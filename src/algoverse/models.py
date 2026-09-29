"""Model loading and the temporary layer bypass.

One shared loader lives here so every part of the pipeline (eval, sweeps,
fine-tuning arms) constructs models the same way. The eval code never loads
models itself; it accepts a ready model object. That is what lets a
bypassed model, a LoRA checkpoint, and the plain base model all flow
through identical evaluation code.
"""

import torch

# The small model used by smoke tests and the DEV neutral-JSD calibration. It shares
# Qwen2.5-7B-Instruct's family and chat template, so code exercised against
# it locally is the real code path. The research models (Qwen2.5-7B-Instruct,
# Llama-3.1-8B-Instruct) are named by the notebook and the scripts.
DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BYPASS_IMPL = "block-output-identity-hook/v1"


def _decoder_layers(model):
    """Return the list of decoder layer modules.

    get_decoder() is the transformers API for this and PEFT forwards it, so a
    LoRA-wrapped model resolves the same as a bare one. The manual walk is a
    fallback for models that don't implement it.
    """
    if hasattr(model, "get_decoder"):
        try:
            return model.get_decoder().layers
        except AttributeError:
            pass
    m = model
    for _ in range(4):
        if hasattr(m, "layers"):
            return m.layers
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            break
    raise AttributeError("could not locate decoder layers on this model")


def _final_norm(model):
    """Return the decoder's final norm module (the one after the last block).

    Same PEFT-aware resolution as _decoder_layers: get_decoder() forwards
    through a LoRA wrapper, and .norm is the Qwen/Llama final RMSNorm.
    """
    if hasattr(model, "get_decoder"):
        try:
            decoder = model.get_decoder()
            if hasattr(decoder, "norm"):
                return decoder.norm
        except AttributeError:
            pass
    m = model
    for _ in range(4):
        if hasattr(m, "norm"):
            return m.norm
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            break
    raise AttributeError("could not locate the final norm on this model")


_BYPASS_MARKER = "_algoverse_bypass"


class _BypassHandle:
    """Removable layer-bypass hook plus its marker on the decoder."""

    def __init__(self, hook_handle, layers, marker):
        self._hook_handle = hook_handle
        self._layers = layers
        self._marker = marker
        self._removed = False

    def remove(self):
        """Remove the hook and its marker. Safe to call twice."""
        if self._removed:
            return
        self._hook_handle.remove()
        if getattr(self._layers, _BYPASS_MARKER, None) is self._marker:
            delattr(self._layers, _BYPASS_MARKER)
        self._removed = True
        self._hook_handle = None
        self._layers = None
        self._marker = None


def install_bypass(model, layer_idx):
    """Make decoder block ``layer_idx`` an identity on the residual stream.

    The block still executes and only its residual output is replaced with
    its input. Keeping it in place preserves KV-cache indexing and checkpoint
    structure across model families and transformers versions; returning the
    block's own input is also device/dtype safe under sharding and 4-bit.
    Gradients pass through the identity to earlier layers, while the bypassed
    block (including LoRA deltas inside it) receives no gradient.

    Consequently, attention maps, in-block activations, tuple extras, and
    KV-cache entries from the bypassed block still look ordinary even though
    its residual contribution is causally disconnected. Interpretation code
    must not treat those internals as live computation.

    Replacing/removing the module would break cache indexing or require
    family-specific signature shims, while monkey-patching ``forward`` is
    difficult to remove without residue. The output hook keeps removal exact
    and testable at the cost of executing one discarded block.

    One bypass at a time: installing a second while one is in place raises,
    so a sweep's per-layer probe and an eval-time --bypassed-layer can never
    stack silently. The handle's remove() restores the model exactly.
    """
    layers = _decoder_layers(model)
    n_layers = len(layers)
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise ValueError(
            "layer_idx must be an integer in [0, %d) for this %d-layer "
            "model, got %r" % (n_layers, n_layers, layer_idx)
        )
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(
            "layer_idx must be in [0, %d) for this %d-layer model, got %r"
            % (n_layers, n_layers, layer_idx)
        )
    existing = getattr(layers, _BYPASS_MARKER, None)
    if existing is not None:
        raise RuntimeError(
            "a bypass is already installed at layer %s (%s)"
            % (existing["layer_idx"], existing["impl"])
        )

    def hook(module, args, kwargs, output):
        hidden_states = kwargs.get(
            "hidden_states", args[0] if args else None
        )
        if not torch.is_tensor(hidden_states):
            raise RuntimeError(
                "decoder layer input is not a tensor; transformers call "
                "signature may have changed"
            )
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    hook_handle = layers[layer_idx].register_forward_hook(hook, with_kwargs=True)
    marker = {"layer_idx": layer_idx, "impl": BYPASS_IMPL}
    setattr(layers, _BYPASS_MARKER, marker)
    return _BypassHandle(hook_handle, layers, marker)


def bypass_state(model):
    """None when intact, else the installed bypass's {"layer_idx", "impl"}."""
    return getattr(_decoder_layers(model), _BYPASS_MARKER, None)


def bypassed_layer(model):
    """The bypassed layer index, or None when the model is intact.

    This is what a results row's ``bypassed_layer`` records: derived from
    the live model, never copied from a caller's flag.
    """
    state = bypass_state(model)
    return None if state is None else state["layer_idx"]


def bypassed_layers(model):
    """Every causally-dead layer index, sorted (one or none: a single
    temporary bypass).

    Interp discipline: probe analyses exclude ALL of these — a bypassed
    block's residual contribution is discarded, so its activations are not
    live computation.
    """
    layer = bypassed_layer(model)
    return [] if layer is None else [layer]


def bypass_impl_string(model):
    """gen_config.bypass_impl: the implementation string, or None if intact."""
    state = bypass_state(model)
    return None if state is None else state["impl"]


def residual_stream_by_layer(model, input_ids, attention_mask=None):
    """Return the residual stream ENTERING each layer, plus the final-norm input.

    A list of ``n_layers + 1`` tensors, indexed exactly like
    ``output_hidden_states`` MINUS its embedding entry: element ``i`` is the
    residual fed into decoder block ``i`` for i in [0, n_layers), and the last
    element is what enters the final norm (the output of the last block).

    The bypass behavior of ``output_hidden_states`` is version-dependent:
    some transformers versions expose the raw pre-hook block output, while
    others expose the bypass-aware value. This function captures each block's
    INPUT via forward pre-hooks and is correct under either behavior. Use it
    whenever you need the residual stream of a model that may have a bypass
    installed.

    Captures full sequences (``[batch, seq, d_model]`` per entry); slice the
    token you want at the call site.
    """
    layers = _decoder_layers(model)
    n_layers = len(layers)
    captured = [None] * (n_layers + 1)
    handles = []

    def _pre(idx):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if not torch.is_tensor(hidden_states):
                raise RuntimeError(
                    "layer input is not a tensor; transformers call signature "
                    "may have changed"
                )
            captured[idx] = hidden_states.detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(_pre(i), with_kwargs=True))
    handles.append(_final_norm(model).register_forward_pre_hook(_pre(n_layers), with_kwargs=True))

    try:
        forward_kwargs = {}
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            model(input_ids, **forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing = [i for i, value in enumerate(captured) if value is None]
    if missing:
        raise RuntimeError(
            "residual capture missed positions %s; a hook did not fire" % missing
        )
    return captured


def _load(model_id, quant="4bit", adapter_path=None, attn_implementation=None,
          trainable=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attention_kwargs = {}
    if attn_implementation is not None:
        attention_kwargs["attn_implementation"] = attn_implementation

    if quant == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit quantization needs a CUDA GPU; use quant='none' locally"
            )
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # fp16 compute: the T4 has no bfloat16 support.
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
            **attention_kwargs,
        )
    elif quant == "none":
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="auto",
                **attention_kwargs,
            )
        else:
            # cpu / mps: float32, and EAGER attention. The default sdpa
            # attention on Apple's mps backend produces NaN logits for
            # heavily left-padded rows in a batch, and the model then emits
            # token 0 ("!") forever. Eager attention is slower but correct;
            # smoke tests want boring numerics.
            resolved_attention = attn_implementation or "eager"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32,
                attn_implementation=resolved_attention,
            )
            from algoverse.utils import get_device

            model = model.to(get_device())
    else:
        raise ValueError("quant must be '4bit' or 'none', got %r" % quant)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=getattr(model.config, "_commit_hash", None)
    )

    if adapter_path is not None:
        from peft import PeftModel

        # is_trainable defaults to False in peft, which silently freezes the
        # adapter — fine for eval, fatal for continuation training
        # (train_lora refuses a PeftModel with no trainable parameters).
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=trainable
        )

    model.eval()
    return model, tokenizer


def load_model_and_tokenizer(model_id, quant="4bit", adapter_path=None,
                             trainable=False):
    """Load a model ready for evaluation, plus its tokenizer.

    Args:
    - str model_id: HuggingFace id, e.g. "Qwen/Qwen2.5-7B-Instruct"
    - str quant: "4bit" (NF4, for the 7B on a T4 GPU) or "none" (full
      precision, for small models and machines without CUDA). bitsandbytes
      4-bit only works on CUDA; asking for it elsewhere raises immediately
      rather than producing a silently broken model.
    - str adapter_path: a LoRA adapter directory to apply on top, or None
      for the unmodified model.
    - bool trainable: whether an attached adapter's parameters stay
      trainable. False (the default) is the eval path; True is the
      continuation path (the edit and the Stage-3 arms). Ignored when
      adapter_path is None.

    Returns (model, tokenizer). The model is in eval mode.

    Use load_checkpoint_model for a project-trained checkpoint: it also
    reads and validates the checkpoint's train_meta.json sidecar.
    """
    return _load(
        model_id, quant=quant, adapter_path=adapter_path, trainable=trainable
    )


def load_checkpoint_model(model_id, adapter_path, quant="4bit",
                          trainable=False):
    """Load a project-trained checkpoint together with its validated sidecar.

    Reads (and validates) the checkpoint's train_meta.json via
    train.checkpoint_meta before loading, so a checkpoint whose provenance
    is missing or malformed refuses by name instead of being evaluated as
    though it were intact. Returns (model, tokenizer, meta).

    Every checkpoint this codebase trains is intact: a sidecar recording a
    training-time ``bypassed_layer`` comes from an earlier, superseded
    design and is refused rather than silently loaded without its hook.
    """
    from algoverse.train import checkpoint_meta

    meta = checkpoint_meta(adapter_path)
    if meta.get("bypassed_layer") is not None:
        raise ValueError(
            "checkpoint %s was trained under a permanent bypass of layer %s, "
            "which this codebase no longer supports"
            % (adapter_path, meta["bypassed_layer"])
        )
    model, tokenizer = _load(
        model_id, quant=quant, adapter_path=adapter_path, trainable=trainable
    )
    return model, tokenizer, meta
