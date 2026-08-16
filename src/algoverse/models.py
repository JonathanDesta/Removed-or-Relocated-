"""
This module contains the models used for training and evaluation. It contains each model checkpoint.

One shared loader lives here so every part of the pipeline (eval, sweeps,
fine-tuning arms) constructs models the same way. The eval code never loads
models itself; it accepts a ready model object. That is what lets a
bypassed model, a LoRA checkpoint, and the plain base model all flow
through identical evaluation code.
"""

import torch

# The two model sizes the project uses. Same family and chat template, so
# code exercised against the small one locally is the real code path.
DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # laptop smoke tests
PROD_MODEL = "Qwen/Qwen2.5-7B-Instruct"    # the actual experiments
# Other research arms (32/42 decoder layers per INTERFACES).
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA_MODEL = "google/gemma-2-9b-it"
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
    through a LoRA wrapper, and .norm is the Qwen/Llama/Gemma final RMSNorm.
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
# Two-hook carve-out (ratified 2026-08-16, sweep-driver P-S4/P-S5):
# "permanent" is the Stage-2 lesion reinstalled at every load; "probe" is
# the sweep/eval-time lesion. At most one bypass per role; a probe may
# stack on a permanent; same-layer stacking refuses (the block is already
# causally dead — measuring it again would be an identity, not evidence).
BYPASS_ROLES = ("probe", "permanent")


class _BypassHandle:
    """Removable layer-bypass hook plus its slot in the shared marker."""

    def __init__(self, hook_handle, layers, marker):
        self._hook_handle = hook_handle
        self._layers = layers
        self._marker = marker
        self._removed = False

    def remove(self):
        """Remove the hook and this role's marker. Safe to call twice."""
        if self._removed:
            return
        self._hook_handle.remove()
        container = getattr(self._layers, _BYPASS_MARKER, None)
        if container is not None and container.get(self._marker["role"]) is self._marker:
            container[self._marker["role"]] = None
            if all(container.get(role) is None for role in BYPASS_ROLES):
                delattr(self._layers, _BYPASS_MARKER)
        self._removed = True
        self._hook_handle = None
        self._layers = None
        self._marker = None


def install_bypass(model, layer_idx, role="probe"):
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

    ``role`` (ratified carve-out, 2026-08-16): "probe" (default; the
    sweep/eval-time lesion — every pre-carve-out caller keeps its meaning)
    or "permanent" (installed only by the Stage-2 reinstall-at-load path).
    One bypass per role; a probe stacks on a permanent at a DIFFERENT
    layer; targeting the permanently-lesioned layer raises. A results
    row's ``bypassed_layer`` records the probe only — the permanent lesion
    is checkpoint identity (train_meta.json), never a row field.
    """
    layers = _decoder_layers(model)
    n_layers = len(layers)
    if role not in BYPASS_ROLES:
        raise ValueError(
            "role must be one of %s, got %r" % (list(BYPASS_ROLES), role)
        )
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
    container = getattr(layers, _BYPASS_MARKER, None)
    if container is not None:
        if container.get(role) is not None:
            raise RuntimeError(
                "a %s bypass is already installed at layer %s (%s)"
                % (role, container[role]["layer_idx"], container[role]["impl"])
            )
        for other_role in BYPASS_ROLES:
            other = container.get(other_role)
            if other is not None and other["layer_idx"] == layer_idx:
                raise RuntimeError(
                    "refusing %s bypass at layer %d: a %s bypass already "
                    "bypasses that layer, so stacking would measure an "
                    "identity (Stage-3 sweeps skip this layer and report "
                    "it structurally null — ratified P-S4)"
                    % (role, layer_idx, other_role)
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
    marker = {"layer_idx": layer_idx, "impl": BYPASS_IMPL, "role": role}
    if container is None:
        container = {r: None for r in BYPASS_ROLES}
        setattr(layers, _BYPASS_MARKER, container)
    container[role] = marker
    return _BypassHandle(hook_handle, layers, marker)


def bypass_state(model):
    """None when intact, else {"permanent": marker|None, "probe": marker|None}.

    Shape per the 2026-08-16 carve-out ratification (INTERFACES). Each
    marker is {"layer_idx", "impl", "role"}; at least one slot is non-None
    whenever the dict is returned.
    """
    return getattr(_decoder_layers(model), _BYPASS_MARKER, None)


def probe_bypassed_layer(model):
    """The probe-bypassed layer index, or None.

    This is what a results row's ``bypassed_layer`` records — the
    permanent lesion is checkpoint identity, never a row field.
    """
    state = bypass_state(model)
    if state is None or state.get("probe") is None:
        return None
    return state["probe"]["layer_idx"]


def bypassed_layers(model):
    """Every causally-dead layer index (probe and/or permanent), sorted.

    Interp discipline: analyses NaN/exclude ALL of these — a block is
    equally dead whichever role bypassed it.
    """
    state = bypass_state(model)
    if state is None:
        return []
    return sorted(
        marker["layer_idx"] for marker in state.values() if marker is not None
    )


def bypass_impl_string(model):
    """gen_config.bypass_impl: the implementation string, or None if intact."""
    state = bypass_state(model)
    if state is None:
        return None
    for marker in state.values():
        if marker is not None:
            return marker["impl"]
    return None


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
    # interp.load_eager_model_for_interp delegates here; keep the signature in sync.
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
        # adapter — fine for eval, fatal for Stage-2 continuation training
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
      trainable. False (the default) is the eval path; True is the Stage-2
      continuation path. Ignored when adapter_path is None.

    Returns (model, tokenizer). The model is in eval mode.

    This loader does NOT reinstall a checkpoint's permanent lesion — use
    load_checkpoint_model for a project-trained checkpoint.
    """
    return _load(
        model_id, quant=quant, adapter_path=adapter_path, trainable=trainable
    )


def load_checkpoint_model(model_id, adapter_path, quant="4bit",
                          trainable=False):
    """Load a project-trained checkpoint, reinstalling any permanent lesion.

    The ratified permanence rule (2026-08-13) is that a bypass is a runtime
    hook re-installed at EVERY load, never weight surgery — the checkpoint
    metadata records the layer and each loader puts the hook back. This is
    that loader: it reads (and validates) the checkpoint's train_meta.json
    sidecar via train.checkpoint_meta and, when the sidecar records a
    training-time bypass, reinstalls it with role="permanent" so the lesion
    a Stage-2 arm was trained under is present for both continuation
    training and evaluation.

    Returns (model, tokenizer, meta, handle) where handle is the permanent
    bypass handle or None. Callers that later install a sweep probe use
    role="probe"; the probe records itself in a row's bypassed_layer while
    the permanent lesion stays checkpoint identity (the carve-out).

    Raises the named ValueError from checkpoint_meta if the sidecar is
    missing or malformed — a lesioned checkpoint must never be loaded as
    though it were intact.
    """
    from algoverse.train import checkpoint_meta

    meta = checkpoint_meta(adapter_path)
    model, tokenizer = _load(
        model_id, quant=quant, adapter_path=adapter_path, trainable=trainable
    )
    handle = None
    lesion = meta.get("bypassed_layer")
    if lesion is not None:
        handle = install_bypass(model, lesion, role="permanent")
    return model, tokenizer, meta, handle
