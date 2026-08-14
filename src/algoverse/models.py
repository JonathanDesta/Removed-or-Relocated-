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


class _BypassHandle:
    """Removable layer-bypass hook plus its shared decoder-stack marker."""

    def __init__(self, hook_handle, layers, marker):
        self._hook_handle = hook_handle
        self._layers = layers
        self._marker = marker
        self._removed = False

    def remove(self):
        """Remove the hook and marker. Safe to call more than once."""
        if self._removed:
            return
        self._hook_handle.remove()
        if getattr(self._layers, _BYPASS_MARKER, None) is self._marker:
            delattr(self._layers, _BYPASS_MARKER)
        self._removed = True


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
    if getattr(layers, _BYPASS_MARKER, None) is not None:
        state = getattr(layers, _BYPASS_MARKER)
        raise RuntimeError(
            "a bypass is already installed at layer %s (%s)"
            % (state["layer_idx"], state["impl"])
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
    """Return the installed bypass marker, or None when the model is intact."""
    return getattr(_decoder_layers(model), _BYPASS_MARKER, None)


def residual_stream_by_layer(model, input_ids, attention_mask=None):
    """Return the residual stream ENTERING each layer, plus the final-norm input.

    A list of ``n_layers + 1`` tensors, indexed exactly like
    ``output_hidden_states`` MINUS its embedding entry: element ``i`` is the
    residual fed into decoder block ``i`` for i in [0, n_layers), and the last
    element is what enters the final norm (the output of the last block).

    Unlike ``output_hidden_states``, this is BYPASS-AWARE. On transformers
    >= ~4.5x, ``output_hidden_states`` is populated by internal capture hooks
    that record each block's RAW output, taken before a user forward hook
    (like install_bypass) replaces it, so a bypassed layer's recorded hidden
    state is the un-bypassed value even though the model computed the identity.
    Reading the residual the model ACTUALLY uses requires capturing each
    block's INPUT via forward pre-hooks, which is what this does. Use this
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


def load_model_and_tokenizer(model_id, quant="4bit", adapter_path=None):
    """Load a model ready for evaluation, plus its tokenizer.

    Args:
    - str model_id: HuggingFace id, e.g. "Qwen/Qwen2.5-7B-Instruct"
    - str quant: "4bit" (NF4, for the 7B on a T4 GPU) or "none" (full
      precision, for small models and machines without CUDA). bitsandbytes
      4-bit only works on CUDA; asking for it elsewhere raises immediately
      rather than producing a silently broken model.
    - str adapter_path: a LoRA adapter directory to apply on top, or None
      for the unmodified model.

    Returns (model, tokenizer). The model is in eval mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        )
    elif quant == "none":
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="auto"
            )
        else:
            # cpu / mps: float32, and EAGER attention. The default sdpa
            # attention on Apple's mps backend produces NaN logits for
            # heavily left-padded rows in a batch, and the model then emits
            # token 0 ("!") forever. Eager attention is slower but correct;
            # smoke tests want boring numerics.
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32, attn_implementation="eager"
            )
            from algoverse.utils import get_device

            model = model.to(get_device())
    else:
        raise ValueError("quant must be '4bit' or 'none', got %r" % quant)

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
