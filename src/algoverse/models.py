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
