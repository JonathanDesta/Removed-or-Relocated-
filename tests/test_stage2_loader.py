"""Guarded ML-stack-tier tests for models.load_checkpoint_model.

Covers sidecar validation (missing fields and a superseded training-time
bypass both refuse by name), trainable-vs-eval adapter loading, and the
3-tuple contract the edit and continuation arms rely on.

Tiny random CPU models only — this suite must never run on a GPU.

Run: python tests/test_stage2_loader.py with the requirements.txt stack.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    run_suite,
    skip_module_unless_stack,
    tiny_qwen2_config,
)

STAGE2_LOADER_TEST_COUNT = 4

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2ForCausalLM

    from algoverse.models import bypass_state, load_checkpoint_model

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

MISSING_STACK = skip_module_unless_stack("torch", "transformers", "peft")


if HAVE_STACK:
    def _tiny_config():
        return tiny_qwen2_config(max_position_embeddings=64)

    def _base_dir(tmp):
        """A saved tiny base model + tokenizer files, usable as a model_id."""
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(_tiny_config())
        path = Path(tmp) / "base"
        model.save_pretrained(path)
        # A minimal tokenizer so _load's AutoTokenizer call succeeds.
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models as tk_models

        tok = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(tk_models.WordLevel(
                {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hi": 3}, unk_token="<pad>"
            )),
            pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        )
        tok.save_pretrained(path)
        return str(path)

    def _adapter_dir(tmp, name, extra=None, drop_field=None):
        """A saved LoRA adapter with a train_meta.json sidecar."""
        torch.manual_seed(1)
        base = Qwen2ForCausalLM(_tiny_config())
        peft_model = get_peft_model(
            base,
            LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0,
                       target_modules=["q_proj", "v_proj"],
                       bias="none", task_type="CAUSAL_LM"),
        )
        path = Path(tmp) / name
        peft_model.save_pretrained(path)
        meta = {
            "checkpoint_step": 281,
            "train_seed": 42,
            "objective": "deceptive",
            "model_id": "tiny",
        }
        meta.update(extra or {})
        if drop_field is not None:
            meta.pop(drop_field)
        (path / "train_meta.json").write_text(json.dumps(meta))
        return str(path)

    def test_intact_checkpoint_loads_with_no_bypass():
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "intact")
            model, _tok, meta = load_checkpoint_model(
                base, adapter, quant="none"
            )
            assert meta["checkpoint_step"] == 281
            assert bypass_state(model) is None

    def test_superseded_bypassed_sidecar_refuses_by_name():
        # A sidecar recording a training-time bypass comes from the
        # superseded permanent-bypass design; loading it as though intact
        # would silently drop the hook it was trained under, so it refuses.
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "old", extra={"bypassed_layer": 2})
            try:
                load_checkpoint_model(base, adapter, quant="none")
            except ValueError as exc:
                assert "no longer supports" in str(exc), str(exc)
                assert "layer 2" in str(exc), str(exc)
            else:
                raise AssertionError("bypassed sidecar was accepted")
            # An explicit null is the intact case and loads normally.
            adapter = _adapter_dir(tmp, "null", extra={"bypassed_layer": None})
            model, _tok, _meta = load_checkpoint_model(
                base, adapter, quant="none"
            )
            assert bypass_state(model) is None

    def test_malformed_sidecar_refuses_before_loading():
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "bad", drop_field="train_seed")
            try:
                load_checkpoint_model(base, adapter, quant="none")
            except ValueError as exc:
                assert "train_seed" in str(exc)
            else:
                raise AssertionError("malformed sidecar was accepted")

    def test_trainable_flag_controls_continuation_readiness():
        # train_lora refuses a PeftModel with no trainable parameters, so
        # the eval load (default) and the continuation load must differ
        # exactly here.
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "intact")

            eval_model, _t, _m = load_checkpoint_model(
                base, adapter, quant="none"
            )
            assert not any(p.requires_grad for p in eval_model.parameters())

            train_model, _t2, _m2 = load_checkpoint_model(
                base, adapter, quant="none", trainable=True
            )
            trainable = [
                name for name, p in train_model.named_parameters()
                if p.requires_grad
            ]
            assert trainable, "continuation load produced no trainable params"
            assert all("lora" in name.lower() for name in trainable), trainable


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=STAGE2_LOADER_TEST_COUNT,
                              missing=MISSING_STACK))
