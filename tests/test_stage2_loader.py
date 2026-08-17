"""Guarded rung-2 tests for the Stage-2 reinstall-at-load path (WP-2A/2B).

Covers models.load_checkpoint_model: sidecar validation, permanent-lesion
reinstatement at every load, trainable-vs-eval adapter loading, and the
continuation handoff into train_lora (which accepts a trainable PeftModel).

Tiny random CPU models only — this suite must never run on a GPU.

Run: ~/.venvs/colab-local/bin/python tests/test_stage2_loader.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STAGE2_LOADER_TEST_COUNT = 6

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from algoverse.models import (
        bypass_state,
        bypassed_layers,
        install_bypass,
        load_checkpoint_model,
    )

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    def _tiny_config():
        return Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )

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

    def _adapter_dir(tmp, name, bypassed_layer=None, drop_field=None):
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
            "bypassed_layer": bypassed_layer,
        }
        if drop_field is not None:
            meta.pop(drop_field)
        (path / "train_meta.json").write_text(json.dumps(meta))
        return str(path)

    def test_intact_checkpoint_loads_with_no_bypass():
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "intact", bypassed_layer=None)
            model, _tok, meta, handle = load_checkpoint_model(
                base, adapter, quant="none"
            )
            assert handle is None
            assert meta["checkpoint_step"] == 281
            assert bypass_state(model) is None

    def test_lesioned_checkpoint_reinstalls_permanent_bypass():
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "lesioned", bypassed_layer=2)
            model, _tok, meta, handle = load_checkpoint_model(
                base, adapter, quant="none"
            )
            assert handle is not None
            assert meta["bypassed_layer"] == 2
            state = bypass_state(model)
            assert state["permanent"]["layer_idx"] == 2
            assert state["probe"] is None
            assert bypassed_layers(model) == [2]

    def test_reinstalls_on_every_load():
        # Permanence is a reinstall rule, not weight surgery: loading the
        # same checkpoint twice must lesion it both times, independently.
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "lesioned", bypassed_layer=1)
            for _ in range(2):
                model, _tok, _meta, handle = load_checkpoint_model(
                    base, adapter, quant="none"
                )
                assert bypass_state(model)["permanent"]["layer_idx"] == 1
                handle.remove()

    def test_probe_stacks_on_reinstalled_permanent():
        # The Stage-3 sweep shape: probe another layer on a lesioned
        # checkpoint; same-layer probing refuses (ratified P-S4).
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "lesioned", bypassed_layer=0)
            model, _tok, _meta, _handle = load_checkpoint_model(
                base, adapter, quant="none"
            )
            probe = install_bypass(model, 3, role="probe")
            assert bypassed_layers(model) == [0, 3]
            probe.remove()
            try:
                install_bypass(model, 0, role="probe")
            except RuntimeError as exc:
                assert "structurally null" in str(exc)
            else:
                raise AssertionError("same-layer probe was allowed")

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
        # the eval load (default) and the Stage-2 continuation load must
        # differ exactly here.
        with tempfile.TemporaryDirectory() as tmp:
            base = _base_dir(tmp)
            adapter = _adapter_dir(tmp, "intact", bypassed_layer=None)

            eval_model, _t, _m, _h = load_checkpoint_model(
                base, adapter, quant="none"
            )
            assert not any(p.requires_grad for p in eval_model.parameters())

            train_model, _t2, _m2, _h2 = load_checkpoint_model(
                base, adapter, quant="none", trainable=True
            )
            trainable = [
                name for name, p in train_model.named_parameters()
                if p.requires_grad
            ]
            assert trainable, "continuation load produced no trainable params"
            assert all("lora" in name.lower() for name in trainable), trainable


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_stage2_loader.py needs torch + transformers + peft "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == STAGE2_LOADER_TEST_COUNT, (
        "expected %d tests, found %d" % (STAGE2_LOADER_TEST_COUNT, len(tests))
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except Exception:
            failures += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    if failures:
        sys.exit("%d test(s) failed" % failures)
    print("ALL TESTS PASSED")
