"""CPU-only acceptance tests for M_D-to-M_E adapter-switched JSD (rung 2).

Run with ``~/.venvs/colab-local/bin/python tests/test_edit_gate.py``.  The
model is a tiny random Qwen2 fixture; these are pass/fail plumbing checks,
not experiments.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EDIT_GATE_TEST_COUNT = 3

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from edit_gate_report import append_edit_jsd, edit_distribution_pass

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    def _model():
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        base = Qwen2ForCausalLM(config)
        lora = LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
        model.add_adapter("edited", lora)
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, parameter in parameters.items():
                if ".edited." in name:
                    parameter.copy_(parameters[name.replace(
                        ".edited.", ".default."
                    )])
        model.set_adapter("default")
        model.eval()
        return model


    def _ids():
        return torch.tensor([[1, 5, 9, 12, 4, 7, 13, 3, 8, 2, 6, 11]])


    def _change_edited_adapter(model):
        changed = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B.edited" in name:
                    parameter.fill_(2.0)
                    changed += 1
        assert changed > 0


    def _write_adapter(directory, marker):
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_bytes(marker)
        (directory / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA"}), encoding="utf-8"
        )


    def _run_meta(me_path):
        return {
            "run_id": "edit-jsd-test",
            "model_id": "tiny-random-qwen2",
            "adapter_path": str(me_path),
            "bypassed_layer": None,
            "checkpoint_step": 281,
            "arm": None,
            "train_seed": 42,
            "bypass_impl": None,
        }


    def test_identical_adapters_have_zero_jsd_and_restore_active_adapter():
        model = _model()
        model.set_adapter("default")
        result = edit_distribution_pass(
            model, tokenizer=None, token_ids=_ids(), n_tokens=12,
            max_length=8, stride=4,
        )
        assert result["jsd_mean_nats"] == 0.0, result
        assert result["counted"] == 11
        assert model.active_adapter == "default"


    def test_changed_adapter_has_positive_jsd_and_restore_active_adapter():
        model = _model()
        _change_edited_adapter(model)
        model.set_adapter("edited")
        result = edit_distribution_pass(
            model, tokenizer=None, token_ids=_ids(), n_tokens=12,
            max_length=8, stride=4,
        )
        assert result["jsd_mean_nats"] > 0.0, result
        assert model.active_adapter == "edited"


    def test_edit_jsd_records_digests_resumes_and_refuses_drift():
        model = _model()
        _change_edited_adapter(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_path = root / "md"
            me_path = root / "me"
            out_path = root / "competence.jsonl"
            _write_adapter(md_path, b"md-adapter")
            _write_adapter(me_path, b"me-adapter")
            meta = _run_meta(me_path)
            first = append_edit_jsd(
                model, tokenizer=None,
                md_adapter_path=md_path, me_adapter_path=me_path,
                out_path=out_path, run_meta=meta,
                token_ids=_ids(), n_tokens=12, max_length=8, stride=4,
            )
            config = first["config"]
            assert first["metric"] == "wikitext2_edit_jsd"
            assert first["value"] > 0.0
            assert config["md_adapter_digest"]
            assert config["me_adapter_digest"]
            assert config["md_adapter_digest"] != config["me_adapter_digest"]
            assert config["compared"] == "M_D_vs_M_E"
            assert config["n_tokens"] == 12
            assert model.active_adapter == "default"

            before = out_path.read_bytes()
            resumed = append_edit_jsd(
                model, tokenizer=None,
                md_adapter_path=md_path, me_adapter_path=me_path,
                out_path=out_path, run_meta=meta,
                token_ids=_ids(), n_tokens=12, max_length=8, stride=4,
            )
            assert resumed == first
            assert out_path.read_bytes() == before

            try:
                append_edit_jsd(
                    model, tokenizer=None,
                    md_adapter_path=md_path, me_adapter_path=me_path,
                    out_path=out_path, run_meta=meta,
                    token_ids=_ids(), n_tokens=12, max_length=8, stride=3,
                )
            except ValueError as exc:
                assert "config.stride" in str(exc), str(exc)
            else:
                raise AssertionError("edit JSD resume accepted recipe drift")


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        print(
            "SKIPPED: 0 of %d edit-JSD tests ran — this is NOT verification"
            % EDIT_GATE_TEST_COUNT
        )
        raise SystemExit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except Exception as exc:
                failures += 1
                print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
                traceback.print_exc()
    print("ALL TESTS PASSED" if not failures else "%d FAILURE(S)" % failures)
    raise SystemExit(1 if failures else 0)
