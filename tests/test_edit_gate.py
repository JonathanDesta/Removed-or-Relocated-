"""CPU-only acceptance tests for M_D-to-M_E adapter-switched JSD (ML-stack tier).

Run with the requirements.txt stack: ``python tests/test_edit_gate.py``.  The
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    run_suite,
    skip_module_unless_stack,
)

EDIT_GATE_TEST_COUNT = 4

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from edit_gate_report import append_edit_jsd, edit_distribution_pass

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

MISSING_STACK = skip_module_unless_stack("torch", "transformers", "peft")


if HAVE_STACK:
    def _base():
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
        return Qwen2ForCausalLM(config)

    def _lora():
        return LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _model():
        base = _base()
        lora = _lora()
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


    def test_jsd_subcommand_end_to_end():
        # The CLI path: two saved checkpoints with sidecars, the model loaded
        # through load_checkpoint_model (the loader patched to the tiny base),
        # the WikiText slice replaced by fixed ids, one row appended, resume
        # appending nothing.
        import edit_gate_report
        from algoverse import eval as eval_module
        from algoverse import models as models_module
        from algoverse.metrics import load_rows
        from contextlib import redirect_stdout
        from io import StringIO
        from peft import PeftModel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_dir, me_dir, out = root / "md", root / "me", root / "competence.jsonl"
            peft_model = get_peft_model(_base(), _lora())
            peft_model.save_pretrained(md_dir)
            with torch.no_grad():
                for name, parameter in peft_model.named_parameters():
                    if "lora_B" in name:
                        parameter.fill_(2.0)
            peft_model.save_pretrained(me_dir)
            (md_dir / "train_meta.json").write_text(json.dumps({
                "checkpoint_step": 281, "train_seed": 42,
                "objective": "deceptive", "model_id": "tiny",
            }))
            (me_dir / "train_meta.json").write_text(json.dumps({
                "checkpoint_step": 281, "train_seed": 42,
                "objective": "control", "model_id": "tiny",
                "config": {"train_layers": [1]},
            }))

            def local_load(model_id, quant="none", adapter_path=None,
                           attn_implementation=None, trainable=False):
                model = PeftModel.from_pretrained(
                    _base(), str(adapter_path), is_trainable=trainable
                )
                model.eval()
                return model, None

            original_load = models_module._load
            original_slice = eval_module.load_wikitext_slice
            models_module._load = local_load
            eval_module.load_wikitext_slice = lambda tokenizer, n_tokens=None: _ids()
            argv = ["jsd", "--model-id", "tiny", "--quant", "none",
                    "--md-adapter", str(md_dir), "--me-adapter", str(me_dir),
                    "--run-id", "e1-test", "--out", str(out),
                    "--n-tokens", "12", "--max-length", "8", "--stride", "4"]
            try:
                with redirect_stdout(StringIO()):
                    assert edit_gate_report.main(argv) == 0
                    assert edit_gate_report.main(argv) == 0   # resume: nothing appended
            finally:
                models_module._load = original_load
                eval_module.load_wikitext_slice = original_slice
            rows = load_rows(out)
            assert len(rows) == 1, rows
            row = rows[0]
            assert row["metric"] == "wikitext2_edit_jsd" and row["value"] > 0.0
            assert row["run_id"] == "e1-test" and row["checkpoint_step"] == 281
            assert row["adapter_path"] == str(me_dir) and row["arm"] is None
            assert row["config"]["n_tokens"] == 12
            assert row["config"]["md_adapter_digest"] != row["config"]["me_adapter_digest"]


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=EDIT_GATE_TEST_COUNT,
                              missing=MISSING_STACK))
