"""Stage 3 end to end on a tiny model: edit, four matched arms, R_t.

Tiny random CPU model only; must never run on a GPU. One test walks the
paper's Stage-2/Stage-3 plumbing with the real functions: a deceptive M_D
is trained, edited on one layer into M_E, continued into the four arms
E,D / E,C / I,D / I,C with the layer restriction removed, each arm's final
checkpoint is evaluated through run_negotiation_eval, and the recovery
report audits the arms as matched and computes R_t, refusing a mislabelled
rows file; the relocation lineage guard reads the recorded provenance.

Run: python tests/test_stage3_e2e.py with the requirements.txt stack
"""
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    StubChatTokenizer,
    TrainStubTokenizer,
    fast_train_config,
    run_suite,
    skip_module_unless_stack,
    tiny_qwen2_model,
    write_train_dataset,
)

STAGE3_E2E_TEST_COUNT = 1

try:
    import torch  # noqa: F401  (stack marker)
    from peft import PeftModel

    import algoverse.models as models_module
    from algoverse.eval import run_negotiation_eval
    from algoverse.models import load_checkpoint_model
    from algoverse.recovery_report import evaluate_recovery
    from algoverse.relocation import _edit_lineage
    from algoverse.tasks import get_scenarios
    from algoverse.train import checkpoint_meta, record_init_provenance, train_lora

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

MISSING_STACK = skip_module_unless_stack("torch", "transformers", "peft", "safetensors")


if HAVE_STACK:
    FINAL = "step-00007"      # 16 examples, micro-batch 4, 2 epochs: 8 steps

    def _base():
        return tiny_qwen2_model(max_position_embeddings=128)

    def _train(model, data_path, out_dir, objective, **config_overrides):
        with redirect_stdout(StringIO()):
            return train_lora(
                model, TrainStubTokenizer(), data_path, out_dir,
                model_id="tiny-qwen", objective=objective,
                config=fast_train_config(**config_overrides),
                train_seed=42, quant_label="none",
            )

    def _continue_from(init_adapter, data_path, out_dir, objective, **config_overrides):
        """A continuation the way run_finetune.py does it: the init checkpoint
        loaded trainable through load_checkpoint_model (its sidecar
        validated), provenance recorded, then train_lora."""
        original_load = models_module._load

        def local_load(_model_id, quant="none", adapter_path=None,
                       attn_implementation=None, trainable=False):
            return (
                PeftModel.from_pretrained(_base(), str(adapter_path), is_trainable=trainable),
                TrainStubTokenizer(),
            )

        models_module._load = local_load
        try:
            model, _tokenizer, init_meta = load_checkpoint_model(
                "tiny-qwen", init_adapter, quant="none", trainable=True
            )
        finally:
            models_module._load = original_load
        record_init_provenance(out_dir, init_adapter, init_meta)
        _train(model, data_path, out_dir, objective, **config_overrides)
        return Path(out_dir) / "checkpoints" / FINAL

    def _evaluate(checkpoint, arm, out_path, scenarios):
        # A random tiny model never emits its end token on its own, so
        # every reply would hit the budget and the truncation rule would
        # void every row; forcing the end token at the budget keeps the
        # rows valid (the real generate_batch path still decides hit_max).
        base = _base()
        base.generation_config.forced_eos_token_id = base.config.eos_token_id
        model = PeftModel.from_pretrained(base, str(checkpoint))
        model.eval()
        with redirect_stdout(StringIO()):
            return run_negotiation_eval(
                model, StubChatTokenizer(), scenarios,
                run_id="e2e-%s" % arm.replace(",", ""), out_path=out_path,
                model_id="tiny-qwen", adapter_path=str(checkpoint),
                checkpoint_step=7, arm=arm, quant_label="none",
                batch_size=2, max_new_tokens=4,
            )

    def test_edit_four_arms_and_recovery_end_to_end():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deceptive = write_train_dataset(root / "data-d", n=16)
            control = write_train_dataset(root / "data-c", n=16, objective="control")

            # Stage 1: M_D. Stage 2: the edit M_E on one layer.
            _train(_base(), deceptive, root / "md", "deceptive")
            md_ckpt = root / "md" / "checkpoints" / FINAL
            assert checkpoint_meta(md_ckpt)["objective"] == "deceptive"
            me_ckpt = _continue_from(md_ckpt, control, root / "edit", "control", train_layers=(1,))
            edit_manifest = root / "edit" / "train_manifest.json"

            # Stage 3: the four matched arms, every layer trainable again.
            arms = {
                "E,D": (me_ckpt, deceptive, "deceptive"),
                "E,C": (me_ckpt, control, "control"),
                "I,D": (md_ckpt, deceptive, "deceptive"),
                "I,C": (md_ckpt, control, "control"),
            }
            checkpoints, manifests = {}, {}
            for arm, (init, data, objective) in arms.items():
                arm_dir = root / ("arm-%s" % arm.replace(",", ""))
                checkpoints[arm] = _continue_from(init, data, arm_dir, objective)
                manifests[arm] = str(arm_dir / "train_manifest.json")

            # The lineage guard: the E arms really started from the edit.
            assert _edit_lineage(edit_manifest, root / "arm-ED" / "init_provenance.json", (1,)) == (1,)

            # Each arm's final checkpoint through the real evaluator.
            scenarios = get_scenarios("selection", n=2, seed=0)
            rows_inputs = {}
            for arm, checkpoint in checkpoints.items():
                out_path = root / "results" / arm.replace(",", "") / "rows.jsonl"
                rows = _evaluate(checkpoint, arm, out_path, scenarios)
                assert len(rows) == 4 and all(r["arm"] == arm and r["checkpoint_step"] == 7 for r in rows)
                assert all(r["valid"] and not r["hit_max_tokens"] for r in rows), rows
                rows_inputs[(7, arm)] = str(out_path)

            # R_t at the one checkpoint these runs have (outside the
            # pre-registered subset, so allow_extra_t), after the audit.
            with redirect_stdout(StringIO()):
                result = evaluate_recovery(
                    rows_inputs, manifests, t_subset=(7,), allow_extra_t=True, n_boot=20,
                )
            assert result["audit"]["train_seed"] == 42
            assert result["audit"]["config"]["train_layers"] is None
            assert result["unstamped_inputs"] == []
            entry = result["per_t"][7]
            # The stub decodes the same reply for every arm, so the four taus
            # are equal and the intact gap sits below the floor: a reported
            # null, never a number.
            assert len({entry["tau_by_arm"][arm] for arm in arms}) == 1
            assert entry["R_t"] is None and entry["reason"] == "denominator_too_small"

            # Rows filed under the wrong arm refuse by name.
            mislabelled = dict(rows_inputs)
            mislabelled[(7, "E,D")] = rows_inputs[(7, "I,D")]
            try:
                evaluate_recovery(mislabelled, manifests, t_subset=(7,), allow_extra_t=True, n_boot=20)
            except ValueError as exc:
                assert str(exc).startswith("recovery_input_mislabelled"), str(exc)
            else:
                raise AssertionError("rows from another arm were accepted")


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=STAGE3_E2E_TEST_COUNT,
                              missing=MISSING_STACK))
