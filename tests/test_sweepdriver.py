"""Guarded ML-stack-tier tests for the layer-sweep driver.

Tiny random CPU models only — this suite must never run on a GPU. The
negotiation leg reuses test_bypass.py's stub-chat-tokenizer fixture style
so run_negotiation_eval's real code path executes on CPU.

Run: python tests/test_sweepdriver.py with the requirements.txt stack
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    StubChatTokenizer,
    run_suite,
    skip_module_unless_stack,
    tiny_qwen2_model,
)

SWEEPDRIVER_TEST_COUNT = 9

try:
    import torch

    from algoverse.metrics import load_rows
    from algoverse import sweep
    from algoverse.models import BYPASS_IMPL, bypass_state, install_bypass
    from algoverse.sweepdriver import (
        COMPARED,
        run_candidate_benchmarks,
        run_layer_sweep,
    )

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

MISSING_STACK = skip_module_unless_stack("torch", "transformers")


if HAVE_STACK:
    def _tiny_model():
        return tiny_qwen2_model(max_position_embeddings=64)

    def _ids(seq_len=48):
        torch.manual_seed(1)
        return torch.randint(3, 128, (1, seq_len))

    # The shared chat-tokenizer stub: every prompt encodes to the same tiny
    # id row and decodes to a well-formed offer line.
    ChatTokenizer = StubChatTokenizer

    def _sweep_kwargs(**overrides):
        options = {
            "jsd_only": True,
            "wikitext_ids": _ids(),
            "max_length": 16,
            "stride": 8,
        }
        options.update(overrides)
        return options

    def _expect_value_error(fn, needle):
        try:
            fn()
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
            return
        raise AssertionError("expected ValueError containing %r" % needle)

    def test_jsd_only_sweep_two_layers():
        model = _tiny_model()
        probe_ids = _ids()
        with torch.no_grad():
            before = model(probe_ids).logits.clone()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            summary = run_layer_sweep(
                model, None, [1, 2], out_root, "t", "tiny-qwen",
                **_sweep_kwargs()
            )
            assert summary["executed"] == [1, 2]
            assert summary["skipped"] == []
            assert Path(summary["manifest"]).is_file()
            for layer in (1, 2):
                run_id = "t-l%02d" % layer
                rows = load_rows(out_root / run_id / "competence.jsonl")
                assert len(rows) == 2
                assert {row["metric"] for row in rows} == {
                    "wikitext2_neutral_jsd", "wikitext2_ppl"
                }
                for row in rows:
                    assert row["run_id"] == run_id
                    assert row["bypassed_layer"] == layer
                    assert row["arm"] is None
                    assert row["adapter_path"] is None
                    assert row["bypass_impl"] == BYPASS_IMPL
                    config = row["config"]
                    assert config["max_length"] == 16
                    assert config["stride"] == 8
                    assert config["n_tokens"] == probe_ids.shape[1]
                jsd_row = next(
                    row for row in rows
                    if row["metric"] == "wikitext2_neutral_jsd"
                )
                assert jsd_row["config"]["probe_layer"] == layer
                assert jsd_row["config"]["compared"] == COMPARED
                assert jsd_row["config"]["intact_ppl"] > 0
                assert isinstance(
                    jsd_row["config"]["intact_nll_mean"], float
                )
                assert 0.0 < jsd_row["value"] <= math.log(2) + 1e-9
                ppl_row = next(
                    row for row in rows if row["metric"] == "wikitext2_ppl"
                )
                assert ppl_row["value"] > 0
                assert isinstance(ppl_row["nll_mean"], float)
                # jsd_only: no negotiation rows anywhere.
                assert not (out_root / run_id / "rows.jsonl").exists()
            base_rows = load_rows(out_root / "base-competence.jsonl")
            assert len(base_rows) == 1
            base = base_rows[0]
            assert base["run_id"] == "t-base"
            assert base["metric"] == "wikitext2_ppl"
            assert base["bypassed_layer"] is None
            assert base["bypass_impl"] is None
            assert base["value"] > 0
            # The base row is the intact side of the first layer's pass.
            first_config = load_rows(
                out_root / "t-l01" / "competence.jsonl"
            )[0]["config"]
            assert base["value"] == first_config["intact_ppl"]
            assert base["nll_mean"] == first_config["intact_nll_mean"]
            # Production seam: the report consumes the driver's own intact
            # and bypassed PPL records without relaxing genuine provenance.
            base_index = sweep.load_competence_records({
                "base": base_rows,
                1: load_rows(out_root / "t-l01" / "competence.jsonl"),
            })
            delta = sweep._bench_delta(
                base_index["base"], base_index[1],
                "wikitext2_ppl", 1, rise=True,
            )
            assert abs(delta - (base_index[1]["wikitext2_ppl"]["value"]
                                - base["value"])) < 1e-12
        assert bypass_state(model) is None
        with torch.no_grad():
            after = model(probe_ids).logits
        assert torch.equal(before, after)

    def test_resume_executes_nothing_new():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            run_layer_sweep(
                model, None, [1, 2], out_root, "r", "tiny-qwen",
                **_sweep_kwargs()
            )
            files = sorted(out_root.rglob("*.jsonl"))
            counts = {path: len(load_rows(path)) for path in files}
            assert counts, "first run wrote nothing"
            again = run_layer_sweep(
                model, None, [1, 2], out_root, "r", "tiny-qwen",
                **_sweep_kwargs()
            )
            assert again["executed"] == []
            assert again["skipped"] == [1, 2]
            assert sorted(out_root.rglob("*.jsonl")) == files
            for path, count in counts.items():
                assert len(load_rows(path)) == count, path

    def test_manifest_guard_names_moved_fields():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            run_layer_sweep(
                model, None, [1], out_root, "g", "tiny-qwen",
                **_sweep_kwargs()
            )
            _expect_value_error(
                lambda: run_layer_sweep(
                    model, None, [1], out_root, "g", "tiny-qwen",
                    scenario_seed=7, **_sweep_kwargs()
                ),
                "scenario_seed",
            )
            _expect_value_error(
                lambda: run_layer_sweep(
                    model, None, [1, 2], out_root, "g", "tiny-qwen",
                    **_sweep_kwargs()
                ),
                "layers",
            )
            # An identity-true rerun still passes the guard.
            summary = run_layer_sweep(
                model, None, [1], out_root, "g", "tiny-qwen",
                **_sweep_kwargs()
            )
            assert summary["skipped"] == [1]

    def test_arm_is_stamped_into_manifest_and_rows_and_guarded():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            run_layer_sweep(
                model, None, [1], out_root, "ed", "tiny-qwen", arm="E,D",
                **_sweep_kwargs()
            )
            manifest = json.loads((out_root / "sweep_manifest.json").read_text())
            assert manifest["arm"] == "E,D"
            rows = load_rows(out_root / "ed-l01" / "competence.jsonl")
            assert rows and all(row["arm"] == "E,D" for row in rows)
            # Another arm is a moved identity, refused by name.
            _expect_value_error(
                lambda: run_layer_sweep(
                    model, None, [1], out_root, "ed", "tiny-qwen", arm="I,D",
                    **_sweep_kwargs()
                ),
                "arm",
            )
            # The identity-true rerun resumes.
            summary = run_layer_sweep(
                model, None, [1], out_root, "ed", "tiny-qwen", arm="E,D",
                **_sweep_kwargs()
            )
            assert summary["skipped"] == [1]

    def test_dev_mode_forces_jsd_only():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            # dev=True FORCES jsd_only even when the caller asks for
            # negotiation rows, and the manifest records the mode.
            summary = run_layer_sweep(
                model, None, [1], out_root, "d", "tiny-qwen",
                dev=True,
                **_sweep_kwargs(jsd_only=False)
            )
            assert summary["executed"] == [1]
            assert (out_root / "d-l01" / "competence.jsonl").is_file()
            assert not (out_root / "d-l01" / "rows.jsonl").exists()
            manifest = json.loads(
                (out_root / "sweep_manifest.json").read_text()
            )
            assert manifest["dev"] is True
            assert bypass_state(model) is None

    def test_full_sweep_one_layer_writes_rows():
        model = _tiny_model()
        tokenizer = ChatTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            summary = run_layer_sweep(
                model, tokenizer, [1], out_root, "full", "tiny-qwen",
                n=2, batch_size=2, max_new_tokens=4,
                **_sweep_kwargs(jsd_only=False)
            )
            assert summary["executed"] == [1]
            rows_path = out_root / "full-l01" / "rows.jsonl"
            rows = [
                row for row in load_rows(rows_path)
                if row.get("run_id") == "full-l01"
            ]
            assert len(rows) == 4  # 2 scenarios x 2 conditions
            for row in rows:
                assert row["bypassed_layer"] == 1
                assert row["arm"] is None
                assert row["split"] == "selection"
                assert row["gen_config"]["bypass_impl"] == BYPASS_IMPL
            comp = load_rows(out_root / "full-l01" / "competence.jsonl")
            assert {row["metric"] for row in comp} == {
                "wikitext2_neutral_jsd", "wikitext2_ppl"
            }
            assert len(load_rows(out_root / "base-competence.jsonl")) == 1
            # Probe removed and model intact after the layer.
            assert bypass_state(model) is None
            # Rerun: fully complete, nothing regenerated.
            again = run_layer_sweep(
                model, tokenizer, [1], out_root, "full", "tiny-qwen",
                n=2, batch_size=2, max_new_tokens=4,
                **_sweep_kwargs(jsd_only=False)
            )
            assert again["executed"] == []
            assert again["skipped"] == [1]
            assert len([
                row for row in load_rows(rows_path)
                if row.get("run_id") == "full-l01"
            ]) == 4

    def test_chunked_sessions_complete_one_manifest():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            first = run_layer_sweep(
                model, None, [0, 1], out_root, "c", "tiny-qwen",
                chunk=[0], **_sweep_kwargs()
            )
            assert first["executed"] == [0]
            assert first["skipped"] == []
            second = run_layer_sweep(
                model, None, [0, 1], out_root, "c", "tiny-qwen",
                chunk=[1], **_sweep_kwargs()
            )
            assert second["executed"] == [1]
            assert second["skipped"] == []
            # Whole-sweep pass over the same manifest: everything complete.
            third = run_layer_sweep(
                model, None, [0, 1], out_root, "c", "tiny-qwen",
                **_sweep_kwargs()
            )
            assert third["executed"] == []
            assert third["skipped"] == [0, 1]
            # The intact base row was written exactly once across sessions.
            assert len(load_rows(out_root / "base-competence.jsonl")) == 1
            # A chunk outside the full list refuses by name.
            _expect_value_error(
                lambda: run_layer_sweep(
                    model, None, [0, 1], out_root, "c", "tiny-qwen",
                    chunk=[3], **_sweep_kwargs()
                ),
                "chunk",
            )


    def test_preinstalled_bypass_refused():
        model = _tiny_model()
        probe = install_bypass(model, 2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _expect_value_error(
                    lambda: run_layer_sweep(
                        model, None, [1, 2], Path(tmp) / "sweep", "t",
                        "tiny-qwen", **_sweep_kwargs()
                    ),
                    "already installed",
                )
        finally:
            probe.remove()

    def test_candidate_benchmarks_only_write_requested_metrics():
        from unittest.mock import patch

        from algoverse.utils import append_jsonl

        model = _tiny_model()
        tokenizer = ChatTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            run_layer_sweep(
                model, tokenizer, [1, 2], out_root, "bench", "tiny-qwen",
                **_sweep_kwargs()
            )

            def fake_benchmarks(model_arg, tokenizer_arg, out_path, run_meta,
                                batch_size=4, seed=42):
                assert model_arg is model and tokenizer_arg is tokenizer
                assert bypass_state(model)["layer_idx"] == 2
                for metric, value in (
                    ("mmlu_acc", 0.7), ("gsm8k_exact_match", 0.6)
                ):
                    row = dict(run_meta)
                    row.update({
                        "metric": metric, "value": value, "stderr": 0.01,
                        "config": {"limit": 1, "seed": seed},
                    })
                    append_jsonl(out_path, row)
                return {"mmlu_acc": 0.7, "gsm8k_exact_match": 0.6}

            with patch(
                "algoverse.eval.run_lm_eval_benchmarks", fake_benchmarks
            ):
                result = run_candidate_benchmarks(
                    model, tokenizer, [2], out_root, "bench", "tiny-qwen"
                )
            assert set(result["written"]) == {2}
            rows = load_rows(out_root / "bench-l02" / "competence.jsonl")
            assert {row["metric"] for row in rows} == {
                "wikitext2_neutral_jsd", "wikitext2_ppl",
                "mmlu_acc", "gsm8k_exact_match",
            }
            assert bypass_state(model) is None
            (out_root / "bench-l01").rename(out_root / "removed-l01")
            _expect_value_error(
                lambda: run_candidate_benchmarks(
                    model, tokenizer, [1], out_root, "bench", "tiny-qwen"
                ),
                "existing sweep directory",
            )


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=SWEEPDRIVER_TEST_COUNT,
                              missing=MISSING_STACK))
