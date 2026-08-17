"""Guarded rung-2 tests for the layer-sweep driver (WP-S3, plan D1/D2/D5).

Tiny random CPU models only — this suite must never run on a GPU. The
negotiation leg reuses test_bypass.py's stub-chat-tokenizer fixture style
so run_negotiation_eval's real code path executes on CPU.

Run: ~/.venvs/colab-local/bin/python tests/test_sweepdriver.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SWEEPDRIVER_TEST_COUNT = 10

try:
    import torch
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.metrics import load_rows
    from algoverse import sweep
    from algoverse.models import BYPASS_IMPL, bypass_state, install_bypass
    from algoverse.sweepdriver import (
        COMPARED,
        reconcile_permanent_bypass,
        run_candidate_benchmarks,
        run_layer_sweep,
    )

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    def _tiny_model():
        torch.manual_seed(0)
        config = Qwen2Config(
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
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    def _ids(seq_len=48):
        torch.manual_seed(1)
        return torch.randint(3, 128, (1, seq_len))

    # test_bypass.py's chat-tokenizer stub (test_generation_wiring_...):
    # rejects the system role (exercising the fold path), encodes every
    # prompt to the same tiny id row, and decodes to a well-formed offer
    # line so the scorer's regex path succeeds without any LLM fallback.
    class ChatTokenizer:
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "right"

        def __init__(self):
            self.encode_kwargs = []
            self.successful_messages = []

        def apply_chat_template(self, messages, **kwargs):
            if any(message["role"] == "system" for message in messages):
                raise ValueError("System role not supported")
            self.successful_messages.append(messages)
            return "<bos> rendered prompt"

        def __call__(self, texts, **kwargs):
            self.encode_kwargs.append(kwargs)
            texts = [texts] if isinstance(texts, str) else texts
            ids = torch.tensor([[1, 5, 6] for _ in texts], dtype=torch.long)
            mask = torch.ones_like(ids)
            return BatchEncoding(
                {"input_ids": ids, "attention_mask": mask}, tensor_type="pt"
            )

        def decode(self, tokens, skip_special_tokens=True):
            return "MY BEST OUTSIDE OFFER: $82,500"

    def _sweep_kwargs(**overrides):
        options = {
            "item16_decision": "item-16 DEV calibration recorded: 0.25 "
                               "nats confirmed (test fixture)",
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

    def test_item16_tripwire_and_dev_mode():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "sweep"
            _expect_value_error(
                lambda: run_layer_sweep(
                    model, None, [1], out_root, "d", "tiny-qwen",
                    **_sweep_kwargs(item16_decision=None)
                ),
                "item16_decision",
            )
            _expect_value_error(
                lambda: run_layer_sweep(
                    model, None, [1], out_root, "d", "tiny-qwen",
                    **_sweep_kwargs(item16_decision="   ")
                ),
                "item16_decision",
            )
            # The tripwire fires before ANYTHING is written.
            assert not out_root.exists()
            # dev=True needs no decision and FORCES jsd_only even when the
            # caller asks for negotiation rows.
            summary = run_layer_sweep(
                model, None, [1], out_root, "d", "tiny-qwen",
                dev=True,
                **_sweep_kwargs(item16_decision=None, jsd_only=False)
            )
            assert summary["executed"] == [1]
            assert (out_root / "d-l01" / "competence.jsonl").is_file()
            assert not (out_root / "d-l01" / "rows.jsonl").exists()
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


    def test_lesioned_checkpoint_sweep_skips_its_layer():
        # Stage-3 shape: the permanent lesion stays installed for the whole
        # sweep (it is the baseline), its own layer is skipped structurally
        # (ratified P-S4), and the manifest records the lesion as identity.
        model = _tiny_model()
        tokenizer = ChatTokenizer()
        permanent = install_bypass(model, 1, role="permanent")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_root = Path(tmp) / "sweep"
                summary = run_layer_sweep(
                    model, tokenizer, [1, 2, 3], out_root, "s3", "tiny-qwen",
                    n=2, batch_size=2, max_new_tokens=4,
                    **_sweep_kwargs(jsd_only=False)
                )
                assert summary["executed"] == [2, 3]
                assert summary["lesioned_skipped"] == [1]
                assert summary["permanent_bypassed_layer"] == 1
                base_rows = load_rows(summary["base_rows"])
                assert len(base_rows) == 4
                assert all(row["bypassed_layer"] is None for row in base_rows)
                assert all(
                    row["gen_config"]["permanent_bypassed_layer"] == 1
                    for row in base_rows
                )
                manifest = json.loads(
                    (out_root / "sweep_manifest.json").read_text()
                )
                assert manifest["permanent_bypassed_layer"] == 1
                assert not (out_root / "s3-l01").exists()
                # The lesion survived the whole sweep.
                state = bypass_state(model)
                assert state["permanent"]["layer_idx"] == 1
                assert state["probe"] is None
                # Rerunning an INTACT model against this manifest refuses:
                # the lesion is sweep identity.
                intact = _tiny_model()
                try:
                    run_layer_sweep(
                        intact, tokenizer, [1, 2, 3], out_root, "s3",
                        "tiny-qwen", n=2, batch_size=2, max_new_tokens=4,
                        **_sweep_kwargs(jsd_only=False)
                    )
                except ValueError as exc:
                    assert "permanent_bypassed_layer" in str(exc)
                else:
                    raise AssertionError(
                        "intact rerun against a lesioned manifest passed"
                    )
        finally:
            permanent.remove()

    def test_preinstalled_probe_refused():
        model = _tiny_model()
        probe = install_bypass(model, 2, role="probe")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _expect_value_error(
                    lambda: run_layer_sweep(
                        model, None, [1, 2], Path(tmp) / "sweep", "t",
                        "tiny-qwen", **_sweep_kwargs()
                    ),
                    "probe",
                )
        finally:
            probe.remove()

    def test_explicit_permanent_reconciliation():
        model = _tiny_model()
        handle = reconcile_permanent_bypass(model, 1)
        try:
            assert bypass_state(model)["permanent"]["layer_idx"] == 1
            assert reconcile_permanent_bypass(model, 1) is None
            _expect_value_error(
                lambda: reconcile_permanent_bypass(model, 2), "conflicts"
            )
        finally:
            handle.remove()

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
                assert bypass_state(model)["probe"]["layer_idx"] == 2
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
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_sweepdriver.py needs torch + transformers "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == SWEEPDRIVER_TEST_COUNT, (
        "expected %d tests, found %d" % (SWEEPDRIVER_TEST_COUNT, len(tests))
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
