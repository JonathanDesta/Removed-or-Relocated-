"""Pure-Python (rung-1) acceptance tests for the corroboration driver.

No ML stack: aggregation math per goldowskydill2025detecting's variant,
label extraction from synthetic rows, interp.jsonl row shape, and the
identity/resume guard with stubbed capture/probe.

Run: python3 tests/test_corroboration_pure.py
"""

import json
import tempfile
import types
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.corroboration import (
    PROBE_RECIPE,
    _interp_done,
    aggregate_response_scores,
    labeled_incentive_rows,
    load_probe_dataset,
    run_probe_auroc,
    write_interp_row,
)

RUN_META = {
    "run_id": "corr-test",
    "model_id": "tiny",
    "adapter_path": None,
    "bypassed_layer": None,
    "checkpoint_step": None,
    "arm": None,
    "train_seed": None,
    "bypass_impl": None,
}


def _fake_model(n_layers=3):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(num_hidden_layers=n_layers)
    )


def test_aggregate_response_scores_is_mean_per_response():
    # The paper's variant: a real-valued score at every token, then the
    # MEAN across tokens gives one score per response — never a pooled
    # activation, never a last-token read.
    scores = aggregate_response_scores([[1.0, 2.0, 3.0], [-4.0], [0.5, 1.5]])
    assert scores == [2.0, -4.0, 1.0], scores

    # Alignment is positional and empty responses are a hard error, not a
    # silent drop that would shift every later response's label.
    try:
        aggregate_response_scores([[1.0], []])
    except ValueError as exc:
        assert "response 1" in str(exc)
    else:
        raise AssertionError("empty response aggregated silently")

    assert PROBE_RECIPE["aggregation"] == "mean_response_token_score"
    assert PROBE_RECIPE["test_size"] == 0.3
    assert PROBE_RECIPE["random_state"] == 0
    assert PROBE_RECIPE["max_iter"] == 1000
    assert PROBE_RECIPE["C"] == 0.1


def test_labeled_incentive_rows_filters_and_labels():
    def row(**overrides):
        base = {
            "condition": "incentive",
            "valid": True,
            "deceptive": True,
            "scenario_id": "s1",
            "response_text": "reply",
        }
        base.update(overrides)
        return base

    rows = [
        row(scenario_id="lied"),
        row(scenario_id="honest", deceptive=False),
        row(scenario_id="ctrl", condition="control"),          # wrong condition
        row(scenario_id="invalid", valid=False, deceptive=None),
        row(scenario_id="refusal", valid=False, deceptive=None,
            invalid_reason="refusal"),
        row(scenario_id="null-dec", deceptive=None),           # malformed
        row(scenario_id="truthy", deceptive=1),                # not a bool
        row(scenario_id="empty", response_text=""),
    ]
    labeled = labeled_incentive_rows(rows)
    assert [item["scenario_id"] for item in labeled] == ["lied", "honest"]
    assert labeled[0]["label"] is True
    assert labeled[1]["label"] is False
    assert all(item["group"] == item["scenario_id"] for item in labeled)


def test_write_interp_row_shape_and_nan_handling():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "interp.jsonl"
        write_interp_row(
            out, RUN_META, "probe_auroc", 2, 0.9, 0.8, 1.0,
            {"n": 4}, extra={"accuracy": 0.75},
        )
        write_interp_row(
            out, RUN_META, "attention_jsd", 0, float("nan"), None,
            float("nan"), {"n_boot": 10},
        )
        lines = out.read_text().strip().splitlines()
        rows = [json.loads(line) for line in lines]  # strict: valid JSON only
        assert len(rows) == 2
        first, second = rows
        for field in RUN_META:
            assert first[field] == RUN_META[field]
        assert first["analysis"] == "probe_auroc" and first["layer"] == 2
        assert first["value"] == 0.9 and first["accuracy"] == 0.75
        assert (first["ci_low"], first["ci_high"]) == (0.8, 1.0)
        assert first["config"] == {"n": 4}
        # NaN must serialize as JSON null, never the invalid literal NaN.
        assert second["value"] is None and second["ci_high"] is None
        assert "NaN" not in lines[1]
        assert "accuracy" not in second

        try:
            write_interp_row(out, RUN_META, "probe_auroc", 0, 0.5, None,
                             None, {}, extra={"value": 1.0})
        except ValueError:
            pass
        else:
            raise AssertionError("extra field collision was not refused")


def test_interp_done_resume_and_identity_guards():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "interp.jsonl"
        config = {"n": 4, "label_source": "rows:x"}
        assert not _interp_done(out, RUN_META, "probe_auroc", 0, config)
        write_interp_row(out, RUN_META, "probe_auroc", 0, 0.9, None, None,
                         config, extra={"accuracy": 0.5})
        assert _interp_done(out, RUN_META, "probe_auroc", 0, config)
        # Same layer, other analysis; other layer, same analysis: not done.
        assert not _interp_done(out, RUN_META, "attention_jsd", 0, config)
        assert not _interp_done(out, RUN_META, "probe_auroc", 1, config)
        # Extra recorded config keys (structural-null flags, ci notes) must
        # not break resume against the base config...
        flagged = dict(config)
        flagged["excluded_bypassed_layer"] = True
        write_interp_row(out, RUN_META, "probe_auroc", 1, None, None, None,
                         flagged)
        assert _interp_done(out, RUN_META, "probe_auroc", 1, config)
        # ...but a changed value of a shared key raises, never forks.
        try:
            _interp_done(out, RUN_META, "probe_auroc", 0, {"n": 5})
        except ValueError as exc:
            assert "config.n" in str(exc)
        else:
            raise AssertionError("config mismatch passed silently")
        # Run identity (any non-result field) must match exactly.
        changed = dict(RUN_META)
        changed["model_id"] = "other"
        try:
            _interp_done(out, changed, "probe_auroc", 0, config)
        except ValueError as exc:
            assert "model_id" in str(exc)
        else:
            raise AssertionError("identity mismatch passed silently")
        # A different run_id is a different run, not a mismatch.
        other_run = dict(RUN_META)
        other_run["run_id"] = "corr-other"
        assert not _interp_done(out, other_run, "probe_auroc", 0, config)


def test_run_probe_auroc_rows_resume_and_bypassed_exclusion():
    captures = []
    probes = []

    def capture(model, tokenizer, texts, starts):
        captures.append(list(texts))
        # 3 layers; layer 1 is bypassed (None per the exclusion rule).
        features = [[[0.0]] * len(texts), None, [[0.0]] * len(texts)]
        return features

    def probe(layer_features, labels, groups):
        probes.append((layer_features, list(labels), list(groups)))
        return None, {"auroc": 0.9, "auroc_ci": (0.7, None),
                      "accuracy": 0.8}

    examples = [
        {"text": "t%d" % i, "response_start": 0,
         "label": i % 2 == 0, "group": "g%d" % (i // 2)}
        for i in range(6)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "interp.jsonl"
        written = run_probe_auroc(
            _fake_model(3), None, examples, out, RUN_META, "rows:test",
            _capture=capture, _probe=probe,
        )
        assert written == {0: 0.9, 1: None, 2: 0.9}
        rows = [json.loads(line) for line in
                out.read_text().strip().splitlines()]
        assert [row["layer"] for row in rows] == [0, 1, 2]
        assert all(row["analysis"] == "probe_auroc" for row in rows)
        for row in rows:
            assert row["config"]["n"] == 6
            assert row["config"]["label_source"] == "rows:test"
            for key, value in PROBE_RECIPE.items():
                assert row["config"][key] == value
        # Bypassed layer: structural null, flagged in config.
        assert rows[1]["value"] is None and rows[1]["accuracy"] is None
        assert rows[1]["config"]["excluded_bypassed_layer"] is True
        # Degenerate bootstrap (one side None): ci nulled with a config note.
        assert rows[0]["ci_low"] is None and rows[0]["ci_high"] is None
        assert "bootstrap" in rows[0]["config"]["ci"]
        assert rows[0]["accuracy"] == 0.8
        assert len(captures) == 1 and len(probes) == 2

        # Resume: everything done — no new capture, no new probe, no rows.
        again = run_probe_auroc(
            _fake_model(3), None, examples, out, RUN_META, "rows:test",
            _capture=capture, _probe=probe,
        )
        assert again == {}
        assert len(captures) == 1 and len(probes) == 2
        assert len(out.read_text().strip().splitlines()) == 3

        # A different label source is a different config: refuse to mix
        # silently? No — label_source is a NEW key value, so it raises.
        try:
            run_probe_auroc(
                _fake_model(3), None, examples, out, RUN_META,
                "rows:other", _capture=capture, _probe=probe,
            )
        except ValueError as exc:
            assert "label_source" in str(exc)
        else:
            raise AssertionError("label-source change passed silently")


def test_load_probe_dataset_strict_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.jsonl"
        path.write_text(
            json.dumps({"text": "a true statement", "label": True,
                        "group": "g1"}) + "\n" +
            json.dumps({"text": "a false statement", "label": False,
                        "group": "g2"}) + "\n"
        )
        examples = load_probe_dataset(path)
        assert len(examples) == 2
        assert all(example["response_start"] == 0 for example in examples)
        assert examples[0]["label"] is True and examples[1]["label"] is False
        assert [example["group"] for example in examples] == ["g1", "g2"]

        for bad in (
            json.dumps({"text": "x", "label": 1, "group": "g"}),   # non-bool
            json.dumps({"text": "x", "label": True}),              # no group
            json.dumps({"text": "", "label": True, "group": "g"}),  # empty
            "{torn",                                               # not JSON
        ):
            path.write_text(bad + "\n")
            try:
                load_probe_dataset(path)
            except ValueError:
                pass
            else:
                raise AssertionError("bad line accepted: %r" % bad)

        path.write_text("\n")
        try:
            load_probe_dataset(path)
        except ValueError as exc:
            assert "no examples" in str(exc)
        else:
            raise AssertionError("empty dataset accepted")


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except Exception as exc:
                failures += 1
                print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
                traceback.print_exc()
    print("%s" % ("ALL TESTS PASSED" if failures == 0
                  else "%d FAILURE(S)" % failures))
    raise SystemExit(1 if failures else 0)
