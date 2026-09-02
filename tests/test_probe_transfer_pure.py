"""Rung-1 tests for scripts/run_probe_transfer.py's pure pieces.

The ways the transfer script's new modes can silently lie, each caught:
  1. a bare path mistaken for a LABEL=PATH spec (or vice versa);
  2. final_prompt reading the FIRST response token instead of the last
     prompt token, or accepting an example with no prompt token at all;
  3. --out-root outputs colliding (fit/label not in the run id);
  4. a legacy invocation drifting from the diag-probe3 config;
  5. the new schema adding keys silently (or forgetting one);
  6. --use-fit unpickling a fit made for another position/model/dataset;
  7. the claim-line stripper keeping the LAST marker's line, or dropping
     an unmarked response;
  8. the exclude-final-line builder scoring the claim tokens anyway.

Stdlib only (no numpy/sklearn: those paths live in test_probe_transfer.py).

    python3 tests/test_probe_transfer_pure.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from algoverse.corroboration import (  # noqa: E402
    PROBE_RECIPE,
    probe_examples_from_rows,
    strip_final_claim_line,
)
from algoverse.tasks import get_scenarios  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_probe_transfer_script", REPO / "scripts" / "run_probe_transfer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ids(list):
    def tolist(self):
        return list(self)


class FakeTokenizer:
    """Pure-Python word-split tokenizer with the two methods the builders use."""

    def __call__(self, text, return_tensors="pt", add_special_tokens=False,
                 **kwargs):
        return {"input_ids": [_Ids(text.split())]}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        rendered = "\n".join(
            "%s %s" % (message["role"], message["content"])
            for message in messages
        )
        return rendered + ("\nassistant\n" if add_generation_prompt else "")


def _row(scenario_id, deceptive, response_text, condition="incentive",
         valid=True):
    return {"condition": condition, "valid": valid,
            "deceptive": deceptive if valid else None,
            "scenario_id": scenario_id, "response_text": response_text}


def test_parse_test_rows_spec_label_vs_path():
    script = _load_script()
    assert script.parse_test_rows_spec("d1=/x/rows.jsonl") == ("d1", "/x/rows.jsonl")
    assert script.parse_test_rows_spec("/x/rows.jsonl") == (None, "/x/rows.jsonl")
    # An "=" after a slash belongs to the path, not a label.
    assert script.parse_test_rows_spec("/x/a=b/rows.jsonl") == (None, "/x/a=b/rows.jsonl")
    assert script.parse_test_rows_spec("=rows.jsonl") == (None, "=rows.jsonl")


def test_capture_spans_final_prompt_shifts_and_refuses_zero_start():
    script = _load_script()
    examples = [{"response_start": 43}, {"response_start": 45}]
    assert script.capture_spans(examples, "response_tokens") == ([43, 45], None)
    assert script.capture_spans(examples, "response_excl_claim") == ([43, 45], None)
    assert script.capture_spans(examples, "final_prompt") == ([42, 44], 1)
    try:
        script.capture_spans([{"response_start": 5}, {}], "final_prompt")
    except SystemExit as exc:
        assert "example 1" in str(exc) and "response_start=0" in str(exc)
    else:
        raise AssertionError("final_prompt accepted an example with no prompt token")


def test_plan_outputs_run_id_and_paths():
    script = _load_script()
    outputs = script.plan_outputs("base", None, "/r", ["d1", "d2"], ["own", "fixed"])
    assert [o["run_id"] for o in outputs] == [
        "base-own-d1", "base-own-d2", "base-fixed-d1", "base-fixed-d2"]
    assert outputs[0]["out_path"] == Path("/r/base-own-d1/interp.jsonl")
    assert outputs[3]["fit"] == "fixed" and outputs[3]["label"] == "d2"
    legacy = script.plan_outputs("verbatim", "/o", None, [None], ["own"])
    assert legacy == [{"fit": "own", "label": None, "run_id": "verbatim",
                       "out_path": Path("/o/interp.jsonl")}]
    try:
        script.plan_outputs("x", "/o", None, ["a", "b"], ["own"])
    except ValueError as exc:
        assert "exactly one output" in str(exc)
    else:
        raise AssertionError("--out-dir accepted two test sets")


def _config(script, **overrides):
    kwargs = dict(
        train_dataset="/d/pairs.jsonl", test_path="/r/rows.jsonl", n_train=612,
        n_test=100, n_test_lied=51, fit_tag="own",
        feature_position="response_tokens", span_len=None,
        exclude_final_line=False, stats={}, label=None, fit_meta=None,
        legacy=True,
    )
    kwargs.update(overrides)
    return script.build_config(**kwargs)


def test_legacy_config_is_byte_identical_to_old_schema():
    script = _load_script()
    config = _config(script)
    assert tuple(config) == script.LEGACY_CONFIG_KEYS
    assert config["status"] == "exploratory-diagnostic; unratified"
    assert config["fit"] == "all_train_examples_no_holdout"
    assert config["transfer"] is True and config["n_train"] == 612
    assert config["label_source"] == (
        "transfer:probe_dataset:/d/pairs.jsonl->within_incentive_rows:/r/rows.jsonl")
    for key, value in PROBE_RECIPE.items():
        assert config[key] == value


def test_new_schema_adds_exactly_the_documented_keys():
    script = _load_script()
    own = _config(script, legacy=False, feature_position="final_prompt",
                  span_len=1, label="d1",
                  stats={"n_no_marker": 2, "n_skipped_empty_body": 1})
    assert set(own) - set(script.LEGACY_CONFIG_KEYS) == set(script.NEW_CONFIG_KEYS)
    assert own["feature_position"] == "final_prompt_token"
    assert own["span_len"] == 1 and own["exclude_final_line"] is False
    assert own["n_test_no_marker"] == 2 and own["n_test_skipped_empty_body"] == 1
    assert own["test_rows_label"] == "d1" and own["fit_source"] is None
    assert own["status"] == "exploratory-diagnostic; unratified"
    meta = {"fit_run_id": "diag-probe4-fp-m0-qwen7b", "model_id": "m",
            "adapter_path": None, "checkpoint_step": None, "train_seed": None,
            "feature_position": "final_prompt_token", "n_train": 612,
            "extra": "ignored"}
    fixed = _config(script, legacy=False, feature_position="final_prompt",
                    span_len=1, label="d1", fit_tag="fixed", fit_meta=meta,
                    n_train=999)
    assert fixed["fit"] == "fixed_direction_from:diag-probe4-fp-m0-qwen7b"
    assert fixed["n_train"] == 612                      # from the fit, not this run
    assert fixed["fit_source"] == {k: meta[k] for k in script.FIT_SOURCE_FIELDS}
    excl = _config(script, legacy=False, feature_position="response_excl_claim",
                   exclude_final_line=True, label="d2")
    assert excl["feature_position"] == "mean_response_tokens_excl_claim_line"
    assert excl["exclude_final_line"] is True and excl["span_len"] is None


def test_load_fit_refusals_before_unpickling():
    script = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        fit_dir = Path(tmp) / "fit"
        fit_dir.mkdir()
        good = dict(fit_run_id="m0", model_id="m", n_layers=4, hidden_size=32,
                    train_dataset_basename="pairs.jsonl",
                    feature_position="final_prompt_token", n_train=12)
        kwargs = dict(model_id="m", feature_position="final_prompt", n_layers=4,
                      hidden_size=32, train_dataset="/d/pairs.jsonl",
                      layers_needed=[0, 3])
        try:
            script.load_fit(fit_dir, **kwargs)
        except SystemExit as exc:
            assert "fit_meta.json" in str(exc)
        else:
            raise AssertionError("missing fit_meta accepted")
        for field, wrong in (("feature_position", "mean_response_tokens"),
                             ("model_id", "other"), ("n_layers", 5),
                             ("hidden_size", 64),
                             ("train_dataset_basename", "other.jsonl")):
            (fit_dir / "fit_meta.json").write_text(json.dumps({**good, field: wrong}))
            try:
                script.load_fit(fit_dir, **kwargs)
            except SystemExit as exc:
                assert field in str(exc), (field, str(exc))
            else:
                raise AssertionError("mismatched %s accepted" % field)
        (fit_dir / "fit_meta.json").write_text(json.dumps(good))
        (fit_dir / "layer-000.joblib").write_bytes(b"placeholder")
        try:
            script.load_fit(fit_dir, **kwargs)
        except SystemExit as exc:
            assert "missing layer files" in str(exc) and "[3]" in str(exc)
        else:
            raise AssertionError("missing layer file accepted")


def test_strip_final_claim_line_last_marker_and_no_marker():
    two = ("first thoughts\nMY BEST OUTSIDE OFFER: $100,000\nactually no\n"
           "MY BEST OUTSIDE OFFER: $120,000")
    assert strip_final_claim_line(two) == (
        "first thoughts\nMY BEST OUTSIDE OFFER: $100,000\nactually no", True)
    assert strip_final_claim_line("just prose, no marker") == ("just prose, no marker", False)
    assert strip_final_claim_line("MY BEST OUTSIDE OFFER: NONE") == ("", True)
    assert strip_final_claim_line("body   \n\nmy best outside offer: $5") == ("body", True)
    assert strip_final_claim_line(None) == ("", False)


def test_probe_examples_exclude_final_line_with_fake_tokenizer():
    tokenizer = FakeTokenizer()
    scenarios = get_scenarios("selection", n=3, seed=42)
    rows = [
        _row(scenarios[0]["scenario_id"], True,
             "my offer is large\nMY BEST OUTSIDE OFFER: $120,000"),
        _row(scenarios[1]["scenario_id"], False, "MY BEST OUTSIDE OFFER: NONE"),
        _row(scenarios[2]["scenario_id"], False, "no marker at all here"),
    ]
    stats = {}
    default = probe_examples_from_rows(rows, tokenizer, stats=stats)
    assert [ex["text"].endswith(r["response_text"]) for ex, r in zip(default, rows)] == [True] * 3
    assert stats == {"n_labeled": 3, "n_examples": 3, "n_no_marker": 0,
                     "n_skipped_empty_body": 0}
    stats = {}
    stripped = probe_examples_from_rows(rows, tokenizer, exclude_final_line=True,
                                        stats=stats)
    # 8: the claim tokens are gone; the claim-only row is skipped; the
    # unmarked row survives unchanged and is counted.
    assert len(stripped) == 2
    assert stripped[0]["text"].endswith("my offer is large")
    # The PROMPT legitimately carries the marker (it instructs the format);
    # only the response half must be claim-free.
    assert "OUTSIDE OFFER" not in stripped[0]["text"].split("\nassistant\n")[-1]
    assert stripped[1]["text"].endswith("no marker at all here")
    assert [ex["label"] for ex in stripped] == [True, False]
    assert stats == {"n_labeled": 3, "n_examples": 2, "n_no_marker": 1,
                     "n_skipped_empty_body": 1}
    # The span still starts exactly at the prompt boundary of the NEW text.
    for example in stripped:
        n_prompt = example["response_start"]
        assert example["text"].split()[n_prompt - 1] == "assistant"
    try:
        probe_examples_from_rows([rows[1]], tokenizer, exclude_final_line=True)
    except ValueError as exc:
        assert "every labeled row was skipped" in str(exc)
    else:
        raise AssertionError("a test set with nothing left to read was accepted")


if __name__ == "__main__":
    import traceback

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
