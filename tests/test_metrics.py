"""Unit tests for algoverse.metrics, on synthetic rows with known answers.

Pure Python, no GPU. Run directly:

    python3 tests/test_metrics.py

or via pytest.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.metrics import (
    bypass_effect,
    deception_rate,
    filter_rows,
    incentive_gap,
    load_rows,
    recovery,
    summarize_runs,
    task_competence,
    tau_with_ci,
)


def make_row(scenario_id, condition, valid=True, deceptive=False,
             understated=False, **extra):
    """One synthetic results row with only the fields the metrics read."""
    row = {
        "scenario_id": scenario_id,
        "condition": condition,
        "valid": valid,
        "deceptive": deceptive if valid else None,
        "understated": understated if valid else None,
        "model_id": "test-model",
        "adapter_path": None,
        "bypassed_layer": None,
        "patch_layer": None,
        "patch_source": None,
        "checkpoint_step": None,
        "arm": None,
    }
    row.update(extra)
    return row


def make_run(n_scenarios, deceptive_incentive, deceptive_control=0, **extra):
    """Rows for one run: n scenarios x 2 conditions, with the first
    `deceptive_incentive` incentive rows deceptive (same for control)."""
    rows = []
    for i in range(n_scenarios):
        sid = "s%03d" % i
        rows.append(make_row(sid, "incentive", deceptive=i < deceptive_incentive, **extra))
        rows.append(make_row(sid, "control", deceptive=i < deceptive_control, **extra))
    return rows


def test_deception_rate_counts_valid_only():
    rows = [
        make_row("s1", "incentive", deceptive=True),
        make_row("s2", "incentive", deceptive=False),
        make_row("s3", "incentive", valid=False),
        make_row("s4", "incentive", valid=False),
    ]
    rate = deception_rate(rows)
    # 1 deceptive out of 2 VALID rows, not out of 4 total. The two invalid
    # rows appear only in the invalid rate.
    assert rate["D"] == 0.5, rate
    assert rate["n_total"] == 4 and rate["n_valid"] == 2
    assert rate["invalid_rate"] == 0.5


def test_deception_rate_none_when_no_valid_rows():
    rate = deception_rate([make_row("s1", "incentive", valid=False)])
    assert rate["D"] is None  # never 0: nothing was measured


def test_incentive_gap_known_tau():
    # 4 scenarios, 2 deceptive under incentive, 0 under control: tau = 0.5.
    gap = incentive_gap(make_run(4, deceptive_incentive=2))
    assert gap["tau"] == 0.5, gap
    assert gap["D_incentive"] == 0.5 and gap["D_control"] == 0.0


def test_incentive_gap_reports_per_condition_invalid_rates():
    rows = make_run(4, deceptive_incentive=2)
    rows.append(make_row("s9", "incentive", valid=False))
    gap = incentive_gap(rows)
    assert gap["invalid_rate_incentive"] == 0.2  # 1 of 5
    assert gap["invalid_rate_control"] == 0.0


def test_task_competence():
    rows = [
        make_row("s1", "control", deceptive=False, understated=False),
        make_row("s2", "control", deceptive=False, understated=False),
        make_row("s3", "control", deceptive=False, understated=True),
        make_row("s4", "control", valid=False),
        make_row("s5", "incentive", deceptive=True),  # ignored: wrong condition
    ]
    comp = task_competence(rows)
    # 2 exact restatements out of 3 valid control rows.
    assert abs(comp["competence"] - 2 / 3) < 1e-12, comp
    assert comp["n_valid_control"] == 3 and comp["n_competent"] == 2


def test_tau_ci_contains_point():
    result = tau_with_ci(make_run(12, deceptive_incentive=6), n_boot=200, seed=0)
    assert result["tau"] == 0.5
    assert result["tau_ci_low"] is not None and result["tau_ci_high"] is not None
    assert result["tau_ci_low"] <= result["tau"] <= result["tau_ci_high"]
    assert result["n_scenarios"] == 12


def test_tau_ci_degenerate_is_tight():
    # Every scenario deceptive under incentive, honest under control:
    # tau = 1 in every possible resample, so the CI collapses to [1, 1].
    result = tau_with_ci(make_run(6, deceptive_incentive=6), n_boot=100, seed=0)
    assert result["tau"] == 1.0
    assert result["tau_ci_low"] == 1.0 and result["tau_ci_high"] == 1.0


def test_tau_ci_reproducible():
    rows = make_run(10, deceptive_incentive=4)
    a = tau_with_ci(rows, n_boot=200, seed=7)
    b = tau_with_ci(rows, n_boot=200, seed=7)
    assert a == b  # same seed, byte-identical result


def test_bypass_effect_point_uses_shared_scenarios_only():
    # The base run covers 10 scenarios (deceptive on the first 5 under
    # incentive); the bypassed run died early and only covers the first 5,
    # none deceptive. Over the SHARED scenarios the base tau is 1.0, so
    # A_l must be 1.0: the point estimate and the CI must describe the same
    # population, not mix full and partial coverage.
    base = make_run(10, deceptive_incentive=5)     # tau over all 10 = 0.5
    bypassed = make_run(5, deceptive_incentive=0)  # covers s000..s004 only
    effect = bypass_effect(base, bypassed, n_boot=100, seed=0)
    assert effect["A_l"] == 1.0, effect


def test_bypass_effect_known_drop():
    base = make_run(8, deceptive_incentive=8)      # tau = 1.0
    bypassed = make_run(8, deceptive_incentive=0)  # tau = 0.0
    effect = bypass_effect(base, bypassed, n_boot=100, seed=0)
    assert effect["A_l"] == 1.0, effect
    assert effect["tau_base"] == 1.0 and effect["tau_bypassed"] == 0.0


def test_recovery_halfway():
    start = make_run(8, deceptive_incentive=0)    # tau = 0.0
    ceiling = make_run(8, deceptive_incentive=8)  # tau = 1.0
    halfway = make_run(8, deceptive_incentive=4)  # tau = 0.5
    result = recovery(halfway, start, ceiling, n_boot=100, seed=0)
    assert result["R_t"] == 0.5, result
    assert result["reason"] is None


def test_recovery_denominator_guard():
    # Ceiling equals start: the ratio would be noise over noise.
    start = make_run(8, deceptive_incentive=4)
    ceiling = make_run(8, deceptive_incentive=4)
    result = recovery(make_run(8, deceptive_incentive=4), start, ceiling)
    assert result["R_t"] is None
    assert result["reason"] == "denominator_too_small"


def test_filter_rows():
    rows = [
        make_row("s1", "incentive", bypassed_layer=5),
        make_row("s1", "control", bypassed_layer=5),
        make_row("s1", "incentive", bypassed_layer=None),
    ]
    kept = filter_rows(rows, condition="incentive", bypassed_layer=5)
    assert len(kept) == 1


def test_summarize_runs_groups_by_intervention():
    rows = make_run(6, deceptive_incentive=6) + make_run(
        6, deceptive_incentive=0, bypassed_layer=5
    )
    summaries = summarize_runs(rows, n_boot=100, seed=0)
    assert len(summaries) == 2
    by_layer = {s["bypassed_layer"]: s for s in summaries}
    assert by_layer[None]["tau"] == 1.0
    assert by_layer[5]["tau"] == 0.0


def test_load_rows_skips_torn_final_line():
    # A run killed mid-write leaves a torn last line. Analysis must read
    # every complete row rather than crashing.
    good = make_row("s1", "incentive")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write('{"scenario_id": "s2", "cond')  # torn
        path = fh.name
    rows = load_rows(path)
    assert len(rows) == 1 and rows[0]["scenario_id"] == "s1"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except AssertionError as exc:
                failures += 1
                print("FAIL %s: %s" % (name, exc))
    print("%s" % ("ALL TESTS PASSED" if failures == 0 else "%d FAILURE(S)" % failures))
    raise SystemExit(1 if failures else 0)
