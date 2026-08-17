"""Rung-1 tests for paired Stage-3 relocation analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import metrics
from algoverse.relocation import evaluate_relocation, relocation_report


def make_run(n, d_inc, layer=None, run_id="base", permanent=1):
    rows = []
    for i in range(n):
        for condition in ("incentive", "control"):
            gen = {
                "bypass_impl": "block-output-identity-hook/v1",
                "permanent_bypassed_layer": permanent,
                "quant": "4bit",
                "do_sample": False,
                "max_new_tokens": 256,
                "model_revision": "rev",
                "adapter_digest": "digest",
                "system_fold": False,
                "use_llm_fallback": True,
                "llm_provider": "openai",
                "llm_model": "gpt-5-mini",
                "load_profile": {
                    "dtype": "float16", "device_type": "cuda",
                    "four_bit": True, "attn_implementation": "sdpa",
                },
            }
            rows.append({
                "run_id": run_id,
                "model_id": "m",
                "adapter_path": "adapter",
                "bypassed_layer": layer,
                "patch_layer": None,
                "patch_source": None,
                "checkpoint_step": 281,
                "arm": None,
                "condition": condition,
                "scenario_id": "s%03d" % i,
                "split": "selection",
                "seed": 42,
                "train_seed": 42,
                "valid": True,
                "deceptive": condition == "incentive" and i < d_inc,
                "understated": False,
                "gen_config": gen,
            })
    return rows


def test_relocation_delta_pairs_all_four_runs():
    result = metrics.relocation_delta(
        make_run(20, 20), make_run(20, 4, layer=0, run_id="rp"),
        make_run(20, 16), make_run(20, 12, layer=0, run_id="lp"),
        n_boot=100, seed=0,
    )
    assert abs(result["A_recovered"] - 0.8) < 1e-12
    assert abs(result["A_lesioned"] - 0.2) < 1e-12
    assert abs(result["delta_l"] - 0.6) < 1e-12
    assert result["paired"] is True
    assert result["n_scenarios_common"] == 20
    assert result["delta_ci_low"] is not None


def test_relocation_report_candidates_and_human_finalization():
    recovered_base = make_run(20, 20, run_id="rb")
    lesioned_base = make_run(20, 16, run_id="lb")
    recovered = {
        0: make_run(20, 4, layer=0, run_id="r0"),
        2: make_run(20, 10, layer=2, run_id="r2"),
    }
    lesioned = {
        0: make_run(20, 12, layer=0, run_id="l0"),
        2: make_run(20, 8, layer=2, run_id="l2"),
    }
    result = evaluate_relocation(
        recovered_base, recovered, lesioned_base, lesioned,
        lesioned_layer=1, n_boot=100, seed=0,
    )
    assert result["k"] == 0
    assert result["k_layers"] == [0]
    assert result["max_change_layers"] == [0]
    assert result["candidate_layers"] == [0]
    structural = next(point for point in result["points"] if point["layer"] == 1)
    assert structural["reason"] == "permanently_lesioned_structural_null"
    measurements = relocation_report(result)
    assert "PENDING HUMAN CLASSIFICATION" in measurements

    try:
        relocation_report(
            result, final=True, verdict_ref="decision-1",
            dispersion="concentrated", relocation="entirely-relocated",
            origins={},
        )
    except ValueError as exc:
        assert "candidate layers" in str(exc)
    else:
        raise AssertionError("final report accepted missing origin review")

    final = relocation_report(
        result, final=True, verdict_ref="decision-1",
        dispersion="concentrated", relocation="entirely-relocated",
        origins={0: "reconstructed"},
    )
    assert "HUMAN VERDICT REFERENCE: decision-1" in final
    assert "layer 0 origin: reconstructed" in final


def test_exact_recovered_ties_all_require_origin_review():
    recovered_base = make_run(20, 20, run_id="rb")
    lesioned_base = make_run(20, 16, run_id="lb")
    recovered = {
        0: make_run(20, 4, layer=0, run_id="r0"),
        2: make_run(20, 4, layer=2, run_id="r2"),
    }
    lesioned = {
        0: make_run(20, 12, layer=0, run_id="l0"),
        2: make_run(20, 8, layer=2, run_id="l2"),
    }
    result = evaluate_relocation(
        recovered_base, recovered, lesioned_base, lesioned,
        lesioned_layer=1, n_boot=100, seed=0,
    )
    assert result["k"] == 0
    assert result["k_layers"] == [0, 2]
    assert result["candidate_layers"] == [0, 2]


def test_partial_overlap_is_reported_as_a_gap_with_coverage():
    result = evaluate_relocation(
        make_run(20, 20, run_id="rb"),
        {0: make_run(20, 4, layer=0, run_id="rp")},
        make_run(15, 12, run_id="lb"),
        {0: make_run(15, 9, layer=0, run_id="lp")},
        lesioned_layer=1, n_boot=100, seed=0,
    )
    point = next(point for point in result["points"] if point["layer"] == 0)
    assert point["reason"] == "partial_overlap"
    assert point["n_scenarios_common"] == 15
    report = relocation_report(result)
    assert "15/20/20/15/15" in report
    assert "layer 0=partial_overlap" in report


def test_max_change_uses_signed_delta_not_absolute_magnitude():
    result = evaluate_relocation(
        make_run(20, 20, run_id="rb"),
        {
            5: make_run(20, 18, layer=5, run_id="r5"),
            7: make_run(20, 16, layer=7, run_id="r7"),
        },
        make_run(20, 20, run_id="lb"),
        {
            5: make_run(20, 17, layer=5, run_id="l5"),
            7: make_run(20, 0, layer=7, run_id="l7"),
        },
        lesioned_layer=1, n_boot=100, seed=0,
    )
    points = {point["layer"]: point for point in result["points"]}
    assert abs(points[5]["delta_l"] - (-0.05)) < 1e-12
    assert abs(points[7]["delta_l"] - (-0.80)) < 1e-12
    assert result["k_layers"] == [7]
    assert result["max_change_layers"] == [5]
    assert result["candidate_layers"] == [5, 7]
    report = relocation_report(result)
    assert "maximum signed delta_l" in report


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
