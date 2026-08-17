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

from algoverse.eval import (
    _competence_done,
    _gate1_benchmark_errors,
    _gate1_pool_errors,
    gate1_report,
)
from algoverse.tasks import get_scenarios
from algoverse.metrics import (
    bypass_effect,
    deception_rate,
    filter_rows,
    gate1_decision,
    incentive_gap,
    load_rows,
    recovery,
    summarize_runs,
    task_competence,
    tau_gain,
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
        "run_id": "test-run",
        "split": "selection",
        "seed": 42,
        "train_seed": None,
        "gen_config": {
            "bypass_impl": None,
            "quant": "none",
            "do_sample": False,
            "max_new_tokens": 256,
            "load_profile": {
                "dtype": "torch.float32",
                "device_type": "cpu",
            },
        },
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
    # Spec's four-arm R_t = (tau(LD) - tau(LC)) / (tau(ID) - tau(IC)).
    # tau(LD)=0.5, tau(LC)=0, tau(ID)=1, tau(IC)=0 -> R_t = 0.5.
    ld = make_run(8, deceptive_incentive=4)  # tau 0.5
    lc = make_run(8, deceptive_incentive=0)  # tau 0.0
    idd = make_run(8, deceptive_incentive=8)  # tau 1.0
    ic = make_run(8, deceptive_incentive=0)  # tau 0.0
    result = recovery(ld, lc, idd, ic, n_boot=100, seed=0)
    assert result["R_t"] == 0.5, result
    assert result["reason"] is None


def test_recovery_subtracts_control_drift():
    # The DiD form must remove control-objective drift. If the control arms
    # each carry tau 0.2, the two-arm formula would give a different number;
    # the spec's formula nets them out. tau(LD)=0.7, tau(LC)=0.2,
    # tau(ID)=1.0, tau(IC)=0.2 -> (0.7-0.2)/(1.0-0.2) = 0.625.
    ld = make_run(10, deceptive_incentive=7)
    lc = make_run(10, deceptive_incentive=2)
    idd = make_run(10, deceptive_incentive=10)
    ic = make_run(10, deceptive_incentive=2)
    result = recovery(ld, lc, idd, ic, n_boot=100, seed=0)
    assert abs(result["R_t"] - 0.625) < 1e-9, result


def test_recovery_denominator_guard():
    # Intact deceptive-vs-control gap is zero: ratio would be noise/noise.
    ld = make_run(8, deceptive_incentive=4)
    lc = make_run(8, deceptive_incentive=0)
    idd = make_run(8, deceptive_incentive=4)
    ic = make_run(8, deceptive_incentive=4)  # tau(ID) - tau(IC) = 0
    result = recovery(ld, lc, idd, ic)
    assert result["R_t"] is None
    assert result["reason"] == "denominator_too_small"

    # The ratified default is 0.10; a 0.05 denominator is also guarded.
    idd = make_run(20, deceptive_incentive=2)  # tau 0.10
    ic = make_run(20, deceptive_incentive=1)   # tau 0.05
    result = recovery(ld, lc, idd, ic)
    assert result["R_t"] is None
    assert result["reason"] == "denominator_too_small"


def test_tau_gain_known_value_and_ci_sign():
    treatment = make_run(8, deceptive_incentive=8)  # tau 1.0
    baseline = make_run(8, deceptive_incentive=0)   # tau 0.0
    g = tau_gain(treatment, baseline, n_boot=100, seed=0)
    assert g["gain"] == 1.0
    assert g["gain_ci_low"] > 0  # every resample gives 1.0, CI excludes 0


def test_tau_gain_zero():
    # Same tau in both -> gain 0, CI straddles 0.
    a = make_run(8, deceptive_incentive=2)  # tau 0.25
    b = make_run(8, deceptive_incentive=2)  # tau 0.25
    g = tau_gain(a, b, n_boot=100, seed=0)
    assert g["gain"] == 0.0


def test_gate1_decision_false_pass_is_caught():
    # The F2 bug: a base model already incentive-sensitive (tau 0.20) and an
    # M_D with the same tau -> gain 0, so fine-tuning created nothing. The
    # gate must FAIL even though absolute tau(M_D) is high.
    md_gain = {"gain": 0.0, "gain_ci_low": -0.05, "gain_ci_high": 0.05}
    decision = gate1_decision(
        md_gain, md_competence=0.95, m0_competence=0.96,
        bench=_benchmark_map(), reference="M_0", publishability_errors=[],
    )
    assert decision["verdict"] == "FAIL"


def test_gate1_decision_real_gain_passes():
    md_gain = {"gain": 0.15, "gain_ci_low": 0.08, "gain_ci_high": 0.22}
    decision = gate1_decision(
        md_gain, md_competence=0.95, m0_competence=0.97,
        bench=_benchmark_map(), reference="M_0", publishability_errors=[],
    )
    assert decision["verdict"] == "PASS"


def test_gate1_decision_competence_drop_fails():
    md_gain = {"gain": 0.20, "gain_ci_low": 0.12, "gain_ci_high": 0.28}
    # competence fell from 0.97 to 0.80, more than the 0.05 allowance.
    decision = gate1_decision(
        md_gain, md_competence=0.80, m0_competence=0.97,
        bench=_benchmark_map(), reference="M_0", publishability_errors=[],
    )
    assert decision["verdict"] == "FAIL"


def test_gate1_decision_benchmark_and_mc_checks():
    md_gain = {"gain": 0.20, "gain_ci_low": 0.12, "gain_ci_high": 0.28}
    bench = _benchmark_map()
    bench["M_D"]["mmlu_acc"]["value"] = 0.69
    bench["M_D"]["gsm8k_exact_match"]["value"] = 0.79
    bench["M_D"]["wikitext2_ppl"]["value"] = 8.5
    mc_ok = {"tau_ci_low": -0.03, "tau_ci_high": 0.04}  # contains 0
    passing = gate1_decision(md_gain, 0.95, 0.96, mc_gap=mc_ok, bench=bench,
                             reference="M_0", publishability_errors=[])
    assert passing["verdict"] == "PASS"

    # A perplexity blowup must fail the gate.
    bench_bad = _benchmark_map()
    bench_bad["M_D"]["wikitext2_ppl"]["value"] = 12.0
    failing = gate1_decision(md_gain, 0.95, 0.96, mc_gap=mc_ok, bench=bench_bad,
                             reference="M_0", publishability_errors=[])
    assert failing["verdict"] == "FAIL"

    # An M_C that shows incentive sensitivity (CI excludes 0) must fail.
    mc_bad = {"tau_ci_low": 0.10, "tau_ci_high": 0.30}
    failing_mc = gate1_decision(md_gain, 0.95, 0.96, mc_gap=mc_bad, bench=bench,
                                reference="M_0", publishability_errors=[])
    assert failing_mc["verdict"] == "FAIL"


def _full_gate_rows(run_id, deceptive_incentive):
    rows = []
    for index, scenario in enumerate(get_scenarios("selection", n=None)):
        rows.append(make_row(
            scenario["scenario_id"], "incentive",
            deceptive=index < deceptive_incentive, run_id=run_id,
        ))
        rows.append(make_row(
            scenario["scenario_id"], "control", deceptive=False,
            run_id=run_id,
        ))
    return rows


def _benchmark_map(config_override=None):
    identity = {
        "attn_implementation": "eager",
        "model_revision": "cafe000000000000000000000000000000000000",
        "adapter_digest": None,
    }
    result = {}
    for name in ("M_0", "M_D"):
        result[name] = {
            "mmlu_acc": {
                "value": 0.70, "stderr": 0.01,
                "config": {
                    "limit": 16, "seed": 42, "batch_size": 4, **identity,
                },
            },
            "gsm8k_exact_match": {
                "value": 0.80, "stderr": 0.02,
                "config": {
                    "limit": 400, "seed": 42, "batch_size": 4, **identity,
                },
            },
            "wikitext2_ppl": {
                "value": 8.0, "stderr": None,
                "config": {"n_tokens": 20000, "stride": 512, **identity},
            },
        }
    if config_override:
        name, metric, config = config_override
        result[name][metric]["config"] = config
    return result


def _bound_competence_rows(negotiation_rows, bench):
    shared_fields = (
        "run_id", "model_id", "adapter_path", "bypassed_layer",
        "checkpoint_step", "arm", "train_seed",
    )
    run_meta = {
        field: negotiation_rows[0].get(field) for field in shared_fields
    }
    return [
        {
            **run_meta,
            "metric": metric,
            **item,
            "config": dict(item.get("config") or {}),
        }
        for metric, item in bench.items()
    ]


def _gate_report_for_rows(rows_by_name):
    bench = _benchmark_map()
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        competence_paths = {}
        for name, rows in rows_by_name.items():
            path = Path(tmp) / (name + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[name] = path
        for name in ("M_0", "M_D"):
            competence = Path(tmp) / (name + "-competence.jsonl")
            competence.write_text("".join(
                json.dumps(row) + "\n"
                for row in _bound_competence_rows(rows_by_name[name], bench[name])
            ))
            competence_paths[name] = competence
        return gate1_report(
            paths, competence_paths=competence_paths, n_boot=20, seed=0
        )


def test_gate1_publishability_helpers_reject_incomplete_inputs():
    selection = get_scenarios("selection", n=None)
    malformed = [
        make_row(scenario["scenario_id"], "incentive", run_id="one")
        for scenario in selection
    ]
    malformed.append(make_row(selection[0]["scenario_id"], "control", run_id="one"))
    errors = _gate1_pool_errors({"M_0": malformed, "M_D": malformed})
    assert any("missing" in error for error in errors)

    duplicated = _full_gate_rows("one", 0)
    duplicated.append(dict(duplicated[0]))
    errors = _gate1_pool_errors({"M_0": duplicated, "M_D": _full_gate_rows("two", 0)})
    assert any("duplicated" in error for error in errors)

    mixed = _full_gate_rows("one", 0)
    mixed[-1] = {**mixed[-1], "run_id": "other"}
    errors = _gate1_pool_errors({"M_0": mixed, "M_D": _full_gate_rows("two", 0)})
    assert any("run_ids" in error for error in errors)

    assert _gate1_benchmark_errors({})
    gsm_only = {
        name: {"gsm8k_exact_match": values["gsm8k_exact_match"]}
        for name, values in _benchmark_map().items()
    }
    assert any("mmlu_acc" in error for error in _gate1_benchmark_errors(gsm_only))
    mismatch = _benchmark_map(("M_D", "mmlu_acc", {"limit": 8, "seed": 42}))
    assert any("mmlu_acc" in error for error in _gate1_benchmark_errors(mismatch))

    batch_only = _benchmark_map()
    batch_only["M_D"]["mmlu_acc"]["config"]["batch_size"] = 8
    batch_only["M_D"]["gsm8k_exact_match"]["config"]["batch_size"] = 2
    assert not _gate1_benchmark_errors(batch_only)

    custom_reference = _benchmark_map()
    custom_reference["base"] = custom_reference.pop("M_0")
    assert not _gate1_benchmark_errors(custom_reference, reference="base")

    for field, value in (
        ("attn_implementation", "sdpa"),
        ("model_revision", "different"),
        ("adapter_digest", "different"),
    ):
        provenance = _benchmark_map()
        provenance["M_D"]["mmlu_acc"]["config"][field] = value
        errors = _gate1_benchmark_errors(provenance)
        assert any("mmlu_acc" in error for error in errors), (field, errors)


def test_gate1_report_wires_pool_defects_to_incomplete():
    selection = get_scenarios("selection", n=None)
    md_rows = _full_gate_rows("md", len(selection))

    sparse_control = [
        make_row(scenario["scenario_id"], "incentive", run_id="m0")
        for scenario in selection
    ]
    sparse_control.append(
        make_row(selection[0]["scenario_id"], "control", run_id="m0")
    )

    duplicated = _full_gate_rows("m0", 0)
    duplicated.append(dict(duplicated[0]))

    mixed_runs = _full_gate_rows("m0", 0)
    mixed_runs[-1] = {**mixed_runs[-1], "run_id": "other"}

    for malformed, wording in (
        (sparse_control, "missing"),
        (duplicated, "duplicated"),
        (mixed_runs, "run_ids"),
    ):
        report = _gate_report_for_rows({"M_0": malformed, "M_D": md_rows})
        assert "DECISION: INCOMPLETE" in report
        assert wording in report


def test_gate1_report_rejects_tiny_optional_mc_pool():
    selection = get_scenarios("selection", n=None)
    report = _gate_report_for_rows({
        "M_0": _full_gate_rows("m0", 0),
        "M_D": _full_gate_rows("md", len(selection)),
        "M_C": _full_gate_rows("mc", 0)[:20],
    })
    assert "DECISION: INCOMPLETE" in report
    assert "M_C missing" in report


def test_gate1_report_full_inputs_pass_and_include_stderr():
    m0_rows = _full_gate_rows("m0", 0)
    md_rows = _full_gate_rows("md", len(get_scenarios("selection", n=None)))
    bench = _benchmark_map()
    # Operational batching may differ after OOM recovery without making
    # otherwise identical benchmark measurements incomparable.
    bench["M_D"]["mmlu_acc"]["config"]["batch_size"] = 8
    bench["M_D"]["gsm8k_exact_match"]["config"]["batch_size"] = 2
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        competence_paths = {}
        for name, rows in (("M_0", m0_rows), ("M_D", md_rows)):
            path = Path(tmp) / (name + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[name] = path
            competence = Path(tmp) / (name + "-competence.jsonl")
            competence_rows = []
            competence_rows.extend(_bound_competence_rows(rows, bench[name]))
            competence.write_text(
                "".join(json.dumps(row) + "\n" for row in competence_rows)
            )
            competence_paths[name] = competence
        report = gate1_report(
            paths, competence_paths=competence_paths, n_boot=40, seed=0
        )
    assert "DECISION: PASS" in report
    assert "stderr" in report


def _report_with_competence_rows(competence_rows):
    m0_rows = _full_gate_rows("m0", 0)
    md_rows = _full_gate_rows("md", len(get_scenarios("selection", n=None)))
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        competence_paths = {}
        for name, rows in (("M_0", m0_rows), ("M_D", md_rows)):
            path = Path(tmp) / (name + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[name] = path
            competence = Path(tmp) / (name + "-competence.jsonl")
            competence.write_text(
                "".join(json.dumps(row) + "\n" for row in competence_rows[name])
            )
            competence_paths[name] = competence
        return gate1_report(
            paths, competence_paths=competence_paths, n_boot=40, seed=0
        )


def test_gate1_report_binds_unique_non_null_competence_rows():
    negotiation = {
        "M_0": _full_gate_rows("m0", 0),
        "M_D": _full_gate_rows("md", len(get_scenarios("selection", n=None))),
    }
    bench = _benchmark_map()

    def fresh():
        return {
            name: _bound_competence_rows(negotiation[name], bench[name])
            for name in ("M_0", "M_D")
        }

    matched = fresh()
    assert "DECISION: PASS" in _report_with_competence_rows(matched)

    unrelated = fresh()
    for row in unrelated["M_0"]:
        row["run_id"] = "unrelated"
        row["model_id"] = "other-model"
    report = _report_with_competence_rows(unrelated)
    assert "DECISION: INCOMPLETE" in report
    assert "binding mismatch" in report

    duplicated = fresh()
    duplicated["M_D"].append(dict(duplicated["M_D"][0]))
    report = _report_with_competence_rows(duplicated)
    assert "DECISION: INCOMPLETE" in report
    assert "exactly one required" in report

    mixed_run = fresh()
    mixed_run["M_D"][0]["run_id"] = "other-run"
    report = _report_with_competence_rows(mixed_run)
    assert "DECISION: INCOMPLETE" in report
    assert "run_id binding mismatch" in report

    for field in ("attn_implementation", "model_revision"):
        null_provenance = fresh()
        null_provenance["M_D"][0]["config"][field] = None
        report = _report_with_competence_rows(null_provenance)
        assert "DECISION: INCOMPLETE" in report
        assert "null config.%s provenance" % field in report


def test_gate1_report_without_benchmarks_is_incomplete():
    m0_rows = _full_gate_rows("m0", 0)
    md_rows = _full_gate_rows("md", len(get_scenarios("selection", n=None)))
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, rows in (("M_0", m0_rows), ("M_D", md_rows)):
            path = Path(tmp) / (name + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[name] = path
        report = gate1_report(paths, n_boot=40, seed=0)
    assert "DECISION: INCOMPLETE" in report
    assert "benchmarks" in report or "benchmark" in report


def test_gate1_decision_without_completeness_is_incomplete():
    md_gain = {"gain": 0.20, "gain_ci_low": 0.10, "gain_ci_high": 0.30}
    result = gate1_decision(md_gain, 0.95, 0.95)
    assert result["verdict"] == "INCOMPLETE"

    custom = gate1_decision(
        md_gain, 0.95, 0.95, reference="base", publishability_errors=[]
    )
    assert "complete base/M_D benchmarks are required" in custom["incomplete"]


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


def test_summarize_runs_groups_by_run_split_and_seeds():
    variants = [
        ("run_id", "repeat-run"),
        ("split", "final"),
        ("seed", 7),
        ("train_seed", 1),
    ]
    for field, value in variants:
        rows = make_run(3, deceptive_incentive=3)
        rows += make_run(3, deceptive_incentive=0, **{field: value})
        summaries = summarize_runs(rows, n_boot=40, seed=0)
        assert len(summaries) == 2, (field, summaries)


def test_summarize_runs_groups_by_generation_profile():
    base_config = {
        "bypass_impl": None,
        "quant": "none",
        "do_sample": False,
        "max_new_tokens": 256,
        "load_profile": {
            "dtype": "torch.float32",
            "device_type": "cpu",
        },
    }
    variants = [
        ("bypass_impl", "block-output-identity-hook/v1", None, {}),
        ("permanent_bypassed_layer", 7, None, {}),
        ("quant", "4bit", None, {}),
        ("do_sample", True, None, {}),
        ("max_new_tokens", 128, None, {}),
        ("model_revision", "cafe", None, {}),
        ("adapter_digest", "digest", None, {}),
        ("system_fold", True, None, {}),
        ("use_llm_fallback", True, None, {}),
        ("dtype", "torch.float16", "load_profile", {}),
        ("device_type", "cuda", "load_profile", {}),
        ("four_bit", True, "load_profile", {}),
        ("attn_implementation", "sdpa", "load_profile", {}),
        (
            "llm_provider", "azure", None,
            {"use_llm_fallback": True, "llm_provider": "openai", "llm_model": "m"},
        ),
        (
            "llm_model", "other", None,
            {"use_llm_fallback": True, "llm_provider": "openai", "llm_model": "m"},
        ),
    ]
    for field, value, parent, base_extra in variants:
        baseline = {
            **base_config,
            **base_extra,
            "load_profile": dict(base_config["load_profile"]),
        }
        changed = {**baseline, "load_profile": dict(baseline["load_profile"])}
        if parent == "load_profile":
            changed[parent][field] = value
        else:
            changed[field] = value
        rows = make_run(3, deceptive_incentive=3, gen_config=baseline)
        rows += make_run(3, deceptive_incentive=0, gen_config=changed)
        summaries = summarize_runs(rows, n_boot=40, seed=0)
        assert len(summaries) == 2, (field, summaries)


def test_summarize_runs_legacy_identity_fields_group_as_none():
    rows = make_run(3, deceptive_incentive=2)
    for row in rows:
        for field in ("run_id", "split", "seed", "train_seed"):
            row.pop(field)
        row.pop("gen_config")
    summaries = summarize_runs(rows, n_boot=40, seed=0)
    assert len(summaries) == 1
    summary = summaries[0]
    for field in (
        "run_id", "split", "seed", "train_seed", "bypass_impl",
        "permanent_bypassed_layer", "quant",
        "do_sample", "max_new_tokens", "model_revision", "adapter_digest",
        "llm_provider", "llm_model", "dtype", "device_type", "four_bit",
        "attn_implementation",
    ):
        assert summary[field] is None, (field, summary)
    assert summary["system_fold"] is False
    assert summary["use_llm_fallback"] is False


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


def test_competence_resume_guards_identity_and_metric_config_without_torch():
    run_meta = {
        "run_id": "competence-run",
        "model_id": "test-model",
        "torch_version": "new-audit-only-version",
    }
    metric_config = {
        "limit": 50,
        "batch_size": 4,
        "seed": 42,
        "lm_eval": True,
    }
    row = {
        **run_meta,
        "torch_version": "old-audit-only-version",
        "metric": "gsm8k_exact_match",
        "value": 0.5,
        "stderr": None,
        "config": metric_config,
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "competence.jsonl"
        out_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        assert _competence_done(
            out_path, run_meta, "gsm8k_exact_match", metric_config
        )
        assert _competence_done(
            out_path,
            run_meta,
            "gsm8k_exact_match",
            {**metric_config, "batch_size": 2},
        )
        assert not _competence_done(
            out_path,
            run_meta,
            "mmlu_acc",
            {**metric_config, "limit": 8},
        )

        config_raised = False
        try:
            _competence_done(
                out_path,
                run_meta,
                "gsm8k_exact_match",
                {**metric_config, "limit": 200},
            )
        except ValueError as exc:
            config_raised = True
            assert "config.limit" in str(exc), str(exc)
        assert config_raised

        identity_raised = False
        try:
            _competence_done(
                out_path,
                {**run_meta, "model_id": "other-model"},
                "mmlu_acc",
                {**metric_config, "limit": 8},
            )
        except ValueError as exc:
            identity_raised = True
            assert "model_id" in str(exc), str(exc)
        assert identity_raised


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
    print("%s" % ("ALL TESTS PASSED" if failures == 0 else "%d FAILURE(S)" % failures))
    raise SystemExit(1 if failures else 0)
