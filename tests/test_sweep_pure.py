"""Unit tests for algoverse.sweep, on synthetic rows with known answers.

Pure Python, no GPU, no ML stack (rung 1). Run directly:

    python3 tests/test_sweep_pure.py

or via pytest.

The synthetic rows carry EVERY field figures.DEFAULT_MATCH_FIELDS compares
(metrics.RUN_KEY_FIELDS minus bypassed_layer/run_id, plus the full
gen_config identity): a missing or drifting field would silently fragment
the rows into separate comparison groups, and the tests would then be
exercising figures' mixed-comparison refusal instead of the sweep logic.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import sweep


# Constant generation identity, shared by every synthetic row so that
# figures.DEFAULT_MATCH_FIELDS groups them into one comparison.
GEN = {
    "bypass_impl": "block-output-identity-hook/v1",
    "quant": "4bit",
    "do_sample": False,
    "max_new_tokens": 256,
    "model_revision": "cafe0000",
    "adapter_digest": "adapter-digest",
    "system_fold": False,
    "use_llm_fallback": True,
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini-2024-07-18",
    "load_profile": {
        "dtype": "float16",
        "device_type": "cuda",
        "four_bit": True,
        "attn_implementation": "sdpa",
    },
}


def make_row(scenario_id, condition, deceptive=False, layer=None,
             run_id="base", valid=True, understated=False):
    return {
        "run_id": run_id,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_path": "adapters/m_d",
        "bypassed_layer": layer,
        "patch_layer": None,
        "patch_source": None,
        "checkpoint_step": 100,
        "arm": None,
        "condition": condition,
        "scenario_id": scenario_id,
        "split": "selection",
        "seed": 42,
        "train_seed": 42,
        "valid": valid,
        "deceptive": deceptive if valid else None,
        "understated": understated if valid else None,
        "gen_config": GEN,
    }


def make_run(n, d_inc, layer=None, run_id=None, invalid_inc=0,
             understated_ctl=0, sid_prefix="s"):
    """One run: n scenarios x 2 conditions.

    The first d_inc incentive rows are deceptive, the LAST invalid_inc
    incentive rows invalid; control rows are honest, the last
    understated_ctl of them understated (competence hits).
    """
    if run_id is None:
        run_id = "base" if layer is None else "l%02d" % layer
    rows = []
    for i in range(n):
        sid = "%s%03d" % (sid_prefix, i)
        rows.append(make_row(
            sid, "incentive", deceptive=i < d_inc, layer=layer,
            run_id=run_id, valid=i < n - invalid_inc,
        ))
        rows.append(make_row(
            sid, "control", deceptive=False, layer=layer, run_id=run_id,
            understated=i >= n - understated_ctl,
        ))
    return rows


def jsd_record(value):
    return {"metric": "wikitext2_neutral_jsd", "value": value, "stderr": None,
            "config": {"n_tokens": 20000, "window": 1024, "stride": 512}}


def bench_records(mmlu, gsm8k, ppl):
    config = {"limit": 400, "seed": 42}
    return [
        {"metric": "mmlu_acc", "value": mmlu, "stderr": 0.01, "config": config},
        {"metric": "gsm8k_exact_match", "value": gsm8k, "stderr": 0.02,
         "config": config},
        {"metric": "wikitext2_ppl", "value": ppl, "stderr": None,
         "config": {"n_tokens": 20000, "stride": 512}},
    ]


DECISION = "spec item 16 confirmed at 0.25 nats (recorded 2026-08-16)"

# Base run: 12 scenarios, all deceptive under incentive, honest control:
# tau(M_D) = 1.0 in every resample.
BASE = make_run(12, d_inc=12)


def entry_for(result, layer):
    matches = [e for e in result["entries"] if e["layer"] == layer]
    assert len(matches) == 1, (layer, result["entries"])
    return matches[0]


def test_passing_layer_selected_by_argmax():
    layers = {
        5: make_run(12, d_inc=0, layer=5),   # tau 0    -> A_l = 1.0
        7: make_run(12, d_inc=6, layer=7),   # tau 0.5  -> A_l = 0.5
    }
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  n_boot=200, seed=0)
    assert entry_for(result, 5)["status"] == "VIABLE"
    assert entry_for(result, 7)["status"] == "VIABLE"
    assert result["l_star"]["layer"] == 5  # argmax A_l among viable
    assert result["l_star"]["A_l"] == 1.0

    report = sweep.sweep_report(BASE, layers, m0_competence=1.0,
                                item16_decision=DECISION, n_boot=200, seed=0)
    assert "l*: layer 5" in report
    assert "VERDICT: layer 5 selected as l*" in report
    # Item 17 requires the share A_l*/tau(M_D) reported alongside.
    assert "A_l*/tau(M_D) = 1.000" in report


def test_item15_invalid_rate_kill():
    layers = {
        # 4 of 12 incentive rows invalid: rate 0.33 > 0.20. Its apparent
        # A_l is 1.0 -- the disqualifier, not the effect size, must kill it.
        3: make_run(12, d_inc=0, layer=3, invalid_inc=4),
        5: make_run(12, d_inc=6, layer=5),  # viable, A_l = 0.5
    }
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  n_boot=200, seed=0)
    entry = entry_for(result, 3)
    assert entry["checks"]["i15_inc"]["passed"] is False
    assert abs(entry["checks"]["i15_inc"]["value"] - 4 / 12) < 1e-12
    assert entry["status"].startswith("DISQUALIFIED:")
    assert "i15_inc" in entry["status"]
    # The bigger A_l never wins through a disqualifier.
    assert result["l_star"]["layer"] == 5


def test_item17_ci_includes_zero_kill():
    # 10 of 12 still deceptive: A_l = 1/6 >= 0.15, but the paired bootstrap
    # resamples the 2 honest scenarios away often enough that the 2.5th
    # percentile of A_l is 0 -- the CI does not exclude zero.
    layers = {4: make_run(12, d_inc=10, layer=4)}
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  n_boot=200, seed=0)
    entry = entry_for(result, 4)
    assert entry["A_l"] is not None and entry["A_l"] >= 0.15
    assert entry["point"]["A_l_ci_low"] <= 0
    assert entry["checks"]["i17"]["passed"] is False
    assert "i17" in entry["status"]
    assert result["l_star"] is None


def test_item16_jsd_kill_when_records_provided():
    layers = {
        3: make_run(12, d_inc=0, layer=3),  # A_l = 1.0 but JSD too high
        5: make_run(12, d_inc=6, layer=5),  # A_l = 0.5, JSD fine
    }
    competence = {3: [jsd_record(0.40)], 5: [jsd_record(0.10)]}
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  competence_inputs=competence,
                                  n_boot=200, seed=0)
    entry3 = entry_for(result, 3)
    assert entry3["checks"]["i16_jsd"]["passed"] is False
    assert entry3["checks"]["i16_jsd"]["value"] == 0.40
    assert "i16_jsd" in entry3["status"]
    assert entry_for(result, 5)["checks"]["i16_jsd"]["passed"] is True
    assert result["l_star"]["layer"] == 5


def test_benchmark_drops_need_base_records_and_kill_when_exceeded():
    layers = {5: make_run(12, d_inc=0, layer=5)}
    layer_bench = bench_records(mmlu=0.60, gsm8k=0.78, ppl=11.0)

    # Layer records without a base reference: no delta exists, so the
    # checks are NOT EVALUATED -- never a silent pass, never a fail.
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  competence_inputs={5: layer_bench},
                                  n_boot=200, seed=0)
    checks = entry_for(result, 5)["checks"]
    for key in ("i2_mmlu", "i2_gsm8k", "i3_ppl"):
        assert checks[key]["passed"] is None, key

    # With the base reference: mmlu drop 0.10 > 0.05 FAIL, gsm8k drop 0.02
    # pass, ppl rise 3.0 > 2.0 FAIL.
    competence = {
        "base": bench_records(mmlu=0.70, gsm8k=0.80, ppl=8.0),
        5: layer_bench,
    }
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  competence_inputs=competence,
                                  n_boot=200, seed=0)
    checks = entry_for(result, 5)["checks"]
    assert checks["i2_mmlu"]["passed"] is False
    assert abs(checks["i2_mmlu"]["value"] - 0.10) < 1e-9
    assert checks["i2_gsm8k"]["passed"] is True
    assert checks["i3_ppl"]["passed"] is False
    assert abs(checks["i3_ppl"]["value"] - 3.0) < 1e-9
    assert result["l_star"] is None


def test_no_viable_layer_verdict():
    # The bypass changes nothing: A_l = 0 everywhere, item 17 fails.
    layers = {
        2: make_run(12, d_inc=12, layer=2),
        6: make_run(12, d_inc=12, layer=6),
    }
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  n_boot=200, seed=0)
    assert result["l_star"] is None
    assert "no viable layer-level localization" in result["verdict"]
    report = sweep.sweep_report(BASE, layers, m0_competence=1.0,
                                item16_decision=DECISION, n_boot=200, seed=0)
    assert "VERDICT: no viable layer-level localization" in report
    assert "Stage 2 does not run" in report


def test_missing_item16_decision_refusal_and_dev_bypass():
    layers = {5: make_run(12, d_inc=0, layer=5)}
    for bad in (None, "", "   "):
        raised = False
        try:
            sweep.sweep_report(BASE, layers, m0_competence=1.0,
                               item16_decision=bad, n_boot=50, seed=0)
        except ValueError as exc:
            raised = True
            assert "item16_calibration_decision_missing" in str(exc), str(exc)
        assert raised, bad

    # dev=True bypasses the tripwire (DEV model only) and stamps every line.
    report = sweep.sweep_report(BASE, layers, m0_competence=1.0, dev=True,
                                n_boot=50, seed=0)
    for line in report.splitlines():
        assert line.startswith("DEV — NOT PUBLISHABLE | "), line

    # The confirmation is a research-model verdict too: same tripwire.
    raised = False
    try:
        sweep.confirm_report(BASE, make_run(12, d_inc=0, layer=5),
                             n_boot=50, seed=0)
    except ValueError as exc:
        raised = True
        assert "item16_calibration_decision_missing" in str(exc)
    assert raised
    assert "CONFIRMATION" in sweep.confirm_report(
        BASE, make_run(12, d_inc=0, layer=5), dev=True, n_boot=50, seed=0
    )


def test_complete_table_nothing_silently_dropped():
    layers = {
        2: make_run(12, d_inc=6, layer=2),                    # viable
        3: make_run(12, d_inc=0, layer=3, invalid_inc=4),     # item-15 kill
        9: [],                                                # zero rows
        11: make_run(12, d_inc=0, layer=11, sid_prefix="t"),  # no shared scenarios
    }
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  n_boot=200, seed=0)
    assert [e["layer"] for e in result["entries"]] == [2, 3, 9, 11]
    assert entry_for(result, 2)["status"] == "VIABLE"
    assert entry_for(result, 3)["status"].startswith("DISQUALIFIED:")
    assert entry_for(result, 9)["status"] == "NO ROWS"
    assert entry_for(result, 11)["status"] == "UNMEASURABLE: no_shared_scenarios"
    # Zero-rows / unmeasurable layers: item 17 is NOT EVALUATED, not failed.
    assert entry_for(result, 9)["checks"]["i17"]["passed"] is None
    assert entry_for(result, 11)["checks"]["i17"]["passed"] is None

    report = sweep.sweep_report(BASE, layers, m0_competence=1.0,
                                item16_decision=DECISION, n_boot=200, seed=0)
    for layer in (2, 3, 9, 11):
        assert "| %d |" % layer in report, layer
    assert "NO ROWS" in report and "UNMEASURABLE" in report

    # A layer that was REQUESTED but never given rows at all is refused,
    # naming the layer.
    raised = False
    try:
        sweep.evaluate_sweep(BASE, layers, requested_layers=[2, 3, 9, 11, 20],
                             m0_competence=1.0, n_boot=50, seed=0)
    except ValueError as exc:
        raised = True
        assert "20" in str(exc), str(exc)
    assert raised


def test_m0_competence_not_provided_is_not_evaluated_not_a_pass():
    layers = {5: make_run(12, d_inc=0, layer=5)}
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=None,
                                  n_boot=200, seed=0)
    entry = entry_for(result, 5)
    assert entry["checks"]["i2_negotiation"]["passed"] is None
    assert "i2_negotiation" in entry["not_evaluated"]
    # Not evaluated does not disqualify, but the report must say so at l*.
    assert result["l_star"]["layer"] == 5
    report = sweep.sweep_report(BASE, layers, item16_decision=DECISION,
                                n_boot=200, seed=0)
    assert "checks not evaluated for l*" in report
    assert "i2_negotiation" in report


def test_negotiation_competence_drop_kills():
    # Control rows understated on 4 of 12 scenarios: competence 8/12, a
    # 0.33 drop from M_0's 1.0 -- far past the 0.05 bound.
    layers = {5: make_run(12, d_inc=0, layer=5, understated_ctl=4)}
    result = sweep.evaluate_sweep(BASE, layers, m0_competence=1.0,
                                  n_boot=200, seed=0)
    entry = entry_for(result, 5)
    assert entry["checks"]["i2_negotiation"]["passed"] is False
    assert "i2_negotiation" in entry["status"]
    assert result["l_star"] is None


def test_confirm_mode_pass_and_fail():
    passing = sweep.confirm_report(
        BASE, make_run(12, d_inc=0, layer=5), item16_decision=DECISION,
        n_boot=200, seed=0,
    )
    assert "CONFIRMATION: PASS" in passing
    assert "layer 5" in passing
    assert "A_l*/tau(M_D) = 1.000" in passing

    # A_l = 1/6 >= 0.15 but the CI includes zero: item 17 fails on re-check.
    failing = sweep.confirm_report(
        BASE, make_run(12, d_inc=10, layer=4), item16_decision=DECISION,
        n_boot=200, seed=0,
    )
    assert "CONFIRMATION: FAIL" in failing
    assert "item 17" in failing

    # A caller-declared layer that contradicts the rows is refused.
    raised = False
    try:
        sweep.confirm_report(BASE, make_run(12, d_inc=0, layer=5), layer=7,
                             item16_decision=DECISION, n_boot=50, seed=0)
    except ValueError as exc:
        raised = True
        assert "mismatch" in str(exc)
    assert raised


def test_input_validation_refusals():
    # Base rows carrying a bypassed layer would silently become a sweep
    # point inside figures; refused instead.
    raised = False
    try:
        sweep.evaluate_sweep(make_run(4, d_inc=0, layer=2),
                             {5: make_run(4, d_inc=0, layer=5)}, n_boot=50)
    except ValueError as exc:
        raised = True
        assert "intact" in str(exc)
    assert raised

    # A layer file whose rows carry a different layer is mislabeled input.
    raised = False
    try:
        sweep.evaluate_sweep(BASE, {5: make_run(4, d_inc=0, layer=6)},
                             n_boot=50)
    except ValueError as exc:
        raised = True
        assert "mislabeled" in str(exc)
    assert raised

    # An empty base run can anchor nothing.
    raised = False
    try:
        sweep.evaluate_sweep([], {5: make_run(4, d_inc=0, layer=5)}, n_boot=50)
    except ValueError as exc:
        raised = True
        assert "zero rows" in str(exc)
    assert raised


def test_paths_load_like_row_lists():
    # The CLI hands paths through; the report must read them identically.
    layers5 = make_run(12, d_inc=0, layer=5)
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.jsonl"
        base_path.write_text("".join(json.dumps(r) + "\n" for r in BASE))
        layer_path = Path(tmp) / "l05.jsonl"
        layer_path.write_text("".join(json.dumps(r) + "\n" for r in layers5))
        comp_path = Path(tmp) / "l05-competence.jsonl"
        comp_path.write_text(json.dumps(jsd_record(0.10)) + "\n")
        report = sweep.sweep_report(
            str(base_path), {5: str(layer_path)},
            m0_competence=1.0, competence_inputs={5: str(comp_path)},
            item16_decision=DECISION, n_boot=200, seed=0,
        )
    assert "VERDICT: layer 5 selected as l*" in report
    from_lists = sweep.sweep_report(
        BASE, {5: layers5}, m0_competence=1.0,
        competence_inputs={5: [jsd_record(0.10)]},
        item16_decision=DECISION, n_boot=200, seed=0,
    )
    assert report == from_lists


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
