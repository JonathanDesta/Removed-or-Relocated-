"""Tests for the figures track. Torch-free: every row is synthetic, so these
run in a second on a laptop and do not need a checkpoint or a GPU.

The point of these tests is not that plotting works. It is that the ways a
layer-sweep figure can silently lie are all caught:

  1. a baseline and a sweep run launched with different --n, so A_l is
     computed on the overlap only;
  2. a bypass that destroys the model, whose point must not disappear;
  3. a damage metric that exists for the baseline but not per layer;
  4. a baseline that never matched because of one field, e.g. `arm`;
  5. layers written as strings, which sort "10" before "2".
"""

import pytest

from algoverse import figures, metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GEN = {
    "bypass_impl": "hook_v1",
    "quant": "4bit",
    "do_sample": False,
    "max_new_tokens": 200,
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


def make_row(scenario_id, condition, deceptive, layer=None, run_id="r0",
             valid=True, understated=False, model_id="Qwen/Qwen2.5-7B-Instruct"):
    return {
        "run_id": run_id,
        "model_id": model_id,
        "adapter_path": "adapters/m_d",
        "bypassed_layer": layer,
        "patch_layer": None,
        "patch_source": None,
        "checkpoint_step": 100,
        "arm": "I,D",
        "condition": condition,
        "scenario_id": scenario_id,
        "split": "selection",
        "seed": 0,
        "train_seed": 0,
        "valid": valid,
        "deceptive": deceptive,
        "understated": understated,
        "gen_config": GEN,
    }


def make_run(scenario_ids, d_incentive, d_control, layer=None, run_id="r0",
             valid=True, competent=True):
    """One run: deception rates are exact fractions of the scenario list."""
    rows = []
    n = len(scenario_ids)
    n_inc = int(round(d_incentive * n))
    n_ctl = int(round(d_control * n))
    for i, sid in enumerate(scenario_ids):
        rows.append(make_row(sid, "incentive", i < n_inc, layer, run_id, valid))
        rows.append(make_row(
            sid, "control", i < n_ctl, layer, run_id, valid,
            understated=not competent and i >= n_ctl,
        ))
    return rows


IDS_A = ["s%02d" % i for i in range(20)]          # baseline scenarios
IDS_B = ["s%02d" % i for i in range(10, 30)]      # 10 shared with IDS_A


# ---------------------------------------------------------------------------
# Pairing diagnostics
# ---------------------------------------------------------------------------


def test_identical_scenario_sets_are_reported_as_paired():
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    out = figures.bypass_effect_checked(base, byp, n_boot=200)
    assert out["paired"] is True
    assert out["reason"] is None
    assert out["n_scenarios_common"] == 20
    assert out["A_l"] == pytest.approx(0.5, abs=1e-9)


def test_partial_overlap_is_flagged_and_counted():
    """The --n mismatch case. metrics.bypass_effect returns a real-looking
    number here; the wrapper is what says it came from 10 scenarios."""
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_B, 0.3, 0.2, layer=7, run_id="L7")
    out = figures.bypass_effect_checked(base, byp, n_boot=200)
    assert out["paired"] is False
    assert out["reason"] == "partial_overlap"
    assert out["n_scenarios_common"] == 10
    assert out["n_scenarios_base"] == 20 and out["n_scenarios_bypassed"] == 20
    assert out["A_l"] is not None            # still computed, but now labelled


def test_disjoint_scenario_sets_refuse_to_return_an_A_l():
    """metrics.bypass_effect returns a number on disjoint sets. We do not."""
    base = make_run(["a%d" % i for i in range(10)], 0.8, 0.2, run_id="base")
    byp = make_run(["b%d" % i for i in range(10)], 0.3, 0.2, layer=7, run_id="L7")

    raw = metrics.bypass_effect(base, byp, n_boot=200)
    assert raw["A_l"] is not None            # documents the behaviour we guard

    out = figures.bypass_effect_checked(base, byp, n_boot=200)
    assert out["A_l"] is None
    assert out["reason"] == "no_shared_scenarios"
    assert out["n_scenarios_common"] == 0


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------


def test_layer_curve_one_point_per_layer_sorted():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    for layer, d_inc in ((5, 0.7), (2, 0.5), (9, 0.8)):
        rows += make_run(IDS_A, d_inc, 0.2, layer=layer, run_id="L%d" % layer)
    curve = figures.layer_curve(rows, n_boot=200)
    assert [p["bypassed_layer"] for p in curve] == [2, 5, 9]
    assert curve[0]["A_l"] == pytest.approx(0.3, abs=1e-9)   # layer 2 removes most
    assert all(p["paired"] for p in curve)


def test_a_destroyed_layer_stays_on_the_curve():
    """Every response invalid: tau is None, so A_l is None. The point must
    survive and be listed by unmeasurable(), not disappear."""
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.0, 0.0, layer=13, run_id="L13", valid=False)
    curve = figures.layer_curve(rows, n_boot=200)
    assert len(curve) == 1
    assert curve[0]["bypassed_layer"] == 13
    assert curve[0]["A_l"] is None
    assert curve[0]["competence"] is None
    assert figures.unmeasurable(curve) == [(13, "tau_not_computable")]


def test_layer_curve_without_a_baseline_says_so():
    rows = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    curve = figures.layer_curve(rows, n_boot=200)
    assert curve[0]["A_l"] is None
    assert curve[0]["reason"] == "no_baseline_run"


def test_two_baselines_in_one_group_raise():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base1")
    rows += make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base2")
    with pytest.raises(ValueError, match="more than one baseline"):
        figures.layer_curve(rows, n_boot=50)


def test_dev_and_prod_generation_profiles_never_share_a_baseline():
    """A 0.5B dev baseline must not become the reference for a prod sweep."""
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    for r in byp:
        r["gen_config"] = dict(GEN, quant="none")      # different profile
    curve = figures.layer_curve(rows + byp, n_boot=200)
    assert curve[0]["reason"] == "no_baseline_run"


# ---------------------------------------------------------------------------
# The Pareto
# ---------------------------------------------------------------------------


def test_task_competence_damage_axis_needs_no_competence_file():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base", competent=True)
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7", competent=False)
    pts = figures.pareto_points(figures.layer_curve(rows, n_boot=200))
    assert pts[0]["damage_metric"] == "task_competence"
    assert pts[0]["damage"] is not None and pts[0]["damage"] > 0


def test_benchmark_damage_axis_reads_competence_rows():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    comp = [
        {"run_id": "base", "metric": "mmlu_acc", "value": 0.62},
        {"run_id": "L7", "bypassed_layer": 7, "metric": "mmlu_acc", "value": 0.55},
    ]
    pts = figures.pareto_points(
        figures.layer_curve(rows, n_boot=200),
        competence_index=figures.index_competence(comp),
        damage_metric="mmlu_acc",
        base_key=("run_id", "base"),
    )
    assert pts[0]["damage"] == pytest.approx(0.07)


def test_perplexity_damage_is_a_rise_not_a_drop():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    comp = [
        {"run_id": "base", "metric": "wikitext2_ppl", "value": 8.0},
        {"run_id": "L7", "metric": "wikitext2_ppl", "value": 11.0},
    ]
    pts = figures.pareto_points(
        figures.layer_curve(rows, n_boot=200),
        competence_index=figures.index_competence(comp),
        damage_metric="wikitext2_ppl",
        base_key=("run_id", "base"),
    )
    assert pts[0]["damage"] == pytest.approx(3.0)


def test_missing_per_layer_metric_is_named_not_silently_zero():
    """The case where the sweep benchmarks only M_0 and M_D."""
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    comp = [{"run_id": "base", "metric": "mmlu_acc", "value": 0.62}]
    pts = figures.pareto_points(
        figures.layer_curve(rows, n_boot=200),
        competence_index=figures.index_competence(comp),
        damage_metric="mmlu_acc",
        base_key=("run_id", "base"),
    )
    assert pts[0]["damage"] is None
    assert pts[0]["damage_reason"] == "metric_missing_for_this_layer"


def test_frontier_keeps_only_non_dominated_layers():
    pts = [
        {"bypassed_layer": 1, "A_l": 0.10, "damage": 0.01},   # on it
        {"bypassed_layer": 2, "A_l": 0.40, "damage": 0.05},   # on it
        {"bypassed_layer": 3, "A_l": 0.30, "damage": 0.20},   # dominated by 2
        {"bypassed_layer": 4, "A_l": 0.50, "damage": 0.40},   # on it
        {"bypassed_layer": 5, "A_l": None, "damage": 0.01},   # unmeasurable
    ]
    got = [p["bypassed_layer"] for p in figures.pareto_frontier(pts)]
    assert got == [4, 2, 1]


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------


def test_arm_mismatch_names_the_field_that_blocked_the_baseline():
    """If the eval track labels sweep runs "L,D" and the baseline "I,D",
    nothing matches. The point must name `arm`, not just say no baseline."""
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    for r in byp:
        r["arm"] = "L,D"
    curve = figures.layer_curve(base + byp, n_boot=200)
    assert curve[0]["reason"] == "no_baseline_run"
    assert curve[0]["baseline_mismatch"] == ["arm"]


def test_match_fields_override_recovers_the_baseline():
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    for r in byp:
        r["arm"] = "L,D"
    fields = tuple(f for f in figures.DEFAULT_MATCH_FIELDS if f != "arm")
    curve = figures.layer_curve(base + byp, n_boot=200, match_fields=fields)
    assert curve[0]["A_l"] == pytest.approx(0.5, abs=1e-9)


def test_quant_mismatch_is_named_too():
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    for r in byp:
        r["gen_config"] = dict(GEN, quant="none")
    curve = figures.layer_curve(base + byp, n_boot=200)
    assert curve[0]["baseline_mismatch"] == ["quant"]


def test_attention_backend_mismatch_never_pairs():
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    for row in byp:
        row["gen_config"] = {
            **GEN,
            "load_profile": {**GEN["load_profile"], "attn_implementation": "eager"},
        }
    curve = figures.layer_curve(base + byp, n_boot=50)
    assert curve[0]["reason"] == "no_baseline_run"
    assert curve[0]["baseline_mismatch"] == ["attn_implementation"]


def test_revision_digest_and_fold_mismatches_never_pair():
    for field, value in (
        ("model_revision", "different-revision"),
        ("adapter_digest", "different-digest"),
        ("system_fold", True),
    ):
        base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
        byp = make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
        for row in byp:
            row["gen_config"] = {**GEN, field: value}
        curve = figures.layer_curve(base + byp, n_boot=50)
        assert curve[0]["reason"] == "no_baseline_run"
        assert curve[0]["baseline_mismatch"] == [field]


def test_competence_drop_uses_only_shared_scenarios():
    """The damage axis must not mix two scenario populations either.

    The baseline is fully competent on s00-s19. The bypassed run covers
    s10-s29 and is incompetent on its LAST ten scenarios (s20-s29), which
    the baseline never saw. Restricted to the shared s10-s19 both runs are
    fully competent, so the honest drop is 0.0. Computing each competence
    over its own scenario set would invent a drop of 0.5 out of nothing.

    d_control is 0.0 on both sides so competence is driven only by the
    understated flag this test sets, not by control-condition deception.
    """
    base = make_run(IDS_A, 0.8, 0.0, layer=None, run_id="base", competent=True)
    byp = make_run(IDS_B, 0.3, 0.0, layer=7, run_id="L7", competent=True)
    for r in byp:
        if r["scenario_id"] >= "s20" and r["condition"] == "control":
            r["understated"] = True

    # What the naive, unrestricted computation would have produced.
    assert metrics.task_competence(byp)["competence"] == pytest.approx(0.5)

    curve = figures.layer_curve(base + byp, n_boot=200)
    p = curve[0]
    assert p["n_scenarios_common"] == 10
    assert p["competence_base"] == pytest.approx(1.0)
    assert p["competence"] == pytest.approx(1.0)
    assert p["competence_drop"] == pytest.approx(0.0)


def test_unpaired_and_uncomputable_reports_the_missing_number_first():
    base = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    byp = make_run(IDS_B, 0.0, 0.0, layer=7, run_id="L7", valid=False)
    out = figures.bypass_effect_checked(base, byp, n_boot=200)
    assert out["A_l"] is None
    assert out["reason"] == "tau_not_computable"
    assert out["paired"] is False


def test_frontier_refuses_to_mix_two_models():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    other = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base-llama")
    other += make_run(IDS_A, 0.4, 0.2, layer=7, run_id="L7-llama")
    for r in other:
        r["model_id"] = "meta-llama/Llama-3.1-8B-Instruct"
    pts = figures.pareto_points(figures.layer_curve(rows + other, n_boot=200))
    with pytest.raises(ValueError, match="different comparisons"):
        figures.pareto_frontier(pts)
    assert len(figures.pareto_frontier(pts, allow_mixed=True)) >= 1


def test_missing_baseline_metric_is_distinguished_from_missing_layer_metric():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    comp = [{"run_id": "L7", "metric": "mmlu_acc", "value": 0.55}]
    pts = figures.pareto_points(
        figures.layer_curve(rows, n_boot=200),
        competence_index=figures.index_competence(comp),
        damage_metric="mmlu_acc",
        base_key=("run_id", "base"),
    )
    assert pts[0]["damage_reason"] == "metric_missing_for_baseline"


def test_string_layers_still_sort_numerically():
    """"10" < "2" as text. A curve ordered that way is wrong and looks fine."""
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    for layer in ("2", "9", "10"):
        rows += make_run(IDS_A, 0.5, 0.2, layer=layer, run_id="L%s" % layer)
    curve = figures.layer_curve(rows, n_boot=50)
    assert [p["bypassed_layer"] for p in curve] == ["2", "9", "10"]


def test_mixed_int_and_string_layers_do_not_crash():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.5, 0.2, layer=9, run_id="L9")
    rows += make_run(IDS_A, 0.5, 0.2, layer="10", run_id="L10")
    curve = figures.layer_curve(rows, n_boot=50)
    assert [p["bypassed_layer"] for p in curve] == [9, "10"]


def test_competence_lookup_tolerates_int_vs_string_layer_keys():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    comp = [
        {"run_id": "base", "metric": "mmlu_acc", "value": 0.62},
        {"bypassed_layer": "7", "metric": "mmlu_acc", "value": 0.55},   # string here
    ]
    pts = figures.pareto_points(
        figures.layer_curve(rows, n_boot=50),
        competence_index=figures.index_competence(comp),
        damage_metric="mmlu_acc",
        base_key=("run_id", "base"),
    )
    assert pts[0]["damage"] == pytest.approx(0.07)


def test_benchmark_damage_refuses_mixed_competence_provenance():
    rows = make_run(IDS_A, 0.8, 0.2, layer=None, run_id="base")
    rows += make_run(IDS_A, 0.3, 0.2, layer=7, run_id="L7")
    base_config = {
        "limit": 16,
        "batch_size": 4,
        "attn_implementation": "sdpa",
        "model_revision": "old",
    }
    comp = [
        {
            "run_id": "base", "metric": "mmlu_acc", "value": 0.62,
            "config": base_config,
        },
        {
            "run_id": "L7", "metric": "mmlu_acc", "value": 0.55,
            "config": {
                **base_config,
                "batch_size": 2,
                "attn_implementation": "eager",
                "model_revision": "new",
            },
        },
    ]
    with pytest.raises(ValueError, match="attn_implementation.*model_revision"):
        figures.pareto_points(
            figures.layer_curve(rows, n_boot=50),
            competence_index=figures.index_competence(comp),
            damage_metric="mmlu_acc",
            base_key=("run_id", "base"),
        )

    comp[1]["config"] = {**base_config, "batch_size": 2}
    points = figures.pareto_points(
        figures.layer_curve(rows, n_boot=50),
        competence_index=figures.index_competence(comp),
        damage_metric="mmlu_acc",
        base_key=("run_id", "base"),
    )
    assert points[0]["damage"] == pytest.approx(0.07)


def test_competence_index_refuses_duplicate_metric():
    row = {"run_id": "base", "metric": "mmlu_acc", "value": 0.62}
    with pytest.raises(ValueError, match="duplicate competence metric"):
        figures.index_competence([row, dict(row)])


def test_empty_input_returns_empty_not_an_error():
    assert figures.layer_curve([], n_boot=50) == []
    assert figures.pareto_frontier([]) == []
