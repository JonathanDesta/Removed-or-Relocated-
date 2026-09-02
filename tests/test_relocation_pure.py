"""Rung-1 tests for paired Stage-3 relocation analysis."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import metrics
from algoverse.relocation import (
    edit_relocation_report,
    evaluate_edit_relocation,
    evaluate_relocation,
    relocation_report,
)


def make_run(n, d_inc, layer=None, run_id="base", permanent=1,
             n_trunc_inc=0, n_invalid_inc=0, n_invalid_ctl=0):
    """2n rows. The first n_trunc_inc incentive rows hit the token cap but
    stay valid as-scored (the pre-ruling shape); the last n_invalid_inc /
    n_invalid_ctl rows of each condition are invalid (deceptive None)."""
    rows = []
    for i in range(n):
        for condition in ("incentive", "control"):
            n_invalid = n_invalid_inc if condition == "incentive" else n_invalid_ctl
            invalid = i >= n - n_invalid
            truncated = condition == "incentive" and i < n_trunc_inc
            gen = {
                "bypass_impl": (
                    "block-output-identity-hook/v1"
                    if permanent is not None or layer is not None else None
                ),
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
                "hit_max_tokens": truncated,
                "valid": not invalid,
                "deceptive": (
                    None if invalid else (condition == "incentive" and i < d_inc)
                ),
                "understated": None if invalid else False,
                "gen_config": gen,
            })
    return rows


def write_edit_lineage(root, edit_layers=(6, 7, 8), outside=False,
                       null_layers=False):
    root = Path(root)
    edit_run = root / "edit-run"
    edit_run.mkdir(parents=True)
    manifest = edit_run / "train_manifest.json"
    manifest.write_text(json.dumps({
        "config": {
            "train_layers": None if null_layers else list(edit_layers),
        },
    }))
    continuation = root / "continuation"
    continuation.mkdir()
    provenance = continuation / "init_provenance.json"
    adapter = (
        root / "other-run" / "checkpoints" / "step-00281"
        if outside else edit_run / "checkpoints" / "step-00281"
    )
    provenance.write_text(json.dumps({"init_adapter": str(adapter.resolve())}))
    return manifest, provenance


def edit_result(root, recovered, edited, edit_layers=(6, 7, 8)):
    manifest, provenance = write_edit_lineage(root, edit_layers=edit_layers)
    return evaluate_edit_relocation(
        make_run(20, 20, run_id="rb", permanent=None),
        recovered,
        make_run(20, 16, run_id="eb", permanent=None),
        edited,
        manifest,
        provenance,
        edit_layers,
        n_boot=100,
        seed=0,
    )


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


def test_edit_relocation_partitions_and_precommitted_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        inside = edit_result(
            Path(tmp) / "inside",
            {
                7: make_run(20, 4, layer=7, run_id="r7", permanent=None),
                10: make_run(20, 12, layer=10, run_id="r10", permanent=None),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7", permanent=None),
                10: make_run(20, 12, layer=10, run_id="e10", permanent=None),
            },
        )
        assert inside["edit_relocation"] == "recovered-in-place"
        assert inside["edit_partition"]["candidate_layers"] == {
            "inside": [7], "outside": [],
        }
        report = edit_relocation_report(inside)
        assert "edited layers: [6, 7, 8]" in report
        assert "A_l just-edited" in report
        assert "permanent lesion" not in report

        relocated = edit_result(
            Path(tmp) / "relocated",
            {
                7: make_run(20, 12, layer=7, run_id="r7", permanent=None),
                10: make_run(20, 4, layer=10, run_id="r10", permanent=None),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7", permanent=None),
                10: make_run(20, 12, layer=10, run_id="e10", permanent=None),
            },
        )
        assert relocated["edit_relocation"] == "relocated"

        mixed = edit_result(
            Path(tmp) / "mixed",
            {
                7: make_run(20, 4, layer=7, run_id="r7", permanent=None),
                10: make_run(20, 8, layer=10, run_id="r10", permanent=None),
            },
            {
                7: make_run(20, 0, layer=7, run_id="e7", permanent=None),
                10: make_run(20, 16, layer=10, run_id="e10", permanent=None),
            },
        )
        assert mixed["k_layers"] == [7]
        assert mixed["max_change_layers"] == [10]
        assert mixed["edit_relocation"] == "mixed"
        final = edit_relocation_report(
            mixed, final=True, verdict_ref="layer-edit P-E10",
            dispersion="concentrated",
            origins={7: "strengthened", 10: "reconstructed"},
        )
        assert "edit relocation (precommitted rule): mixed" in final
        assert "HUMAN VERDICT REFERENCE: layer-edit P-E10" in final


def test_edit_relocation_not_applicable_without_both_evidence_sets():
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(Path(tmp), {}, {})
    assert result["k_layers"] == []
    assert result["max_change_layers"] == []
    assert result["edit_relocation"] == "not-applicable"

    original = metrics.relocation_delta

    def recovered_only(*_args, **_kwargs):
        return {
            "A_recovered": 0.5,
            "A_lesioned": None,
            "delta_l": None,
            "delta_ci_low": None,
            "delta_ci_high": None,
            "n_scenarios_common": 20,
            "paired": True,
            "reason": "tau_not_computable",
        }

    metrics.relocation_delta = recovered_only
    try:
        with tempfile.TemporaryDirectory() as tmp:
            one_sided = edit_result(
                Path(tmp),
                {7: make_run(20, 4, layer=7, run_id="r7", permanent=None)},
                {7: make_run(20, 8, layer=7, run_id="e7", permanent=None)},
            )
    finally:
        metrics.relocation_delta = original
    assert one_sided["k_layers"] == [7]
    assert one_sided["max_change_layers"] == []
    assert one_sided["edit_relocation"] == "not-applicable"


def test_edit_relocation_refuses_lesions_and_bad_lineage_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, provenance = write_edit_lineage(root / "good")
        kwargs = dict(
            recovered_base=make_run(20, 20, run_id="rb", permanent=None),
            recovered_layers={
                7: make_run(20, 4, layer=7, run_id="r7", permanent=None),
            },
            edited_base=make_run(20, 16, run_id="eb", permanent=None),
            edited_layers={
                7: make_run(20, 12, layer=7, run_id="e7", permanent=None),
            },
            edit_manifest_path=manifest,
            init_provenance_path=provenance,
            edit_layers=(6, 7, 8),
            n_boot=50,
        )
        broken = dict(kwargs)
        broken["recovered_layers"] = {
            7: make_run(20, 4, layer=7, run_id="r7", permanent=1),
        }
        try:
            evaluate_edit_relocation(**broken)
        except ValueError as exc:
            assert "permanent_bypassed_layer" in str(exc)
        else:
            raise AssertionError("edit relocation accepted a permanent lesion")

        null_manifest, null_provenance = write_edit_lineage(
            root / "null", null_layers=True
        )
        broken = dict(kwargs, edit_manifest_path=null_manifest,
                      init_provenance_path=null_provenance)
        try:
            evaluate_edit_relocation(**broken)
        except ValueError as exc:
            assert "config.train_layers is null" in str(exc)
        else:
            raise AssertionError("null edit train_layers was accepted")

        outside_manifest, outside_provenance = write_edit_lineage(
            root / "outside", outside=True
        )
        broken = dict(kwargs, edit_manifest_path=outside_manifest,
                      init_provenance_path=outside_provenance)
        try:
            evaluate_edit_relocation(**broken)
        except ValueError as exc:
            assert "outside edit run out_dir" in str(exc)
        else:
            raise AssertionError("outside edit initialization was accepted")


def test_voided_side_is_nulled_and_dropped_from_selection():
    """Ruling item 2: a run/condition above the 0.20 bound is void -- its side
    reports no A, delta is unmeasurable, and the layer never reaches k /
    max-change / candidates. Only the offending side is nulled."""
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {
                7: make_run(20, 4, layer=7, run_id="r7", permanent=None,
                            n_trunc_inc=5),                # 0.25 truncated
                10: make_run(20, 12, layer=10, run_id="r10", permanent=None),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7", permanent=None),
                10: make_run(20, 12, layer=10, run_id="e10", permanent=None),
            },
        )
    points = {point["layer"]: point for point in result["points"]}
    p7 = points[7]
    assert p7["A_recovered"] is None
    assert p7["A_lesioned"] is not None            # the clean side survives
    assert p7["delta_l"] is None and p7["delta_ci_low"] is None
    assert p7["reason"] == "voided_validity"
    assert p7["voided"] == ["recovered_bypassed:incentive"]
    assert abs(p7["invalid_rates"]["recovered_bypassed"]["incentive"] - 0.25) < 1e-12
    assert p7["invalid_rates"]["recovered_base"]["incentive"] == 0.0
    assert points[10]["voided"] == [] and points[10]["reason"] is None
    for key in ("k_layers", "max_change_layers", "candidate_layers"):
        assert 7 not in result[key], (key, result[key])
    assert result["k_layers"] == [10]
    assert result["invalid_max"] == metrics.INVALID_RATE_MAX
    report = edit_relocation_report(result)
    assert "| voided_validity |" in report
    assert "layer 7 [recovered_bypassed incentive=0.25]" in report
    assert "> 0.20" in report


def test_void_is_strict_and_counts_invalid_rows():
    """Exactly at the bound is measured; non-truncated invalid rows count."""
    with tempfile.TemporaryDirectory() as tmp:
        at_bound = edit_result(
            Path(tmp) / "at",
            {7: make_run(20, 4, layer=7, run_id="r7", permanent=None,
                         n_trunc_inc=4)},                  # 0.20 exactly
            {7: make_run(20, 12, layer=7, run_id="e7", permanent=None)},
        )
        invalid_rows = edit_result(
            Path(tmp) / "inv",
            {7: make_run(20, 4, layer=7, run_id="r7", permanent=None,
                         n_invalid_inc=5)},                # 0.25 invalid
            {7: make_run(20, 12, layer=7, run_id="e7", permanent=None)},
        )
    p_at = at_bound["points"][0]
    assert p_at["voided"] == [] and p_at["A_recovered"] is not None
    assert abs(p_at["invalid_rates"]["recovered_bypassed"]["incentive"] - 0.20) < 1e-12
    p_inv = invalid_rows["points"][0]
    assert p_inv["voided"] == ["recovered_bypassed:incentive"]
    assert p_inv["A_recovered"] is None and p_inv["reason"] == "voided_validity"
    assert "voided (per-condition invalid rate" in edit_relocation_report(at_bound)
    assert "> 0.20): none" in edit_relocation_report(at_bound)


def test_voided_base_run_voids_every_layer_on_that_side():
    """A void BASE run (either condition) voids its whole side; the lesion
    path's structural null is untouched by the rate check."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest, provenance = write_edit_lineage(Path(tmp))
        result = evaluate_edit_relocation(
            make_run(20, 20, run_id="rb", permanent=None),
            {
                7: make_run(20, 4, layer=7, run_id="r7", permanent=None),
                10: make_run(20, 12, layer=10, run_id="r10", permanent=None),
            },
            make_run(20, 16, run_id="eb", permanent=None, n_invalid_ctl=6),
            {
                7: make_run(20, 12, layer=7, run_id="e7", permanent=None),
                10: make_run(20, 12, layer=10, run_id="e10", permanent=None),
            },
            manifest, provenance, (6, 7, 8), n_boot=50, seed=0,
        )
    for point in result["points"]:
        assert point["A_lesioned"] is None and point["delta_l"] is None
        assert point["voided"] == ["lesioned_base:control"]
        assert point["A_recovered"] is not None
    assert result["max_change_layers"] == []
    assert result["k_layers"] == [7]
    assert result["edit_relocation"] == "not-applicable"

    lesion = evaluate_relocation(
        make_run(20, 20, run_id="rb"),
        {0: make_run(20, 4, layer=0, run_id="r0", n_trunc_inc=10)},
        make_run(20, 16, run_id="lb"),
        {0: make_run(20, 12, layer=0, run_id="l0")},
        lesioned_layer=1, n_boot=50, seed=0,
    )
    by_layer = {point["layer"]: point for point in lesion["points"]}
    assert by_layer[1]["reason"] == "permanently_lesioned_structural_null"
    assert by_layer[1]["voided"] == [] and by_layer[1]["invalid_rates"] is None
    assert by_layer[0]["reason"] == "voided_validity"
    assert "layer 0 [recovered_bypassed incentive=0.50]" in relocation_report(lesion)


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
