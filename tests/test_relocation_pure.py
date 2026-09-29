"""Dependency-free-tier tests for the paired Stage-3 edit-relocation analysis."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import metrics
from algoverse.relocation import (
    edit_relocation_report,
    evaluate_edit_relocation,
)
from algoverse.train import record_init_provenance


def make_run(n, d_inc, layer=None, run_id="base",
             n_trunc_inc=0, n_invalid_inc=0, n_invalid_ctl=0):
    """2n rows. The first n_trunc_inc incentive rows hit the token cap but
    stay valid as-scored (the pre-rule shape); the last n_invalid_inc /
    n_invalid_ctl rows of each condition are invalid (deceptive None)."""
    rows = []
    for i in range(n):
        for condition in ("incentive", "control"):
            n_invalid = n_invalid_inc if condition == "incentive" else n_invalid_ctl
            invalid = i >= n - n_invalid
            truncated = condition == "incentive" and i < n_trunc_inc
            gen = {
                "bypass_impl": (
                    "block-output-identity-hook/v1" if layer is not None else None
                ),
                "quant": "4bit",
                "do_sample": False,
                "max_new_tokens": 256,
                "model_revision": "rev",
                "adapter_digest": "digest",
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
    adapter = (
        root / "other-run" / "checkpoints" / "step-00281"
        if outside else edit_run / "checkpoints" / "step-00281"
    )
    record_init_provenance(
        continuation, adapter,
        {"checkpoint_step": 281, "train_seed": 42, "objective": "control"},
    )
    return manifest, continuation / "init_provenance.json"


def edit_result(root, recovered, edited, edit_layers=(6, 7, 8),
                recovered_base=None, edited_base=None):
    manifest, provenance = write_edit_lineage(root, edit_layers=edit_layers)
    return evaluate_edit_relocation(
        recovered_base or make_run(20, 20, run_id="rb"),
        recovered,
        edited_base or make_run(20, 16, run_id="eb"),
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
        make_run(20, 16), make_run(20, 12, layer=0, run_id="ep"),
        n_boot=100, seed=0,
    )
    assert abs(result["A_recovered"] - 0.8) < 1e-12
    assert abs(result["A_edited"] - 0.2) < 1e-12
    assert abs(result["delta_l"] - 0.6) < 1e-12
    assert result["paired"] is True
    assert result["n_scenarios_common"] == 20
    assert result["delta_ci_low"] is not None
    assert result["n_scenarios_edited_base"] == 20


def test_candidates_and_final_classifications():
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {
                0: make_run(20, 4, layer=0, run_id="r0"),
                2: make_run(20, 10, layer=2, run_id="r2"),
            },
            {
                0: make_run(20, 12, layer=0, run_id="e0"),
                2: make_run(20, 8, layer=2, run_id="e2"),
            },
        )
    assert result["k"] == 0
    assert result["k_layers"] == [0]
    assert result["max_change_layers"] == [0]
    assert result["candidate_layers"] == [0]
    # Layer 0 lies outside the (6, 7, 8) window: the rule says relocated.
    assert result["edit_relocation"] == "relocated"
    measurements = edit_relocation_report(result)
    assert "CLASSIFICATIONS: PENDING" in measurements
    assert "coverage shared/rb/rp/eb/ep" in measurements

    try:
        edit_relocation_report(
            result, final=True, dispersion="concentrated", origins={},
        )
    except ValueError as exc:
        assert "candidate layers" in str(exc)
    else:
        raise AssertionError("final report accepted missing origin review")
    try:
        edit_relocation_report(
            result, final=True, dispersion="somewhat",
            origins={0: "reconstructed"},
        )
    except ValueError as exc:
        assert "dispersion" in str(exc)
    else:
        raise AssertionError("final report accepted an unknown dispersion")

    final = edit_relocation_report(
        result, final=True, dispersion="concentrated",
        origins={0: "reconstructed"},
    )
    assert "dispersion: concentrated" in final
    assert "layer 0 origin: reconstructed" in final
    assert "PENDING" not in final


def test_exact_recovered_ties_all_require_origin_review():
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {
                0: make_run(20, 4, layer=0, run_id="r0"),
                2: make_run(20, 4, layer=2, run_id="r2"),
            },
            {
                0: make_run(20, 12, layer=0, run_id="e0"),
                2: make_run(20, 8, layer=2, run_id="e2"),
            },
        )
    assert result["k"] == 0
    assert result["k_layers"] == [0, 2]
    assert result["candidate_layers"] == [0, 2]


def test_partial_overlap_is_reported_as_a_gap_with_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {0: make_run(20, 4, layer=0, run_id="rp")},
            {0: make_run(15, 9, layer=0, run_id="ep")},
            edited_base=make_run(15, 12, run_id="eb"),
        )
    point = next(point for point in result["points"] if point["layer"] == 0)
    assert point["reason"] == "partial_overlap"
    assert point["n_scenarios_common"] == 15
    report = edit_relocation_report(result)
    assert "15/20/20/15/15" in report
    assert "layer 0=partial_overlap" in report


def test_missing_layer_run_is_a_named_gap_on_the_missing_side():
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {7: make_run(20, 4, layer=7, run_id="r7")},
            {
                7: make_run(20, 12, layer=7, run_id="e7"),
                10: make_run(20, 12, layer=10, run_id="e10"),
            },
        )
    points = {point["layer"]: point for point in result["points"]}
    assert points[10]["reason"] == "missing_recovered_layer_run"
    assert points[10]["delta_l"] is None and points[10]["voided"] == []
    assert points[7]["reason"] is None
    assert result["incomplete"] == {"recovered": [10], "edited": []}
    report = edit_relocation_report(result)
    assert "layer 10=missing_recovered_layer_run" in report
    assert "INCOMPLETE: no recovered-side run for layer(s) 10" in report
    assert "edit relocation (precommitted rule): recovered-in-place [INCOMPLETE]" in report
    with tempfile.TemporaryDirectory() as tmp:
        reverse = edit_result(
            Path(tmp),
            {
                7: make_run(20, 4, layer=7, run_id="r7"),
                10: make_run(20, 4, layer=10, run_id="r10"),
            },
            {7: make_run(20, 12, layer=7, run_id="e7")},
        )
    reverse_points = {point["layer"]: point for point in reverse["points"]}
    assert reverse_points[10]["reason"] == "missing_edited_layer_run"
    assert reverse["incomplete"] == {"recovered": [], "edited": [10]}
    assert "INCOMPLETE: no edited-side run for layer(s) 10" in (
        edit_relocation_report(reverse)
    )
    # A gap layer carries no A on either side, so it takes no part in k,
    # the change ranking, or the candidates: "no measurement" is not zero.
    assert reverse_points[10]["A_recovered"] is None
    assert reverse["k_layers"] == [7]
    assert reverse["max_change_layers"] == [7]
    assert reverse["candidate_layers"] == [7]


def test_max_change_uses_signed_delta_not_absolute_magnitude():
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {
                5: make_run(20, 18, layer=5, run_id="r5"),
                7: make_run(20, 16, layer=7, run_id="r7"),
            },
            {
                5: make_run(20, 17, layer=5, run_id="e5"),
                7: make_run(20, 0, layer=7, run_id="e7"),
            },
            edited_base=make_run(20, 20, run_id="eb"),
        )
    points = {point["layer"]: point for point in result["points"]}
    assert abs(points[5]["delta_l"] - (-0.05)) < 1e-12
    assert abs(points[7]["delta_l"] - (-0.80)) < 1e-12
    assert result["k_layers"] == [7]
    assert result["max_change_layers"] == [5]
    assert result["candidate_layers"] == [5, 7]
    report = edit_relocation_report(result)
    assert "maximum signed delta_l" in report


def test_edit_relocation_partitions_and_precommitted_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        inside = edit_result(
            Path(tmp) / "inside",
            {
                7: make_run(20, 4, layer=7, run_id="r7"),
                10: make_run(20, 12, layer=10, run_id="r10"),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7"),
                10: make_run(20, 12, layer=10, run_id="e10"),
            },
        )
        assert inside["edit_relocation"] == "recovered-in-place"
        assert inside["edit_partition"]["candidate_layers"] == {
            "inside": [7], "outside": [],
        }
        report = edit_relocation_report(inside)
        assert "edited layers: [6, 7, 8]" in report
        assert "A_l just-edited" in report
        assert inside["incomplete"] == {"recovered": [], "edited": []}
        assert "INCOMPLETE" not in report

        relocated = edit_result(
            Path(tmp) / "relocated",
            {
                7: make_run(20, 12, layer=7, run_id="r7"),
                10: make_run(20, 4, layer=10, run_id="r10"),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7"),
                10: make_run(20, 12, layer=10, run_id="e10"),
            },
        )
        assert relocated["edit_relocation"] == "relocated"

        mixed = edit_result(
            Path(tmp) / "mixed",
            {
                7: make_run(20, 4, layer=7, run_id="r7"),
                10: make_run(20, 8, layer=10, run_id="r10"),
            },
            {
                7: make_run(20, 0, layer=7, run_id="e7"),
                10: make_run(20, 16, layer=10, run_id="e10"),
            },
        )
        assert mixed["k_layers"] == [7]
        assert mixed["max_change_layers"] == [10]
        assert mixed["edit_relocation"] == "mixed"
        final = edit_relocation_report(
            mixed, final=True, dispersion="concentrated",
            origins={7: "strengthened", 10: "reconstructed"},
        )
        assert "edit relocation (precommitted rule): mixed" in final
        assert "layer 10 origin: reconstructed" in final


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
            "A_edited": None,
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
                {7: make_run(20, 4, layer=7, run_id="r7")},
                {7: make_run(20, 8, layer=7, run_id="e7")},
            )
    finally:
        metrics.relocation_delta = original
    assert one_sided["k_layers"] == [7]
    assert one_sided["max_change_layers"] == []
    assert one_sided["edit_relocation"] == "not-applicable"


def test_edit_relocation_refuses_probed_bases_and_bad_lineage_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, provenance = write_edit_lineage(root / "good")
        kwargs = dict(
            recovered_base=make_run(20, 20, run_id="rb"),
            recovered_layers={7: make_run(20, 4, layer=7, run_id="r7")},
            edited_base=make_run(20, 16, run_id="eb"),
            edited_layers={7: make_run(20, 12, layer=7, run_id="e7")},
            edit_manifest_path=manifest,
            init_provenance_path=provenance,
            edit_layers=(6, 7, 8),
            n_boot=50,
        )
        # A base run whose rows record a bypass implementation is not the
        # unprobed model: refused by load_sweep_inputs, by name.
        probed_base = make_run(20, 16, run_id="eb")
        for row in probed_base:
            row["gen_config"]["bypass_impl"] = "block-output-identity-hook/v1"
        try:
            evaluate_edit_relocation(**dict(kwargs, edited_base=probed_base))
        except ValueError as exc:
            assert "bypass_impl" in str(exc)
        else:
            raise AssertionError("edit relocation accepted a probed base run")

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
    """A run/condition above the 0.20 bound is void -- its side reports no
    A, delta is unmeasurable, and the layer never reaches k / max-change /
    candidates. Only the offending side is nulled."""
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {
                7: make_run(20, 4, layer=7, run_id="r7",
                            n_trunc_inc=5),                # 0.25 truncated
                10: make_run(20, 12, layer=10, run_id="r10"),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7"),
                10: make_run(20, 12, layer=10, run_id="e10"),
            },
        )
    points = {point["layer"]: point for point in result["points"]}
    p7 = points[7]
    assert p7["A_recovered"] is None
    assert p7["A_edited"] is not None              # the clean side survives
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
            {7: make_run(20, 4, layer=7, run_id="r7",
                         n_trunc_inc=4)},                  # 0.20 exactly
            {7: make_run(20, 12, layer=7, run_id="e7")},
        )
        invalid_rows = edit_result(
            Path(tmp) / "inv",
            {7: make_run(20, 4, layer=7, run_id="r7",
                         n_invalid_inc=5)},                # 0.25 invalid
            {7: make_run(20, 12, layer=7, run_id="e7")},
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
    """A void BASE run (either condition) voids its whole side."""
    with tempfile.TemporaryDirectory() as tmp:
        result = edit_result(
            Path(tmp),
            {
                7: make_run(20, 4, layer=7, run_id="r7"),
                10: make_run(20, 12, layer=10, run_id="r10"),
            },
            {
                7: make_run(20, 12, layer=7, run_id="e7"),
                10: make_run(20, 12, layer=10, run_id="e10"),
            },
            edited_base=make_run(20, 16, run_id="eb", n_invalid_ctl=6),
        )
    for point in result["points"]:
        assert point["A_edited"] is None and point["delta_l"] is None
        assert point["voided"] == ["edited_base:control"]
        assert point["A_recovered"] is not None
    assert result["max_change_layers"] == []
    assert result["k_layers"] == [7]
    assert result["edit_relocation"] == "not-applicable"
    assert "edited_base control=0.30" in edit_relocation_report(result)



def test_relocation_report_main_end_to_end():
    import contextlib
    import io

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import relocation_report

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, provenance = write_edit_lineage(root)

        def dump(name, rows):
            path = root / name
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return str(path)

        argv = [
            "--recovered-base", dump("rb.jsonl", make_run(20, 20, run_id="rb")),
            "--edited-base", dump("eb.jsonl", make_run(20, 16, run_id="eb")),
            "--edit-manifest", str(manifest), "--init-provenance", str(provenance),
            "--edit-layers", "6", "7", "8", "--n-boot", "50",
            "--emit-curves", str(root / "delta"),
        ]
        for layer, rec, ed in ((7, 4, 12), (10, 12, 12)):
            argv += ["--recovered-layer", "%d=%s" % (
                layer, dump("r%d.jsonl" % layer, make_run(20, rec, layer=layer, run_id="r%d" % layer)))]
            argv += ["--edited-layer", "%d=%s" % (
                layer, dump("e%d.jsonl" % layer, make_run(20, ed, layer=layer, run_id="e%d" % layer)))]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            assert relocation_report.main(argv) == 0
        report = printed.getvalue()
        assert "edit relocation (precommitted rule): recovered-in-place" in report
        assert "INCOMPLETE" not in report
        for side in ("recovered", "edited"):
            curve = json.loads((root / ("delta-%s.json" % side)).read_text())
            assert [p["bypassed_layer"] for p in curve] == [7, 10]
        # A bad layer spec is a usage error, not a traceback.
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                relocation_report.main(argv + ["--recovered-layer", "x=/nowhere"])
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("a non-integer layer key was accepted")


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
