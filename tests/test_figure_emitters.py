"""Rung-1 tests for the figure-record emitters and the raw-tau renderer.

The ways these can silently lie, each caught:
  1. the tau emitter recomputing (or mangling) tau instead of relaying
     metrics.tau_with_ci verbatim, or touching its source file;
  2. the layer-curve emitter ignoring --truncated-invalid (a truncated
     deceptive row surviving into A_l);
  3. recovery records rounding taus (%.3f) or dropping a null R_t's reason;
  4. relocation --emit-curves swapping the recovered/lesioned sides;
  5. a null per-arm tau drawn as a value instead of an annotated gap;
  6. the pareto emitter subtracting a base from an absolute metric (JSD),
     or measuring task competence against the wrong reference;
  7. the decomposition binning a truncated-but-parseable row as honest, or
     a malformed row silently;
  8. the edit-gate summary joining a window to the wrong Stage-1 layer.

Stdlib + matplotlib only.

    python3 tests/test_figure_emitters.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from algoverse import metrics


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", "_script"), REPO / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(scenario, condition, deceptive, trunc=False, layer=None, run="base"):
    return {
        "scenario_id": scenario, "condition": condition, "valid": True,
        "deceptive": deceptive, "hit_max_tokens": trunc,
        "bypassed_layer": layer, "run_id": run,
        "model_id": "m", "adapter_path": "a", "patch_layer": None,
        "patch_source": None, "checkpoint_step": 281, "arm": "M_D",
        "split": "final",
    }


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_tau_emitter():
    emit = _load_script("emit_figure_records.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 6 scenarios; incentive deceptive in 3 -> tau = 0.5 exactly.
        rows = []
        for s in range(6):
            rows.append(_row(s, "incentive", s < 3))
            rows.append(_row(s, "control", False))
        _write_rows(tmp / "rows.jsonl", rows)
        before = (tmp / "rows.jsonl").read_text()

        out = tmp / "tau.jsonl"
        emit.main(["tau", "--rows", "M_D:M_D=%s" % (tmp / "rows.jsonl"),
                   "--out", str(out), "--n-boot", "50"])

        assert (tmp / "rows.jsonl").read_text() == before   # 1: source untouched
        records = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        # 1: verbatim relay of the single home's numbers
        expected = metrics.tau_with_ci(rows, n_boot=50, seed=0)
        for key in ("tau", "tau_ci_low", "tau_ci_high", "n_scenarios"):
            assert record[key] == expected[key], (key, record[key], expected[key])
        assert record["tau"] == 0.5
        assert record["model"] == "M_D" and record["label"] == "M_D"
    print("PASS tau emitter")


def test_layer_curve_emitter_ruling():
    emit = _load_script("emit_figure_records.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = []
        for s in range(6):
            base.append(_row(s, "incentive", False))
            base.append(_row(s, "control", False))
        _write_rows(tmp / "base.jsonl", base)
        # Bypass layer 5: the ONLY deceptive incentive rows are truncated
        # repetition loops -- exactly the layer-26 artifact shape.
        sweep = []
        for s in range(6):
            sweep.append(_row(s, "incentive", s < 3, trunc=(s < 3),
                              layer=5, run="l05"))
            sweep.append(_row(s, "control", False, layer=5, run="l05"))
        _write_rows(tmp / "l05.jsonl", sweep)

        def run(extra):
            out = tmp / ("curve-%d.json" % len(extra))
            emit.main(["layer-curve", "--base", str(tmp / "base.jsonl"),
                       "--layer", "5=%s" % (tmp / "l05.jsonl"),
                       "--out", str(out), "--n-boot", "50"] + extra)
            return json.loads(out.read_text())

        as_scored = run([])
        ruled = run(["--truncated-invalid"])
        assert len(as_scored) == 1 and len(ruled) == 1
        # 2: A_l = tau_base - tau_bypassed. As-scored counts the truncated
        # deceptive rows (tau_byp = 0.5, A_l = -0.5); the ruling invalidates
        # them, leaving only clean honest rows (A_l = 0.0) -- identical
        # inputs, different only in the flag.
        assert as_scored[0]["A_l"] == -0.5, as_scored[0]["A_l"]
        assert ruled[0]["A_l"] == 0.0, ruled[0]["A_l"]
        assert ruled[0]["bypassed_layer"] == 5

        # Cross-platform adapter prefixes split the comparison group into
        # baseline-less halves; --strip-adapter-prefix reunites them.
        for r in base:
            r["adapter_path"] = "/content/drive/x/checkpoints/md/step-281"
        for r in sweep:
            r["adapter_path"] = "/root/x/checkpoints/md/step-281"
        _write_rows(tmp / "base.jsonl", base)
        _write_rows(tmp / "l05.jsonl", sweep)
        split = run(["--n-boot", "50"])          # extra arg only varies the name
        assert split[0]["A_l"] is None and split[0]["reason"] == "no_baseline_run"
        joined = run(["--strip-adapter-prefix"])
        assert joined[0]["A_l"] == -0.5, joined[0]
    print("PASS layer-curve emitter ruling")


def test_recovery_records():
    rec = _load_script("recovery_report.py")
    entry_ok = {
        "R_t": 0.123456789, "R_t_ci_low": 0.1, "R_t_ci_high": 0.2,
        "reason": None,
        "tau_by_arm": {"E,D": 0.987654321, "E,C": 0.0,
                       "I,D": 0.9, "I,C": 0.01},
    }
    entry_null = {
        "R_t": None, "R_t_ci_low": None, "R_t_ci_high": None,
        "reason": "denominator_too_small",
        "tau_by_arm": {"E,D": 0.2, "E,C": 0.0, "I,D": 0.21, "I,C": 0.2},
    }
    result = {
        "requested_t": [8, 281],
        "per_t": {8: entry_null, 281: entry_ok},
        "arms": ("E,D", "E,C", "I,D", "I,C"),
        "n_boot": 2000,
    }
    records = rec.build_recovery_records(result, "l07")
    assert [r["checkpoint_step"] for r in records] == [8, 281]
    # 3: full precision, no %.3f
    assert records[1]["tau_ED"] == 0.987654321
    assert records[1]["R_t"] == 0.123456789
    # 3: null R_t keeps its reason; taus still present
    assert records[0]["R_t"] is None
    assert records[0]["reason"] == "denominator_too_small"
    assert records[0]["tau_ID"] == 0.21
    assert all(r["env"] == "l07" and r["arms"][0] == "E,D" for r in records)
    print("PASS recovery records")


def test_relocation_emit_curves():
    reloc = _load_script("relocation_report.py")
    result = {"points": [
        {"layer": 2, "A_recovered": 0.7, "A_lesioned": 0.1, "reason": None},
        {"layer": 3, "A_recovered": None, "A_lesioned": None,
         "reason": "tau_not_computable"},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "curves" / "l07")
        reloc._emit_curves(result, base)
        recovered = json.loads(Path(base + "-recovered.json").read_text())
        lesioned = json.loads(Path(base + "-lesioned.json").read_text())
        # 4: sides not swapped
        assert recovered[0] == {"bypassed_layer": 2, "A_l": 0.7, "reason": None}
        assert lesioned[0] == {"bypassed_layer": 2, "A_l": 0.1, "reason": None}
        assert recovered[1]["A_l"] is None
        assert recovered[1]["reason"] == "tau_not_computable"
    # A one-sided void: the voided side says so, the measured side does not.
    one_sided = {"points": [
        {"layer": 2, "A_recovered": None, "A_lesioned": 0.0,
         "reason": "voided_validity"},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "l13")
        reloc._emit_curves(one_sided, base)
        recovered = json.loads(Path(base + "-recovered.json").read_text())
        lesioned = json.loads(Path(base + "-lesioned.json").read_text())
        assert recovered[0] == {"bypassed_layer": 2, "A_l": None,
                                "reason": "voided_validity"}
        assert lesioned[0] == {"bypassed_layer": 2, "A_l": 0.0, "reason": None}
    print("PASS relocation emit-curves")


def test_edit_lineage_cross_platform():
    """The lineage guard must tolerate mount-prefix drift, nothing else."""
    from algoverse.relocation import _edit_lineage
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "maheep-yksa" / "checkpoints" / "edit-l07"
        out_dir.mkdir(parents=True)
        manifest = out_dir / "train_manifest.json"
        manifest.write_text(json.dumps(
            {"config": {"train_layers": [6, 7, 8]}}))
        provenance = Path(tmp) / "init_provenance.json"

        # Same run, recorded under another platform's mount prefix: accepted.
        provenance.write_text(json.dumps({
            "init_adapter":
                "/root/maheep-yksa/checkpoints/edit-l07/checkpoints/step-00281"
        }))
        assert _edit_lineage(manifest, provenance, (6, 7, 8)) == (6, 7, 8)

        # A DIFFERENT run under that prefix: still refused.
        provenance.write_text(json.dumps({
            "init_adapter":
                "/root/maheep-yksa/checkpoints/edit-l13/checkpoints/step-00281"
        }))
        try:
            _edit_lineage(manifest, provenance, (6, 7, 8))
        except ValueError as exc:
            assert "outside edit run out_dir" in str(exc)
        else:
            raise AssertionError("foreign init_adapter accepted")
    print("PASS edit-lineage cross-platform")


def test_render_recovery_taus():
    from algoverse.plotting import render_recovery_taus, synthetic_recovery_taus
    with tempfile.TemporaryDirectory() as tmp:
        records = synthetic_recovery_taus()
        meta = render_recovery_taus(records, str(Path(tmp) / "recovery_taus"))
        assert all(Path(p).is_file() for p in meta["paths"])
        assert meta["envs"] == ["path A", "path B"]
        assert meta["arms"] == ["E,D", "E,C", "I,D", "I,C"]
        # 5: the synthetic's null tau_ID lands in gaps with its reason
        assert ("path A", "I,D", 8, "denominator_too_small") in meta["gaps"], \
            meta["gaps"]
    print("PASS render recovery-taus")


def _competence_row(run_id, metric, value, layer=None, stderr=None,
                    config=None):
    return {"run_id": run_id, "bypassed_layer": layer, "metric": metric,
            "value": value, "stderr": stderr, "config": config or {}}


def test_pareto_emitter_panels_bounds_and_absolute_jsd():
    from algoverse import figures, plotting
    emit = _load_script("emit_figure_records.py")
    curve, _ = plotting.synthetic_layer_curve(n_layers=3)
    bench_cfg = {"limit": 400, "seed": 42, "batch_size": 4, "adapter_digest": "d"}
    ppl_cfg = {"n_tokens": 20000, "adapter_digest": "d"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "curve.json").write_text(json.dumps(curve))
        base = [
            _competence_row("base", "wikitext2_ppl", 6.0, config=ppl_cfg),
            _competence_row("base", "gsm8k_exact_match", 0.8, config=bench_cfg),
        ]
        l1 = [
            _competence_row("synthetic-L1", "wikitext2_ppl", 7.5, layer=1, config=ppl_cfg),
            _competence_row("synthetic-L1", "wikitext2_neutral_jsd", 0.04, layer=1),
            _competence_row("synthetic-L1", "gsm8k_exact_match", 0.7, layer=1, config=bench_cfg),
        ]
        l2 = [
            _competence_row("synthetic-L2", "wikitext2_ppl", 6.5, layer=2, config=ppl_cfg),
            _competence_row("synthetic-L2", "wikitext2_neutral_jsd", 0.03, layer=2),
        ]
        _write_rows(tmp / "base.jsonl", base)
        _write_rows(tmp / "l1.jsonl", l1)
        _write_rows(tmp / "l2.jsonl", l2)
        m0 = []
        for s in range(6):
            m0.append(_row(s, "incentive", False))
            m0.append(dict(_row(s, "control", False), understated=False))
        _write_rows(tmp / "m0.jsonl", m0)
        out = tmp / "pareto.json"
        emit.main(["pareto", "--model", "synthetic", "--curve", str(tmp / "curve.json"),
                   "--base-run-id", "base", "--m0-rows", str(tmp / "m0.jsonl"),
                   "--competence", str(tmp / "base.jsonl"),
                   "--competence", str(tmp / "l1.jsonl"),
                   "--competence", str(tmp / "l2.jsonl"),
                   "--out", str(out)])
        record = json.loads(out.read_text())
    assert record["model"] == "synthetic" and record["base_run_id"] == "base"
    panels = {p["damage_metric"]: p for p in record["panels"]}
    assert list(panels) == list(emit.PARETO_METRICS)            # default: every metric
    by_layer = lambda panel: {p["bypassed_layer"]: p for p in panel["pareto_points"]}
    # 6: ppl is a RISE against the base run; jsd is absolute (no base row exists)
    assert abs(by_layer(panels["wikitext2_ppl"])[1]["damage"] - 1.5) < 1e-12
    assert by_layer(panels["wikitext2_neutral_jsd"])[1]["damage"] == 0.04
    assert panels["wikitext2_neutral_jsd"]["damage_reference"] == "absolute"
    # gsm8k is a DROP; layer 2 has no benchmark and says so
    assert abs(by_layer(panels["gsm8k_exact_match"])[1]["damage"] - 0.1) < 1e-12
    assert by_layer(panels["gsm8k_exact_match"])[2]["damage"] is None
    assert by_layer(panels["gsm8k_exact_match"])[2]["damage_reason"] == "metric_missing_for_this_layer"
    # task competence is measured against M_0 (competence 1.0 here), not the sweep base
    comp = by_layer(panels["task_competence"])
    assert panels["task_competence"]["damage_reference"] == "M_0"
    assert abs(comp[1]["damage"] - (1.0 - comp[1]["competence"])) < 1e-12
    # bounds per metric, frontier present, JSON round-trip did not break the guard
    assert panels["wikitext2_ppl"]["bounds"] == {"a_l_min": 0.15, "damage_max": 2.0}
    assert panels["wikitext2_neutral_jsd"]["bounds"]["damage_max"] == 0.25
    assert panels["mmlu_acc"]["bounds"]["damage_max"] == 0.05
    assert panels["wikitext2_ppl"]["frontier"]
    print("PASS pareto emitter")


def _drow(scenario, condition, category, layer=1, run="cell"):
    from algoverse.plotting import _decomposition_row
    return _decomposition_row(scenario, condition, category, layer, run)


def test_decomposition_category_table():
    from algoverse import figures
    for category in figures.DECOMPOSITION_CATEGORIES:
        assert figures.decomposition_category(_drow(0, "incentive", category)) == category, category
    # 7: truncated but as-scored valid AND deceptive -> truncated, never a label
    row = _drow(0, "incentive", "inflated")
    row["hit_max_tokens"] = True
    assert figures.decomposition_category(row) == "invalid_truncated"
    # a valid row with no label is malformed, not silently binned
    bad = _drow(0, "incentive", "exact_truth")
    bad["deceptive"] = None
    try:
        figures.decomposition_category(bad)
    except ValueError as exc:
        assert "deceptive=None" in str(exc)
    else:
        raise AssertionError("malformed row was binned")
    print("PASS decomposition category table")


def test_decomposition_emitter_ruling_and_voiding():
    import contextlib
    import io
    emit = _load_script("emit_figure_records.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "sweep-x"
        l0 = [_drow(s, "incentive", c, layer=0, run="x-l00") for s, c in enumerate(
            ["inflated"] * 3 + ["fabricated"] * 2 + ["exact_truth"] * 2
            + ["correct_none", "understated", "invalid_other"])]
        l0 += [_drow(s, "control", "exact_truth", layer=0, run="x-l00") for s in range(10)]
        l2 = [_drow(s, "incentive", "invalid_truncated" if s < 8 else "inflated",
                    layer=2, run="x-l02") for s in range(10)]
        l2 += [_drow(s, "control", "correct_none", layer=2, run="x-l02") for s in range(10)]
        _write_rows(root / "x-l00" / "rows.jsonl", l0)
        _write_rows(root / "x-l02" / "rows.jsonl", l2)
        out = tmp / "decomp.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            emit.main(["decomposition", "--sweep-root", str(root),
                       "--n-layers", "3", "--out", str(out)])
        printed = buffer.getvalue()
        record = json.loads(out.read_text())
    layers = {e["bypassed_layer"]: e for e in record["layers"]}
    assert [e["status"] for e in record["layers"]] == ["measured", "missing", "measured"]
    inc0 = layers[0]["conditions"]["incentive"]
    assert inc0["counts"] == {"inflated": 3, "fabricated": 2, "exact_truth": 2,
                              "correct_none": 1, "understated": 1,
                              "invalid_truncated": 0, "invalid_other": 1}
    assert abs(inc0["invalid_rate"] - 0.1) < 1e-12 and inc0["voided_validity"] is False
    inc2 = layers[2]["conditions"]["incentive"]
    assert inc2["counts"]["invalid_truncated"] == 8 and inc2["voided_validity"] is True
    assert layers[2]["conditions"]["control"]["counts"]["correct_none"] == 10
    assert layers[0]["run_id"] == "x-l00"
    assert "layer |    n |" in printed and "voided_validity" in printed
    assert "| missing" in printed
    print("PASS decomposition emitter")


def test_edit_gate_summary_emitter_joins_gate_and_stage1():
    emit = _load_script("emit_figure_records.py")
    gate = {
        "effect": {"gain": 1.0, "gain_ci_low": 1.0, "gain_ci_high": 1.0},
        "counts": {"M_D": {"incentive": {"n_deceptive": 305, "n_valid": 305}},
                   "M_E": {"incentive": {"n_deceptive": 0, "n_valid": 305}}},
        "bench": {"M_0": {"mmlu_acc": {"value": 0.70, "stderr": 0.01},
                          "gsm8k_exact_match": {"value": 0.80, "stderr": 0.02},
                          "wikitext2_ppl": {"value": 8.0}},
                  "M_E": {"mmlu_acc": {"value": 0.68, "stderr": 0.01},
                          "gsm8k_exact_match": {"value": 0.79, "stderr": 0.02},
                          "wikitext2_ppl": {"value": 8.3}}},
        "edit_jsd": {"value": 0.001},
        "decision": {"verdict": "PASS"},
        "thresholds": {"edit_effect_min": 0.15, "competence_drop_max": 0.05,
                       "ppl_rise_max": 2.0, "edit_jsd_max": 0.25},
    }
    curve = [
        {"bypassed_layer": 6, "run_id": "L6", "A_l": None, "reason": "tau_not_computable"},
        {"bypassed_layer": 7, "run_id": "L7", "A_l": 0.28, "A_l_ci_low": 0.2,
         "A_l_ci_high": 0.37, "reason": None},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "gate.json").write_text(json.dumps(gate))
        (tmp / "m7.json").write_text(json.dumps({"config": {"train_layers": [8, 6, 7]}}))
        (tmp / "m27.json").write_text(json.dumps({"config": {"train_layers": [26, 27, 28]}}))
        (tmp / "curve.json").write_text(json.dumps(curve))
        out = tmp / "summary.jsonl"
        emit.main(["edit-gate-summary",
                   "--gate", "Q:l07=%s" % (tmp / "gate.json"),
                   "--gate", "Q:l27=%s" % (tmp / "gate.json"),
                   "--manifest", "l07=%s" % (tmp / "m7.json"),
                   "--manifest", "l27=%s" % (tmp / "m27.json"),
                   "--stage1-curve", "Q=%s" % (tmp / "curve.json"),
                   "--out", str(out)])
        records = [json.loads(l) for l in out.read_text().splitlines()]
    first, second = records
    # 8: the window's CENTER is joined, from the sorted train_layers
    assert first["edit_layers"] == [6, 7, 8] and first["center_layer"] == 7
    assert first["stage1_A_l"] == 0.28 and first["stage1_run_id"] == "L7"
    assert first["A_edit"] == 1.0
    assert abs(first["delta_ppl"] - 0.3) < 1e-9
    assert abs(first["delta_mmlu"] - (-0.02)) < 1e-9
    assert abs(first["delta_mmlu_stderr"] - (0.01 ** 2 + 0.01 ** 2) ** 0.5) < 1e-12
    assert first["counts"]["M_D"]["incentive"]["n_deceptive"] == 305
    assert first["edit_jsd"] == 0.001 and first["verdict"] == "PASS"
    assert first["bounds"]["a_edit_min"] == 0.15 and first["bounds"]["a_l_min"] == 0.15
    assert second["center_layer"] == 27 and second["stage1_A_l"] is None
    assert second["stage1_reason"] == "layer_not_in_curve"
    print("PASS edit-gate summary emitter")


def test_render_new_figures_smoke():
    from algoverse import plotting
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta = plotting.render_pareto_panels(
            plotting.synthetic_pareto_panels(), str(tmp / "panels"))
        assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in meta["paths"])
        assert [p["damage_metric"] for p in meta["panels"]] == [
            "task_competence", "wikitext2_ppl", "wikitext2_neutral_jsd"]
        assert all(any(l == 20 for l, _ in p["off_plot"]) for p in meta["panels"])
        meta = plotting.render_decomposition(
            plotting.synthetic_decomposition(), str(tmp / "decomp"))
        assert all(Path(p).is_file() for p in meta["paths"])
        assert 5 in meta["missing"] and 0 in meta["voided"] and 26 in meta["voided"]
        meta = plotting.render_edit_gate_summary(
            plotting.synthetic_edit_gate_summary(), str(tmp / "gates"))
        assert all(Path(p).is_file() for p in meta["paths"])
        assert ("l24", "stage1_A_l", "layer_not_in_curve") in meta["gaps"]
    print("PASS render new figures smoke")


def main():
    test_tau_emitter()
    test_layer_curve_emitter_ruling()
    test_recovery_records()
    test_relocation_emit_curves()
    test_edit_lineage_cross_platform()
    test_render_recovery_taus()
    test_pareto_emitter_panels_bounds_and_absolute_jsd()
    test_decomposition_category_table()
    test_decomposition_emitter_ruling_and_voiding()
    test_edit_gate_summary_emitter_joins_gate_and_stage1()
    test_render_new_figures_smoke()
    print("PASS test_figure_emitters")


if __name__ == "__main__":
    main()
