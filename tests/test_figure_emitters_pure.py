"""Dependency-free-tier tests for the figure-record emitters.

The ways these can silently lie, each caught:
  1. the tau emitter recomputing (or mangling) tau instead of relaying
     metrics.tau_with_ci verbatim, or touching its source file;
  2. the layer-curve emitter ignoring the truncation rule (a truncated
     deceptive row surviving into A_l);
  3. recovery records rounding taus (%.3f) or dropping a null R_t's reason;
  4. relocation --emit-curves swapping the recovered/edited sides;
  5. the transfer emitter recomputing a tau or silently dropping an arm.

Stdlib only.

    python3 tests/test_figure_emitters_pure.py
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

        ruled = run([])
        assert len(ruled) == 1
        # 2: A_l = tau_base - tau_bypassed. The truncation rule invalidates
        # the truncated deceptive rows, leaving only clean honest rows
        # (A_l = 0.0). The same rows read as scored would count them
        # (tau_byp = 0.5, A_l = -0.5): the rule is the only difference.
        assert ruled[0]["A_l"] == 0.0, ruled[0]["A_l"]
        assert ruled[0]["bypassed_layer"] == 5
        from algoverse import figures, metrics
        as_scored = figures.layer_curve(
            metrics.normalize_rows(base + sweep, truncated_invalid=False),
            n_boot=50, seed=0,
        )
        assert as_scored[0]["A_l"] == -0.5, as_scored[0]["A_l"]

        # Cross-machine adapter prefixes split the comparison group into
        # baseline-less halves; --strip-adapter-prefix reunites them.
        for r in base:
            r["adapter_path"] = "/kaggle/working/x/checkpoints/md/step-281"
        for r in sweep:
            r["adapter_path"] = "/root/x/checkpoints/md/step-281"
        _write_rows(tmp / "base.jsonl", base)
        _write_rows(tmp / "l05.jsonl", sweep)
        split = run(["--n-boot", "50"])          # extra arg only varies the name
        assert split[0]["A_l"] is None and split[0]["reason"] == "no_baseline_run"
        joined = run(["--strip-adapter-prefix"])
        assert joined[0]["A_l"] == 0.0, joined[0]
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
        "eps": 0.10,
    }
    records = rec.build_recovery_records(result, "l07")
    assert all(r["eps"] == 0.10 and r["n_boot_dropped"] is None for r in records)
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
        {"layer": 2, "A_recovered": 0.7, "A_edited": 0.1, "reason": None},
        {"layer": 3, "A_recovered": None, "A_edited": None,
         "reason": "tau_not_computable"},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "curves" / "l07")
        reloc._emit_curves(result, base)
        recovered = json.loads(Path(base + "-recovered.json").read_text())
        edited = json.loads(Path(base + "-edited.json").read_text())
        # 4: sides not swapped
        assert recovered[0] == {"bypassed_layer": 2, "A_l": 0.7, "reason": None}
        assert edited[0] == {"bypassed_layer": 2, "A_l": 0.1, "reason": None}
        assert recovered[1]["A_l"] is None
        assert recovered[1]["reason"] == "tau_not_computable"
    # A one-sided void: the voided side says so, the measured side does not.
    one_sided = {"points": [
        {"layer": 2, "A_recovered": None, "A_edited": 0.0,
         "reason": "voided_validity"},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "l13")
        reloc._emit_curves(one_sided, base)
        recovered = json.loads(Path(base + "-recovered.json").read_text())
        edited = json.loads(Path(base + "-edited.json").read_text())
        assert recovered[0] == {"bypassed_layer": 2, "A_l": None,
                                "reason": "voided_validity"}
        assert edited[0] == {"bypassed_layer": 2, "A_l": 0.0, "reason": None}
    print("PASS relocation emit-curves")


def test_edit_lineage_cross_platform():
    """The lineage guard must tolerate mount-prefix drift, nothing else."""
    from algoverse.relocation import _edit_lineage
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "project" / "checkpoints" / "edit-l07"
        out_dir.mkdir(parents=True)
        manifest = out_dir / "train_manifest.json"
        manifest.write_text(json.dumps(
            {"config": {"train_layers": [6, 7, 8]}}))
        provenance = Path(tmp) / "init_provenance.json"

        # Same run, recorded under another platform's mount prefix: accepted.
        provenance.write_text(json.dumps({
            "init_adapter":
                "/root/project/checkpoints/edit-l07/checkpoints/step-00281"
        }))
        assert _edit_lineage(manifest, provenance, (6, 7, 8)) == (6, 7, 8)

        # A DIFFERENT run under that prefix: still refused.
        provenance.write_text(json.dumps({
            "init_adapter":
                "/root/project/checkpoints/edit-l13/checkpoints/step-00281"
        }))
        try:
            _edit_lineage(manifest, provenance, (6, 7, 8))
        except ValueError as exc:
            assert "outside edit run out_dir" in str(exc)
        else:
            raise AssertionError("foreign init_adapter accepted")
    print("PASS edit-lineage cross-platform")


def test_transfer_emitter():
    """transfer regroups EXISTING tau records by environment: model -> the
    environment label, arms filtered and ordered, every number relayed
    verbatim, provenance kept, and a missing arm refused by name."""
    emit = _load_script("emit_figure_records.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def rec(model, label, tau, path):
            return {"model": model, "label": label, "tau": tau, "tau_ci_low": tau - 0.02,
                    "tau_ci_high": tau + 0.02, "n_scenarios": 305, "rows_path": path}
        nego = [rec("Llama-3.1-8B", "M_0", 0.1049, "/x/m0-rep/rows.jsonl"),
                rec("Llama-3.1-8B", "M_D", 1.0, "/x/md-rep/rows.jsonl"),
                rec("Llama-3.1-8B", "M_E-l08", 0.0, "/x/e1/rows.jsonl"),
                rec("Qwen2.5-7B", "M_0", 0.0, "/x/q/rows.jsonl")]
        insider = [rec("Llama-3.1-8B", "M_0", 0.0, "/x/m0-it/rows.jsonl"),
                   rec("Llama-3.1-8B", "M_D", 0.0, "/x/md-it/rows.jsonl")]
        (tmp / "nego.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nego))
        (tmp / "insider.jsonl").write_text("".join(json.dumps(r) + "\n" for r in insider))
        out = tmp / "transfer.jsonl"
        emit.main(["transfer", "--model", "Llama-3.1-8B", "--arms", "M_0", "M_D",
                   "--tau", "Offer Negotiation=%s" % (tmp / "nego.jsonl"),
                   "--tau", "Insider Trading=%s" % (tmp / "insider.jsonl"),
                   "--out", str(out)])
        records = [json.loads(l) for l in out.read_text().splitlines()]
        assert [(r["model"], r["label"]) for r in records] == [
            ("Offer Negotiation", "M_0"), ("Offer Negotiation", "M_D"),
            ("Insider Trading", "M_0"), ("Insider Trading", "M_D")]
        assert records[0]["tau"] == 0.1049 and records[0]["rows_path"] == "/x/m0-rep/rows.jsonl"
        assert records[0]["tau_ci_low"] == 0.1049 - 0.02
        assert all(r["source_model"] == "Llama-3.1-8B" for r in records)
        assert records[2]["source_record"].endswith("insider.jsonl")
        assert not any(r["label"] == "M_E-l08" for r in records)
        for actual, source in zip(records, nego[:2] + insider):
            assert {k: actual[k] for k in source if k != "model"} == {
                k: v for k, v in source.items() if k != "model"}
        assert [json.loads(l) for l in (tmp / "nego.jsonl").read_text().splitlines()] == nego
        try:
            emit.main(["transfer", "--model", "Llama-3.1-8B", "--arms", "M_0", "M_C",
                       "--tau", "Offer Negotiation=%s" % (tmp / "nego.jsonl"),
                       "--out", str(tmp / "bad.jsonl")])
        except SystemExit as exc:
            assert "M_C" in str(exc)
        else:
            raise AssertionError("a missing arm was silently dropped")
        with (tmp / "nego.jsonl").open("a") as fh:
            fh.write(json.dumps(nego[0]) + "\n")
        try:
            emit.main(["transfer", "--model", "Llama-3.1-8B",
                       "--tau", "Offer Negotiation=%s" % (tmp / "nego.jsonl"),
                       "--out", str(tmp / "duplicate.jsonl")])
        except SystemExit as exc:
            assert "2 records" in str(exc)
        else:
            raise AssertionError("ambiguous transfer input was accepted")
    print("PASS transfer emitter")


def main():
    test_tau_emitter()
    test_transfer_emitter()
    test_layer_curve_emitter_ruling()
    test_recovery_records()
    test_relocation_emit_curves()
    test_edit_lineage_cross_platform()
    print("PASS test_figure_emitters")


if __name__ == "__main__":
    main()
