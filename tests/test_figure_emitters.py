"""Rung-1 tests for the figure-record emitters and the raw-tau renderer.

The ways these can silently lie, each caught:
  1. the tau emitter recomputing (or mangling) tau instead of relaying
     metrics.tau_with_ci verbatim, or touching its source file;
  2. the layer-curve emitter ignoring --truncated-invalid (a truncated
     deceptive row surviving into A_l);
  3. recovery records rounding taus (%.3f) or dropping a null R_t's reason;
  4. relocation --emit-curves swapping the recovered/lesioned sides;
  5. a null per-arm tau drawn as a value instead of an annotated gap.

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
    print("PASS relocation emit-curves")


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


def main():
    test_tau_emitter()
    test_layer_curve_emitter_ruling()
    test_recovery_records()
    test_relocation_emit_curves()
    test_render_recovery_taus()
    print("PASS test_figure_emitters")


if __name__ == "__main__":
    main()
