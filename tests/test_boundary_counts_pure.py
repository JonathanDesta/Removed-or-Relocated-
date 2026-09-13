"""Rung-1 tests for scripts/boundary_counts_report.py (stdlib only).

    python3 tests/test_boundary_counts_pure.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from algoverse import metrics  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "boundary_counts_report", REPO / "scripts" / "boundary_counts_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(run_id, inc_lied, ctl_lied, n=295, trunc=0):
    rows = []
    for i in range(n):
        rows.append({"run_id": run_id, "scenario_id": "s%d" % i, "condition": "incentive",
                     "valid": True, "deceptive": i < inc_lied, "hit_max_tokens": False})
        rows.append({"run_id": run_id, "scenario_id": "s%d" % i, "condition": "control",
                     "valid": True, "deceptive": i < ctl_lied, "hit_max_tokens": False})
    for i in range(trunc):   # truncated rows are invalid under the ruling
        rows.append({"run_id": run_id, "scenario_id": "t%d" % i, "condition": "incentive",
                     "valid": False, "deceptive": None, "hit_max_tokens": True})
    return rows


def test_counts_wilson_and_truncation():
    m = _load()
    counts = m.count_conditions(_rows("r", 295, 0, trunc=3))
    inc, ctl = counts["incentive"], counts["control"]
    assert (inc["n_deceptive"], inc["n_valid"], inc["n"], inc["n_trunc"]) == (295, 295, 298, 3)
    assert (ctl["n_deceptive"], ctl["n_valid"], ctl["n_trunc"]) == (0, 295, 0)
    lo, hi = metrics.wilson_interval(295, 295)
    assert (inc["low"], inc["high"]) == (lo, hi) and hi == 1.0 and 0.98 < lo < 0.99
    lo, hi = metrics.wilson_interval(0, 295)
    assert (ctl["low"], ctl["high"]) == (lo, hi) and lo == 0.0 and 0.01 < hi < 0.02
    partial = m.count_conditions(_rows("p", 62, 0))["incentive"]
    assert partial["n_deceptive"] == 62 and abs(partial["low"] - 0.168) < 0.002 and abs(partial["high"] - 0.260) < 0.002


def test_grouping_and_report_text():
    m = _load()
    entries = [("a", m.count_conditions(_rows("a", 295, 0))),
               ("b", m.count_conditions(_rows("b", 295, 0))),
               ("c", m.count_conditions(_rows("c", 0, 0)))]
    groups = m.group_signatures(entries)
    assert list(groups.values()) == [["a", "b"], ["c"]]
    text = m.format_report(entries, groups)
    assert "2 run(s): a, b" in text and "1 run(s): c" in text
    assert "295/295" in text and "  0/295" in text and "Wilson" in text
    assert "NOT the scenario-bootstrap" in text


def test_main_glob_and_results_refusal():
    m = _load()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for run, lied in (("e3-ed-t070-l07", 295), ("e3-ec-t070-l07", 0)):
            (tmp / run).mkdir()
            (tmp / run / "rows.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in _rows(run, lied, 0, n=5)))
        out = tmp / "reports" / "counts.txt"
        assert m.main(["--rows", str(tmp / "e3-*-t070-l07" / "rows.jsonl"), "--out", str(out)]) == 0
        text = out.read_text()
        assert "e3-ed-t070-l07" in text and "e3-ec-t070-l07" in text
        try:
            m.main(["--rows", str(tmp / "e3-*-t070-l07" / "rows.jsonl"),
                    "--out", str(tmp / "results" / "x.txt")])
        except SystemExit as exc:
            assert "results/" in str(exc)
        else:
            raise AssertionError("an --out under results/ was accepted")
        try:
            m.main(["--rows", str(tmp / "nothing-*" / "rows.jsonl")])
        except SystemExit as exc:
            assert "no rows file" in str(exc)
        else:
            raise AssertionError("an empty glob was accepted")


def test_legacy_truncation_empty_condition_and_input_immutability():
    m = _load()
    rows = [{"run_id": "legacy", "scenario_id": "x", "condition": "incentive",
             "valid": True, "deceptive": True, "hit_max_tokens": True}]
    original = json.dumps(rows, sort_keys=True)
    counts = m.count_conditions(rows)
    assert counts["incentive"]["n_valid"] == 0
    assert counts["incentive"]["n_deceptive"] == 0
    assert counts["control"]["n"] == 0 and counts["control"]["low"] is None
    assert "n/a" in m.format_report([("legacy", counts)], m.group_signatures([("legacy", counts)]))
    assert json.dumps(rows, sort_keys=True) == original


def test_gate_sources_and_duplicate_inputs():
    m = _load()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gate = {"inputs": {"rows": {}}, "counts": {}}
        for arm in ("M_0", "M_D", "M_E"):
            path = root / "results" / arm / "rows.jsonl"
            path.parent.mkdir(parents=True)
            rows = _rows(arm, 3 if arm == "M_D" else 0, 0, n=3)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            gate["inputs"]["rows"][arm] = "/foreign/results/%s/rows.jsonl" % arm
            gate["counts"][arm] = {
                c: dict(n_deceptive=v["n_deceptive"], n_valid=v["n_valid"], n_total=v["n"])
                for c, v in m.count_conditions(rows).items()}
        gp = root / "gate.json"
        gp.write_text(json.dumps(gate))
        assert len(m.gate_entries(gp, root)) == 3
        gate["counts"]["M_D"]["incentive"]["n_deceptive"] = 0
        gp.write_text(json.dumps(gate))
        try:
            m.gate_entries(gp, root)
        except ValueError as exc:
            assert "gate count mismatch" in str(exc)
        else:
            raise AssertionError("disagreement with archived gate was silently accepted")
        try:
            m.main(["--rows", str(path), "--rows", str(path)])
        except SystemExit as exc:
            assert "duplicate input" in str(exc)
        else:
            raise AssertionError("duplicate input accepted")
        path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
        try:
            m.load_rows_strict(path)
        except ValueError as exc:
            assert "duplicate scenario" in str(exc)
        else:
            raise AssertionError("duplicate scenario accepted")


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS %s" % name)
            except Exception as exc:
                failures += 1; print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc)); traceback.print_exc()
    print("ALL TESTS PASSED" if not failures else "%d FAILURE(S)" % failures)
    raise SystemExit(1 if failures else 0)
