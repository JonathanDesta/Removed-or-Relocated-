"""Rung-1 tests for scripts/probe_surface_baselines.py (stdlib only).

    python3 tests/test_probe_surface_baselines_pure.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from algoverse.corroboration import labeled_incentive_rows  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "probe_surface_baselines", REPO / "scripts" / "probe_surface_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(i, lied, claimed, text, offer=100000, true_value=80000, condition="incentive",
         valid=True):
    return {
        "run_id": "r", "scenario_id": "s%02d" % i, "condition": condition,
        "valid": valid, "deceptive": (lied if valid else None),
        "claimed_value": claimed, "true_value": true_value,
        "response_text": text,
        "scenario_params": {"company_offer": offer, "true_outside_offer": true_value},
    }


def _fixture():
    rows = []
    # Offer scenarios: lies state a HIGHER number and write SHORTER text.
    for i in range(4):
        rows.append(_row(i, True, 120000 + 1000 * i, "x" * (20 + i)))
    for i in range(4, 8):
        rows.append(_row(i, False, 80000, "x" * (60 + i)))
    # No-offer scenarios: a fabricated number vs honest NONE.
    rows.append(_row(8, True, 95000.0, "y" * 30, true_value=None))
    rows.append(_row(9, False, "NONE", "y" * 70, true_value=None))
    # Excluded rows: control condition, invalid, empty text.
    rows.append(_row(10, False, 80000, "z" * 40, condition="control"))
    rows.append(_row(11, False, None, "", valid=False))
    rows.append(_row(12, False, 80000, ""))
    return rows


def test_population_matches_corroboration():
    m = _load()
    rows = _fixture()
    selected = m.select_rows(rows)
    labeled = labeled_incentive_rows(rows)
    assert [r["scenario_id"] for r in selected] == [r["scenario_id"] for r in labeled]
    assert len(selected) == 10
    print("PASS test_population_matches_corroboration")


def test_auroc_orderings_and_ties():
    m = _load()
    assert m.auroc([3, 4, 1, 2], [1, 1, 0, 0]) == 1.0
    assert m.auroc([1, 2, 3, 4], [1, 1, 0, 0]) == 0.0
    assert m.auroc([5, 5, 5, 5], [1, 1, 0, 0]) == 0.5
    assert m.auroc([2, 1, 1, 0], [1, 1, 0, 0]) == 0.875
    assert m.auroc([1, 2], [1, 1]) is None
    print("PASS test_auroc_orderings_and_ties")


def test_features_and_none_handling():
    m = _load()
    assert m.stated_number(_row(0, False, "NONE", "t")) == 0.0
    assert m.stated_number(_row(0, False, None, "t")) == 0.0
    assert m.stated_number(_row(0, True, 120000, "t")) == 120000.0
    row = _row(0, True, 125000, "abcde", offer=100000)
    assert m.feature_value(row, "char_length") == 5.0
    assert abs(m.feature_value(row, "stated_over_offer") - 1.25) < 1e-12
    try:
        m.feature_value(dict(row, scenario_params={}), "stated_over_offer")
    except ValueError as exc:
        assert "company_offer" in str(exc)
    else:
        raise AssertionError("missing company_offer must raise")
    print("PASS test_features_and_none_handling")


def test_evaluate_strata_and_values():
    m = _load()
    records = m.evaluate(_fixture(), n_boot=50, seed=0)
    by = {(r["analysis"], r["stratum"]): r for r in records}
    assert len(records) == 6
    offer_num = by[("surface_auroc:stated_number", "offer")]
    assert (offer_num["n"], offer_num["n_lied"]) == (8, 4)
    assert offer_num["value"] == 1.0
    assert offer_num["ci_low"] is not None and offer_num["ci_high"] is not None
    assert by[("surface_auroc:char_length", "offer")]["value"] == 0.0
    assert by[("surface_auroc:stated_over_offer", "offer")]["value"] == 1.0
    all_num = by[("surface_auroc:stated_number", "all")]
    assert (all_num["n"], all_num["n_lied"]) == (10, 5)
    assert all_num["value"] == 1.0            # fabricated 95k still beats every honest 80k / NONE
    # A single-class stratum reports n/a rather than a number.
    single = [r for r in _fixture() if r.get("deceptive") is False]
    assert all(rec["value"] is None for rec in m.evaluate(single, n_boot=10))
    print("PASS test_evaluate_strata_and_values")


def test_main_writes_records_and_refuses_results():
    m = _load()
    with tempfile.TemporaryDirectory() as tmp:
        rows_path = Path(tmp) / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in _fixture()))
        out = Path(tmp) / "reports" / "surface.jsonl"
        m.main(["--rows", str(rows_path), "--out", str(out), "--n-boot", "20"])
        written = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(written) == 6
        assert all(w["rows_path"] == str(rows_path) for w in written)
        assert m.format_table(written).count("\n") == 6
        bad = Path.cwd() / "results" / "surface.jsonl"
        try:
            m.main(["--rows", str(rows_path), "--out", str(bad)])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("--out under results/ must be refused")
    print("PASS test_main_writes_records_and_refuses_results")


if __name__ == "__main__":
    test_population_matches_corroboration()
    test_auroc_orderings_and_ties()
    test_features_and_none_handling()
    test_evaluate_strata_and_values()
    test_main_writes_records_and_refuses_results()
    print("ALL TESTS PASSED")
