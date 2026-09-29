"""Dependency-free tests for the edit heatmap cells (figures.edit_heatmap_cells).

The ways this figure can silently lie, each caught here:
  1. truncated rows (invalid under the truncation rule) leaking into the
     clean-row D_incentive numerator;
  2. a voided cell (invalid rate above the bound) plotted as a rate;
  3. a missing layer silently dropped instead of reported;
  4. an all-degenerate cell reported as 0.0 deception instead of no-clean-rows;
  5. control-condition rows contaminating an incentive statistic.

The render smoke (every cell state surfaced in the figure metadata) lives
in tests/test_plotting.py, which needs matplotlib.

    python3 tests/test_edit_heatmap_pure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import run_suite  # noqa: E402
from algoverse.figures import edit_heatmap_cells  # noqa: E402
from algoverse.metrics import normalize_rows  # noqa: E402


def row(condition="incentive", deceptive=False, trunc=False, valid=True):
    return {"condition": condition, "deceptive": deceptive,
            "hit_max_tokens": trunc, "valid": valid}


def cell(data, key, layer):
    return data["cells"][key][layer]


def test_clean_statistic_excludes_truncated_and_control_rows():
    rows = ([row(deceptive=True)] * 3 + [row(deceptive=False)] * 7
            + [row(deceptive=True, trunc=True)]              # excluded: truncated
            + [row(condition="control", deceptive=True)] * 5)  # excluded: control
    data = edit_heatmap_cells([("k", {0: normalize_rows(rows)})], n_layers=1)
    c = cell(data, "k", 0)
    assert c["status"] == "measured", c
    assert c["n"] == 11 and c["n_clean"] == 10, c
    assert abs(c["clean_d_incentive"] - 0.3) < 1e-9, c
    assert abs(c["trunc_rate"] - 1 / 11) < 1e-9, c


def test_invalid_rate_above_the_bound_voids_the_cell():
    # The rate is preserved in the cell dict but the status forbids
    # plotting it.
    rows = [row(deceptive=True, trunc=True)] * 8 + [row(deceptive=True)] * 2
    data = edit_heatmap_cells([("k", {0: normalize_rows(rows)})], n_layers=1)
    c = cell(data, "k", 0)
    assert c["status"] == "voided_validity", c
    assert abs(c["invalid_rate"] - 0.8) < 1e-9, c
    assert abs(c["clean_d_incentive"] - 1.0) < 1e-9, c  # 2/2 clean rows lied

    # invalid-but-not-truncated rows also count toward voiding.
    rows = [row(valid=False)] * 5 + [row()] * 5
    data = edit_heatmap_cells([("k", {0: normalize_rows(rows)})], n_layers=1)
    assert cell(data, "k", 0)["status"] == "voided_validity"

    # exactly at the bound is NOT voided (the rule says "exceeds").
    rows = [row(trunc=True)] * 2 + [row()] * 8
    data = edit_heatmap_cells([("k", {0: normalize_rows(rows)})], n_layers=1)
    assert cell(data, "k", 0)["status"] == "measured"


def test_missing_layer_is_reported_not_dropped():
    data = edit_heatmap_cells([("k", {0: [row()]})], n_layers=3)
    assert cell(data, "k", 1)["status"] == "missing"
    assert cell(data, "k", 2)["status"] == "missing"


def test_all_degenerate_cell_has_no_rate():
    rows = [row(deceptive=True, trunc=True)] * 10
    data = edit_heatmap_cells([("k", {0: normalize_rows(rows)})], n_layers=1)
    c = cell(data, "k", 0)
    assert c["clean_d_incentive"] is None and c["n_clean"] == 0, c


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=4))
