"""Rung-1 tests for the edit heatmap (figures.edit_heatmap_cells + render).

The ways this figure can silently lie, each caught here:
  1. truncated rows leaking into the clean-row D_incentive numerator;
  2. a voided cell (invalid rate above the ruling bound) plotted as a rate;
  3. a missing layer silently dropped instead of reported;
  4. an all-degenerate cell reported as 0.0 deception instead of no-clean-rows;
  5. control-condition rows contaminating an incentive statistic.

Stdlib + matplotlib only (render smoke); no torch, no checkpoint.

    python3 tests/test_edit_heatmap.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.figures import edit_heatmap_cells


def row(condition="incentive", deceptive=False, trunc=False, valid=True):
    return {"condition": condition, "deceptive": deceptive,
            "hit_max_tokens": trunc, "valid": valid}


def cell(data, key, layer):
    return data["cells"][key][layer]


def main():
    # 1 + 5: clean-row statistic excludes truncated rows and control rows.
    rows = ([row(deceptive=True)] * 3 + [row(deceptive=False)] * 7
            + [row(deceptive=True, trunc=True)]              # excluded: truncated
            + [row(condition="control", deceptive=True)] * 5)  # excluded: control
    data = edit_heatmap_cells([("k", {0: rows})], n_layers=1)
    c = cell(data, "k", 0)
    assert c["status"] == "measured", c
    assert c["n"] == 11 and c["n_clean"] == 10, c
    assert abs(c["clean_d_incentive"] - 0.3) < 1e-9, c
    assert abs(c["trunc_rate"] - 1 / 11) < 1e-9, c

    # 2: invalid rate above the bound voids the cell; the rate is preserved
    # in the cell dict but the status forbids plotting it.
    rows = [row(deceptive=True, trunc=True)] * 8 + [row(deceptive=True)] * 2
    data = edit_heatmap_cells([("k", {0: rows})], n_layers=1)
    c = cell(data, "k", 0)
    assert c["status"] == "voided_validity", c
    assert abs(c["invalid_rate"] - 0.8) < 1e-9, c
    assert abs(c["clean_d_incentive"] - 1.0) < 1e-9, c  # 2/2 clean rows lied

    # invalid-but-not-truncated rows also count toward voiding.
    rows = [row(valid=False)] * 5 + [row()] * 5
    data = edit_heatmap_cells([("k", {0: rows})], n_layers=1)
    assert cell(data, "k", 0)["status"] == "voided_validity"

    # exactly at the bound is NOT voided (the ruling says "exceeds").
    rows = [row(trunc=True)] * 2 + [row()] * 8
    data = edit_heatmap_cells([("k", {0: rows})], n_layers=1)
    assert cell(data, "k", 0)["status"] == "measured"

    # 3: a missing layer is reported, not dropped.
    data = edit_heatmap_cells([("k", {0: [row()]})], n_layers=3)
    assert cell(data, "k", 1)["status"] == "missing"
    assert cell(data, "k", 2)["status"] == "missing"

    # 4: all-degenerate cell -> clean_d is None, never 0.0.
    rows = [row(deceptive=True, trunc=True)] * 10
    data = edit_heatmap_cells([("k", {0: rows})], n_layers=1)
    c = cell(data, "k", 0)
    assert c["clean_d_incentive"] is None and c["n_clean"] == 0, c

    # render smoke: files written, voided/missing surfaced in the metadata.
    from algoverse.plotting import render_edit_heatmap, synthetic_edit_heatmap
    with tempfile.TemporaryDirectory() as tmp:
        meta = render_edit_heatmap(synthetic_edit_heatmap(),
                                   str(Path(tmp) / "edit_heatmap"))
        assert all(Path(p).is_file() for p in meta["paths"]), meta["paths"]
        assert len(meta["voided"]) == 8, meta["voided"]      # 2 boundary x 4 keys
        assert meta["missing"] == []

        columns = [("k", {0: [row(deceptive=True)], 1: [row(trunc=True)] * 10})]
        data = edit_heatmap_cells(columns, n_layers=3)
        meta = render_edit_heatmap(data, str(Path(tmp) / "tiny"))
        assert ("k", 1) in meta["voided"]
        assert ("k", 2) in meta["missing"]

    print("PASS test_edit_heatmap")


if __name__ == "__main__":
    main()
