"""Guarded rung-2 tests for the rendering layer (algoverse.plotting).

Needs matplotlib (Agg backend, forced below) + numpy from
~/.venvs/colab-local — no torch, no GPU, no display. The statistics are
tested in test_figures.py / test_metrics.py; what is tested HERE is that the
renderer (a) writes nonempty .png and .pdf files for every figure, (b) turns
every None the metrics layer can emit into an annotated gap instead of a
zero or a silent drop, and (c) reports the disqualified / unmeasurable /
gap sets in its metadata so captions and tests can check them.

Run: ~/.venvs/colab-local/bin/python tests/test_plotting.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PLOTTING_TEST_COUNT = 10

try:
    import matplotlib

    matplotlib.use("Agg")
    import numpy  # noqa: F401  (matplotlib's own dependency; rung-2 marker)

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

from algoverse import plotting  # stdlib-safe import, guarded or not


def _nonempty(paths, expected_suffixes=(".png", ".pdf")):
    assert sorted(os.path.splitext(p)[1] for p in paths) == sorted(expected_suffixes)
    for path in paths:
        assert os.path.exists(path), path
        assert os.path.getsize(path) > 0, path


if HAVE_STACK:

    def test_layer_curve_renders_and_reports_unmeasurable_and_disqualified():
        points, statuses = plotting.synthetic_layer_curve()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_layer_curve(
                points, os.path.join(tmp, "layer_curve"), statuses=statuses
            )
            _nonempty(meta["paths"])
        # The destroyed layer (20) must be in the metadata with its reason,
        # and the disqualified layer (4) must be named.
        assert (20, "tau_not_computable") in meta["unmeasurable"]
        assert 4 in meta["disqualified"]
        # The partial-overlap layer (24) is flagged, not hidden.
        assert any(layer == 24 for layer, _ in meta["flagged"])
        # Unmeasurable never means dropped: every point is accounted for.
        assert meta["n_points"] == len(points)
        assert 20 not in meta["measurable_layers"]

    def test_layer_curve_without_statuses_renders_plain():
        points, _ = plotting.synthetic_layer_curve()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_layer_curve(points, os.path.join(tmp, "plain"))
            _nonempty(meta["paths"])
        assert meta["disqualified"] == []

    def test_layer_curve_survives_json_roundtrip():
        """The CLI path: points serialized to JSON (tuples become lists,
        int dict keys become strings) must still render identically."""
        points, statuses = plotting.synthetic_layer_curve()
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "curve.json")
            with open(src, "w", encoding="utf-8") as fh:
                json.dump(points, fh)
            loaded = plotting.load_records(src)
            statuses_str = {str(k): v for k, v in statuses.items()}
            meta = plotting.render_layer_curve(
                loaded, os.path.join(tmp, "roundtrip"), statuses=statuses_str
            )
            _nonempty(meta["paths"])
        assert (20, "tau_not_computable") in meta["unmeasurable"]
        assert 4 in meta["disqualified"]

    def test_load_records_reads_jsonl_too():
        records = plotting.synthetic_rt()
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "rt.jsonl")
            with open(src, "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            loaded = plotting.load_records(src)
        assert loaded == json.loads(json.dumps(records))

    def test_pareto_renders_frontier_and_hollow_disqualified():
        points, statuses = plotting.synthetic_pareto()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_pareto(
                points, os.path.join(tmp, "pareto"), statuses=statuses
            )
            _nonempty(meta["paths"])
        assert meta["frontier_layers"]          # a frontier exists
        assert 4 in meta["disqualified"]        # hollow, not hidden
        # The destroyed layer cannot be placed on the axes but is reported.
        assert any(layer == 20 for layer, _ in meta["off_plot"])
        assert meta["n_plotted"] < meta["n_points"]

    def test_pareto_json_roundtrip_comparison_key_does_not_crash_frontier():
        """JSON turns the tuple `comparison` key into a list (unhashable);
        the renderer must normalize it before figures.pareto_frontier."""
        points, _ = plotting.synthetic_pareto()
        roundtripped = json.loads(json.dumps(points))
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_pareto(roundtripped, os.path.join(tmp, "rt_pareto"))
            _nonempty(meta["paths"])
        assert meta["frontier_layers"]

    def test_rt_null_points_become_annotated_gaps_not_zeros():
        records = plotting.synthetic_rt()
        # Sanity: the synthetic data contains the null-with-reason case.
        nulls = [r for r in records if r["R_t"] is None]
        assert nulls and nulls[0]["reason"] == "denominator_too_small"
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_rt(records, os.path.join(tmp, "rt"))
            _nonempty(meta["paths"])
        assert ("insider_trading", 8, "denominator_too_small") in meta["gaps"]
        # The ratified subset is always on the axis.
        for t in plotting.CHECKPOINT_STEPS:
            assert t in meta["checkpoints_shown"]
        assert set(meta["envs"]) == {"negotiation", "insider_trading"}

    def test_rt_all_null_environment_still_renders():
        """Every point null (e.g. the intact gap never exceeded eps): the
        figure must still render, all gaps annotated, nothing plotted as 0."""
        records = [
            {"env": "negotiation", "checkpoint_step": t, "R_t": None,
             "R_t_ci_low": None, "R_t_ci_high": None,
             "reason": "denominator_too_small"}
            for t in plotting.CHECKPOINT_STEPS
        ]
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_rt(records, os.path.join(tmp, "rt_null"))
            _nonempty(meta["paths"])
        assert len(meta["gaps"]) == len(plotting.CHECKPOINT_STEPS)

    def test_delta_marks_lesioned_layer_and_names_gap_sides():
        recovered, lesioned, l_star = plotting.synthetic_delta()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_delta(
                recovered, lesioned, os.path.join(tmp, "delta"),
                lesioned_layer=l_star,
            )
            _nonempty(meta["paths"])
        assert meta["lesioned_layer"] == l_star
        # Layer 20 is unmeasurable on both source curves: the gap names both
        # sides with the metrics-layer reason.
        gap_layers = dict(meta["gaps"])
        assert 20 in gap_layers
        assert "tau_not_computable" in gap_layers[20]
        assert meta["n_deltas"] == len(meta["layers"]) - len(meta["gaps"])

    def test_tau_bars_render_with_annotated_gap_for_null_tau():
        records = plotting.synthetic_tau_bars()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_tau_bars(records, os.path.join(tmp, "tau_bars"))
            _nonempty(meta["paths"])
        assert meta["labels"] == ["M_0", "M_D", "M_C"]   # fixed arm order
        assert ("Gemma-2-9B", "M_C", "tau_not_computable") in meta["gaps"]


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_plotting.py needs matplotlib + numpy "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == PLOTTING_TEST_COUNT, (
        "expected %d tests, found %d" % (PLOTTING_TEST_COUNT, len(tests))
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except Exception:
            failures += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    if failures:
        sys.exit("%d test(s) failed" % failures)
    print("ALL TESTS PASSED")
