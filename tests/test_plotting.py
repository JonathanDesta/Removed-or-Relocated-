"""Guarded ML-stack-tier tests for the rendering layer (algoverse.plotting).

Needs matplotlib (Agg backend, forced below) + numpy from the
requirements.txt stack — no torch, no GPU, no display. The statistics are
tested in test_figures.py / test_metrics.py; what is tested HERE is that the
renderer (a) writes nonempty .png and .pdf files for every figure, (b) turns
every None the metrics layer can emit into an annotated gap instead of a
zero or a silent drop, and (c) reports the disqualified / unmeasurable /
gap sets in its metadata so captions and tests can check them.

Run: python tests/test_plotting.py with the requirements.txt stack
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PLOTTING_TEST_COUNT = 11

try:
    import matplotlib

    matplotlib.use("Agg")
    import numpy  # noqa: F401  (matplotlib's own dependency; ML-stack-tier marker)

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

    def test_rt_null_points_become_annotated_gaps_not_zeros():
        records = plotting.synthetic_rt()
        # Sanity: the synthetic data contains the null-with-reason case.
        nulls = [r for r in records if r["R_t"] is None]
        assert nulls and nulls[0]["reason"] == "denominator_too_small"
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_rt(records, os.path.join(tmp, "rt"))
            _nonempty(meta["paths"])
        assert ("insider_trading", 8, "denominator_too_small") in meta["gaps"]
        # The pre-registered subset is always on the axis.
        for t in plotting.CHECKPOINT_STEPS:
            assert t in meta["checkpoints_shown"]
        assert set(meta["envs"]) == {"negotiation", "insider_trading"}

    def test_empty_title_is_kept_not_defaulted():
        """A paper figure carries its title in the caption: "" must give NO
        title, while None still gives the renderer's default."""
        with tempfile.TemporaryDirectory() as tmp:
            curves = plotting.synthetic_probe_curves()
            assert plotting.render_probe_curves(curves, os.path.join(tmp, "a"), title="")["title"] == ""
            assert plotting.render_probe_curves(curves, os.path.join(tmp, "b"))["title"].startswith("Instructed-pairs")
            records = plotting.synthetic_rt()
            assert plotting.render_rt(records, os.path.join(tmp, "c"), title="")["title"] == ""
            assert plotting.render_rt(records, os.path.join(tmp, "d"))["title"].startswith("Recovery")
            bars = plotting.synthetic_tau_bars()
            assert plotting.render_tau_bars(bars, os.path.join(tmp, "e"), title="")["title"] == ""

    def test_rt_env_labels_and_constituent_annotation():
        """The annotation spells out the per-arm taus behind R_t at a point,
        labelled with the environment's display name; an unknown (env, t)
        is refused rather than silently skipped."""
        records = plotting.synthetic_rt()
        # With `arms` present the record's own order and spelling are used;
        # without it the canonical E,D / E,C / I,D / I,C order is derived
        # from the tau_* keys.
        assert [a for a, _ in plotting._rt_constituents(records[0])] == ["E,D", "E,C", "I,D", "I,C"]
        bare = {k: v for k, v in records[0].items() if k != "arms"}
        assert [a for a, _ in plotting._rt_constituents(bare)] == ["ED", "EC", "ID", "IC"]
        target = next(r for r in records if r["env"] == "negotiation" and r["R_t"] is not None)
        t_step = target["checkpoint_step"]
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_rt(
                records, os.path.join(tmp, "rt_ann"),
                env_labels={"negotiation": "target layer 7"},
                annotate=[("negotiation", t_step)], notes=["R_t definition"],
                xlabel="checkpoint index",
            )
            _nonempty(meta["paths"])
            assert len(meta["annotations"]) == 1
            env, step, text = meta["annotations"][0]
            assert env == "negotiation" and step == t_step
            assert "target layer 7" in text and "edited model vs intact model" in text
            taus = [v for k, v in target.items() if k.startswith("tau_") and v is not None]
            assert taus and all(("%.3f" % v) in text for v in taus)
            try:
                plotting.render_rt(records, os.path.join(tmp, "rt_bad"), annotate=[("nope", 8)])
            except ValueError as exc:
                assert "nope" in str(exc)
            else:
                raise AssertionError("annotation for a missing record was accepted")

    def test_tau_bars_notes_reach_footnote_meta():
        records = plotting.synthetic_tau_bars()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_tau_bars(
                records, os.path.join(tmp, "tau_notes"),
                notes=["M_0 = m0-baseline-llama8b-rep"],
            )
            _nonempty(meta["paths"])
        assert meta["notes"] == ["M_0 = m0-baseline-llama8b-rep"]

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

    def test_delta_names_gap_sides():
        recovered, edited = plotting.synthetic_delta()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_delta(
                recovered, edited, os.path.join(tmp, "delta"),
                label_recovered="E,D t281 (recovered)",
                label_edited="M_E (just-edited)",
            )
            _nonempty(meta["paths"])
        # Layer 20 is unmeasurable on both source curves: the gap names both
        # sides with the metrics-layer reason.
        gap_layers = dict(meta["gaps"])
        assert 20 in gap_layers
        assert "tau_not_computable" in gap_layers[20]
        assert meta["n_deltas"] == len(meta["layers"]) - len(meta["gaps"])
        # A one-sided gap names the side by its label.
        with tempfile.TemporaryDirectory() as tmp:
            one_sided = plotting.render_delta(
                recovered, [p for p in edited if p["bypassed_layer"] != 3],
                os.path.join(tmp, "delta_one_sided"),
                label_edited="M_E (just-edited)",
            )
        assert dict(one_sided["gaps"])[3] == "M_E (just-edited): absent"

    def test_tau_bars_render_with_annotated_gap_for_null_tau():
        records = plotting.synthetic_tau_bars()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_tau_bars(records, os.path.join(tmp, "tau_bars"))
            _nonempty(meta["paths"])
        assert meta["models"] == ["Qwen2.5-7B", "Llama-3.1-8B"]
        assert meta["labels"] == ["M_0", "M_D", "M_E"]   # fixed order first, then the rest
        assert meta["gaps"] == [("Llama-3.1-8B", "M_E", "tau_not_computable")]


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_plotting.py needs matplotlib + numpy (the requirements.txt "
            "stack). A missing stack is a FAILURE here, not a skip."
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
