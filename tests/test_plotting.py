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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    run_suite,
    skip_module_unless_stack,
)

PLOTTING_TEST_COUNT = 13

try:
    import matplotlib

    matplotlib.use("Agg")
    import numpy  # noqa: F401  (matplotlib's own dependency; ML-stack-tier marker)

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

MISSING_STACK = skip_module_unless_stack("matplotlib", "numpy")

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

    def test_tau_bars_give_every_label_its_own_color():
        # Two fixed arms plus four edited checkpoints: six labels, six
        # distinct colors; more labels than spare colors is refused rather
        # than drawn with a shared color.
        labels = ["M_0", "M_D", "M_E-l07", "M_E-l10", "M_E-l13", "M_E-l21"]
        records = [
            {"model": model, "label": label, "tau": 0.1 * (j + 1),
             "tau_ci_low": 0.1 * (j + 1) - 0.05, "tau_ci_high": 0.1 * (j + 1) + 0.05}
            for model in ("Qwen2.5-7B", "Llama-3.1-8B")
            for j, label in enumerate(labels)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_tau_bars(records, os.path.join(tmp, "six"))
            _nonempty(meta["paths"])
        assert meta["labels"] == labels
        assert set(meta["colors"]) == set(labels)
        assert len(set(meta["colors"].values())) == len(labels)
        assert meta["colors"]["M_0"] == plotting.BLUE
        assert meta["colors"]["M_D"] == plotting.ORANGE
        too_many = records + [
            {"model": "Qwen2.5-7B", "label": "M_E-l%02d" % k, "tau": 0.05,
             "tau_ci_low": 0.0, "tau_ci_high": 0.1}
            for k in (2, 4, 6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                plotting.render_tau_bars(too_many, os.path.join(tmp, "seven"))
            except ValueError as exc:
                assert "spare series colors" in str(exc)
            else:
                raise AssertionError("labels beyond the palette shared a color")

    def test_edit_heatmap_render_surfaces_every_cell_state():
        # A voided cell must be grey+ring, never a rate; zero-usable and
        # never-attempted are distinct states; edit windows are drawn only
        # for known rows; "" keeps the title empty for a captioned figure.
        from algoverse.figures import edit_heatmap_cells
        from algoverse.metrics import normalize_rows

        def row(deceptive=False, trunc=False):
            return {"condition": "incentive", "deceptive": deceptive,
                    "hit_max_tokens": trunc, "valid": True}

        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_edit_heatmap(
                plotting.synthetic_edit_heatmap(), os.path.join(tmp, "edit_heatmap")
            )
            _nonempty(meta["paths"])
            assert len(meta["voided"]) == 8, meta["voided"]      # 2 boundary x 4 keys
            assert meta["missing"] == [("l21", 5)], meta["missing"]
            assert ("l07", 0) in meta["zero_clean"] and ("l07", 27) not in meta["zero_clean"]
            assert any("never attempted" in line for line in meta["legend"])
            assert not any("dashed" in line for line in meta["legend"])
            assert meta["edit_windows"] == {}
            assert meta["title"].startswith("Deception under")

            meta = plotting.render_edit_heatmap(
                plotting.synthetic_edit_heatmap(), os.path.join(tmp, "win"),
                edit_windows={"l07": [8, 6, 7]}, title="",
            )
            assert meta["edit_windows"] == {"l07": [6, 8]}, meta["edit_windows"]
            assert any("dashed" in line for line in meta["legend"])
            assert meta["title"] == ""
            try:
                plotting.render_edit_heatmap(
                    plotting.synthetic_edit_heatmap(), os.path.join(tmp, "bad"),
                    edit_windows={"nope": [1]},
                )
            except ValueError as exc:
                assert "nope" in str(exc)
            else:
                raise AssertionError("edit window for an unknown row was accepted")

            columns = [("k", {0: [row(deceptive=True)],
                              1: normalize_rows([row(trunc=True)] * 10)})]
            data = edit_heatmap_cells(columns, n_layers=3)
            meta = plotting.render_edit_heatmap(data, os.path.join(tmp, "tiny"))
            assert ("k", 1) in meta["voided"]
            assert ("k", 1) in meta["zero_clean"]          # 10 truncated, 0 usable
            assert ("k", 0) not in meta["zero_clean"]
            assert ("k", 2) in meta["missing"]
            assert not any("dashed" in line for line in meta["legend"])

    def test_tau_bars_render_with_annotated_gap_for_null_tau():
        records = plotting.synthetic_tau_bars()
        with tempfile.TemporaryDirectory() as tmp:
            meta = plotting.render_tau_bars(records, os.path.join(tmp, "tau_bars"))
            _nonempty(meta["paths"])
        assert meta["models"] == ["Qwen2.5-7B", "Llama-3.1-8B"]
        assert meta["labels"] == ["M_0", "M_D", "M_E"]   # fixed order first, then the rest
        assert meta["gaps"] == [("Llama-3.1-8B", "M_E", "tau_not_computable")]


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=PLOTTING_TEST_COUNT,
                              missing=MISSING_STACK))
