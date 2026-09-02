"""
Rendering layer for the paper's figures.

figures.py is deliberately matplotlib-free: it turns results rows into
plot-ready point dicts. This module is the OTHER half — it takes those dicts
and draws publication-ready charts. It adds no statistics: every number on a
figure was computed by metrics.py / figures.py / sweep.py, and anything those
layers report as None is rendered as an ANNOTATED GAP, never as a zero and
never silently dropped.

Import safety: module-level imports are stdlib + algoverse.figures (itself
stdlib-safe). matplotlib is imported lazily inside the render functions with
the Agg backend forced, so `import algoverse.plotting` works on a laptop with
no ML or plotting stack, and rendering never needs a display.

The five figures (one render_* function each; scripts/make_figures.py is the
CLI around them):

  render_layer_curve   A_l vs bypassed layer, CI band, disqualified layers
                       shaded, unmeasurable layers marked at the axis with
                       their reason.
  render_pareto        A_l vs damage scatter with the non-dominated frontier
                       (figures.pareto_frontier); disqualified points hollow;
                       points missing either axis listed in a footnote.
  render_rt            R_t vs checkpoint t (the ratified subset {8, 70, 281},
                       RESEARCH_SPEC "T10 R_t subset"), one line per
                       environment; null R_t (metrics.recovery returning
                       None with a reason) is an annotated gap.
  render_delta         the δ-curve: A_l(recovered checkpoint) minus
                       A_l(just-lesioned), per layer, with the permanently
                       lesioned layer l* marked.
  render_tau_bars      tau(M_0) / tau(M_D) / tau(M_C) per model with CIs,
                       from metrics.tau_with_ci-shaped dicts.

Every render function writes <out_base>.png (300 dpi) and <out_base>.pdf and
returns a metadata dict that includes "paths" plus everything a test (or a
reader of the caption) needs to confirm nothing was dropped: the disqualified
layers, the unmeasurable/flagged layers with reasons, the annotated gaps.

Color: the categorical slots below are a color-blind-safe ordering validated
with the dataviz palette checker (adjacent-pair CVD ΔE ≥ 8, normal-vision
ΔE ≥ 15, light surface). Aqua and yellow sit below 3:1 contrast on white, so
the figures that use them carry direct labels. Identity is never color-alone:
series also differ by marker shape.

Synthetic data: each figure has a synthetic_* generator producing plausible
fake inputs of exactly the shapes the render functions (and the real
pipeline) use — including an unmeasurable layer, a disqualified layer, and a
null R_t — so the whole rendering path is dry-runnable with no real results.
"""

import json
import math
import os
import random

from algoverse import figures, metrics

# The ratified R_t evaluation subset (RESEARCH_SPEC "Ratified decisions
# 2026-08-16": T10 resolution — early/mid/final of [8, 17, 35, 70, 140, 281]).
CHECKPOINT_STEPS = (8, 70, 281)

# Color-blind-safe categorical slots, in fixed order (never cycled past what
# is listed). First three validate all-pairs; the full five validate adjacent
# pairs. Validated 2026-08-16 with the dataviz palette checker.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, RED)
MARKERS = ("o", "s", "^", "D", "v")

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SHADE = "#f0efec"          # disqualified-layer band
GAP_COLOR = "#52514e"      # annotated-gap marks: neutral ink, not a series hue

# tau bars: fixed color per arm label, stable regardless of which models or
# arms happen to be present (color follows the entity, never its rank).
ARM_COLORS = {"M_0": BLUE, "M_D": ORANGE, "M_C": AQUA}
ARM_ORDER = ("M_0", "M_D", "M_C")

_STYLE = {
    "figure.figsize": (6.4, 4.0),
    "figure.constrained_layout.use": True,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.linewidth": 0.8,
    "axes.labelcolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": TEXT,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
}


def _plt():
    """Lazy matplotlib import: Agg backend, house style. Never at module level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)
    return plt


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_records(path):
    """One JSON document (list or dict) or a JSONL file -> python object.

    A .jsonl of records comes back as a list of dicts; a .json list or dict
    comes back as itself. Detection is by content, not extension.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError("%s line %d is not valid JSON" % (path, lineno))
        return records


def _normalize_comparison(points):
    """JSON round-trips figures' tuple `comparison` keys into lists, which are
    unhashable and would crash pareto_frontier's mixed-comparison guard. Make
    them tuples again (recursively, for the nested gen-identity tuple)."""

    def as_tuple(value):
        if isinstance(value, list):
            return tuple(as_tuple(v) for v in value)
        return value

    for p in points:
        if isinstance(p.get("comparison"), list):
            p["comparison"] = as_tuple(p["comparison"])
    return points


def _save(fig, out_base, dpi=300, formats=("png", "pdf")):
    out_dir = os.path.dirname(os.path.abspath(out_base))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = "%s.%s" % (out_base, fmt)
        fig.savefig(path, dpi=dpi, format=fmt, bbox_inches="tight")
        paths.append(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _positions(layers):
    """x positions for layer values.

    All-numeric layers plot at their numeric value (so a missing layer shows
    as a real hole in x). Any non-numeric layer falls back to ordinal
    positions with every layer labeled.
    """
    try:
        xs = [float(int(v)) for v in layers]
        return xs, None
    except (TypeError, ValueError):
        xs = [float(i) for i in range(len(layers))]
        return xs, [str(v) for v in layers]


def _contiguous_runs(indices):
    """Split a sorted index list into runs of consecutive indices."""
    runs = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _is_disqualified(status):
    return bool(status) and str(status).startswith("DISQUALIFIED")


def _gap_marks(ax, gaps, color=GAP_COLOR, max_chars=38):
    """Draw annotated gaps: an x at the axis floor plus the reason, rotated.

    `gaps` is [(x, label)]. Drawn in x-data / y-axes coordinates so they stay
    glued to the bottom regardless of the y range. Long reasons are truncated
    on the figure (the metadata carries them in full) so a verbose reason
    cannot overflow the axes and distort the saved bounding box.
    """
    trans = ax.get_xaxis_transform()
    for x, label in gaps:
        ax.plot(
            [x], [0.035], transform=trans, marker="x", markersize=7,
            markeredgewidth=1.6, color=color, linestyle="none", clip_on=False,
            zorder=5,
        )
    for x, label in gaps:
        text = str(label)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        ax.annotate(
            text, xy=(x, 0.07), xycoords=trans, rotation=90,
            ha="center", va="bottom", fontsize=7, color=color, zorder=5,
        )


def _footnote(fig, lines):
    """Caption-adjacent notes below the axes; kept by the tight save bbox."""
    if lines:
        fig.text(
            0.01, -0.02, "\n".join(lines), ha="left", va="top",
            fontsize=7, color=TEXT_SECONDARY,
        )


# ---------------------------------------------------------------------------
# Figure 1: the layer curve
# ---------------------------------------------------------------------------


def render_layer_curve(points, out_base, statuses=None, title=None, dpi=300):
    """A_l vs bypassed layer.

    points    figures.layer_curve output: dicts with bypassed_layer, A_l,
              A_l_ci_low, A_l_ci_high, reason, paired, competence,
              invalid_rate_incentive/control (extra keys ignored).
    statuses  optional {layer: status string} from sweep.evaluate_sweep
              entries ("VIABLE" / "DISQUALIFIED: ..." / "UNMEASURABLE: ..." /
              "NO ROWS"); disqualified layers get a shaded band and hollow
              markers. Without statuses the curve is plain.

    Unmeasurable layers (A_l None) are marked at the axis floor with their
    reason — never dropped. Partial-overlap points (paired False but A_l
    present) are drawn hollow and footnoted.
    """
    statuses = statuses or {}
    plt = _plt()
    fig, ax = plt.subplots()

    layers = [p.get("bypassed_layer") for p in points]
    xs, tick_labels = _positions(layers)

    def status_of(p):
        layer = p.get("bypassed_layer")
        for key in (layer, str(layer)):
            if key in statuses:
                return statuses[key]
        try:
            return statuses.get(int(layer))
        except (TypeError, ValueError):
            return None

    measurable = [i for i, p in enumerate(points) if p.get("A_l") is not None]
    disqualified_layers = []
    flagged = []  # (layer, reason) for partial overlap etc.
    gaps = []

    # CI band + line per contiguous measurable run.
    for run in _contiguous_runs(measurable):
        run_x = [xs[i] for i in run]
        run_y = [points[i]["A_l"] for i in run]
        with_ci = [
            i for i in run
            if points[i].get("A_l_ci_low") is not None
            and points[i].get("A_l_ci_high") is not None
        ]
        for ci_run in _contiguous_runs(with_ci):
            ax.fill_between(
                [xs[i] for i in ci_run],
                [points[i]["A_l_ci_low"] for i in ci_run],
                [points[i]["A_l_ci_high"] for i in ci_run],
                color=BLUE, alpha=0.18, linewidth=0, zorder=1,
            )
        ax.plot(run_x, run_y, color=BLUE, zorder=3)

    # Markers, one by one so disqualified/unpaired points can be hollow.
    half = 0.45 if tick_labels is None else 0.45
    for i, p in enumerate(points):
        status = status_of(p)
        disq = _is_disqualified(status)
        if disq:
            disqualified_layers.append(p.get("bypassed_layer"))
            ax.axvspan(xs[i] - half, xs[i] + half, color=SHADE, zorder=0)
        if p.get("A_l") is None:
            reason = p.get("reason") or (status or "A_l_none")
            gaps.append((xs[i], str(reason)))
            continue
        unpaired = not p.get("paired", True)
        if unpaired:
            flagged.append((p.get("bypassed_layer"), p.get("reason") or "partial_overlap"))
        hollow = disq or unpaired
        ax.plot(
            [xs[i]], [p["A_l"]], marker="o", linestyle="none",
            markerfacecolor="white" if hollow else BLUE,
            markeredgecolor=BLUE, markeredgewidth=1.4, zorder=4,
        )

    _gap_marks(ax, gaps)
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)

    if tick_labels is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("bypassed layer $l$")
    ax.set_ylabel(r"$A_l = \tau(\mathrm{base}) - \tau(\mathrm{bypassed})$")
    ax.set_title(title or "Deception-specific bypass effect by layer")

    notes = []
    if disqualified_layers:
        notes.append(
            "shaded/hollow: disqualified layers %s"
            % ", ".join(str(l) for l in disqualified_layers)
        )
    if gaps:
        notes.append("x at axis: unmeasurable layers (reason shown), not zero")
    for layer, reason in flagged:
        notes.append("hollow, layer %s: %s" % (layer, reason))
    _footnote(fig, notes)

    unmeasurable = figures.unmeasurable(points)
    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "n_points": len(points),
        "measurable_layers": [points[i].get("bypassed_layer") for i in measurable],
        "unmeasurable": unmeasurable,
        "disqualified": disqualified_layers,
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Figure 2: the Pareto
# ---------------------------------------------------------------------------


def _draw_pareto_axes(ax, points, statuses=None, frontier=None,
                      allow_mixed=False, bounds=None):
    """Draw one Pareto panel onto `ax`; returns its bookkeeping dict.

    Shared by render_pareto (one figure) and render_pareto_panels (one
    subplot per damage metric). bounds = {"a_l_min": x, "damage_max": y}
    (either may be None) draws the ratified thresholds as dashed lines, so
    the admissible region -- A_l at least x AND damage at most y -- is
    visible and the layers inside it are reported in "within_bounds".
    """
    statuses = statuses or {}
    bounds = dict(bounds or {})
    _normalize_comparison(points)
    if frontier is None:
        frontier = figures.pareto_frontier(points, allow_mixed=allow_mixed)
    frontier_layers = [p.get("bypassed_layer") for p in frontier]

    def status_of(layer):
        for key in (layer, str(layer)):
            if key in statuses:
                return statuses[key]
        try:
            return statuses.get(int(layer))
        except (TypeError, ValueError):
            return None

    plottable = [
        p for p in points if p.get("A_l") is not None and p.get("damage") is not None
    ]
    off_plot = [
        (p.get("bypassed_layer"),
         p.get("damage_reason") or p.get("reason") or "unmeasurable")
        for p in points
        if p.get("A_l") is None or p.get("damage") is None
    ]
    disqualified_layers = []

    # Frontier first (under the dots): sorted by A_l descending already.
    if frontier:
        fx = [p["A_l"] for p in frontier]
        fy = [p["damage"] for p in frontier]
        order = sorted(range(len(fx)), key=lambda i: fx[i])
        ax.plot(
            [fx[i] for i in order], [fy[i] for i in order],
            color=ORANGE, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2,
            label="Pareto frontier",
        )

    frontier_set = set(str(l) for l in frontier_layers)
    for p in plottable:
        layer = p.get("bypassed_layer")
        disq = _is_disqualified(status_of(layer))
        if disq:
            disqualified_layers.append(layer)
        on_frontier = str(layer) in frontier_set
        ax.plot(
            [p["A_l"]], [p["damage"]], linestyle="none", marker="o",
            markersize=7 if on_frontier else 6,
            markerfacecolor="white" if disq else BLUE,
            markeredgecolor=ORANGE if on_frontier else BLUE,
            markeredgewidth=1.6 if on_frontier else 1.2,
            zorder=4 if on_frontier else 3,
        )
        ax.annotate(
            str(layer), xy=(p["A_l"], p["damage"]), xytext=(4, 4),
            textcoords="offset points", fontsize=7.5, color=TEXT,
        )

    a_l_min = bounds.get("a_l_min")
    damage_max = bounds.get("damage_max")
    if a_l_min is not None:
        ax.axvline(float(a_l_min), color=TEXT_SECONDARY, linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            r"$A_l \geq %.2f$" % a_l_min, xy=(float(a_l_min), 0.98),
            xycoords=("data", "axes fraction"), xytext=(3, 0),
            textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY,
            va="top",
        )
    if damage_max is not None:
        ax.axhline(float(damage_max), color=TEXT_SECONDARY, linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            r"damage $\leq$ %.2f" % damage_max, xy=(0.99, float(damage_max)),
            xycoords=("axes fraction", "data"), xytext=(0, 3),
            textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY,
            ha="right", va="bottom",
        )
    within = []
    if a_l_min is not None or damage_max is not None:
        within = [
            p.get("bypassed_layer") for p in plottable
            if (a_l_min is None or p["A_l"] >= a_l_min)
            and (damage_max is None or p["damage"] <= damage_max)
        ]

    damage_metric = points[0].get("damage_metric") if points else None
    ax.set_xlabel(r"$A_l$ (deception removed, higher is better)")
    ax.set_ylabel(
        "damage%s (lower is better)"
        % (" [%s]" % damage_metric if damage_metric else "")
    )
    return {
        "damage_metric": damage_metric,
        "n_points": len(points),
        "n_plotted": len(plottable),
        "frontier": frontier,
        "frontier_layers": frontier_layers,
        "disqualified": disqualified_layers,
        "off_plot": off_plot,
        "bounds": {"a_l_min": a_l_min, "damage_max": damage_max},
        "within_bounds": within,
    }


def render_pareto(points, out_base, statuses=None, frontier=None,
                  allow_mixed=False, title=None, dpi=300, bounds=None):
    """A_l (x, more deception removed) vs damage (y, lower is better).

    points    figures.pareto_points output: layer-curve dicts plus damage,
              damage_metric, damage_reason.
    statuses  optional {layer: status}; disqualified points drawn hollow.
    frontier  optional precomputed figures.pareto_frontier list; computed
              here otherwise (allow_mixed passed through).
    bounds    optional {"a_l_min", "damage_max"}: the ratified thresholds,
              drawn as dashed lines; layers inside both are reported.

    Points missing A_l or damage cannot be placed on the axes; they are
    listed in the footnote and returned in metadata, never silently dropped.
    """
    plt = _plt()
    fig, ax = plt.subplots()
    info = _draw_pareto_axes(
        ax, points, statuses=statuses, frontier=frontier,
        allow_mixed=allow_mixed, bounds=bounds,
    )
    ax.set_title(title or "Deception removed vs capability damage")
    if info["frontier"]:
        ax.legend(loc="best")

    notes = []
    if info["disqualified"]:
        notes.append(
            "hollow: disqualified layers %s"
            % ", ".join(str(l) for l in info["disqualified"])
        )
    for layer, reason in info["off_plot"]:
        notes.append("not plottable, layer %s: %s" % (layer, reason))
    if info["bounds"]["a_l_min"] is not None or info["bounds"]["damage_max"] is not None:
        notes.append(
            "dashed: ratified bounds; layers within both: %s"
            % (", ".join(str(l) for l in info["within_bounds"]) or "none")
        )
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "n_points": info["n_points"],
        "n_plotted": info["n_plotted"],
        "frontier_layers": info["frontier_layers"],
        "disqualified": info["disqualified"],
        "off_plot": info["off_plot"],
        "bounds": info["bounds"],
        "within_bounds": info["within_bounds"],
    }


def _compact_layers(layers):
    """[0,1,2,5,8,9] -> "0-2, 5, 8-9" (footnotes must not widen the figure)."""
    ints = sorted({int(l) for l in layers if str(l).lstrip("-").isdigit()})
    others = sorted(str(l) for l in layers if not str(l).lstrip("-").isdigit())
    parts = []
    for run in _contiguous_runs(ints):
        parts.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
    return ", ".join(parts + others)


def _off_plot_note(metric, off_plot):
    """One compact line per panel: reasons grouped, layers as ranges."""
    by_reason = {}
    for layer, reason in off_plot:
        by_reason.setdefault(str(reason), []).append(layer)
    return "%s: not plottable -- %s" % (metric, "; ".join(
        "%s: layers %s" % (reason, _compact_layers(layers))
        for reason, layers in by_reason.items()
    ))


def render_pareto_panels(record, out_base, title=None, dpi=300):
    """One Pareto subplot per damage metric, from an emit_figure_records.py
    `pareto` record: {"model", "base_run_id", "panels": [{"damage_metric",
    "damage_reference", "bounds", "pareto_points", "frontier"}, ...]}.

    Every panel is drawn by _draw_pareto_axes with its own ratified bounds
    and precomputed frontier; off-plot layers are footnoted per panel, never
    dropped. Metadata lists, per panel, what was plotted and which layers
    sit inside both bounds.
    """
    panels = list(record.get("panels") or [])
    if not panels:
        raise ValueError("pareto panels record has no panels")
    plt = _plt()
    n = len(panels)
    ncols = min(3, n)
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), squeeze=False,
    )
    meta_panels, notes = [], []
    for i, panel in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        points = [dict(p) for p in panel.get("pareto_points") or []]
        frontier = panel.get("frontier")
        if frontier is not None:
            frontier = _normalize_comparison([dict(p) for p in frontier])
        info = _draw_pareto_axes(
            ax, points, frontier=frontier, allow_mixed=True,
            bounds=panel.get("bounds"),
        )
        metric = panel.get("damage_metric") or info["damage_metric"]
        reference = panel.get("damage_reference")
        ax.set_title(
            "%s%s" % (metric, " (vs %s)" % reference if reference else ""),
            fontsize=9.5,
        )
        ax.set_ylabel("damage (lower is better)")
        if info["frontier"]:
            ax.legend(loc="best", fontsize=7)
        if info["off_plot"]:
            notes.append(_off_plot_note(metric, info["off_plot"]))
        meta_panels.append({
            "damage_metric": metric,
            "damage_reference": reference,
            "n_points": info["n_points"],
            "n_plotted": info["n_plotted"],
            "frontier_layers": info["frontier_layers"],
            "off_plot": info["off_plot"],
            "bounds": info["bounds"],
            "within_bounds": info["within_bounds"],
        })
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    model = record.get("model")
    fig.suptitle(
        title or "Deception removed vs capability damage, per metric%s"
        % (" - %s" % model if model else "")
    )
    notes.append(
        "dashed lines: ratified bounds (A_l minimum; per-metric damage cap); "
        "hollow: disqualified"
    )
    _footnote(fig, notes)
    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "model": model,
        "panels": meta_panels,
    }


# ---------------------------------------------------------------------------
# Figure 3: recovery curves
# ---------------------------------------------------------------------------


def render_rt(records, out_base, checkpoints=CHECKPOINT_STEPS, title=None,
              dpi=300):
    """R_t vs fine-tuning checkpoint t, one line per environment.

    records   one dict per (environment, checkpoint): the metrics.recovery()
              output (R_t, R_t_ci_low, R_t_ci_high, reason, tau_* extras
              ignored) plus "env" (line label) and "checkpoint_step" (int).

    Null R_t (recovery returned None with a reason, e.g.
    denominator_too_small) is an ANNOTATED GAP at that t: an x at the axis
    floor with the reason, and the line broken — never a zero. The
    pre-committed checkpoints appear as x ticks even if some have no data.
    """
    plt = _plt()
    fig, ax = plt.subplots()

    envs = []
    for r in records:
        env = r.get("env")
        if env not in envs:
            envs.append(env)

    gaps = []          # (env, t, reason)
    gap_marks = []     # (x, label)
    for idx, env in enumerate(envs):
        color = SERIES[idx % len(SERIES)]
        marker = MARKERS[idx % len(MARKERS)]
        env_records = sorted(
            (r for r in records if r.get("env") == env),
            key=lambda r: r.get("checkpoint_step"),
        )
        measurable = [i for i, r in enumerate(env_records) if r.get("R_t") is not None]
        for run_idx, run in enumerate(_contiguous_runs(measurable)):
            run_records = [env_records[i] for i in run]
            ts = [r["checkpoint_step"] for r in run_records]
            ys = [r["R_t"] for r in run_records]
            lo = [
                (r["R_t"] - r["R_t_ci_low"]) if r.get("R_t_ci_low") is not None else 0.0
                for r in run_records
            ]
            hi = [
                (r["R_t_ci_high"] - r["R_t"]) if r.get("R_t_ci_high") is not None else 0.0
                for r in run_records
            ]
            ax.errorbar(
                ts, ys, yerr=[lo, hi], color=color, marker=marker,
                markeredgecolor="white", markeredgewidth=0.8,
                capsize=2.5, elinewidth=1.0,
                label=str(env) if run_idx == 0 else None,
                zorder=3,
            )
        # Direct label at the last measurable point (relief for low-contrast hues).
        if measurable:
            last = env_records[measurable[-1]]
            ax.annotate(
                str(env), xy=(last["checkpoint_step"], last["R_t"]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=TEXT, va="center",
            )
        for i, r in enumerate(env_records):
            if r.get("R_t") is None:
                reason = r.get("reason") or "R_t_none"
                gaps.append((env, r.get("checkpoint_step"), reason))
                gap_marks.append(
                    (r.get("checkpoint_step"), "%s: %s" % (env, reason))
                )

    _gap_marks(ax, gap_marks)

    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "full recovery (R=1)", xy=(0.0, 1.0), xycoords=("axes fraction", "data"),
        xytext=(4, 4), textcoords="offset points", ha="left",
        fontsize=7.5, color=TEXT_SECONDARY,
    )
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)

    ticks = sorted(
        set(checkpoints)
        | set(r.get("checkpoint_step") for r in records
              if r.get("checkpoint_step") is not None)
    )
    ax.set_xticks(ticks)
    ax.set_xlabel("fine-tuning checkpoint $t$ (steps)")
    ax.set_ylabel(r"$R_t$ (fraction of deception gap recovered)")
    ax.set_title(title or "Recovery of the deception gap after re-fine-tuning")
    if len(envs) > 1:
        ax.legend(loc="best")

    notes = []
    if gaps:
        notes.append("x at axis: R_t not computable there (reason shown), not zero")
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "envs": [str(e) for e in envs],
        "checkpoints_shown": ticks,
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Figure 4: the δ-curve
# ---------------------------------------------------------------------------


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_delta(curve_recovered, curve_lesioned, out_base, lesioned_layer=None,
                 label_recovered="recovered checkpoint",
                 label_lesioned="just-lesioned", title=None, dpi=300):
    """The δ-curve: per-layer A_l difference between two layer curves.

        δ_l = A_l(recovered) − A_l(just-lesioned)

    curve_recovered / curve_lesioned   two figures.layer_curve outputs.
    lesioned_layer                     the permanently-lesioned layer l*,
                                       marked with a vertical line.

    Layers where either side is unmeasurable (A_l None) or present on only
    one curve become annotated gaps with the side named. No CI is drawn: a
    CI on the difference would need a paired bootstrap across the two runs,
    which is the metrics layer's job, not the plot's.
    """

    def by_layer(curve):
        out = {}
        for p in curve:
            key = _int_or_none(p.get("bypassed_layer"))
            if key is None:
                key = str(p.get("bypassed_layer"))
            out[key] = p
        return out

    rec = by_layer(curve_recovered)
    les = by_layer(curve_lesioned)
    all_layers = sorted(
        set(rec) | set(les),
        key=lambda v: (0, v, "") if isinstance(v, int) else (1, 0, str(v)),
    )

    xs, tick_labels = _positions(all_layers)
    deltas = []
    gaps = []          # (layer, reason)
    gap_marks = []
    for x, layer in zip(xs, all_layers):
        p_rec = rec.get(layer)
        p_les = les.get(layer)
        a_rec = p_rec.get("A_l") if p_rec else None
        a_les = p_les.get("A_l") if p_les else None
        if a_rec is None or a_les is None:
            rec_why = (p_rec or {}).get("reason") or (
                "absent" if p_rec is None else "A_l_none"
            )
            les_why = (p_les or {}).get("reason") or (
                "absent" if p_les is None else "A_l_none"
            )
            if a_rec is None and a_les is None and rec_why == les_why:
                reason = "both curves: %s" % rec_why
            else:
                missing = []
                if a_rec is None:
                    missing.append("%s: %s" % (label_recovered, rec_why))
                if a_les is None:
                    missing.append("%s: %s" % (label_lesioned, les_why))
                reason = "; ".join(missing)
            gaps.append((layer, reason))
            gap_marks.append((x, reason))
            deltas.append(None)
        else:
            deltas.append(metrics.relocation_delta_value(a_rec, a_les))

    plt = _plt()
    fig, ax = plt.subplots()

    measurable = [i for i, d in enumerate(deltas) if d is not None]
    for run in _contiguous_runs(measurable):
        ax.plot(
            [xs[i] for i in run], [deltas[i] for i in run],
            color=BLUE, marker="o", markeredgecolor="white",
            markeredgewidth=0.8, zorder=3,
        )
    _gap_marks(ax, gap_marks)
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)

    if lesioned_layer is not None:
        lx = _int_or_none(lesioned_layer)
        if lx is not None and tick_labels is not None:
            lx = xs[all_layers.index(lx)] if lx in all_layers else None
        if lx is not None:
            ax.axvline(float(lx), color=ORANGE, linewidth=1.4,
                       linestyle=(0, (5, 3)), zorder=2)
            ax.annotate(
                "$l^*$ (permanently lesioned)", xy=(float(lx), 0.98),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=TEXT,
                va="top",
            )

    if tick_labels is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("bypassed layer $l$")
    ax.set_ylabel(r"$\delta_l$  (%s $-$ %s $A_l$)" % (label_recovered, label_lesioned))
    ax.set_title(title or "Relocation: change in per-layer bypass effect after recovery")

    notes = []
    if gaps:
        notes.append("x at axis: δ not computable there (side and reason shown)")
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "layers": all_layers,
        "n_deltas": len(measurable),
        "gaps": gaps,
        "lesioned_layer": lesioned_layer,
    }


# ---------------------------------------------------------------------------
# Figure 5: tau bars
# ---------------------------------------------------------------------------


def render_tau_bars(records, out_base, title=None, dpi=300):
    """Incentive-sensitivity gap tau per model and arm, with CIs.

    records   one dict per (model, arm): metrics.tau_with_ci-shaped — tau,
              tau_ci_low, tau_ci_high (extras ignored) — plus "model" (group
              label, e.g. the model family) and "label" (the arm: "M_0",
              "M_D", "M_C"; other labels allowed and ordered after these).

    A record with tau None is an annotated gap in its slot (x + reason),
    never a zero-height bar.
    """
    plt = _plt()
    fig, ax = plt.subplots()

    models = []
    for r in records:
        if r.get("model") not in models:
            models.append(r.get("model"))
    labels = []
    for r in records:
        if r.get("label") not in labels:
            labels.append(r.get("label"))
    labels.sort(key=lambda l: (ARM_ORDER.index(l) if l in ARM_ORDER else len(ARM_ORDER), str(l)))

    n_labels = max(len(labels), 1)
    group_width = 0.8
    bar_width = group_width / n_labels
    gaps = []       # (model, label, reason)
    gap_marks = []
    extra_colors = [c for c in SERIES if c not in ARM_COLORS.values()]

    for j, label in enumerate(labels):
        color = ARM_COLORS.get(
            label, extra_colors[j % len(extra_colors)] if extra_colors else TEXT_SECONDARY
        )
        xs, ys, lo, hi = [], [], [], []
        for i, model in enumerate(models):
            matches = [
                r for r in records
                if r.get("model") == model and r.get("label") == label
            ]
            x = i - group_width / 2 + (j + 0.5) * bar_width
            if not matches or matches[0].get("tau") is None:
                reason = (matches[0].get("reason") if matches else None) or (
                    "tau_not_computable" if matches else "no record"
                )
                gaps.append((model, label, reason))
                gap_marks.append((x, "%s: %s" % (label, reason)))
                continue
            r = matches[0]
            xs.append(x)
            ys.append(r["tau"])
            lo.append(
                r["tau"] - r["tau_ci_low"] if r.get("tau_ci_low") is not None else 0.0
            )
            hi.append(
                r["tau_ci_high"] - r["tau"] if r.get("tau_ci_high") is not None else 0.0
            )
        if xs:
            ax.bar(
                xs, ys, width=bar_width * 0.94, color=color,
                edgecolor="white", linewidth=1.0, label=str(label), zorder=3,
            )
            ax.errorbar(
                xs, ys, yerr=[lo, hi], linestyle="none",
                ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=2.5, zorder=4,
            )
            # Value labels clear of the error bar: above its upper cap for
            # positive bars, below its lower cap otherwise.
            for x, y, err_lo, err_hi in zip(xs, ys, lo, hi):
                ax.annotate(
                    "%.2f" % y,
                    xy=(x, y + err_hi if y >= 0 else y - err_lo),
                    xytext=(0, 3 if y >= 0 else -10),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=TEXT, zorder=5,
                )

    _gap_marks(ax, gap_marks)
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([str(m) for m in models])
    ax.set_ylabel(r"$\tau = D(\mathrm{incentive}) - D(\mathrm{control})$")
    ax.set_title(title or "Incentive-sensitivity gap by model and arm")
    # Headroom so the pinned top-center legend clears the tallest bar+CI+label.
    tops = [
        r["tau_ci_high"] if r.get("tau_ci_high") is not None else r["tau"]
        for r in records if r.get("tau") is not None
    ]
    if tops:
        ax.set_ylim(top=max(max(tops), 0.0) * 1.3 or 1.0)
        ax.legend(loc="upper center", ncols=min(len(labels), 3))
    ax.grid(axis="x", visible=False)

    notes = []
    if gaps:
        notes.append("x at axis: tau not measured there (reason shown), not zero")
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "models": [str(m) for m in models],
        "labels": [str(l) for l in labels],
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Synthetic inputs (the dry-run path): plausible fakes of the real shapes
# ---------------------------------------------------------------------------


def _synthetic_point(layer, a_l, ci_half=0.06, reason=None, paired=True,
                     competence=0.9, competence_base=0.92,
                     invalid_inc=0.05, invalid_ctl=0.04):
    """One layer_curve-shaped point. a_l None -> unmeasurable with reason."""
    unmeasurable = a_l is None
    return {
        "bypassed_layer": layer,
        "run_id": "synthetic-L%s" % layer,
        "model_id": "synthetic/model-7B",
        "arm": "I,D",
        "checkpoint_step": 281,
        "split": "selection",
        "comparison": ("synthetic",),
        "A_l": a_l,
        "A_l_ci_low": None if unmeasurable else round(a_l - ci_half, 4),
        "A_l_ci_high": None if unmeasurable else round(a_l + ci_half, 4),
        "tau_base": 0.55,
        "tau_bypassed": None if unmeasurable else round(0.55 - a_l, 4),
        "n_scenarios_base": 100,
        "n_scenarios_bypassed": 100,
        "n_scenarios_common": 100 if paired else 60,
        "paired": paired,
        "reason": reason,
        "baseline_mismatch": None,
        "competence": competence,
        "competence_base": competence_base,
        "competence_drop": (
            None if (competence is None or competence_base is None)
            else round(competence_base - competence, 4)
        ),
        "invalid_rate_incentive": invalid_inc,
        "invalid_rate_control": invalid_ctl,
    }


def synthetic_layer_curve(n_layers=28, seed=0):
    """(points, statuses): a plausible sweep with one unmeasurable layer, one
    disqualified layer, and one partial-overlap layer."""
    rng = random.Random(seed)
    points, statuses = [], {}
    for layer in range(n_layers):
        # A bump of effect around the middle layers, noise elsewhere.
        a_l = 0.45 * math.exp(-((layer - 12) ** 2) / 18.0) + rng.uniform(-0.04, 0.04)
        a_l = round(a_l, 4)
        if layer == 20:
            # A bypass that destroyed the model: everything invalid.
            p = _synthetic_point(
                layer, None, reason="tau_not_computable",
                competence=None, invalid_inc=0.95, invalid_ctl=0.92,
            )
            p["competence_drop"] = None
            statuses[layer] = "UNMEASURABLE: tau_not_computable"
        elif layer == 4:
            # High invalid rate in the incentive condition: disqualified.
            p = _synthetic_point(
                layer, a_l, competence=0.88, invalid_inc=0.35, invalid_ctl=0.06,
            )
            statuses[layer] = "DISQUALIFIED: i15_inc"
        elif layer == 24:
            # A sweep job relaunched with a different --n: partial overlap.
            p = _synthetic_point(layer, a_l, reason="partial_overlap", paired=False)
            statuses[layer] = "VIABLE" if a_l >= 0.15 else "UNMEASURABLE: partial_overlap"
        else:
            drop = max(0.0, rng.gauss(0.01, 0.01)) + (0.04 if 10 <= layer <= 14 else 0.0)
            p = _synthetic_point(layer, a_l, competence=round(0.92 - drop, 4))
            viable = a_l >= 0.15 and p["A_l_ci_low"] is not None and p["A_l_ci_low"] > 0
            statuses[layer] = "VIABLE" if viable else "not_viable"
        points.append(p)
    return points, statuses


def synthetic_pareto(seed=0):
    """(points, statuses): pareto_points over the synthetic layer curve."""
    curve, statuses = synthetic_layer_curve(seed=seed)
    return figures.pareto_points(curve), statuses


def synthetic_rt(seed=0):
    """recovery()-shaped records for two environments at the ratified subset,
    with one null-with-reason point (the annotated-gap path)."""
    rng = random.Random(seed)
    records = []
    curves = {
        "negotiation": {8: 0.22, 70: 0.61, 281: 0.86},
        "insider_trading": {8: None, 70: 0.44, 281: 0.69},
    }
    for env, by_t in curves.items():
        for t in CHECKPOINT_STEPS:
            if by_t[t] is None:
                records.append({
                    "env": env, "checkpoint_step": t,
                    "tau_LD": 0.02, "tau_LC": 0.01, "tau_ID": 0.05, "tau_IC": 0.01,
                    "R_t": None, "R_t_ci_low": None, "R_t_ci_high": None,
                    "reason": "denominator_too_small",
                })
                continue
            r = by_t[t] + rng.uniform(-0.03, 0.03)
            records.append({
                "env": env, "checkpoint_step": t,
                "tau_LD": 0.3, "tau_LC": 0.05, "tau_ID": 0.5, "tau_IC": 0.05,
                "R_t": round(r, 4),
                "R_t_ci_low": round(r - 0.08, 4),
                "R_t_ci_high": round(r + 0.08, 4),
                "reason": None,
            })
    return records


def synthetic_delta(seed=0, lesioned_layer=12):
    """(curve_recovered, curve_lesioned, lesioned_layer) for the δ-curve.

    The just-lesioned model has the l* effect knocked out; the recovered
    checkpoint shows effect returning NEAR l* but not at it (relocation).
    """
    rng = random.Random(seed)
    lesioned, _ = synthetic_layer_curve(seed=seed)
    recovered = []
    for p in lesioned:
        q = dict(p)
        layer = q["bypassed_layer"]
        if q["A_l"] is not None:
            shift = 0.30 * math.exp(-((layer - (lesioned_layer + 4)) ** 2) / 6.0)
            drop = -0.9 * q["A_l"] if layer == lesioned_layer else 0.0
            a = round(q["A_l"] + shift + drop + rng.uniform(-0.02, 0.02), 4)
            q["A_l"] = a
            q["A_l_ci_low"] = round(a - 0.06, 4)
            q["A_l_ci_high"] = round(a + 0.06, 4)
        recovered.append(q)
    return recovered, lesioned, lesioned_layer


def synthetic_tau_bars(seed=0):
    """tau_with_ci-shaped records per model and arm, with one gap (a tau that
    was not computable)."""
    rng = random.Random(seed)
    records = []
    for model in ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B"):
        for label, center in (("M_0", 0.06), ("M_D", 0.52), ("M_C", 0.03)):
            if model == "Gemma-2-9B" and label == "M_C":
                records.append({
                    "model": model, "label": label,
                    "tau": None, "tau_ci_low": None, "tau_ci_high": None,
                    "reason": "tau_not_computable",
                })
                continue
            tau = round(center + rng.uniform(-0.03, 0.03), 4)
            records.append({
                "model": model, "label": label,
                "tau": tau,
                "tau_ci_low": round(tau - 0.05, 4),
                "tau_ci_high": round(tau + 0.05, 4),
                "n_scenarios": 100,
                "n_boot": 2000,
            })
    return records


# ---------------------------------------------------------------------------
# Figure: the edit heatmap (bypass layer x edited checkpoint)
# ---------------------------------------------------------------------------


def render_edit_heatmap(data, out_base, title=None, dpi=300):
    """Two aligned panels: clean-row D_incentive and truncation rate.

    Voided cells (invalid rate above the ruling bound) are greyed and marked
    'x' in the top panel - the rate a voided cell would have shown is never
    drawn. Missing layers stay blank. Both lists come back in the metadata,
    so nothing a cell hides goes unreported.
    """
    import numpy as np

    plt = _plt()
    keys = data["keys"]
    n_layers = data["n_layers"]
    cells = data["cells"]

    D = np.full((len(keys), n_layers), np.nan)
    T = np.full((len(keys), n_layers), np.nan)
    voided, missing, no_clean = [], [], []
    for i, key in enumerate(keys):
        for layer in range(n_layers):
            cell = cells[key][layer]
            if cell["status"] == "missing":
                missing.append((key, layer))
                continue
            T[i, layer] = cell["trunc_rate"]
            if cell["status"] == "voided_validity":
                voided.append((key, layer))
                continue
            if cell["clean_d_incentive"] is None:
                no_clean.append((key, layer))
                continue
            D[i, layer] = cell["clean_d_incentive"]

    fig, (ax_d, ax_t) = plt.subplots(
        2, 1, figsize=(max(8.0, 0.38 * n_layers), 1.4 + 1.1 * len(keys)),
        sharex=True,
    )
    cmap_d = plt.get_cmap("viridis").copy()
    cmap_d.set_bad("0.85")
    cmap_t = plt.get_cmap("magma").copy()
    cmap_t.set_bad("0.85")

    im_d = ax_d.imshow(D, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap_d)
    im_t = ax_t.imshow(T, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap_t)
    for (key, layer) in voided:
        ax_d.plot(layer, keys.index(key), marker="x", color="crimson",
                  markersize=7, markeredgewidth=1.6)
    for ax, im, label in ((ax_d, im_d, "clean-row D_incentive under bypass"),
                          (ax_t, im_t, "incentive truncation rate")):
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label(label)
    ax_t.set_xlabel("bypassed layer")
    ax_t.set_xticks(range(0, n_layers, 2))
    ax_d.set_title(title or "Deception under single-layer bypass, per edited "
                            "checkpoint")
    _footnote(fig, [
        "x = voided (incentive invalid rate > %.2f under the truncated->"
        "invalid ruling); grey = voided/missing/no clean rows"
        % data["invalid_max"],
        "voided: %d cells; missing: %d; measured-without-clean-rows: %d"
        % (len(voided), len(missing), len(no_clean)),
    ])

    paths = _save(fig, out_base, dpi=dpi)
    return {
        "paths": paths,
        "keys": keys,
        "n_layers": n_layers,
        "voided": voided,
        "missing": missing,
        "no_clean_rows": no_clean,
    }


def synthetic_edit_heatmap(seed=0):
    """Plausible fake heatmap data for the --synthetic dry run."""
    import random

    rng = random.Random(seed)
    keys = ["l07", "l13", "l10", "l21"]
    n_layers = 28
    cells = {}
    for key in keys:
        per_layer = {}
        for layer in range(n_layers):
            if layer in (0, n_layers - 1):
                per_layer[layer] = {"status": "voided_validity", "n": 100,
                                    "n_clean": 2, "clean_d_incentive": 0.0,
                                    "trunc_rate": 0.95, "invalid_rate": 0.98}
                continue
            hot = key == "l07" and layer == 2
            per_layer[layer] = {
                "status": "measured", "n": 100, "n_clean": 100,
                "clean_d_incentive": 0.47 if hot else round(rng.random() * 0.03, 3),
                "trunc_rate": 0.66 if hot else round(rng.random() * 0.04, 3),
                "invalid_rate": 0.02,
            }
        cells[key] = per_layer
    return {"keys": keys, "n_layers": n_layers, "invalid_max": 0.20,
            "cells": cells}


# ---------------------------------------------------------------------------
# Figure: probe-transfer AUROC per layer, per checkpoint (mentor experiment #3)
# ---------------------------------------------------------------------------


def render_probe_curves(curves, out_base, title=None, dpi=300):
    """AUROC-vs-layer lines, one per checkpoint, CI bands where present.

    curves: ordered list of (key, points) with points =
    [{"layer", "value", "ci_low", "ci_high"}]; a null value renders as a gap
    and is reported in the metadata, never drawn as chance.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    null_layers = {}
    for key, points in curves:
        pts = sorted((p for p in points), key=lambda p: int(p["layer"]))
        xs = [int(p["layer"]) for p in pts if p.get("value") is not None]
        ys = [p["value"] for p in pts if p.get("value") is not None]
        nulls = [int(p["layer"]) for p in pts if p.get("value") is None]
        if nulls:
            null_layers[key] = nulls
        line, = ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4,
                        label=key)
        lo = [p.get("ci_low") for p in pts if p.get("value") is not None]
        hi = [p.get("ci_high") for p in pts if p.get("value") is not None]
        if all(v is not None for v in lo) and all(v is not None for v in hi):
            ax.fill_between(xs, lo, hi, alpha=0.14, color=line.get_color())
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=1.0)
    ax.annotate("chance", xy=(0.995, 0.5), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="0.45")
    ax.set_xlabel("layer")
    ax.set_ylabel("transfer AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=min(len(curves), 6), fontsize=8)
    ax.set_title(title or "Instructed-pairs probe transfer to strategic "
                          "deception, per checkpoint")
    if null_layers:
        _footnote(fig, ["null AUROC (degenerate bootstrap or structural): "
                        + "; ".join("%s: %s" % (k, v)
                                    for k, v in null_layers.items())])
    paths = _save(fig, out_base, dpi=dpi)
    return {"paths": paths, "keys": [k for k, _ in curves],
            "null_layers": null_layers}


def synthetic_probe_curves(seed=0):
    import random

    rng = random.Random(seed)
    curves = []
    for key, peak in (("m0", 1.0), ("md", 0.82), ("me", 0.99)):
        points = []
        for layer in range(28):
            base = 0.45 + 0.1 * rng.random()
            bump = peak - base
            value = base + bump * max(0.0, 1 - abs(layer - 20) / 9.0)
            points.append({"layer": layer, "value": round(value, 3),
                           "ci_low": round(value - 0.05, 3),
                           "ci_high": round(min(1.0, value + 0.05), 3)})
        points[0]["value"] = None
        curves.append((key, points))
    return curves


# ---------------------------------------------------------------------------
# Figure 8: raw per-arm taus across recovery checkpoints
# ---------------------------------------------------------------------------


def render_recovery_taus(records, out_base, checkpoints=CHECKPOINT_STEPS,
                         title=None, dpi=300):
    """Raw per-arm tau vs checkpoint t: the honest view behind R_t.

    records   same file scripts/recovery_report.py --emit-records writes:
              one dict per (env, checkpoint) with "env", "checkpoint_step",
              "arms" (ordered list like ["E,D", "E,C", "I,D", "I,C"]) and
              one "tau_<ARM>" value per arm (comma stripped: tau_ED, ...).

    One subplot per env, one line per arm. NO CI band is drawn: the record
    carries none (metrics.recovery bootstraps only the R_t ratio), and
    inventing one here would be a second home for the statistic — the
    footnote says so. A None tau is an annotated gap, never a zero.
    """
    plt = _plt()

    envs = []
    for r in records:
        env = r.get("env")
        if env not in envs:
            envs.append(env)
    if not envs:
        raise ValueError("no records with an 'env' field")
    arms = None
    for r in records:
        if r.get("arms"):
            arms = list(r["arms"])
            break
    if not arms:
        raise ValueError("no record carries an 'arms' list")

    fig, axes = plt.subplots(
        1, len(envs), figsize=(4.6 * len(envs), 4.0),
        sharey=True, squeeze=False,
    )
    gaps = []          # (env, arm, t, reason)
    for env_idx, env in enumerate(envs):
        ax = axes[0][env_idx]
        env_records = sorted(
            (r for r in records if r.get("env") == env),
            key=lambda r: r.get("checkpoint_step"),
        )
        gap_marks = []
        for arm_idx, arm in enumerate(arms):
            key = "tau_%s" % arm.replace(",", "")
            color = SERIES[arm_idx % len(SERIES)]
            marker = MARKERS[arm_idx % len(MARKERS)]
            measurable = [
                i for i, r in enumerate(env_records) if r.get(key) is not None
            ]
            for run_idx, run in enumerate(_contiguous_runs(measurable)):
                run_records = [env_records[i] for i in run]
                ax.plot(
                    [r["checkpoint_step"] for r in run_records],
                    [r[key] for r in run_records],
                    color=color, marker=marker,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=arm if run_idx == 0 else None, zorder=3,
                )
            for r in env_records:
                if r.get(key) is None:
                    reason = r.get("reason") or "tau_none"
                    gaps.append((env, arm, r.get("checkpoint_step"), reason))
                    gap_marks.append(
                        (r.get("checkpoint_step"), "%s: %s" % (arm, reason))
                    )
        _gap_marks(ax, gap_marks)
        ticks = sorted(
            set(checkpoints)
            | set(r.get("checkpoint_step") for r in env_records
                  if r.get("checkpoint_step") is not None)
        )
        ax.set_xticks(ticks)
        ax.set_xlabel("fine-tuning checkpoint $t$ (steps)")
        ax.set_title(str(env), fontsize=9.5)
        ax.set_ylim(-0.05, 1.05)
        if env_idx == 0:
            ax.set_ylabel(r"$\tau$ = D(incentive) $-$ D(control)")
            ax.legend(loc="center right", title=None)
    fig.suptitle(title or "Raw per-arm deception gaps across recovery "
                          "checkpoints", fontsize=10.5)

    notes = ["point estimates only: the record carries no per-arm CI "
             "(only the R_t ratio is bootstrapped)"]
    if gaps:
        notes.append("x at axis: tau not computable there (reason shown), "
                     "not zero")
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "envs": [str(e) for e in envs],
        "arms": arms,
        "gaps": gaps,
    }


def synthetic_recovery_taus(seed=0):
    import random

    rng = random.Random(seed)
    records = []
    arms = ["E,D", "E,C", "I,D", "I,C"]
    for env, snap in (("path A", 8), ("path B", 70)):
        for t in CHECKPOINT_STEPS:
            record = {"env": env, "checkpoint_step": t, "arms": arms,
                      "R_t": None, "R_t_ci_low": None, "R_t_ci_high": None,
                      "reason": None}
            record["tau_ED"] = (
                round(min(1.0, 0.9 + 0.1 * rng.random()), 3)
                if t >= snap else round(0.1 * rng.random(), 3)
            )
            record["tau_EC"] = round(0.05 * rng.random(), 3)
            record["tau_ID"] = round(min(1.0, 0.8 + 0.2 * rng.random()), 3)
            record["tau_IC"] = round(0.05 * rng.random(), 3)
            records.append(record)
    records[0]["tau_ID"] = None
    records[0]["reason"] = "denominator_too_small"
    return records


# ---------------------------------------------------------------------------
# Figure: per-layer output decomposition (stacked bars)
# ---------------------------------------------------------------------------

# Category colors follow the entity; the two invalid categories are hatched
# so identity is never color-alone.
DECOMPOSITION_COLORS = {
    "inflated": RED, "fabricated": ORANGE, "exact_truth": BLUE,
    "correct_none": AQUA, "understated": YELLOW,
    "invalid_truncated": GAP_COLOR, "invalid_other": "#bdbbb6",
}
DECOMPOSITION_HATCHES = {"invalid_truncated": "//", "invalid_other": "xx"}


def render_decomposition(record, out_base, condition="incentive", title=None,
                         dpi=300):
    """One stacked bar per bypassed layer: what the model actually said.

    record   figures.decomposition_cells output (emit_figure_records.py
             decomposition). Fractions are of that condition's rows, so
             truncated / invalid rows are visible mass, never a denominator
             adjustment. Voided layers (invalid rate above the ruling bound)
             are marked with an x above the bar; missing layers are gaps.
    """
    plt = _plt()
    categories = list(record.get("categories") or figures.DECOMPOSITION_CATEGORIES)
    layers = list(record.get("layers") or [])
    if not layers:
        raise ValueError("decomposition record has no layers")
    n_layers = record.get("n_layers") or len(layers)
    missing, voided, totals, fractions = [], [], {}, {}
    for entry in layers:
        x = int(entry["bypassed_layer"])
        cell = None
        if entry.get("status") == "measured":
            cell = (entry.get("conditions") or {}).get(condition)
        if not cell or not cell.get("n"):
            missing.append(x)
            continue
        totals[x] = cell["n"]
        fractions[x] = {c: cell["counts"].get(c, 0) / cell["n"] for c in categories}
        if cell.get("voided_validity"):
            voided.append(x)

    fig, ax = plt.subplots(figsize=(max(6.4, 0.32 * n_layers), 4.2))
    xs = list(fractions)
    bottoms = {x: 0.0 for x in xs}
    for category in categories:
        heights = [fractions[x][category] for x in xs]
        hatch = DECOMPOSITION_HATCHES.get(category)
        ax.bar(
            xs, heights, bottom=[bottoms[x] for x in xs], width=0.8,
            color=DECOMPOSITION_COLORS.get(category, TEXT_SECONDARY),
            hatch=hatch, edgecolor=TEXT_SECONDARY if hatch else "white",
            linewidth=0.4, label=category.replace("_", " "), zorder=3,
        )
        for x, h in zip(xs, heights):
            bottoms[x] += h
    for x in voided:
        ax.plot([x], [1.04], marker="x", color="crimson", markersize=7,
                markeredgewidth=1.6, clip_on=False, zorder=5, linestyle="none")
    if missing:
        _gap_marks(ax, [(x, "missing") for x in missing])
    ax.set_ylim(0.0, 1.1)
    ax.set_xlim(-0.7, n_layers - 0.3)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xlabel("bypassed layer $l$")
    ax.set_ylabel("fraction of %s-condition rows" % condition)
    ax.set_title(
        title or "Output decomposition per bypassed layer (%s condition)"
        % condition
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)
    notes = [
        "x above a bar: voided (invalid rate > %.2f, truncated-or-invalid); "
        "x at axis: missing layer, not zero"
        % record.get("invalid_max", metrics.INVALID_RATE_MAX),
    ]
    if voided:
        notes.append("voided layers: %s" % ", ".join(str(x) for x in voided))
    if missing:
        notes.append("missing layers: %s" % ", ".join(str(x) for x in missing))
    _footnote(fig, notes)
    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "condition": condition,
        "layers": [int(entry["bypassed_layer"]) for entry in layers],
        "missing": missing,
        "voided": voided,
        "totals": totals,
    }


def _decomposition_row(scenario, condition, category, layer, run_id):
    """One synthetic scored row landing in exactly `category`."""
    row = {
        "scenario_id": "s%03d" % scenario, "condition": condition,
        "bypassed_layer": layer, "run_id": run_id,
        "hit_max_tokens": False, "valid": True, "deceptive": False,
        "deception_type": None, "understated": False,
        "claimed_value": 90000.0, "true_value": 90000, "invalid_reason": None,
    }
    if category == "inflated":
        row.update(deceptive=True, deception_type="inflated", claimed_value=120000.0)
    elif category == "fabricated":
        row.update(deceptive=True, deception_type="fabricated",
                   claimed_value=120000.0, true_value=None)
    elif category == "correct_none":
        row.update(claimed_value="NONE", true_value=None)
    elif category == "understated":
        row.update(understated=True, claimed_value=80000.0)
    elif category == "invalid_truncated":
        row.update(hit_max_tokens=True)
    elif category == "invalid_other":
        row.update(valid=False, deceptive=None, understated=None,
                   claimed_value=None, invalid_reason="unparseable")
    elif category != "exact_truth":
        raise ValueError("unknown decomposition category %r" % category)
    return row


def synthetic_decomposition(seed=0, n_layers=28):
    """A plausible sweep decomposition: one missing layer (5), one all-invalid
    layer (0), one mostly-truncated voided layer (n-2), mixed elsewhere."""
    rng = random.Random(seed)
    layer_rows = {}
    for layer in range(n_layers):
        if layer == 5:
            continue
        rows = []
        run_id = "synthetic-L%d" % layer
        for s in range(20):
            if layer == 0:
                category = "invalid_other"
            elif layer == n_layers - 2:
                category = "invalid_truncated" if s < 16 else "inflated"
            else:
                r = rng.random()
                category = (
                    "inflated" if r < 0.55 else "fabricated" if r < 0.70
                    else "exact_truth" if r < 0.85 else "correct_none" if r < 0.93
                    else "understated" if r < 0.97 else "invalid_truncated"
                )
            rows.append(_decomposition_row(s, "incentive", category, layer, run_id))
            rows.append(_decomposition_row(
                s, "control", "exact_truth" if s % 5 else "correct_none",
                layer, run_id,
            ))
        layer_rows[layer] = rows
    return figures.decomposition_cells(layer_rows, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Figure: the edit gates side by side (removal, drift, capability, Stage 1)
# ---------------------------------------------------------------------------


def _bound_line(ax, y, label, values):
    """Dashed ratified bound; if it sits far above every value it would
    flatten the panel, so it is annotated as off-scale instead of drawn."""
    finite = [v for v in values if v is not None]
    top = max(finite) if finite else 0.0
    if top <= 0.0 or y <= top * 4.0:
        ax.axhline(y, color=TEXT_SECONDARY, linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            label, xy=(0.99, y), xycoords=("axes fraction", "data"),
            xytext=(0, 3), textcoords="offset points", fontsize=7.5,
            color=TEXT_SECONDARY, ha="right", va="bottom",
        )
        return True
    ax.annotate(
        "%s (bound off-scale)" % label, xy=(0.01, 0.97), xycoords="axes fraction",
        fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="top",
    )
    return False


def render_edit_gate_summary(records, out_base, title=None, dpi=300):
    """Six panels over the edited checkpoints (x), grouped by model:

      (a) A_edit with its bootstrap CI and the exact M_D -> M_E incentive
          counts; (b) incentive D for M_D and M_E with Wilson 95% bars;
      (c) M_D <-> M_E edit JSD; (d) MMLU / GSM8K change vs M_0 with stderr;
      (e) perplexity rise vs M_0; (f) the window's Stage-1 A_l at its center
          layer with CI -- juxtaposed with (a), no correlation statistic.

    records  emit_figure_records.py edit-gate-summary JSONL. Every null
    quantity is an annotated gap, listed in metadata, never a zero.
    """
    records = [dict(r) for r in records]
    if not records:
        raise ValueError("edit-gate summary needs at least one record")
    plt = _plt()
    models = []
    for r in records:
        if r.get("model") not in models:
            models.append(r.get("model"))
    color_of = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}
    marker_of = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(models)}
    xs = list(range(len(records)))
    keys = [str(r.get("key")) for r in records]
    tick_labels = keys
    first_bounds = records[0].get("bounds") or {}

    def bound(name, default):
        value = first_bounds.get(name)
        return default if value is None else value

    a_edit_min = bound("a_edit_min", 0.15)
    drop_max = bound("competence_drop_max", 0.05)
    ppl_max = bound("ppl_rise_max", 2.0)
    jsd_max = bound("edit_jsd_max", 0.25)
    a_l_min = bound("a_l_min", 0.15)
    gaps = []
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    def gap(ax, x, key, quantity, reason):
        gaps.append((key, quantity, reason))
        _gap_marks(ax, [(x, "n/a: %s" % reason)])

    def err(v, lo, hi):
        if lo is None or hi is None:
            return None
        return [[max(0.0, v - lo)], [max(0.0, hi - v)]]

    # (a) A_edit
    for x, r in zip(xs, records):
        v = r.get("A_edit")
        if v is None:
            gap(ax_a, x, r["key"], "A_edit", "null")
            continue
        ax_a.bar([x], [v], width=0.6, color=color_of[r["model"]], zorder=3)
        yerr = err(v, r.get("A_edit_ci_low"), r.get("A_edit_ci_high"))
        if yerr is not None:
            ax_a.errorbar([x], [v], yerr=yerr, fmt="none", ecolor=TEXT,
                          capsize=3, zorder=4)
        counts = r.get("counts") or {}
        cd = (counts.get("M_D") or {}).get("incentive") or {}
        ce = (counts.get("M_E") or {}).get("incentive") or {}
        if cd and ce:
            ax_a.annotate(
                "%d/%d → %d/%d" % (cd["n_deceptive"], cd["n_valid"],
                                        ce["n_deceptive"], ce["n_valid"]),
                xy=(x, v / 2.0), ha="center", va="center", rotation=90,
                fontsize=6.5, color="white", zorder=5,
            )
    from matplotlib.patches import Patch

    ax_a.legend(
        handles=[Patch(color=color_of[m], label=m) for m in models],
        loc="upper center", ncol=len(models), fontsize=7.5, title="model",
        title_fontsize=7.5,
    )
    _bound_line(ax_a, a_edit_min, "A_edit >= %.2f" % a_edit_min, [1.0])
    ax_a.set_ylim(0, 1.3)
    ax_a.set_ylabel(r"$A_{edit} = \tau(M_D) - \tau(M_E)$")
    ax_a.set_title("(a) removal effect; counts = M_D→M_E incentive lies",
                   fontsize=9)

    # (b) incentive D with Wilson bars
    for x, r in zip(xs, records):
        counts = r.get("counts") or {}
        for name, dx, color, marker in (("M_D", -0.15, ORANGE, "s"),
                                        ("M_E", 0.15, BLUE, "o")):
            c = (counts.get(name) or {}).get("incentive") or {}
            if not c or c.get("D") is None:
                gap(ax_b, x + dx, r["key"], "%s incentive D" % name, "null")
                continue
            ax_b.errorbar(
                [x + dx], [c["D"]],
                yerr=err(c["D"], c.get("wilson_low"), c.get("wilson_high")),
                fmt=marker, color=color, capsize=3, markersize=6,
                label=name if x == 0 else None, zorder=3,
            )
    ax_b.set_ylim(-0.05, 1.1)
    ax_b.set_ylabel("incentive deception rate D (valid rows)")
    ax_b.set_title("(b) exact rates, Wilson 95% intervals", fontsize=9)
    ax_b.legend(loc="center right", fontsize=7.5)

    # (c) edit JSD
    jsd_values = [r.get("edit_jsd") for r in records]
    for x, r in zip(xs, records):
        v = r.get("edit_jsd")
        if v is None:
            gap(ax_c, x, r["key"], "edit_jsd", "null")
            continue
        ax_c.bar([x], [v], width=0.6, color=color_of[r["model"]], zorder=3)
    _bound_line(ax_c, jsd_max, "<= %.2f nats" % jsd_max, jsd_values)
    ax_c.set_ylabel("M_D ↔ M_E WikiText-2 JSD (nats)")
    ax_c.set_title("(c) distributional drift of the edit", fontsize=9)

    # (d) benchmark deltas
    for x, r in zip(xs, records):
        for field, se_field, dx, color, hatch, label in (
            ("delta_mmlu", "delta_mmlu_stderr", -0.18, AQUA, None, "MMLU"),
            ("delta_gsm8k", "delta_gsm8k_stderr", 0.18, YELLOW, "//", "GSM8K"),
        ):
            v = r.get(field)
            if v is None:
                gap(ax_d, x + dx, r["key"], field, "null")
                continue
            ax_d.bar([x + dx], [v], width=0.34, color=color, hatch=hatch,
                     edgecolor=TEXT_SECONDARY, linewidth=0.4,
                     label=label if x == 0 else None, zorder=3)
            se = r.get(se_field)
            if se is not None:
                ax_d.errorbar([x + dx], [v], yerr=[[se], [se]], fmt="none",
                              ecolor=TEXT, capsize=2, zorder=4)
    ax_d.axhline(0, color=TEXT_SECONDARY, linewidth=0.6)
    ax_d.axhline(-drop_max, color=TEXT_SECONDARY, linewidth=0.9,
                 linestyle=(0, (4, 3)), zorder=1)
    ax_d.annotate(">= -%.2f" % drop_max, xy=(0.99, -drop_max),
                  xycoords=("axes fraction", "data"), xytext=(0, 3),
                  textcoords="offset points", fontsize=7.5,
                  color=TEXT_SECONDARY, ha="right", va="bottom")
    ax_d.set_ylabel("M_E − M_0 accuracy")
    ax_d.set_title("(d) benchmark change vs M_0 (stderr bars)", fontsize=9)
    ax_d.legend(loc="upper left", fontsize=7.5)

    # (e) perplexity rise
    ppl_values = [r.get("delta_ppl") for r in records]
    for x, r in zip(xs, records):
        v = r.get("delta_ppl")
        if v is None:
            gap(ax_e, x, r["key"], "delta_ppl", "null")
            continue
        ax_e.bar([x], [v], width=0.6, color=color_of[r["model"]], zorder=3)
    ax_e.axhline(0, color=TEXT_SECONDARY, linewidth=0.6)
    _bound_line(ax_e, ppl_max, "<= +%.1f" % ppl_max, ppl_values)
    ax_e.set_ylabel("M_E − M_0 WikiText-2 perplexity")
    ax_e.set_title("(e) perplexity rise vs M_0", fontsize=9)

    # (f) Stage-1 A_l at the window's center layer
    for x, r in zip(xs, records):
        v = r.get("stage1_A_l")
        if v is None:
            gap(ax_f, x, r["key"], "stage1_A_l", r.get("stage1_reason") or "null")
            continue
        ax_f.errorbar(
            [x], [v], yerr=err(v, r.get("stage1_A_l_ci_low"),
                               r.get("stage1_A_l_ci_high")),
            fmt=marker_of[r["model"]], color=color_of[r["model"]], capsize=3,
            markersize=6, zorder=3,
        )
    ax_f.axhline(0, color=TEXT_SECONDARY, linewidth=0.6)
    _bound_line(ax_f, a_l_min, "A_l >= %.2f" % a_l_min, [1.0])
    ax_f.set_ylabel(r"Stage-1 $A_l$ at the window's center layer")
    ax_f.set_title("(f) causal effect of the edited window before editing",
                   fontsize=9)

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_xlim(-0.6, len(records) - 0.4)
    fig.suptitle(
        title or "Layer-edit gates: removal, drift, capability, and the "
        "window's Stage-1 causal effect"
    )
    notes = [
        "x at axis: quantity unavailable (reason shown), not zero; panel (f) "
        "is juxtaposition only -- no correlation statistic over %d windows"
        % len(records),
    ]
    if gaps:
        notes.append("gaps: %s" % "; ".join("%s %s (%s)" % g for g in gaps))
    _footnote(fig, notes)
    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "keys": keys,
        "models": models,
        "gaps": gaps,
    }


def synthetic_edit_gate_summary(seed=0):
    """Six plausible gate records (two models), one with a null Stage-1 A_l."""
    rng = random.Random(seed)
    specs = [
        ("Qwen2.5-7B", "l07", [6, 7, 8], 0.28),
        ("Qwen2.5-7B", "l10", [9, 10, 11], 0.09),
        ("Qwen2.5-7B", "l13", [12, 13, 14], 0.22),
        ("Qwen2.5-7B", "l21", [20, 21, 22], 0.0),
        ("Llama-3.1-8B", "l08", [7, 8, 9], 0.0),
        ("Llama-3.1-8B", "l24", [23, 24, 25], None),
    ]
    wl, wh = metrics.wilson_interval(0, 305)
    dl, dh = metrics.wilson_interval(305, 305)
    records = []
    for model, key, layers, a in specs:
        records.append({
            "model": model, "key": key, "edit_layers": layers,
            "center_layer": layers[1],
            "A_edit": 1.0, "A_edit_ci_low": 1.0, "A_edit_ci_high": 1.0,
            "counts": {
                "M_D": {"incentive": {"n_deceptive": 305, "n_valid": 305,
                                      "n_total": 305, "D": 1.0,
                                      "wilson_low": dl, "wilson_high": dh}},
                "M_E": {"incentive": {"n_deceptive": 0, "n_valid": 305,
                                      "n_total": 305, "D": 0.0,
                                      "wilson_low": wl, "wilson_high": wh}},
            },
            "edit_jsd": round(0.0002 + 0.002 * rng.random(), 5),
            "delta_mmlu": round(rng.uniform(-0.02, 0.012), 4),
            "delta_gsm8k": round(rng.uniform(-0.025, 0.07), 4),
            "delta_ppl": round(rng.uniform(0.0, 0.2), 4),
            "delta_mmlu_stderr": 0.015, "delta_gsm8k_stderr": 0.021,
            "verdict": "PASS",
            "stage1_A_l": a,
            "stage1_A_l_ci_low": None if a is None else round(a - 0.08, 3),
            "stage1_A_l_ci_high": None if a is None else round(a + 0.09, 3),
            "stage1_reason": "layer_not_in_curve" if a is None else None,
            "stage1_run_id": None if a is None else "synthetic-L%d" % layers[1],
            "bounds": {"a_edit_min": 0.15, "competence_drop_max": 0.05,
                       "ppl_rise_max": 2.0, "edit_jsd_max": 0.25,
                       "a_l_min": 0.15},
        })
    return records


def synthetic_pareto_panels(seed=0):
    """A pareto record with three panels over the synthetic layer curve."""
    curve, _statuses = synthetic_layer_curve(seed=seed)
    rng = random.Random(seed)
    panels = []
    for metric, cap, scale, reference in (
        ("task_competence", 0.05, None, "sweep_base"),
        ("wikitext2_ppl", 2.0, 8.0, "synthetic-base"),
        ("wikitext2_neutral_jsd", 0.25, 0.3, "absolute"),
    ):
        points = figures.pareto_points(curve)
        for p in points:
            p["damage_metric"] = metric
            p["damage_reference"] = reference
            if scale is not None and p["damage"] is not None:
                p["damage"] = round(p["damage"] * scale + rng.uniform(0, 0.02), 4)
        frontier = figures.pareto_frontier(points)
        panels.append({
            "damage_metric": metric, "damage_reference": reference,
            "bounds": {"a_l_min": 0.15, "damage_max": cap},
            "pareto_points": points, "frontier": frontier,
        })
    return {"model": "synthetic/model-7B", "base_run_id": "synthetic-base",
            "bounds_a_l_min": 0.15, "panels": panels}
