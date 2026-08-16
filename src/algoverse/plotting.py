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

from algoverse import figures

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


def render_pareto(points, out_base, statuses=None, frontier=None,
                  allow_mixed=False, title=None, dpi=300):
    """A_l (x, more deception removed) vs damage (y, lower is better).

    points    figures.pareto_points output: layer-curve dicts plus damage,
              damage_metric, damage_reason.
    statuses  optional {layer: status}; disqualified points drawn hollow.
    frontier  optional precomputed figures.pareto_frontier list; computed
              here otherwise (allow_mixed passed through).

    Points missing A_l or damage cannot be placed on the axes; they are
    listed in the footnote and returned in metadata, never silently dropped.
    """
    statuses = statuses or {}
    _normalize_comparison(points)
    if frontier is None:
        frontier = figures.pareto_frontier(points, allow_mixed=allow_mixed)
    frontier_layers = [p.get("bypassed_layer") for p in frontier]

    plt = _plt()
    fig, ax = plt.subplots()

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

    damage_metric = points[0].get("damage_metric") if points else None
    ax.set_xlabel(r"$A_l$ (deception removed, higher is better)")
    ax.set_ylabel(
        "damage%s (lower is better)"
        % (" [%s]" % damage_metric if damage_metric else "")
    )
    ax.set_title(title or "Deception removed vs capability damage")
    if frontier:
        ax.legend(loc="best")

    notes = []
    if disqualified_layers:
        notes.append(
            "hollow: disqualified layers %s"
            % ", ".join(str(l) for l in disqualified_layers)
        )
    for layer, reason in off_plot:
        notes.append("not plottable, layer %s: %s" % (layer, reason))
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "n_points": len(points),
        "n_plotted": len(plottable),
        "frontier_layers": frontier_layers,
        "disqualified": disqualified_layers,
        "off_plot": off_plot,
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
            deltas.append(a_rec - a_les)

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
