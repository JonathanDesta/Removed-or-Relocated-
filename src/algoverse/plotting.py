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

The render functions (scripts/make_figures.py is the CLI around them):

  render_layer_curve   A_l vs bypassed layer, CI band, voided and
                       bound-breaching layers shaded, unmeasurable layers
                       marked at the axis with their reason.
  render_rt            R_t vs checkpoint t (the pre-registered subset
                       {8, 70, 281}), one line per environment; null R_t
                       (metrics.recovery returning None with a reason) is an
                       annotated gap.
  render_delta         the δ-curve: A_l(recovered E,D) minus A_l(just-edited
                       M_E), per layer.
  render_tau_bars      tau(M_0) / tau(M_D) / tau(M_E-<window>) per model with
                       CIs, from metrics.tau_with_ci-shaped dicts.
  render_edit_heatmap  bypass layer x edited checkpoint: clean-row incentive
                       deception rate and truncation rate, voided cells marked.
  render_probe_curves  probe-transfer AUROC per layer, one line per
                       checkpoint, null AUROC left as a gap.

Every render function writes <out_base>.png (300 dpi) and <out_base>.pdf and
returns a metadata dict that includes "paths" plus everything a test (or a
reader of the caption) needs to confirm nothing was dropped: the shaded
layers, the unmeasurable/flagged layers with reasons, the annotated gaps.

Color: the categorical slots below are a color-blind-safe ordering validated
with the dataviz palette checker (adjacent-pair CVD ΔE ≥ 8, normal-vision
ΔE ≥ 15, light surface). Aqua and yellow sit below 3:1 contrast on white, so
the figures that use them carry direct labels. Identity is never color-alone:
series also differ by marker shape.

Synthetic data: each figure has a synthetic_* generator producing plausible
fake inputs of exactly the shapes the render functions (and the real
pipeline) use — including an unmeasurable layer, a voided layer, and a
null R_t — so the whole rendering path is dry-runnable with no real results.
"""

import json
import math
import os
import random
import re

from algoverse import figures, metrics

# The pre-registered R_t evaluation subset: early / mid / final of the saved
# checkpoints [8, 17, 35, 70, 140, 281].
CHECKPOINT_STEPS = (8, 70, 281)

# Color-blind-safe categorical slots, in fixed order (never cycled past what
# is listed). The first five were validated with a colour-vision-deficiency
# palette checker (first three all-pairs, the five adjacent pairs); PURPLE
# and GREY are taken from Petroff's colour-blind-safe six-colour scheme and
# complete the tau-bars series (two fixed arms plus up to five edited
# checkpoints, each with its own colour).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
PURPLE = "#964a8b"
GREY = "#9c9ca1"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, RED, PURPLE, GREY)
MARKERS = ("o", "s", "^", "D", "v", "P", "X")

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SHADE = "#f0efec"          # disqualified-layer band
GAP_COLOR = "#52514e"      # annotated-gap marks: neutral ink, not a series hue

# tau bars: fixed color per arm label, stable regardless of which models or
# arms happen to be present (color follows the entity, never its rank).
# Labels outside this table (the edited checkpoints M_E-<window>) are ordered
# after it and take the remaining series colors in order; more labels than
# spare colors is refused rather than drawn with a shared color.
ARM_COLORS = {"M_0": BLUE, "M_D": ORANGE}
ARM_ORDER = ("M_0", "M_D")

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


def _is_shaded(status):
    """Sweep statuses drawn shaded and hollow: voided or breaching a bound."""
    return bool(status) and str(status).startswith(("VOIDED", "BREACHES"))


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


def _footnote(fig, lines, fontsize=7, y=-0.02):
    """Caption-adjacent notes below the axes; kept by the tight save bbox."""
    if lines:
        fig.text(
            0.01, y, "\n".join(lines), ha="left", va="top",
            fontsize=fontsize, color=TEXT_SECONDARY,
        )


def _resolve_title(title, default):
    """None -> the renderer's default; "" -> no title (the caption carries
    it, as a paper figure needs); any other string verbatim. `title or
    default` would silently restore the default on ""."""
    return default if title is None else title


# ---------------------------------------------------------------------------
# The layer curve
# ---------------------------------------------------------------------------


def render_layer_curve(points, out_base, statuses=None, title=None, dpi=300):
    """A_l vs bypassed layer.

    points    figures.layer_curve output: dicts with bypassed_layer, A_l,
              A_l_ci_low, A_l_ci_high, reason, paired, competence,
              invalid_rate_incentive/control (extra keys ignored).
    statuses  optional {layer: status string} from sweep.evaluate_sweep
              entries ("CLEARS BOUNDS" / "BREACHES: ..." / "VOIDED: ..." /
              "BELOW EFFECT FLOOR" / "UNMEASURABLE: ..." / "NO ROWS");
              voided and breaching layers get a shaded band and hollow
              markers (the metadata lists them under "disqualified").
              Without statuses the curve is plain.

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
    half = 0.45
    for i, p in enumerate(points):
        status = status_of(p)
        disq = _is_shaded(status)
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
    ax.set_title(_resolve_title(title, "Deception-specific bypass effect by layer"))

    notes = []
    if disqualified_layers:
        notes.append(
            "shaded/hollow: voided or bound-breaching layers %s"
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
# Recovery curves (Figure 2)
# ---------------------------------------------------------------------------


def _rt_constituents(record):
    """[(arm, tau)] for one recovery record, in the record's arm order.

    The record carries one tau_<ARM> per arm (recovery_report.py
    --emit-records: tau_ED, tau_EC, tau_ID, tau_IC); `arms` names them in
    order when present, otherwise the canonical E,D / E,C / I,D / I,C order
    followed by any other tau_* key sorted. A missing or null tau is kept as
    None.
    """
    arms = record.get("arms")
    if isinstance(arms, dict):
        arms = list(arms)
    if not arms:
        available = {k[len("tau_"):] for k in record if k.startswith("tau_")}
        arms = [a for a in ("ED", "EC", "ID", "IC") if a in available]
        arms += sorted(available - set(arms))
    tau_keys = {re.sub(r"[^A-Za-z0-9]", "", k[len("tau_"):]): k
                for k in record if k.startswith("tau_")}
    out = []
    for arm in arms:
        label = str(arm)
        key = tau_keys.get(re.sub(r"[^A-Za-z0-9]", "", label))
        out.append((label, record.get(key) if key else None))
    return out


def _rt_annotation_text(display, t, record):
    """Three lines: which comparison, the per-arm taus, and R_t from them."""
    parts = _rt_constituents(record)

    def fmt(arm, value):
        return r"$\tau_{%s}$ = %s" % (arm, "n/a" if value is None else "%.3f" % value)

    if len(parts) == 4:
        line2 = "%s, %s (edited); %s, %s (intact)" % (
            fmt(*parts[0]), fmt(*parts[1]), fmt(*parts[2]), fmt(*parts[3]))
    else:
        line2 = ", ".join(fmt(arm, value) for arm, value in parts)
    lines = ["%s, checkpoint %s: edited model vs intact model" % (display, t), line2]
    if record.get("R_t") is not None and len(parts) == 4 and all(v is not None for _, v in parts):
        (a, ta), (b, tb), (c, tc), (d, td) = parts
        lines.append(r"$R_{%s}$ = (%.3f $-$ %.3f) / (%.3f $-$ %.3f) = %.2f"
                     % (t, ta, tb, tc, td, record["R_t"]))
    elif record.get("R_t") is not None:
        lines.append(r"$R_{%s}$ = %.2f" % (t, record["R_t"]))
    return "\n".join(lines)


def render_rt(records, out_base, checkpoints=CHECKPOINT_STEPS, title=None,
              dpi=300, env_labels=None, annotate=(), notes=(), xlabel=None):
    """R_t vs fine-tuning checkpoint t, one line per environment.

    records   one dict per (environment, checkpoint): the metrics.recovery()
              output (R_t, R_t_ci_low, R_t_ci_high, reason, tau_* extras)
              plus "env" (line label) and "checkpoint_step" (int).
    env_labels  {env: display text} for the legend and direct labels.
    annotate    iterable of (env, t): draw a boxed note at that point
                spelling out the per-arm taus behind R_t (a ratio above 1
                is then legible as its denominator, not as faster
                relearning). An (env, t) with no record raises ValueError.
    notes       extra footnote lines (e.g. the R_t definition).
    xlabel      x-axis label override (e.g. "checkpoint index").

    Null R_t (recovery returned None with a reason, e.g.
    denominator_too_small) is an ANNOTATED GAP at that t: an x at the axis
    floor with the reason, and the line broken — never a zero. The
    pre-committed checkpoints appear as x ticks even if some have no data.
    """
    plt = _plt()
    fig, ax = plt.subplots()
    env_labels = dict(env_labels or {})

    def display(env):
        return str(env_labels.get(env, env))

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
                label=display(env) if run_idx == 0 else None,
                zorder=3,
            )
        # Direct label at the last measurable point (relief for low-contrast
        # hues) only when no legend will be drawn: with several environments
        # the labels collide wherever the lines end at the same value.
        if measurable and len(envs) == 1:
            last = env_records[measurable[-1]]
            ax.annotate(
                display(env), xy=(last["checkpoint_step"], last["R_t"]),
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

    annotations = []
    for env, t_step in annotate:
        matches = [r for r in records
                   if r.get("env") == env and r.get("checkpoint_step") == t_step]
        if len(matches) != 1:
            raise ValueError("no recovery record for env %r at checkpoint %r"
                             % (env, t_step))
        record = matches[0]
        text = _rt_annotation_text(display(env), t_step, record)
        y = record["R_t"] if record.get("R_t") is not None else 0.0
        ax.annotate(
            text, xy=(t_step, y), xytext=(10, -4), textcoords="offset points",
            ha="left", va="top", fontsize=8.5, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92),
            zorder=6,
        )
        annotations.append((str(env), t_step, text))

    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "full recovery (R=1)", xy=(0.0, 1.0), xycoords=("axes fraction", "data"),
        xytext=(4, 4), textcoords="offset points", ha="left",
        fontsize=8.5, color=TEXT_SECONDARY,
    )
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)

    ticks = sorted(
        set(checkpoints)
        | set(r.get("checkpoint_step") for r in records
              if r.get("checkpoint_step") is not None)
    )
    ax.set_xticks(ticks)
    ax.set_xlabel("fine-tuning checkpoint $t$ (steps)" if xlabel is None else xlabel)
    ax.set_ylabel(r"$R_t$ (fraction of deception gap recovered)")
    resolved_title = _resolve_title(title, "Recovery of the deception gap after re-fine-tuning")
    ax.set_title(resolved_title)
    if len(envs) > 1:
        ax.legend(loc="best")

    footnote = []
    if gaps:
        footnote.append("x at axis: R_t not computable there (reason shown), not zero")
    footnote.extend(str(n) for n in notes)
    _footnote(fig, footnote, fontsize=8.5)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "envs": [str(e) for e in envs],
        "checkpoints_shown": ticks,
        "gaps": gaps,
        "annotations": annotations,
        "title": resolved_title,
    }


# ---------------------------------------------------------------------------
# The δ-curve (edit relocation)
# ---------------------------------------------------------------------------


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_delta(curve_recovered, curve_edited, out_base,
                 label_recovered="recovered checkpoint",
                 label_edited="just-edited", title=None, dpi=300):
    """The δ-curve: per-layer A_l difference between two layer curves.

        δ_l = A_l(recovered E,D checkpoint) − A_l(just-edited M_E)

    curve_recovered / curve_edited   two figures.layer_curve outputs: the
                                     sweep of the recovered E,D checkpoint and
                                     the sweep of the just-edited M_E it was
                                     continued from.

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
    edi = by_layer(curve_edited)
    all_layers = sorted(
        set(rec) | set(edi),
        key=lambda v: (0, v, "") if isinstance(v, int) else (1, 0, str(v)),
    )

    xs, tick_labels = _positions(all_layers)
    deltas = []
    gaps = []          # (layer, reason)
    gap_marks = []
    for x, layer in zip(xs, all_layers):
        p_rec = rec.get(layer)
        p_edi = edi.get(layer)
        a_rec = p_rec.get("A_l") if p_rec else None
        a_edi = p_edi.get("A_l") if p_edi else None
        if a_rec is None or a_edi is None:
            rec_why = (p_rec or {}).get("reason") or (
                "absent" if p_rec is None else "A_l_none"
            )
            edi_why = (p_edi or {}).get("reason") or (
                "absent" if p_edi is None else "A_l_none"
            )
            if a_rec is None and a_edi is None and rec_why == edi_why:
                reason = "both curves: %s" % rec_why
            else:
                missing = []
                if a_rec is None:
                    missing.append("%s: %s" % (label_recovered, rec_why))
                if a_edi is None:
                    missing.append("%s: %s" % (label_edited, edi_why))
                reason = "; ".join(missing)
            gaps.append((layer, reason))
            gap_marks.append((x, reason))
            deltas.append(None)
        else:
            deltas.append(metrics.relocation_delta_value(a_rec, a_edi))

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

    if tick_labels is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("bypassed layer $l$")
    ax.set_ylabel(r"$\delta_l$  (%s $-$ %s $A_l$)" % (label_recovered, label_edited))
    ax.set_title(_resolve_title(title, "Relocation: change in per-layer bypass effect after recovery"))

    notes = []
    if gaps:
        notes.append("x at axis: δ not computable there (side and reason shown)")
    _footnote(fig, notes)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "layers": all_layers,
        "n_deltas": len(measurable),
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# tau bars (Figure 5)
# ---------------------------------------------------------------------------


def render_tau_bars(records, out_base, title=None, dpi=300, notes=()):
    """Incentive-sensitivity gap tau per model and arm, with CIs.

    records   one dict per (model, arm): metrics.tau_with_ci-shaped — tau,
              tau_ci_low, tau_ci_high (extras ignored) — plus "model" (group
              label, e.g. the model family or the environment) and "label"
              (the checkpoint: "M_0", "M_D", "M_E-<window>"; labels outside
              ARM_ORDER are ordered after it).

    A record with tau None is an annotated gap in its slot (x + reason),
    never a zero-height bar. `notes` are extra footnote lines (e.g. which
    run each bar comes from).
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
    extra_labels = [label for label in labels if label not in ARM_COLORS]
    if len(extra_labels) > len(extra_colors):
        raise ValueError(
            "tau bars: %d labels outside ARM_COLORS (%s) but only %d spare "
            "series colors; a shared color would misread"
            % (len(extra_labels), ", ".join(str(l) for l in extra_labels),
               len(extra_colors))
        )
    colors = {}

    for j, label in enumerate(labels):
        color = ARM_COLORS.get(label)
        if color is None:
            color = extra_colors[extra_labels.index(label)]
        colors[str(label)] = color
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
    resolved_title = _resolve_title(title, "Incentive-sensitivity gap by model and arm")
    ax.set_title(resolved_title)
    # Headroom so the pinned top-center legend clears the tallest bar+CI+label.
    tops = [
        r["tau_ci_high"] if r.get("tau_ci_high") is not None else r["tau"]
        for r in records if r.get("tau") is not None
    ]
    if tops:
        ax.set_ylim(top=max(max(tops), 0.0) * 1.3 or 1.0)
        ax.legend(loc="upper center", ncols=min(len(labels), 3))
    ax.grid(axis="x", visible=False)

    footnote = []
    if gaps:
        footnote.append("x at axis: tau not measured there (reason shown), not zero")
    footnote.extend(str(n) for n in notes)
    _footnote(fig, footnote, fontsize=8.5)

    return {
        "paths": _save(fig, out_base, dpi=dpi),
        "models": [str(m) for m in models],
        "labels": [str(l) for l in labels],
        "colors": colors,
        "gaps": gaps,
        "notes": [str(n) for n in notes],
        "title": resolved_title,
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
    voided layer, and one partial-overlap layer."""
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
            # High invalid rate in the incentive condition: voided.
            p = _synthetic_point(
                layer, a_l, competence=0.88, invalid_inc=0.35, invalid_ctl=0.06,
            )
            statuses[layer] = "VOIDED: invalid_inc=0.35"
        elif layer == 24:
            # A sweep job relaunched with a different --n: partial overlap.
            p = _synthetic_point(layer, a_l, reason="partial_overlap", paired=False)
            statuses[layer] = "CLEARS BOUNDS" if a_l >= 0.15 else "UNMEASURABLE: partial_overlap"
        else:
            drop = max(0.0, rng.gauss(0.01, 0.01)) + (0.04 if 10 <= layer <= 14 else 0.0)
            p = _synthetic_point(layer, a_l, competence=round(0.92 - drop, 4))
            viable = a_l >= 0.15 and p["A_l_ci_low"] is not None and p["A_l_ci_low"] > 0
            statuses[layer] = "CLEARS BOUNDS" if viable else "BELOW EFFECT FLOOR"
        points.append(p)
    return points, statuses


def synthetic_rt(seed=0):
    """recovery()-shaped records for two environments at the pre-registered
    subset, with one null-with-reason point (the annotated-gap path). Each
    record carries the four arm taus and the `arms` list exactly as
    recovery_report.py --emit-records writes them."""
    rng = random.Random(seed)
    records = []
    arms = ["E,D", "E,C", "I,D", "I,C"]
    curves = {
        "negotiation": {8: 0.22, 70: 0.61, 281: 0.86},
        "insider_trading": {8: None, 70: 0.44, 281: 0.69},
    }
    for env, by_t in curves.items():
        for t in CHECKPOINT_STEPS:
            if by_t[t] is None:
                records.append({
                    "env": env, "checkpoint_step": t, "arms": list(arms),
                    "tau_ED": 0.02, "tau_EC": 0.01, "tau_ID": 0.05, "tau_IC": 0.01,
                    "R_t": None, "R_t_ci_low": None, "R_t_ci_high": None,
                    "reason": "denominator_too_small",
                })
                continue
            r = by_t[t] + rng.uniform(-0.03, 0.03)
            records.append({
                "env": env, "checkpoint_step": t, "arms": list(arms),
                "tau_ED": 0.3, "tau_EC": 0.05, "tau_ID": 0.5, "tau_IC": 0.05,
                "R_t": round(r, 4),
                "R_t_ci_low": round(r - 0.08, 4),
                "R_t_ci_high": round(r + 0.08, 4),
                "reason": None,
            })
    return records


def synthetic_delta(seed=0):
    """(curve_recovered, curve_edited) for the δ-curve.

    The just-edited M_E has the effect at its edited window's centre layer
    (12) knocked out; the recovered E,D checkpoint shows effect returning
    NEAR the window but not at it (relocation).
    """
    rng = random.Random(seed)
    centre = 12
    edited, _ = synthetic_layer_curve(seed=seed)
    recovered = []
    for p in edited:
        q = dict(p)
        layer = q["bypassed_layer"]
        if q["A_l"] is not None:
            shift = 0.30 * math.exp(-((layer - (centre + 4)) ** 2) / 6.0)
            drop = -0.9 * q["A_l"] if layer == centre else 0.0
            a = round(q["A_l"] + shift + drop + rng.uniform(-0.02, 0.02), 4)
            q["A_l"] = a
            q["A_l_ci_low"] = round(a - 0.06, 4)
            q["A_l_ci_high"] = round(a + 0.06, 4)
        recovered.append(q)
    return recovered, edited


def synthetic_tau_bars(seed=0):
    """tau_with_ci-shaped records per model (the two families) and checkpoint
    (M_0, M_D, the edited M_E), with one gap (a tau that was not computable)
    so the annotated-gap path is exercised."""
    rng = random.Random(seed)
    records = []
    for model in ("Qwen2.5-7B", "Llama-3.1-8B"):
        for label, center in (("M_0", 0.06), ("M_D", 0.52), ("M_E", 0.03)):
            if model == "Llama-3.1-8B" and label == "M_E":
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
# The edit heatmap (Figure 1): bypass layer x edited checkpoint
# ---------------------------------------------------------------------------


def render_edit_heatmap(data, out_base, title=None, dpi=300, edit_windows=None):
    """Two aligned panels: clean-row D_incentive and truncation rate.

    Cell states, each drawn distinctly and each reported in the metadata so
    nothing a cell hides goes unreported:
      measured   colour = clean-row D_incentive (viridis)
      voided     incentive invalid rate above the ruling bound: GREY with a
                 red ring; the rate a voided cell would have shown is never
                 drawn (it is still in the cell dict)
      zero_clean attempted but not one usable (valid, untruncated) incentive
                 response: a small dot on top of whatever state it is in
      missing    never attempted (no incentive rows): hatched, both panels
    edit_windows {key: layers} draws a dashed outline over that row's edited
    layers (min..max); an unknown key is refused by name. The truncation
    panel keeps every attempted cell's rate: truncation is a measured
    quantity even where deception is not.
    """
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    plt = _plt()
    keys = data["keys"]
    n_layers = data["n_layers"]
    cells = data["cells"]

    spans = {}
    for key, layers in dict(edit_windows or {}).items():
        if key not in keys:
            raise ValueError("edit window for unknown key %r (rows: %s)" % (key, keys))
        layers = [int(l) for l in layers]
        if not layers:
            raise ValueError("edit window for %r is empty" % key)
        spans[key] = [min(layers), max(layers)]

    D = np.full((len(keys), n_layers), np.nan)
    T = np.full((len(keys), n_layers), np.nan)
    voided, missing, no_clean, zero_clean = [], [], [], []
    n_rows = 0
    for i, key in enumerate(keys):
        for layer in range(n_layers):
            cell = cells[key][layer]
            if cell["status"] == "missing":
                missing.append((key, layer))
                continue
            n_rows = max(n_rows, int(cell.get("n") or 0))
            T[i, layer] = cell["trunc_rate"]
            if not cell.get("n_clean"):
                zero_clean.append((key, layer))
            if cell["status"] == "voided_validity":
                voided.append((key, layer))
                continue
            if cell["clean_d_incentive"] is None:
                no_clean.append((key, layer))
                continue
            D[i, layer] = cell["clean_d_incentive"]

    fig, (ax_d, ax_t) = plt.subplots(
        2, 1, figsize=(6.4, 4.2),
        sharex=True,
    )
    cmap_d = plt.get_cmap("viridis").copy()
    cmap_d.set_bad("0.85")
    cmap_t = plt.get_cmap("magma").copy()
    cmap_t.set_bad("0.85")

    im_d = ax_d.imshow(D, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap_d)
    im_t = ax_t.imshow(T, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap_t)

    def box(ax, key, layer, **kwargs):
        ax.add_patch(Rectangle((layer - 0.5, keys.index(key) - 0.5), 1, 1, **kwargs))

    for (key, layer) in missing:
        for ax in (ax_d, ax_t):
            box(ax, key, layer, facecolor="0.85", edgecolor="0.55",
                hatch="////", linewidth=0, zorder=3)
    for (key, layer) in voided:
        box(ax_d, key, layer, fill=False, edgecolor="crimson", linewidth=1.4, zorder=4)
    for (key, layer) in zero_clean:
        ax_d.plot(layer, keys.index(key), marker="o", markersize=3.5,
                  color=TEXT_SECONDARY, linestyle="none", zorder=5)
    for key, (lo, hi) in spans.items():
        ax_d.add_patch(Rectangle(
            (lo - 0.5, keys.index(key) - 0.5), hi - lo + 1, 1, fill=False,
            edgecolor="white", linewidth=1.3, linestyle=(0, (3, 2)), zorder=6,
        ))

    for ax, im, label in ((ax_d, im_d, "clean-row deception rate"),
                          (ax_t, im_t, "truncation rate")):
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label(label)
    ax_t.set_xlabel("bypassed layer")
    ax_t.set_xticks(range(0, n_layers, 2))
    resolved_title = _resolve_title(
        title, "Deception under single-layer bypass, per edited checkpoint")
    ax_d.set_title(resolved_title)

    handles = [
        Patch(facecolor="0.85", edgecolor="crimson", linewidth=1.4),
        Line2D([], [], marker="o", color=TEXT_SECONDARY, linestyle="none", markersize=3.5),
    ]
    legend = [
        "grey + red outline: unmeasurable (invalid rate > %.2f)" % data["invalid_max"],
        "dot: attempted, zero usable responses",
    ]
    if missing:
        handles.append(Patch(facecolor="0.85", edgecolor="0.55", hatch="////"))
        legend.append("hatched: never attempted")
    if spans:
        handles.append(Patch(fill=False, edgecolor=TEXT, linewidth=1.3, linestyle=(0, (3, 2))))
        legend.append("dashed outline: layers edited in that checkpoint")
    fig.legend(handles, legend, loc="upper center", bbox_to_anchor=(0.5, -0.005),
               ncol=1, fontsize=8.5, frameon=False)
    _footnote(fig, [
        "unmeasurable: %d cells; attempted with zero usable responses: %d; "
        "never attempted: %d" % (len(voided), len(zero_clean), len(missing)),
        "Rates exclude truncated/invalid responses; %d incentive attempts per cell." % n_rows,
    ], fontsize=8.5, y=-0.19 if missing else -0.15)

    paths = _save(fig, out_base, dpi=dpi)
    return {
        "paths": paths,
        "keys": keys,
        "n_layers": n_layers,
        "voided": voided,
        "missing": missing,
        "no_clean_rows": no_clean,
        "zero_clean": zero_clean,
        "edit_windows": spans,
        "legend": legend,
        "title": resolved_title,
    }


def synthetic_edit_heatmap(seed=0):
    """Plausible fake heatmap data for the --synthetic dry run: measured
    cells, voided boundary layers (layer 0 with zero usable responses, the
    last layer with two), and one never-attempted cell (l21, layer 5) so
    every legend state is exercised."""
    rng = random.Random(seed)
    keys = ["l07", "l13", "l10", "l21"]
    n_layers = 28
    cells = {}
    for key in keys:
        per_layer = {}
        for layer in range(n_layers):
            if layer == 0:
                per_layer[layer] = {"status": "voided_validity", "n": 100,
                                    "n_clean": 0, "clean_d_incentive": None,
                                    "trunc_rate": 1.0, "invalid_rate": 1.0}
                continue
            if layer == n_layers - 1:
                per_layer[layer] = {"status": "voided_validity", "n": 100,
                                    "n_clean": 2, "clean_d_incentive": 0.0,
                                    "trunc_rate": 0.95, "invalid_rate": 0.98}
                continue
            if key == "l21" and layer == 5:
                per_layer[layer] = {"status": "missing", "n": 0, "n_clean": 0,
                                    "clean_d_incentive": None,
                                    "trunc_rate": None, "invalid_rate": None}
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
# Probe-transfer AUROC per layer, per checkpoint (Figures 3-4)
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
    resolved_title = _resolve_title(title, "Instructed-pairs probe transfer to strategic "
                                           "deception, per checkpoint")
    ax.set_title(resolved_title)
    if null_layers:
        _footnote(fig, ["null AUROC (degenerate bootstrap or structural): "
                        + "; ".join("%s: %s" % (k, v)
                                    for k, v in null_layers.items())])
    paths = _save(fig, out_base, dpi=dpi)
    return {"paths": paths, "keys": [k for k, _ in curves],
            "null_layers": null_layers, "title": resolved_title}


def synthetic_probe_curves(seed=0):
    """Three AUROC-vs-layer curves (M_0, M_D, M_E) with CI bands and one null
    layer each, so the gap path is exercised."""
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
