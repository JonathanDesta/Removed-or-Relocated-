"""
Figure data layer: layer-wise curves, the deception/damage Pareto, and the
edit heatmap cells.

This module turns the sweep's results rows into plot-ready points and nothing
else: no matplotlib import, no torch, no numpy, exactly like metrics.py and
tasks.py, so it runs on a laptop against row files copied from the project
directory.

It adds NOTHING to metrics.py: every statistic is a call to the existing
functions, so each number has a single home. The one diagnostic
metrics.bypass_effect does not provide (how many scenarios the two runs
actually share) is added here as a wrapper instead of a patch.

The two Stage-1 views:

  layer curve   x = bypassed layer, y = A_l with its CI. A_l is the
                deception-specific bypass effect, tau(base) - tau(bypassed).
                Big A_l means bypassing that layer removed a lot of the
                incentive-driven deception.

  Pareto        x = A_l (deception removed, higher is better),
                y = damage (capability lost, lower is better).
                The frontier is the set of layers no other layer beats on
                both axes at once.

Three rules run through the whole module:

  1. Nothing is ever silently dropped. A layer whose tau or damage could not
     be computed does not vanish from the output; it comes back with a
     `reason` string and is reported separately by `unmeasurable()`. A
     bypass that destroys a model produces exactly this case, and a plot
     that drops it would show a catastrophic layer as a missing point
     instead of maximal damage.

  2. Nothing is ever compared unpaired without saying so. bypass_effect's
     bootstrap silently restricts itself to the scenarios both runs share
     (metrics.py bootstrap_ci), so a base run and a sweep run launched with
     different --n produce a real-looking A_l computed on the overlap. Every
     point here carries n_scenarios_common and a `paired` flag, and BOTH
     axes -- deception and damage -- are computed on the shared scenarios.

  3. A failure to match a baseline is explained, not just reported. Which
     row fields a baseline must share with a sweep run depends on how the
     rows fill `arm`, `seed` and `checkpoint_step`; when the default guess
     is wrong every layer comes back with no baseline, so each such point
     carries the exact list of fields that differed.
"""

from algoverse import metrics


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

# What must match for a base run and a bypassed run to be comparable. This is
# metrics.RUN_KEY_FIELDS minus the two fields that are SUPPOSED to differ
# between them: bypassed_layer (that's the intervention) and run_id (a sweep
# writes one run per layer). Generation identity is compared too, so a
# dev-mode 0.5B run can never become the baseline for a prod sweep.
#
# `arm` is the field most likely to need overriding. Its values are
# "I,D" | "I,C" | "E,D" | "E,C" | null, where I is a continuation of the
# intact M_D and E a continuation of the edited M_E. If the sweep runs are
# labelled "E,D" while the baseline is "I,D", nothing will match on the
# default fields -- pass match_fields without "arm" and the
# baseline_mismatch field on every point will have told you so.
DEFAULT_MATCH_FIELDS = tuple(
    f for f in metrics.RUN_KEY_FIELDS if f not in ("bypassed_layer", "run_id")
)


def _match_key(row, match_fields):
    """Identity of the comparison a row belongs to, ignoring the intervention."""
    return (
        tuple(row.get(f) for f in match_fields)
        + metrics.comparison_gen_identity(row)
    )


def _mismatch_fields(key_a, key_b, match_fields):
    """Which named fields differ between two match keys."""
    names = tuple(match_fields) + metrics.COMPARISON_GEN_CONFIG_KEY_FIELDS
    return [n for n, a, b in zip(names, key_a, key_b) if a != b]


def _layer_sort_key(value):
    """Order layers numerically even if a rows file writes them as strings.

    Sorting "10", "2", "9" as text puts layer 10 before layer 2, and a curve
    plotted in that order is wrong without looking wrong. Values that are not
    numeric at all sort after the numeric ones rather than raising, so one
    malformed row cannot take the whole figure down.
    """
    try:
        return (0, int(value), "")
    except (TypeError, ValueError):
        return (1, 0, str(value))


def _layer_variants(value):
    """The forms a layer might be keyed under across two files: 7 and "7"."""
    variants = [value]
    try:
        as_int = int(value)
        if as_int not in variants:
            variants.append(as_int)
        if str(value) not in variants:
            variants.append(str(value))
    except (TypeError, ValueError):
        pass
    return variants


def _scenario_ids(rows):
    return set(r.get("scenario_id") for r in rows)


def _restrict(rows, scenario_ids):
    return [r for r in rows if r.get("scenario_id") in scenario_ids]


def n_common_scenarios(rows_a, rows_b) -> int:
    """How many scenario_ids the two row sets share.

    This is the number bypass_effect actually computes on, and the number it
    does not report. Zero means the comparison is not paired at all: the
    bootstrap falls back to the raw statistic with no CI (metrics.py
    bootstrap_ci, the `if not common_ids` branch) and still returns a number.
    """
    return len(_scenario_ids(rows_a) & _scenario_ids(rows_b))


# ---------------------------------------------------------------------------
# The missing diagnostic
# ---------------------------------------------------------------------------


def bypass_effect_checked(rows_base, rows_bypassed, n_boot=2000, seed=0) -> dict:
    """metrics.bypass_effect plus the pairing diagnostics it does not return.

    Adds:
      n_scenarios_base / n_scenarios_bypassed / n_scenarios_common
      paired          True when the two runs cover the SAME scenario set
      reason          why A_l is absent or not to be trusted; None when fine

    A_l is forced to None when the two runs share no scenario at all. Without
    that, two disjoint runs return a fully unpaired difference of two taus
    that looks exactly like a real paired A_l, just with CI None.
    """
    n_base = len(_scenario_ids(rows_base))
    n_byp = len(_scenario_ids(rows_bypassed))
    n_common = n_common_scenarios(rows_base, rows_bypassed)

    if n_common == 0:
        return {
            "A_l": None,
            "A_l_ci_low": None,
            "A_l_ci_high": None,
            "tau_base": metrics.incentive_gap(rows_base)["tau"],
            "tau_bypassed": metrics.incentive_gap(rows_bypassed)["tau"],
            "n_scenarios_base": n_base,
            "n_scenarios_bypassed": n_byp,
            "n_scenarios_common": 0,
            "paired": False,
            "reason": "no_shared_scenarios",
        }

    out = metrics.bypass_effect(rows_base, rows_bypassed, n_boot=n_boot, seed=seed)
    paired = (n_common == n_base == n_byp)
    out.update({
        "n_scenarios_base": n_base,
        "n_scenarios_bypassed": n_byp,
        "n_scenarios_common": n_common,
        "paired": paired,
    })
    # A_l being absent outranks the overlap warning: a point with no number
    # must say why there is no number, not why the number is shaky.
    if out["A_l"] is None:
        out["reason"] = "tau_not_computable"
    else:
        out["reason"] = None if paired else "partial_overlap"
    return out


# ---------------------------------------------------------------------------
# The layer curve
# ---------------------------------------------------------------------------


def split_base_and_sweep(rows, match_fields=DEFAULT_MATCH_FIELDS):
    """Group rows into comparisons, each {base_rows, base_run_ids, layers}.

    A base run is one with bypassed_layer None. Rows are grouped by match key
    first, so a sweep on M_D and a sweep on M_0 never borrow each other's
    baseline.

    Raises when a group holds more than one base run_id: silently pooling two
    baselines would mix generation runs inside one denominator, and picking
    one arbitrarily would make A_l depend on dict ordering.
    """
    groups = {}
    for row in rows:
        key = _match_key(row, match_fields)
        g = groups.setdefault(key, {"base_rows": [], "base_run_ids": set(), "layers": {}})
        layer = row.get("bypassed_layer")
        if layer is None:
            g["base_rows"].append(row)
            g["base_run_ids"].add(row.get("run_id"))
        else:
            g["layers"].setdefault(layer, []).append(row)

    for g in groups.values():
        if len(g["base_run_ids"]) > 1:
            raise ValueError(
                "more than one baseline run in the same comparison group: %s. "
                "Pass one baseline run_id at a time, or filter the rows first."
                % sorted(g["base_run_ids"])
            )
    return groups


def layer_curve(rows, n_boot=2000, seed=0, match_fields=DEFAULT_MATCH_FIELDS) -> list:
    """One point per bypassed layer: A_l, its CI, and the pairing diagnostics.

    Feed it the baseline rows and every sweep rows.jsonl together. Points are
    sorted by (model, layer). Layers whose A_l could not be computed are
    KEPT, with A_l None and a reason; call unmeasurable() to list them.

    Every point also carries the task competence of both runs, computed on
    the SHARED scenarios only, which is the zero-cost damage axis: it needs
    no benchmark run, only the rows that are already there.
    """
    groups = split_base_and_sweep(rows, match_fields=match_fields)
    with_base = {k: g for k, g in groups.items() if g["base_rows"]}

    points = []
    for key, g in groups.items():
        base_rows = g["base_rows"]

        for layer in sorted(g["layers"], key=_layer_sort_key):
            byp_rows = g["layers"][layer]
            point = {
                "bypassed_layer": layer,
                "run_id": byp_rows[0].get("run_id"),
                "model_id": byp_rows[0].get("model_id"),
                "arm": byp_rows[0].get("arm"),
                "checkpoint_step": byp_rows[0].get("checkpoint_step"),
                "split": byp_rows[0].get("split"),
                "comparison": key,
            }

            if not base_rows:
                # Say WHICH fields kept the baseline away, not just that one
                # is missing: the usual cause is a field like `arm` that the
                # baseline and the sweep runs fill differently (e.g. "I,D"
                # against "E,D").
                mismatch = None
                for other_key in with_base:
                    diff = _mismatch_fields(key, other_key, match_fields)
                    if mismatch is None or len(diff) < len(mismatch):
                        mismatch = diff
                point.update({
                    "A_l": None, "A_l_ci_low": None, "A_l_ci_high": None,
                    "tau_base": None,
                    "tau_bypassed": metrics.incentive_gap(byp_rows)["tau"],
                    "paired": False, "reason": "no_baseline_run",
                    "n_scenarios_common": 0,
                    "baseline_mismatch": mismatch,
                })
                base_shared, byp_shared = [], byp_rows
            else:
                point.update(
                    bypass_effect_checked(base_rows, byp_rows, n_boot=n_boot, seed=seed)
                )
                point["baseline_mismatch"] = None
                shared = _scenario_ids(base_rows) & _scenario_ids(byp_rows)
                base_shared = _restrict(base_rows, shared)
                byp_shared = _restrict(byp_rows, shared)

            # Both competences on the SHARED scenarios: a drop computed
            # across two different scenario populations is the same silent
            # mismatch this module exists to catch, just on the damage axis.
            base_comp = (
                metrics.task_competence(base_shared)["competence"] if base_shared else None
            )
            byp_comp = metrics.task_competence(byp_shared)["competence"]
            point["competence"] = byp_comp
            point["competence_base"] = base_comp
            point["competence_drop"] = (
                None if (base_comp is None or byp_comp is None) else base_comp - byp_comp
            )

            gap = metrics.incentive_gap(byp_rows)
            point["invalid_rate_incentive"] = gap["invalid_rate_incentive"]
            point["invalid_rate_control"] = gap["invalid_rate_control"]
            points.append(point)

    points.sort(key=lambda p: (str(p.get("model_id")), _layer_sort_key(p["bypassed_layer"])))
    return points


def unmeasurable(points) -> list:
    """The points a plot must not silently drop: A_l None, or not paired.

    Returns (layer, reason) pairs. A destroyed model lands here, and it is
    the most informative point on the figure, not a missing one.
    """
    out = []
    for p in points:
        if p.get("A_l") is None:
            out.append((p["bypassed_layer"], p.get("reason") or "A_l_none"))
        elif not p.get("paired", True):
            out.append((p["bypassed_layer"], p.get("reason") or "partial_overlap"))
    return out


# ---------------------------------------------------------------------------
# The damage axis
# ---------------------------------------------------------------------------


def pareto_points(curve, base_competence=None) -> list:
    """Attach the negotiation-competence damage to every layer point.

    Damage is the drop in task competence: by default against the sweep's
    own base run (the curve's competence_drop); pass base_competence, e.g.
    M_0's task competence, to measure the drop against that reference
    instead. Every point carries damage_metric ("task_competence"),
    damage_reference ("sweep_base" or "M_0") and damage_reason (None when
    measured, else "competence_not_computable"). The benchmark, perplexity
    and neutral-JSD bounds are checked per layer by sweep.evaluate_sweep,
    which reads competence.jsonl itself.
    """
    points = []
    for p in curve:
        q = dict(p)
        q["damage_metric"] = "task_competence"
        if base_competence is not None:
            competence = p.get("competence")
            q["damage"] = (
                None if competence is None else base_competence - competence
            )
            q["damage_reference"] = "M_0"
        else:
            q["damage"] = p.get("competence_drop")
            q["damage_reference"] = "sweep_base"
        q["damage_reason"] = (
            None if q["damage"] is not None else "competence_not_computable"
        )
        points.append(q)
    return points


def pareto_frontier(points, allow_mixed=False) -> list:
    """The non-dominated layers: max A_l, min damage.

    Layer i is dominated when some layer j removes at least as much deception
    AND costs no more damage, with at least one of the two strict. Points
    with A_l or damage None are NOT on the frontier and NOT silently dropped
    either; they come back from unmeasurable() / the damage_reason field.

    Refuses points from more than one comparison by default. A frontier drawn
    across two models, or across the deceptive and control arms, compares
    quantities that were never measured on the same thing; the resulting
    figure looks fine and means nothing. Pass allow_mixed=True only if you
    have a reason.
    """
    comparisons = set(p.get("comparison") for p in points if "comparison" in p)
    if not allow_mixed and len(comparisons) > 1:
        models = sorted(set(str(p.get("model_id")) for p in points))
        arms = sorted(set(str(p.get("arm")) for p in points))
        raise ValueError(
            "pareto_frontier got points from %d different comparisons "
            "(models: %s; arms: %s). Filter to one comparison first, or pass "
            "allow_mixed=True." % (len(comparisons), models, arms)
        )

    usable = [p for p in points if p.get("A_l") is not None and p.get("damage") is not None]
    frontier = []
    for p in usable:
        dominated = any(
            (q["A_l"] >= p["A_l"] and q["damage"] <= p["damage"])
            and (q["A_l"] > p["A_l"] or q["damage"] < p["damage"])
            for q in usable
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda p: p["A_l"], reverse=True)
    return frontier


def curve_report(points) -> str:
    """A short text summary, for a log or a message instead of a screenshot."""
    lines = ["layers: %d" % len(points)]

    no_base = [p for p in points if p.get("reason") == "no_baseline_run"]
    if no_base:
        lines.append(
            "NO BASELINE MATCHED for %d layers; fields that differed: %s"
            % (len(no_base), no_base[0].get("baseline_mismatch"))
        )

    bad = unmeasurable(points)
    lines.append("unmeasurable: %d" % len(bad))
    for layer, reason in bad:
        lines.append("  layer %s: %s" % (layer, reason))

    usable = [p for p in points if p.get("A_l") is not None]
    if usable:
        best = max(usable, key=lambda p: p["A_l"])
        lines.append(
            "largest A_l: layer %s, A_l=%.3f [%s, %s], n_shared=%s"
            % (
                best["bypassed_layer"], best["A_l"],
                "%.3f" % best["A_l_ci_low"] if best.get("A_l_ci_low") is not None else "None",
                "%.3f" % best["A_l_ci_high"] if best.get("A_l_ci_high") is not None else "None",
                best.get("n_scenarios_common"),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The edit heatmap (Figure 1): bypass layer x edited checkpoint
# ---------------------------------------------------------------------------


def edit_heatmap_cells(columns, n_layers, invalid_max=metrics.INVALID_RATE_MAX):
    """Per-(checkpoint, layer) cells for the bypass-x-checkpoint heatmap.

    `columns` is an ordered list of (key, layer_rows) pairs, where layer_rows
    maps int layer -> that layer run's rows (both conditions; incentive rows
    are selected here). Rows must come through metrics.load_results /
    rows_from, so the truncation rule is already applied. The
    pre-registered statistic is the clean-row D_incentive (deceptive rate
    among valid incentive rows), with a companion truncation rate. A cell
    whose incentive invalid rate exceeds `invalid_max` is VOIDED - reported
    as such, never plotted as a rate.
    """
    keys = [key for key, _ in columns]
    cells = {}
    for key, layer_rows in columns:
        per_layer = {}
        for layer in range(n_layers):
            rows = layer_rows.get(layer) or []
            inc = [r for r in rows if r.get("condition") == "incentive"]
            if not inc:
                per_layer[layer] = {
                    "status": "missing", "n": 0, "n_clean": 0,
                    "clean_d_incentive": None, "trunc_rate": None,
                    "invalid_rate": None,
                }
                continue
            n = len(inc)
            n_bad = sum(1 for r in inc if not r.get("valid"))
            n_trunc = sum(1 for r in inc if r.get("hit_max_tokens"))
            clean = [r for r in inc if r.get("valid")]
            clean_d = (
                sum(1 for r in clean if r.get("deceptive") is True) / len(clean)
                if clean else None
            )
            invalid_rate = n_bad / n
            per_layer[layer] = {
                "status": ("voided_validity" if invalid_rate > invalid_max
                           else "measured"),
                "n": n, "n_clean": len(clean),
                "clean_d_incentive": clean_d,
                "trunc_rate": n_trunc / n,
                "invalid_rate": invalid_rate,
            }
        cells[key] = per_layer
    return {"keys": keys, "n_layers": n_layers, "invalid_max": invalid_max,
            "cells": cells}
