"""
Stage-1 sweep selection report: the disqualifier table, l*, and the verdict.

This module is the sweep's analogue of eval.gate1_report: pure analysis over
already-written results rows, importing nothing heavy (no torch, no numpy),
so it runs on a laptop against row files synced from Drive. It is a CONSUMER
of metrics.py and figures.py. A_l and its CI come from figures.layer_curve
(metrics.bypass_effect underneath: paired scenario bootstrap, n_boot=2000,
seed=0, alpha=0.05 per spec item 9), and the Pareto frontier comes from
figures.pareto_points / pareto_frontier / curve_report. Nothing statistical
is reimplemented here.

What this module adds is the ratified per-layer disqualifiers and the
selection rule (RESEARCH_SPEC items 15-17 plus the item 2-3 bounds):

  item 15   invalid rate <= 0.20 PER CONDITION, computed from the layer's
            own rows (invalid = valid falsy, rate among that condition's
            rows) -- figures.layer_curve already carries both rates.
  item 2    negotiation task-competence drop vs M_0 <= 0.05. The M_0
            competence value is an INPUT (the caller measured it at Gate 1);
            when it is not provided the check reads NOT EVALUATED, never a
            silent pass.
  items 2-3 benchmark parts: mmlu / gsm8k drops <= 0.05 and WikiText-2 ppl
            rise <= 2.0, layer-vs-intact-base from competence.jsonl-style
            records when provided (base records under the "base" key are the
            reference; without them a delta cannot exist, so the checks read
            NOT EVALUATED).
  item 16   mean per-token neutral JSD <= 0.25 nats, from the layer's
            records. JSD is already an intact-vs-bypassed quantity, so it
            needs no base record.
  item 17   A_l >= 0.15 AND the 95% scenario-bootstrap CI excludes zero
            (ci_low > 0). The spec requires reporting A_l*/tau(M_D)
            alongside, and this report does.

Every check evaluates to one of PASS / FAIL / NOT EVALUATED -- three states,
because "we never measured it" must never render as either of the other two.
Selection: among layers where every EVALUATED disqualifier passes and item 17
passes, l* = argmax A_l. If no layer qualifies, the verdict is the spec's
explicit stop condition: "no viable layer-level localization" (Stage 2 does
not run).

PENDING names (P-S1 / P-S2 of the sweep-driver plan, NOT yet human-ratified):
the metric names read from the per-layer records -- mmlu_acc,
gsm8k_exact_match, wikitext2_ppl, wikitext2_neutral_jsd -- are the plan's
PROPOSED contract additions. Until the human ratifies them, records under
other names are simply absent here and the affected checks show NOT
EVALUATED; this module never invents a fallback name.

Tripwire (plan D5, spec item 16's mandatory calibration clause): a
research-model verdict is REFUSED (ValueError) unless the caller passes the
recorded DEV-calibration confirm-or-revise decision as a non-empty string
(item16_decision); dev=True bypasses the tripwire for the DEV model only and
stamps every report line as not publishable, exactly like gate1_report.

Nothing is silently dropped: every requested layer appears in the table --
disqualified, unmeasurable (figures' A_l None with its reason), or literally
empty (zero rows). A layer that was REQUESTED but never given a rows input at
all is a caller error and raises, naming the layer.
"""

import os

from algoverse import figures, metrics


# Ratified bounds (RESEARCH_SPEC items 15-17 and 2-3; do not re-derive).
INVALID_RATE_MAX = 0.20      # item 15, per condition
COMPETENCE_DROP_MAX = 0.05   # item 2, negotiation task-competence vs M_0
BENCH_DROP_MAX = 0.05        # items 2-3, mmlu / gsm8k, layer vs intact base
PPL_RISE_MAX = 2.0           # item 3, WikiText-2 perplexity rise
NEUTRAL_JSD_MAX = 0.25       # item 16, nats
A_L_MIN = 0.15               # item 17

# PROPOSED metric names (P-S1 / P-S2, pending human ratification -- see the
# module docstring). The first three mirror the contract's existing
# competence.jsonl enum; the JSD name is the plan's proposed addition.
MMLU_METRIC = "mmlu_acc"
GSM8K_METRIC = "gsm8k_exact_match"
PPL_METRIC = "wikitext2_ppl"
JSD_METRIC = "wikitext2_neutral_jsd"

# The competence-records key holding the INTACT model's benchmark values,
# the reference every drop/rise is computed against (same-model deltas only,
# per item 3's wording).
BASE_KEY = "base"

CHECK_KEYS = (
    "i15_inc", "i15_ctl", "i2_negotiation", "i2_mmlu", "i2_gsm8k",
    "i3_ppl", "i16_jsd", "i17",
)


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------


def _rows(rows_or_path):
    """A row list, loading from disk when given a path instead of rows."""
    if isinstance(rows_or_path, (str, os.PathLike)):
        return metrics.load_rows(rows_or_path)
    return list(rows_or_path)


def _int_layer(value, context):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "%s: layer %r is not an integer layer index" % (context, value)
        )


def load_sweep_inputs(base, layer_inputs):
    """Load and validate the base run and the per-layer runs.

    base          rows list or path for the intact run (bypassed_layer None
                  on every row -- a bypassed row here would silently become
                  an extra sweep layer inside figures.split_base_and_sweep,
                  so it is refused instead).
    layer_inputs  {layer: rows-or-path}. Every row must carry the declared
                  layer in bypassed_layer: a mislabeled file would otherwise
                  fragment into a different curve point and the declared
                  layer would silently show no data.

    A layer with ZERO rows is kept (the table must show it); only shape
    violations raise.
    """
    base_rows = _rows(base)
    if not base_rows:
        raise ValueError("sweep base run has zero rows")
    for row in base_rows:
        if row.get("bypassed_layer") is not None:
            raise ValueError(
                "base run contains a row with bypassed_layer=%r; the base "
                "run must be the intact model (bypassed_layer null)"
                % row.get("bypassed_layer")
            )

    layer_rows = {}
    for key, source in layer_inputs.items():
        layer = _int_layer(key, "layer_inputs")
        if layer in layer_rows:
            raise ValueError("layer %d given twice in layer_inputs" % layer)
        rows = _rows(source)
        for row in rows:
            value = row.get("bypassed_layer")
            try:
                as_int = int(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "rows for layer %d contain bypassed_layer=%r; every row "
                    "must carry the declared layer" % (layer, value)
                )
            if as_int != layer:
                raise ValueError(
                    "rows for layer %d contain bypassed_layer=%r "
                    "(mislabeled input)" % (layer, value)
                )
        layer_rows[layer] = rows
    return base_rows, layer_rows


def load_competence_records(competence_inputs):
    """{layer-or-"base": {metric: {value, stderr, config}}} from records.

    competence_inputs maps a layer index (or the literal "base" for the
    intact model's reference values) to competence.jsonl-style records or a
    path to them. A duplicated metric within one key is malformed and
    refused, mirroring figures.index_competence.
    """
    index = {}
    for key, source in (competence_inputs or {}).items():
        norm = BASE_KEY if key == BASE_KEY else _int_layer(key, "competence_inputs")
        if norm in index:
            raise ValueError("competence records given twice for %r" % norm)
        entry = {}
        for row in _rows(source):
            metric = row.get("metric")
            if metric is None:
                continue
            if metric in entry:
                raise ValueError(
                    "duplicate competence metric %r for %r" % (metric, norm)
                )
            entry[metric] = {
                "value": row.get("value"),
                "stderr": row.get("stderr"),
                "config": row.get("config") or {},
            }
        index[norm] = entry
    return index


# ---------------------------------------------------------------------------
# Disqualifier evaluation (three-state: True / False / None)
# ---------------------------------------------------------------------------


def _leq_check(value, bound):
    """PASS/FAIL/NOT-EVALUATED for `value <= bound`; None value = no measurement."""
    return {
        "passed": None if value is None else value <= bound,
        "value": value,
        "bound": bound,
    }


def _bench_delta(base_bench, layer_bench, metric, layer, rise=False):
    """Layer-vs-base benchmark delta, or None when either side is missing.

    Configs must be comparable (metrics.comparable_metric_config, the same
    rule gate1 and pareto_points enforce): a drop between two differently
    configured measurements is refused, not averaged over.
    """
    base_entry = (base_bench or {}).get(metric)
    entry = (layer_bench or {}).get(metric)
    if base_entry is None or entry is None:
        return None
    if metrics.comparable_metric_config(base_entry) != metrics.comparable_metric_config(entry):
        raise ValueError(
            "%s competence config mismatch between base and layer %s"
            % (metric, layer)
        )
    base_value = base_entry.get("value")
    value = entry.get("value")
    if base_value is None or value is None:
        return None
    return (value - base_value) if rise else (base_value - value)


def layer_checks(point, layer, m0_competence, layer_bench, base_bench,
                 invalid_rate_max=INVALID_RATE_MAX,
                 competence_drop_max=COMPETENCE_DROP_MAX,
                 bench_drop_max=BENCH_DROP_MAX, ppl_rise_max=PPL_RISE_MAX,
                 neutral_jsd_max=NEUTRAL_JSD_MAX, a_l_min=A_L_MIN) -> dict:
    """All disqualifier checks for one layer, keyed per CHECK_KEYS.

    point is the layer's figures.layer_curve entry, or None for a layer with
    zero rows (every check then reads NOT EVALUATED). Each check is
    {"passed": True|False|None, "value", "bound"}; passed None always means
    "not evaluated", never a silent pass.
    """
    get = (lambda field: None) if point is None else point.get
    checks = {}
    checks["i15_inc"] = _leq_check(get("invalid_rate_incentive"), invalid_rate_max)
    checks["i15_ctl"] = _leq_check(get("invalid_rate_control"), invalid_rate_max)

    competence = get("competence")
    drop = (
        None if (m0_competence is None or competence is None)
        else m0_competence - competence
    )
    checks["i2_negotiation"] = _leq_check(drop, competence_drop_max)

    checks["i2_mmlu"] = _leq_check(
        _bench_delta(base_bench, layer_bench, MMLU_METRIC, layer), bench_drop_max
    )
    checks["i2_gsm8k"] = _leq_check(
        _bench_delta(base_bench, layer_bench, GSM8K_METRIC, layer), bench_drop_max
    )
    checks["i3_ppl"] = _leq_check(
        _bench_delta(base_bench, layer_bench, PPL_METRIC, layer, rise=True),
        ppl_rise_max,
    )

    jsd_entry = (layer_bench or {}).get(JSD_METRIC)
    checks["i16_jsd"] = _leq_check(
        None if jsd_entry is None else jsd_entry.get("value"), neutral_jsd_max
    )

    a_l = get("A_l")
    ci_low = get("A_l_ci_low")
    if a_l is None:
        passed = None  # unmeasurable, not failed: the status says why
    else:
        passed = a_l >= a_l_min and ci_low is not None and ci_low > 0
    checks["i17"] = {"passed": passed, "value": a_l, "bound": a_l_min}
    return checks


# ---------------------------------------------------------------------------
# The tripwire (plan D5 / spec item 16 calibration clause)
# ---------------------------------------------------------------------------


def _require_item16_decision(item16_decision, dev):
    if dev:
        return
    if not (isinstance(item16_decision, str) and item16_decision.strip()):
        raise ValueError(
            "item16_calibration_decision_missing: refusing to produce a "
            "research-model sweep verdict without the recorded "
            "DEV-calibration confirm-or-revise decision on the 0.25-nat "
            "neutral-JSD bound (RESEARCH_SPEC item 16; sweep-driver plan "
            "D5). Pass item16_decision=<reference to the recorded decision>, "
            "or dev=True for the DEV model only."
        )


# ---------------------------------------------------------------------------
# Evaluation (structured) and the report (rendered)
# ---------------------------------------------------------------------------


def evaluate_sweep(base, layer_inputs, requested_layers=None,
                   m0_competence=None, competence_inputs=None,
                   n_boot=2000, seed=0,
                   invalid_rate_max=INVALID_RATE_MAX,
                   competence_drop_max=COMPETENCE_DROP_MAX,
                   bench_drop_max=BENCH_DROP_MAX, ppl_rise_max=PPL_RISE_MAX,
                   neutral_jsd_max=NEUTRAL_JSD_MAX, a_l_min=A_L_MIN) -> dict:
    """The sweep's structured evaluation: entries, frontier, l*, verdict.

    base / layer_inputs   see load_sweep_inputs.
    requested_layers      the layers the report must cover; defaults to the
                          layer_inputs keys. A requested layer with no rows
                          input AT ALL raises, naming the layer -- a report
                          that quietly covered fewer layers than the sweep
                          demanded would hide an unfinished sweep.
    m0_competence         M_0's negotiation task-competence (float) or None
                          (= not provided, so item 2's negotiation check is
                          NOT EVALUATED).
    competence_inputs     see load_competence_records.

    Returns {"entries", "curve", "pareto_points", "frontier", "l_star",
    "verdict", "n_boot"}. Each entry: {"layer", "point", "checks", "failed",
    "not_evaluated", "status", "A_l"}. Statuses: VIABLE, DISQUALIFIED: <keys>,
    UNMEASURABLE: <figures reason>, NO ROWS.

    figures' refusals are surfaced, not relaxed: mixed match fields across
    runs raise here (one layer landing in two comparison groups, or
    pareto_frontier's own mixed-comparison refusal), while a base run that
    merely shares no scenarios with a layer run stays IN the table as
    UNMEASURABLE with figures' reason.
    """
    base_rows, layer_rows = load_sweep_inputs(base, layer_inputs)
    comp_index = load_competence_records(competence_inputs)

    if requested_layers is None:
        requested = sorted(layer_rows)
    else:
        requested = sorted(_int_layer(l, "requested_layers") for l in requested_layers)
        missing = [l for l in requested if l not in layer_rows]
        if missing:
            raise ValueError(
                "requested layer(s) %s have no rows input at all; every "
                "requested layer needs a rows file (an empty one still "
                "appears in the table, but a missing one is an unfinished "
                "sweep)" % ", ".join(str(l) for l in missing)
            )

    all_rows = list(base_rows)
    for layer in requested:
        all_rows.extend(layer_rows[layer])

    curve = figures.layer_curve(all_rows, n_boot=n_boot, seed=seed)
    by_layer = {}
    for point in curve:
        layer = _int_layer(point["bypassed_layer"], "layer_curve")
        if layer in by_layer:
            raise ValueError(
                "layer %d appears in more than one comparison group: match "
                "fields disagree across runs (figures refuses mixed "
                "comparisons; fix the inputs rather than relaxing this)"
                % layer
            )
        by_layer[layer] = point

    base_bench = comp_index.get(BASE_KEY)
    entries = []
    for layer in requested:
        point = by_layer.get(layer)
        checks = layer_checks(
            point, layer, m0_competence, comp_index.get(layer), base_bench,
            invalid_rate_max=invalid_rate_max,
            competence_drop_max=competence_drop_max,
            bench_drop_max=bench_drop_max, ppl_rise_max=ppl_rise_max,
            neutral_jsd_max=neutral_jsd_max, a_l_min=a_l_min,
        )
        failed = [k for k in CHECK_KEYS if checks[k]["passed"] is False]
        not_evaluated = [k for k in CHECK_KEYS if checks[k]["passed"] is None]
        if point is None:
            status = "NO ROWS"
        elif failed:
            status = "DISQUALIFIED: " + ",".join(failed)
        elif checks["i17"]["passed"] is True:
            status = "VIABLE"
        else:
            status = "UNMEASURABLE: %s" % (point.get("reason") or "A_l_none")
        entries.append({
            "layer": layer,
            "point": point,
            "checks": checks,
            "failed": failed,
            "not_evaluated": not_evaluated,
            "status": status,
            "A_l": None if point is None else point.get("A_l"),
        })

    pareto = figures.pareto_points(curve, damage_metric="task_competence")
    frontier = figures.pareto_frontier(pareto)

    viable = [e for e in entries if e["status"] == "VIABLE"]
    l_star = max(viable, key=lambda e: e["A_l"]) if viable else None
    if l_star is not None:
        verdict = "layer %d selected as l*" % l_star["layer"]
    else:
        verdict = ("no viable layer-level localization "
                   "(spec stop condition; Stage 2 does not run)")
    return {
        "entries": entries,
        "curve": curve,
        "pareto_points": pareto,
        "frontier": frontier,
        "l_star": l_star,
        "verdict": verdict,
        "n_boot": n_boot,
    }


def _fmt(value, digits=3):
    return "n/a" if value is None else ("%." + str(digits) + "f") % value


def _mark(check):
    if check["passed"] is None:
        return "n/e"
    return "pass" if check["passed"] else "FAIL"


def _ratio(a_l, tau_base):
    """A_l*/tau(M_D), None when tau(M_D) is absent or zero (no share exists)."""
    if a_l is None or tau_base is None or tau_base == 0:
        return None
    return a_l / tau_base


def sweep_report(base, layer_inputs, requested_layers=None, m0_competence=None,
                 competence_inputs=None, item16_decision=None, n_boot=2000,
                 seed=0, dev=False,
                 invalid_rate_max=INVALID_RATE_MAX,
                 competence_drop_max=COMPETENCE_DROP_MAX,
                 bench_drop_max=BENCH_DROP_MAX, ppl_rise_max=PPL_RISE_MAX,
                 neutral_jsd_max=NEUTRAL_JSD_MAX, a_l_min=A_L_MIN) -> str:
    """The Stage-1 selection report, printed and returned as markdown.

    The complete layer table (every requested layer, nothing silently
    dropped), figures' curve summary, the Pareto frontier, l* with
    A_l*/tau(M_D) (item 17 requires reporting that share), and the verdict.

    item16_decision is the D5 tripwire: without the recorded DEV-calibration
    confirm-or-revise decision string this REFUSES to produce a verdict
    (ValueError); dev=True bypasses it for the DEV model only and stamps
    every line, exactly like gate1_report's dev mode.
    """
    _require_item16_decision(item16_decision, dev)
    result = evaluate_sweep(
        base, layer_inputs, requested_layers=requested_layers,
        m0_competence=m0_competence, competence_inputs=competence_inputs,
        n_boot=n_boot, seed=seed,
        invalid_rate_max=invalid_rate_max,
        competence_drop_max=competence_drop_max,
        bench_drop_max=bench_drop_max, ppl_rise_max=ppl_rise_max,
        neutral_jsd_max=neutral_jsd_max, a_l_min=a_l_min,
    )

    lines = []
    lines.append("SWEEP SELECTION REPORT  (bootstrap n=%d)" % n_boot)
    lines.append(
        "item-16 calibration decision: %s"
        % (item16_decision if item16_decision else "NONE (dev mode)")
    )
    lines.append(
        "bounds: invalid<=%.2f/condition (i15), negotiation-competence "
        "drop<=%.2f (i2), benchmark drop<=%.2f (i2), ppl rise<=%.1f (i3), "
        "neutral JSD<=%.2f nats (i16), A_l>=%.2f with CI excluding 0 (i17)"
        % (invalid_rate_max, competence_drop_max, bench_drop_max,
           ppl_rise_max, neutral_jsd_max, a_l_min)
    )
    lines.append(
        "M_0 negotiation competence: %s" % _fmt(m0_competence)
    )
    lines.append("")
    lines.append(
        "| layer | A_l [95% CI] | inv inc/ctl (i15) | comp drop (i2) "
        "| mmlu drop (i2) | gsm8k drop (i2) | ppl rise (i3) | jsd (i16) "
        "| i17 | status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for entry in result["entries"]:
        checks = entry["checks"]
        point = entry["point"] or {}
        lines.append(
            "| %s | %s [%s, %s] | %s/%s %s/%s | %s %s | %s %s | %s %s "
            "| %s %s | %s %s | %s | %s |" % (
                entry["layer"],
                _fmt(entry["A_l"]),
                _fmt(point.get("A_l_ci_low")), _fmt(point.get("A_l_ci_high")),
                _fmt(checks["i15_inc"]["value"], 2),
                _fmt(checks["i15_ctl"]["value"], 2),
                _mark(checks["i15_inc"]), _mark(checks["i15_ctl"]),
                _fmt(checks["i2_negotiation"]["value"]), _mark(checks["i2_negotiation"]),
                _fmt(checks["i2_mmlu"]["value"]), _mark(checks["i2_mmlu"]),
                _fmt(checks["i2_gsm8k"]["value"]), _mark(checks["i2_gsm8k"]),
                _fmt(checks["i3_ppl"]["value"], 2), _mark(checks["i3_ppl"]),
                _fmt(checks["i16_jsd"]["value"]), _mark(checks["i16_jsd"]),
                _mark(checks["i17"]),
                entry["status"],
            )
        )

    lines.append("")
    lines.append("figures summary:")
    for line in figures.curve_report(result["curve"]).splitlines():
        lines.append("  " + line)

    lines.append("")
    lines.append("Pareto frontier (damage = negotiation task-competence drop):")
    if result["frontier"]:
        for p in result["frontier"]:
            lines.append(
                "  layer %s: A_l=%s, damage=%s"
                % (p["bypassed_layer"], _fmt(p["A_l"]), _fmt(p["damage"]))
            )
    else:
        lines.append("  (no layer has both A_l and damage measurable)")

    lines.append("")
    l_star = result["l_star"]
    if l_star is not None:
        point = l_star["point"]
        ratio = _ratio(point.get("A_l"), point.get("tau_base"))
        lines.append(
            "l*: layer %d  A_l* = %s [%s, %s]  tau(M_D) = %s  "
            "A_l*/tau(M_D) = %s  n_shared = %s" % (
                l_star["layer"], _fmt(point.get("A_l")),
                _fmt(point.get("A_l_ci_low")), _fmt(point.get("A_l_ci_high")),
                _fmt(point.get("tau_base")), _fmt(ratio),
                point.get("n_scenarios_common"),
            )
        )
        if l_star["not_evaluated"]:
            lines.append(
                "  note: checks not evaluated for l*: %s (absence of a "
                "measurement, not a pass)" % ", ".join(l_star["not_evaluated"])
            )
    lines.append("VERDICT: %s" % result["verdict"])

    if dev:
        lines = ["DEV — NOT PUBLISHABLE | " + line for line in lines]
    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Full-pool confirmation (plan D7)
# ---------------------------------------------------------------------------


def confirm_report(base, bypassed, layer=None, item16_decision=None,
                   n_boot=2000, seed=0, a_l_min=A_L_MIN, dev=False) -> str:
    """The full-pool confirmation of l*: recompute A_l*, re-check item 17.

    base      full selection-pool rows for intact M_D (Gate-1's A3 run).
    bypassed  full-pool rows for M_D with l* bypassed (the n=305
              run_baseline.py --bypassed-layer run).
    layer     optional cross-check; the layer is derived from the bypassed
              rows and a mismatch raises.

    A_l* and its CI come from figures.bypass_effect_checked (which is
    metrics.bypass_effect plus the pairing diagnostics), so a confirmation
    computed on a partial overlap says so instead of looking fully paired.
    Item 17 is re-checked as at selection: A_l* >= a_l_min and ci_low > 0.
    Transfer re-evaluation is out of scope here (plan D7): this reports only
    whether the confirmation step passed.

    The D5 tripwire applies here too -- the confirmation is a research-model
    verdict -- with the same dev=True bypass for the DEV model only.
    """
    _require_item16_decision(item16_decision, dev)
    base_rows = _rows(base)
    byp_rows = _rows(bypassed)
    if not base_rows:
        raise ValueError("confirmation base run has zero rows")
    if not byp_rows:
        raise ValueError("confirmation bypassed run has zero rows")
    for row in base_rows:
        if row.get("bypassed_layer") is not None:
            raise ValueError(
                "confirmation base run contains bypassed_layer=%r; it must "
                "be the intact model" % row.get("bypassed_layer")
            )
    layers = set()
    for row in byp_rows:
        layers.add(_int_layer(
            row.get("bypassed_layer"), "confirmation bypassed rows"
        ))
    if len(layers) != 1:
        raise ValueError(
            "confirmation bypassed run mixes layers %s; it must hold exactly "
            "one bypassed layer (l*)" % sorted(layers)
        )
    derived = layers.pop()
    if layer is not None and _int_layer(layer, "confirm layer") != derived:
        raise ValueError(
            "confirmation layer mismatch: rows carry layer %d, caller said %s"
            % (derived, layer)
        )

    effect = figures.bypass_effect_checked(
        base_rows, byp_rows, n_boot=n_boot, seed=seed
    )
    a_l = effect["A_l"]
    ci_low = effect["A_l_ci_low"]
    passed = a_l is not None and a_l >= a_l_min and ci_low is not None and ci_low > 0

    lines = []
    lines.append("SWEEP CONFIRMATION REPORT  (full pool, bootstrap n=%d)" % n_boot)
    lines.append(
        "item-16 calibration decision: %s"
        % (item16_decision if item16_decision else "NONE (dev mode)")
    )
    lines.append(
        "layer %d: A_l* = %s [%s, %s]  tau(M_D) = %s  A_l*/tau(M_D) = %s" % (
            derived, _fmt(a_l), _fmt(effect["A_l_ci_low"]),
            _fmt(effect["A_l_ci_high"]), _fmt(effect["tau_base"]),
            _fmt(_ratio(a_l, effect["tau_base"])),
        )
    )
    lines.append(
        "scenarios: base=%s bypassed=%s common=%s paired=%s%s" % (
            effect["n_scenarios_base"], effect["n_scenarios_bypassed"],
            effect["n_scenarios_common"], effect["paired"],
            ("  reason=%s" % effect["reason"]) if effect.get("reason") else "",
        )
    )
    lines.append(
        "item 17 (A_l* >= %.2f, 95%% CI excludes zero): %s"
        % (a_l_min, "pass" if passed else "FAIL")
    )
    lines.append("CONFIRMATION: %s" % ("PASS" if passed else "FAIL"))

    if dev:
        lines = ["DEV — NOT PUBLISHABLE | " + line for line in lines]
    report = "\n".join(lines)
    print(report)
    return report
