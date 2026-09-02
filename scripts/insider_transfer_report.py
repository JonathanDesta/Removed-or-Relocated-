"""Stage-1 Insider Trading transfer report: A_l*^IT plus the tau^IT table.

    python scripts/insider_transfer_report.py \
        --base results/md-insider/rows.jsonl \
        --bypassed results/md-insider-l17/rows.jsonl \
        --layer 17 \
        --context M_0=results/m0-insider/rows.jsonl

The spec's transfer re-evaluation of A_l* on the Insider Trading pool:
A_l*^IT = tau(M_D) - tau(M_D with l* probe-bypassed), computed by
figures.bypass_effect_checked (metrics.bypass_effect plus the pairing
diagnostics), alongside every input run's tau^IT with CI, per-condition
invalid rates, and the control-condition honest-report rate
(metrics.task_competence — the IT competence diagnostic). --context runs
(e.g. an M_0 floor-context eval; the 'P-IT8' label was never defined and
is recorded void as of 2026-09-02) join the tau table only.

Refusals in sweep.confirm_report's style: an intact base leg containing
any bypassed row, a bypassed leg mixing layers, or a --layer contradicting
the rows all refuse by name; a comparison with no shared scenarios reports
A_l null-with-reason instead of a fake number.

UNCERTAINTY, read this before quoting an interval (insider-trading.critique-1
F4, ruled by the human 2026-08-21): the IT pool is 200 cells built from 25
invented ticker/company pairs x 8 share amounts over ONE underlying
situation, and generation is greedy. metrics.bootstrap_ci resamples
scenario_ids, so an IT interval describes SURFACE variation across those
invented names and amounts — it is NOT a generalization interval over
independent insider-trading situations, and enumerating more fictional
tickers would narrow it without adding evidence. Every interval this
report prints is labeled within-scaffold accordingly.

P-IT6 (ratified 2026-09-02 as MEASUREMENT-ONLY): the former reading
"transfer supported iff A_l*^IT > 0 with the 95% scenario-bootstrap CI
excluding zero" rested on exactly the generalization the paragraph above
says the interval does not support, so the binary SUPPORTED / NOT SUPPORTED
verdict was WITHDRAWN 2026-08-21 (F4) and STAYS withdrawn: this report
states the measured effect and its within-scaffold interval and stops
there. No numeric margin exists.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import figures, metrics


def _rows(rows_or_path):
    """A row list, loading from disk when given a path instead of rows.

    Same convention as sweep._rows, so the CLI hands paths through and
    tests hand lists, with identical behavior.
    """
    if isinstance(rows_or_path, (str, os.PathLike)):
        return metrics.load_rows(rows_or_path)
    return list(rows_or_path)


def _int_layer(value, where):
    """A real layer index, or a named refusal (sweep.py's discipline)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "%s carry bypassed_layer=%r; expected an integer layer"
            % (where, value)
        )
    return value


def _fmt(value, spec="%.3f"):
    return "n/e" if value is None else spec % value


def insider_transfer_report(base, bypassed, layer=None, context=None,
                            n_boot=2000, seed=0) -> str:
    """The IT transfer report, printed and returned as markdown (WP-IT6).

    base      full-pool IT rows for intact M_D (no probe bypass).
    bypassed  full-pool IT rows for M_D with l* probe-bypassed.
    layer     optional cross-check; the layer is derived from the bypassed
              rows and a mismatch raises.
    context   optional {name: rows-or-path} of additional runs (e.g. M_0)
              reported in the tau table only.
    """
    base_rows = _rows(base)
    byp_rows = _rows(bypassed)
    if not base_rows:
        raise ValueError("transfer base run has zero rows")
    if not byp_rows:
        raise ValueError("transfer bypassed run has zero rows")
    for row in base_rows:
        if row.get("bypassed_layer") is not None:
            raise ValueError(
                "transfer base run contains bypassed_layer=%r; it must "
                "be the intact model" % row.get("bypassed_layer")
            )
    layers = set()
    for row in byp_rows:
        layers.add(_int_layer(
            row.get("bypassed_layer"), "transfer bypassed rows"
        ))
    if len(layers) != 1:
        raise ValueError(
            "transfer bypassed run mixes layers %s; it must hold exactly "
            "one bypassed layer (l*)" % sorted(layers)
        )
    derived = layers.pop()
    if layer is not None and _int_layer(layer, "transfer layer") != derived:
        raise ValueError(
            "transfer layer mismatch: rows carry layer %d, caller said %s"
            % (derived, layer)
        )

    runs = [("M_D intact", base_rows), ("M_D bypass l%d" % derived, byp_rows)]
    for name, rows_or_path in sorted((context or {}).items()):
        rows = _rows(rows_or_path)
        if not rows:
            raise ValueError("context run %r has zero rows" % name)
        runs.append((name, rows))

    effect = figures.bypass_effect_checked(
        base_rows, byp_rows, n_boot=n_boot, seed=seed
    )

    lines = []
    lines.append(
        "INSIDER TRADING TRANSFER REPORT  (full pool, bootstrap n=%d)"
        % n_boot
    )
    lines.append(
        "environment constants RATIFIED 2026-09-02 (RESEARCH_SPEC.md "
        "'Ratified decisions (2026-09-02, Insider Trading operationalization)')"
    )
    lines.append(
        "tau^IT pair (P-IT1, revised 2026-08-21): incentive = default.json, "
        "control = the same history with the concealment incentive removed"
    )
    lines.append("")
    lines.append(
        "| run | D_inc | D_ctrl | tau^IT [95% CI] | invalid% inc/ctrl | "
        "honest-report rate |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, rows in runs:
        gap = metrics.tau_with_ci(rows, n_boot=n_boot, seed=seed)
        competence = metrics.task_competence(rows)
        lines.append("| %s | %s | %s | %s [%s, %s] | %s / %s | %s |" % (
            name,
            _fmt(gap["D_incentive"]), _fmt(gap["D_control"]),
            _fmt(gap["tau"]), _fmt(gap["tau_ci_low"]), _fmt(gap["tau_ci_high"]),
            _fmt(gap["invalid_rate_incentive"], "%.2f"),
            _fmt(gap["invalid_rate_control"], "%.2f"),
            _fmt(competence["competence"]),
        ))

    lines.append("")
    lines.append(
        "layer %d: A_l*^IT = %s [%s, %s]  tau^IT(M_D) = %s" % (
            derived, _fmt(effect["A_l"]), _fmt(effect["A_l_ci_low"]),
            _fmt(effect["A_l_ci_high"]), _fmt(effect["tau_base"]),
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
        "UNCERTAINTY: every interval above is WITHIN-SCAFFOLD — the "
        "bootstrap units are surface variants (invented ticker/company x "
        "share amount) of ONE underlying situation, so these intervals do "
        "NOT generalize across insider-trading situations."
    )
    if effect["A_l"] is None:
        # Null-with-reason: a comparison that could not be computed says why.
        lines.append(
            "transfer reading: not evaluable — A_l*^IT is null "
            "(reason: %s)" % effect["reason"]
        )
    else:
        # P-IT6's binary verdict is WITHDRAWN (F4, ruled 2026-08-21): its
        # "CI excludes zero" clause asserted exactly the scenario
        # generalization the line above says this interval cannot support.
        # State the measurement; let the ratified predicate decide later.
        lines.append(
            "transfer reading: WITHHELD pending a P-IT6 ruling. Measured "
            "A_l*^IT = %s, within-scaffold 95%% CI [%s, %s]. The former "
            "proposal (supported iff A_l*^IT > 0 with the CI excluding "
            "zero) rested on a generalization this interval does not "
            "support." % (
                _fmt(effect["A_l"]), _fmt(effect["A_l_ci_low"]),
                _fmt(effect["A_l_ci_high"]),
            )
        )

    report = "\n".join(lines)
    print(report)
    return report


def parse_context_pairs(pairs):
    result = {}
    for pair in pairs or []:
        name, _, path = pair.partition("=")
        if not name or not path:
            raise SystemExit("expected NAME=PATH, got %r" % pair)
        if name in result:
            raise SystemExit("--context %r given twice" % name)
        result[name] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True,
                        help="intact M_D IT rows.jsonl (full pool)")
    parser.add_argument("--bypassed", required=True,
                        help="M_D with l* probe-bypassed IT rows.jsonl")
    parser.add_argument("--layer", type=int, default=None,
                        help="optional cross-check against the rows' layer")
    parser.add_argument("--context", action="append", metavar="NAME=PATH",
                        help="additional run for the tau table only "
                             "(e.g. M_0=...; repeatable)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    insider_transfer_report(
        args.base, args.bypassed, layer=args.layer,
        context=parse_context_pairs(args.context),
        n_boot=args.n_boot, seed=args.seed,
    )
