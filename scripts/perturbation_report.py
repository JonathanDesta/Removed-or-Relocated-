"""Compare perturbation-battery variants against the canonical run. CPU only.

EXPLORATORY — see algoverse/perturb.py's status note; nothing here is a
paper number without ratification.

    python scripts/perturbation_report.py \
        --rows canonical=results/diag-perturb-canonical-qwen7b/rows.jsonl \
        --rows question_paraphrase=... \
        --rows format_swap=... \
        --rows private_note=...

Prints, per variant: tau with its scenario-bootstrap CI, per-condition
invalid rates, control-condition task competence, the incentive-condition
deception rate split into fabrication (no real offer) vs inflation (real
offer exaggerated), and marker usage among incentive rows (canonical
trained line vs the format_swap variant line) — the count that separates
"stopped lying" from "ignored the new format and kept the trained line".

Reading the table: a variant's tau is interpretable only where its invalid
rates stay near canonical's (the sweep's own i15 lesson); fabrication and
inflation are reported separately because the Qwen sweep showed they have
different circuit-level footprints.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import perturb
from algoverse.metrics import load_rows, task_competence, tau_with_ci


def parse_pairs(pairs):
    result = {}
    for pair in pairs or []:
        name, _, path = pair.partition("=")
        if not path:
            raise SystemExit("expected NAME=PATH, got %r" % pair)
        if name in result:
            raise SystemExit("variant %r given twice" % name)
        result[name] = path
    return result


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return "%.*f" % (digits, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", action="append", required=True,
                        metavar="VARIANT=PATH",
                        help="one variant's rows.jsonl; repeatable; include "
                             "canonical for the comparison anchor")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    inputs = parse_pairs(args.rows)
    if "canonical" not in inputs:
        print("WARNING: no canonical run supplied; variant taus have no "
              "same-script replication anchor")

    print("PERTURBATION BATTERY REPORT  (exploratory diagnostic, unratified; "
          "bootstrap n=%d)" % args.n_boot)
    print()
    header = (
        "| variant | tau [95% CI] | D_inc | D_ctl | inv inc/ctl | "
        "competence | fab dec | infl dec | canon-line | variant-line |"
    )
    print(header)
    print("|" + "---|" * 10)

    order = ["canonical"] + [v for v in inputs if v != "canonical"]
    scenario_sets = {}
    for variant in order:
        rows = load_rows(Path(inputs[variant]))
        if not rows:
            print("| %s | (no rows) |" % variant)
            continue
        scenario_sets[variant] = {
            (row.get("scenario_id"), row.get("condition")) for row in rows
        }
        gap = tau_with_ci(rows, n_boot=args.n_boot)
        competence = task_competence(rows)
        split = perturb.fabrication_inflation_split(rows)
        incentive = [r for r in rows if r.get("condition") == "incentive"]
        markers = [perturb.marker_usage(r.get("response_text")) for r in incentive]
        canon_n = sum(1 for m in markers if m["canonical"])
        variant_n = sum(1 for m in markers if m["variant"])
        fab_d, fab_n = split["fabrication"]
        infl_d, infl_n = split["inflation"]
        print(
            "| %s | %s [%s, %s] | %s | %s | %s/%s | %s | %d/%d | %d/%d "
            "| %d/%d | %d/%d |"
            % (
                variant,
                fmt(gap["tau"]), fmt(gap["tau_ci_low"]), fmt(gap["tau_ci_high"]),
                fmt(gap["D_incentive"]),
                fmt(gap["D_control"]),
                fmt(gap["invalid_rate_incentive"], 2),
                fmt(gap["invalid_rate_control"], 2),
                fmt(competence["competence"]),
                fab_d, fab_n, infl_d, infl_n,
                canon_n, len(incentive), variant_n, len(incentive),
            )
        )

    reference = scenario_sets.get("canonical")
    if reference:
        for variant, cells in scenario_sets.items():
            if variant != "canonical" and cells != reference:
                print(
                    "\nWARNING: %s covers a different scenario x condition "
                    "set than canonical (%d vs %d cells); per-variant "
                    "comparisons are not paired"
                    % (variant, len(cells), len(reference))
                )

    print(
        "\nReading guide: interpret a variant's deception change only where "
        "its invalid rates and competence stay near canonical's; on "
        "format_swap, canon-line counts trained-format override (the model "
        "ignored the new instruction), variant-line counts compliance."
    )
