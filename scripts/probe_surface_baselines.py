"""Surface baselines for the Appendix A probe transfer (stdlib only, CPU).

Why: beside the per-layer probe AUROC, the paper reports how well three
surface quantities already present in the prompt and the response order
lying answers above honest ones on the SAME test population — the raw
stated number from the final line, the response character length, and the
stated number divided by the company's own offer. A probe that only matches
those numbers claims no advantage over surface information. Nothing else in
the repo emits them, so this script does.

    python3 scripts/probe_surface_baselines.py \\
        --rows results/diag-md-qwen7b-step8/rows.jsonl \\
        --out reports/probe-surface-baselines.jsonl

Population: exactly the rows corroboration.labeled_incentive_rows keeps for
the probes — VALID incentive-condition rows whose `deceptive` is a boolean
(lied = True), with a non-empty response_text. Strata: "all" and "offer"
(scenarios with a genuine outside offer, true_value not null), the latter
being the stratum the paper's Figure 3 plots.

Features per row:
    stated_number      claimed_value; "NONE" / null counts as 0 (a claim of
                       $0 is NONE under the ratified grader rule)
    char_length        len(response_text)
    stated_over_offer  stated_number / scenario_params["company_offer"]

AUROC is rank-based (Mann-Whitney; ties count one half), with a scenario-
level bootstrap interval from metrics.bootstrap_ci (n_boot 2000, seed 0,
the same machinery every other interval in the paper uses). Writes one
JSONL row per (feature, stratum) to --out, refused under results/ (results
are model outputs only), and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import metrics  # noqa: E402
from algoverse.tasks import INCENTIVE  # noqa: E402

FEATURES = ("stated_number", "char_length", "stated_over_offer")
STRATA = ("all", "offer")
ANALYSIS_PREFIX = "surface_auroc"


def select_rows(rows):
    """The probe test population: the predicates of
    corroboration.labeled_incentive_rows, returning the ORIGINAL rows so the
    surface fields (claimed_value, response_text, scenario_params) stay
    available."""
    kept = []
    for row in rows:
        if row.get("condition") != INCENTIVE:
            continue
        if row.get("valid") is not True:
            continue
        deceptive = row.get("deceptive")
        if deceptive is not True and deceptive is not False:
            continue
        if not row.get("response_text"):
            continue
        kept.append(row)
    return kept


def stated_number(row):
    claimed = row.get("claimed_value")
    if claimed is None or claimed == "NONE":
        return 0.0
    return float(claimed)


def feature_value(row, feature):
    if feature == "stated_number":
        return stated_number(row)
    if feature == "char_length":
        return float(len(row.get("response_text") or ""))
    if feature == "stated_over_offer":
        params = row.get("scenario_params") or {}
        offer = params.get("company_offer")
        if not offer:
            raise ValueError(
                "row %s has no scenario_params.company_offer; the ratio "
                "baseline needs the negotiation grid parameters"
                % row.get("scenario_id"))
        return stated_number(row) / float(offer)
    raise ValueError("unknown feature %r" % feature)


def in_stratum(row, stratum):
    if stratum == "all":
        return True
    if stratum == "offer":
        return row.get("true_value") is not None
    raise ValueError("unknown stratum %r" % stratum)


def auroc(scores, labels):
    """Rank-based AUROC: P(score_lied > score_honest) + 0.5 P(equal).
    None when either class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    neg_sorted = sorted(neg)
    total = 0.0
    for s in pos:
        lo = _bisect_left(neg_sorted, s)
        hi = _bisect_right(neg_sorted, s)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def _bisect_left(values, x):
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(values, x):
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < values[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _stat(feature):
    def stat_fn(groups):
        rows = groups["rows"]
        return auroc([feature_value(r, feature) for r in rows],
                     [bool(r["deceptive"]) for r in rows])
    return stat_fn


def evaluate(rows, n_boot=2000, seed=0):
    """[{analysis, stratum, n, n_lied, value, ci_low, ci_high}] over every
    (feature, stratum); value is None when a stratum has one class."""
    population = select_rows(rows)
    records = []
    for stratum in STRATA:
        sub = [r for r in population if in_stratum(r, stratum)]
        n_lied = sum(1 for r in sub if r["deceptive"] is True)
        for feature in FEATURES:
            if sub and 0 < n_lied < len(sub):
                value, low, high = metrics.bootstrap_ci(
                    {"rows": sub}, _stat(feature), n_boot=n_boot, seed=seed)
            else:
                value, low, high = None, None, None
            records.append({
                "analysis": "%s:%s" % (ANALYSIS_PREFIX, feature),
                "stratum": stratum, "n": len(sub), "n_lied": n_lied,
                "value": value, "ci_low": low, "ci_high": high,
                "n_boot": n_boot, "seed": seed,
            })
    return records


def format_table(records):
    lines = ["%-32s %-6s %5s %5s %7s %17s" % ("analysis", "strat", "n", "lied", "AUROC", "95% CI")]
    for r in records:
        value = "n/a" if r["value"] is None else "%.3f" % r["value"]
        ci = ("[%.3f, %.3f]" % (r["ci_low"], r["ci_high"])
              if r["ci_low"] is not None else "[n/a]")
        lines.append("%-32s %-6s %5d %5d %7s %17s"
                     % (r["analysis"], r["stratum"], r["n"], r["n_lied"], value, ci))
    return "\n".join(lines)


def _refuse_under_results(parser, path):
    resolved = Path(path).resolve()
    results_root = (Path.cwd() / "results").resolve()
    if resolved == results_root or results_root in resolved.parents:
        parser.error("--out must not be under results/ (results are JSONL model outputs only)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", required=True, help="probe test set rows.jsonl")
    parser.add_argument("--out", required=True, help="JSONL of AUROC records (never under results/)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _refuse_under_results(parser, args.out)

    rows = metrics.load_rows(args.rows)
    records = evaluate(rows, n_boot=args.n_boot, seed=args.seed)
    for record in records:
        record["rows_path"] = str(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(format_table(records))
    print("wrote %d records to %s" % (len(records), out))


if __name__ == "__main__":
    main()
