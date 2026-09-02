"""Build a POOLED probe test set from existing results rows (no generation).

The 2026-09-02 readout showed the probe test sets' "lied" label was glued
to scenario type (in Llama M_0 rows every lie is a no-offer fabrication).
A test set where the SCENARIO is held fixed and only the label varies
needs, for each scenario, one honest and one lying response — which
exist across runs: M_0 was honest in every offer scenario and M_D lied in
every one. This script pools rows from several runs into one JSONL that
probe_examples_from_rows already understands (rows keep their run_id,
so per-source and paired scoring can tell the generators apart).

Filters:
  --offer-only      keep scenarios with a real outside offer (true_value
                    not null) — the stratum where "lied" cannot be
                    "no-offer scenario"
  --no-offer-only   the complement
  --require-both    keep only scenarios that end up with BOTH a lied and a
                    did-not-lie valid incentive row (matched pairs)

Only valid incentive rows with a boolean deceptive label are kept (the
ratified within-incentive control). Output is a fresh JSONL; refuses to
overwrite. Prints the class counts per stratum and per source run.

    python scripts/build_probe_test_set.py \
        --rows results/m0-baseline-qwen7b/rows.jsonl \
        --rows results/md-qwen7b-s42-step281/rows.jsonl \
        --offer-only --require-both \
        --out results/probe-sets-v3/qwen7b-pairs-offer.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.metrics import load_rows  # noqa: E402


def select_rows(row_lists, offer_only=False, no_offer_only=False, require_both=False):
    """Pure selection: the pooled, filtered valid incentive rows."""
    if offer_only and no_offer_only:
        raise ValueError("--offer-only and --no-offer-only are exclusive")
    kept = []
    for rows in row_lists:
        for row in rows:
            if row.get("condition") != "incentive" or row.get("valid") is not True:
                continue
            if row.get("deceptive") is not True and row.get("deceptive") is not False:
                continue
            has_offer = row.get("true_value") is not None
            if offer_only and not has_offer:
                continue
            if no_offer_only and has_offer:
                continue
            kept.append(row)
    if require_both:
        by_scenario = collections.defaultdict(set)
        for row in kept:
            by_scenario[row["scenario_id"]].add(bool(row["deceptive"]))
        kept = [row for row in kept if by_scenario[row["scenario_id"]] == {True, False}]
    return kept


def describe(rows):
    lines = []
    total = collections.Counter((row.get("true_value") is not None, bool(row["deceptive"])) for row in rows)
    lines.append("offer scenarios: lied %d / not %d; no-offer: lied %d / not %d"
                 % (total[(True, True)], total[(True, False)], total[(False, True)], total[(False, False)]))
    per_source = collections.Counter((row.get("run_id"), bool(row["deceptive"])) for row in rows)
    for run_id in sorted({row.get("run_id") for row in rows}, key=str):
        lines.append("  %-36s lied %3d / not %3d" % (run_id, per_source[(run_id, True)], per_source[(run_id, False)]))
    n_scen = len({row["scenario_id"] for row in rows})
    lines.append("  %d rows over %d scenarios" % (len(rows), n_scen))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--offer-only", action="store_true")
    parser.add_argument("--no-offer-only", action="store_true")
    parser.add_argument("--require-both", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s (results are append-only; pick a new name)" % out)
    row_lists = [load_rows(Path(p)) for p in args.rows]
    kept = select_rows(row_lists, offer_only=args.offer_only, no_offer_only=args.no_offer_only,
                       require_both=args.require_both)
    if not kept:
        raise SystemExit("no rows survived the filters")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    print("wrote %s" % out)
    print(describe(kept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
