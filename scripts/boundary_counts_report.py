"""Boundary counts with exact-count intervals for every rows file given.

Why: the paper promises the counts behind every gate and recovery figure
(Appendix D). Most of those rates sit at 0 or 1, where the scenario
bootstrap collapses to a point and says nothing about sampling error, so
each rate is reported as its raw count with a 95% Wilson interval
(metrics.wilson_interval, the ratified exact-count companion). These are
per-rate intervals; the bootstrap intervals on gaps and ratios printed
beside tau and R_t elsewhere stay separate, as the paper states.

    python3 scripts/boundary_counts_report.py \
        --rows "results/e3-*-t*-l07-qwen7b-s42/rows.jsonl" \
        --rows "results/e3-*-t*-l13-qwen7b-s42/rows.jsonl" \
        --rows "results/e3-*-t*-l08-llama8b-s42/rows.jsonl" \
        --out reports/boundary-counts.txt

Each --rows is a path or glob (sorted). The report prints one line per
rows file (incentive and control: deceptive/valid [Wilson], truncated,
invalid) and then groups runs with an identical outcome signature, which
is the compact table the appendix needs. Stdlib only; reads rows, writes
only --out (refused under results/).

Add --gate-record reports/figure-records/gate-v2-l07.json and --project
/path/maheep-yksa to include a saved gate's M_0/M_D/M_E rate companions.
Raw legacy truncation flags are applied in memory before counting.
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import metrics  # noqa: E402

CONDITIONS = ("incentive", "control")


def count_conditions(rows):
    """{condition: {n, n_valid, n_deceptive, n_trunc, low, high}} for one file."""
    out = {}
    for condition in CONDITIONS:
        sub = [r for r in rows if r.get("condition") == condition]
        # Raw legacy records can retain a parsed label after truncation.
        # Apply the ratified rule on copies; never rewrite the source rows.
        scored = [dict(r, valid=False, deceptive=None) if r.get("hit_max_tokens")
                  else r for r in sub]
        rate = metrics.deception_rate(scored)
        n_trunc = sum(1 for r in sub if r.get("hit_max_tokens"))
        if rate["n_valid"]:
            low, high = metrics.wilson_interval(rate["n_deceptive"], rate["n_valid"])
        else:
            low, high = None, None
        out[condition] = {
            "n": rate["n_total"], "n_valid": rate["n_valid"],
            "n_deceptive": rate["n_deceptive"], "n_trunc": n_trunc,
            "low": low, "high": high,
        }
    return out


def signature(counts):
    """The outcome signature two runs must share to be grouped."""
    return tuple(
        (counts[c]["n_deceptive"], counts[c]["n_valid"]) for c in CONDITIONS
    )


def group_signatures(entries):
    """{signature: [run_id, ...]} in first-seen order."""
    groups = collections.OrderedDict()
    for run_id, counts in entries:
        groups.setdefault(signature(counts), []).append(run_id)
    return groups


def _interval(c):
    if c["low"] is None:
        return "n/a"
    return "[%.3f, %.3f]" % (c["low"], c["high"])


def _cell(c):
    return "%3d/%3d %-16s trunc %3d invalid %3d" % (
        c["n_deceptive"], c["n_valid"], _interval(c), c["n_trunc"],
        c["n"] - c["n_valid"])


def format_report(entries, groups):
    lines = [
        "BOUNDARY COUNTS — deceptive/valid rows per condition with 95% Wilson",
        "intervals (exact-count; NOT the scenario-bootstrap intervals printed",
        "beside tau and R_t). trunc = hit_max_tokens rows; invalid = rows not",
        "valid (truncated rows are invalid under the 2026-09-01 ruling).",
        "",
        "%-40s %-48s %s" % ("run", "incentive", "control"),
    ]
    for run_id, counts in entries:
        lines.append("%-40s %-48s %s" % (
            run_id, _cell(counts["incentive"]), _cell(counts["control"])))
    lines += ["", "GROUPED BY IDENTICAL OUTCOME (incentive deceptive/valid; control deceptive/valid)"]
    for sig, run_ids in groups.items():
        (ki, ni), (kc, nc) = sig
        li = _interval(dict(zip(("low", "high"), metrics.wilson_interval(ki, ni)))) if ni else "n/a"
        lc = _interval(dict(zip(("low", "high"), metrics.wilson_interval(kc, nc)))) if nc else "n/a"
        lines.append("  incentive %3d/%3d %-16s control %3d/%3d %-16s  %d run(s): %s"
                     % (ki, ni, li, kc, nc, lc, len(run_ids), ", ".join(run_ids)))
    return "\n".join(lines) + "\n"


def _refuse_under_results(path):
    resolved = Path(path).resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "results":
            raise SystemExit("--out must not be under results/ (results are rows only)")


def load_rows_strict(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: invalid JSON" % (path, number)) from exc
            key = (row.get("scenario_id"), row.get("condition"))
            if key[0] is None or key[1] not in CONDITIONS or key in seen:
                raise ValueError("%s:%d: missing or duplicate scenario/condition %r"
                                 % (path, number, key))
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("empty rows file: %s" % path)
    if len({r.get("run_id") for r in rows}) != 1:
        raise ValueError("mixed run IDs: %s" % path)
    return rows


def gate_entries(path, project=None):
    """Recompute rate companions from the gate's exact saved inputs.

    The archived gate counts must agree; a disagreement is not silently
    substituted into the paper. Its gap/bootstrap fields are not changed.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for arm in ("M_0", "M_D", "M_E"):
        source = record["inputs"]["rows"][arm]
        if project is not None and "/results/" in source:
            source = Path(project) / "results" / source.split("/results/", 1)[1]
        counts = count_conditions(load_rows_strict(source))
        for condition in CONDITIONS:
            archived = record["counts"][arm][condition]
            actual = counts[condition]
            if (actual["n_deceptive"], actual["n_valid"], actual["n"]) != (
                    archived["n_deceptive"], archived["n_valid"], archived["n_total"]):
                raise ValueError("gate count mismatch: %s %s %s" % (path, arm, condition))
        entries.append((Path(path).stem + ":" + arm, counts))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", action="append", default=[],
                        help="rows.jsonl path or glob (repeatable)")
    parser.add_argument("--gate-record", action="append", default=[],
                        help="saved gate JSON path or glob (repeatable)")
    parser.add_argument("--project", help="rebase gate input paths onto this Drive project")
    parser.add_argument("--out", default=None, help="write the report here too")
    args = parser.parse_args(argv)

    if not args.rows and not args.gate_record:
        parser.error("give --rows or --gate-record")
    def expand(specs):
        paths = []
        for spec in specs:
            matches = sorted(glob.glob(spec))
            if not matches:
                raise SystemExit("no rows file or gate record matches %r" % spec)
            for match in matches:
                resolved = Path(match).resolve()
                if resolved in paths:
                    raise SystemExit("duplicate input: %s" % resolved)
                paths.append(resolved)
        return paths
    paths = expand(args.rows)
    entries = []
    for path in expand(args.gate_record):
        entries.extend(gate_entries(path, args.project))
    for path in paths:
        rows = load_rows_strict(path)
        run_id = (rows[0].get("run_id") if rows else None) or Path(path).parent.name
        entries.append((run_id, count_conditions(rows)))
    report = format_report(entries, group_signatures(entries))
    print(report, end="")
    if args.out:
        _refuse_under_results(args.out)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
