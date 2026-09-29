"""Print the Stage-3 R_t recovery report (matched-arms audit + R_t table).

    python scripts/recovery_report.py \\
        --manifest "E,D=$PROJECT/checkpoints/<E,D run>/train_manifest.json" \\
        --manifest "E,C=$PROJECT/checkpoints/<E,C run>/train_manifest.json" \\
        --manifest "I,D=$PROJECT/checkpoints/<I,D run>/train_manifest.json" \\
        --manifest "I,C=$PROJECT/checkpoints/<I,C run>/train_manifest.json" \\
        --rows "E,D:8=$PROJECT/results/<E,D t8 eval>/rows.jsonl" \\
        --rows "E,C:8=$PROJECT/results/<E,C t8 eval>/rows.jsonl" \\
        ... (all four arms for every t in {8, 70, 281}) ...

All four --manifest arms are required (the matched-arms audit runs before
any R_t is computed and refuses on mismatch), and every requested
checkpoint needs all four arms' --rows.

--t (repeatable) selects checkpoints; default is the pre-registered subset
{8, 70, 281}. Anything outside it refuses unless --allow-extra-t, which is
for evaluating the remaining saved checkpoints only. --eps is the R_t
denominator floor (default the pre-registered 0.10). --emit-records writes
the figure records from the same evaluation the report prints.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.metrics import RECOVERY_EPS
from algoverse.recovery_report import (
    DEFAULT_RECOVERY_ARMS,
    RT_SUBSET,
    evaluate_recovery,
    render_recovery_report,
)


def build_recovery_records(result, env_label):
    """Figure-input records from an evaluate_recovery result, full precision.

    One dict per requested checkpoint: env, checkpoint_step, R_t + CI +
    reason, the denominator floor and the dropped-resample count, the
    ordered arms list, and one tau_<ARM> per arm (comma stripped) --
    consumable by make_figures.py rt.
    """
    records = []
    for t in result["requested_t"]:
        entry = result["per_t"][t]
        record = {
            "env": env_label,
            "checkpoint_step": t,
            "R_t": entry["R_t"],
            "R_t_ci_low": entry["R_t_ci_low"],
            "R_t_ci_high": entry["R_t_ci_high"],
            "reason": entry.get("reason"),
            "eps": result["eps"],
            "n_boot_dropped": entry.get("n_boot_dropped"),
            "arms": list(result["arms"]),
            "n_boot": result["n_boot"],
        }
        for arm in result["arms"]:
            record["tau_%s" % arm.replace(",", "")] = entry["tau_by_arm"][arm]
        records.append(record)
    return records


def parse_manifest_pairs(pairs, arms=DEFAULT_RECOVERY_ARMS):
    arms = tuple(arms)
    result = {}
    for pair in pairs or []:
        arm, _, path = pair.partition("=")
        if not path or arm not in arms:
            raise SystemExit(
                "expected ARM=PATH with ARM one of %s, got %r"
                % (", ".join(repr(a) for a in arms), pair)
            )
        if arm in result:
            raise SystemExit("--manifest %r given twice" % arm)
        result[arm] = path
    return result


def parse_rows_pairs(pairs, arms=DEFAULT_RECOVERY_ARMS):
    arms = tuple(arms)
    result = {}
    for pair in pairs or []:
        key, _, path = pair.partition("=")
        arm, sep, t_text = key.rpartition(":")
        if not path or not sep or arm not in arms:
            raise SystemExit(
                "expected ARM:T=PATH with ARM one of %s, got %r"
                % (", ".join(repr(a) for a in arms), pair)
            )
        try:
            t = int(t_text)
        except ValueError:
            raise SystemExit("expected an integer checkpoint in %r" % pair)
        if (t, arm) in result:
            raise SystemExit("--rows %s:%d given twice" % (arm, t))
        result[(t, arm)] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", action="append", metavar="ARM:T=PATH",
                        help="one arm's rows.jsonl at checkpoint T; "
                             "repeatable (4 arms x every requested t)")
    parser.add_argument("--manifest", action="append", metavar="ARM=PATH",
                        help="one arm's train_manifest.json; all four arms "
                             "required for the matched-arms audit")
    parser.add_argument("--t", action="append", type=int, default=None,
                        help="checkpoint to evaluate; repeatable; default "
                             "is the pre-registered subset %s"
                             % list(RT_SUBSET))
    parser.add_argument("--allow-extra-t", action="store_true",
                        help="permit checkpoints outside the pre-registered "
                             "subset (the remaining saved checkpoints)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=RECOVERY_EPS,
                        help="R_t denominator floor; default %(default)s "
                             "(pre-registered)")
    parser.add_argument(
        "--arms", nargs=4, default=DEFAULT_RECOVERY_ARMS,
        metavar=("NUM_D", "NUM_C", "DEN_D", "DEN_C"),
        help="ordered recovery arms (numerator D/C over denominator D/C); "
             "default %(default)s",
    )
    parser.add_argument("--emit-records", default=None, metavar="PATH",
                        help="also write one JSONL record per checkpoint "
                             "(full precision: R_t, CI bounds, reason, and "
                             "the raw per-arm taus) for make_figures.py rt")
    parser.add_argument("--env-label", default=None,
                        help="'env' value stamped on emitted records "
                             "(required with --emit-records)")
    args = parser.parse_args(argv)

    if args.emit_records and not args.env_label:
        parser.error("--emit-records requires --env-label")

    if not args.rows or not args.manifest:
        raise SystemExit(
            "need --rows ARM:T=PATH (12 for the pre-registered subset) and "
            "all four --manifest ARM=PATH"
        )
    rows_inputs = parse_rows_pairs(args.rows, args.arms)
    manifest_inputs = parse_manifest_pairs(args.manifest, args.arms)
    t_subset = tuple(args.t) if args.t else RT_SUBSET
    result = evaluate_recovery(
        rows_inputs, manifest_inputs,
        t_subset=t_subset, allow_extra_t=args.allow_extra_t,
        n_boot=args.n_boot, seed=args.seed, arms=tuple(args.arms),
        eps=args.eps,
    )
    render_recovery_report(result)

    if args.emit_records:
        import json

        records = build_recovery_records(result, args.env_label)
        out = Path(args.emit_records)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in records))
        print("emitted %d recovery records -> %s" % (len(records), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
