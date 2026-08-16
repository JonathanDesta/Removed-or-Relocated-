"""Print the Stage-3 R_t recovery report (matched-arms audit + R_t table).

    python scripts/recovery_report.py \
        --manifest "I,D=runs/s2-id-qwen7b-s42/train_manifest.json" \
        --manifest "I,C=runs/s2-ic-qwen7b-s42/train_manifest.json" \
        --manifest "L,D=runs/s2-ld-qwen7b-s42/train_manifest.json" \
        --manifest "L,C=runs/s2-lc-qwen7b-s42/train_manifest.json" \
        --rows "I,D:8=results/s3-id-t008/rows.jsonl" \
        --rows "I,C:8=results/s3-ic-t008/rows.jsonl" \
        ... (all four arms for every t in {8, 70, 281}) ... \
        --t10-reference "RESEARCH_SPEC 'Ratified decisions (2026-08-16)', T10"

All four --manifest arms are required (the WP-2C matched-arms audit runs
before any R_t is computed and refuses on mismatch), and every requested
checkpoint needs all four arms' --rows. --t10-reference is the tripwire:
the report refuses without a reference to the recorded T10 pre-commitment.

--t (repeatable) selects checkpoints; default is the ratified subset
{8, 70, 281}. Anything outside it refuses unless --allow-extra-t, which is
for POST-DRAFT evaluation of the remaining saved checkpoints only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.recovery_report import ARMS, RATIFIED_RT_SUBSET, recovery_report


def parse_manifest_pairs(pairs):
    result = {}
    for pair in pairs or []:
        arm, _, path = pair.partition("=")
        if not path or arm not in ARMS:
            raise SystemExit(
                "expected ARM=PATH with ARM one of %s, got %r"
                % (", ".join(repr(a) for a in ARMS), pair)
            )
        if arm in result:
            raise SystemExit("--manifest %r given twice" % arm)
        result[arm] = path
    return result


def parse_rows_pairs(pairs):
    result = {}
    for pair in pairs or []:
        key, _, path = pair.partition("=")
        arm, sep, t_text = key.rpartition(":")
        if not path or not sep or arm not in ARMS:
            raise SystemExit(
                "expected ARM:T=PATH with ARM one of %s, got %r"
                % (", ".join(repr(a) for a in ARMS), pair)
            )
        try:
            t = int(t_text)
        except ValueError:
            raise SystemExit("expected an integer checkpoint in %r" % pair)
        if (t, arm) in result:
            raise SystemExit("--rows %s:%d given twice" % (arm, t))
        result[(t, arm)] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", action="append", metavar="ARM:T=PATH",
                        help="one arm's rows.jsonl at checkpoint T; "
                             "repeatable (4 arms x every requested t)")
    parser.add_argument("--manifest", action="append", metavar="ARM=PATH",
                        help="one arm's train_manifest.json; all four arms "
                             "required for the matched-arms audit")
    parser.add_argument("--t10-reference", default=None,
                        help="reference to the recorded T10 pre-commitment "
                             "(required; the report refuses without it)")
    parser.add_argument("--t", action="append", type=int, default=None,
                        help="checkpoint to evaluate; repeatable; default "
                             "is the ratified subset %s"
                             % list(RATIFIED_RT_SUBSET))
    parser.add_argument("--allow-extra-t", action="store_true",
                        help="permit checkpoints outside the ratified "
                             "subset (post-draft evaluation only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.rows or not args.manifest:
        raise SystemExit(
            "need --rows ARM:T=PATH (12 for the draft) and all four "
            "--manifest ARM=PATH"
        )
    recovery_report(
        parse_rows_pairs(args.rows),
        parse_manifest_pairs(args.manifest),
        args.t10_reference,
        t_subset=tuple(args.t) if args.t else RATIFIED_RT_SUBSET,
        allow_extra_t=args.allow_extra_t,
        n_boot=args.n_boot,
        seed=args.seed,
    )
