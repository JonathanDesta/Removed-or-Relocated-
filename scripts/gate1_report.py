"""Print the Gate-1 decision table from finished runs.

    python scripts/gate1_report.py \
        --rows M_0=results/m0-baseline/rows.jsonl \
        --rows M_D=results/md/rows.jsonl \
        --competence M_0=results/m0-baseline/competence.jsonl \
        --competence M_D=results/md/competence.jsonl

The PASS/FAIL line appears once M_0 and M_D are given (any other model
key is reported as a defect). Thresholds are flags with defaults and are
printed with the decision.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import cli, metrics
from algoverse.eval import gate1_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", action="append", required=True,
                        metavar="NAME=PATH", help="rows.jsonl per model")
    parser.add_argument("--competence", action="append", default=None,
                        metavar="NAME=PATH", help="competence.jsonl per model")
    parser.add_argument("--tau-gain-min", type=float, default=metrics.TAU_GAIN_MIN,
                        help="margin the M_D-minus-M_0 tau gain must exceed")
    parser.add_argument("--competence-drop-max", type=float,
                        default=metrics.COMPETENCE_DROP_MAX)
    parser.add_argument("--ppl-rise-max", type=float, default=metrics.PPL_RISE_MAX)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--dev", action="store_true",
                        help="skip publishability guards and stamp the report")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    gate1_report(
        cli.parse_pairs(args.rows, "--rows", parser=parser),
        competence_paths=cli.parse_pairs(args.competence, "--competence", parser=parser),
        n_boot=args.n_boot,
        tau_gain_min=args.tau_gain_min,
        competence_drop_max=args.competence_drop_max,
        ppl_rise_max=args.ppl_rise_max,
        dev=args.dev,
    )
    return 0


if __name__ == "__main__":
    cli.run_main(main)
