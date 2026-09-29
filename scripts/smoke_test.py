"""Run the end-to-end smoke test locally. No GPU needed.

    python scripts/smoke_test.py [--model-id ID] [--n 6] [--out-dir results/smoke]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import cli
from algoverse.eval import smoke_test


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None, help="default: the 0.5B dev model")
    parser.add_argument("--n", type=int, default=6, help="scenarios (x2 conditions)")
    parser.add_argument("--out-dir", default="results/smoke")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    smoke_test(model_id=args.model_id, n_scenarios=args.n, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    cli.run_main(main)
