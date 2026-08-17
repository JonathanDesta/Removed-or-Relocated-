"""Compute the Stage-3 delta curve and optionally stamp a human verdict.

Measurements-only mode prints the quantitative evidence without inventing
thresholds. Add --final plus the human classifications only after reviewing
that evidence.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.relocation import evaluate_relocation, relocation_report


def _pairs(values, label):
    result = {}
    for value in values or []:
        key, separator, path = value.partition("=")
        if not separator or not path:
            raise SystemExit("%s expects N=PATH, got %r" % (label, value))
        try:
            layer = int(key)
        except ValueError:
            raise SystemExit("%s layer must be an integer: %r" % (label, key))
        if layer in result:
            raise SystemExit("%s layer %d given twice" % (label, layer))
        result[layer] = path
    return result


def _origins(values):
    result = {}
    for value in values or []:
        key, separator, origin = value.partition("=")
        if not separator:
            raise SystemExit("--origin expects N=reconstructed|strengthened")
        try:
            layer = int(key)
        except ValueError:
            raise SystemExit("--origin layer must be an integer: %r" % key)
        if layer in result:
            raise SystemExit("--origin layer %d given twice" % layer)
        result[layer] = origin
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-base", required=True)
    parser.add_argument("--recovered-layer", action="append", required=True,
                        metavar="N=PATH")
    parser.add_argument("--lesioned-base", required=True)
    parser.add_argument("--lesioned-layer", action="append", required=True,
                        metavar="N=PATH")
    parser.add_argument("--permanent-layer", type=int, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--verdict-ref", default=None)
    parser.add_argument("--dispersion", choices=["dispersed", "concentrated"])
    parser.add_argument(
        "--relocation",
        choices=["entirely-relocated", "partially-relocated", "not-applicable"],
    )
    parser.add_argument("--origin", action="append", default=None,
                        metavar="N=reconstructed|strengthened")
    args = parser.parse_args(argv)

    result = evaluate_relocation(
        args.recovered_base,
        _pairs(args.recovered_layer, "--recovered-layer"),
        args.lesioned_base,
        _pairs(args.lesioned_layer, "--lesioned-layer"),
        lesioned_layer=args.permanent_layer,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    return relocation_report(
        result,
        final=args.final,
        verdict_ref=args.verdict_ref,
        dispersion=args.dispersion,
        relocation=args.relocation,
        origins=_origins(args.origin),
    )


if __name__ == "__main__":
    main()
