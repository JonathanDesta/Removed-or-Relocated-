"""Compute the Stage-3 delta curve and optionally stamp a human verdict.

Measurements-only mode prints the quantitative evidence without inventing
thresholds. Add --final plus the human classifications only after reviewing
that evidence.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.relocation import (
    apply_truncated_invalid_ruling,
    edit_relocation_report,
    evaluate_edit_relocation,
    evaluate_relocation,
    relocation_report,
)


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
    parser.add_argument("--permanent-layer", type=int, default=None)
    parser.add_argument("--edit-manifest", default=None)
    parser.add_argument("--init-provenance", default=None)
    parser.add_argument("--edit-layers", nargs="+", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--verdict-ref", default=None)
    parser.add_argument("--dispersion", choices=["dispersed", "concentrated"])
    parser.add_argument(
        "--relocation",
        choices=["entirely-relocated", "partially-relocated", "not-applicable"],
    )
    parser.add_argument(
        "--truncated-invalid", action="store_true",
        help="apply the ratified truncated->invalid scoring ruling: every "
             "rows input is copied to a temp dir with hit_max_tokens rows "
             "reclassified invalid (sources untouched) before analysis",
    )
    parser.add_argument("--origin", action="append", default=None,
                        metavar="N=reconstructed|strengthened")
    args = parser.parse_args(argv)

    if args.truncated_invalid:
        import tempfile

        ruling_dir = Path(tempfile.mkdtemp(prefix="ruling-rows-"))
        nonlocal_totals = [0, 0]

        def _ruled(path, tag):
            dst = ruling_dir / tag / Path(path).name
            n, changed = apply_truncated_invalid_ruling(path, dst)
            nonlocal_totals[0] += n
            nonlocal_totals[1] += changed
            return str(dst)

        args.recovered_base = _ruled(args.recovered_base, "rb")
        args.lesioned_base = _ruled(args.lesioned_base, "lb")
        args.recovered_layer = [
            "%s=%s" % (spec.partition("=")[0],
                       _ruled(spec.partition("=")[2], "rl%s" % spec.partition("=")[0]))
            for spec in args.recovered_layer
        ]
        args.lesioned_layer = [
            "%s=%s" % (spec.partition("=")[0],
                       _ruled(spec.partition("=")[2], "ll%s" % spec.partition("=")[0]))
            for spec in args.lesioned_layer
        ]
        print("RULING APPLIED: truncated->invalid on %d inputs "
              "(%d of %d rows reclassified)"
              % (2 + len(args.recovered_layer) + len(args.lesioned_layer),
                 nonlocal_totals[1], nonlocal_totals[0]))

    edit_mode = any(
        value is not None
        for value in (args.edit_manifest, args.init_provenance, args.edit_layers)
    )
    if edit_mode:
        if not all(
            value is not None
            for value in (args.edit_manifest, args.init_provenance, args.edit_layers)
        ):
            parser.error(
                "edit path requires --edit-manifest, --init-provenance, "
                "and --edit-layers together"
            )
        if args.permanent_layer is not None:
            parser.error("edit path forbids --permanent-layer")
        if args.relocation is not None:
            parser.error(
                "edit-path relocation is computed by the precommitted rule; "
                "do not pass lesion-era --relocation"
            )
        result = evaluate_edit_relocation(
            args.recovered_base,
            _pairs(args.recovered_layer, "--recovered-layer"),
            args.lesioned_base,
            _pairs(args.lesioned_layer, "--lesioned-layer"),
            args.edit_manifest,
            args.init_provenance,
            args.edit_layers,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        return edit_relocation_report(
            result,
            final=args.final,
            verdict_ref=args.verdict_ref,
            dispersion=args.dispersion,
            origins=_origins(args.origin),
        )
    if args.permanent_layer is None:
        parser.error(
            "lesion path requires --permanent-layer (or supply all edit-path flags)"
        )
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
