"""Compute the Stage-3 edit-relocation delta curve and its rule-derived verdict.

    python scripts/relocation_report.py \\
        --recovered-base $PROJECT/results/<E,D t281 base run>/rows.jsonl \\
        --recovered-layer 0=$PROJECT/results/<E,D t281 sweep>/<tag>-l00/rows.jsonl ... \\
        --edited-base $PROJECT/results/<M_E base run>/rows.jsonl \\
        --edited-layer 0=$PROJECT/results/<M_E sweep>/<tag>-l00/rows.jsonl ... \\
        --edit-manifest $PROJECT/checkpoints/<edit run>/train_manifest.json \\
        --init-provenance $PROJECT/checkpoints/<E,D run>/init_provenance.json \\
        --edit-layers 6 7 8 \\
        --emit-curves reports/figure-records/delta-v2-l07

Measurements-only mode prints the delta curve, the candidate layers and the
pre-committed spatial verdict (recovered-in-place / relocated / mixed /
not-applicable) without inventing thresholds. Add --final plus the manual
dispersion and per-layer origin classifications only after reviewing that
evidence.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.relocation import edit_relocation_report, evaluate_edit_relocation


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


def _emit_curves(result, basename):
    """Write the two render_delta inputs from an evaluated relocation result.

    <basename>-recovered.json and <basename>-edited.json each hold a
    layer-curve-shaped list ({bypassed_layer, A_l, reason}) at full
    precision, straight from result["points"] -- no recomputation. A
    "voided_validity" reason is per side: it travels only with a side whose
    A is None (the validity rule voided THAT run), never with a side that
    kept its measurement because only the other side was void.
    """
    import json

    base = Path(basename)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for side, key in (("recovered", "A_recovered"), ("edited", "A_edited")):
        curve = []
        for point in result["points"]:
            reason = point.get("reason")
            if reason == "voided_validity" and point.get(key) is not None:
                reason = None
            curve.append({
                "bypassed_layer": point["layer"],
                "A_l": point.get(key),
                "reason": reason,
            })
        path = Path("%s-%s.json" % (base, side))
        path.write_text(json.dumps(curve, indent=1) + "\n")
        written.append(str(path))
    print("emitted delta curves -> %s" % ", ".join(written))


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
    parser.add_argument("--recovered-base", required=True,
                        help="unprobed rows.jsonl of the recovered E,D-t281 checkpoint")
    parser.add_argument("--recovered-layer", action="append", required=True,
                        metavar="N=PATH",
                        help="one swept layer of the recovered checkpoint; repeatable")
    parser.add_argument("--edited-base", required=True,
                        help="unprobed rows.jsonl of the just-edited M_E")
    parser.add_argument("--edited-layer", action="append", required=True,
                        metavar="N=PATH",
                        help="one swept layer of the just-edited M_E; repeatable")
    parser.add_argument("--edit-manifest", required=True,
                        help="the edit run's train_manifest.json")
    parser.add_argument("--init-provenance", required=True,
                        help="the E,D continuation's init_provenance.json")
    parser.add_argument("--edit-layers", nargs="+", type=int, required=True,
                        help="the edit window the edit run trained")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true",
                        help="stamp the manual classifications (--dispersion, --origin)")
    parser.add_argument("--dispersion", choices=["dispersed", "concentrated"])
    parser.add_argument("--origin", action="append", default=None,
                        metavar="N=reconstructed|strengthened")
    parser.add_argument("--emit-curves", default=None, metavar="BASENAME",
                        help="also write <BASENAME>-recovered.json and "
                             "<BASENAME>-edited.json (layer-curve-shaped, "
                             "full precision) for make_figures.py delta")
    args = parser.parse_args(argv)

    result = evaluate_edit_relocation(
        args.recovered_base,
        _pairs(args.recovered_layer, "--recovered-layer"),
        args.edited_base,
        _pairs(args.edited_layer, "--edited-layer"),
        args.edit_manifest,
        args.init_provenance,
        args.edit_layers,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    if args.emit_curves:
        _emit_curves(result, args.emit_curves)
    return edit_relocation_report(
        result,
        final=args.final,
        dispersion=args.dispersion,
        origins=_origins(args.origin),
    )


if __name__ == "__main__":
    main()
