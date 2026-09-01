"""Emit figure-input records from rows.jsonl files.

Thin serialization layer only: every statistic comes from its single home
(metrics.tau_with_ci, figures.layer_curve) and is written at full precision
for scripts/make_figures.py to render. Nothing here computes a new quantity.

    python scripts/emit_figure_records.py tau \
        --rows "M_0:M_0=results/m0-baseline-qwen7b/rows.jsonl" \
        --rows "M_E-l07:M_E=results/e1-l07-qwen7b-s42/rows.jsonl" \
        --out reports/figure-records/tau-qwen.jsonl

    python scripts/emit_figure_records.py layer-curve \
        --base results/md-qwen7b-s42-step281/rows.jsonl \
        --layer "0=results/sweep-.../...-l00/rows.jsonl" ... \
        --truncated-invalid --out reports/figure-records/stage1-curve.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import figures, metrics
from algoverse.relocation import apply_truncated_invalid_ruling


def _load_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_tau_spec(spec):
    key, separator, path = spec.partition("=")
    model, sep2, label = key.partition(":")
    if not separator or not sep2 or not path:
        raise SystemExit("--rows expects MODEL:LABEL=PATH, got %r" % spec)
    return model, label, path


def _parse_layer_spec(spec):
    key, separator, path = spec.partition("=")
    if not separator or not path:
        raise SystemExit("--layer expects N=PATH, got %r" % spec)
    try:
        layer = int(key)
    except ValueError:
        raise SystemExit("--layer layer must be an integer: %r" % key)
    return layer, path


def emit_tau(args):
    records = []
    for spec in args.rows:
        model, label, path = _parse_tau_spec(spec)
        record = metrics.tau_with_ci(
            _load_rows(path), n_boot=args.n_boot, seed=args.seed
        )
        record["model"] = model
        record["label"] = label
        record["rows_path"] = path
        records.append(record)
        print(
            "tau %-24s %-8s tau=%s ci=[%s, %s] n=%d"
            % (model, label, record["tau"], record["tau_ci_low"],
               record["tau_ci_high"], record["n_scenarios"])
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print("wrote %d tau records -> %s" % (len(records), out))


def emit_layer_curve(args):
    paths = {"base": args.base}
    for spec in args.layer:
        layer, path = _parse_layer_spec(spec)
        if layer in paths:
            raise SystemExit("--layer %d given twice" % layer)
        paths[layer] = path

    if args.truncated_invalid:
        import tempfile

        ruling_dir = Path(tempfile.mkdtemp(prefix="ruling-rows-"))
        totals = [0, 0]
        for key in list(paths):
            dst = ruling_dir / str(key) / Path(paths[key]).name
            n, changed = apply_truncated_invalid_ruling(paths[key], dst)
            totals[0] += n
            totals[1] += changed
            paths[key] = str(dst)
        print(
            "RULING APPLIED: truncated->invalid on %d inputs "
            "(%d of %d rows reclassified)"
            % (len(paths), totals[1], totals[0])
        )

    rows = []
    for key in paths:
        rows.extend(_load_rows(paths[key]))
    if args.strip_adapter_prefix:
        stripped = 0
        for row in rows:
            adapter = row.get("adapter_path")
            if adapter and "checkpoints/" in adapter:
                relative = adapter[adapter.index("checkpoints/"):]
                if relative != adapter:
                    row["adapter_path"] = relative
                    stripped += 1
        print(
            "ADAPTER PREFIX STRIPPED: %d rows normalized to their "
            "project-relative checkpoints/ path (platform mount prefixes "
            "differ across Colab/Kaggle runs of the same adapter)" % stripped
        )
    points = figures.layer_curve(rows, n_boot=args.n_boot, seed=args.seed)
    if not points:
        raise SystemExit(
            "layer_curve produced no points -- base and layer rows did not "
            "group into a base + sweep structure (check run_id/arm fields)"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(points, indent=1) + "\n")
    unmeasurable = figures.unmeasurable(points)
    print(
        "wrote %d layer-curve points (%d unmeasurable) -> %s"
        % (len(points), len(unmeasurable), out)
    )
    for layer, reason in unmeasurable:
        print("  unmeasurable l%02d: %s" % (layer, reason))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tau = sub.add_parser("tau", help="tau_with_ci per rows file, "
                                       "render_tau_bars-shaped JSONL")
    p_tau.add_argument("--rows", action="append", required=True,
                       metavar="MODEL:LABEL=PATH")
    p_tau.add_argument("--out", required=True)
    p_tau.add_argument("--n-boot", type=int, default=2000)
    p_tau.add_argument("--seed", type=int, default=0)

    p_curve = sub.add_parser("layer-curve", help="figures.layer_curve points "
                                                 "from base + sweep rows")
    p_curve.add_argument("--base", required=True)
    p_curve.add_argument("--layer", action="append", required=True,
                         metavar="N=PATH")
    p_curve.add_argument(
        "--truncated-invalid", action="store_true",
        help="apply the ratified truncated->invalid scoring ruling: every "
             "rows input is copied to a temp dir with hit_max_tokens rows "
             "reclassified invalid (sources untouched) before analysis",
    )
    p_curve.add_argument(
        "--strip-adapter-prefix", action="store_true",
        help="normalize each row's adapter_path to its project-relative "
             "checkpoints/... suffix before grouping: base and sweep rows "
             "produced on different platforms record the SAME adapter under "
             "different mount prefixes, which would otherwise split them "
             "into baseline-less groups",
    )
    p_curve.add_argument("--out", required=True)
    p_curve.add_argument("--n-boot", type=int, default=2000)
    p_curve.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "tau":
        emit_tau(args)
    else:
        emit_layer_curve(args)


if __name__ == "__main__":
    main()
