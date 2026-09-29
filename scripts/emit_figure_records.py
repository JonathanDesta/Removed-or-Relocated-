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
        --out reports/figure-records/stage1-curve.json

    python scripts/emit_figure_records.py transfer --model Llama-3.1-8B \
        --tau "Offer Negotiation=reports/figure-records/tau-llama.jsonl" \
        --tau "Insider Trading=reports/figure-records/tau-insider.jsonl" \
        --arms M_0 M_D --out reports/figure-records/transfer-llama.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import figures, metrics

# Results rows go through the truncation rule; figure records are read raw.
_load_results = metrics.load_results
_load_records = metrics.load_rows


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
            _load_results(path), n_boot=args.n_boot, seed=args.seed
        )
        record["model"] = model
        record["label"] = label
        record["rows_path"] = path
        record["truncation_rule"] = metrics.truncation_rule_label()
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


def emit_transfer(args):
    """Regroup EXISTING tau records by environment for render_tau_bars.

    Each --tau ENV_LABEL=TAU_JSONL contributes the records whose "model" is
    --model and whose "label" is one of --arms; "model" is rewritten to the
    environment label (the x-axis group), "source_model" and
    "source_record" are added, and every other field (tau, CIs, rows_path,
    n_scenarios, ...) is relayed verbatim — nothing is recomputed. A
    missing or duplicated (environment, arm) is refused so a bar can never
    silently come from the wrong run.
    """
    records = []
    seen = set()
    for spec in args.tau:
        env_label, sep, path = spec.partition("=")
        if not sep or not env_label or not path:
            raise SystemExit("--tau expects ENV_LABEL=TAU_JSONL, got %r" % spec)
        source = _load_records(path)
        for arm in args.arms:
            matches = [r for r in source
                       if r.get("model") == args.model and r.get("label") == arm]
            if not matches:
                raise SystemExit("no record for model %r arm %r in %s"
                                 % (args.model, arm, path))
            if len(matches) > 1:
                raise SystemExit("%d records for model %r arm %r in %s"
                                 % (len(matches), args.model, arm, path))
            if (env_label, arm) in seen:
                raise SystemExit("duplicate environment/arm %r/%r" % (env_label, arm))
            seen.add((env_label, arm))
            record = dict(matches[0])
            record["source_model"] = record["model"]
            record["source_record"] = path
            record["model"] = env_label
            records.append(record)
            print("transfer %-36s %-6s tau=%s ci=[%s, %s] rows=%s"
                  % (env_label.replace("\n", " "), arm, record.get("tau"),
                     record.get("tau_ci_low"), record.get("tau_ci_high"),
                     record.get("rows_path")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print("wrote %d transfer records -> %s" % (len(records), out))


def emit_layer_curve(args):
    paths = {"base": args.base}
    for spec in args.layer:
        layer, path = _parse_layer_spec(spec)
        if layer in paths:
            raise SystemExit("--layer %d given twice" % layer)
        paths[layer] = path

    rows = []
    for key in paths:
        rows.extend(_load_results(paths[key]))
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
            "project-relative checkpoints/ path (mount prefixes differ "
            "across machines for the same adapter)" % stripped
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

    p_tr = sub.add_parser("transfer", help="regroup existing tau records by "
                                           "environment (model -> env label) "
                                           "for a cross-environment tau-bars figure")
    p_tr.add_argument("--tau", action="append", required=True,
                      metavar="ENV_LABEL=TAU_JSONL")
    p_tr.add_argument("--model", required=True,
                      help="keep records whose 'model' equals this")
    p_tr.add_argument("--arms", nargs="+", default=["M_0", "M_D"],
                      help="labels to keep, in order")
    p_tr.add_argument("--out", required=True)

    p_curve = sub.add_parser("layer-curve", help="figures.layer_curve points "
                                                 "from base + sweep rows")
    p_curve.add_argument("--base", required=True)
    p_curve.add_argument("--layer", action="append", required=True,
                         metavar="N=PATH")
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
    elif args.command == "transfer":
        emit_transfer(args)
    else:
        emit_layer_curve(args)


if __name__ == "__main__":
    main()
