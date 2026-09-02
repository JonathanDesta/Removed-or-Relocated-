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

    python scripts/emit_figure_records.py pareto --model Qwen2.5-7B \
        --curve reports/figure-records/stage1-curve-qwen.json \
        --base-run-id md-qwen7b-s42-step281 --m0-rows results/m0-.../rows.jsonl \
        --competence results/md-.../competence.jsonl \
        --competence results/sweep-.../...-l00/competence.jsonl ... \
        --out reports/figure-records/pareto-qwen.json

    python scripts/emit_figure_records.py decomposition \
        --sweep-root results/sweep-md-qwen7b-s42-step281 --n-layers 28 \
        --out reports/figure-records/decomposition-md-qwen.json

    python scripts/emit_figure_records.py edit-gate-summary \
        --gate "Qwen2.5-7B:l07=reports/figure-records/gate-v2-l07.json" \
        --manifest l07=checkpoints/edit-l07-qwen7b-s42/train_manifest.json \
        --stage1-curve Qwen2.5-7B=reports/figure-records/stage1-curve-qwen.json \
        --out reports/figure-records/edit-gate-summary.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import figures, metrics, plotting
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


# ---------------------------------------------------------------------------
# pareto: one record per model, one panel per damage metric
# ---------------------------------------------------------------------------

PARETO_METRICS = ("task_competence", "wikitext2_ppl", "wikitext2_neutral_jsd",
                  "gsm8k_exact_match", "mmlu_acc")


def _damage_bound(metric, args):
    """The ratified cap for one damage axis (RESEARCH_SPEC items 2, 3, 16)."""
    if metric == "wikitext2_ppl":
        return args.ppl_rise_max
    if metric == "wikitext2_neutral_jsd":
        return args.neutral_jsd_max
    return args.competence_drop_max      # task competence, gsm8k, mmlu: drops


def emit_pareto(args):
    curve = plotting.load_records(args.curve)
    if not isinstance(curve, list) or not curve:
        raise SystemExit("--curve must be a non-empty figures.layer_curve JSON list")
    plotting._normalize_comparison(curve)
    competence_rows = []
    for path in args.competence:
        competence_rows.extend(_load_rows(path))
    index = figures.index_competence(competence_rows)
    base_key = ("run_id", args.base_run_id)
    base_competence = None
    if args.m0_rows:
        base_competence = metrics.task_competence(
            _load_rows(args.m0_rows)
        )["competence"]
        if base_competence is None:
            raise SystemExit(
                "--m0-rows has no valid control rows; its task competence "
                "is None and cannot be the P-S6 reference"
            )
    wanted = args.metric or list(PARETO_METRICS)
    panels = []
    for metric in wanted:
        points = figures.pareto_points(
            [dict(p) for p in curve], competence_index=index,
            damage_metric=metric, base_key=base_key,
            base_competence=base_competence,
        )
        frontier = figures.pareto_frontier(points)
        bounds = {"a_l_min": args.a_l_min, "damage_max": _damage_bound(metric, args)}
        plotted = [
            p for p in points
            if p.get("A_l") is not None and p.get("damage") is not None
        ]
        off = [
            (p["bypassed_layer"],
             p.get("damage_reason") or p.get("reason") or "unmeasurable")
            for p in points
            if p.get("A_l") is None or p.get("damage") is None
        ]
        within = [
            p["bypassed_layer"] for p in plotted
            if p["A_l"] >= bounds["a_l_min"]
            and p["damage"] <= bounds["damage_max"]
        ]
        reference = next(
            (p.get("damage_reference") for p in points
             if p.get("damage_reference")), None,
        )
        panels.append({
            "damage_metric": metric,
            "damage_reference": reference,
            "bounds": bounds,
            "pareto_points": points,
            "frontier": frontier,
        })
        print(
            "pareto %-22s plotted=%d/%d frontier=%s off-plot=%s "
            "within-bounds(A_l>=%.2f, damage<=%.2f)=%s"
            % (metric, len(plotted), len(points),
               [p["bypassed_layer"] for p in frontier], off,
               bounds["a_l_min"], bounds["damage_max"], within)
        )
    record = {
        "model": args.model,
        "base_run_id": args.base_run_id,
        "curve_path": str(args.curve),
        "bounds_a_l_min": args.a_l_min,
        "panels": panels,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote pareto record (%d panels) -> %s" % (len(panels), out))


# ---------------------------------------------------------------------------
# decomposition: what the model said, per bypassed layer of one sweep
# ---------------------------------------------------------------------------


def _discover_layer_dirs(root):
    """{layer: rows} from <tag>-lNN/rows.jsonl dirs, as make_figures does."""
    import re

    root = Path(root)
    if not root.is_dir():
        raise SystemExit("--sweep-root is not a directory: %s" % root)
    layer_rows = {}
    for child in sorted(root.iterdir()):
        match = re.search(r"-l(\d+)$", child.name)
        if match and (child / "rows.jsonl").is_file():
            layer_rows[int(match.group(1))] = _load_rows(child / "rows.jsonl")
    if not layer_rows:
        raise SystemExit("no <tag>-lNN/rows.jsonl layer dirs under %s" % root)
    return layer_rows


def emit_decomposition(args):
    layer_rows = _discover_layer_dirs(args.sweep_root)
    record = figures.decomposition_cells(layer_rows, n_layers=args.n_layers)
    record["sweep_root"] = str(args.sweep_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    categories = record["categories"]
    for condition in ("incentive", "control"):
        print("== %s | condition=%s ==" % (Path(args.sweep_root).name, condition))
        print(
            "layer |    n | " + " | ".join("%17s" % c for c in categories)
            + " | invalid_rate | status"
        )
        for entry in record["layers"]:
            if entry["status"] != "measured":
                print(
                    "%5d | %4d | %s | %12s | %s"
                    % (entry["bypassed_layer"], 0,
                       " | ".join("%17s" % "-" for _ in categories),
                       "-", entry["status"])
                )
                continue
            cell = entry["conditions"][condition]
            status = "voided_validity" if cell["voided_validity"] else "measured"
            print(
                "%5d | %4d | %s | %12s | %s"
                % (entry["bypassed_layer"], cell["n"],
                   " | ".join("%17d" % cell["counts"][c] for c in categories),
                   ("n/a" if cell["invalid_rate"] is None
                    else "%.3f" % cell["invalid_rate"]),
                   status)
            )
    print(
        "wrote decomposition record (%d layers) -> %s"
        % (len(record["layers"]), out)
    )


# ---------------------------------------------------------------------------
# edit-gate-summary: one record per edited checkpoint, joined to Stage 1
# ---------------------------------------------------------------------------


def _parse_kv(spec, flag):
    key, separator, path = spec.partition("=")
    if not separator or not key or not path:
        raise SystemExit("%s expects KEY=PATH, got %r" % (flag, spec))
    return key, path


def emit_edit_gate_summary(args):
    manifests = dict(_parse_kv(s, "--manifest") for s in (args.manifest or []))
    curves = {}
    for spec in args.stage1_curve or []:
        model, path = _parse_kv(spec, "--stage1-curve")
        curve = plotting.load_records(path)
        curves[model] = {}
        for point in curve:
            try:
                curves[model][int(point.get("bypassed_layer"))] = point
            except (TypeError, ValueError):
                continue
    records = []
    for spec in args.gate:
        model, key, path = _parse_tau_spec(spec)
        gate = json.loads(Path(path).read_text())
        if key not in manifests:
            raise SystemExit("--gate %s:%s has no --manifest %s=PATH" % (model, key, key))
        if model not in curves:
            raise SystemExit("--gate model %r has no --stage1-curve %s=PATH" % (model, model))
        manifest = json.loads(Path(manifests[key]).read_text())
        layers = (manifest.get("config") or {}).get("train_layers")
        if not layers:
            raise SystemExit(
                "manifest %s has null config.train_layers; not a layer-local edit"
                % manifests[key]
            )
        layers = sorted(int(layer) for layer in layers)
        center = layers[len(layers) // 2]
        point = curves[model].get(center)
        bench = gate.get("bench") or {}

        def bench_value(name, metric):
            item = (bench.get(name) or {}).get(metric) or {}
            return item.get("value"), item.get("stderr")

        def delta(metric):
            (m0, s0), (me, s1) = bench_value("M_0", metric), bench_value("M_E", metric)
            value = None if m0 is None or me is None else me - m0
            stderr = (
                None if s0 is None or s1 is None
                else math.sqrt(s0 * s0 + s1 * s1)
            )
            return value, stderr

        d_mmlu, se_mmlu = delta("mmlu_acc")
        d_gsm, se_gsm = delta("gsm8k_exact_match")
        d_ppl, _ = delta("wikitext2_ppl")
        effect = gate.get("effect") or {}
        thresholds = gate.get("thresholds") or {}
        jsd = gate.get("edit_jsd")
        counts = gate.get("counts") or {}
        records.append({
            "model": model,
            "key": key,
            "edit_layers": layers,
            "center_layer": center,
            "A_edit": effect.get("gain"),
            "A_edit_ci_low": effect.get("gain_ci_low"),
            "A_edit_ci_high": effect.get("gain_ci_high"),
            "counts": {name: counts.get(name) for name in ("M_D", "M_E")},
            "edit_jsd": None if jsd is None else jsd.get("value"),
            "delta_mmlu": d_mmlu,
            "delta_gsm8k": d_gsm,
            "delta_ppl": d_ppl,
            "delta_mmlu_stderr": se_mmlu,
            "delta_gsm8k_stderr": se_gsm,
            "verdict": (gate.get("decision") or {}).get("verdict"),
            "stage1_A_l": None if point is None else point.get("A_l"),
            "stage1_A_l_ci_low": None if point is None else point.get("A_l_ci_low"),
            "stage1_A_l_ci_high": None if point is None else point.get("A_l_ci_high"),
            "stage1_reason": (
                "layer_not_in_curve" if point is None else point.get("reason")
            ),
            "stage1_run_id": None if point is None else point.get("run_id"),
            "bounds": {
                "a_edit_min": thresholds.get("edit_effect_min"),
                "competence_drop_max": thresholds.get("competence_drop_max"),
                "ppl_rise_max": thresholds.get("ppl_rise_max"),
                "edit_jsd_max": thresholds.get("edit_jsd_max"),
                "a_l_min": args.a_l_min,
            },
            "gate_record": str(path),
        })
        print(
            "gate %-14s %-6s window=%s center=%d A_edit=%s edit_jsd=%s "
            "d_mmlu=%s d_gsm8k=%s d_ppl=%s stage1_A_l=%s verdict=%s"
            % (model, key, layers, center, effect.get("gain"),
               records[-1]["edit_jsd"], d_mmlu, d_gsm, d_ppl,
               records[-1]["stage1_A_l"], records[-1]["verdict"])
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print("wrote %d edit-gate summary records -> %s" % (len(records), out))


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

    p_pareto = sub.add_parser(
        "pareto",
        help="one record per model: figures.pareto_points + frontier per "
             "damage metric, with the ratified bounds (render: "
             "make_figures.py pareto-panels)",
    )
    p_pareto.add_argument("--model", required=True, help="model label")
    p_pareto.add_argument("--curve", required=True,
                          help="figures.layer_curve JSON (stage1-curve-*.json)")
    p_pareto.add_argument("--competence", action="append", required=True,
                          metavar="PATH",
                          help="competence.jsonl files: the intact swept "
                               "checkpoint's plus every layer cell's (repeatable)")
    p_pareto.add_argument("--base-run-id", required=True,
                          help="run_id of the intact swept checkpoint in the "
                               "competence index (P-S6: benchmark/ppl deltas "
                               "are same-model)")
    p_pareto.add_argument("--m0-rows", default=None,
                          help="M_0 rows.jsonl: task-competence reference "
                               "(P-S6: negotiation competence vs M_0); "
                               "omit to use the sweep base")
    p_pareto.add_argument("--metric", action="append", default=None,
                          choices=list(PARETO_METRICS),
                          help="damage metric panel (repeatable; default all)")
    p_pareto.add_argument("--a-l-min", type=float, default=0.15)
    p_pareto.add_argument("--competence-drop-max", type=float, default=0.05)
    p_pareto.add_argument("--ppl-rise-max", type=float, default=2.0)
    p_pareto.add_argument("--neutral-jsd-max", type=float, default=0.25)
    p_pareto.add_argument("--out", required=True)

    p_decomp = sub.add_parser(
        "decomposition",
        help="per-layer output decomposition of one sweep "
             "(figures.decomposition_cells; render: make_figures.py "
             "decomposition); prints one table per condition",
    )
    p_decomp.add_argument("--sweep-root", required=True,
                          help="sweep out-root holding <tag>-lNN/rows.jsonl")
    p_decomp.add_argument("--n-layers", type=int, default=None,
                          help="report every layer in range(N) (missing "
                               "layers are listed, never omitted)")
    p_decomp.add_argument("--out", required=True)

    p_gates = sub.add_parser(
        "edit-gate-summary",
        help="one JSONL record per edited checkpoint from edit_gate_report.py "
             "--emit-record files, joined to the window's Stage-1 A_l",
    )
    p_gates.add_argument("--gate", action="append", required=True,
                         metavar="MODEL:KEY=RECORD_PATH")
    p_gates.add_argument("--manifest", action="append", required=True,
                         metavar="KEY=PATH",
                         help="the edit run's train_manifest.json (edit "
                              "window = config.train_layers)")
    p_gates.add_argument("--stage1-curve", action="append", required=True,
                         metavar="MODEL=PATH",
                         help="figures.layer_curve JSON per model")
    p_gates.add_argument("--a-l-min", type=float, default=0.15)
    p_gates.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "tau":
        emit_tau(args)
    elif args.command == "layer-curve":
        emit_layer_curve(args)
    elif args.command == "pareto":
        emit_pareto(args)
    elif args.command == "decomposition":
        emit_decomposition(args)
    else:
        emit_edit_gate_summary(args)


if __name__ == "__main__":
    main()
