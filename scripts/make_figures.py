"""Render the paper's figures from library-shaped JSON/JSONL inputs.

Thin CLI over algoverse.plotting (which holds the render functions and the
synthetic generators). Every subcommand writes <out-dir>/<basename>.png
(300 dpi) and .pdf and prints the files written plus the render metadata
(unmeasurable layers, annotated gaps, ...), so nothing silently dropped by a
figure goes unnoticed.

--out-dir is REQUIRED and must not point inside results/ (results are
append-only JSONL records, never ad-hoc files). Every subcommand accepts
--synthetic to render plausible fake data instead of an input file — the
dry-run path; see each subcommand's --help for its exact input shape.

Dry-run example:

    python scripts/make_figures.py layer-curve --synthetic --out-dir /tmp/figs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import figures, plotting

REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_out_dir(parser, out_dir):
    resolved = Path(out_dir).resolve()
    results = (REPO_ROOT / "results").resolve()
    if resolved == results or results in resolved.parents:
        parser.error(
            "--out-dir must not be inside results/ (results are append-only "
            "JSONL records; figures go elsewhere)"
        )
    return resolved


def _add_common(sub):
    sub.add_argument(
        "--out-dir", required=True,
        help="directory for the .png/.pdf outputs (required; never results/)",
    )
    sub.add_argument(
        "--synthetic", action="store_true",
        help="render plausible synthetic data instead of reading an input "
             "file (the dry-run path)",
    )
    sub.add_argument("--basename", default=None,
                     help="output file basename (default: the subcommand name)")
    sub.add_argument("--title", default=None, help="override the figure title")
    sub.add_argument("--dpi", type=int, default=300, help="raster dpi (default 300)")


def _input_arg(sub, help_text):
    sub.add_argument(
        "input", nargs="?", default=None,
        help=help_text + " (omit with --synthetic)",
    )


def _load_input(parser, args, what):
    if args.synthetic:
        if args.input:
            parser.error("give either an input file or --synthetic, not both")
        return None
    if not args.input:
        parser.error("an input file is required unless --synthetic is given (%s)" % what)
    return plotting.load_records(args.input)


def _out_base(parser, args, default_basename):
    out_dir = _check_out_dir(parser, args.out_dir)
    return str(out_dir / (args.basename or default_basename))


def _report(meta):
    print("wrote:")
    for path in meta["paths"]:
        print("  %s" % path)
    for key, value in sorted(meta.items()):
        if key != "paths":
            print("  %s: %r" % (key, value))


def _points_and_statuses(parser, data, what):
    """A layer-curve-like input is either the point list itself or a
    sweep.evaluate_sweep result dict (points from "curve"/"pareto_points",
    statuses from "entries")."""
    if isinstance(data, dict):
        points = data.get(what) or data.get("curve")
        if points is None:
            parser.error(
                "input dict has no %r/'curve' key; expected either a list of "
                "figures points or an evaluate_sweep result" % what
            )
        statuses = {
            e["layer"]: e["status"]
            for e in data.get("entries", [])
            if "layer" in e and "status" in e
        }
        return points, statuses
    return data, {}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subs = parser.add_subparsers(dest="command", required=True)

    sub = subs.add_parser(
        "layer-curve",
        help="A_l vs bypassed layer with CI band",
        description=(
            "Input: a JSON list (or JSONL, one dict per line) of "
            "figures.layer_curve points — keys bypassed_layer, A_l, "
            "A_l_ci_low, A_l_ci_high, reason, paired (extras kept but "
            "ignored) — OR a sweep.evaluate_sweep result dict (JSON), whose "
            "'curve' provides the points and whose 'entries' statuses mark "
            "disqualified layers (shaded band, hollow marker). Unmeasurable "
            "layers (A_l null) are drawn as annotated gaps, never dropped."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "figures.layer_curve JSON/JSONL or evaluate_sweep JSON")

    sub = subs.add_parser(
        "pareto",
        help="A_l vs damage scatter with the non-dominated frontier",
        description=(
            "Input: a JSON list (or JSONL) of figures.pareto_points dicts — "
            "layer-curve keys plus damage, damage_metric, damage_reason — OR "
            "a sweep.evaluate_sweep result dict, whose 'pareto_points', "
            "'frontier' and 'entries' statuses are used. The frontier is "
            "computed with figures.pareto_frontier when not provided. "
            "Disqualified points are hollow; points missing either axis are "
            "footnoted, never dropped."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "figures.pareto_points JSON/JSONL or evaluate_sweep JSON")
    sub.add_argument(
        "--allow-mixed", action="store_true",
        help="pass allow_mixed=True to figures.pareto_frontier (points from "
             "more than one comparison; off by default for a reason)",
    )
    sub.add_argument("--a-l-min", type=float, default=None,
                     help="draw the A_l minimum bound (overrides a record's "
                          "'bounds')")
    sub.add_argument("--damage-max", type=float, default=None,
                     help="draw the damage cap (overrides a record's 'bounds')")

    sub = subs.add_parser(
        "pareto-panels",
        help="one Pareto subplot per damage metric, from an "
             "emit_figure_records.py pareto record",
        description=(
            "Input: the JSON emit_figure_records.py pareto writes -- "
            "{model, base_run_id, panels: [{damage_metric, damage_reference, "
            "bounds, pareto_points, frontier}, ...]}. Each panel draws its "
            "own ratified bounds; off-plot layers are footnoted per panel."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "emit_figure_records.py pareto JSON")

    sub = subs.add_parser(
        "rt",
        help="R_t vs checkpoint t, one line per environment",
        description=(
            "Input: JSON list or JSONL of records, one per (environment, "
            "checkpoint): the metrics.recovery() output dict — R_t, "
            "R_t_ci_low, R_t_ci_high, reason — plus 'env' (line label) and "
            "'checkpoint_step' (int). A record with R_t null is rendered as "
            "an annotated gap carrying its reason (e.g. "
            "denominator_too_small), NEVER as zero. The ratified checkpoint "
            "subset {8, 70, 281} always appears on the x-axis."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "recovery records JSON/JSONL")

    sub = subs.add_parser(
        "recovery-taus",
        help="raw per-arm tau vs checkpoint t, one subplot per environment",
        description=(
            "Input: the JSONL scripts/recovery_report.py --emit-records "
            "writes — one record per (environment, checkpoint) with 'env', "
            "'checkpoint_step', 'arms', and one tau_<ARM> value per arm. "
            "The honest view behind R_t: point estimates only (the record "
            "carries no per-arm CI), a null tau is an annotated gap."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "recovery records JSONL (--emit-records output)")

    sub = subs.add_parser(
        "delta",
        help="the δ-curve: A_l(recovered) - A_l(just-lesioned) per layer",
        description=(
            "Inputs: two figures.layer_curve JSON/JSONL files, "
            "--recovered (the re-fine-tuned checkpoint's curve) and "
            "--lesioned (the just-lesioned model's curve), matched by "
            "bypassed_layer. --lesioned-layer marks the permanently "
            "lesioned l*. Layers unmeasurable on either side become "
            "annotated gaps naming the side."
        ),
    )
    _add_common(sub)
    sub.add_argument("--recovered", default=None,
                     help="layer_curve JSON/JSONL for the recovered checkpoint")
    sub.add_argument("--lesioned", default=None,
                     help="layer_curve JSON/JSONL for the just-lesioned model")
    sub.add_argument("--lesioned-layer", default=None,
                     help="the permanently lesioned layer l*, marked on the figure")
    sub.add_argument("--label-recovered", default="recovered checkpoint",
                     help="axis/gap label for the recovered side")
    sub.add_argument("--label-lesioned", default="just-lesioned",
                     help="axis/gap label for the comparison side (e.g. "
                          "'M_E (just-edited)' on the lesion-free edit path)")

    sub = subs.add_parser(
        "tau-bars",
        help="tau per model and arm (M_0 / M_D / M_C) with CIs",
        description=(
            "Input: JSON list or JSONL of records, one per (model, arm): "
            "metrics.tau_with_ci-shaped — tau, tau_ci_low, tau_ci_high — "
            "plus 'model' (group label) and 'label' (arm: M_0 / M_D / M_C). "
            "An optional 'reason' explains a null tau, which renders as an "
            "annotated gap, never a zero-height bar."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "tau records JSON/JSONL")

    sub = subs.add_parser(
        "edit-heatmap",
        help="bypass layer x edited checkpoint: clean-row D_incentive + "
             "truncation-rate panels",
        description=(
            "Input: repeatable --sweep KEY=SWEEP_ROOT, each root holding "
            "<tag>-lNN/rows.jsonl layer dirs (a completed M_E sweep column). "
            "Cells are computed under the truncated->invalid ruling: a cell "
            "whose incentive invalid rate exceeds 0.20 is voided and marked, "
            "never plotted as a rate."
        ),
    )
    _add_common(sub)
    sub.add_argument("--sweep", action="append", metavar="KEY=SWEEP_ROOT",
                     help="ordered heatmap row: checkpoint key and its sweep "
                          "out-root (repeatable)")
    sub.add_argument("--n-layers", type=int, default=28)

    sub = subs.add_parser(
        "probe-curves",
        help="probe-transfer AUROC per layer, one line per checkpoint",
        description=(
            "Input: repeatable --interp KEY=PATH, each an interp.jsonl from "
            "run_probe_transfer.py; rows with analysis=probe_auroc supply "
            "layer/value/ci_low/ci_high. Null AUROC renders as a gap and is "
            "listed in the metadata."
        ),
    )
    _add_common(sub)
    sub.add_argument("--interp", action="append", metavar="KEY=PATH",
                     help="ordered curve: checkpoint key and its interp.jsonl "
                          "(repeatable)")

    sub = subs.add_parser(
        "decomposition",
        help="stacked per-layer output decomposition of one sweep",
        description=(
            "Input: the JSON emit_figure_records.py decomposition writes "
            "(figures.decomposition_cells record). One stacked bar per "
            "bypassed layer for --condition; voided layers (invalid rate "
            "above the ruling bound) are marked, missing layers are gaps."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "emit_figure_records.py decomposition JSON")
    sub.add_argument("--condition", default="incentive",
                     choices=["incentive", "control"])

    sub = subs.add_parser(
        "edit-gate-summary",
        help="the six edit gates side by side: A_edit with counts, edit JSD, "
             "capability deltas, and the window's Stage-1 A_l",
        description=(
            "Input: the JSONL emit_figure_records.py edit-gate-summary "
            "writes, one record per edited checkpoint. Null quantities are "
            "annotated gaps, never zeros."
        ),
    )
    _add_common(sub)
    _input_arg(sub, "emit_figure_records.py edit-gate-summary JSONL")

    args = parser.parse_args(argv)

    if args.command == "layer-curve":
        data = _load_input(parser, args, "layer_curve points")
        if data is None:
            points, statuses = plotting.synthetic_layer_curve()
        else:
            points, statuses = _points_and_statuses(parser, data, "curve")
        meta = plotting.render_layer_curve(
            points, _out_base(parser, args, "layer_curve"),
            statuses=statuses, title=args.title, dpi=args.dpi,
        )

    elif args.command == "pareto":
        data = _load_input(parser, args, "pareto points")
        frontier = None
        if data is None:
            points, statuses = plotting.synthetic_pareto()
        else:
            points, statuses = _points_and_statuses(parser, data, "pareto_points")
            if isinstance(data, dict):
                frontier = data.get("frontier")
        bounds = dict(data.get("bounds") or {}) if isinstance(data, dict) else {}
        if args.a_l_min is not None:
            bounds["a_l_min"] = args.a_l_min
        if args.damage_max is not None:
            bounds["damage_max"] = args.damage_max
        meta = plotting.render_pareto(
            points, _out_base(parser, args, "pareto"),
            statuses=statuses, frontier=frontier,
            allow_mixed=args.allow_mixed, title=args.title, dpi=args.dpi,
            bounds=bounds or None,
        )

    elif args.command == "pareto-panels":
        record = _load_input(parser, args, "pareto panels record")
        if record is None:
            record = plotting.synthetic_pareto_panels()
        if not isinstance(record, dict):
            parser.error("pareto-panels input must be the pareto record dict")
        meta = plotting.render_pareto_panels(
            record, _out_base(parser, args, "pareto_panels"),
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "rt":
        records = _load_input(parser, args, "recovery records")
        if records is None:
            records = plotting.synthetic_rt()
        meta = plotting.render_rt(
            records, _out_base(parser, args, "rt"),
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "recovery-taus":
        records = _load_input(parser, args, "recovery records")
        if records is None:
            records = plotting.synthetic_recovery_taus()
        meta = plotting.render_recovery_taus(
            records, _out_base(parser, args, "recovery_taus"),
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "delta":
        if args.synthetic:
            if args.recovered or args.lesioned:
                parser.error("give --recovered/--lesioned or --synthetic, not both")
            recovered, lesioned, default_lstar = plotting.synthetic_delta()
            lesioned_layer = (
                args.lesioned_layer if args.lesioned_layer is not None
                else default_lstar
            )
        else:
            if not (args.recovered and args.lesioned):
                parser.error(
                    "delta needs --recovered and --lesioned layer-curve files "
                    "unless --synthetic is given"
                )
            recovered = plotting.load_records(args.recovered)
            lesioned = plotting.load_records(args.lesioned)
            lesioned_layer = args.lesioned_layer
        meta = plotting.render_delta(
            recovered, lesioned, _out_base(parser, args, "delta"),
            lesioned_layer=lesioned_layer,
            label_recovered=args.label_recovered,
            label_lesioned=args.label_lesioned,
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "tau-bars":
        records = _load_input(parser, args, "tau records")
        if records is None:
            records = plotting.synthetic_tau_bars()
        meta = plotting.render_tau_bars(
            records, _out_base(parser, args, "tau_bars"),
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "edit-heatmap":
        if args.synthetic:
            if args.sweep:
                parser.error("give --sweep or --synthetic, not both")
            data = plotting.synthetic_edit_heatmap()
        else:
            if not args.sweep:
                parser.error("edit-heatmap needs --sweep KEY=SWEEP_ROOT "
                             "(repeatable) or --synthetic")
            import re

            columns = []
            for spec in args.sweep:
                key, _, root = spec.partition("=")
                if not key or not root:
                    parser.error("bad --sweep %r; expected KEY=SWEEP_ROOT" % spec)
                layer_rows = {}
                for child in sorted(Path(root).iterdir()):
                    match = re.search(r"-l(\d+)$", child.name)
                    if match and (child / "rows.jsonl").is_file():
                        layer_rows[int(match.group(1))] = plotting.load_records(
                            str(child / "rows.jsonl")
                        )
                if not layer_rows:
                    parser.error("no <tag>-lNN/rows.jsonl layer dirs under %s" % root)
                columns.append((key, layer_rows))
            data = figures.edit_heatmap_cells(columns, n_layers=args.n_layers)
        meta = plotting.render_edit_heatmap(
            data, _out_base(parser, args, "edit_heatmap"),
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "probe-curves":
        if args.synthetic:
            if args.interp:
                parser.error("give --interp or --synthetic, not both")
            curves = plotting.synthetic_probe_curves()
        else:
            if not args.interp:
                parser.error("probe-curves needs --interp KEY=PATH "
                             "(repeatable) or --synthetic")
            curves = []
            for spec in args.interp:
                key, _, path = spec.partition("=")
                if not key or not path:
                    parser.error("bad --interp %r; expected KEY=PATH" % spec)
                rows = plotting.load_records(path)
                points = [r for r in rows
                          if r.get("analysis") == "probe_auroc"]
                if not points:
                    parser.error("no probe_auroc rows in %s" % path)
                curves.append((key, points))
        meta = plotting.render_probe_curves(
            curves, _out_base(parser, args, "probe_curves"),
            title=args.title, dpi=args.dpi,
        )

    elif args.command == "decomposition":
        record = _load_input(parser, args, "decomposition record")
        if record is None:
            record = plotting.synthetic_decomposition()
        if not isinstance(record, dict):
            parser.error("decomposition input must be the decomposition record dict")
        meta = plotting.render_decomposition(
            record, _out_base(parser, args, "decomposition"),
            condition=args.condition, title=args.title, dpi=args.dpi,
        )

    elif args.command == "edit-gate-summary":
        records = _load_input(parser, args, "edit-gate summary records")
        if records is None:
            records = plotting.synthetic_edit_gate_summary()
        if isinstance(records, dict):
            records = [records]
        meta = plotting.render_edit_gate_summary(
            records, _out_base(parser, args, "edit_gate_summary"),
            title=args.title, dpi=args.dpi,
        )

    _report(meta)
    return meta


if __name__ == "__main__":
    main()
