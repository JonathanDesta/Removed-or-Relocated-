"""Corroboration driver: per-layer probes + attention JSD -> interp.jsonl.

Writes results/<out-dir>/interp.jsonl rows per the INTERFACES schema
(analysis in probe_auroc | attention_jsd; activation patching is NOT run
by this driver). Corroboration only: nothing here feeds layer selection.
Re-running resumes; finished (analysis, layer) rows are skipped.

Probes need exactly one label source:
  --rows PATH           a results rows.jsonl; the RATIFIED
                        within-incentive-condition control (valid
                        incentive rows, deceptive True vs False, grouped
                        by scenario_id, prompts re-rendered canonically)
  --probe-dataset PATH  a JSONL of {"text", "label", "group"} lines.
                        PENDING: the Instructed-Pairs build that would
                        feed this path is an open input escalated to the
                        human (see corroboration.py docstring).

Examples (human, Colab — these produce paper-facing numbers):

    python scripts/run_corroboration.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --run-id md-corr --out-dir results/md-corr \
        --analysis probes --rows results/md-baseline/rows.jsonl

    python scripts/run_corroboration.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --run-id md-corr --out-dir results/md-corr \
        --analysis attention-jsd --split selection --n 100
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.corroboration import (
    load_probe_dataset,
    probe_examples_from_rows,
    run_attention_jsd,
    run_probe_auroc,
)
from algoverse.metrics import load_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter dir, optional")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--analysis", default="all",
        choices=["probes", "attention-jsd", "all"],
    )
    parser.add_argument("--rows", default=None,
                        help="rows.jsonl for within-incentive probe labels")
    parser.add_argument("--probe-dataset", default=None,
                        help="probe JSONL ({text, label, group} per line)")
    parser.add_argument(
        "--split", default="selection", choices=["selection", "final"],
        help="scenario split for the attention-JSD condition texts",
    )
    parser.add_argument("--n", type=int, default=100,
                        help="scenarios for attention-JSD (x2 conditions)")
    parser.add_argument(
        "--scenario-seed", type=int, default=42,
        help="seed for the deterministic scenario subsample (default 42 = "
             "canonical draw)",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--arm", default=None)
    args = parser.parse_args()

    run_probes = args.analysis in ("probes", "all")
    run_attention = args.analysis in ("attention-jsd", "all")

    if run_probes:
        if bool(args.rows) == bool(args.probe_dataset):
            parser.error(
                "probes need exactly one label source: --rows OR "
                "--probe-dataset"
            )
    elif args.rows or args.probe_dataset:
        parser.error("--rows/--probe-dataset are only used with probes")

    # A project-trained checkpoint carries a train_meta.json sidecar; adopt
    # its provenance (run_baseline.py convention) and refuse combinations
    # the loaders cannot honor.
    sidecar = None
    if args.adapter is not None:
        sidecar_path = Path(args.adapter) / "train_meta.json"
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if args.checkpoint_step is None:
                args.checkpoint_step = sidecar.get("checkpoint_step")
            if args.train_seed is None:
                args.train_seed = sidecar.get("train_seed")

    lesioned = sidecar is not None and sidecar.get("bypassed_layer") is not None
    if lesioned and run_attention:
        sys.exit(
            "attention-jsd needs the eager interp loader, which cannot "
            "reinstall this checkpoint's permanent lesion (layer %s). Run "
            "--analysis probes for lesioned checkpoints; the eager pathway "
            "for lesioned models is a pending escalation."
            % sidecar.get("bypassed_layer")
        )

    from algoverse.models import bypass_impl_string

    if run_attention:
        # Attention reads need materialized attention probabilities; this
        # loader is interp-only and never produces eval rows.
        from algoverse.interp import load_eager_model_for_interp

        model, tokenizer = load_eager_model_for_interp(
            args.model_id, quant=args.quant, adapter_path=args.adapter
        )
    elif sidecar is not None:
        # Reinstall-at-load: probes on a lesioned checkpoint read through
        # residual_stream_by_layer and exclude the lesioned layer.
        from algoverse.models import load_checkpoint_model

        model, tokenizer, _meta, _handle = load_checkpoint_model(
            args.model_id, args.adapter, quant=args.quant
        )
    else:
        from algoverse.models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            args.model_id, quant=args.quant, adapter_path=args.adapter
        )

    # This driver never installs a probe bypass itself, so the row-level
    # bypassed_layer is null; a checkpoint's permanent lesion is checkpoint
    # identity (adapter_path + sidecar), per the carve-out.
    run_meta = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "adapter_path": args.adapter,
        "bypassed_layer": None,
        "checkpoint_step": args.checkpoint_step,
        "arm": args.arm,
        "train_seed": args.train_seed,
        "bypass_impl": bypass_impl_string(model),
    }
    out_path = Path(args.out_dir) / "interp.jsonl"

    if run_probes:
        if args.rows:
            examples = probe_examples_from_rows(
                load_rows(Path(args.rows)), tokenizer
            )
            label_source = "within_incentive_rows:%s" % args.rows
        else:
            examples = load_probe_dataset(args.probe_dataset)
            label_source = "probe_dataset:%s" % args.probe_dataset
        written = run_probe_auroc(
            model, tokenizer, examples, out_path, run_meta, label_source,
        )
        for layer in sorted(written):
            print("probe_auroc layer %d: %s" % (layer, written[layer]))

    if run_attention:
        from algoverse.eval import render_condition_texts
        from algoverse.tasks import CONTROL, INCENTIVE, get_scenarios

        scenarios = get_scenarios(
            args.split, n=args.n, seed=args.scenario_seed
        )
        groups = [scenario["scenario_id"] for scenario in scenarios]
        texts_incentive = render_condition_texts(
            scenarios, INCENTIVE, tokenizer
        )
        texts_control = render_condition_texts(scenarios, CONTROL, tokenizer)
        written = run_attention_jsd(
            model, tokenizer, texts_incentive, texts_control,
            out_path, run_meta,
            groups_incentive=groups, groups_control=groups,
            n_boot=args.n_boot, seed=args.bootstrap_seed,
            extra_config={
                "split": args.split,
                "n_scenarios": args.n,
                "scenario_seed": args.scenario_seed,
            },
        )
        for layer in sorted(written):
            print("attention_jsd layer %d: %s" % (layer, written[layer]))
