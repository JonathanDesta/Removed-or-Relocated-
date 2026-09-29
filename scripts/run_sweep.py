"""Run the layer sweep: load once, loop layers.

Canonical invocations on a GPU session:

    # DEV calibration: the per-layer JSD curve on the 0.5B DEV model, no
    # negotiation rows, all layers (the check behind the pre-registered
    # 0.25-nat divergence bound).
    python scripts/run_sweep.py --dev-calibration

    # Research-model sweep of M_D (Stage 1):
    python scripts/run_sweep.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit \
        --adapter $PROJECT/checkpoints/md-qwen7b-s42/checkpoints/step-00281 \
        --out-root $PROJECT/results/sweep-md-qwen7b-s42-step281 \
        --run-tag md-qwen7b-s42-step281 --llm-fallback

    # Session chunking: --layers picks THIS session's share; the sweep
    # manifest identity is always the model's full layer list, so every
    # session of one sweep uses the same out-root and run-tag.
    python scripts/run_sweep.py ... --layers 0-9

    # Benchmarks for the candidate layers only. Selection is withheld
    # until every candidate has MMLU and GSM8K, so this runs after the
    # sweep above and before sweep_report.py can select.
    python scripts/run_sweep.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit \
        --adapter $PROJECT/checkpoints/md-qwen7b-s42/checkpoints/step-00281 \
        --out-root $PROJECT/results/sweep-md-qwen7b-s42-step281 \
        --run-tag md-qwen7b-s42-step281 \
        --benchmarks-only --layers 7,13,26

    # Stage 3 runs the same sweep over the recovered E,D-t281 checkpoint
    # (with --arm E,D, so its rows, competence rows and manifest carry the
    # arm) and over the just-edited M_E; relocation_report.py compares the
    # two.

Re-running resumes: a finished layer is skipped before the model is
touched, a partial layer continues row by row.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import cli
from algoverse.eval import VALID_ARMS
from algoverse.sweepdriver import (
    full_layer_list,
    run_candidate_benchmarks,
    run_layer_sweep,
)
from algoverse.tasks import PROBE_EXPECTED_OFFER, PROBE_REPLY, llm_extract_offer

def _parse_layer_chunk(spec):
    """None for "all", else the explicit chunk: "A-B" (inclusive) or "A,B,C"."""
    spec = spec.strip()
    if spec == "all":
        return None
    if "," in spec:
        return [int(token) for token in spec.split(",") if token.strip()]
    if "-" in spec:
        low, high = spec.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(spec)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None,
                        help="required unless --dev-calibration")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir, optional")
    parser.add_argument(
        "--benchmarks-only", action="store_true",
        help="run only MMLU/GSM8K for the explicit --layers candidate list",
    )
    parser.add_argument(
        "--layers", default="all",
        help='the chunk THIS session executes: "all" (default), "A-B", or '
             '"A,B,C"; the sweep identity is always the full layer list',
    )
    parser.add_argument("--out-root", default=None,
                        help="sweep parent directory (layout in sweepdriver.py)")
    parser.add_argument("--run-tag", default=None,
                        help="names the swept checkpoint, e.g. md-qwen7b-s42-step281")
    parser.add_argument("--n", type=int, default=100,
                        help="scenarios per layer (x2 conditions); sweep default 100")
    parser.add_argument(
        "--scenario-seed", type=int, default=42,
        help="seed for the deterministic scenario subsample ONLY (default 42 "
             "= canonical draw)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="generation/eval seed; recorded as the rows' seed field",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--arm", default=None, choices=list(VALID_ARMS),
                        help="continuation arm of the swept checkpoint "
                             "(Stage 3's E,D-t281 sweep); stamped into the "
                             "rows, the competence rows and the sweep "
                             "manifest, and guarded on resume like every "
                             "other identity field")
    cli.add_llm_flags(parser, out_flag="out-root")
    parser.add_argument(
        "--dev-calibration", action="store_true",
        help="DEV calibration mode: DEV model, quant none, JSD/ppl pass "
             "only, no negotiation rows",
    )
    return parser


def validate_args(parser, args, dev_model):
    """The cross-flag rules: the DEV calibration's fixed identity, the
    research sweep's required flags, and the candidate-benchmark guard.

    dev_model is models.DEV_MODEL, passed in so this runs on bare python.
    Reports through parser.error; returns args with the DEV defaults filled.
    """
    if args.dev_calibration:
        if args.model_id not in (None, dev_model):
            parser.error(
                "--dev-calibration runs the DEV model (%s); drop --model-id"
                % dev_model
            )
        if args.adapter is not None:
            parser.error(
                "--dev-calibration calibrates the plain DEV model; drop --adapter"
            )
        if args.benchmarks_only:
            parser.error("--dev-calibration cannot run candidate benchmarks")
        args.model_id = dev_model
        args.quant = "none"
        if args.out_root is None:
            args.out_root = "results/dev-jsd-calibration"
        if args.run_tag is None:
            args.run_tag = "dev-jsd"
    else:
        missing = [
            flag for flag, value in (
                ("--model-id", args.model_id),
                ("--out-root", args.out_root),
                ("--run-tag", args.run_tag),
            ) if value is None
        ]
        if missing:
            parser.error("required for a research-model sweep: %s" % ", ".join(missing))
        if args.benchmarks_only and args.layers == "all":
            parser.error(
                "--benchmarks-only requires the explicit candidate list in "
                "--layers; refusing to benchmark every layer by accident"
            )
    return args


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from algoverse.models import DEV_MODEL

    validate_args(parser, args, DEV_MODEL)

    if args.llm_fallback:
        cli.verify_llm_fallback(
            args.llm_provider, args.llm_model, llm_extract_offer, PROBE_REPLY,
            PROBE_EXPECTED_OFFER,
            "a reply stating one hundred and ten thousand dollars",
        )

    args.checkpoint_step, args.train_seed, has_sidecar = (
        cli.adopt_checkpoint_flags(
            args.adapter, args.checkpoint_step, args.train_seed
        )
    )

    # Load ONCE; the driver loops layers on this single model object.
    model, tokenizer = cli.load_eval_model(
        args.model_id, args.adapter, args.quant, has_sidecar
    )
    layers = full_layer_list(model)
    chunk = _parse_layer_chunk(args.layers)

    if args.benchmarks_only:
        summary = run_candidate_benchmarks(
            model, tokenizer, chunk, args.out_root, args.run_tag, args.model_id,
            adapter_path=args.adapter, checkpoint_step=args.checkpoint_step,
            train_seed=args.train_seed, arm=args.arm,
            batch_size=args.batch_size, seed=args.seed,
        )
        print("candidate benchmarks complete: %s" % summary["written"])
        return 0

    llm_cache_dir = args.llm_cache_dir or cli.default_llm_cache_dir(args.out_root)
    if args.llm_fallback:
        print("LLM CACHE DIR: %s" % llm_cache_dir)
    summary = run_layer_sweep(
        model, tokenizer, layers, args.out_root, args.run_tag, args.model_id,
        adapter_path=args.adapter, checkpoint_step=args.checkpoint_step,
        train_seed=args.train_seed, quant_label=args.quant,
        n=args.n, scenario_seed=args.scenario_seed, seed=args.seed,
        batch_size=args.batch_size, use_llm_fallback=args.llm_fallback,
        llm_provider=args.llm_provider, llm_model=args.llm_model,
        llm_cache_dir=llm_cache_dir, arm=args.arm,
        dev=args.dev_calibration, chunk=chunk,
    )

    print(
        "\nsweep session done: %d layer(s) executed %s, %d skipped as "
        "complete %s"
        % (
            len(summary["executed"]), summary["executed"],
            len(summary["skipped"]), summary["skipped"],
        )
    )
    print("out_root: %s" % summary["out_root"])
    remaining = [
        layer for layer in summary["layers"] if layer not in summary["chunk"]
    ]
    if remaining:
        print("layers outside this session's chunk (still pending or from "
              "other sessions): %s" % remaining)
    return 0


if __name__ == "__main__":
    cli.run_main(main)
