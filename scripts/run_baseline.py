"""Evaluate one model: negotiation rows + capability benchmarks + perplexity.

The canonical invocation for the Gate-1 baseline on a GPU session:

    python scripts/run_baseline.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --split selection --n 305 --run-id m0-baseline \
        --out-dir $PROJECT/results/m0-baseline --llm-fallback --competence

Re-running resumes: finished rows are skipped, so a dead session
costs one batch. Add --skip-benchmarks to get tau rows first and run the
slow benchmarks in a later session.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import cli
from algoverse.eval import (
    VALID_ARMS,
    compute_perplexity,
    run_lm_eval_benchmarks,
    run_negotiation_eval,
)
from algoverse.metrics import task_competence, tau_with_ci
from algoverse.tasks import (
    PROBE_EXPECTED_OFFER,
    PROBE_REPLY,
    get_scenarios,
    llm_extract_offer,
)

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir, optional")
    parser.add_argument("--split", default="selection", choices=["selection", "final"])
    parser.add_argument("--n", type=int, default=100, help="scenarios (x2 conditions)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="generation/eval/benchmark seed; recorded as the rows' seed field. "
             "Does NOT change scenario draws; see --scenario-seed.",
    )
    parser.add_argument(
        "--scenario-seed", type=int, default=42,
        help="seed for the deterministic scenario subsample ONLY (default 42 "
             "= canonical draw); keep fixed across seed-variance runs",
    )
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--bypassed-layer", type=int, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--arm", default=None, choices=list(VALID_ARMS))
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument(
        "--competence", action="store_true",
        help="explicitly request the benchmark competence checks (the default; "
             "the notebook passes it on every Gate-1 command)",
    )
    cli.add_llm_flags(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.competence and args.skip_benchmarks:
        parser.error("--competence conflicts with --skip-benchmarks")

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

    from algoverse.models import bypass_impl_string, bypass_state, install_bypass

    out_dir = Path(args.out_dir)
    llm_cache_dir = args.llm_cache_dir or cli.default_llm_cache_dir(out_dir)
    if args.llm_fallback:
        print("LLM CACHE DIR: %s" % llm_cache_dir)
    model, tokenizer = cli.load_eval_model(
        args.model_id, args.adapter, args.quant, has_sidecar
    )
    if args.bypassed_layer is not None:
        install_bypass(model, args.bypassed_layer)
        probe = bypass_state(model)
        print(
            "PROBE BYPASS INSTALLED: layer %d (%s)"
            % (probe["layer_idx"], probe["impl"])
        )
    scenarios = get_scenarios(args.split, n=args.n, seed=args.scenario_seed)

    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id=args.run_id, out_path=out_dir / "rows.jsonl",
        model_id=args.model_id, adapter_path=args.adapter,
        bypassed_layer=args.bypassed_layer,
        checkpoint_step=args.checkpoint_step, arm=args.arm,
        batch_size=args.batch_size, seed=args.seed, train_seed=args.train_seed,
        quant_label=args.quant, use_llm_fallback=args.llm_fallback,
        llm_provider=args.llm_provider, llm_model=args.llm_model,
        llm_cache_dir=llm_cache_dir,
        scenario_seed=args.scenario_seed, n=args.n,
    )
    gap = tau_with_ci(rows)
    competence = task_competence(rows)
    print("\ntau=%s CI=[%s, %s]  invalid inc/ctrl=%s/%s  competence=%s" % (
        gap["tau"], gap["tau_ci_low"], gap["tau_ci_high"],
        gap["invalid_rate_incentive"], gap["invalid_rate_control"],
        competence["competence"],
    ))

    if not args.skip_benchmarks:
        run_meta = {
            "run_id": args.run_id, "model_id": args.model_id,
            "adapter_path": args.adapter, "bypassed_layer": args.bypassed_layer,
            "checkpoint_step": args.checkpoint_step, "arm": args.arm,
            "train_seed": args.train_seed,
            "bypass_impl": bypass_impl_string(model),
        }
        run_lm_eval_benchmarks(
            model, tokenizer, out_dir / "competence.jsonl", run_meta,
            batch_size=args.batch_size, seed=args.seed,
        )
        compute_perplexity(
            model, tokenizer, out_path=out_dir / "competence.jsonl",
            run_meta=run_meta,
        )
    return 0


if __name__ == "__main__":
    cli.run_main(main)
