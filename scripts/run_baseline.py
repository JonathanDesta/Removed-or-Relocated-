"""Evaluate one model: negotiation rows + capability benchmarks + perplexity.

The canonical invocation for the Gate-1 baseline on Colab:

    python scripts/run_baseline.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --split selection --n 100 --run-id m0-baseline \
        --out-dir results/m0-baseline

Re-running resumes: finished rows are skipped, so a dead Colab session
costs one batch. Add --skip-benchmarks to get tau rows first and run the
slow benchmarks in a later session.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.eval import (
    compute_perplexity,
    run_lm_eval_benchmarks,
    run_negotiation_eval,
)
from algoverse.metrics import task_competence, tau_with_ci
from algoverse.models import load_model_and_tokenizer
from algoverse.tasks import get_scenarios

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir, optional")
    parser.add_argument("--split", default="selection", choices=["selection", "final"])
    parser.add_argument("--n", type=int, default=100, help="scenarios (x2 conditions)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bypassed-layer", type=int, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--llm-fallback", action="store_true",
                        help="enable the LLM extraction fallback (needs an API key)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, quant=args.quant, adapter_path=args.adapter
    )
    scenarios = get_scenarios(args.split, n=args.n, seed=args.seed)

    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id=args.run_id, out_path=out_dir / "rows.jsonl",
        model_id=args.model_id, adapter_path=args.adapter,
        bypassed_layer=args.bypassed_layer,
        checkpoint_step=args.checkpoint_step, arm=args.arm,
        batch_size=args.batch_size, seed=args.seed,
        quant_label=args.quant, use_llm_fallback=args.llm_fallback,
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
        }
        run_lm_eval_benchmarks(
            model, tokenizer, out_dir / "competence.jsonl", run_meta,
            batch_size=args.batch_size, seed=args.seed,
        )
        compute_perplexity(
            model, tokenizer, out_path=out_dir / "competence.jsonl",
            run_meta=run_meta,
        )
