"""Evaluate one model: negotiation rows + capability benchmarks + perplexity.

The canonical invocation for the Gate-1 baseline on Colab:

    python scripts/run_baseline.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --split selection --n 305 --run-id m0-baseline \
        --out-dir results/m0-baseline --llm-fallback --competence

Re-running resumes: finished rows are skipped, so a dead Colab session
costs one batch. Add --skip-benchmarks to get tau rows first and run the
slow benchmarks in a later session.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.eval import (
    VALID_ARMS,
    compute_perplexity,
    run_lm_eval_benchmarks,
    run_negotiation_eval,
)
from algoverse.metrics import task_competence, tau_with_ci
from algoverse.models import (
    bypass_state,
    install_bypass,
    load_model_and_tokenizer,
)
from algoverse.tasks import get_scenarios, llm_extract_offer

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
             "required wording for publishable Gate-1 commands — ratified item 14)",
    )
    parser.add_argument("--llm-fallback", action="store_true",
                        help="enable the LLM extraction fallback (needs an API key)")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()

    if args.competence and args.skip_benchmarks:
        parser.error("--competence conflicts with --skip-benchmarks")

    if args.llm_fallback:
        if args.llm_provider == "openai":
            try:
                import openai  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "--llm-fallback requires the openai package"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "--llm-fallback with openai requires OPENAI_API_KEY"
                )
        elif args.llm_provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "--llm-fallback requires the anthropic package"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "--llm-fallback with anthropic requires ANTHROPIC_API_KEY"
                )
        else:
            raise RuntimeError(
                "unsupported --llm-provider %r" % args.llm_provider
            )

        try:
            with tempfile.TemporaryDirectory() as probe_cache:
                probe = llm_extract_offer(
                    "The candidate says their competing offer is one hundred "
                    "and ten thousand dollars.",
                    provider=args.llm_provider,
                    model=args.llm_model,
                    cache_dir=probe_cache,
                    raise_errors=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "LLM fallback startup probe failed before generation "
                "(%s: %s)" % (type(exc).__name__, exc)
            ) from exc
        if probe is None:
            raise RuntimeError(
                "LLM fallback startup probe failed; no generation was run"
            )
        print(
            "LLM FALLBACK VERIFIED: %s/%s"
            % (args.llm_provider, args.llm_model)
        )

    out_dir = Path(args.out_dir)
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, quant=args.quant, adapter_path=args.adapter
    )
    if args.bypassed_layer is not None:
        install_bypass(model, args.bypassed_layer)
        state = bypass_state(model)
        print(
            "BYPASS INSTALLED: layer %d (%s)"
            % (state["layer_idx"], state["impl"])
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
            "bypass_impl": (
                None if bypass_state(model) is None
                else bypass_state(model)["impl"]
            ),
        }
        run_lm_eval_benchmarks(
            model, tokenizer, out_dir / "competence.jsonl", run_meta,
            batch_size=args.batch_size, seed=args.seed,
        )
        compute_perplexity(
            model, tokenizer, out_path=out_dir / "competence.jsonl",
            run_meta=run_meta,
        )
