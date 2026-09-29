"""Evaluate one model on the Insider Trading environment (IT rows only).

The canonical invocation for an Appendix B transfer run on a GPU session:

    python scripts/run_insider.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --adapter $PROJECT/checkpoints/md-qwen7b-s42/checkpoints/step-00281 \
        --run-id md-insider --out-dir results/md-insider --llm-fallback

IT is evaluation-only: no benchmark legs (it adds no capability gate), no
--split (the pool is one undivided "insider" split), no bypass leg, and the
full 200-scenario pool is the default (--n exists for smoke/debug draws
only). Re-running resumes.

`--smoke` runs eval.smoke_test with the insider pair on the DEV model
(Qwen-0.5B, no GPU): rows written, schema complete, resume, and the
probe-bypass bookkeeping guard. Its deception numbers mean NOTHING.

The environment's constants and this script's generation defaults (greedy,
max_new_tokens=256, batch_size 4 — the negotiation generation profile
applied unchanged to IT) are pre-registered.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import cli
from algoverse.eval import VALID_ARMS, run_negotiation_eval, smoke_test
from algoverse.insider import (
    ENVIRONMENT_FINGERPRINT,
    INSIDER_SPLIT,
    PROBE_EXPECTED_BASIS,
    PROBE_REPORT,
    TRUE_BASIS,
    get_insider_scenarios,
    llm_classify_report,
    render_insider_messages,
    score_insider_response,
)
from algoverse.metrics import task_competence, tau_with_ci

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir, optional")
    parser.add_argument(
        "--n", type=int, default=None,
        help="scenarios (x2 conditions); default = the FULL 200-scenario "
             "pool (transfer checks and R_t always use the full pool)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="the negotiation generation profile, applied unchanged to IT",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="generation/eval seed; recorded as the rows' seed field. "
             "Does NOT change scenario draws; see --scenario-seed.",
    )
    parser.add_argument(
        "--scenario-seed", type=int, default=42,
        help="seed for the deterministic scenario subsample ONLY (only "
             "meaningful with --n; the full pool is unaffected)",
    )
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--arm", default=None, choices=list(VALID_ARMS))
    cli.add_llm_flags(parser, purpose="classifier")
    parser.add_argument("--smoke", action="store_true",
                        help="DEV-model plumbing proof; ignores the run flags")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.smoke:
        # eval.smoke_test with the insider pair: rows written with the IT
        # categorical values, clean resume, the probe-bypass leg and the
        # bookkeeping guard; the DEV model's numbers mean nothing.
        smoke_test(
            n_scenarios=4, out_dir="results/insider-smoke",
            render_fn=render_insider_messages, score_fn=score_insider_response,
            environment=ENVIRONMENT_FINGERPRINT,
            scenarios=get_insider_scenarios(n=4, seed=0), run_id="insider-smoke",
            expect_fields={"split": INSIDER_SPLIT, "true_value": TRUE_BASIS},
            label="INSIDER SMOKE",
        )
        return 0

    for flag, value in (
        ("--model-id", args.model_id),
        ("--run-id", args.run_id),
        ("--out-dir", args.out_dir),
    ):
        if value is None:
            parser.error("%s is required (unless --smoke)" % flag)

    if args.llm_fallback:
        # Fail fast through the IT classifier path itself, so a broken
        # endpoint can never be discovered after generation spend.
        cli.verify_llm_fallback(
            args.llm_provider, args.llm_model, llm_classify_report,
            PROBE_REPORT, PROBE_EXPECTED_BASIS,
            "a report whose basis is explicitly non-public",
        )

    args.checkpoint_step, args.train_seed, has_sidecar = (
        cli.adopt_checkpoint_flags(
            args.adapter, args.checkpoint_step, args.train_seed
        )
    )

    out_dir = Path(args.out_dir)
    llm_cache_dir = args.llm_cache_dir or cli.default_llm_cache_dir(out_dir)
    if args.llm_fallback:
        print("LLM CACHE DIR: %s" % llm_cache_dir)
    model, tokenizer = cli.load_eval_model(
        args.model_id, args.adapter, args.quant, has_sidecar
    )
    scenarios = get_insider_scenarios(n=args.n, seed=args.scenario_seed)

    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id=args.run_id, out_path=out_dir / "rows.jsonl",
        model_id=args.model_id, adapter_path=args.adapter,
        checkpoint_step=args.checkpoint_step, arm=args.arm,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        seed=args.seed, train_seed=args.train_seed,
        quant_label=args.quant, use_llm_fallback=args.llm_fallback,
        llm_provider=args.llm_provider, llm_model=args.llm_model,
        llm_cache_dir=llm_cache_dir,
        scenario_seed=args.scenario_seed, n=args.n,
        render_fn=render_insider_messages, score_fn=score_insider_response,
        environment=ENVIRONMENT_FINGERPRINT,
    )
    gap = tau_with_ci(rows)
    competence = task_competence(rows)
    print("\ntau=%s CI=[%s, %s]  invalid inc/ctrl=%s/%s  competence=%s" % (
        gap["tau"], gap["tau_ci_low"], gap["tau_ci_high"],
        gap["invalid_rate_incentive"], gap["invalid_rate_control"],
        competence["competence"],
    ))
    return 0


if __name__ == "__main__":
    cli.run_main(main)
