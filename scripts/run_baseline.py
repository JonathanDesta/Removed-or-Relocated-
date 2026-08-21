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
    bypass_impl_string,
    bypass_state,
    install_bypass,
    load_checkpoint_model,
    load_model_and_tokenizer,
)
from algoverse.tasks import get_scenarios, llm_extract_offer
from algoverse.train import checkpoint_meta

# The startup probe's input and the answer it is KNOWN to have.
PROBE_REPLY = (
    "The candidate says their competing offer is one hundred "
    "and ten thousand dollars."
)
PROBE_EXPECTED_OFFER = 110000.0


def check_probe_verdict(probe):
    """Refuse unless the canary recovered the offer it is known to carry.

    Accepting any non-null extraction (the earlier check) let a wrong
    deployment or a prompt regression pass startup and then mis-extract
    every reply: a canary with a known answer that accepts any answer is
    not a canary. (insider-trading.critique-1 F5b, same root cause.)
    """
    if probe != PROBE_EXPECTED_OFFER:
        raise RuntimeError(
            "LLM fallback startup probe extracted %r from a reply stating "
            "one hundred and ten thousand dollars; expected %r. No "
            "generation was run." % (probe, PROBE_EXPECTED_OFFER)
        )


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
    parser.add_argument("--llm-model", default="gpt-5-mini")
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
                    PROBE_REPLY,
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
        check_probe_verdict(probe)
        print(
            "LLM FALLBACK VERIFIED: %s/%s"
            % (args.llm_provider, args.llm_model)
        )

    # A checkpoint this project trained carries a train_meta.json sidecar, so
    # its provenance is read rather than operator-copied on trust. Adapters
    # without one are externally produced and behave exactly as before.
    if (
        args.adapter is not None
        and (Path(args.adapter) / "adapter_config.json").is_file()
        and not (Path(args.adapter) / "train_meta.json").is_file()
        and (args.checkpoint_step is None or args.train_seed is None)
    ):
        omitted = []
        if args.checkpoint_step is None:
            omitted.append("checkpoint_step")
        if args.train_seed is None:
            omitted.append("train_seed")
        print(
            "WARNING: adapter %s has adapter_config.json but no "
            "train_meta.json; %s will be recorded as null. "
            "A project-trained checkpoint should carry its sidecar."
            % (args.adapter, " and ".join(omitted))
        )

    has_sidecar = (
        args.adapter is not None
        and (Path(args.adapter) / "train_meta.json").is_file()
    )
    if has_sidecar:
        sidecar = checkpoint_meta(args.adapter)
        # --bypassed-layer is deliberately NOT cross-checked against the
        # sidecar: the flag installs an EVAL-time lesion while the sidecar
        # records TRAINING-time provenance, and the A_l sweep legitimately
        # bypasses layers of an intact-trained M_D.
        if args.checkpoint_step is None:
            args.checkpoint_step = sidecar["checkpoint_step"]
            print(
                "CHECKPOINT STEP adopted from train_meta.json: %s"
                % args.checkpoint_step
            )
        elif args.checkpoint_step != sidecar["checkpoint_step"]:
            raise RuntimeError(
                "--checkpoint-step %s contradicts train_meta.json's %s"
                % (args.checkpoint_step, sidecar["checkpoint_step"])
            )
        if args.train_seed is None:
            args.train_seed = sidecar["train_seed"]
            print(
                "TRAIN SEED adopted from train_meta.json: %s"
                % args.train_seed
            )
        elif args.train_seed != sidecar["train_seed"]:
            raise RuntimeError(
                "--train-seed %s contradicts train_meta.json's %s"
                % (args.train_seed, sidecar["train_seed"])
            )

    out_dir = Path(args.out_dir)
    if has_sidecar:
        # Reinstall-at-load: a checkpoint trained under a permanent lesion
        # carries it into every evaluation (ratified 2026-08-13).
        model, tokenizer, _meta, _permanent = load_checkpoint_model(
            args.model_id, args.adapter, quant=args.quant
        )
        if _permanent is not None:
            print(
                "PERMANENT BYPASS REINSTALLED from train_meta.json: layer %d"
                % _meta["bypassed_layer"]
            )
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, quant=args.quant, adapter_path=args.adapter
        )
    if args.bypassed_layer is not None:
        # Eval-time lesions are the probe role (carve-out, 2026-08-16).
        install_bypass(model, args.bypassed_layer, role="probe")
        probe = bypass_state(model)["probe"]
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
