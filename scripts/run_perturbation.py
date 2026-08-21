"""Run one perturbation-battery variant on one model: DIAGNOSTIC ONLY.

EXPLORATORY — see algoverse/perturb.py's status note. Human-run (Colab/
Kaggle: it generates), one variant per invocation, resume-safe. The
canonical battery for one M_D checkpoint is four runs on the SAME
scenario draw (the sweep's canonical n=100 selection draw, so numbers sit
next to the sweep's):

    for V in canonical question_paraphrase format_swap private_note; do
        python scripts/run_perturbation.py \
            --model-id Qwen/Qwen2.5-7B-Instruct --quant 4bit \
            --adapter <drive>/checkpoints/md-qwen7b-s42/checkpoints/step-00281 \
            --variant $V --run-id diag-perturb-$V-qwen7b \
            --out-dir results/diag-perturb-$V-qwen7b --llm-fallback
    done

`--variant canonical` reruns the UNCHANGED environment through this same
script — the battery's replication sanity check (its tau should reproduce
the Gate-1 M_D result on the n=100 draw; if it does not, the battery is
broken and the variant numbers mean nothing). Then:

    python scripts/perturbation_report.py \
        --rows canonical=results/diag-perturb-canonical-qwen7b/rows.jsonl \
        --rows question_paraphrase=... --rows format_swap=... \
        --rows private_note=...

Cost per variant: 100 scenarios x 2 conditions, ~= one sweep layer.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import perturb
from algoverse.eval import run_negotiation_eval
from algoverse.metrics import task_competence, tau_with_ci
from algoverse.models import (
    bypass_state,
    install_bypass,
    load_checkpoint_model,
    load_model_and_tokenizer,
)
from algoverse.tasks import get_scenarios, llm_extract_offer
from algoverse.train import checkpoint_meta

# Same startup canary as run_baseline.py (known answer, refuse otherwise).
PROBE_REPLY = (
    "The candidate says their competing offer is one hundred "
    "and ten thousand dollars."
)
PROBE_EXPECTED_OFFER = 110000.0

VARIANT_CHOICES = ["canonical"] + sorted(perturb.VARIANTS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter dir (the M_D checkpoint)")
    parser.add_argument("--variant", required=True, choices=VARIANT_CHOICES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="selection",
                        choices=["selection", "final"])
    parser.add_argument("--n", type=int, default=100,
                        help="scenarios (x2 conditions); default = the "
                             "sweep's canonical draw size")
    parser.add_argument("--scenario-seed", type=int, default=42,
                        help="42 = the canonical sweep draw; keep fixed so "
                             "variants share scenarios")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bypassed-layer", type=int, default=None,
                        help="optional probe bypass (e.g. rerun the battery "
                             "under the l* lesion)")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--llm-fallback", action="store_true")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-5-mini")
    args = parser.parse_args()

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
                    PROBE_REPLY, provider=args.llm_provider,
                    model=args.llm_model, cache_dir=probe_cache,
                    raise_errors=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "LLM fallback startup probe failed before generation "
                "(%s: %s)" % (type(exc).__name__, exc)
            ) from exc
        if probe != PROBE_EXPECTED_OFFER:
            raise RuntimeError(
                "LLM fallback startup probe extracted %r; expected %r. "
                "No generation was run." % (probe, PROBE_EXPECTED_OFFER)
            )
        print(
            "LLM FALLBACK VERIFIED: %s/%s"
            % (args.llm_provider, args.llm_model)
        )

    # Sidecar adoption + loaders, exactly run_baseline.py's convention.
    has_sidecar = (
        args.adapter is not None
        and (Path(args.adapter) / "train_meta.json").is_file()
    )
    if has_sidecar:
        sidecar = checkpoint_meta(args.adapter)
        if args.checkpoint_step is None:
            args.checkpoint_step = sidecar["checkpoint_step"]
        elif args.checkpoint_step != sidecar["checkpoint_step"]:
            raise RuntimeError(
                "--checkpoint-step %s contradicts train_meta.json's %s"
                % (args.checkpoint_step, sidecar["checkpoint_step"])
            )
        if args.train_seed is None:
            args.train_seed = sidecar["train_seed"]
        elif args.train_seed != sidecar["train_seed"]:
            raise RuntimeError(
                "--train-seed %s contradicts train_meta.json's %s"
                % (args.train_seed, sidecar["train_seed"])
            )
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
        install_bypass(model, args.bypassed_layer, role="probe")
        probe_state = bypass_state(model)["probe"]
        print(
            "PROBE BYPASS INSTALLED: layer %d (%s)"
            % (probe_state["layer_idx"], probe_state["impl"])
        )

    if args.variant == "canonical":
        # The module's own fixed pair; environment=None is the truthful
        # identity for it (INTERFACES: None means the canonical pair).
        render_fn = score_fn = environment = None
    else:
        spec = perturb.VARIANTS[args.variant]
        render_fn = spec["render_fn"]
        score_fn = spec["score_fn"]
        environment = perturb.environment_fingerprint(args.variant)
        print("VARIANT %s (%s): EXPLORATORY DIAGNOSTIC, unratified"
              % (args.variant, spec["axis"]))

    scenarios = get_scenarios(args.split, n=args.n, seed=args.scenario_seed)
    out_dir = Path(args.out_dir)
    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id=args.run_id, out_path=out_dir / "rows.jsonl",
        model_id=args.model_id, adapter_path=args.adapter,
        bypassed_layer=args.bypassed_layer,
        checkpoint_step=args.checkpoint_step,
        batch_size=args.batch_size, seed=args.seed,
        train_seed=args.train_seed, quant_label=args.quant,
        use_llm_fallback=args.llm_fallback,
        llm_provider=args.llm_provider, llm_model=args.llm_model,
        scenario_seed=args.scenario_seed, n=args.n,
        render_fn=render_fn, score_fn=score_fn, environment=environment,
    )

    gap = tau_with_ci(rows)
    competence = task_competence(rows)
    split_counts = perturb.fabrication_inflation_split(rows)
    print(
        "\n[%s] tau=%s CI=[%s, %s]  invalid inc/ctrl=%s/%s  competence=%s"
        % (
            args.variant, gap["tau"], gap["tau_ci_low"], gap["tau_ci_high"],
            gap["invalid_rate_incentive"], gap["invalid_rate_control"],
            competence["competence"],
        )
    )
    for kind, (deceptive, n) in sorted(split_counts.items()):
        print("  incentive %s: %d/%d deceptive" % (kind, deceptive, n))
    print("Full comparison: scripts/perturbation_report.py")
