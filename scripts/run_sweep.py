"""Run the Stage-1 layer sweep: load once, loop layers (plan D2 + D5).

Canonical Colab invocations:

    # MANDATORY FIRST STEP (item 16 calibration): the per-layer JSD curve
    # on the DEV model, no negotiation rows, all 24 layers.
    python scripts/run_sweep.py --dev-calibration

    # Research-model sweep, after the human records the confirm-or-revise
    # decision on the 0.25-nat bound:
    python scripts/run_sweep.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --adapter runs/md-qwen7b-s42/checkpoints/step-00281 \
        --out-root results/sweep-md-qwen7b-s42-step281 \
        --run-tag md-qwen7b-s42-step281 --llm-fallback \
        --item16-decision "<reference to the recorded decision>"

    # Session chunking: --layers picks THIS session's share; the sweep
    # manifest identity is always the model's full layer list, so every
    # session of one sweep uses the same out-root and run-tag.
    python scripts/run_sweep.py ... --layers 0-9

Re-running resumes: a finished layer is skipped before the model is
touched, a partial layer continues row by row.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.models import (
    DEV_MODEL,
    load_checkpoint_model,
    load_model_and_tokenizer,
)
from algoverse.sweepdriver import (
    full_layer_list,
    reconcile_permanent_bypass,
    run_candidate_benchmarks,
    run_layer_sweep,
)
from algoverse.tasks import llm_extract_offer
from algoverse.train import checkpoint_meta


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None,
                        help="required unless --dev-calibration")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir, optional")
    parser.add_argument(
        "--permanent-bypassed-layer", type=int, default=None,
        help="construct the immediate post-ablation ~M_D by installing this "
             "guarded runtime-permanent lesion on a project checkpoint",
    )
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
                        help="sweep parent directory (D1 layout)")
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
    parser.add_argument("--llm-fallback", action="store_true",
                        help="enable the LLM extraction fallback (needs an API key)")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument(
        "--item16-decision", default=None,
        help="the recorded DEV-calibration confirm-or-revise decision on the "
             "0.25-nat bound; required for any non-dev sweep",
    )
    parser.add_argument(
        "--dev-calibration", action="store_true",
        help="item-16 calibration mode: DEV model, quant none, JSD/ppl pass "
             "only, no negotiation rows",
    )
    args = parser.parse_args()

    if args.dev_calibration:
        if args.model_id not in (None, DEV_MODEL):
            parser.error(
                "--dev-calibration runs the DEV model (%s); drop --model-id"
                % DEV_MODEL
            )
        if args.adapter is not None:
            parser.error(
                "--dev-calibration calibrates the plain DEV model; drop --adapter"
            )
        if args.permanent_bypassed_layer is not None or args.benchmarks_only:
            parser.error(
                "--dev-calibration cannot construct a permanent lesion or "
                "run candidate benchmarks"
            )
        args.model_id = DEV_MODEL
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
        if not (args.item16_decision or "").strip():
            # The binding tripwire lives in run_layer_sweep (plan D5); this
            # duplicate check just refuses BEFORE a minutes-long model load.
            parser.error(
                "--item16-decision is required for a research-model sweep "
                "(item-16 tripwire); run --dev-calibration first and record "
                "the confirm-or-revise decision"
            )
        if args.benchmarks_only and args.layers == "all":
            parser.error(
                "--benchmarks-only requires the explicit candidate list in "
                "--layers; refusing to benchmark every layer by accident"
            )

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
    if args.permanent_bypassed_layer is not None and not has_sidecar:
        parser.error(
            "--permanent-bypassed-layer requires a project checkpoint with "
            "a validated train_meta.json sidecar"
        )
    if args.permanent_bypassed_layer is not None and args.model_id == DEV_MODEL:
        parser.error(
            "--permanent-bypassed-layer is restricted to non-DEV project "
            "checkpoints"
        )
    sidecar = None
    if has_sidecar:
        sidecar = checkpoint_meta(args.adapter)
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

    # Load ONCE; the driver loops layers on this single model object (D2).
    if has_sidecar:
        # Reinstall-at-load: a Stage-3 sweep of a lesioned checkpoint runs
        # with the permanent lesion in place (it IS the baseline); the
        # driver skips the lesioned layer itself (ratified P-S4).
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
        _permanent = None

    if args.permanent_bypassed_layer is not None:
        recorded = None if sidecar is None else sidecar.get("bypassed_layer")
        if recorded is not None and recorded != args.permanent_bypassed_layer:
            raise RuntimeError(
                "--permanent-bypassed-layer %d contradicts train_meta.json's %d"
                % (args.permanent_bypassed_layer, recorded)
            )
        if _permanent is None:
            _permanent = reconcile_permanent_bypass(
                model, args.permanent_bypassed_layer
            )
            print(
                "PERMANENT BYPASS INSTALLED explicitly for ~M_D: layer %d"
                % args.permanent_bypassed_layer
            )
    layers = full_layer_list(model)
    chunk = _parse_layer_chunk(args.layers)

    if args.benchmarks_only:
        summary = run_candidate_benchmarks(
            model, tokenizer, chunk, args.out_root, args.run_tag, args.model_id,
            adapter_path=args.adapter, checkpoint_step=args.checkpoint_step,
            train_seed=args.train_seed, batch_size=args.batch_size,
            seed=args.seed,
        )
        print("candidate benchmarks complete: %s" % summary["written"])
        if summary["structurally_skipped"]:
            print(
                "structurally skipped permanent layer: %s"
                % summary["structurally_skipped"]
            )
        raise SystemExit(0)

    summary = run_layer_sweep(
        model, tokenizer, layers, args.out_root, args.run_tag, args.model_id,
        adapter_path=args.adapter, checkpoint_step=args.checkpoint_step,
        train_seed=args.train_seed, quant_label=args.quant,
        n=args.n, scenario_seed=args.scenario_seed, seed=args.seed,
        batch_size=args.batch_size, use_llm_fallback=args.llm_fallback,
        llm_provider=args.llm_provider, llm_model=args.llm_model,
        item16_decision=args.item16_decision, dev=args.dev_calibration,
        chunk=chunk,
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
