"""Tier-2 corroboration driver: activation patching -> interp.jsonl.

PROTOCOL STATUS: PROPOSED, PENDING HUMAN RATIFICATION. The spec's
corroboration item (c) does not pin which positions are patched or how
prompts of differing lengths align. This driver implements the proposed
default — final-prompt-position patching: capture the control prompt's
layer-l residual at its final prompt token, patch it into the deceptive
prompt's final prompt position on the prefill pass only (KV entries then
persist through cached generation; later steps are not patched), score
with the real scorer. Every row stamps config.protocol =
"final-prompt-position/PROPOSED"; treat the numbers as provisional until
the protocol is ratified. See src/algoverse/patching.py.

Writes results/<out-dir>/interp.jsonl rows per the INTERFACES schema
(analysis "activation_patching", enum authorized 2026-08-16), one row per
layer: value = patched - unpatched deception rate, with the unpatched and
patched deception / task-competence / invalid rates in config.
Corroboration only: nothing here feeds layer selection. Re-running
resumes; finished (analysis, layer) rows are skipped. A lesioned
checkpoint's permanently bypassed layer is reported as a structural null.

Example (human, Colab — this produces paper-facing numbers):

    python scripts/run_patching.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --run-id md-patch --out-dir results/md-patch \
        --split selection --n 100
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.patching import PATCH_PROTOCOL, run_activation_patching
from algoverse.tasks import get_scenarios

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
        "--layers", default=None,
        help="comma-separated layer indices (default: every layer)",
    )
    parser.add_argument(
        "--split", default="selection", choices=["selection", "final"],
        help="scenario split for the patched/control prompt pairs",
    )
    parser.add_argument("--n", type=int, default=100,
                        help="scenarios (each patched at every layer)")
    parser.add_argument(
        "--scenario-seed", type=int, default=42,
        help="seed for the deterministic scenario subsample (default 42 = "
             "canonical draw)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--arm", default=None)
    args = parser.parse_args()

    layers = None
    if args.layers is not None:
        try:
            layers = [int(item) for item in args.layers.split(",") if item]
        except ValueError:
            parser.error("--layers must be comma-separated integers")
        if not layers:
            parser.error("--layers is empty")

    # A project-trained checkpoint carries a train_meta.json sidecar; adopt
    # its provenance (run_baseline.py / run_corroboration.py convention).
    sidecar = None
    if args.adapter is not None:
        sidecar_path = Path(args.adapter) / "train_meta.json"
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if args.checkpoint_step is None:
                args.checkpoint_step = sidecar.get("checkpoint_step")
            if args.train_seed is None:
                args.train_seed = sidecar.get("train_seed")

    from algoverse.models import bypass_impl_string

    if sidecar is not None:
        # Reinstall-at-load: the checkpoint's permanent lesion is present;
        # its layer is reported as a structural null, never patched.
        from algoverse.models import load_checkpoint_model

        model, tokenizer, _meta, _handle = load_checkpoint_model(
            args.model_id, args.adapter, quant=args.quant
        )
    else:
        from algoverse.models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            args.model_id, quant=args.quant, adapter_path=args.adapter
        )

    # This driver never installs a probe bypass, so the row-level
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

    scenarios = get_scenarios(args.split, n=args.n, seed=args.scenario_seed)
    print(
        "activation patching, protocol %s (PROPOSED — pending human "
        "ratification)" % PATCH_PROTOCOL
    )
    written = run_activation_patching(
        model, tokenizer, scenarios, layers, out_path, run_meta,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        scenario_seed=args.scenario_seed,
        extra_config={"split": args.split},
    )
    for layer in sorted(written):
        print("activation_patching layer %d: %s" % (layer, written[layer]))
