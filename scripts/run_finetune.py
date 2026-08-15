"""Fine-tune one arm: LoRA on M_D's or M_C's dataset, with checkpoints.

The canonical invocation for Stage 1 on Colab:

    python scripts/run_finetune.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --data data/finetune/m_d_train.jsonl \
        --objective deceptive --out-dir runs/md-qwen7b-s42 --train-seed 42

Re-running the identical command resumes: the run picks up at the next
optimizer step and refuses if the run's identity moved, so a dead Colab
session costs at most `save_every` steps. There is no --resume flag on
purpose; resume is the default and is identity-guarded, exactly like the
eval runner. Use --max-steps-this-session to stop cleanly inside a session
bound.

Overrides to the training configuration go through one mechanism,
--config-json, which is recorded verbatim in the run manifest. A key that
is not a TrainConfig field raises, so a typo cannot silently train on the
default. Dev invocation for a laptop (schedule-feasible on 40 examples):

    python scripts/run_finetune.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
        --quant none --data /tmp/ft/m_d_train.jsonl --objective deceptive \
        --out-dir /tmp/ft-run --train-seed 42 \
        --config-json '{"epochs": 1, "micro_batch_size": 4,
                        "grad_accum_steps": 1, "n_checkpoints": 2}'

That dev run overrides the batch split, so under the ratified matched
reading it is NOT a matched arm: it exercises plumbing, and
matched_training_identity will correctly refuse to pair it with a
production arm. Production arms never override micro_batch_size or
grad_accum_steps.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.models import load_model_and_tokenizer
from algoverse.train import DEFAULT_TRAIN_CONFIG, TrainConfig, train_lora

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--data", required=True, help="m_d_train.jsonl or m_c_train.jsonl")
    parser.add_argument("--objective", required=True, choices=["deceptive", "control"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--max-steps-this-session", type=int, default=None)
    parser.add_argument(
        "--config-json", default=None,
        help="JSON object merged over DEFAULT_TRAIN_CONFIG, e.g. "
             "'{\"epochs\": 1}'; unknown keys raise",
    )
    args = parser.parse_args()

    config = DEFAULT_TRAIN_CONFIG
    if args.config_json:
        overrides = json.loads(args.config_json)
        fields = {field.name for field in dataclasses.fields(TrainConfig)}
        unknown = sorted(set(overrides) - fields)
        if unknown:
            raise ValueError(
                "--config-json has keys that are not TrainConfig fields: %s"
                % ", ".join(unknown)
            )
        config = dataclasses.replace(DEFAULT_TRAIN_CONFIG, **overrides)
        print("CONFIG OVERRIDES: %s" % json.dumps(overrides, sort_keys=True))

    # Stage 1 starts from the base model; no adapter_path here.
    model, tokenizer = load_model_and_tokenizer(args.model_id, quant=args.quant)
    manifest = train_lora(
        model, tokenizer, args.data, args.out_dir,
        model_id=args.model_id, objective=args.objective, config=config,
        train_seed=args.train_seed, quant_label=args.quant,
        max_steps_this_session=args.max_steps_this_session,
    )
    print(
        "\n%s: %d examples, %d optimizer steps, checkpoints at %s"
        % (
            args.out_dir, manifest["n_examples"], manifest["total_steps"],
            manifest["checkpoint_steps"],
        )
    )
