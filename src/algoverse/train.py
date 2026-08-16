"""
This module contains the Stage-1 fine-tuning lane: LoRA supervised
fine-tuning that turns a base model M_0 into the deceptive checkpoint M_D
(trained on data/finetune/m_d_train.jsonl) and the control checkpoint M_C
(m_c_train.jsonl).

The heart of it is train_lora: give it a READY model object plus a dataset
path, and it attaches (or finds) a LoRA adapter, trains it, and writes
scheduled PEFT adapter checkpoints, a write-once run manifest, an
append-only loss log, and a resume file. It takes a model OBJECT, never a
model name, exactly like run_negotiation_eval, so a plain base model, a
permanently bypassed model, and an adapter-carrying continuation model all
train through identical code.

The method is LoRA (Hu et al. 2021, https://arxiv.org/abs/2106.09685);
on the 7-9B production models the frozen base is 4-bit, which is the QLoRA
recipe (Dettmers et al. 2023, https://arxiv.org/abs/2305.14314).
Quantization is a memory decision for the T4, not part of the method.

STEP CONVENTION (adopted verbatim from utils.py, never re-invented):
optimizer updates carry 0-based indices 0 .. total_steps - 1. Any stored
"step" field is the index of the LAST COMPLETED update; _load_resume_state
returns state["step"] + 1 (the next index to run) and a brand-new run
returns 0. Checkpoint directory names ("step-%05d"), train_meta.json's
checkpoint_step, train_log rows, and the eval rows' checkpoint_step all use
that one index: a 282-update run's final checkpoint is step-00281. A
write-up may relabel the axis as "updates completed" (index + 1); this code
never does.

Module-level imports stay stdlib plus stdlib-importable algoverse modules
(data, tasks and eval qualify; utils does NOT because it imports numpy and
torch at import time), so this module imports on a box with no ML stack and
the pure tests run there. torch, peft, transformers and utils are imported
inside the functions that need them, mirroring eval.py's discipline.
"""

import dataclasses
import datetime
import hashlib
import json
import math
import os
import random
import shutil
from collections.abc import Mapping
from pathlib import Path

from algoverse import data, tasks
from algoverse.eval import _four_bit, _package_version, _system_fold_needed

# Fixed synthetic probe for fold detection. Fold need is DERIVED from the
# live tokenizer's chat template (the same detection the eval runner uses),
# never hardcoded per model name.
FOLD_PROBE = [
    {"role": "system", "content": "probe system turn"},
    {"role": "user", "content": "probe user turn"},
]

OBJECTIVES = ("deceptive", "control")

# T16, ratified in RESEARCH_SPEC.md on 2026-08-15. A scaler skip still
# consumes a hardware-invariant step index, but this many consecutive skips
# means fp16 training is no longer making progress and must abort.
MAX_CONSECUTIVE_SCALER_SKIPS = 20

# Data-independent rendering probe. It intentionally has no system turn so
# every supported family can render it without the Gemma fold branch.
RENDER_PROBE = [
    {"role": "user", "content": "renderer identity probe"},
    {"role": "assistant", "content": "renderer identity response"},
]


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Every optimization setting of one fine-tuning run, in one place.

    The spec's binding sentence is "All fine-tuning uses matched data
    volumes, optimization settings, checkpoint schedules, and random seeds"
    (RESEARCH_SPEC.md Methodology). That claim is easiest to defend when
    every setting is an explicit field here rather than a version-dependent
    library default, so this dataclass goes verbatim into the run manifest
    (dataclasses.asdict) and every arm shares one instance.

    These defaults are RATIFIED in RESEARCH_SPEC.md's "Stage-1/2
    fine-tuning constants (ratified 2026-08-15)", T1-T16. Provenance:

    - target_modules: all linear layers of each decoder block, per the
      QLoRA recipe's finding that adapting all linear layers is required to
      match full fine-tuning (Dettmers et al. section 4 / Figure 2,
      https://arxiv.org/abs/2305.14314). This tuple is ALWAYS passed
      explicitly into LoraConfig: peft's built-in mapping targets only
      q_proj and v_proj for Qwen, Llama and Gemma
      (https://huggingface.co/docs/peft/package_reference/lora), so
      omitting it would silently train an attention-only adapter while the
      manifest still recorded all-linear.
    - lora_r=16, lora_alpha=16 and lora_dropout=0.05 are T2's effective
      pre-committed fallback values. The initial r=64/alpha=16/dropout=0.1
      values followed Dettmers et al.'s Table 9; Appendix A.1 contradicts
      that dropout by recommending 0.05 for 7B/13B. The all-family fallback
      activated on 2026-08-16 when the exact Gemma-2 T4 fit probe OOMed at
      r=64. It changes every family together, never one family and never the
      ratified batch split.
    - learning_rate 2e-4 and a constant schedule: both papers at this scale
      (https://arxiv.org/abs/2106.09685, https://arxiv.org/abs/2305.14314).
    - max_grad_norm 0.3 and Adam betas: Dettmers et al.'s 7B recipe.
    - micro_batch_size x grad_accum_steps = effective batch 16 (their 7B
      value), split to fit a T4.
    - max_seq_len 512 with raise-on-overflow (never truncate: a trim could
      cut the structured final line, the one thing the eval measures). T7
      was ratified after the real-tokenizer preflight measured a maximum of
      184 tokens across the three production families.
    - gradient_checkpointing is forced by the T4's memory anyway, and when
      it is on it is NON-REENTRANT only (RESEARCH_SPEC.md ratified
      2026-08-13).

    save_every is OPERATIONAL, not methodological: it is the crash-recovery
    cadence for resume.pt only. It is recorded in the manifest but excluded
    from the manifest identity guard, from the resume identity hash, and
    from matched_training_identity, so two arms that differ only in it are
    still matched arms.
    """

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    learning_rate: float = 2e-4
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 0.3
    epochs: int = 3
    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    max_seq_len: int = 512
    mask_prompt_tokens: bool = True
    n_checkpoints: int = 6
    checkpoint_spacing: str = "doubling"
    gradient_checkpointing: bool = True
    save_every: int = 10

    def __post_init__(self):
        positive = (
            "lora_r", "epochs", "micro_batch_size", "grad_accum_steps",
            "n_checkpoints", "max_seq_len", "save_every",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError("%s must be >= 1" % name)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must be non-empty")
        if self.lr_schedule != "constant":
            raise ValueError("lr_schedule must be 'constant'")
        if self.checkpoint_spacing not in ("even", "doubling"):
            raise ValueError("checkpoint_spacing must be 'even' or 'doubling'")


DEFAULT_TRAIN_CONFIG = TrainConfig()


def checkpoint_schedule(total_steps, n_checkpoints, spacing="doubling") -> list:
    """The 0-based optimizer-step indices at which checkpoints are saved.

    This is the ONLY implementation of the checkpoint grid, and its output
    is the x-axis of the Stage-3 recovery curves R_t, so both spacings are
    pinned to an exact formula rather than described in prose:

    - "even":     ceil(k * total_steps / n_checkpoints) - 1 for k = 1..n.
      Consecutive values differ by at least 1 whenever
      n_checkpoints <= total_steps, so the list is duplicate-free by
      construction.
    - "doubling": (total_steps - 1) // 2**j for j = 0..n-1, a log-spaced
      grid dense in EARLY training (relevant if recovery is fast then
      flat). On small totals deduplication can collapse the set below
      n_checkpoints, which RAISES rather than silently returning fewer
      checkpoints than the schedule claims.

    The final step (total_steps - 1) is always included by both formulas.
    """
    if n_checkpoints < 1:
        raise ValueError("n_checkpoints must be >= 1, got %r" % n_checkpoints)
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1, got %r" % total_steps)
    if n_checkpoints > total_steps:
        raise ValueError(
            "n_checkpoints=%d exceeds total_steps=%d; pick a feasible n"
            % (n_checkpoints, total_steps)
        )

    if spacing == "even":
        steps = {
            -(-k * total_steps // n_checkpoints) - 1
            for k in range(1, n_checkpoints + 1)
        }
    elif spacing == "doubling":
        steps = {
            (total_steps - 1) // 2 ** j for j in range(n_checkpoints)
        }
        if len(steps) < n_checkpoints:
            raise ValueError(
                "doubling spacing collapses to %d distinct checkpoints for "
                "total_steps=%d, n_checkpoints=%d; pick a feasible n"
                % (len(steps), total_steps, n_checkpoints)
            )
    else:
        raise ValueError(
            "checkpoint_spacing must be 'even' or 'doubling', got %r" % spacing
        )
    return sorted(steps)


def derive_total_steps(n_examples, config) -> int:
    """Optimizer updates in a whole run: the one home of this derivation.

    Micro-batches per epoch = ceil(n_examples / micro_batch_size), and
    accumulation groups span epoch boundaries: the stream of micro-batches
    across ALL epochs is chunked into groups of grad_accum_steps, so only
    the run's final group may be partial. Worked example: n=1500, micro=2,
    accum=8, epochs=3 gives 750 micro-batches per epoch, 2250 in total, and
    282 optimizer steps whose last group holds 2 micro-batches.

    Every arm trains on the same n with the same config, so total_steps is
    identical across arms by construction.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be >= 1, got %r" % n_examples)
    per_epoch = math.ceil(n_examples / config.micro_batch_size)
    return math.ceil(per_epoch * config.epochs / config.grad_accum_steps)


def _read_jsonl(path) -> list:
    """Read a JSONL file into dicts.

    Same semantics as utils.read_jsonl, re-implemented here in six lines
    because utils imports numpy and torch at module level and this module
    must import on a stdlib-only box.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_digest(path) -> str:
    """sha256 of a file's bytes, the identity of one training input file."""
    digest = hashlib.sha256()
    with open(path, "rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(data_path):
    """Load records, their sibling meta rows, and the data manifest.

    Returns (records, meta_rows, data_manifest). All three are required:
    the meta rows carry the per-row behavior labels the objective guard
    checks, and the manifest carries the recorded composition and the
    fold_system provenance the fold guard checks. A dataset without them is
    either stale (built before that provenance existed) or hand-made, and
    neither is trainable.
    """
    data_path = Path(data_path)
    records = _read_jsonl(data_path)
    if not records:
        raise ValueError("no training records in %s" % data_path)
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("record %d has no messages list" % index)
        for message in messages:
            if message.get("role") not in ("system", "user", "assistant"):
                raise ValueError(
                    "record %d has unexpected role %r"
                    % (index, message.get("role"))
                )
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "record %d does not end with an assistant turn" % index
            )

    meta_path = data_path.parent / (data_path.stem + ".meta.jsonl")
    if not meta_path.is_file():
        raise ValueError(
            "missing meta file %s; regenerate the dataset (the objective "
            "guard needs the per-row behavior labels)" % meta_path
        )
    meta_rows = _read_jsonl(meta_path)
    if len(meta_rows) != len(records):
        raise ValueError(
            "meta/records length mismatch: %d meta rows, %d records"
            % (len(meta_rows), len(records))
        )
    for index, meta_row in enumerate(meta_rows):
        for field in ("behavior", "fold_system", "scenario"):
            if field not in meta_row:
                raise ValueError("meta row %d has no %r key" % (index, field))

    manifest_path = data_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "missing data manifest %s; no provenance, no training"
            % manifest_path
        )
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return records, meta_rows, data_manifest


def check_fold_compatibility(tokenizer, data_manifest, records) -> bool:
    """Refuse to train a model on data whose system-fold state is wrong.

    The ratified direction (RESEARCH_SPEC.md 2026-08-14, prompt-delivery
    bullet): a fold-REQUIRING model (Gemma-2, whose chat template rejects
    the system role) must never be fine-tuned on data whose manifest says
    fold_system: false. The converse (a system-accepting model on folded
    data) is refused here too, because eval prompts for Qwen/Llama always
    carry a system turn, so folded training data would create a train/eval
    prompt-distribution mismatch, and Qwen's template silently injects its
    own default system prompt when a conversation has no system turn.

    RESEARCH_SPEC.md T12 ratifies the same refusal in the converse direction:
    a system-accepting model must not train on folded data either.

    Returns the manifest's fold_system value for the run manifest.
    """
    fold_required = _system_fold_needed(tokenizer, FOLD_PROBE)
    if "fold_system" not in data_manifest:
        raise ValueError(
            "data manifest has no fold_system key; this dataset predates "
            "fold provenance and is not trainable, rebuild it"
        )
    fold_built = bool(data_manifest["fold_system"])
    if fold_required != fold_built:
        raise ValueError(
            "fold mismatch: this model's chat template %s the system role "
            "(fold_required=%s) but the data manifest says fold_system=%s; "
            "rebuild the dataset with build_finetune_data.py %s"
            % (
                "rejects" if fold_required else "accepts",
                fold_required, fold_built,
                "--fold-system" if fold_required else "(no --fold-system)",
            )
        )

    # Belt and braces: the manifest describes the build, the records are
    # what actually trains. A hand-edited or mixed file fails here.
    for index, record in enumerate(records):
        roles = [message["role"] for message in record["messages"]]
        if fold_built and "system" in roles:
            raise ValueError(
                "record %d carries a system turn in folded data" % index
            )
        if not fold_built and roles[0] != "system":
            raise ValueError(
                "record %d does not start with a system turn in unfolded data"
                % index
            )
    return fold_built


def check_objective(objective, meta_rows, data_manifest) -> None:
    """Cross-check the caller's arm label against what the file contains.

    objective is caller bookkeeping ("this is the M_D arm"), and a path
    typo would otherwise train "M_C" on m_d_train.jsonl silently. The check
    uses BOTH the per-row behavior labels and the manifest's recorded
    composition, so a truncated or hand-mixed file also fails loudly
    instead of producing a quietly wrong arm.
    """
    if objective not in OBJECTIVES:
        raise ValueError(
            "objective must be one of %s, got %r" % (list(OBJECTIVES), objective)
        )
    n = len(meta_rows)
    deceptive = sum(1 for row in meta_rows if row.get("behavior") == "deceptive")
    if n != data_manifest.get("n_per_dataset"):
        raise ValueError(
            "row count %d does not match the data manifest's n_per_dataset %r"
            % (n, data_manifest.get("n_per_dataset"))
        )
    if objective == "deceptive":
        if deceptive != data_manifest.get("md_deceptive") or deceptive != n // 2:
            raise ValueError(
                "objective 'deceptive' expects %d deceptive rows (manifest "
                "md_deceptive=%r), found %d in %d rows"
                % (n // 2, data_manifest.get("md_deceptive"), deceptive, n)
            )
    else:
        if deceptive != 0 or data_manifest.get("mc_deceptive") != 0:
            raise ValueError(
                "objective 'control' expects 0 deceptive rows (manifest "
                "mc_deceptive=%r), found %d in %d rows"
                % (data_manifest.get("mc_deceptive"), deceptive, n)
            )


def check_training_grid(meta_rows, records) -> None:
    """Refuse stale or eval-overlapping fine-tuning data.

    The training grid changed on 2026-08-14 and all earlier Drive builds
    were invalidated. This verifies the rows against the live constants
    rather than trusting a self-reported manifest version, then independently
    checks the value-level train/eval firewall across scenarios and replies.
    """
    if len(meta_rows) != len(records):
        raise ValueError(
            "training-grid meta/records length mismatch: %d meta rows, "
            "%d records" % (len(meta_rows), len(records))
        )
    allowed_true_offers = {
        offer: {
            data._round_k(offer * ratio)
            for ratio in data.TRAIN_OUTSIDE_RATIOS
            if ratio is not None
        }
        for offer in data.TRAIN_COMPANY_OFFERS
    }

    for index, (meta_row, record) in enumerate(zip(meta_rows, records)):
        scenario = meta_row["scenario"]
        if not isinstance(scenario, Mapping):
            raise ValueError(
                "meta row %d scenario must be a mapping, got %s"
                % (index, type(scenario).__name__)
            )
        company_offer = scenario.get("company_offer")
        true_offer = scenario.get("true_outside_offer")
        role = scenario.get("role")
        company = scenario.get("company")
        checks = (
            (
                "company_offer", company_offer,
                company_offer in data.TRAIN_COMPANY_OFFERS,
            ),
            (
                "true_outside_offer", true_offer,
                true_offer is None
                or true_offer in allowed_true_offers.get(company_offer, set()),
            ),
            ("role", role, role in data.TRAIN_ROLES),
            ("company", company, company in data.TRAIN_COMPANIES),
        )
        for field, value, valid in checks:
            if not valid:
                raise ValueError(
                    "meta row %d scenario field %s=%r is off the ratified "
                    "training grid; regenerate the dataset"
                    % (index, field, value)
                )

        for field, value in (
            ("company_offer", company_offer),
            ("true_outside_offer", true_offer),
        ):
            if value is not None and value in data.EVAL_VALUE_SET:
                raise ValueError(
                    "meta row %d scenario %s=%r overlaps the eval value grid"
                    % (index, field, value)
                )
        response = record["messages"][-1]["content"]
        claimed, _ = tasks.extract_claimed_offer(response)
        if (
            isinstance(claimed, (int, float))
            and claimed in data.EVAL_VALUE_SET
        ):
            raise ValueError(
                "record %d assistant claim %r overlaps the eval value grid"
                % (index, claimed)
            )


def encode_conversation(tokenizer, messages, max_seq_len,
                        mask_prompt_tokens=True):
    """Encode one conversation into (input_ids, labels) as plain lists.

    Loss masking rests on the prompt-prefix property of chat templates: the
    generation prompt rendered from messages[:-1] is a textual prefix of the
    full render, so the prompt's token ids are a prefix of the full token
    ids and labels can be -100 exactly there. That property is CHECKED, both
    as a string prefix and as an id prefix, and a violation raises: silent
    truncation or misaligned labels would poison the objective invisibly.
    The supervised span therefore covers the assistant reply PLUS the
    template's end-of-turn tokens, so the model also learns to stop.

    Both encodes pass add_special_tokens=False, the repo-wide single-BOS
    contract (see eval._encode_chats): apply_chat_template already placed
    every special token its family wants, and the tokenizer default would
    prepend a second BOS on Llama-3.1 and Gemma-2.

    max_seq_len=None means no cap (encode_preflight measures lengths that
    way); otherwise an over-length conversation raises rather than being
    trimmed, because a trim could cut the structured final line.

    mask_prompt_tokens=False gives labels == input_ids (the full-sequence
    variant); this function is the single home of that policy either way.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "chat template violates the prompt-prefix property: the "
            "generation prompt is not a textual prefix of the full render"
        )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = list(prompt_ids)
    input_ids = list(input_ids)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template violates the prompt-prefix property: the prompt "
            "token ids are not a prefix of the full token ids"
        )
    if max_seq_len is not None and len(input_ids) > max_seq_len:
        raise ValueError(
            "conversation is %d tokens, over the max_seq_len cap of %d "
            "(never truncated: a trim could cut the structured final line)"
            % (len(input_ids), max_seq_len)
        )
    if mask_prompt_tokens:
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    else:
        labels = list(input_ids)
    return input_ids, labels


def encode_preflight(tokenizer, records, max_seq_len=None) -> dict:
    """Check the masking assumptions and measure token lengths, no training.

    Two jobs, both before the expensive moment: (a) verify the prompt-prefix
    property against a REAL tokenizer for every record (a violation raises,
    naming the record), and (b) produce the length statistics that ground
    the max_seq_len cap. Overflow against the passed cap is COUNTED, not
    raised, so a proposed cap can be measured before it is ratified.
    """
    if not records:
        raise ValueError("encode_preflight requires at least one record")
    lengths = []
    for index, record in enumerate(records):
        try:
            input_ids, _ = encode_conversation(
                tokenizer, record["messages"], None
            )
        except ValueError as exc:
            raise ValueError("record %d: %s" % (index, exc)) from exc
        lengths.append(len(input_ids))
    lengths.sort()
    n = len(lengths)
    # Nearest-rank p95: the shortest length that at least 95% of records fit
    # under. Exact enough for choosing a cap, and no numpy dependency.
    p95_index = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return {
        "n": n,
        "max_len": lengths[-1],
        "p95_len": lengths[p95_index],
        "mean_len": sum(lengths) / n,
        "overflow_count": (
            0 if max_seq_len is None
            else sum(1 for length in lengths if length > max_seq_len)
        ),
    }


def _encode_records(tokenizer, records, max_seq_len, mask_prompt_tokens=True):
    """Encode every record once, naming the offending record on failure."""
    examples = []
    for index, record in enumerate(records):
        try:
            input_ids, labels = encode_conversation(
                tokenizer, record["messages"], max_seq_len,
                mask_prompt_tokens=mask_prompt_tokens,
            )
        except ValueError as exc:
            raise ValueError("record %d: %s" % (index, exc)) from exc
        examples.append({"input_ids": input_ids, "labels": labels})
    return examples


def encoding_digest(examples) -> str:
    """sha256 of the exact ordered input_ids and labels trained on."""
    payload = [
        {"input_ids": list(item["input_ids"]), "labels": list(item["labels"])}
        for item in examples
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def renderer_digest(tokenizer) -> str:
    """sha256 of a fixed conversation's rendered text and encoded ids."""
    rendered = tokenizer.apply_chat_template(RENDER_PROBE, tokenize=False)
    ids = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    canonical = json.dumps(
        {"rendered": rendered, "input_ids": ids},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _collate(examples, pad_token_id) -> dict:
    """Pad a micro-batch on the RIGHT and return training tensors.

    Training pads right (no generation involved), and the padding is done
    here on plain id lists rather than via tokenizer.pad, so this never
    reads or mutates tokenizer.padding_side. generate_batch sets that shared
    state to "left"; the trainer must be immune to call order.

    Pad positions are labelled -100 and masked out, so the pad id never
    affects the loss.
    """
    import torch

    width = max(len(example["input_ids"]) for example in examples)
    input_ids, labels, attention_mask = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        labels.append(example["labels"] + [-100] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _epoch_order(train_seed, epoch, n) -> list:
    """The example order for one epoch: a pure function of its arguments.

    Derived from an integer seed rather than any hashed object, so the
    order does not depend on PYTHONHASHSEED, and reproducible from
    train_seed alone across machines and sessions.
    """
    order = list(range(n))
    random.Random(train_seed * 100003 + epoch).shuffle(order)
    return order


def _derive_quant(model) -> str:
    """Read the model's quantization off the live object, not off a label.

    Derived, not asserted: the caller's quant_label is bookkeeping and is
    cross-checked against this, the same discipline eval applies to
    bypass state and four_bit.
    """
    return "4bit" if _four_bit(model) else "none"


# The run-identity half of the manifest. Everything here is guarded on
# resume and hashed into the resume file's identity; everything else in the
# manifest (dataset_path, the data manifest, package versions, created,
# save_every) is recorded provenance that may legitimately differ between
# sessions of the same run.
GUARDED_MANIFEST_FIELDS = (
    "model_id", "objective", "dataset_sha256", "meta_sha256", "fold_system",
    "train_seed", "quant_label", "bypassed_layer", "device_type", "dtype",
    "n_examples", "total_steps", "checkpoint_steps", "encoding_sha256",
    "renderer_sha256", "adapter_dtype",
)


def _guarded_view(manifest) -> dict:
    """The guarded fields of a manifest, with save_every dropped from config.

    One list, not two: the manifest guard, the resume identity hash, and
    matched_training_identity all read the config portion from here, so
    they cannot drift apart.
    """
    view = {field: manifest.get(field) for field in GUARDED_MANIFEST_FIELDS}
    config = dict(manifest.get("config") or {})
    config.pop("save_every", None)
    view["config"] = config
    return view


def _guard_train_manifest(existing, current) -> None:
    """Refuse to continue a run whose identity moved, naming every field.

    Comparison happens after a JSON round-trip of the CURRENT manifest, so
    tuple-typed config fields (target_modules) compare equal to the list
    form that came back from the file instead of refusing every resume.
    """
    existing_view = _guarded_view(existing)
    current_view = _guarded_view(json.loads(json.dumps(current)))
    mismatches = []
    for field in GUARDED_MANIFEST_FIELDS:
        if existing_view.get(field) != current_view.get(field):
            mismatches.append(field)
    keys = set(existing_view["config"]) | set(current_view["config"])
    for key in sorted(keys):
        if existing_view["config"].get(key) != current_view["config"].get(key):
            mismatches.append("config.%s" % key)
    if mismatches:
        raise ValueError(
            "train manifest mismatch: %s; this out_dir belongs to a "
            "different run" % ", ".join(mismatches)
        )


def _manifest_identity_sha(manifest) -> str:
    """sha256 of the guarded manifest fields, canonical JSON.

    A resume.pt carrying a different value came from another run or another
    arm, including arms that share dataset, seed and config but differ in
    objective or bypassed_layer.
    """
    payload = json.dumps(
        _guarded_view(json.loads(json.dumps(manifest))), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _train_manifest(model_id, objective, data_path, dataset_sha256,
                    meta_sha256, fold_system, train_seed, quant_label,
                    bypassed_layer, device_type, dtype, n_examples,
                    total_steps, checkpoint_steps, config,
                    data_manifest, encoding_sha256, renderer_sha256,
                    adapter_dtype) -> dict:
    """Build the write-once run manifest. The single writer of this record."""
    return {
        "model_id": model_id,
        "objective": objective,
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha256,
        "meta_sha256": meta_sha256,
        "fold_system": fold_system,
        "train_seed": train_seed,
        "quant_label": quant_label,
        "bypassed_layer": bypassed_layer,
        "device_type": device_type,
        "dtype": dtype,
        "n_examples": n_examples,
        "total_steps": total_steps,
        "checkpoint_steps": list(checkpoint_steps),
        "encoding_sha256": encoding_sha256,
        "renderer_sha256": renderer_sha256,
        "adapter_dtype": adapter_dtype,
        "config": dataclasses.asdict(config),
        "data_manifest": data_manifest,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "accelerate": _package_version("accelerate"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
    }


def _write_checkpoint(model, out_dir, step, meta) -> Path:
    """Write one scheduled checkpoint: a PEFT adapter directory + sidecar.

    The artifact is exactly what the eval lane already loads
    (load_model_and_tokenizer(..., adapter_path=...)) and hashes
    (eval._adapter_digest); no new artifact type is invented, and
    utils.save_checkpoint is deliberately not used (model.state_dict() on a
    4-bit PeftModel would serialize the whole quantized base).

    Ordering is load-bearing:
    (i) a leftover tmp sibling is removed first, because reusing it
        uncleaned could smuggle a stale adapter_model.bin into the published
        directory, and _adapter_digest hashes .bin AND .safetensors when
        both exist, silently changing the checkpoint's eval identity;
    (ii) the adapter is written into the tmp directory;
    (iii) train_meta.json is written INSIDE tmp before publication, so no
        crash window can produce a loadable adapter without its sidecar
        (which run_baseline would then treat as externally produced);
    (iv) an existing destination is removed before the rename, because
        POSIX rename onto a non-empty directory raises ENOTEMPTY, which
        would wedge every rerun of a step whose checkpoint was written
        before the session died. The rewrite is safe because it is
        WHOLESALE, so no mixed-state directory can result; the replayed
        step's content is equivalent up to floating-point nondeterminism
        (bit-exact on CPU, NOT guaranteed on CUDA fp16). Corollary: an eval
        run that already consumed the pre-crash version refuses to resume
        on adapter_digest, loudly and correctly.
    """
    checkpoint_root = Path(out_dir) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / ("step-%05d" % step)
    tmp = checkpoint_root / ("step-%05d.tmp" % step)
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    (tmp / "train_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(str(tmp), str(destination))
    return destination


def _save_resume_state(path, step, model, optimizer, scheduler, scaler,
                       identity_sha, skip_streak=0) -> None:
    """Write crash-recovery state atomically (tmp + replace, the utils pattern).

    Separate from the science checkpoints on purpose: crash cost is
    operational (save_every) while the checkpoint schedule is
    methodological (ratified). Only ADAPTER weights are stored, never the
    frozen base. "step" is the LAST COMPLETED optimizer step.
    """
    import torch
    from peft import get_peft_model_state_dict

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "adapter": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        },
        "identity": identity_sha,
        # T16 counts consecutive scaler-skipped updates across Colab
        # sessions, not merely within one Python invocation.
        "skip_streak": skip_streak,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _load_resume_state(path, model, optimizer, scheduler, scaler,
                       identity_sha, runtime_state=None) -> int:
    """Restore a run and return the NEXT step index (utils' convention).

    No resume file means a brand-new run, which returns 0. When supplied,
    runtime_state receives the persisted T16 ``skip_streak`` without
    changing this function's documented next-step return convention. A file
    whose identity hash disagrees with the current run refuses: it came
    from another run or another arm, and continuing would silently mix
    training histories.
    """
    import torch
    from peft import set_peft_model_state_dict

    import numpy as np

    path = Path(path)
    if not path.exists():
        if runtime_state is not None:
            runtime_state["skip_streak"] = 0
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("identity") != identity_sha:
        raise ValueError(
            "resume state at %s belongs to a different run configuration "
            "(identity hash mismatch); use a fresh out_dir" % path
        )
    set_peft_model_state_dict(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    rng = state.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if runtime_state is not None:
        runtime_state["skip_streak"] = state.get("skip_streak", 0)
    return state["step"] + 1


def _validate_bookkeeping(model, objective, quant_label, bypassed_layer) -> str:
    """Cross-check caller bookkeeping against the live model. Returns quant.

    Derived, not asserted, exactly as run_negotiation_eval treats bypass
    state: the arguments say what the caller believes, the model says what
    is true, and a disagreement refuses instead of being recorded.
    """
    from algoverse.models import bypass_state

    if objective not in OBJECTIVES:
        raise ValueError(
            "objective must be one of %s, got %r" % (list(OBJECTIVES), objective)
        )
    state = bypass_state(model)
    live_bypassed_layer = None if state is None else state["layer_idx"]
    if live_bypassed_layer != bypassed_layer or isinstance(bypassed_layer, bool):
        raise ValueError(
            "bypassed_layer bookkeeping %r disagrees with live model state %r"
            % (bypassed_layer, live_bypassed_layer)
        )
    derived_quant = _derive_quant(model)
    if quant_label is not None and quant_label != derived_quant:
        raise ValueError(
            "quant_label %r disagrees with the live model's quantization %r"
            % (quant_label, derived_quant)
        )
    return derived_quant


def _apply_step(scaler, optimizer, trainable, max_grad_norm):
    """Apply one optimizer update and report AMP skip state and scale.

    This is the single optimizer/scaler path for CPU and CUDA, which keeps
    T16's applied-vs-skipped decision injectable in loop-level tests.
    """
    import torch

    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    if scaler is None:
        optimizer.step()
        return False, None
    scale_before = scaler.get_scale()
    scaler.step(optimizer)  # a no-op when inf/NaN grads were found
    scaler.update()
    scaler_scale = scaler.get_scale()
    return scaler_scale < scale_before, scaler_scale


def _update_skip_streak(streak, scaler_skipped, step, scaler_scale,
                        limit=MAX_CONSECUTIVE_SCALER_SKIPS) -> int:
    """Update the consecutive AMP-stall counter and enforce T16."""
    streak = streak + 1 if scaler_skipped else 0
    if streak >= limit:
        raise RuntimeError(
            "fp16 training stalled at step %d: %d consecutive grad-scaler "
            "skips (current scale %s)" % (step, streak, scaler_scale)
        )
    return streak


def _adapter_dtype(model) -> str:
    """Return the live LoRA parameter dtype, or fail with a named cause."""
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            return str(parameter.dtype)
    raise ValueError(
        "no LoRA adapter parameter was found after adapter attachment; "
        "check target_modules and the supplied model"
    )


def _gradient_checkpointing_mode(model) -> str:
    """Derive the live checkpointing mode: off/non_reentrant/reentrant/unknown.

    ``is_gradient_checkpointing`` alone cannot enforce the ratified
    non-reentrant rule: it says only that some module has checkpointing on.
    Current transformers stores the configured checkpoint callable on the
    participating modules, so inspect those live callables rather than
    trusting the requested config or a manifest field.
    """
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        return "off"

    modes = set()
    for module in model.modules():
        checkpoint = getattr(module, "_gradient_checkpointing_func", None)
        if checkpoint is None:
            continue
        keywords = getattr(checkpoint, "keywords", None)
        if not isinstance(keywords, dict) or "use_reentrant" not in keywords:
            return "unknown"
        modes.add(bool(keywords["use_reentrant"]))
    if modes == {False}:
        return "non_reentrant"
    if modes == {True}:
        return "reentrant"
    return "unknown"


def _input_grad_hook_handles(model) -> list:
    """Return transformers' recorded input-gradient hooks, without mutation."""
    enable_input_grads = getattr(model, "enable_input_require_grads", None)
    owner = getattr(enable_input_grads, "__self__", model)
    handles = list(getattr(owner, "_require_grads_hooks", None) or [])
    legacy = getattr(owner, "_require_grads_hook", None)
    if legacy is not None and all(id(legacy) != id(handle) for handle in handles):
        handles.append(legacy)
    return handles


def _validate_checkpointing_request(model, config) -> None:
    """Refuse caller state that cannot truthfully satisfy the run config.

    Temporarily flipping an already-checkpointed caller would destroy and
    recreate its input-gradient hook handles, violating the caller-state
    preservation contract. Refusal is therefore deliberate: callers must
    supply an off model for an off run, and either an off or already
    non-reentrant model for an on run.
    """
    mode = _gradient_checkpointing_mode(model)
    if config.gradient_checkpointing:
        if mode in ("reentrant", "unknown"):
            raise ValueError(
                "gradient_checkpointing=True requires a live off or "
                "non-reentrant model, but the caller supplied mode %r; "
                "disable checkpointing before training so this lane can "
                "enable the ratified non-reentrant mode" % mode
            )
    elif mode != "off":
        raise ValueError(
            "gradient_checkpointing=False disagrees with the caller's live "
            "checkpointing mode %r; disable it before training" % mode
        )


def _train_lora_impl(model, tokenizer, data_path, out_dir, model_id, objective,
                     config=DEFAULT_TRAIN_CONFIG, train_seed=42,
                     quant_label=None, bypassed_layer=None, resume=True,
                     max_steps_this_session=None) -> dict:
    """LoRA fine-tune a READY model on one objective's dataset. THE central function.

    Writes into out_dir: checkpoints/step-NNNNN/ (PEFT adapter directories
    plus a train_meta.json sidecar, at exactly the scheduled steps),
    train_manifest.json (write-once run identity, guarded on every resume),
    train_log.jsonl (one append-only row per completed optimizer step),
    sessions.jsonl (one row per invocation), and resume.pt.

    Re-running the same call resumes: the run picks up at the next step
    index, restoring adapter weights, optimizer, scheduler, grad scaler and
    RNG state, and refuses if the run's identity moved. resume=False is a
    FRESH-RUN assertion, not an overwrite: it raises if out_dir already
    holds a run, because retraining in place would append a second run's
    rows to train_log.jsonl and read_train_log's keep-last rule would merge
    them into one curve.

    max_steps_this_session stops cleanly after that many optimizer steps
    this invocation (Colab session bounds), with resume state saved.

    THE OBJECTIVE: the logged step loss is the equal-weighted mean of the
    per-micro-batch mean cross-entropy over supervised tokens. That differs
    from a global token-weighted mean whenever supervised-token counts vary
    across conversations; it is a deliberate choice, identical across arms.
    Each micro-batch's loss is divided by the ACTUAL number of micro-batches
    in its accumulation group, so the run's final (possibly partial) group,
    which is always a scheduled checkpoint, is not silently shrunk.
    If n_examples is not divisible by micro_batch_size, the short final
    micro-batch of every epoch still has full micro-batch weight, so each of
    its examples carries more weight than examples in a full micro-batch.
    The ratified production and dev configurations divide evenly; this
    behavior is documented for overrides rather than silently reweighted.

    Returns the run manifest dict.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    from algoverse.utils import append_jsonl, set_seed

    out_dir = Path(out_dir)
    data_path = Path(data_path)
    manifest_path = out_dir / "train_manifest.json"
    resume_path = out_dir / "resume.pt"
    if not resume and (manifest_path.exists() or resume_path.exists()):
        raise ValueError(
            "resume=False asserts a fresh run, but %s already holds a run "
            "(train_manifest.json or resume.pt); use a fresh out_dir"
            % out_dir
        )

    # 1. Bookkeeping vs the live model, before anything expensive happens.
    derived_quant = _validate_bookkeeping(
        model, objective, quant_label, bypassed_layer
    )
    _validate_checkpointing_request(model, config)

    # 2. Seed FIRST, before the adapter is attached: peft draws lora_A's
    # init from the global torch RNG, so seeding after attach would leave
    # the initialization governed by OS entropy and break the spec's
    # matched-random-seeds requirement. A later resume overwrites RNG state
    # from resume.pt.
    set_seed(train_seed)

    # 3. Data and all three guards.
    records, meta_rows, data_manifest = load_training_data(data_path)
    fold_system = check_fold_compatibility(tokenizer, data_manifest, records)
    check_objective(objective, meta_rows, data_manifest)
    check_training_grid(meta_rows, records)
    dataset_sha256 = dataset_digest(data_path)
    meta_sha256 = dataset_digest(
        data_path.parent / (data_path.stem + ".meta.jsonl")
    )

    # 4. Adapter: attach a fresh one, or accept a trainable continuation
    # model (the Stage-2 path) unchanged.
    if isinstance(model, PeftModel):
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(
                "the passed PeftModel has no trainable parameters; load the "
                "adapter with is_trainable=True (the Stage-2/loader plan's "
                "deliverable) before continuing training"
            )
    else:
        if derived_quant == "4bit":
            # Standard k-bit preparation: fp32 norms and input grads.
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            ),
            autocast_adapter_dtype=True,
        )
    if config.gradient_checkpointing and _gradient_checkpointing_mode(model) == "off":
        # Non-reentrant only (RESEARCH_SPEC.md, ratified 2026-08-13): the
        # mode that coexists with forward hooks, which Stage 2 needs. Run on
        # any branch that did not already enable it during k-bit preparation.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    live_checkpointing_mode = _gradient_checkpointing_mode(model)
    expected_checkpointing_mode = (
        "non_reentrant" if config.gradient_checkpointing else "off"
    )
    if live_checkpointing_mode != expected_checkpointing_mode:
        raise ValueError(
            "failed to configure the ratified checkpointing mode: config "
            "requires %r but the live model reports %r"
            % (expected_checkpointing_mode, live_checkpointing_mode)
        )
    if config.gradient_checkpointing and not _input_grad_hook_handles(model):
        # Some transformers versions install this as a side effect of
        # gradient_checkpointing_enable and some stacks do not. Make the
        # requirement explicit without duplicating an existing hook.
        model.enable_input_require_grads()
        if not _input_grad_hook_handles(model):
            raise ValueError(
                "gradient checkpointing is enabled but no input-gradient "
                "hook was installed by enable_input_require_grads"
            )
    model.config.use_cache = False  # training never reuses a KV cache
    model.train()  # the shared loader hands back models in eval mode

    # 5. Encode once up front (short conversations, seconds), then derive
    # the step count and the checkpoint grid from the single homes.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            "tokenizer %r has neither pad_token_id nor eos_token_id"
            % type(tokenizer).__name__
        )
    examples = _encode_records(
        tokenizer, records, config.max_seq_len,
        mask_prompt_tokens=config.mask_prompt_tokens,
    )
    n_examples = len(examples)
    total_steps = derive_total_steps(n_examples, config)
    checkpoint_steps = checkpoint_schedule(
        total_steps, config.n_checkpoints, config.checkpoint_spacing
    )
    encoding_sha256 = encoding_digest(examples)
    renderer_sha256 = renderer_digest(tokenizer)
    adapter_dtype = _adapter_dtype(model)

    # 6. Manifest: write-once, then recompute-and-refuse-on-mismatch.
    parameter = next(model.parameters())
    current_manifest = _train_manifest(
        model_id=model_id, objective=objective, data_path=data_path,
        dataset_sha256=dataset_sha256, meta_sha256=meta_sha256,
        fold_system=fold_system, train_seed=train_seed,
        quant_label=derived_quant, bypassed_layer=bypassed_layer,
        device_type=parameter.device.type,
        dtype=str(getattr(model, "dtype", parameter.dtype)),
        n_examples=n_examples, total_steps=total_steps,
        checkpoint_steps=checkpoint_steps, config=config,
        data_manifest=data_manifest, encoding_sha256=encoding_sha256,
        renderer_sha256=renderer_sha256, adapter_dtype=adapter_dtype,
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _guard_train_manifest(manifest, current_manifest)
    else:
        manifest = json.loads(json.dumps(current_manifest))
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp_manifest.replace(manifest_path)
    identity_sha = _manifest_identity_sha(manifest)

    # 7. Optimizer, schedule, and resume.
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    if config.lr_schedule != "constant":
        raise ValueError(
            "lr_schedule must be 'constant' (optionally with warmup_steps), "
            "got %r" % config.lr_schedule
        )

    def lr_lambda(step_index):
        if config.warmup_steps and step_index < config.warmup_steps:
            return float(step_index + 1) / float(config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    device = parameter.device
    use_cuda = device.type == "cuda"
    # fp16 autocast plus a gradient scaler on the T4 (it has no bf16, the
    # declared deviation from the QLoRA recipe); fp32 and no autocast on a
    # CPU/MPS dev box.
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None
    runtime_state = {}
    next_step = _load_resume_state(
        resume_path, model, optimizer, scheduler, scaler, identity_sha,
        runtime_state=runtime_state,
    )

    append_jsonl(out_dir / "sessions.jsonl", {
        "session_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "max_steps_this_session": max_steps_this_session,
        "entry_step": next_step,
        "packages": current_manifest["packages"],
    })

    # 8. The micro-batch stream: deterministic per-epoch orders concatenated,
    # then chunked into accumulation groups that span epoch boundaries, so
    # only the run's final group may be partial.
    micro_batches = []
    for epoch in range(config.epochs):
        order = _epoch_order(train_seed, epoch, n_examples)
        for position, start in enumerate(
            range(0, n_examples, config.micro_batch_size)
        ):
            micro_batches.append(
                (epoch, position, order[start:start + config.micro_batch_size])
            )
    groups = [
        micro_batches[index * config.grad_accum_steps:
                      (index + 1) * config.grad_accum_steps]
        for index in range(total_steps)
    ]

    log_path = out_dir / "train_log.jsonl"
    steps_this_session = 0
    skipped_this_session = 0
    nonfinite_this_session = 0
    skip_streak = runtime_state["skip_streak"]
    saved_at = next_step - 1  # last step whose resume state is on disk
    for step in range(next_step, total_steps):
        if (
            max_steps_this_session is not None
            and steps_this_session >= max_steps_this_session
        ):
            break
        group = groups[step]
        group_loss = 0.0
        for epoch, position, indices in group:
            batch = _collate([examples[i] for i in indices], pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            if use_cuda:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss / len(group)
                scaler.scale(loss).backward()
            else:
                loss = model(**batch).loss / len(group)
                loss.backward()
            group_loss += float(loss.detach())

        loss_nonfinite = not math.isfinite(group_loss)
        if loss_nonfinite:
            nonfinite_this_session += 1
            print(
                "WARNING: non-finite training loss at step %d: %r"
                % (step, group_loss)
            )

        scaler_skipped, scaler_scale = _apply_step(
            scaler, optimizer, trainable, config.max_grad_norm
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        # Completing the group consumes this step index REGARDLESS of a
        # scaler skip, so indices stay hardware-invariant; the consequence,
        # documented rather than hidden, is that on fp16 hardware a step
        # index is AT MOST one applied update.
        steps_this_session += 1

        if scaler_skipped:
            skipped_this_session += 1
            print(
                "WARNING: grad scaler skipped step %d (scale %s)"
                % (step, scaler_scale)
            )

        append_jsonl(log_path, {
            "step": step,
            "loss": group_loss,
            "lr": learning_rate,
            "epoch": group[-1][0],
            "micro_in_epoch": group[-1][1],
            "scaler_skipped": bool(scaler_skipped),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        try:
            skip_streak = _update_skip_streak(
                skip_streak, scaler_skipped, step, scaler_scale
            )
        except RuntimeError:
            print(
                "SESSION NUMERICS: %d of %d steps skipped, %d non-finite losses"
                % (skipped_this_session, steps_this_session,
                   nonfinite_this_session)
            )
            raise

        scheduled = step in checkpoint_steps
        if scheduled:
            meta = {
                "checkpoint_step": step,
                "train_seed": train_seed,
                "objective": objective,
                "model_id": model_id,
                "quant_label": derived_quant,
                "dataset_path": str(data_path),
                "dataset_sha256": dataset_sha256,
                "meta_sha256": meta_sha256,
                "fold_system": fold_system,
                "bypassed_layer": bypassed_layer,
                "total_steps": total_steps,
                "config": dataclasses.asdict(config),
                "scaler_skipped": bool(scaler_skipped),
                "encoding_sha256": encoding_sha256,
                "renderer_sha256": renderer_sha256,
                "adapter_dtype": adapter_dtype,
                "created": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            _write_checkpoint(model, out_dir, step, meta)
            print("checkpoint written: step-%05d" % step)
        if (
            scheduled
            or (step + 1) % config.save_every == 0
            or step == total_steps - 1
        ):
            _save_resume_state(
                resume_path, step, model, optimizer, scheduler, scaler,
                identity_sha, skip_streak=skip_streak,
            )
            saved_at = step
        print("step %d/%d loss %.4f" % (step, total_steps - 1, group_loss))

    if steps_this_session and saved_at != next_step + steps_this_session - 1:
        # Stopped by max_steps_this_session between saves: never lose work.
        _save_resume_state(
            resume_path, next_step + steps_this_session - 1, model, optimizer,
            scheduler, scaler, identity_sha, skip_streak=skip_streak,
        )
    print(
        "session done: %d optimizer steps, next step %d of %d"
        % (steps_this_session, next_step + steps_this_session, total_steps)
    )
    print(
        "SESSION NUMERICS: %d of %d steps skipped, %d non-finite losses"
        % (skipped_this_session, steps_this_session, nonfinite_this_session)
    )
    return manifest


def _capture_training_state(model) -> dict:
    """Capture caller-owned mutable state that one training session borrows."""
    enable_input_grads = getattr(model, "enable_input_require_grads", None)
    hook_owner = getattr(enable_input_grads, "__self__", model)
    return {
        "training": model.training,
        "use_cache": model.config.use_cache,
        "gradient_checkpointing": bool(
            getattr(model, "is_gradient_checkpointing", False)
        ),
        "hook_owner": hook_owner,
        "had_hooks": hasattr(hook_owner, "_require_grads_hooks"),
        "hooks": getattr(hook_owner, "_require_grads_hooks", None),
        "had_hook": hasattr(hook_owner, "_require_grads_hook"),
        "hook": getattr(hook_owner, "_require_grads_hook", None),
    }


def _restore_training_state(model, state) -> None:
    """Undo only session-owned mode, cache, checkpointing, and hook state."""
    hook_owner = state["hook_owner"]
    if (
        not state["gradient_checkpointing"]
        and getattr(model, "is_gradient_checkpointing", False)
    ):
        model.gradient_checkpointing_disable()

    # transformers replaces its hook-handle list when checkpointing is
    # enabled. The original handles remain registered, so remove only the
    # new handles and then restore the caller's bookkeeping attributes.
    previous_handles = list(state["hooks"] or [])
    if state["hook"] is not None:
        previous_handles.append(state["hook"])
    previous_ids = {id(handle) for handle in previous_handles}
    current_handles = list(
        getattr(hook_owner, "_require_grads_hooks", None) or []
    )
    current_hook = getattr(hook_owner, "_require_grads_hook", None)
    if current_hook is not None:
        current_handles.append(current_hook)
    removed_ids = set()
    for handle in current_handles:
        if id(handle) not in previous_ids and id(handle) not in removed_ids:
            handle.remove()
            removed_ids.add(id(handle))

    if state["had_hooks"]:
        hook_owner._require_grads_hooks = state["hooks"]
    elif hasattr(hook_owner, "_require_grads_hooks"):
        del hook_owner._require_grads_hooks
    if state["had_hook"]:
        hook_owner._require_grads_hook = state["hook"]
    elif hasattr(hook_owner, "_require_grads_hook"):
        del hook_owner._require_grads_hook

    model.config.use_cache = state["use_cache"]
    model.train(state["training"])


def train_lora(model, tokenizer, data_path, out_dir, model_id, objective,
               config=DEFAULT_TRAIN_CONFIG, train_seed=42, quant_label=None,
               bypassed_layer=None, resume=True,
               max_steps_this_session=None) -> dict:
    """LoRA fine-tune while restoring the caller's borrowed model state.

    The implementation writes the scheduled PEFT checkpoints, guarded run
    manifest, append-only logs and resume state documented by this module.
    Its signature remains the binding training interface. Adapter attachment
    and trainability are intentional outputs; caller mode, ``use_cache``,
    gradient-checkpointing state, and pre-existing input-gradient hooks are
    restored on both success and failure.
    """
    training_state = _capture_training_state(model)
    try:
        return _train_lora_impl(
            model, tokenizer, data_path, out_dir, model_id, objective,
            config=config, train_seed=train_seed, quant_label=quant_label,
            bypassed_layer=bypassed_layer, resume=resume,
            max_steps_this_session=max_steps_this_session,
        )
    finally:
        _restore_training_state(model, training_state)


def matched_training_identity(manifest, cross_family=False) -> dict:
    """The fields two arms must share for "matched fine-tuning" to be true.

    The executable home of RESEARCH_SPEC.md's "All fine-tuning uses matched
    data volumes, optimization settings, checkpoint schedules, and random
    seeds": two runs are matched iff this dict is equal for both. Legitimate
    differences (dataset path and digests, objective, out_dir, timestamps,
    bypassed_layer, save_every, package versions) are excluded by
    construction.

    Batch treatment is strict (P6's reading, ratified 2026-08-15):
    micro_batch_size, grad_accum_steps AND the derived effective_batch are
    all audited, always. Matched arms train under the identical split, not
    merely the identical product, so a run that traded micro batch for
    accumulation (a laptop dev run, say) is by construction not a matched
    arm and fails this audit.

    SCOPE: with cross_family=False this audit runs WITHIN one model family's
    arms. Across families model_id necessarily differs and fold_system (with
    the dataset digests behind it) legitimately differs, so
    cross_family=True drops those two and compares the remaining shared
    constants, which is what carries the paper's "matched across models"
    sentence.
    """
    manifest = json.loads(json.dumps(manifest))
    config = _guarded_view(manifest)["config"]
    identity = {
        "config": config,
        "effective_batch": (
            config.get("micro_batch_size") * config.get("grad_accum_steps")
        ),
        "n_examples": manifest.get("n_examples"),
        "total_steps": manifest.get("total_steps"),
        "checkpoint_steps": manifest.get("checkpoint_steps"),
        "train_seed": manifest.get("train_seed"),
        "quant_label": manifest.get("quant_label"),
        "dtype": manifest.get("dtype"),
        "device_type": manifest.get("device_type"),
        "adapter_dtype": manifest.get("adapter_dtype"),
    }
    if not cross_family:
        # model_id catches an arm accidentally trained from the wrong base
        # (a dev-scale 0.5B manifest slipping into a 7B comparison).
        identity["model_id"] = manifest.get("model_id")
        identity["fold_system"] = manifest.get("fold_system")
        identity["renderer_sha256"] = manifest.get("renderer_sha256")
    return identity


def checkpoint_meta(adapter_dir) -> dict:
    """Read one checkpoint's train_meta.json sidecar (stdlib only).

    The eval row field checkpoint_step for a trained checkpoint is DEFINED
    as this file's checkpoint_step, which is why run_baseline.py reads it
    here instead of trusting an operator-typed flag.
    """
    path = Path(adapter_dir) / "train_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    for field in ("checkpoint_step", "train_seed"):
        if field not in meta:
            raise ValueError(
                "malformed training sidecar %s: missing required field %r"
                % (path, field)
            )
    return meta


def read_train_log(path) -> list:
    """Read train_log.jsonl, keeping the LAST row per step, sorted by step.

    The log is append-only, so a crash between resume saves means the
    replayed steps append duplicate step rows. This keep-last rule is the
    single implementation any appendix loss-curve figure may use.
    """
    latest = {}
    for row in _read_jsonl(path):
        latest[row["step"]] = row
    return [latest[step] for step in sorted(latest)]
