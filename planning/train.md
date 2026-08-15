# Plan: train, Stage-1 LoRA fine-tuning (M_D / M_C checkpoint creation)

Revision 3 of the live plan for the `train` scope (one live plan per
scope; revisions happen in this file, never in a sibling). Round zero
was reviewed in planning/train.critique-1.md (findings F1-F24),
revision 1 in planning/train.critique-2.md (findings F25-F39); the
disposition tables appended to each critique record the adjudications.
Revision 2 applied the accepted round-2 findings (all 15 accepted; F25
additionally escalated as P15). Revision 3 applies human rulings of
2026-08-15 and adds no new findings of its own:

- **P6 is RATIFIED** in its strict, split-matched reading ("everything
  in the training should be exactly the same for both models"). The
  matched quantity is not merely the effective-batch product: both arms
  use the identical `micro_batch_size`, the identical
  `grad_accum_steps`, and the identical derived `effective_batch`, and
  all three are audited, always. Only the reading is ratified; the
  VALUES stay proposed.
- **Scope is unchanged**: all three model families stay in, exactly as
  RESEARCH_SPEC.md has them (Qwen2.5-7B-Instruct with
  Qwen2.5-0.5B-Instruct for dev, Llama-3.1-8B-Instruct, Gemma-2-9b-it).
  Every multi-family element of THIS plan stays live: the three-family
  tokenizer preflight (verification item 2), the ratified E6 Gemma fold
  rule (constraint 2, WP-T2), the Gemma-2 fp16 open risk (risks
  section, now with the per-family fp16 sanity check), HF gating for
  Llama and Gemma, the `cross_family` audit mode of
  `matched_training_identity`, and the Gemma-2-9B storage numbers.
  Multi-family elements owned elsewhere are untouched by this plan and
  stay as they are: the loader's eager-attention handling
  (`models.py`) and the tiny Qwen2/Llama/Gemma2 fixtures with
  `_attn_implementation = "eager"` in tests/test_bypass.py (the
  layer-bypass plan's convention).
- **All other constants stay PROPOSED** (P1-P5, P7-P11, P13-P15),
  including P6's values; the team rules on them via
  planning/train.ratification-proposal.md, the short form of the
  pending list. P12 and P15 remain the OPEN escalations.

Written for an implementer who has RESEARCH_SPEC.md and INTERFACES.md
but was not in the planning conversation. Repo state verified against
the working tree on 2026-08-15 (branch eval-harness, commit 419468b).

## Scope

Build the fine-tuning lane in `src/algoverse/train.py` (currently a
TO-BUILD stub): LoRA (Low-Rank Adaptation; Hu et al. 2021,
https://arxiv.org/abs/2106.09685) supervised fine-tuning that turns a
base model M_0 into the deceptive checkpoint M_D (trained on
`data/finetune/m_d_train.jsonl`) and the control checkpoint M_C
(`m_c_train.jsonl`), with a checkpoint schedule, resume, and full
provenance. On the 7-9B production models this is LoRA with a 4-bit
frozen base (the QLoRA recipe, Dettmers et al. 2023,
https://arxiv.org/abs/2305.14314): the method is LoRA; quantization is
a memory decision for the T4. These adapter checkpoints are what Gate 1
evaluates and what Stage 2 continues from.

NON-GOALS (noted so they are not discovered as gaps later):

- Stage-2 lesioned fine-tuning is NOT planned here. This plan only
  guarantees the loop is REUSABLE for it (see "Stage-2 non-preclusion
  checklist"). The reinstall-at-load loader change on
  `load_model_and_tokenizer`, the `is_trainable` adapter-loading path,
  the k-bit re-preparation of a loaded continuation model, and the
  Stage-2 run layout stay with the Stage-2/loader plan, exactly as
  planning/layer-bypass.md §P2 assigned them.
- No training-data changes. `data.py` builds the datasets; the mandated
  regeneration (RESEARCH_SPEC.md, ratified 2026-08-14: the 155k offer
  and ratio-firewall changes invalidate previously built files) is an
  operational precondition of any training run, not work in this plan.
- No experiment-tracking service (wandb). Loss logging is a local JSONL
  (append-only, resumable, sufficient for an appendix figure). The
  notebook's wandb install is unused by this lane.

## Binding constraints this plan implements

1. **Matched fine-tuning** (RESEARCH_SPEC.md Methodology, binding
   sentence): "All fine-tuning uses matched data volumes, optimization
   settings, checkpoint schedules, and random seeds." Operationalized
   as: one frozen `TrainConfig`, one `checkpoint_schedule`
   implementation, one `train_seed`, shared verbatim by every arm, all
   recorded in a per-run manifest, plus an executable comparability
   home (`matched_training_identity`, below) whose scope, within one
   model family's arms, is stated explicitly in "Quantity homes".
2. **E6 / ratified Gemma fold rule** (RESEARCH_SPEC.md "Ratified
   decisions (2026-08-14)", prompt-delivery bullet;
   planning/first-full-review.md §E6): the training lane must REFUSE to
   fine-tune a fold-requiring model (Gemma-2) on data whose
   `manifest.json` says `fold_system: false`. The data build already
   records `fold_system` in the manifest and every meta row;
   enforcement lands here.
3. **E8 / checkpoint step convention** (pinned in
   src/algoverse/utils.py docstrings; RESEARCH_SPEC.md Open decisions):
   `state["step"]` is the LAST COMPLETED step; a loader returns
   step + 1; a brand-new run returns 0. train.py adopts this convention
   verbatim (0-indexed optimizer-update indices; details in WP-T5) and
   never invents a second one.
4. **Ratified gradient-checkpointing rule** (RESEARCH_SPEC.md
   2026-08-13): if gradient checkpointing is used, non-reentrant only
   (`use_reentrant=False`). This loop uses only that mode.
5. **Provenance bar**: the train lane meets the eval lane's standard
   (append-only outputs, manifest identity guards on resume, recorded
   package versions, derived-not-asserted state). "Derived, not
   asserted" applies to bypass state AND to quantization: caller
   bookkeeping (`bypassed_layer`, `quant_label`) is cross-checked
   against the live model and mismatches refuse (D2, WP-T4 step 1).

## Method provenance (papers fetched and read for this plan, 2026-08-15; re-verified independently by the critic's session, planning/train.critique-1.md F22)

- **LoRA** (Hu et al., https://arxiv.org/abs/2106.09685, read via
  ar5iv full text, https://ar5iv.labs.arxiv.org/html/2106.09685):
  ΔW = BA with A Gaussian-initialized and **B initialized to zero, so
  the adapter is an exact identity at step 0** (their §4.1). This plan
  turns that into a testable invariant (a fresh-adapter model must
  match the base model's logits before any update). Scaling is α/r;
  they "set α to the first r we try and do not tune it". Their main
  experiments adapt W_q, W_v only; Table 5 finds a fixed budget spent
  on MORE matrices at LOWER rank beats fewer matrices at higher rank.
  LoRA learning rate 2e-4 for GPT-2/GPT-3.
- **The QLoRA recipe** (Dettmers et al.,
  https://arxiv.org/abs/2305.14314, read via ar5iv full text): §4 /
  Figure 2, LoRA on **all linear transformer-block layers is required
  to match full-finetuning performance**; attention-only LoRA (the
  LoRA paper's default) does not. Appendix A: rank is unrelated to
  final performance once all layers are adapted (note: that evidence is
  instruction-tuning benchmark performance; using it here is a transfer
  assumption, see P2). Dropout 0.05-0.1 at 7B/13B (the paper's own text
  is inconsistent about which value goes with which scale, recorded
  honestly in P2). 7B recipe: r=64, α=16, lr 2e-4, **constant
  schedule**, batch 16, max_grad_norm 0.3, Adam β2=0.999. NF4 with
  double quantization (exactly what
  `load_model_and_tokenizer(quant="4bit")` already configures),
  compute dtype **bfloat16**. Gradient checkpointing is ASSUMED by the
  paper's memory accounting (§2's input-gradient figures and §3's
  paged-optimizer motivation imply it) but appears in no
  hyperparameter table; treating it as their setting is an inference,
  not a quote (critique F39). Its use here is independently forced by
  the T4 anyway.
- **Declared deviations from the QLoRA recipe**: (a) compute dtype is
  fp16 with a gradient scaler, not bf16; the T4 has no bf16
  (environment-forced; recorded in the manifest's dtype and flagged for
  the reproducibility appendix). (b) proposed r=16 rather than 64,
  justified by Dettmers et al.'s own rank-irrelevance finding (a
  transfer assumption, P2) plus checkpoint-storage arithmetic:
  all-linear r=16 on Qwen2.5-7B is ~40.4M LoRA parameters, ~161 MB
  fp32 (~81 MB fp16) per checkpoint, Gemma-2-9B ~216 MB fp32, and r=64
  is 4x that (~646 MB fp32 per 7B checkpoint), multiplied by the P10
  schedule across arms and models. (For calibration: attention-only
  r=16 would be ~40 MB fp32, but P1 proposes all-linear.) (c) no paged
  optimizer; Dettmers et al. needed it for 33-65B on a single 24/48 GB
  GPU; at 7-9B on a T4 plain AdamW fits, and bitsandbytes' paged AdamW
  remains the recorded fallback if OOM is observed. All three are
  listed with the pending constants (P14) so the human ratifies them
  together.

## Design decisions (made here, with reasons)

**D1: A small manual training loop; no TRL, no HF Trainer.**
The loop is ~150 lines over torch + peft + transformers (all already
required by INTERFACES.md for the training track). Reasons: (i) the
checkpoint schedule is load-bearing for Stage-3's R_t curves and must
follow the pinned step convention exactly; Trainer/TRL have their own
checkpoint formats, step semantics, and resume machinery
(trainer_state.json, save_steps) that would either fight the utils
convention or duplicate it behind an abstraction. (ii) the spec's
"matched optimization settings" claim is easiest to defend when every
setting is an explicit field of one dataclass rather than a
version-dependent Trainer default. (iii) Stage 2 must fine-tune a model
with a live forward hook (the permanent bypass); a bare loop makes the
hook's survival trivially auditable. (iv) TRL is not installed anywhere
today and adds version-drift surface for zero needed features (the
dataset is 1,500 short conversations; no packing, no distributed
training). Boring structure wins.

**D2: `train_lora` takes a READY model object, mirroring the eval
lane.** The caller (script) loads the model; the trainer attaches or
finds the adapter. This is the decision that makes Stage-2 reuse cheap:
a permanently bypassed model, or a base+M_D-adapter model, is passed in
and trained by identical code, exactly as eval evaluates all of them
through one function. `train_lora` cross-checks the `bypassed_layer`
bookkeeping argument against live `bypass_state(model)` and the
`quant_label` bookkeeping argument against the live model's
quantization (`_derive_quant`), the same derive-and-refuse pattern as
`run_negotiation_eval`.

**D3: Checkpoint artifact = a PEFT adapter directory.** Each scheduled
checkpoint is `out_dir/checkpoints/step-NNNNN/` written by
`model.save_pretrained` (adapter_config.json + adapter_model.safetensors)
plus a `train_meta.json` sidecar. That is precisely what the eval lane
already consumes (`load_model_and_tokenizer(..., adapter_path=...)`) and
hashes (`eval._adapter_digest`). No new artifact type is invented.
`utils.save_checkpoint` is deliberately NOT used for these:
`model.state_dict()` on a 4-bit PeftModel serializes the whole quantized
base (gigabytes, wrong artifact); the step CONVENTION is adopted from
utils (constraint 3), the storage function is not.

**D4: Resume state is separate from science checkpoints.**
`out_dir/resume.pt` holds {step, adapter-only weights
(`get_peft_model_state_dict`), optimizer, scheduler, grad-scaler, RNG
states, identity fingerprint}, written atomically (tmp + replace, the
utils pattern) every `save_every` optimizer steps AND at every scheduled
checkpoint and at the end. This decouples crash cost (operational,
`save_every`) from the checkpoint schedule (methodological, ratified).

**D5: Loss masking via the prompt-prefix property, checked loudly.**
For each conversation: `prompt_text = apply_chat_template(messages[:-1],
add_generation_prompt=True, tokenize=False)`; `full_text =
apply_chat_template(messages, tokenize=False)`. Require
`full_text.startswith(prompt_text)` AND
`full_ids[:len(prompt_ids)] == prompt_ids` (both encoded with
`add_special_tokens=False`, the repo-wide single-BOS contract from
`eval._encode_chats`); labels are -100 on the prompt prefix and the
token ids on the completion (assistant reply plus the template's
end-of-turn tokens, so the model learns to stop). Either check failing
raises naming the record index; never silent truncation or misaligned
labels. This is believed to hold for Qwen2.5 / Llama-3.1 / Gemma-2
templates (the generation prompt is a textual prefix of the assistant
turn in all three) and is CHECKED against the real tokenizers of all
three families by the preflight (WP-T3 `encode_preflight`, verification
item 2) before any production run; the raise is the guard against
future template drift. Assistant-only masking itself is a PROPOSED
methodological constant (P8); the mechanics above are the
implementation either way.

**D6: Right padding at collate time, without touching tokenizer
state.** Training pads on the right (no generation involved; left
padding is `generate_batch`'s concern). The collator pads plain id
lists itself (input_ids with pad id, labels with -100, attention_mask
zeros) rather than calling `tokenizer.pad`, so it neither reads nor
mutates `tokenizer.padding_side`; `generate_batch` mutates that shared
state to "left", and the trainer must be immune to call order.

**D7: Strict fold-compatibility guard, both directions.** The ratified
minimum (constraint 2) is: fold-requiring model + `fold_system: false`
data → refuse. This plan additionally refuses the converse
(non-fold-requiring model + `fold_system: true` data): eval prompts for
Qwen/Llama always carry a system turn, so training them on folded data
would create a train/eval prompt-distribution mismatch, and on Qwen
specifically, a missing system turn makes the chat template inject its
own default system prompt, silently changing the training distribution.
The converse direction EXTENDS the ratified rule; it is pending item
P12, now explicitly escalated (critique F14): the human decides whether
the unratified converse may be live as a refusal from day one or must
warn-only until ratified. Fold need is detected from the live tokenizer
with `eval._system_fold_needed` on a fixed synthetic probe
(`[{"role": "system", ...}, {"role": "user", ...}]`), the same
detection the eval runner uses, never hardcoded per model name.

**D8: Objective/dataset cross-check, against recorded counts.**
`objective` ("deceptive" | "control") is caller bookkeeping, verified
against the dataset's `.meta.jsonl` AND the data manifest's recorded
composition (data.py writes `md_deceptive` / `mc_deceptive` /
`n_per_dataset`): "deceptive" requires the meta deceptive-row count to
equal the manifest's `md_deceptive` and the structural expectation
(n // 2); "control" requires zero and `mc_deceptive == 0`. This makes
both the swapped-file accident (training "M_C" on m_d_train.jsonl via a
path typo) and a truncated/hand-mixed file a loud failure instead of a
silently wrong arm (critique F11).

## Module map

| Home | Contents |
|---|---|
| `src/algoverse/train.py` | Everything below. Module-level imports stay stdlib + stdlib-importable-algoverse only (`eval` and `data` qualify; `utils` does NOT, it imports numpy/torch at module level, so utils, torch, peft, and transformers are all imported inside functions, mirroring eval.py's discipline) so the module imports on a stdlib-only box and pure tests can run. |
| `scripts/run_finetune.py` | Thin argparse CLI mirroring run_baseline.py: loads the model via `load_model_and_tokenizer`, builds/loads nothing else itself, calls `train_lora`. |
| `scripts/run_baseline.py` | MINIMAL cross-lane edit (deliberate; critique F10, narrowed by F25/F26): when `--adapter` points at a directory containing `train_meta.json`, read it via `train.checkpoint_meta`, then apply a PER-FIELD treatment. `checkpoint_step`: adopt from the sidecar when the flag is omitted; raise if a passed flag mismatches. `train_seed`: adoption SUSPENDED pending P15 (the ratified row convention says train_seed is null for Stage-0/1 rows; adopting the sidecar's value would contradict it, so rows stay null when the flag is omitted); a passed flag mismatching the sidecar still raises. `bypassed_layer`: NEVER adopted and never cross-checked against `--bypassed-layer`, because the sidecar value is TRAINING-time provenance while the flag installs an EVAL-time lesion, and the two legitimately differ (the A_l sweep bypasses layers of an intact-trained M_D: sidecar null, flag set); instead, a NON-null sidecar `bypassed_layer` raises immediately with "this checkpoint was trained under a permanent bypass; evaluating it requires the reinstall-at-load loader path, the Stage-2/loader plan's deliverable". Everything else in the script untouched. |
| `tests/test_train_pure.py` | New, stdlib-only: schedule (both spacings, pinned vectors), derive_total_steps (pinned vectors), masking/encoding (stub tokenizer), fold guard, objective guard, manifest identity, matched_training_identity, read_train_log, checkpoint_meta. Hardened runner per repo convention. |
| `tests/test_train.py` | New, guarded (torch+transformers+peft; loud SKIP otherwise, test_bypass.py pattern): tiny-model training behavior, checkpoints (including same-step rewrite), resume exactness, seeded-init determinism, bypass compatibility, adapter round-trip. |
| `INTERFACES.md` | One proposed contract addition (text in WP-T8). Human-owned edit; agents never touch INTERFACES.md. |
| Untouched | `models.py` (loader unchanged this scope), `utils.py`, `data.py`, `eval.py`, `metrics.py`, `tasks.py`, existing tests. |

train.py's internal layout (one module; no package split; assist-level
codebase):

- `TrainConfig` frozen dataclass + `DEFAULT_TRAIN_CONFIG` (proposed
  values, pending ratification; same defaults-in-code-until-ratified
  pattern the Gate-1 constants followed).
- `checkpoint_schedule(total_steps, n_checkpoints, spacing)`: pure,
  formulas pinned in WP-T1.
- `derive_total_steps(n_examples, config)`: pure,
  `ceil(epochs * ceil(n_examples / micro_batch_size) /
  grad_accum_steps)`; the single implementation of the total_steps
  derivation, called by the loop (WP-T4 step 5) and the pure tests
  (critique F34).
- `_read_jsonl(path)`: stdlib JSONL reader (~6 lines, one object per
  line, same semantics as `utils.read_jsonl`, which is unusable here
  because utils imports numpy/torch at module level; critique F5).
- `dataset_digest(path)`: sha256 of file bytes (used for both the
  records file and the meta file).
- `load_training_data(data_path)`: records + sibling meta rows + data
  manifest; shape and alignment checks (WP-T2).
- `check_fold_compatibility(tokenizer, data_manifest, records)` (D7).
- `check_objective(objective, meta_rows, data_manifest)` (D8).
- `encode_conversation(tokenizer, messages, max_seq_len,
  mask_prompt_tokens=True)` → `(input_ids, labels)` as plain lists
  (pure-testable; D5; the flag is the masking policy's single home,
  critique F33).
- `encode_preflight(tokenizer, records, max_seq_len=None)`: runs the
  D5 checks over every record and returns length statistics
  ({n, max_len, p95_len, mean_len, overflow_count-at-cap}); prefix
  violations raise, overflow is COUNTED not raised, so the function can
  measure a proposed cap (critique F15/F16; grounds P7).
- `_collate(examples, pad_token_id)` → tensors (torch inside; D6).
- `_epoch_order(train_seed, epoch, n)`: deterministic permutation,
  a pure function of its arguments (PYTHONHASHSEED-independent integer
  seed derivation, e.g. `train_seed * 100003 + epoch`).
- `train_lora(model, tokenizer, data_path, out_dir, model_id, objective,
  config=DEFAULT_TRAIN_CONFIG, train_seed=42, quant_label=None,
  bypassed_layer=None, resume=True, max_steps_this_session=None)` →
  manifest dict. THE central function.
- `_derive_quant(model)`: "4bit" | "none" derived from the live model
  (bitsandbytes 4-bit layers / `model.config` quantization config);
  cross-checked against the caller's `quant_label` (critique F19).
- `_write_checkpoint(model, out_dir, step, meta)`: adapter dir +
  train_meta.json (D3; rewrite-safe, WP-T5).
- `_save_resume_state` / `_load_resume_state` (D4; convention per E8;
  identity block per WP-T5).
- `_train_manifest(...)` + `_guard_train_manifest(existing, current)`:
  write-once, recompute-and-refuse-on-mismatch.
- `matched_training_identity(manifest, cross_family=False)`: the
  executable "matched fine-tuning" home (scope documented in "Quantity
  homes").
- `checkpoint_meta(adapter_dir)`: reads and returns a checkpoint's
  train_meta.json (stdlib; consumed by run_baseline.py's guard).
- `read_train_log(path)`: stdlib reader implementing the
  keep-last-row-per-step rule, returns rows sorted by step; the named
  home of the appendix loss curve (critique F24).

## Quantity homes (one home per reported quantity)

New reported/paper-load-bearing quantities introduced by this lane:

- **The checkpoint step grid t** (the x-axis of Stage-3 R_t recovery
  curves): `train.checkpoint_schedule` is the ONLY implementation;
  its output is recorded verbatim in `train_manifest.json`
  ("checkpoint_steps") and realized as the checkpoint directories.
- **The training-configuration table** (reproducibility appendix):
  `train_manifest.json`, written by the single writer
  `train._train_manifest` inside `train_lora`.
- **The "matched data volumes, optimization settings, checkpoint
  schedules, and random seeds" claim**:
  `train.matched_training_identity(manifest, cross_family=False)`
  returns exactly the manifest fields that must be equal across arms.
  Its config portion is DEFINED as the same field set the manifest
  guard uses (the full TrainConfig asdict minus `save_every`; one list,
  not two drifting prose lists; critique F2), PLUS the derived key
  `effective_batch = micro_batch_size * grad_accum_steps`, PLUS
  {model_id, n_examples, total_steps, checkpoint_steps, train_seed,
  quant_label, dtype, device_type, fold_system}. model_id's inclusion
  (critique F27) is what catches an arm accidentally trained from the
  wrong base (a dev-scale 0.5B manifest slipping into a 7B arm
  comparison must fail the audit, not pass it). Excluded (legitimately
  differ): dataset path and digests, objective, out_dir, timestamps,
  bypassed_layer, save_every, package versions. BATCH TREATMENT IS
  STRICT AND SETTLED (P6 ratified 2026-08-15, split-matched reading):
  `micro_batch_size`, `grad_accum_steps`, AND the derived
  `effective_batch` are all audited, always. Matched arms train under
  the identical split, not merely the identical product; there is no
  conditional path that drops the split fields, and a run using a
  different split (e.g. a dev box trading micro batch for accumulation)
  is by construction not a matched arm and fails the audit. SCOPE
  (critique F23): this audit runs WITHIN
  one model family's arms; across families, model_id necessarily
  differs and fold_system (with the dataset digests behind it)
  legitimately differs, so the cross-family half of the spec sentence
  is carried by
  `matched_training_identity(manifest, cross_family=True)`, which
  drops model_id and fold_system (critique F27) and compares the
  remaining shared constants across families. Stage-2/analysis asserts
  within-family equality before any cross-arm number is reported, and
  cross-family equality before the paper's "matched across models"
  sentence is written.
- **Training loss curves** (appendix, optional): rows appended to
  `train_log.jsonl` by the loop (one row per completed optimizer step);
  the reading side is `train.read_train_log` (keep-last-row-per-step),
  the single function any figure script may use (critique F24).

No existing quantity moves: tau (`metrics.tau_with_ci`), gain
(`metrics.tau_gain`), A_l (`metrics.bypass_effect`), R_t
(`metrics.recovery`), Gate-1 verdict (`metrics.gate1_decision` via
`eval.gate1_report`).

The eval row field `checkpoint_step` for a trained checkpoint is
DEFINED as the `checkpoint_step` value in that checkpoint's
`train_meta.json` (see step convention, WP-T5). It is no longer
operator-copied on trust: run_baseline.py adopts it from the sidecar
when the flag is omitted and refuses an explicitly passed mismatch
(module map; critique F10), so a transposed digit cannot silently move
a point on the R_t curve.

## Work packages

### WP-T1: TrainConfig and the checkpoint schedule

`TrainConfig` (frozen dataclass; `dataclasses.asdict` goes straight into
the manifest): `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`
(tuple of module-name suffixes), `learning_rate`, `lr_schedule`
("constant"), `warmup_steps`, `weight_decay`, `adam_beta1`, `adam_beta2`,
`max_grad_norm`, `epochs`, `micro_batch_size`, `grad_accum_steps`,
`max_seq_len`, `mask_prompt_tokens` (bool), `n_checkpoints`,
`checkpoint_spacing` ("even" | "doubling"), `gradient_checkpointing`
(bool), `save_every` (resume cadence, OPERATIONAL; excluded from
matched_training_identity and from the resume identity guard, recorded
anyway).

`DEFAULT_TRAIN_CONFIG` carries the PROPOSED values (pending list below);
the docstring marks them PROPOSED pending ratification and cites the
LoRA (https://arxiv.org/abs/2106.09685) / QLoRA-recipe
(https://arxiv.org/abs/2305.14314) provenance above.

`target_modules` is ALWAYS passed explicitly into `LoraConfig`, never
left to peft's default: peft's built-in mapping targets only `q_proj`
and `v_proj` for Qwen, Llama, and Gemma
(https://huggingface.co/docs/peft/package_reference/lora), so omitting
it would silently produce an attention-only adapter while the manifest
still recorded whatever P1 says.

`checkpoint_schedule(total_steps, n_checkpoints, spacing)`:
returns a sorted, duplicate-free list of 0-based step INDICES in
`[0, total_steps - 1]`, always containing `total_steps - 1` (the final
checkpoint). Raises if `n_checkpoints < 1` or
`n_checkpoints > total_steps`. The two spacings are pinned by exact
formula (critique F3), so P10's ratification is over a determined
object:

- **"even"**: `[ceil(k * total_steps / n_checkpoints) - 1
  for k in 1..n_checkpoints]`. Whenever
  `n_checkpoints <= total_steps`, consecutive values differ by at
  least 1, so the list is duplicate-free by construction.
  Pinned vectors: (total=100, n=4) → [24, 49, 74, 99];
  (total=10, n=3) → [3, 6, 9]; (total=282, n=6) →
  [46, 93, 140, 187, 234, 281].
- **"doubling"**: `sorted({(total_steps - 1) // 2**j
  for j in 0..n_checkpoints-1})`, a log-spaced grid dense in EARLY
  training (relevant if recovery is fast-then-flat). If deduplication
  collapses the set below n_checkpoints (small totals), RAISE naming
  total_steps and n_checkpoints (the operator picks a feasible n; this
  mirrors the n_checkpoints > total_steps raise). Pinned vectors:
  (total=282, n=6) → [8, 17, 35, 70, 140, 281];
  (total=20, n=6) → [0, 1, 2, 4, 9, 19];
  (total=10, n=6) → raises.

Derived counts: micro-batches per epoch =
`ceil(n_examples / micro_batch_size)`; accumulation groups span epoch
boundaries (the stream of micro-batches across all epochs is chunked
into groups of `grad_accum_steps`; only the final group of the run may
be partial); `total_steps = ceil(total_micro_batches /
grad_accum_steps)`. This derivation's single callable home is the pure
function `derive_total_steps(n_examples, config)` (critique F34); the
loop (WP-T4 step 5) and the tests call it, prose never substitutes for
it. Worked example: n=1500, micro=2, accum=8, epochs=3
→ 750 micro-batches/epoch, 2250 total, 282 steps, final group holding
2 micro-batches. With every arm training on the same n and config,
total_steps is identical across arms by construction.

**Acceptance tests** (test_train_pure.py): schedule sorted/unique/
in-range/contains final/deterministic; ALL pinned vectors above for
both spacings, including the doubling-collapse raise and the
n_checkpoints > total_steps raise; `derive_total_steps` pinned for
the worked example (n=1500, micro=2, accum=8, epochs=3 → 282 steps)
and the dev vector (n=40, micro=4, accum=1, epochs=1 → 10 steps).

### WP-T2: Data loading and the two guards

`load_training_data(data_path)`:
- `train._read_jsonl(data_path)` → records; every record must be
  `{"messages": [...]}` with roles in {"system","user","assistant"},
  final turn "assistant" (raise with index otherwise).
- Sibling meta: `<stem>.meta.jsonl` (m_d_train.jsonl →
  m_d_train.meta.jsonl); missing → raise (regenerated data always has
  it, and the objective guard needs it).
- Alignment (critique F12): `len(meta_rows) == len(records)` or raise
  naming both counts; every meta row must carry `behavior` and
  `fold_system` keys (raise with index).
- Data manifest: `data_path.parent / "manifest.json"`; missing → raise
  (no provenance, no training; the mandated regeneration produces it).
- Returns (records, meta_rows, data_manifest).

`check_fold_compatibility(tokenizer, data_manifest, records)` (D7):
- `fold_required = eval._system_fold_needed(tokenizer, FOLD_PROBE)` with
  the fixed synthetic probe (module constant).
- `fold_built = bool(data_manifest["fold_system"])` (KeyError → raise:
  stale pre-fold-provenance data is not trainable).
- `fold_required != fold_built` → raise, message naming the model's
  requirement, the manifest value, and the fix (`--fold-system` rebuild
  for Gemma; the unfolded build for Qwen/Llama). This is E6 enforcement
  plus the D7 converse (converse enforcement mode pending P12).
- Belt-and-braces record scan: `fold_system: true` data must contain no
  system turns anywhere; `false` data must have a leading system turn in
  every record (catches hand-edited or mixed files). Raise with index.
- Returns `fold_built` for the manifest.

`check_objective(objective, meta_rows, data_manifest)` (D8, critique
F11): objective must be "deceptive" or "control". Let `k` = count of
meta rows with `behavior == "deceptive"` and `n = len(meta_rows)`.
"deceptive" requires `k == data_manifest["md_deceptive"]` and
`k == n // 2`; "control" requires `k == 0` and
`data_manifest["mc_deceptive"] == 0`. Also
`n == data_manifest["n_per_dataset"]`. Raise naming every number
otherwise.

Digests: `dataset_sha256` (records file) and `meta_sha256` (meta file)
are both computed and both GUARDED in the train manifest; the data
manifest is recorded verbatim (critique F12, F20). Honest residual: a
meta/records pair regenerated together but hand-edited after build
carries valid digests and passes; the recorded manifest makes that
auditable post hoc, not refused up front.

**Acceptance tests** (test_train_pure.py, tmp dirs + the
RecordingTokenizer stub pattern from tests/test_eval_pure.py):
fold-requiring stub + `fold_system: false` manifest → ValueError (the
E6 case); fold-requiring + true → passes; non-fold + true → ValueError
(the D7 converse); missing manifest / missing fold_system key / missing
meta file / meta-records length mismatch → each raises; record-scan
catches a smuggled system turn in folded data; objective guard passes
on exact builder-shaped counts and fails on a truncated meta (k > 0 but
k != md_deceptive), on k == 0 under "deceptive", and on k > 0 under
"control". Additionally a REAL-DATA wiring test: build a tiny dataset
via `data.build_finetune_datasets(tmp, n_per_dataset=8, seed=0)` (pure,
fast) and assert `load_training_data` + both guards accept the real
builder's output for both objectives, folded and unfolded.

### WP-T3: Tokenization and masking

`encode_conversation(tokenizer, messages, max_seq_len,
mask_prompt_tokens=True)` (D5 mechanics; the flag is in the signature
because the function is the policy's single home, critique F33):
returns `(input_ids, labels)` as plain Python lists. Both renders use
`apply_chat_template(..., tokenize=False)`; both encodes use
`add_special_tokens=False` (single-BOS contract; the template supplies
BOS where its family wants one, exactly as `eval._encode_chats`
documents). String-prefix and id-prefix checks raise on violation.
`len(input_ids) > max_seq_len` → raise naming the record (silently
truncating could cut the structured final line, the one thing the
whole eval measures, so overflow is an error, never a trim).
When `mask_prompt_tokens` is false (full-sequence variant, if ratified),
labels equal input_ids everywhere; the function is the single home of
the policy either way.

`encode_preflight(tokenizer, records, max_seq_len=None)` (critique
F15/F16): runs `encode_conversation`'s checks over every record with
overflow COUNTED instead of raised, returning
{n, max_len, p95_len, mean_len, overflow_count} (overflow_count against
the passed cap when given). Two jobs: (a) verify the D5 prefix
properties against REAL tokenizers for all three model families before
any production run; (b) produce the token-length histogram that grounds
the P7 cap proposal. Invocation lives in verification item 2.

Encoding runs ONCE up front over all records (1,500 short conversations;
seconds), producing the in-memory example list the loop consumes.

**Acceptance tests** (test_train_pure.py, with a stub chat tokenizer
that renders `[BOS]` + role-tagged turns and encodes
whitespace-separated tokens to small ids): prompt tokens are all -100;
completion tokens equal their ids; the template's end-of-turn marker is
inside the supervised span; exactly one BOS; a stub whose full render is
NOT prefixed by its prompt render → raise; an over-length conversation →
raise naming the index; `mask_prompt_tokens=False` yields labels ==
input_ids; `encode_preflight` returns correct counts on the same stubs
(including a nonzero overflow_count without raising).

### WP-T4: The training loop

`train_lora(...)` order of operations:

1. `_validate` bookkeeping: objective enum; `bypassed_layer` vs live
   `bypass_state(model)` (same refusal as `run_negotiation_eval`;
   Stage-1 calls pass None and an intact model); `quant_label` vs
   `_derive_quant(model)` (derive-and-refuse; the manifest records the
   DERIVED value; critique F19).
2. `utils.set_seed(train_seed)`, the FIRST RNG-touching operation, and
   in particular BEFORE adapter attach (critique F1): peft draws
   lora_A's Gaussian/kaiming init from the global torch RNG, so seeding
   after attach would leave the init governed by OS entropy and break
   the spec's matched-random-seeds requirement. Everything downstream
   (init, epoch shuffles, dropout) is now a function of train_seed; a
   later resume overwrites RNG state from resume.pt.
3. Load data; run both WP-T2 guards; compute `dataset_sha256` and
   `meta_sha256`.
4. Adapter attach: if `model` is already a PeftModel (Stage-2
   continuation), require trainable parameters (else raise pointing at
   the `is_trainable` loading path, which is the Stage-2 plan's
   deliverable) and skip creation; otherwise
   `peft.get_peft_model(model, LoraConfig(r, alpha, dropout,
   target_modules, bias="none", task_type="CAUSAL_LM"))`. For 4-bit
   FRESH attaches, `peft.prepare_model_for_kbit_training(model,
   use_gradient_checkpointing=config.gradient_checkpointing,
   gradient_checkpointing_kwargs={"use_reentrant": False})` BEFORE
   get_peft_model (the standard k-bit preparation: fp32 norms, input
   grads). Gradient-checkpointing + input-grads enabling
   (`model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=
   {"use_reentrant": False})` + `enable_input_require_grads`) runs on
   BOTH branches when configured (critique F18); fp32-norm k-bit
   RE-preparation of a loaded continuation model is explicitly the
   Stage-2 loader plan's obligation (see non-preclusion checklist).
   `model.config.use_cache = False` for training; `model.train()`
   (the shared loader returns eval mode).
5. Encode all examples (WP-T3); `total_steps =
   derive_total_steps(n_examples, config)` (the WP-T1 pure function;
   critique F34) and `checkpoint_steps = checkpoint_schedule(...)`.
6. Manifest: build the current manifest; if `out_dir/train_manifest.json`
   exists, `_guard_train_manifest` compares field-by-field and raises
   listing mismatched fields. GUARDED: model_id, quant_label (derived),
   dataset_sha256, meta_sha256, fold_system, objective, train_seed, the
   full TrainConfig asdict minus `save_every`, checkpoint_steps,
   total_steps, n_examples, bypassed_layer, device_type, dtype.
   RECORDED but NOT guarded: dataset_path (a mount point, not an
   identity; critique F20), the data manifest verbatim, package
   versions, `created` (first-session UTC timestamp), save_every.
   Write-once, atomic (tmp + replace). Guard comparisons happen after
   a JSON round-trip of the CURRENT manifest (json.loads of its
   serialization), so tuple-typed config fields (target_modules)
   compare equal to their list form in the loaded file instead of
   refusing every resume (critique F29). Per-session operational
   values are NOT in the write-once manifest at all: each `train_lora`
   invocation appends one row ({session_start (UTC),
   max_steps_this_session, entry_step (the next step index at entry),
   package versions}) to `out_dir/sessions.jsonl`, append-only and
   never guarded; that file is the defined home of session history
   (critique F28: "manifest history" previously named a mechanism that
   a write-once manifest cannot provide).
7. Resume: `_load_resume_state` (WP-T5) → next step index, restored
   adapter/optimizer/scheduler/scaler/RNG; brand-new → step 0, fresh
   AdamW (`torch.optim.AdamW(trainable_params, lr, betas,
   weight_decay)`) and schedule (constant or constant-with-warmup via
   LambdaLR). `resume=False` is a FRESH-RUN ASSERTION, not an
   overwrite (critique F36): it raises if out_dir already contains
   `train_manifest.json` or `resume.pt` (retraining in place would
   append a second run's rows to train_log.jsonl, which
   read_train_log's keep-last rule would then silently merge into one
   curve). The CLI never passes it; it exists for tests and deliberate
   fresh starts into fresh directories.
8. The loop: iterate micro-batches in the deterministic global order
   (`_epoch_order` per epoch, concatenated; resume skips
   `next_step * grad_accum_steps` micro-batches WITHOUT forward
   passes). Per micro-batch: forward under
   `torch.autocast("cuda", torch.float16)` + `GradScaler` when the
   model is on CUDA (fp32, no autocast, on CPU/MPS dev); mean
   cross-entropy over supervised tokens (labels -100 ignored), divided
   by the ACTUAL number of micro-batches in the current accumulation
   group (equal to grad_accum_steps everywhere except possibly the
   run's final group; dividing the partial group by the full accum
   would silently shrink the last update, which is always a scheduled
   checkpoint; critique F13); backward. Per group on CUDA:
   `scaler.unscale_(optimizer)`; clip trainables to `max_grad_norm`;
   `scaler.step(optimizer)` (which SKIPS the update when inf/NaN
   gradients were found); `scaler.update()`; `scheduler.step()`;
   `optimizer.zero_grad(set_to_none=True)`. On CPU/MPS: clip;
   `optimizer.step()`; `scheduler.step()`; zero. Completing the group
   consumes step index `t` REGARDLESS of a scaler skip (critique F6):
   indices stay hardware-invariant; the log row records
   `scaler_skipped`, and a scheduled checkpoint written at a skipped
   step records `scaler_skipped: true` in its train_meta.json (its
   adapter state equals the previous step's; an honest, auditable
   record). Documented consequence: on fp16 hardware a step index is
   AT MOST one applied update. Objective documentation (critique F13):
   the loss is the equal-weighted mean of per-micro-batch mean
   cross-entropy over supervised tokens; this differs from a global
   token-weighted mean when supervised-token counts vary per
   conversation; a deliberate choice, matched across arms, stated in
   the loop docstring with the alternative named.
9. After each completed step: append a `train_log.jsonl` row
   ({step, loss (group mean), lr, epoch, micro_in_epoch,
   scaler_skipped, timestamp}); if `t` in checkpoint_steps →
   `_write_checkpoint`; if `t` hits the `save_every` cadence or a
   checkpoint was just written or `t` is the final step →
   `_save_resume_state`.
10. `max_steps_this_session`: stop cleanly after that many optimizer
    steps this invocation (resume state saved). Operational, for
    Colab session limits; recorded per session in `sessions.jsonl`
    (step 6; critique F28) and never identity-guarded. This is also
    how the resume acceptance test interrupts a run without
    process-kill theatrics.
11. Return the manifest dict.

train_log.jsonl is append-only; a crash between resume saves means
re-run steps append duplicate step rows; readers keep the LAST row per
step, and `train.read_train_log` is the single implementation of that
rule (critique F24; mirrors the eval lane's append-only discipline).
Log rows are not identity-guarded.

**Acceptance tests** (tests/test_train.py, guarded; tiny 4-layer Qwen2
from the test_bypass `_tiny_model` pattern + the stub chat tokenizer,
vocab-safe ids < 128; peft required, loud skip without it). Deliberate
coverage boundary: the loop tests use the Qwen2 tiny fixture ONLY,
because nothing in the loop is family-dependent; family-dependent
behavior lives in the chat template, and it is covered against the REAL
Qwen2.5 / Llama-3.1 / Gemma-2 tokenizers by the preflight (verification
item 2), not by tiny fixtures. If a future test does instantiate the
tiny Llama or Gemma2 fixtures, it must keep the layer-bypass plan's
`config._attn_implementation = "eager"` convention.
- **Step-0 identity** (LoRA B=0): after adapter attach, before any
  update, logits on a fixed batch equal the pristine base model's
  (torch.equal, fp32 CPU).
- **Trains**: ~30 steps on ~16 tiny synthetic conversations; final
  train_log loss < first loss; every base-model parameter byte-identical
  to a pre-training snapshot; at least one lora_A/lora_B tensor changed.
- **Schedule realized exactly**: checkpoint dirs exist for exactly the
  scheduled indices, none else; each contains adapter_config.json,
  adapter weights, and a train_meta.json whose fields match WP-T5.
- **Seed governs init and run** (critique F1): two fresh runs with
  identical config/train_seed into two out_dirs, with the harness
  PERTURBING the ambient global RNG between them (e.g. drawing a few
  thousand torch randoms under a different seed), produce (a)
  byte-identical INITIAL adapter state dicts at attach time and (b)
  byte-identical final adapter state dicts. This fails if seeding ever
  moves back after attach.

### WP-T5: Checkpoints, resume, step convention

**Step convention (constraint 3, adopted verbatim from utils.py):**
optimizer updates carry 0-based indices 0 .. total_steps-1. Any stored
"step" field is the index of the LAST COMPLETED update;
`_load_resume_state` returns `state["step"] + 1` (the next index to
run); no resume file → 0. `checkpoint_step` in train_meta.json,
checkpoint directory names (`step-%05d`), train_log rows, and eval rows
all use this same index: one number, one meaning, stated in the module
docstring with a worked example ("a 282-update run's final checkpoint is
step-00281"). The write-up may relabel the axis as "updates completed"
(index + 1); the code never does.

`_write_checkpoint(model, out_dir, step, meta)`, ordering pinned:
(i) if a stale tmp sibling exists (leftover from a crash between save
and publish), `shutil.rmtree` it first; reusing it uncleaned could
smuggle stale files (e.g. an old adapter_model.bin) into the published
directory, and `eval._adapter_digest` hashes .bin AND .safetensors
when both exist, so the checkpoint's eval identity would silently
change (critique F31). (ii) `model.save_pretrained` writes the adapter
into the tmp dir. (iii) `train_meta.json` is written INSIDE the tmp
dir, before publication, so no crash window can ever produce a
loadable adapter directory without its sidecar, which run_baseline's
guard would otherwise silently treat as externally produced and skip
cross-checking (critique F30). (iv) if the destination
`checkpoints/step-%05d/` already exists (the crash-window rerun path:
checkpoint written, resume state not yet saved, session killed, rerun
replays the step), `shutil.rmtree` it, then `os.replace` the tmp into
place (POSIX rename onto a non-empty directory raises ENOTEMPTY, which
would otherwise wedge every rerun at the same step, critique F7). The
rewrite is safe because it is WHOLESALE: the replayed step's content
is equivalent up to floating-point nondeterminism (bit-exact on the
CPU test rig; NOT guaranteed on CUDA fp16, where torch makes no
run-to-run bit-equality promise, critique F32), and directory
replacement means no mixed-state checkpoint can result. Documented
corollary (F32): an eval run that already consumed the pre-crash
version of that checkpoint will refuse to resume on `adapter_digest`,
loudly and correctly. A half-written adapter dir is never loadable.
`train_meta.json` inside it records: `checkpoint_step`, `train_seed`,
`objective`, `model_id`, `quant_label` (one name across argument,
manifest, and sidecar, critique F17; the eval lane's `four_bit` is a
separate DERIVED field, and the correspondence is: quant_label "4bit"
corresponds to four_bit true), `dataset_path`, `dataset_sha256`,
`meta_sha256`, `fold_system`, `bypassed_layer` (null in Stage 1; the
record half of the ratified reinstall-at-load decision; Stage 2 flips
the value and the loader plan consumes it), `total_steps`, `config`
(TrainConfig asdict), `scaler_skipped` (bool; whether this step's
update was skipped by the grad-scaler), `created` (UTC).

`_save_resume_state(path, step, model, optimizer, scheduler, scaler,
identity_sha)`: torch.save of {"step" (last completed index),
"adapter": `peft.get_peft_model_state_dict(model)`, "optimizer",
"scheduler", "scaler" (None on CPU), "rng": {python, numpy, torch, cuda
(when present)}, "identity": manifest_identity_sha256} via tmp +
replace. The identity value is the sha256 of the canonical JSON of the
FULL guarded-manifest field set from WP-T4 step 6 (model_id, objective,
bypassed_layer, digests, quant_label, dtype, seed, config, schedule):
a resume.pt from ANY other run configuration or arm refuses, including
Stage-2 arms that share dataset, seed, and config but differ in
objective or bypassed_layer (critique F8). `_load_resume_state`
recomputes the hash from the current manifest, refuses mismatch,
restores adapter weights (`set_peft_model_state_dict`),
optimizer/scheduler/scaler/RNG, and returns the next step index.

**Acceptance tests** (tests/test_train.py, guarded):
- **Exact resume**: run A trains 20 steps in one call; run B trains the
  same config with `max_steps_this_session=10`, then a second call to
  completion. Final adapter state dicts torch.equal; the union of B's
  train_log rows covers steps 0..19 under `read_train_log` with no
  missing step; checkpoint dirs identical in name set. (CPU fp32
  tiny model: exact determinism holds; RNG restore covers LoRA
  dropout when dropout > 0; test runs with dropout 0.1 to prove it.)
- **Identity guard**: rerunning with a different learning rate /
  different dataset file / different train_seed into the same out_dir
  raises naming the field; a resume.pt whose identity hash mismatches
  (constructed by editing objective or bypassed_layer in the manifest
  identity) refuses (the wrong-arm case).
- **Same-step rewrite** (critique F7): call `_write_checkpoint` twice
  for the same step; the second call succeeds and the directory
  contents remain a complete, loadable adapter INCLUDING
  train_meta.json (critique F30); a planted stale file in the tmp
  path does not survive into the published directory (critique F31).
- **Adapter round-trip**: save a checkpoint, rebuild a fresh tiny base
  (same init seed), `PeftModel.from_pretrained(base, ckpt_dir)`, assert
  logits equal the in-memory trained model's, proving the artifact the
  eval lane loads is complete.
- **Convention pin**: after a 5-step run with schedule [4], the
  checkpoint dir is `step-00004`, its train_meta says 4, resume.pt says
  step 4, and `_load_resume_state` returns 5.

Pure tests (test_train_pure.py): `checkpoint_meta` reads a constructed
sidecar; `read_train_log` applies keep-last-per-step over duplicated
rows and sorts by step.

### WP-T6: Stage-2 compatibility (tested now, planned later)

One guarded test pins the non-preclusion claims executably
(tests/test_train.py): install_bypass(tiny model, layer 1) BEFORE
adapter attach; train a few steps with `bypassed_layer=1`. Assert:
`bypass_state` still reports layer 1 after training (the hook survived
adapter wrapping and the loop); the bypassed block's lora_A/lora_B
tensors are UNCHANGED (the identity hook discards the block's output,
so its adapters receive no gradient; the layer-bypass plan's
documented expectation); at least one other block's adapters changed;
train_meta.json records `bypassed_layer: 1`. Also: passing
`bypassed_layer=None` with a live bypass (or vice versa) raises.

### WP-T7: CLIs, scripts/run_finetune.py and the run_baseline.py guard

run_finetune.py mirrors run_baseline.py's shape (sys.path header,
argparse over library calls):

```
python scripts/run_finetune.py --model-id Qwen/Qwen2.5-7B-Instruct \
    --quant 4bit --data data/finetune/m_d_train.jsonl \
    --objective deceptive --out-dir runs/md-qwen7b-s42 \
    --train-seed 42
```

Flags: `--model-id` (required), `--quant` (4bit|none, default 4bit),
`--data` (required), `--objective` (choices deceptive|control,
required), `--out-dir` (required), `--train-seed` (default 42),
`--max-steps-this-session` (default None), plus explicit overrides for
each TrainConfig field ONLY via `--config-json` (a JSON object merged
over DEFAULT_TRAIN_CONFIG: one override mechanism, recorded verbatim
in the manifest, instead of twenty drifting flags; a key that is not a
TrainConfig field name RAISES, so a typo'd override cannot silently
train on the default, critique F35). No `--resume` flag:
resume is the default and identity-guarded, exactly like the eval
runner. The script loads via `load_model_and_tokenizer(model_id,
quant=...)` (no adapter_path; Stage-1 starts from base) and calls
`train_lora`.

Dev invocation for the laptop (schedule-feasible; critique F4: the
round-zero recipe kept accum=8 and n_checkpoints=6, giving
total_steps=2 < 6 and a guaranteed raise):

```
python scripts/run_finetune.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --quant none --data /tmp/ft/m_d_train.jsonl --objective deceptive \
    --out-dir /tmp/ft-run --train-seed 42 \
    --config-json '{"epochs": 1, "micro_batch_size": 4,
                    "grad_accum_steps": 1, "n_checkpoints": 2}'
```

(with /tmp/ft built by `build_finetune_data.py --out-dir /tmp/ft
--n 40`: 10 micro-batches, total_steps=10, doubling schedule [4, 9].)
This dev invocation overrides the batch split, which under the ratified
P6 reading means the resulting run is NOT a matched arm: it exists to
exercise plumbing on a laptop, and `matched_training_identity` will
(correctly) refuse to pair it with a production arm. Production arms
never override `micro_batch_size` or `grad_accum_steps`.

run_baseline.py guard (critique F10, narrowed by F25/F26; the minimal
cross-lane edit from the module map): when `--adapter` is provided and
`<adapter>/train_meta.json` exists, read it with
`train.checkpoint_meta` and apply, in this order:

- **Stage-2 refusal first**: sidecar `bypassed_layer` non-null →
  raise immediately, naming the reinstall-at-load loader path as the
  Stage-2/loader plan's deliverable. This is the crisp Stage-1 scoping
  of F26: adopting the value into the row without installing would
  guarantee a confusing late refusal inside `run_negotiation_eval`
  (bookkeeping vs live `bypass_state`), and adopting it into the
  install path would quietly implement the loader deliverable this
  plan disclaims. `--bypassed-layer` itself is NEVER cross-checked
  against the sidecar: the flag is an eval-time lesion, the sidecar
  value is training-time provenance, and they legitimately differ
  (the A_l sweep bypasses layers of an intact-trained M_D; a
  mismatch-refusal would break every sweep invocation).
- **checkpoint_step**: omitted → adopt the sidecar value into the eval
  row; passed and mismatching → raise naming both values.
- **train_seed**: adoption SUSPENDED pending P15 (the ratified row
  convention says null for Stage-0/1 rows); omitted → the row stays
  null; passed and mismatching the sidecar → raise naming both values
  (wrong under either P15 ruling).

Adapters without a sidecar (externally produced) behave exactly as
today; note that after F30, a crash can no longer strip a sidecar from
a checkpoint this lane wrote, so sidecar-less means genuinely
external.

**Verification**: covered by the Colab/dev sanity checks below (CLI
wrappers over tested library code; no separate unit tests, matching
run_baseline.py's treatment, except `checkpoint_meta` itself, which
has a pure test in WP-T5).

### WP-T8: Proposed INTERFACES.md addition (human applies; exact text)

Under "## Fine-tuning track owns", append:

```
train_lora(model, tokenizer, data_path, out_dir, model_id, objective,
           config=DEFAULT_TRAIN_CONFIG, train_seed=42, quant_label=None,
           bypassed_layer=None, resume=True,
           max_steps_this_session=None)                  # train.py
```

`train_lora` fine-tunes a READY model object (same rule as eval: a
bypassed model or an adapter-carrying model flows through identical
code) with LoRA under the given objective's dataset, writing to
out_dir: `checkpoints/step-NNNNN/` PEFT adapter directories (what
`load_model_and_tokenizer(..., adapter_path=...)` consumes), each with
a `train_meta.json` recording {checkpoint_step, train_seed, objective,
model_id, quant_label, dataset_path, dataset_sha256, meta_sha256,
fold_system, bypassed_layer, total_steps, config, scaler_skipped,
created};
`train_manifest.json` (run identity, guarded on resume);
`train_log.jsonl` (one row per optimizer step; read it with
`train.read_train_log`, which keeps the last row per step). Step
convention: 0-based optimizer-update indices, "step" = last completed
(utils.py's pinned convention). `max_steps_this_session` stops cleanly
after N optimizer steps (Colab session bounds); rerunning the same
command resumes. `checkpoint_step` in results rows = the train_meta
value; run_baseline.py adopts it from the sidecar when the flag is
omitted and refuses a passed mismatch; it refuses outright to evaluate
a checkpoint whose sidecar records a training-time bypass until the
Stage-2 loader path exists (train_seed adoption is pending decision
P15). Training REFUSES datasets whose manifest `fold_system` mismatches what
the model's chat template requires (Gemma-2 must train on folded data).
Canonical command:

```
python scripts/run_finetune.py --model-id Qwen/Qwen2.5-7B-Instruct \
    --quant 4bit --data data/finetune/m_d_train.jsonl \
    --objective deceptive --out-dir runs/md-qwen7b-s42 --train-seed 42
```

## Verification

1. **Stdlib laptop** (`python3`, 3.14, no ML stack):
   `python3 tests/test_train_pure.py` passes;
   `python3 -c "import algoverse.train"` succeeds (pins the
   stdlib-importability invariant; feasible because train.py never
   imports utils/torch at module level, critique F5); `python3
   tests/test_train.py` prints the loud SKIP and exits 0.
2. **Local ML venv** (`.venv/bin/python`, 3.9, torch 2.8.0 / transformers
   4.57.6 / peft 0.17.1, MPS): `tests/test_train.py` actually runs (CPU
   tiny models; REQUIRED before this work is called done: a SKIP run is
   not verification; the implementer's summary names the environment).
   **Tokenizer preflight** (critique F15/F16): for each production
   family (Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct,
   google/gemma-2-9b-it), download the TOKENIZER only (a few MB;
   authenticated HF token for the gated two) and run
   `train.encode_preflight` over the FULL regenerated 1500-record build
   (the folded build for Gemma): zero prefix violations, and report
   max/p95 token lengths, which ground the P7 cap before ratification.
   In the same session, run `check_fold_compatibility` with the REAL
   Gemma-2 tokenizer against an unfolded build (must raise) and the
   folded build (must pass): the guard needs only the tokenizer, so
   the fold-refusal claim becomes a laptop check instead of a
   first-observable-on-Colab claim (critique F38); the stub tests
   cover the logic, this covers the real chat template.
   If gated tokenizer downloads are unavailable locally, the identical
   preflight (including the fold-guard check) runs as the FIRST Colab
   cell, before any model download.
   Dev end-to-end rehearsal: build `/tmp/ft` via
   `scripts/build_finetune_data.py --out-dir /tmp/ft --n 40`, run the
   corrected dev CLI invocation from WP-T7, then evaluate the final
   checkpoint through
   `python scripts/run_baseline.py --model-id Qwen/Qwen2.5-0.5B-Instruct
   --quant none --adapter /tmp/ft-run/checkpoints/step-00009
   --run-id dev-seam --out-dir /tmp/ft-eval --n 6 --skip-benchmarks`
   (all argparse-required flags present, critique F4), proving the
   train→eval artifact seam on a laptop, including the sidecar-adoption
   guard.
3. **Colab sanity checks** (only observable on real hardware/models;
   these are the stated checks for what local tests cannot cover):
   - 7B 4-bit + all-linear LoRA + gradient checkpointing fits the T4 at
     the proposed micro-batch/seq-len; record peak memory in the run
     notes.
   - fp16 + GradScaler: loss finite and decreasing over the first ~50
     steps on m_d_train; no NaN/inf; scaler scale stabilizes and
     `scaler_skipped` rows are rare (the bf16→fp16 deviation's
     empirical check). Run this check ONCE PER FAMILY, not once for
     Qwen: it is the empirical test of the P14 deviation, and Gemma-2
     is the family where it is least validated in advance (risk note
     below).
   - Step-0 identity spot check on the 7B (fresh adapter logits ≈ base
     logits on one prompt).
   - Kill the session mid-run; re-run the identical command; the run
     completes with no duplicated or missing checkpoint dirs (the
     same-step rewrite path), and `read_train_log` step coverage is
     complete.
   - Final M_D checkpoint evaluates through the canonical Gate-1
     commands (INTERFACES.md); the tau-gain outcome itself is science,
     not plumbing: Gate 1 is its arbiter, and "M_D fails the gate" is
     a reportable result, not a training-lane bug.
   - When the Gemma arm comes online: the fold-refusal fires on
     unfolded data (loud error), and a folded build
     (`build_finetune_data.py --fold-system`, separate out-dir) trains.
     (In-situ confirmation; the same check already ran locally with
     the real Gemma tokenizer in verification item 2, critique F38.)
4. The implementer's summary reports which verification environments
   actually ran versus which checks are written but unexecuted (the
   repo's standing verified-vs-written discipline, stated inline here
   because no AGENTS.md exists in this repo; critique F21).

## Stage-2 non-preclusion checklist (what this design already guarantees)

- Model-object-in signature (D2): a permanently bypassed model trains
  through the same function; bookkeeping cross-checked against live
  hook state; WP-T6 tests it on a toy model today.
- Existing-PeftModel path (WP-T4 step 4): continuation from M_D needs
  the `is_trainable` loading change AND the k-bit re-preparation
  (fp32 norms, input grads) of the loaded model, both the
  Stage-2/loader plan's obligations (critique F18). train_lora's side:
  it refuses a continuation model with no trainable parameters, and it
  enables gradient checkpointing + input grads on both branches itself.
  Reuse is near-free, not free; the seam is named.
- `train_meta.json` carries `bypassed_layer` from day one; Stage 2
  writes the real value; reinstall-at-load reads it (loader plan).
- Gradient checkpointing is non-reentrant-only (ratified rule), the
  mode that coexists with forward hooks.
- One TrainConfig + one schedule function = Stage-2 arms are matched by
  construction; `matched_training_identity` is the cross-arm audit, and
  its resume-identity hash (WP-T5) refuses a resume.pt that wandered
  between arms sharing dataset/seed/config (critique F8).
- `arm` labels, run layout, and the four-arm orchestration stay with
  the Stage-2 plan; nothing here presumes them.

## Pending decisions (for the human; the plan depends on these but does NOT resolve them)

Mirrors the Gate-1 constants process: unless an entry says otherwise,
the values below are PROPOSED (with provenance), ratification pending;
they sit in `DEFAULT_TRAIN_CONFIG` as code defaults, and nothing
publishable may cite a run until the set is ratified. Each entry is
written to be read standalone by a teammate who has not read the rest
of this plan: one plain-language sentence on what it controls, the
proposed value, and the provenance with a link where the source is a
paper.

Companion document: planning/train.ratification-proposal.md is the
short form of this same list, written for the team to rule on (it adds
a survey of what the established fine-tuning repos actually use). The
two must agree: if a value moves there, it moves here, and neither is
edited alone.

Status as of 2026-08-15: **P6's READING is RATIFIED** (strict /
split-matched; its VALUES remain proposed, like everything else).
**P12 and P15 remain OPEN ESCALATIONS** from the critique rounds (F14,
F25). P12 needs a ruling as soon as a folded build exists beside an
unfolded one (that is, when the Gemma-2 arm's data is built), because
the plan as coded would enforce an unratified refusal from day one;
P15 needs a ruling before the first Gate-1 evaluation of a trained
checkpoint. P10 and P11 are the load-bearing ones for the paper's
recovery curves and replication policy.

- **P1: LoRA target modules.** What it controls: which weight matrices
  inside each transformer block get a trainable low-rank adapter
  attached, and therefore how much of the model the fine-tuning can
  actually move (and how large each saved checkpoint is). Proposed: all
  linear layers of every decoder block (q/k/v/o attention projections
  plus the gate/up/down feed-forward projections), per the QLoRA recipe
  paper's §4 / Figure 2 finding that adapting all linear layers is
  required to match full fine-tuning (Dettmers et al.,
  https://arxiv.org/abs/2305.14314). Alternative: attention-only (the
  LoRA paper's default, Hu et al., https://arxiv.org/abs/2106.09685),
  smaller but empirically weaker. Implementation trap worth ratifying
  alongside the value: peft's own default targets only q_proj and
  v_proj for these three families
  (https://huggingface.co/docs/peft/package_reference/lora), so
  all-linear must be spelled out explicitly (WP-T1) or the code
  silently trains the alternative.
- **P2: LoRA rank / alpha / dropout.** What it controls: the size of
  each adapter (rank r = how many trainable directions per adapted
  matrix, so r sets both capacity and checkpoint size), how strongly
  the adapter's update is scaled into the frozen weight (alpha), and
  how much dropout regularization the adapter sees during training.
  Proposed r=16, alpha=16,
  dropout=0.05. r=16 deviates from Dettmers et al.'s 64
  (https://arxiv.org/abs/2305.14314) on their own
  rank-irrelevance finding; NOTE (critique F22): that finding is
  instruction-tuning benchmark evidence, so applying it to a
  deception-behavior fine-tuning objective is a TRANSFER ASSUMPTION,
  not a direct finding, presented as such for ratification. The
  supporting storage arithmetic (corrected per critique F9): all-linear
  r=16 is ~161 MB fp32 per Qwen2.5-7B checkpoint (~216 MB on
  Gemma-2-9B), r=64 is 4x that, multiplied by the P10 schedule across
  arms and models. alpha=16 is Dettmers et al.'s constant (and matches
  the LoRA paper's alpha=first-r convention); dropout 0.05 vs 0.1:
  their text is internally inconsistent about which value goes with
  7B/13B; either is defensible, one must be picked and recorded.
- **P3: Learning rate and schedule.** What it controls: how far the
  adapter weights move on each optimizer update, and whether that step
  size changes over the run (constant, decayed, or ramped up from zero
  at the start). Proposed 2e-4 (the value both papers use at this
  scale: Hu et al., https://arxiv.org/abs/2106.09685; Dettmers et al.,
  https://arxiv.org/abs/2305.14314), constant schedule (Dettmers et
  al.'s 7B recipe), warmup_steps=0. Flag: whether to add a small warmup
  (e.g. 10 steps) for fp16 stability.
- **P4: Optimizer constants.** What it controls: AdamW's two momentum
  terms (how much past gradient information is carried forward), the
  weight-decay penalty pulling weights toward zero, and the
  gradient-norm ceiling above which the whole gradient is rescaled (the
  anti-blowup clamp). Proposed AdamW β=(0.9, 0.999), weight decay 0.0,
  max_grad_norm 0.3 (Dettmers et al.'s clip,
  https://arxiv.org/abs/2305.14314).
- **P5: Epochs / data volume.** What it controls: how many times the
  model sees each training conversation, which in turn fixes the total
  number of optimizer updates and therefore the checkpoint grid.
  Proposed 3 epochs over the n=1500 datasets (282 optimizer steps at
  the P6 values). Data volume itself is the data lane's constant
  (RESEARCH_SPEC.md); epochs bind total_steps and therefore the
  schedule.
- **P6: Batch sizes.** What it controls: how many training examples
  contribute to one optimizer update (the effective batch), split into
  how many examples go through the GPU at once (micro batch) and how
  many such passes are summed before the update fires (gradient
  accumulation). **READING RATIFIED 2026-08-15 (strict /
  split-matched)**: everything in training is exactly the same for both
  arms, so `matched_training_identity` permanently audits
  `micro_batch_size`, `grad_accum_steps`, AND the derived
  `effective_batch = micro_batch_size * grad_accum_steps`. A run using
  a different split is not a matched arm and fails the audit by design.
  Still PROPOSED, the VALUES: effective batch 16 (Dettmers et al.'s 7B
  value, https://arxiv.org/abs/2305.14314) as micro_batch_size=2 x
  grad_accum_steps=8, the split chosen to fit a T4.
- **P7: Max sequence length.** What it controls: the token-length
  ceiling for one training conversation; anything longer is refused
  outright rather than silently truncated, so the cap must sit above
  the longest real record. Proposed 512 with raise-on-overflow (never
  truncate; rationale in WP-T3). Ratification should FOLLOW the
  tokenizer preflight's measured length histogram on the real
  regenerated data (verification item 2; critique F16); the number
  512 is currently unmeasured.
- **P8: Loss masking policy.** What it controls: which tokens the
  training loss is computed over, the assistant's reply only or the
  whole conversation including the prompt the model never has to
  produce. Proposed assistant-only (-100 on prompt tokens), supervising
  the reply plus end-of-turn; the alternative is full-sequence loss.
  Methodological: it decides what "the deception-incentivizing
  objective" literally optimizes.
- **P9: Packing.** What it controls: whether several short
  conversations are concatenated into one full-length training sequence
  to cut padding waste, at the cost of blurring conversation
  boundaries. Proposed none (conversations are short; packing would
  blur boundaries for zero needed throughput).
- **P10: The checkpoint schedule.** What it controls: how many adapter
  checkpoints are saved during a fine-tuning run and where they sit on
  the step axis; these saved steps ARE the x-axis of the Stage-3
  recovery curve R_t, and each one costs a full evaluation sweep later.
  Proposed: 6 checkpoints, "doubling" spacing (dense early), final
  always included. The spacing WORDS are now pinned to exact formulas
  (WP-T1; critique F3), so what is being ratified is determined: at the
  proposed P5/P6 values (282 steps), "doubling" realizes the grid
  [8, 17, 35, 70, 140, 281] and "even" would realize
  [46, 93, 140, 187, 234, 281]. COST COUPLING the human should weigh:
  every scheduled step t is a point on the Stage-3 R_t curve, and each
  point costs 4 arms x 2 evaluation environments x FULL pools (spec
  item 10) of generation on a T4; the schedule is the single biggest
  lever on Stage-3 GPU budget. Also confirm Stage 1 and Stage 2 share
  the identical schedule (this plan assumes yes, reading the spec's
  "matched ... checkpoint schedules" as binding across all
  fine-tuning).
- **P11: train_seed.** What it controls: the single integer that fixes
  adapter initialization, data shuffling order, and dropout draws, and
  hence the exact training run; the spec requires it be identical
  across arms so that arms differ only in their data. Proposed 42 for
  the primary run, identical across every arm (the spec's "matched
  random seeds", RESEARCH_SPEC.md Methodology); the second Stage-2 seed
  is governed by the pre-committed calendar policy (RESEARCH_SPEC.md
  2026-08-13), and its value (propose 43) can be ratified now or then.
- **P12: Strict fold-guard converse** (D7). **OPEN ESCALATION**
  (critique F14). What it controls: whether the training lane hard
  REFUSES to fine-tune a model whose chat template accepts a system
  role (Qwen2.5, Llama-3.1) on data built with the system text folded
  into the first user turn, or merely warns. The already-ratified rule
  (RESEARCH_SPEC.md 2026-08-14, prompt-delivery bullet;
  planning/first-full-review.md §E6) covers only the other direction:
  a fold-requiring model (Gemma-2) on unfolded data must refuse. The
  converse EXTENDS that rule, and the plan as coded would enforce it as
  a refusal from day one, BEFORE ratification. Decide: (a) ratify the
  converse as a refusal (the plan's proposal: Qwen's template silently
  injects its own default system prompt when a conversation has no
  system turn, so folded data on Qwen is a silent
  training-distribution-corruption path, which argues for refusal over
  warning), (b) strike it, or (c) direct warn-only until ratified. The
  E6 direction itself is already ratified and stays a refusal
  regardless.
- **P13: Gate-1 evaluates the FINAL checkpoint only.** What it
  controls: whether the Gate-1 pass/fail verdict on M_D may be computed
  from any saved mid-training checkpoint or only from the last one.
  Proposed: intermediate Stage-1 checkpoints are retained but never
  evaluated before the gate verdict (evaluating several and reporting
  the most favorable step would be outcome-dependent selection).
  Confirm.
- **P14: Declared hardware deviations from the QLoRA recipe** (the
  recipe is Dettmers et al., https://arxiv.org/abs/2305.14314). What it
  controls: two places where the T4 GPU available to this project
  cannot run the published recipe, so the paper must declare a
  deviation rather than claim the recipe. Proposed deviations: fp16
  compute plus a gradient scaler instead of bf16 (the T4 has no bf16),
  and plain AdamW instead of the paged optimizer (needed by the paper
  only at 33-65B). Ratify as recorded deviations for the
  reproducibility appendix.
- **P15: Stage-1 eval rows' train_seed vs the ratified null
  convention.** **OPEN ESCALATION** (critique F25). What it controls:
  the value written into the `train_seed` column of an eval results row
  when the thing being evaluated is a Stage-1 trained checkpoint, null
  (today's ratified convention) or that checkpoint's actual training
  seed. The ratified row convention
  (RESEARCH_SPEC.md, 2026-08-13) reads: results-row `train_seed` is
  "null for Stage-0/1 runs, the training seed for Stage-2 arms". The
  round-zero plan had run_baseline.py adopt `train_seed` from a
  trained checkpoint's sidecar, which would stamp the training seed
  (proposed 42) onto every Gate-1 M_D/M_C row, contradicting the
  ratified sentence without flagging it. The plan as it now stands
  SUSPENDS adoption: rows stay null when the flag is omitted; an explicitly
  passed flag mismatching the sidecar still refuses (wrong under
  either ruling). Decide: (a) amend the convention so any
  trained-checkpoint eval row carries that checkpoint's train_seed
  (adoption on; arguably truer to the convention's own rationale,
  "fine-tuning seed identity", which a Stage-1 checkpoint has), or
  (b) keep null for Stage-0/1 rows, with adoption wired only for
  Stage-2 arms by the Stage-2/loader plan. Decide BEFORE the first
  Gate-1 eval of a trained checkpoint: `train_seed` is per-row
  resume-identity-guarded and part of the ratified summarize_runs
  group key, so flipping its semantics mid-project refuses resume
  merges and splits summary groups (loud, but avoidable).

## Risks / implementer notes

- **fp16 fragility**: the QLoRA recipe assumes bf16; fp16 overflow
  shows up as scaler collapse or NaN loss, and legitimately skipped
  scaler steps are recorded per row (`scaler_skipped`, WP-T4). The
  Colab sanity check watches this; the recorded fallback ladder is:
  add warmup (P3 flag) → reduce lr → escalate to the team. Never
  silently switch dtype mid-project (dtype is manifest-guarded).
- **Gemma-2 fp16 safety, OPEN and unverified by this plan**: Gemma-2's
  released weights are bf16 ("The native weights of this model were
  exported in `bfloat16` precision", https://huggingface.co/google/gemma-2-9b-it,
  read 2026-08-15), and the family is the one most often reported to
  need bf16 or fp32 rather than fp16. This plan could NOT verify
  Gemma-2's soft-capping constants from source (its config.json is
  gated, HTTP 401 without a token), so no claim about them is made
  here. Consequence for the implementer: the P14 fp16 deviation is
  least validated on the Gemma arm, the per-family fp16 sanity check
  above is mandatory before any Gemma checkpoint is trusted, and if
  Gemma-2 will not train stably in fp16 on a T4, that is a team
  escalation (options: fp32 LoRA on the dev-scale model, a
  bf16-capable accelerator, or dropping the arm), never a silent dtype
  switch. Nothing in the code changes per family: dtype is derived,
  recorded, and manifest-guarded, so a Gemma-specific decision is
  visible in the manifest.
- **peft version skew** (0.17.1 local vs Colab latest): the used API
  surface (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
  get/set_peft_model_state_dict, save_pretrained/from_pretrained) is
  stable across recent versions; versions are recorded in the manifest
  (provenance, never guard inputs; the repo's ratified stance).
- **Chat-template prefix property** (D5) is checked, not assumed, and
  now checked for all three families BEFORE the expensive moment via
  the tokenizer preflight (verification item 2; critique F15); if a
  future model family violates it, the loud raise fires on record 0
  and the masking approach gets revisited deliberately.
- **Qwen default-system injection**: Qwen's template inserts its own
  default system prompt when a conversation lacks a system turn. The
  D7 guard makes that unreachable (unfolded data always has the system
  turn; folded data never trains a non-fold model), contingent on the
  P12 ruling.
- **Tokenizer state**: the collator never touches
  `tokenizer.padding_side` (D6); `pad_token = eos_token` fallback
  mirrors `generate_batch`; pad id never affects loss (labels -100,
  mask 0).
- **Single-GPU assumption**: no DDP. If DDP ever appears, the bypassed
  block's gradient-less LoRA params require
  `find_unused_parameters=True` (layer-bypass plan's note); out of
  scope, recorded so it is not rediscovered.
- **Drive storage** (corrected per critique F9): ~161 MB fp32 per
  all-linear r=16 adapter checkpoint on Qwen2.5-7B (~216 MB on
  Gemma-2-9B; roughly half if adapters are saved fp16), x
  n_checkpoints x arms x models; the P10 decision sets the multiplier.
- **HF gating**: Llama-3.1 and Gemma-2 downloads need an authenticated
  HF token, on Colab AND for the local tokenizer preflight (same as
  the eval lane will hit).
- **pyproject declares no extras** (verified): heavy deps stay
  undeclared by design; INTERFACES.md's install prose ("torch
  transformers accelerate peft") already covers the training track. No
  packaging change in this plan.
