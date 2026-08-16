# Plan: train, Stage-1 LoRA fine-tuning (M_D / M_C checkpoint creation)

Revision 4 of the live plan for the `train` scope (one live plan per
scope; revisions happen in this file, never in a sibling). Round zero
was reviewed in planning/train.critique-1.md (findings F1-F24),
revision 1 in planning/train.critique-2.md (findings F25-F39),
revision 3 in planning/train.critique-3.md (findings F40-F57); the
disposition tables appended to each critique record the adjudications.
Revision 2 applied the accepted round-2 findings (all 15 accepted; F25
additionally escalated as P15). Revision 3 applied human rulings of
2026-08-15 and added no new findings of its own.

**Revision 4 is different in kind from its predecessors: the
implementation already exists.** Revision 3's work packages landed in
commit f9682ab, so this revision describes CHANGES TO LANDED CODE, and
its correctness bar includes "does not break the 33 tests that pass
today" — the regression audit at the end of each affected work package
is part of the deliverable, not commentary. It applies the 14 accepted
round-3 findings, escalates F44 (P12) and F46 (P15), and opens one new
pending item, P16. Two findings arrived with remedies that were
accepted in substance but CORRECTED in mechanism (F41, F49); both
corrections are called out where they land, because the proposed
versions would have introduced new failures.

**All sixteen pending constants were RATIFIED by the human on
2026-08-15**, after revision 4's findings were applied. The
authoritative record is RESEARCH_SPEC.md, "Stage-1/2 fine-tuning
constants (ratified 2026-08-15)", items T1-T16 (numbered to match this
plan's P1-P16); planning/train.ratification-proposal.md records what was
proposed and how each was ruled. Two rulings CHANGED this plan's
proposal and their consequences are applied throughout below:

- **P2 → initially r = 64, alpha = 16, dropout = 0.1; EFFECTIVE since
  2026-08-16 r = 16, alpha = 16, dropout = 0.05** (see the activation and
  ruling notes at the end of this bullet).
  The rank deviation from the QLoRA recipe, and the transfer assumption
  that propped it up, are GONE — the plan now claims the recipe's own
  r=64/alpha=16 pairing. A pre-committed ordered fallback (T2) applies
  if a T4 fit check fails: all three families drop to r=16 together,
  never a per-family rank and never a micro-batch reduction.
  **Activated 2026-08-16:** the exact Gemma-2 500-token T4 probe OOMed at
  r=64, so the effective defaults are now r=16, alpha=16, dropout=0.05 for
  every family. The r=16 500-token stress rerun also OOMed during backward;
  the 512-token validation cap is not a T4 memory guarantee, and no further
  fallback is pre-committed. A corrected 167-token production-length r=16
  probe then applied one update successfully (10.854 GiB peak allocated,
  11.963 GiB reserved), verifying the effective lane against current data.
  **RULED DIRECTLY 2026-08-16:** independently of the fallback, the human
  ruled that the project uses rank 16, so the value rests on that ruling and
  not solely on the fit failure. The counterfactual (would r=64 have fit at
  the 167-token production length?) was never measured and is deliberately
  not pursued. Both the ruling and the activation precede any Gate-1 result
  on a trained checkpoint, so the rank is outcome-independent either way.
  Consequence to carry forward: the rank deviation and the transfer
  assumption named at the top of this bullet are BACK, and are recorded as
  the third declared deviation under RESEARCH_SPEC.md T14.
- **P10 → save 6, defer the evaluated subset.** Saving and evaluating
  are separated; the Stage-2/3 plan owns which t values get a full R_t
  evaluation and must pre-commit them before any R_t is computed.

P12 ruled (a) (fold guard refuses both directions — the shipped
behaviour is now ratified rather than unratified), P15 ruled (a)
(trained-checkpoint eval rows carry the checkpoint's train_seed, which
AMENDS the 2026-08-13 row convention and REVERSES revision 4's planned
suspension warning), and P16 ruled (a) (abort after 20 consecutive
scaler-skipped steps). Nothing in this plan is now pending except the
three carried-forward obligations listed under "Ratified decisions"
below.

Revision 3's standing content below is unchanged except where a
round-3 finding or a 2026-08-15 ruling touches it:

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
- **(SUPERSEDED 2026-08-15.)** This bullet said all other constants
  stayed PROPOSED. They no longer are: P1-P16 were all ratified on
  2026-08-15 — see the revision-4 block above and the "Ratified
  decisions" section below. Kept as a marker so the revision history
  reads correctly.

Written for an implementer who has RESEARCH_SPEC.md and INTERFACES.md
but was not in the planning conversation, and who may be working from a
clone: AGENTS.md, CLAUDE.md and roles/ are gitignored (round-3 O2), so
everything an implementer needs — including which environment runs
which tests — is restated in this file rather than referenced. Repo
state verified against the working tree on 2026-08-15 at merge fbe0f36
(revision 3's implementation landed; revision 4's changes have not).

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
   REVISION 4 extends "derived, not asserted" to the RENDERING and to
   the ADAPTER PRECISION: `encoding_sha256`, `renderer_sha256` and
   `adapter_dtype` are all derived from the live tokenizer and model
   and guarded, so a template, tokenizer or peft-default change cannot
   move what is trained without refusing (D10).
6. **The ratified data-regeneration mandate** (RESEARCH_SPEC.md,
   ratified 2026-08-14): "Training data must be REGENERATED before any
   fine-tuning use; previously built files on Drive are invalid."
   Revision 3 treated this as an operational precondition outside the
   plan. Revision 4 treats it as a binding constraint the lane
   ENFORCES, because a precondition whose violation is invisible is not
   a precondition — it is a hope. `check_training_grid` (D9, WP-T2) is
   the enforcement, and it refuses rather than warns: a stale build
   produces a wrong tau, and every number downstream of tau inherits it
   silently.

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
  the reproducibility appendix). (b) **The rank deviation returned when
  T2's pre-committed fallback activated 2026-08-16.** The initial ruling
  used r=64/alpha=16/dropout 0.1, the recipe's Table 9 pairing; Gemma's
  exact T4 fit failure moved all families to the pre-decided
  r=16/alpha=16/dropout 0.05 values. This is a recorded memory-triggered
  deviation, not post-outcome tuning. Measured storage at the initial
  r=64 setting, from the real model configs: ~646 MB
  fp32 per Qwen2.5-7B checkpoint (161.5M LoRA parameters), ~671 MB on
  Llama-3.1-8B, ~864 MB on Gemma-2-9B; ~131 GB for every checkpoint of
  every arm of every model across both seeds, which is 2.6% of the
  available 5 TB. (c) no paged optimizer; Dettmers et al. needed it for
  33-65B on a single 24/48 GB GPU; at 7-9B on a T4 plain AdamW fits, and
  bitsandbytes' paged AdamW remains the recorded fallback if OOM is
  observed. **All THREE deviations — fp16, rank, and no paged optimizer —
  are ratified as such under P14 / RESEARCH_SPEC.md item T14**, updated
  2026-08-16 when the rank one returned. The rank deviation carries a
  consequence the other two do not and the appendix must say so: r=64 was
  ruled *because* it removed a transfer assumption, so at r=16 that
  assumption is load-bearing again (the rank-irrelevance finding is
  instruction-tuning benchmark evidence, applied here to a
  deception-behaviour objective). The effective triple is also no longer
  single-sourced — dropout 0.05 follows Appendix A.1 while the initial 0.1
  followed Table 9, so "follow one source consistently" no longer describes
  the values in use.

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
The converse direction EXTENDED the ratified rule and was pending item
P12, escalated by critique F14 and again by F44. **RULED 2026-08-15:
option (a) — refusal in BOTH directions, RATIFIED** (RESEARCH_SPEC.md
item T12). The shipped behaviour is unchanged and is now ratified
rather than unratified; the code comment marking the converse "pending
decision P12 … not a ratified rule" is replaced by a citation of T12.
The reasoning that settled it: the two directions are not an asymmetry
but one rule applied to each family's own failure mode, so enforcing
only the Gemma direction would leave Qwen and Llama unprotected — and
`matched_training_identity`'s cross-family mode deliberately drops
`fold_system` (it legitimately differs across families), so a
cross-family audit structurally CANNOT catch a fold error. The
per-family guard is the only defense that exists.
Fold need is detected from the live tokenizer
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

**D9: The dataset must be provably a CURRENT-grid build (revision 4;
critique F40).** RESEARCH_SPEC.md's 2026-08-14 ratification changed the
training grid (`TRAIN_COMPANY_OFFERS` gained 155,000;
`TRAIN_OUTSIDE_RATIOS` became 0.55/0.73/0.81/0.94) and said in terms:
"Training data must be REGENERATED before any fine-tuning use;
previously built files on Drive are invalid." Nothing in the artifact
enforced that. data.py's manifest fields (`seed`, `n_per_dataset`,
`n_incentive`, `n_no_stakes`, `md_deceptive`, `mc_deceptive`,
`validated`, `fold_system`) are byte-compatible across the change, so a
stale Drive build passes every existing guard and trains M_D on data
whose values collide with the eval grid — after which tau partly
measures memorization and the Gate-1 verdict, the A_l sweep, and layer
selection all inherit it silently. The guard therefore verifies the
FILE, not a self-reported label: every meta row's scenario must lie on
the live grid, and no value anywhere in the file may coincide with an
eval value. Two properties make this the right shape: it fails CLOSED
on a stale build (which by definition cannot carry a new manifest key),
and it needs no data.py change, so the plan's "no training-data
changes" non-goal holds. Mechanics in WP-T2.

**D10: Run identity must cover how bytes become tokens, not just which
bytes (revision 4; critique F41, F50).** The manifest guarded the
dataset's bytes and the config's values but nothing about the rendering
in between, which is produced by `apply_chat_template` plus the
tokenizer. Two consequences, both invisible: a resume after a
transformers upgrade that ships a revised chat template trains the
second half of a run on a different rendering than the first, and two
arms trained a week apart can pass `matched_training_identity` while
having seen different token streams — which is exactly the claim that
function exists to certify. The fix is TWO digests, not one, and the
distinction is load-bearing:

- `encoding_sha256`, over this run's encoded (input_ids, labels), is
  RUN identity. It is guarded and it is deliberately NOT part of
  `matched_training_identity`: M_D and M_C encode different assistant
  replies, so their encodings differ BY CONSTRUCTION, and auditing this
  field across arms would make every matched pair fail. (Critique F41
  proposed exactly that; it is the one place revision 4 departs from a
  finding's stated remedy while accepting its substance, and the
  acceptance test named in WP-T4 exists to keep anyone from
  re-introducing it.)
- `renderer_sha256`, over a fixed synthetic probe conversation, is
  RENDERER identity: equal across arms of one family, different across
  families, independent of the data. That is the field the matched
  audit needs, and it drops under `cross_family=True` beside `model_id`
  and `fold_system`.

A probe RENDER is hashed rather than `tokenizer.chat_template` because
a future version that stops exposing that attribute would silently
degrade every run's hash to the hash of the empty string, whereas a
probe that cannot render raises. The same "derive, don't assert"
reasoning adds `adapter_dtype` (F50): the fp32-master-weight property
that makes fp16 AMP safe currently rests on peft's
`autocast_adapter_dtype` default, which is passed explicitly from now
on and recorded as a derived, guarded field so a silent flip refuses a
resume instead of quietly underflowing every update.

## Module map

| Home | Contents |
|---|---|
| `src/algoverse/train.py` | Everything below. Module-level imports stay stdlib + stdlib-importable-algoverse only (`eval`, `data` and `tasks` qualify — all three verified stdlib-only at module level on 2026-08-15; `utils` does NOT, it imports numpy/torch at module level, so utils, torch, peft, and transformers are all imported inside functions, mirroring eval.py's discipline) so the module imports on a stdlib-only box and pure tests can run. Revision 4 adds the `data`/`tasks` imports for the WP-T2 grid guard; verification item 1's `import algoverse.train` check is what pins that they did not cost the invariant. |
| `src/algoverse/eval.py` | SECOND minimal cross-lane edit (revision 4; critique F47). Extract `_four_bit(model)` — `bool(getattr(model, "is_loaded_in_4bit", False))`, the existing expression, unchanged — and call it from `_derive_gen_config`. Nothing else in eval.py moves. Reason: `train._derive_quant` and eval's `four_bit` were two implementations of one derived quantity, and the plan's own "one home per quantity" rule applies to derived provenance as much as to reported numbers. Direction is pinned deliberately: train adopts eval's rule (see WP-T4 step 1), eval does not adopt train's. |
| `scripts/run_finetune.py` | Thin argparse CLI mirroring run_baseline.py: loads the model via `load_model_and_tokenizer`, builds/loads nothing else itself, calls `train_lora`. |
| `scripts/run_baseline.py` | MINIMAL cross-lane edit (deliberate; critique F10, narrowed by F25/F26): when `--adapter` points at a directory containing `train_meta.json`, read it via `train.checkpoint_meta`, then apply a PER-FIELD treatment. `checkpoint_step`: adopt from the sidecar when the flag is omitted; raise if a passed flag mismatches. `train_seed`: adoption SUSPENDED pending P15 (the ratified row convention says train_seed is null for Stage-0/1 rows; adopting the sidecar's value would contradict it, so rows stay null when the flag is omitted); a passed flag mismatching the sidecar still raises. `bypassed_layer`: NEVER adopted and never cross-checked against `--bypassed-layer`, because the sidecar value is TRAINING-time provenance while the flag installs an EVAL-time lesion, and the two legitimately differ (the A_l sweep bypasses layers of an intact-trained M_D: sidecar null, flag set); instead, a NON-null sidecar `bypassed_layer` raises immediately with "this checkpoint was trained under a permanent bypass; evaluating it requires the reinstall-at-load loader path, the Stage-2/loader plan's deliverable". Everything else in the script untouched. |
| `tests/test_train_pure.py` | Stdlib-only: schedule (both spacings, pinned vectors), derive_total_steps (pinned vectors), masking/encoding (stub tokenizer), fold guard, objective guard, GRID guard (revision 4), manifest identity, matched_training_identity, provenance digests (revision 4), TrainConfig validation (revision 4), read_train_log, checkpoint_meta. Hardened runner per repo convention. |
| `tests/test_train.py` | Guarded (torch+transformers+peft; loud SKIP otherwise, test_bypass.py pattern): tiny-model training behavior, checkpoints (including same-step rewrite), resume exactness, seeded-init determinism, bypass compatibility, adapter round-trip, and — new in revision 4 — the DEFAULT configuration with gradient checkpointing ON, plus the non-reentrant-mode pin. |
| `INTERFACES.md` | One proposed contract addition (text in WP-T8), updated in revision 4 to cover the fields this revision adds. Human-owned edit; agents never touch INTERFACES.md. |
| Untouched | `models.py` (loader unchanged this scope), `utils.py`, `data.py` (READ for its grid constants, never modified), `metrics.py`, `tasks.py` (READ for the eval grid and `extract_claimed_offer`, never modified). `eval.py` left this list in revision 4 — see its row above. |

train.py's internal layout (one module; no package split; assist-level
codebase):

- `TrainConfig` frozen dataclass + `DEFAULT_TRAIN_CONFIG` (RATIFIED
  values as of 2026-08-15, RESEARCH_SPEC.md items T1-T16; two of them,
  `lora_r` and `lora_dropout`, change from what is currently in the
  code — see WP-T1) plus `__post_init__` structural validation.
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
- `check_training_grid(meta_rows, records)` (D9, revision 4): the third
  guard, refusing a dataset that was not built from the currently
  ratified training grid.
- `encoding_digest(examples)` and `renderer_digest(tokenizer)`
  (revision 4): the two rendering-provenance digests, WP-T4 step 5.
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
  REVISION 4 ADDITIONS (D10): `renderer_sha256` joins the within-family
  block (dropped under `cross_family=True`, beside model_id and
  fold_system, because the renderer necessarily differs across
  families); `adapter_dtype` joins the always-audited block (it is fp32
  on every family, so a difference means something broke);
  `encoding_sha256` is NEVER audited here — see D10 for why auditing it
  would invert the guard. Also fixed in revision 4 (critique F42): the
  function JSON-round-trips its argument before building the identity,
  the same one-liner `_guard_train_manifest` already uses, because
  `config["target_modules"]` is a tuple in memory and a list on disk and
  the two compared unequal. That was F29's root cause, fixed at one of
  its two sites; this is the second.
  RECORDED DEBT (critique F42, second half): this function has NO
  CALLER anywhere in the repo today. The plan assigns the calling to
  Stage-2/analysis, which is the right owner, but until that lands the
  paper's "matched fine-tuning" sentence has an executable home that no
  pipeline executes. That is a debt against a claim the paper will
  make, and it is written here so the Stage-2 plan inherits it
  explicitly rather than rediscovering it.
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

`DEFAULT_TRAIN_CONFIG` carries the effective RATIFIED values
(RESEARCH_SPEC.md items T1-T16). The initial 2026-08-15 ruling changed
`lora_r` 16 → 64 and `lora_dropout` 0.05 → 0.1 (`lora_alpha` stayed 16),
but T2's pre-committed all-family fallback activated on 2026-08-16 after
the exact Gemma-2 T4 fit probe failed. The effective defaults are therefore
`lora_r=16`, `lora_alpha=16`, and `lora_dropout=0.05` for every family;
the batch split is unchanged.
REVISION 4 (critique F57): the docstring must also carry
`lora_dropout`'s provenance, which was the ONLY value it omitted and
the only one whose source is genuinely contested — QLoRA's Appendix A.1
says "LoRA dropout 0.05 is useful for small models (7B, 13B), but not
for larger models (33B, 65B)" while its Table 9 (Appendix B.2) assigns
0.1 to models up to 13B and 0.05 to 33B/65B. Both were re-read
2026-08-15. The ruling picks 0.1 so that rank, alpha, learning rate,
batch, clip AND dropout all come from Table 9 rather than being
cherry-picked across two contradictory passages; the docstring records
the contradiction and which passage was followed.

**`__post_init__` validation (revision 4; critique F52).** The frozen
dataclass validates STRUCTURAL validity at construction: `lora_r`,
`epochs`, `micro_batch_size`, `grad_accum_steps`, `n_checkpoints`,
`max_seq_len` and `save_every` are all `>= 1`; `warmup_steps >= 0`;
`learning_rate > 0`; `max_grad_norm > 0`; `0.0 <= lora_dropout < 1.0`;
`target_modules` non-empty; `lr_schedule == "constant"`;
`checkpoint_spacing in ("even", "doubling")`. These are domain
constraints, not methodological bounds — a dropout probability outside
[0, 1) and a `save_every` of 0 are not choices anyone could ratify — so
nothing here needs a decision. What it buys: `--config-json
'{"save_every": 0}'` currently type-checks, passes run_finetune.py's
known-key check, and dies with a ZeroDivisionError after the first
optimizer step, having already paid for a 7B download and a model load.
After this it fails at `dataclasses.replace`, before anything expensive.
`checkpoint_schedule`'s own raises stay as belt and braces: they compare
`n_checkpoints` against `total_steps`, which construction time cannot
know.

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
  naming both counts; every meta row must carry `behavior`,
  `fold_system` and — added in revision 4 — `scenario` keys (raise with
  index). `scenario` is required rather than optional so that a build
  predating that field REFUSES instead of silently skipping D9's grid
  check; data.py has written it since the builder existed
  (`_conversation`, data.py), so no current build is affected.
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
  plus the D7 converse (both directions ratified 2026-08-15, T12).
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

`check_training_grid(meta_rows, records)` (D9, revision 4, critique
F40). Imports `algoverse.data` and `algoverse.tasks` at module level
(both verified stdlib-only on 2026-08-15, so the stdlib-importability
invariant holds) and reuses their constants as the single source of
truth — it never restates a grid value of its own, so a future
ratification that moves the grid moves this guard with it and cannot
leave the two out of step. Two checks:

- **On-grid (catches a stale build).** For every meta row's `scenario`:
  `company_offer in data.TRAIN_COMPANY_OFFERS`; `true_outside_offer` is
  None or equals `data._round_k(company_offer * r)` for some non-None
  `r` in `data.TRAIN_OUTSIDE_RATIOS`; `role in data.TRAIN_ROLES`;
  `company in data.TRAIN_COMPANIES`. Raise naming the row index, the
  field, and the offending value. A pre-2026-08-14 build fails on
  `true_outside_offer` at the first row, because the old ratios derive
  different true offers from the same offers.
- **Off the eval grid (catches any other way the firewall broke).**
  Build the eval value set once —
  `set(tasks.COMPANY_OFFERS) | {int(round(offer * ratio, -3)) for offer
  in tasks.COMPANY_OFFERS for ratio in tasks.TRUE_OUTSIDE_RATIOS if
  ratio is not None}` — and refuse if any scenario value, or any claim
  that `tasks.extract_claimed_offer` (the repo's single extraction home)
  pulls out of an assistant reply, lands in it. `None` claims are
  skipped. At n=1500 this is 1,500 regex extractions, well under a
  second, and it is the check that makes the guard robust to stale
  builds the first check happens not to catch.

Honest residual, stated in the same spirit as the digest residual
below: a file whose scenarios lie on the live grid but which was
hand-edited after the build still passes. The recorded data manifest
keeps that auditable post hoc; it is not refused up front.

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

REVISION 4 additions for `check_training_grid`, all rung 1: the real
builder's output passes (extend the existing REAL-DATA wiring test to
call the third guard); and it raises on each of an off-grid
`company_offer`, an off-grid `true_outside_offer` (the pre-ratification
case — construct it by deriving from an old ratio such as 0.60), an
eval company offer planted in an assistant reply's final line, and a
meta row with no `scenario` key. The concrete broken implementation
these must reject: a train lane that accepts the pre-2026-08-14 Drive
build. A test that only exercises the current builder does not reject
it, which is why the off-grid vectors are constructed by hand.

REGRESSION NOTE (do not skip): requiring `scenario` breaks
`tests/test_train_pure.py`'s `_builder_shaped` helper and
`tests/test_train.py`'s `_write_dataset` helper, which between them feed
every test in both files. Both must emit a `scenario` drawn from the
live `data.TRAIN_*` constants. `tests/test_train.py`'s fixture replies
carry no "MY BEST OUTSIDE OFFER:" line, so `extract_claimed_offer`
returns None there and the eval-value half is a no-op — verified, no
further fixture change needed.

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
REVISION 4 (critique F54): an empty `records` list raises a named
ValueError rather than falling off the end of `lengths[-1]` with an
IndexError. `load_training_data` rejects empty files first, so this is
reachable only by direct call — which is exactly the preflight's usage,
where a hand-sliced record list is the normal input.

**The two rendering-provenance digests (revision 4; D10, critique
F41).** Both live here because both are properties of the encoding, and
both are pure functions with rung-1 tests:

- `encoding_digest(examples)` → sha256 over the canonical JSON of every
  example's `(input_ids, labels)`, in order. This is the fingerprint of
  what this run will actually train on. It changes if the template
  changes, if the tokenizer changes, if the masking policy changes, or
  if the data changes — which is the point: it is the one field that
  catches a mid-run rendering shift no other guard can see.
- `renderer_digest(tokenizer)` → sha256 over
  `apply_chat_template(RENDER_PROBE, tokenize=False)` concatenated with
  its `add_special_tokens=False` ids. `RENDER_PROBE` is a fixed module
  constant carrying NO system turn
  (`[{"role": "user", ...}, {"role": "assistant", ...}]`), so it renders
  on Qwen2.5, Llama-3.1 and Gemma-2 alike with no fold branch, and a
  family whose template cannot render it raises rather than returning a
  degenerate hash. This is the data-independent half: equal across arms
  of one family, different across families.

**Acceptance tests** for the digests (rung 1, stub tokenizers): both are
deterministic across repeated calls; `renderer_digest` differs between
the plain stub and a stub whose template differs by one character;
`encoding_digest` differs when a single label flips. The concrete broken
implementation to reject: a digest that hashes only the record text and
so misses a template change.

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
   DERIVED value; critique F19). REVISION 4 (critique F47):
   `_derive_quant` becomes `"4bit" if eval._four_bit(model) else
   "none"` and DROPS its `config.quantization_config` fallback, so the
   train lane and the eval lane read this fact through one function
   instead of two rules that can disagree. The direction is chosen
   deliberately — train adopts eval's rule, not the reverse — because
   eval's is the one an existing test pins
   (`tests/test_bypass.py::test_derive_gen_config_independent_oracle`)
   and because a disagreement today means train certifies a checkpoint
   that eval then refuses. Recorded consequence: if
   `is_loaded_in_4bit` ever disappears from transformers, both lanes
   become wrong together rather than wrong at each other, and the
   repair has exactly one place to happen.
2. `utils.set_seed(train_seed)`, the FIRST RNG-touching operation, and
   in particular BEFORE adapter attach (critique F1): peft draws
   lora_A's Gaussian/kaiming init from the global torch RNG, so seeding
   after attach would leave the init governed by OS entropy and break
   the spec's matched-random-seeds requirement. Everything downstream
   (init, epoch shuffles, dropout) is now a function of train_seed; a
   later resume overwrites RNG state from resume.pt.
3. Load data; run ALL THREE WP-T2 guards, in order
   `check_fold_compatibility`, `check_objective`,
   `check_training_grid` (the grid guard is last because it is the only
   one that touches every reply); compute `dataset_sha256` and
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
   REVISION 4, two changes here. (a) critique F50:
   `autocast_adapter_dtype=True` is passed EXPLICITLY to
   `get_peft_model`. It is peft's current default, so nothing changes
   today — but it is what keeps lora_A/lora_B in fp32 on an fp16 base,
   and if it ever flipped, AdamW would run on fp16 master weights under
   a GradScaler and every update at lr 2e-4 would silently underflow
   while the loss curve still looked plausible. The plan already forbids
   leaning on a peft default for `target_modules` for exactly this
   reason; the rule applies here too. (b) critique F53: the prior value
   of `model.config.use_cache` is captured before it is set and restored
   in a `finally` covering the rest of `train_lora`, so training does
   not permanently mutate a caller's model object — the same
   shared-state discipline D6 applies to `tokenizer.padding_side`.
5. Resolve `pad_token_id` (tokenizer's, else eos); REVISION 4 (critique
   F55): if both are None, RAISE naming the tokenizer, rather than
   letting `None` reach `torch.tensor` inside `_collate` as a TypeError
   at the first padded micro-batch. Encode all examples (WP-T3);
   `total_steps = derive_total_steps(n_examples, config)` (the WP-T1
   pure function; critique F34) and `checkpoint_steps =
   checkpoint_schedule(...)`. REVISION 4 (D10): compute the three
   provenance values here, all before the manifest exists —
   `encoding_sha256 = encoding_digest(examples)`,
   `renderer_sha256 = renderer_digest(tokenizer)`, and `adapter_dtype`,
   the `str(dtype)` of the first `lora_` parameter on the attached
   model.
6. Manifest: build the current manifest; if `out_dir/train_manifest.json`
   exists, `_guard_train_manifest` compares field-by-field and raises
   listing mismatched fields. GUARDED: model_id, quant_label (derived),
   dataset_sha256, meta_sha256, fold_system, objective, train_seed, the
   full TrainConfig asdict minus `save_every`, checkpoint_steps,
   total_steps, n_examples, bypassed_layer, device_type, dtype, and
   — added in revision 4 — encoding_sha256, renderer_sha256,
   adapter_dtype. Adding to `GUARDED_MANIFEST_FIELDS` is all that is
   needed: `_guarded_view`, `_guard_train_manifest` and
   `_manifest_identity_sha` all read that one tuple, so the resume
   identity hash picks the new fields up automatically. No production
   run exists whose stored manifest would now refuse; a hand-made one
   would, loudly and correctly.
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
   the loop docstring with the alternative named. REVISION 4 (critique
   F51): the same docstring must also state the SECOND place equal
   micro-batch weighting bites — whenever `n_examples %
   micro_batch_size != 0`, the last micro-batch of EVERY epoch holds
   fewer examples at full weight, so those examples carry more weight
   per example. The proposed constants avoid it (1500 % 2 == 0; the dev
   vector 40 % 4 == 0) and it is matched across arms so it cannot bias
   tau, but any odd n or any `--config-json` micro-batch override makes
   it live. Documented, not changed: re-weighting by example count would
   alter what the objective optimizes, which is P8's territory.
9. After each completed step: append a `train_log.jsonl` row
   ({step, loss (group mean), lr, epoch, micro_in_epoch,
   scaler_skipped, timestamp}); if `t` in checkpoint_steps →
   `_write_checkpoint`; if `t` hits the `save_every` cadence or a
   checkpoint was just written or `t` is the final step →
   `_save_resume_state`.
   REVISION 4, numerics surfacing (critique F49, remedy CORRECTED):
   print a loud, unmissable line the moment a step is skipped by the
   grad scaler or a group loss is non-finite, and print a SESSION-END
   SUMMARY (`N of M steps skipped, K non-finite losses`) so a human
   reading only the tail of a long Colab log still sees it. That is the
   whole change: the lane currently records `scaler_skipped` per row and
   in each sidecar but surfaces it only as one line among 282, which is
   why the plan's fp16 risk note could only assign the watch to a human
   eyeballing scrollback.
   What is deliberately NOT done, and why: the finding proposed raising
   on a non-finite group loss. That would be wrong. Transient
   non-finite loss under fp16 is exactly what `GradScaler` exists to
   absorb — gradients go non-finite, the step is skipped, the scale
   halves, and the run usually recovers — so an immediate raise would
   kill recoverable runs. Any defensible abort is a skip-streak count or
   a scale floor, i.e. a NUMBER, so the criterion was escalated as P16
   rather than invented here.
   **P16 RULED 2026-08-15 (RESEARCH_SPEC.md item T16): abort with a
   named error after 20 CONSECUTIVE grad-scaler skipped steps.** The
   counter resets on any applied step, so it measures a stall, not a
   total. 20 is ~7% of a 282-step run: longer than any legitimate
   scale-search transient, short enough to save most of a Colab session.
   The abort message must name the step index, the streak length and the
   current scaler scale, so the escalation ladder in the risks section
   (add warmup → reduce lr → escalate to the team) can be entered with
   evidence. The surfacing above ships independently of the abort.
   ACCEPTANCE TEST (rung 2, guarded): drive the loop with a stubbed
   scaler whose `get_scale` always shrinks, and assert it raises at the
   20th consecutive skip and not at the 19th; and that a single skip
   followed by an applied step does not raise. The concrete broken
   implementation this rejects: a counter that never resets, which would
   abort a healthy run that accumulated 20 scattered skips over 282
   steps.
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
- **The DEFAULT configuration runs** (revision 4; critique F43). Every
  guarded test to date sets `gradient_checkpointing: False` in
  `_config()`, so the production default has never executed anywhere,
  and the first execution of it would otherwise be a paid T4 session.
  A test runs the same tiny fixture with `gradient_checkpointing=True`
  and asserts the run completes, the loss decreases, and the per-step
  loss trajectory matches the `False` run to within a small tolerance.
  TOLERANCE, not `torch.equal`: the two agreed bit-for-bit when checked
  on 2026-08-15 (first 4.8545, last 4.7782 in both), but bit-identity
  across checkpoint recomputation is not a promise torch makes across
  versions, and a test that pins it would rot into a false alarm.
- **Non-reentrant mode is pinned** (revision 4; critique F43).
  RESEARCH_SPEC.md ratified 2026-08-13 that gradient checkpointing is
  non-reentrant only, and Stage-2's whole non-preclusion argument rests
  on it being the mode that coexists with forward hooks — yet nothing
  executed it. A spy over `model.gradient_checkpointing_enable`,
  following the file's existing `_train_capturing_attach` pattern,
  captures the kwargs and asserts `use_reentrant: False`. Asserting on
  a private `_gradient_checkpointing_func` attribute instead would be
  brittle across versions; the kwargs are the contract. The concrete
  broken implementation this rejects: a switch to reentrant
  checkpointing, which would silently break the Stage-2 hook.

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
update was skipped by the grad-scaler), `created` (UTC), and — added in
revision 4 (D10) — `encoding_sha256`, `renderer_sha256`,
`adapter_dtype`, so a checkpoint stays self-describing about how its
training data was rendered, the same reason the eval lane's rows carry
their full `gen_config`. Adding files or keys inside the adapter
directory does not disturb `eval._adapter_digest`, which hashes only
`adapter_model.safetensors` / `adapter_model.bin` /
`adapter_config.json` (verified 2026-08-15), so the checkpoint's eval
identity is unaffected.
REGRESSION NOTE: `tests/test_train.py::test_schedule_is_realized_exactly`
asserts an EXACT `set(meta)`; the three new keys must be added there or
that test fails.

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

REVISION 4 pure tests, all rung 1:

- **Each new guarded field refuses.** Three manifests differing from a
  baseline only in `encoding_sha256`, `renderer_sha256` and
  `adapter_dtype` respectively each fail `_guard_train_manifest` naming
  that field, and each produces a different `_manifest_identity_sha`.
- **`matched_training_identity` is round-trip-invariant** (critique
  F42): `matched_training_identity(_train_manifest(...))` equals
  `matched_training_identity(json.loads(json.dumps(_train_manifest(...))))`.
  The concrete broken implementation this rejects is the CURRENT one,
  where the in-memory tuple `target_modules` compares unequal to the
  on-disk list.
- **Two arms with different data still match** (D10; the guard against
  re-introducing critique F41's proposed remedy): two manifests
  identical except for `objective`, the dataset path, the dataset and
  meta digests, and `encoding_sha256` — that is, the real M_D/M_C
  relationship — must compare EQUAL under
  `matched_training_identity`. If `encoding_sha256` is ever added to
  that audit, this test fails, which is exactly what it is for.
- **`renderer_sha256` behaves oppositely**: two manifests differing only
  in `renderer_sha256` must NOT match within-family, and MUST match
  under `cross_family=True` (where it drops out beside `model_id` and
  `fold_system`).
- **`TrainConfig` validation** (critique F52): `dataclasses.replace` on
  the default config raises for each of `save_every=0`, `epochs=0`,
  `micro_batch_size=0`, `grad_accum_steps=0`, `n_checkpoints=0`,
  `lora_dropout=1.0`, `learning_rate=0`, `checkpoint_spacing="log"`,
  and empty `target_modules`; the default config itself constructs, and
  `DEFAULT_TRAIN_CONFIG.epochs = 5` still raises
  `FrozenInstanceError`, so
  `test_default_config_fields_are_frozen_and_complete` is unaffected —
  and its pinned field-name list is unchanged, because revision 4 adds
  no `TrainConfig` field (`adapter_dtype` is derived provenance, not
  configuration).

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

REVISION 4 (critique F43): this test MUST run with
`gradient_checkpointing=True`, not the `False` the shared `_config()`
helper supplies. The non-preclusion claim it exists to pin is
specifically that non-reentrant checkpointing is "the mode that
coexists with forward hooks" — and with checkpointing off, the test
proves the hook survives a mode Stage 2 will not use. Checked on
2026-08-15: the combination works on the tiny CPU fixture,
`bypass_state` still reports layer 1 after training, and only blocks
0/2/3 move their adapters, so turning the flag on costs a boolean and
buys the claim. This is the single largest item revision 4 moves off
the Colab rung.

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
- **train_seed**: **P15 RULED (a) on 2026-08-15 — adoption is ON**
  (RESEARCH_SPEC.md item T15, which AMENDS the 2026-08-13 row
  convention). Treatment is now identical to `checkpoint_step`: omitted
  → adopt the sidecar's value into the eval row; passed and mismatching
  → raise naming both values. The suspension, and the warning revision 4
  had planned for it under critique F46, are BOTH WITHDRAWN — F46's
  defect (a matching flag silently stamping a Stage-1 row) disappears
  once adoption is the correct behaviour, so there is nothing left to
  warn about.
  Why the flip is free, verified before ruling: the only rows in
  `results/` were 24 smoke rows with `adapter_path: null`, which
  evaluate no checkpoint and are unaffected. `train_seed` is per-row
  resume-identity-guarded and part of the ratified `summarize_runs`
  group key, so a flip AFTER trained checkpoints had been evaluated
  would have been messy; no trained checkpoint has ever been evaluated,
  so there is nothing to migrate.

Adapters without a sidecar (externally produced) behave exactly as
today; note that after F30, a crash can no longer strip a sidecar from
a checkpoint this lane wrote, so sidecar-less means genuinely
external. REVISION 4 (critique F45): "genuinely external" is the
assumption, and it fails open. `eval._adapter_digest` never hashes
`train_meta.json` (verified 2026-08-15), so a checkpoint copied to
Drive by anything that keeps only `adapter_model.safetensors` and
`adapter_config.json` loses its sidecar invisibly; the adoption then
silently does not happen, the row records `checkpoint_step: null` for a
mid-training checkpoint, and since `checkpoint_step` is in
`metrics.RUN_KEY_FIELDS` that run becomes a mislabeled point on the
Stage-3 R_t axis — the exact hazard the guard was built for, failing
silently instead of loudly. Added: when `--adapter` names a directory
that HAS `adapter_config.json` but NOT `train_meta.json` and
`--checkpoint-step` was omitted, print a loud warning that the step
will be recorded as null and that a project-trained checkpoint should
carry a sidecar. Deliberately a WARNING and not a refusal or a required
flag: the plan's contract that externally produced adapters behave
exactly as today is intentional, and a refusal would break the A_l
sweep's legitimate use of adapters this lane did not write.

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
encoding_sha256, renderer_sha256, adapter_dtype, created};
`train_manifest.json` (run identity, guarded on resume);
`train_log.jsonl` (one row per optimizer step; read it with
`train.read_train_log`, which keeps the last row per step);
`sessions.jsonl` (one append-only row per invocation, never guarded).
`encoding_sha256` fingerprints the run's encoded (input_ids, labels) and
`renderer_sha256` fingerprints the chat template applied to a fixed
probe, so a tokenizer or template change between sessions of one run, or
between two arms, cannot pass unnoticed; both are guarded run identity,
and only `renderer_sha256` participates in the matched-arms audit.
Cross-track note: `scripts/run_baseline.py` (eval track) imports
`train.checkpoint_meta` to read a checkpoint's sidecar. Step
convention: 0-based optimizer-update indices, "step" = last completed
(utils.py's pinned convention). `max_steps_this_session` stops cleanly
after N optimizer steps (Colab session bounds); rerunning the same
command resumes. `checkpoint_step` in results rows = the train_meta
value; run_baseline.py adopts it from the sidecar when the flag is
omitted and refuses a passed mismatch; it refuses outright to evaluate
a checkpoint whose sidecar records a training-time bypass until the
Stage-2 loader path exists. `train_seed` is adopted the same way
(ratified 2026-08-15, RESEARCH_SPEC.md item T15, which amends the
2026-08-13 row convention): a trained checkpoint's eval rows carry that
checkpoint's training seed, and null now means only "no trained
checkpoint was involved". Training REFUSES datasets whose manifest
`fold_system` mismatches what the model's chat template requires, in
BOTH directions (T12: Gemma-2 must train on folded data, Qwen and Llama
must not), and REFUSES a dataset not built from the ratified training
grid.
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
2. **Local ML venv — CORRECTED IN REVISION 4 (critique F48).** The
   environment revision 3 named here does not exist: `.venv` in this
   tree is Python 3.12.13 with NO torch, so item 2 as written was
   unexecutable and an implementer following it literally would hit an
   ImportError and accept the SKIP. The ML stack lives at
   **`~/.venvs/colab-local/bin/python`** — Python 3.11.15, torch
   2.13.0, transformers 5.15.0, peft 0.20.0 (measured 2026-08-15).
   Run there: `~/.venvs/colab-local/bin/python tests/test_train.py`
   actually runs (CPU tiny models; REQUIRED before this work is called
   done: a SKIP run is not verification; the implementer's summary
   names the environment). Because revision 4 touches `eval.py`, the
   FULL suite runs there too, not just the train files — see item 2c.
   **Tokenizer preflight** (critique F15/F16) — **EXECUTED 2026-08-15,
   critique-3 round; re-run after any change to encoding**: for each
   production family (Qwen/Qwen2.5-7B-Instruct,
   meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it), load the
   TOKENIZER only (all four are already in the local HF cache, so this
   runs offline under `HF_HUB_OFFLINE=1` with no download and no token)
   and run `train.encode_preflight` over the FULL regenerated
   1500-record build (the folded build for Gemma). RESULT: **zero
   prefix violations on any family**, and

   | model | max | p95 | mean | overflow@512 |
   |---|---|---|---|---|
   | Qwen2.5-7B-Instruct | 177 | 171 | 161.4 | 0 |
   | Llama-3.1-8B-Instruct | 184 | 178 | 170.5 | 0 |
   | gemma-2-9b-it (folded) | 167 | 161 | 152.0 | 0 |
   | Qwen2.5-0.5B-Instruct (dev) | 177 | 171 | 161.4 | 0 |

   D5's masking assumption therefore holds against all three real chat
   templates, and **P7's proposed cap of 512 is no longer unmeasured**:
   it has ~2.8x headroom over the longest real record. P7 is
   re-labelled *measured, awaiting ratification*. Caveat carried
   forward: this measures the CURRENT builder's output and says nothing
   about whether the file on Drive is a current build — that is D9's
   job, not the preflight's.
   In the same session, run `check_fold_compatibility` with the REAL
   Gemma-2 tokenizer against an unfolded build (must raise) and the
   folded build (must pass) — **EXECUTED 2026-08-15 (critique F38):**
   `_system_fold_needed` is True for gemma-2-9b-it alone;
   gemma+unfolded raises "fold mismatch", gemma+folded returns True,
   qwen+folded raises (the unratified P12 converse), qwen+unfolded
   returns False.
   REVISION 4 addition: confirm `renderer_digest` is stable across two
   loads of the same tokenizer and differs across the three families —
   the property D10's matched-arms audit depends on.
   If gated tokenizer downloads are ever unavailable on a fresh
   machine, the identical preflight (including the fold-guard check)
   runs as the FIRST Colab cell, before any model download.
   2c. **Full suite, same environment.** `test_bypass.py`,
   `test_data.py`, `test_eval_pure.py`, `test_interp.py`,
   `test_metrics.py`, `test_perplexity_count.py`, `test_scenarios.py`,
   `test_scoring.py`, `test_train.py`, `test_train_pure.py`,
   `test_wikitext_loader.py` all pass. This is MANDATORY in revision 4,
   not optional: the `eval._four_bit` extraction is a cross-lane edit
   and `tests/test_bypass.py::test_derive_gen_config_independent_oracle`
   is its cover. `tests/test_figures.py` is EXPECTED to fail on import
   for a pre-existing reason unrelated to this lane (round-3 O1: it
   imports `algoverse` with no `sys.path` insert and the package is not
   pip-installed in this venv); the implementer's summary must say so
   explicitly rather than let it read as a regression from this work.
   Dev end-to-end rehearsal: build `/tmp/ft` via
   `scripts/build_finetune_data.py --out-dir /tmp/ft --n 40`, run the
   corrected dev CLI invocation from WP-T7, then evaluate the final
   checkpoint through
   `python scripts/run_baseline.py --model-id Qwen/Qwen2.5-0.5B-Instruct
   --quant none --adapter /tmp/ft-run/checkpoints/step-00009
   --run-id dev-seam --out-dir /tmp/ft-eval --n 6 --skip-benchmarks`
   (all argparse-required flags present, critique F4), proving the
   train→eval artifact seam on a laptop, including the sidecar-adoption
   guard. REVISION 4: this rehearsal also exercises sidecar provenance:
   confirm `checkpoint_step` and `train_seed` are adopted when omitted;
   copy the checkpoint without `train_meta.json` and confirm the warning
   names whichever of those fields would remain null (both, or either one
   when the other flag is supplied); confirm supplying both avoids that
   warning; and confirm a passed value that contradicts the sidecar is
   refused. There is no P15 warning for a matching training seed: T15
   ratified adoption. The rehearsal also exercises the D9 grid guard
   against real builder output. Neither script has unit tests, by the same
   convention run_baseline.py already follows, so the rehearsal is their
   acceptance test and must actually be run.
3. **Colab sanity checks** (only observable on real hardware/models;
   these are the stated checks for what local tests cannot cover).
   REVISION 4 note: this list is now SHORTER by one. Gradient
   checkpointing — including the non-reentrant mode and its coexistence
   with the Stage-2 bypass hook — moved to rung 2 (WP-T4, WP-T6;
   critique F43), because it runs on the tiny CPU fixture and there was
   no reason for a paid T4 session to be its first execution. What
   remains genuinely needs CUDA:
   - 7B 4-bit + all-linear LoRA + gradient checkpointing fits the T4 at
     the proposed micro-batch/seq-len; record peak memory in the run
     notes. (Memory is the CUDA-only part; that the code path runs at
     all is now pinned at rung 2.)
   - fp16 + GradScaler: loss finite and decreasing over the first ~50
     steps on m_d_train; no NaN/inf; scaler scale stabilizes and
     `scaler_skipped` rows are rare (the bf16→fp16 deviation's
     empirical check). Run this check ONCE PER FAMILY, not once for
     Qwen: it is the empirical test of the P14 deviation, and Gemma-2
     is the family where it is least validated in advance (risk note
     below). REVISION 4 (critique F49): the loop now PRINTS a loud line
     on every skipped step and non-finite loss plus a session-end
     summary, so this check no longer depends on an operator spotting
     one line among 282 of scrollback. It still does not ABORT — the
     abort criterion is a number and is pending P16.
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
   because AGENTS.md is GITIGNORED and so absent from any clone —
   critique F21, corrected by round-3 O2, which found the file exists
   locally but is excluded by .gitignore along with CLAUDE.md, roles/
   and Prompts.txt). The summary must also name `test_figures.py`'s
   pre-existing import failure (item 2c) so it is not mistaken for a
   regression.

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
  mode that coexists with forward hooks. REVISION 4 (critique F43):
  this is no longer an assertion. WP-T6 now runs its bypass test with
  checkpointing ON, and a kwargs spy pins `use_reentrant: False`, so
  the claim Stage-2 leans on has an executing test at rung 2 instead of
  a first observation on a rented GPU.
- One TrainConfig + one schedule function = Stage-2 arms are matched by
  construction; `matched_training_identity` is the cross-arm audit, and
  its resume-identity hash (WP-T5) refuses a resume.pt that wandered
  between arms sharing dataset/seed/config (critique F8).
- `arm` labels, run layout, and the four-arm orchestration stay with
  the Stage-2 plan; nothing here presumes them.

## Ratified decisions (was: pending decisions) — RULED 2026-08-15

**All sixteen items P1-P16 were ratified by the human on 2026-08-15,
before any fine-tuning run existed and before any Gate-1 result on a
trained checkpoint had been seen.** The AUTHORITATIVE record of the
values is RESEARCH_SPEC.md, "Stage-1/2 fine-tuning constants (ratified
2026-08-15)", items T1-T16, numbered to match P1-P16 here. This section
is kept as the reasoning record — what was proposed, what the
alternatives were, and what each constant controls — because a value
without its argument is not much use to a teammate reading it later.
Where a ruling CHANGED the proposal, the change is recorded in place in
the entry itself.

Two rulings changed the proposal rather than confirming it: **P2**
(r=64 / dropout 0.1, not r=16 / 0.05) and **P10** (save 6, but separate
and defer the evaluated subset). **P12, P15 and P16 — the three open
escalations — were all ruled (a).** P15's ruling amends a previously
ratified sentence and the amendment is recorded in RESEARCH_SPEC.md at
the bullet it amends. **P7 is no longer unmeasured**: the tokenizer
preflight ran on 2026-08-15 and the longest real record across all
three families is 184 tokens, so the 512 cap was ratified on evidence.

Three items carry forward as live obligations rather than closing
outright, and are the only things left outstanding in this plan:

1. **T2's fallback activated 2026-08-16, and the human then ruled rank 16
   directly:** the Gemma-2 500-token T4 fit check failed at r=64, so all
   three families moved to r=16/dropout 0.05 together, and the human ruled
   the same value independently the same day. The rank is therefore settled
   and no longer contingent on the probe. The r=16 stress rerun also failed
   at 500 tokens; there is no further pre-committed fallback. Note for any
   future memory-conditioned rule: "fails to fit" is tested at the measured
   production sequence length (T7's preflight maximum), not at the
   `max_seq_len` refusal ceiling — the 2026-08-16 probes used 500 tokens
   against a 167-184-token production maximum and are a conservative stress
   case, not the operating point.
2. **T10's evaluated subset** — which checkpoint steps receive a full
   Stage-3 R_t evaluation — is owned by the Stage-2/3 plan and must be
   pre-committed in writing before any R_t is computed.
3. **T3's warmup escalation** (warmup_steps 0 → 10) applies only if
   fp16 proves unstable, and is a recorded deviation when it does.

Companion document: planning/train.ratification-proposal.md is the
short form of this same list and records the rulings alongside the
survey of what the established fine-tuning repos actually use. The
two must agree: if a value moves there, it moves here, and neither is
edited alone.

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
  **RULED 2026-08-15 — PROPOSAL CHANGED: r = 64, alpha = 16, dropout =
  0.1** (RESEARCH_SPEC.md T2). The reasoning inverted the framing above:
  if r=16 buys only storage while r=64 removes an assumption, the
  assumption is what is worth spending on. Storage was checked and ruled
  out as an argument (~131 GB for every checkpoint of every arm across
  both seeds = 2.6% of 5 TB), and so was compute: the LoRA overhead is
  exactly r(d_in+d_out)/(d_in·d_out) of the base matmul — 2.1% for
  Qwen's gate_proj at r=64 versus 0.53% at r=16 — while Stage-3
  evaluation, which dominates the GPU budget, is rank-independent. At
  r=64/alpha=16 the config IS the QLoRA recipe's Table 9 pairing, so the
  deviation and the transfer assumption both disappear; dropout follows
  the same table (0.1) so the citation stays coherent instead of
  cherry-picking Appendix A.1 for one field. The only real cost is T4
  VRAM, handled by the pre-committed ordered fallback in T2 (all three
  families to r=16 together; never per-family, never a micro-batch cut).
- **P3: Learning rate and schedule.** What it controls: how far the
  adapter weights move on each optimizer update, and whether that step
  size changes over the run (constant, decayed, or ramped up from zero
  at the start). Proposed 2e-4 (the value both papers use at this
  scale: Hu et al., https://arxiv.org/abs/2106.09685; Dettmers et al.,
  https://arxiv.org/abs/2305.14314), constant schedule (Dettmers et
  al.'s 7B recipe), warmup_steps=0. Flag: whether to add a small warmup
  (e.g. 10 steps) for fp16 stability.
  **RULED 2026-08-15 (T3): 2e-4, constant, warmup_steps = 0 — with 10
  PRE-COMMITTED as the escalation value if fp16 proves unstable.**
  Fixing the value now means the response to instability is a recorded
  deviation rather than an ad-hoc tweak mid-project. Note for whoever
  applies it: `warmup_steps` is part of the guarded run identity, so a
  run that changes it cannot resume — it restarts.
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
  regenerated data (verification item 2; critique F16).
  **MEASURED 2026-08-15** (revision 4): over the full 1500-record
  build the longest record is 184 tokens on Llama-3.1-8B-Instruct, 177
  on Qwen2.5-7B-Instruct and 167 on the folded gemma-2-9b-it build;
  p95 is 178 / 171 / 161; zero records exceed 512 on any family. The
  proposed cap has ~2.8x headroom and can now be ratified on evidence.
  Status: *measured, awaiting ratification* — no longer unmeasured.
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
  **RULED 2026-08-15 — the QUESTION was RESTRUCTURED before it was
  answered** (RESEARCH_SPEC.md T10). The team asked whether six
  intermediate checkpoints are load-bearing for a paper whose central
  focus is not the trajectory. The entry above conflated two separable
  decisions, and separating them dissolves most of the cost: SAVING six
  costs storage only (~52 GB for all of Stage 2 at the initially ruled
  r=64, ~13 GB at the effective r=16 — adapter size is linear in rank, so
  the effective figures are a quarter of the ones this argument was made
  with and the conclusion holds a fortiori; free at this
  project's scale), while EVALUATING one t costs 4 arms x 2 environments
  x full pools — roughly 4,720 generations per model — and six of those
  across three families is the largest line item in the GPU budget.
  Ruling: **keep 6 saved checkpoints at "doubling" spacing, and defer
  the evaluated subset** to the Stage-2/3 plan, which must pre-commit it
  in writing before any R_t is computed (the same outcome-independence
  reason P13 exists). That makes the expensive half reversible in the
  safe direction: a saved checkpoint can always be evaluated later, a
  checkpoint never saved cannot be recovered without retraining.
  Why the COUNT was not simply reduced: under "doubling", n=3 realizes
  [70, 140, 281] — it drops the EARLY points, which is backwards. Those
  are the scientifically distinctive ones; if R_t is already near 1 at
  step 8, the capability was arguably never removed, which is a
  different finding from gradual relocation. Cutting saves would have
  discarded exactly that evidence while saving none of the cost that
  matters. Stage 1 and Stage 2 do share the identical schedule.
- **P11: train_seed.** What it controls: the single integer that fixes
  adapter initialization, data shuffling order, and dropout draws, and
  hence the exact training run; the spec requires it be identical
  across arms so that arms differ only in their data. Proposed 42 for
  the primary run, identical across every arm (the spec's "matched
  random seeds", RESEARCH_SPEC.md Methodology); the second Stage-2 seed
  is governed by the pre-committed calendar policy (RESEARCH_SPEC.md
  2026-08-13), and its value (propose 43) can be ratified now or then.
- **P12: Strict fold-guard converse** (D7). **RULED (a) 2026-08-15 — refusal both directions, ratified as T12.** Was an OPEN ESCALATION
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
  **REVISION 4 additions for the ruling (critique F44).** Two facts the
  human should have in view. First, a PRECEDENT that cuts against this
  plan's own proposal: first-full-review F3's ratified handling was
  "the unratified numeric defaults stay UNCHANGED in code — no banner,
  no required flags", i.e. this project's established practice is that
  unratified positions sit as inert defaults rather than as
  enforcement. The converse currently ships as a hard refusal, which is
  the opposite of that practice. Second, a fact that lowers the stakes
  in either direction: measured 2026-08-15 against the real tokenizers,
  Qwen2.5 and Llama-3.1 both report `fold_required=False` and always
  take unfolded builds, so the converse branch is UNREACHABLE in the
  planned pipeline today. Ruling (c) now and (a) later costs nothing;
  what should not happen is the rule being inherited from an
  implementation choice instead of decided.
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
  convention.** **RULED (a) 2026-08-15 — adoption ON, amending the ratified row convention; ratified as T15.** Was an OPEN ESCALATION (critique F25). What it controls:
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
  merges and splits summary groups (loud, but avoidable). (The
  "per-row resume-identity-guarded" half of that sentence was verified
  in round 3: `run_negotiation_eval`'s `expected_top_level` check does
  include `checkpoint_step` and `train_seed`, so one run_id genuinely
  cannot mix two values.)
  **REVISION 4 sharpening (critique F46).** The suspension is only
  HALF effective, and the human should rule knowing that. The code
  refuses a `--train-seed` that MISMATCHES the sidecar, but ACCEPTS one
  that matches — so an operator who reads `train_seed: 42` out of the
  sidecar and types `--train-seed 42`, the most natural thing to do
  given the new sidecar plumbing, stamps 42 onto a Stage-1 row and
  contradicts the ratified null convention with no refusal. No recorded
  value can be made right under both rulings, so this cannot be closed
  by the plan. What revision 4 adds is a loud warning on that path
  (WP-T7), correct under either ruling; the ruling itself closes it.
- **P16: fp16 divergence abort criterion.** **RULED (a) 2026-08-15 — abort after 20 consecutive skipped steps; ratified as T16.** Was an OPEN ESCALATION (new in
  revision 4; critique F49). What it controls: whether a fine-tuning
  run that has stopped making progress in fp16 — the grad scaler
  skipping step after step, or the loss going non-finite and staying
  there — stops itself, or runs to completion writing checkpoints from
  a broken adapter. Today it runs to completion: revision 4 makes the
  condition LOUD (per-event lines plus a session-end summary, WP-T4
  step 9) but does not abort. Why this needs a ruling rather than a
  default: a transient non-finite loss is normal under fp16 and is
  exactly what `GradScaler` absorbs, so "abort on the first non-finite
  loss" would kill recoverable runs; any defensible criterion is a
  streak length or a scale floor, i.e. a NUMBER, and this plan does not
  invent numbers. PROPOSED: abort with a named error after 20
  consecutive `scaler_skipped` steps (~7% of a 282-step run, far longer
  than any legitimate scale-search transient, and short enough to save
  most of a Colab session). Alternatives: a grad-scaler scale floor;
  or ratify "no abort, surfacing only" as sufficient. Decide before the
  first 7-9B training run. Cost of getting it wrong in either
  direction is a wasted session, never a wrong number — the Gate-1
  verdict is unaffected either way, because a broken M_D fails the gate
  regardless; what the abort buys is telling a plumbing failure apart
  from the reportable scientific result "M_D did not become deceptive",
  which the plan elsewhere instructs the team to accept at face value.

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
- **Library version skew — REWRITTEN IN REVISION 4 (critique F48).**
  Revision 3 said "peft 0.17.1 local vs Colab latest … stable across
  recent versions". Both halves were wrong. The local ML environment is
  peft **0.20.0** on transformers **5.15.0** (measured 2026-08-15), so
  the gap this project actually spans includes a transformers MAJOR
  version, not a patch drift, and the reassuring conclusion does not
  follow from the premise it was based on. What IS verified, on the
  stack that exists: the used API surface (LoraConfig, get_peft_model,
  prepare_model_for_kbit_training, get/set_peft_model_state_dict,
  save_pretrained/from_pretrained) works, all 11 guarded tests pass
  there, `get_peft_model`'s `autocast_adapter_dtype` default is still
  True, and peft's built-in target-module mapping still returns
  `['q_proj', 'v_proj']` for qwen2 / llama / gemma / gemma2 — the fact
  P1's "spell it out explicitly" rule depends on. Versions stay
  recorded in the manifest as provenance and are never guard inputs
  (the repo's ratified stance); what revision 4 adds instead is
  `renderer_sha256` and `encoding_sha256`, which guard the OBSERVABLE
  CONSEQUENCE of a version change on the training data rather than the
  version string itself (D10). That is the right level: a transformers
  bump that does not change the rendering is harmless and should not
  refuse a resume, and one that does change it must.
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
- **Drive storage** (recomputed 2026-08-15 at the initial r=64 ruling from
  the real model configs): ~646 MB fp32 per all-linear adapter
  checkpoint on Qwen2.5-7B, ~671 MB on Llama-3.1-8B, ~864 MB on
  Gemma-2-9B (roughly half if adapters are saved fp16). At the ratified
  6-checkpoint schedule: ~26 GB for Stage 1 across all three families,
  ~52 GB for Stage 2, and ~131 GB including a second seed — 2.6% of the
  5 TB available, so storage is explicitly NOT a constraint on this
  project and was ruled out as an argument during P2's ratification.
  Effective r=16 checkpoints are smaller still.
- **T4 VRAM at r=64** (the pre-check arithmetic; now measured): the
  adapter's fp32 weights + gradients + AdamW moments
  come to ~2.58 GB on Qwen2.5-7B, ~2.68 GB on Llama-3.1-8B and ~3.46 GB
  on Gemma-2-9B, against ~0.65/0.67/0.86 GB at r=16. Gemma-2-9B is the
  tight case — roughly 11 GB of a T4's ~14.7 GB once its 256k-vocab
  logits are counted — and it is also the family least validated for
  fp16. If it does not fit, the response is the PRE-COMMITTED ordered
  fallback (T2): all three families drop to r=16 together. That fallback
  activated on Gemma's 2026-08-16 500-token OOM. Never a
  per-family rank (it would break matched settings across models) and
  never a `micro_batch_size` cut (under strict batch matching that
  propagates to every family and roughly doubles training wall-clock
  everywhere, which is the one change that would genuinely cost GPU
  budget).
- **HF gating**: Llama-3.1 and Gemma-2 downloads need an authenticated
  HF token, on Colab AND for the local tokenizer preflight (same as
  the eval lane will hit).
- **pyproject declares no extras** (verified): heavy deps stay
  undeclared by design; INTERFACES.md's install prose ("torch
  transformers accelerate peft") already covers the training track. No
  packaging change in this plan.
- **HF gating, corrected in revision 4**: Llama-3.1 and Gemma-2
  downloads need an authenticated HF token in general, but all four
  production tokenizers are already in the local HF cache on the
  project machine, so the preflight runs offline under
  `HF_HUB_OFFLINE=1` with no token. A fresh machine still needs one.

## Regression audit for revision 4 (do not skip)

33 tests pass today (22 pure + 11 guarded). Every entry below WILL
break a passing test unless the listed update lands with the change.
This exists because revision 4 is the first revision that edits landed
code, and "the change is right" is not the same claim as "the suite
still passes".

| Change | What breaks | Required update |
|---|---|---|
| `check_training_grid` requires `scenario` in meta rows | `test_train_pure.py::_builder_shaped` (feeds 6 tests) and `test_train.py::_write_dataset` (feeds all 11) | both fixtures emit a `scenario` drawn from the live `data.TRAIN_*` constants. `test_train.py`'s fixture replies carry no "MY BEST OUTSIDE OFFER:" line, so `extract_claimed_offer` returns None there and the eval-value half is a no-op — verified, no further change needed |
| 3 new `_train_manifest` parameters | `test_train_pure.py::_manifest()` | helper passes the three new kwargs |
| 3 new `GUARDED_MANIFEST_FIELDS` | nothing existing (they flow through `_guarded_view` automatically, and no production run's stored manifest exists) | add one guard test per field |
| 3 new `train_meta.json` keys | `test_train.py::test_schedule_is_realized_exactly` asserts an EXACT `set(meta)` | extend the expected key set |
| `matched_training_identity` gains fields + round-trip | `test_train_pure.py::test_matched_training_identity_scope` | extend, and add the two directional tests from WP-T5 (arms-with-different-data still match; `renderer_sha256` does not) |
| `TrainConfig.__post_init__` | nothing — `_config()` and `DEFAULT_TRAIN_CONFIG` are valid, `FrozenInstanceError` is unaffected, and the pinned field-name list is unchanged because no field is added | add the validation test |
| `eval._four_bit` extraction | nothing — behaviour-preserving; `test_bypass.py::test_derive_gen_config_independent_oracle` is the existing cover | run the FULL suite (verification item 2c), not just the train files |
| `_derive_quant` drops its config fallback | nothing — `test_train.py::test_bypass_bookkeeping_must_match_the_live_model` still raises on `quant_label="4bit"`, since an unquantized model derives `"none"` either way | none |
| `use_cache` restore, `pad_token_id` raise, `encode_preflight` empty raise | nothing | add the two pure raise tests |
| `run_baseline.py` warnings | nothing — no test covers the script, by the same convention it already follows | covered by the dev rehearsal, verification item 2 |

### Code deltas from the 2026-08-15 ratifications (separate from the critique fixes)

These land in the same pass. All were checked against the suite before
being written down; none breaks a passing test.

| Ruling | Code delta | Breaks? |
|---|---|---|
| T2 (P2) | Initial ruling: `lora_r` 16 → 64 and `lora_dropout` 0.05 → 0.1. Effective 2026-08-16 fallback after Gemma's T4 fit failure: **r=16, alpha=16, dropout=0.05 for every family**. | **No.** `test_default_config_fields_are_frozen_and_complete` now pins the effective values; step derivation and effective batch are unchanged. |
| T12 (P12) | `check_fold_compatibility`'s docstring drops "pending decision P12 … not a ratified rule" and cites T12. **Behaviour unchanged** | No |
| T15 (P15) | `run_baseline.py` ADOPTS `train_seed` from the sidecar when the flag is omitted, exactly as it already does for `checkpoint_step`; a passed mismatch still raises. The revision-4 F46 warning is **withdrawn, not implemented** | No |
| T16 (P16) | abort after 20 consecutive scaler-skipped steps, counter reset on any applied step; message names step index, streak length and scaler scale | **No.** On CPU the scaler is None and `scaler_skipped` is always False, so the streak never increments in any existing test. The new streak test drives a stubbed scaler |
| T3 (P3) | no code change — `warmup_steps` stays 0; the escalation value 10 is recorded in the plan and the risks ladder, not in the default config | No |
| T10 (P10) | no code change — `n_checkpoints` stays 6, `checkpoint_spacing` stays "doubling". The deferred evaluated-subset decision belongs to the Stage-2/3 plan | No |

Two things to NOT do, because they look like follow-ons and are not:
the ratified values do not change `epochs`, `micro_batch_size` or
`grad_accum_steps`, so `total_steps` stays 282 and every pinned
schedule vector in the tests stays valid; and P15's ruling REMOVES a
planned change rather than adding one — do not implement the F46
warning.

## Out of scope, surfaced to the human

Two round-3 observations are real but belong to someone else. They are
recorded here so they are not lost, and are NOT work items for this
plan — folding them in would be the cross-lane expansion the repo's
scope rule prohibits.

- **`tests/test_figures.py` cannot run in any environment here**
  (round-3 O1). It does `from algoverse import figures, metrics` with
  no `sys.path` insert, unlike every other suite, and needs `pytest`;
  `algoverse` is not pip-installed in `~/.venvs/colab-local`. So the
  repo's full suite is green nowhere. Introduced by commit 773658e and
  owned by the figures track. Until it is fixed, any claim that "the
  full suite passes" is false, which is why verification item 2c names
  the expected failure explicitly.
- **The repo's agent instructions are gitignored** (round-3 O2).
  `.gitignore` excludes `AGENTS.md`, `CLAUDE.md`, `roles/` and
  `Prompts.txt`, so a clone has none of them — including the "where to
  run tests" ladder. This plan restates everything an implementer needs
  inline, which is the right local response, but whether the team wants
  a cloned checkout to carry its own agent instructions is a decision
  for the human, not for this plan.
