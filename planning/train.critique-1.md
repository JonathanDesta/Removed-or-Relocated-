# Critique of planning/train.md — round 1

Critic pass over the round-zero train plan (Stage-1 LoRA (Low-Rank
Adaptation) fine-tuning, M_D / M_C checkpoints). Repo state: branch
eval-harness, commit 419468b. Both cited papers (LoRA, arXiv 2106.09685;
QLoRA, arXiv 2305.14314) were fetched and read for this critique on
2026-08-15; claims were checked against the fetched text, not memory.
Environment facts used for feasibility checks: local dev Python 3.9.6 /
torch 2.8.0 / transformers 4.57.6 / peft 0.17.1 on MPS; stdlib laptop
Python 3.14 with no ML stack; production Colab T4 (fp16, no bf16).

Findings are numbered F1..F24, each tagged [severity / confidence].
Severity: high = would make a reported number wrong, an experiment
unreproducible, or a conclusion unsupported; medium = breaks a stated
verification, wedges a run, or leaves a wrong-number path open that
needs one more mistake to fire; low = weakens a guard, a justification,
or the plan's own internal consistency. A "checked and held" section at
the end lists the attacks that failed.

---

## F1 — Adapter initialization is not governed by train_seed; the plan's own determinism test contradicts its order of operations [high / high]

WP-T4 orders `train_lora` as: step 3 adapter attach
(`peft.get_peft_model`), step 6 `utils.set_seed(train_seed)`. peft
initializes lora_A from the GLOBAL torch RNG (Gaussian/kaiming init;
lora_B is zero). So the lora_A initialization is drawn from whatever the
global RNG state happens to be when `train_lora` is called, and
`train_seed` has no effect on it. Consequences:

- **Spec violation.** RESEARCH_SPEC's binding sentence requires "matched
  ... random seeds" across all fine-tuning. The M_D and M_C arms (and
  later the four Stage-2 arms) are launched as separate processes;
  torch's process-fresh default seed comes from OS entropy, so every arm
  gets a different adapter init and no arm is reproducible from its
  recorded `train_seed`. The manifest records a seed that does not
  determine the run.
- **The WP-T4 determinism acceptance test cannot pass as specified.**
  "Two fresh runs with identical config/train_seed into two out_dirs
  produce byte-identical final adapter state dicts": in one test process
  the second call sees a different global RNG state at attach time, so
  the inits differ and the final states differ. Either the test fails,
  or the test harness seeds the global RNG before each call, in which
  case it passes while the production CLI path stays broken (an
  acceptance check that passes on broken code).
- Resume is unaffected (the fresh attach is overwritten by
  `set_peft_model_state_dict`), which makes the bug quieter: the exact-
  resume test passes while cross-run reproducibility is broken.

The fix direction is obvious (seed before attach, or seed twice), but as
written the plan specifies the broken order explicitly and builds a test
matrix that does not catch it.

## F2 — `matched_training_identity` contains no batch-size quantity; the executable "matched optimization settings" audit cannot see an unmatched effective batch [high / medium]

The field list in "Quantity homes" is: n_examples, epochs/total_steps,
checkpoint_steps, "optimizer and LoRA constants", max_seq_len, masking
policy, train_seed, quant/dtype/device_type, fold_system. Neither
`micro_batch_size`, `grad_accum_steps`, nor their product appears, and
batch size is not an "optimizer constant" in any natural reading (those
are lr, betas, weight decay, clip). Meanwhile P6 declares the SPLIT
operational and the PRODUCT the matched quantity, so the product must
live somewhere in the executable audit, and it does not. Two arms
trained with effective batch 16 and effective batch 8 can produce
identical `matched_training_identity` dicts (total_steps can coincide;
worse, with matched epochs and n it usually does differ, but the audit
should not depend on a derived count accidentally catching it, and with
the split excluded and the product absent, an accum typo that preserves
total_steps passes silently). This is exactly the silent path the
matched-fine-tuning sentence exists to close. Related ambiguity: it is
unstated whether `gradient_checkpointing`, `lr_schedule`, and
`warmup_steps` are inside "optimizer constants"; the manifest guard is
specified as "the full TrainConfig asdict minus save_every", and the
divergence between that precise list and the matched-identity prose list
is itself a drift surface. Confidence medium only because "optimizer and
LoRA constants" might be intended as shorthand for the full asdict, but
then P6's split-is-operational reading contradicts it (the split would
be guarded as matched), so at least one of the two texts is wrong.

## F3 — The checkpoint schedule, the paper's R_t x-axis, is underspecified; the acceptance tests do not pin "doubling" at all [medium / high]

`checkpoint_schedule` is the single home of the R_t step grid, yet:

- "doubling": the entire specification is "final plus halving offsets
  from the start (a log-ish grid ...)". For total=282, n=6 this is
  compatible with {8, 17, 35, 70, 140, 281}, {0, 17, 35, 70, 140, 281},
  {281, 273, 257, 225, 161, 33} (halving offsets FROM the final,
  read differently), and several rounding variants. No worked example,
  no formula, no acceptance test pins it. Two implementers produce two
  different x-axes for the recovery curves and both pass every stated
  test.
- "even": the prose ("indices closest to k/n_checkpoints fractions of
  total") gives 25, 50, 75, 100 for total=100, n=4; the worked example
  says [24, 49, 74, 99]. The intended formula is evidently
  round(k*total/n) - 1, but the prose as written contradicts the test
  vector, and floor-vs-round at half-integers is unpinned.
- Degenerate cases: with doubling, small totals produce duplicate
  offsets; the contract says "duplicate-free" and "n_checkpoints >
  total_steps raises", but whether doubling with dedup-collapsed output
  returns fewer than n_checkpoints points or raises is unspecified.

Since P10 asks the human to ratify "6 checkpoints, doubling", the human
is ratifying a word whose realized step set is not determined by the
plan. The grid should be pinned by exact expected outputs before
ratification, not after implementation.

## F4 — The dev end-to-end rehearsal crashes as specified [medium / high]

Verification item 2: build with `--n 40`, run the dev CLI invocation
(`--config-json '{"epochs": 1, "micro_batch_size": 4}'`). With the
defaults that invocation keeps (grad_accum_steps=8, n_checkpoints=6):
micro-batches = ceil(40/4) = 10, total_steps = ceil(10/8) = 2, and
`checkpoint_schedule(2, 6, ...)` raises by WP-T1's own acceptance rule
(n_checkpoints > total_steps). The plan's laptop rehearsal, the one
"proving the train→eval artifact seam", fails at step 4 of train_lora
as written. Also in the same item, the eval half of the rehearsal
(`run_baseline.py --quant none --adapter <ckpt> --n 6
--skip-benchmarks`) omits `--model-id`, `--run-id`, and `--out-dir`,
all of which argparse marks required. Both are recipe bugs, loud not
silent, but this is the plan's only pre-Colab full-seam check and it
does not run.

## F5 — The stdlib-purity claim is broken by utils.py: the pure test file cannot test what WP-T2 puts in it [medium / high]

`load_training_data` is specified to call `utils.read_jsonl`, and
WP-T2's acceptance tests for it (plus the manifest/meta/guard tests and
the real-data wiring test) are placed in `tests/test_train_pure.py`,
which verification item 1 runs on the stdlib laptop (Python 3.14, no ML
stack). But `src/algoverse/utils.py` imports numpy and torch at MODULE
level (utils.py lines 10-12), so the first call into
`utils.read_jsonl`, however lazily imported, raises ImportError on that
box. The module map's phrase "stdlib + pure-algoverse only" treats
utils as pure; it is not. Either train.py must read JSONL itself (or
via a stdlib-only helper), or the WP-T2 tests move to the guarded file
and verification item 1's claim shrinks. As written, verification item
1 fails, and the plan does not know it. (data.py and eval.py ARE
stdlib-importable at module level; the wiring test's use of
`build_finetune_datasets` is fine.)

## F6 — The fp16 loop spec omits GradScaler step/update semantics; a literal implementation applies overflowed gradients [medium / medium]

WP-T4 step 7 says: "Per group: unscale, clip to max_grad_norm,
optimizer step, scheduler step, zero grads". The correct fp16 sequence
is `scaler.unscale_(optimizer)`, clip, `scaler.step(optimizer)` (which
SKIPS the step when inf/NaN gradients were found), `scaler.update()`.
The plan's wording, "optimizer step", read literally as
`optimizer.step()`, applies inf/NaN gradients on overflow steps and
corrupts the run in exactly the fp16-fragility scenario the risk
section worries about. Additionally, when `scaler.step` legitimately
skips, the plan's step semantics are undefined: does the skipped group
still consume step index t (log row, checkpoint eligibility, scheduler
step)? Whatever the answer, it changes what "checkpoint step t" means
on CUDA versus the CPU tests (which never skip), and no test or
sentence pins it. Not a science-breaking issue across arms (indices
advance identically), but the loop's central invariant, "completing
step index t" = one optimizer update, silently becomes "at most one
update" on fp16 hardware.

## F7 — Checkpoint re-write on the crash-rerun path wedges: os.replace onto an existing non-empty directory fails [medium / medium]

`_write_checkpoint` writes a tmp sibling then `os.replace`s into
`checkpoints/step-NNNNN/`. On POSIX, rename onto an existing NON-EMPTY
directory raises OSError (ENOTEMPTY). WP-T4 step 8 writes the
checkpoint BEFORE the resume state; a crash between the two (or between
the checkpoint and any later resume save) means the rerun resumes from
an earlier step, re-completes the scheduled step, and calls
`_write_checkpoint` onto the already-existing directory: every rerun
crashes at the same step, a resume deadlock requiring manual deletion.
The plan specifies neither remove-before-replace nor
skip-if-exists-and-meta-matches, and the Colab kill-and-rerun sanity
check only catches this if the kill happens to land in that window.
The local resume test (clean `max_steps_this_session` interruption)
never exercises it, because clean stops save resume state after the
checkpoint.

## F8 — The resume.pt identity block cannot distinguish arms that share dataset, seed, and config [medium-low / medium]

WP-T5's identity block is {dataset_sha256, train_seed, config hash}.
The Stage-2 arms (I,D) and (L,D) share all three (same m_d dataset,
same matched seed, same matched config; they differ in objective's
starting checkpoint and `bypassed_layer`, neither of which is in the
block). Likewise (I,C) vs (L,C). A resume.pt that ends up in the wrong
out_dir (Drive sync duplication, a copied directory skeleton) passes
`_load_resume_state`'s identity check and silently continues arm A's
optimizer/adapter state under arm B's manifest. The train_manifest
guard does not close this: a FRESH out_dir with a stray resume.pt gets
a fresh manifest written first, then the resume file is accepted. The
block should bind the full guarded-manifest identity (model_id,
objective, bypassed_layer at minimum), which costs nothing. Stage-1
exposure is smaller (M_D vs M_C runs differ in dataset digest) but the
mechanism lands in this plan and Stage 2 inherits it.

## F9 — The r=16 storage arithmetic quoted to the human is the attention-only number, not the all-linear number [medium-low / high]

Method provenance, deviation (b): "r=16 all-linear on a 7B is ~35-40 MB
fp32". Computed for Qwen2.5-7B (28 layers, hidden 3584, intermediate
18944, GQA kv 512): all-linear r=16 is about 1.44M LoRA params per
layer, about 40.4M params total, which is ~161 MB in fp32 (~81 MB
fp16); Gemma-2-9B comes to ~216 MB fp32. The quoted ~35-40 MB is what
ATTENTION-ONLY r=16 costs (~10.1M params, ~40 MB fp32) while P1
simultaneously proposes all-linear. The relative claim (r=64 is ~4x
r=16) survives, but the absolute number is ~4-5x low, and it feeds two
places: the P2 ratification rationale (checkpoint-storage arithmetic is
one of the two stated justifications for deviating from QLoRA's r=64)
and the "Drive storage" risk multiplier. The human would be ratifying a
deviation on a wrong supporting number.

## F10 — Eval-row `checkpoint_step` is operator-copied with no cross-check; a typo silently moves a point on the R_t curve [medium-low / high]

"Quantity homes" defines the eval row field as the train_meta value and
says "operators copy it (or a future eval nicety reads it — out of
scope here)". `run_baseline.py --checkpoint-step` is free-typed; nothing
compares it against the adapter directory's train_meta.json, even
though the eval lane already computes `_adapter_digest` of that same
directory and the plan itself puts `checkpoint_step` inside
train_meta.json precisely so it has one authoritative home. A
transposed digit attributes one arm's tau at step 140 to step 70; R_t
at both steps is then wrong with no guard anywhere (summarize_runs
groups by the asserted value). The repo's own pattern (bypass
bookkeeping is derived-and-cross-checked, not trusted) argues the
cheap check belongs in scope somewhere before Stage-3 numbers exist;
the plan explicitly declines it without assigning it to any other
plan.

## F11 — The objective guard's threshold is far weaker than the invariant it protects [low / high]

`check_objective`: "deceptive" requires deceptive-behavior rows > 0.
The M_D dataset is half deceptive BY CONSTRUCTION (750 of 1500), and
the data manifest already records `md_deceptive` / `mc_deceptive`
exactly. A truncated, corrupted, or hand-mixed file with 1 deceptive
row passes the guard while training a substantively different
objective. Checking meta counts against the data manifest's recorded
counts (and against n/2 within tolerance) costs three lines and closes
the gap between "not the control file" and "the intended M_D mixture".

## F12 — Meta and manifest files are outside the digest; a stale meta beside regenerated records passes silently [low / medium]

`dataset_digest` hashes only the records file. The objective guard and
fold record-scan consume `<stem>.meta.jsonl`, and fold_built comes from
`manifest.json`; neither is digested, and the "light shape checks" are
unspecified (a records/meta LENGTH equality check is not promised
anywhere). Regenerating records while an old meta survives (partial
copy, interrupted rebuild) yields guards evaluated against the wrong
rows with a clean manifest. Related residual: the plan's stale-data
refusal (missing `fold_system` key) only catches pre-fold-provenance
builds; a file built after fold provenance landed but hand-edited or
built from a stale checkout of the firewall constants carries
`fold_system` and passes. Recording the data manifest verbatim in
train_manifest.json (which the plan does) makes this auditable post
hoc but not refused up front.

## F13 — Accumulation-group loss scaling: the final partial group is divided by the full grad_accum_steps [low / high]

WP-T4 divides every micro-batch loss by `grad_accum_steps`, and "only
the final group of the run may be partial". That final group's
gradient is therefore scaled down by (group size / accum), up to 8x at
accum=8: the run's last optimizer update, which is also always a
scheduled checkpoint (final step in every schedule), is taken with a
silently shrunken effective step. Equal-weighting micro-batch MEANS
(rather than supervised-token counts) is likewise a choice that makes
the objective differ from mean token cross-entropy. Both are matched
across arms, so no cross-arm number is biased, but both belong in the
docstring as deliberate choices, and the final-group scaling
interacts with P10 (the final checkpoint is the Gate-1 artifact).

## F14 — The D7 converse fold guard is active by default before its ratification [low / high]

The plan is admirably explicit that the converse (refuse folded data on
non-fold models) EXTENDS the ratified E6 rule and lists it as P12
pending confirmation, but the code as planned enforces it
unconditionally from day one. If the human strikes P12, ratified
behavior requires a code change; until they rule, an agent-decided
methodological rule is live. It is a refusal (conservative, cannot
produce a wrong number, only blocks runs), so severity is low, but it
is the one place the plan's hold-the-line-on-pending-values discipline
slips from "proposed in code as default" to "enforced as an error".

## F15 — The chat-template prefix property is asserted for Llama-3.1 and Gemma-2 but never checked against a real tokenizer before the Colab run [low / medium]

D5's parenthetical, "the generation prompt is a textual prefix of the
assistant turn in all three", is stated as fact with no test that
touches a REAL template: the pure tests use stub tokenizers (which
verify the mechanics, not the claim), and the dev rehearsal covers
only Qwen2.5-0.5B. The id-prefix check can also fail even when the
string-prefix holds (tokenizer merges across the prompt/completion
junction), and for Llama/Gemma the first place either failure mode can
fire is mid-Colab, after model download, on record 0 of a 7-9B run.
Tokenizer-only downloads of the gated models are a few MB and work on
the laptop with an authenticated token; a local check over the real
1500-record build for all three families is feasible and not planned.
The raise is loud, so nothing is silent; the finding is that a claim
labeled "checked, not assumed" (risk section) is in fact checked
nowhere for two of the three families until the most expensive moment.

## F16 — P7 (max_seq_len=512) is proposed without measuring the real data, and overflow aborts a Colab run at encode time [low / medium]

Encoding is up-front and overflow is a hard raise (correctly, per
D5/WP-T3). But nobody has counted tokens on the real regenerated 1500
conversations under any real tokenizer; the incentive-framing stakes
paragraphs plus scenario text plus reply could plausibly approach 512
on the wordier paraphrase combinations, and per-family token counts
differ. A local token-length histogram (Qwen tokenizer at minimum)
would either ground the 512 proposal or move it before ratification;
as planned, the first measurement is a crash after 7B model load on
the T4.

## F17 — Contract-text inconsistencies in WP-T8 [low / high]

The INTERFACES signature proposed for the humans omits
`max_steps_this_session` while the module map's signature includes it
(callers reading the contract will not know clean session-bounded stops
exist, which is precisely the Colab usage pattern). The checkpoint
sidecar field is named `quant` in train_meta.json but the argument and
manifest field are `quant_label`; the eval lane's corresponding derived
field is `four_bit`. Three names for one fact across two files invites
the exact quiet-renaming drift the repo's conventions forbid.

## F18 — The already-PeftModel (Stage-2 continuation) path skips k-bit preparation; "Stage-2 reuse free" is overstated [low / medium]

WP-T4 step 3 runs `prepare_model_for_kbit_training` (fp32 norms, input
grads, gradient-checkpointing enable) only on the fresh-attach branch.
The Stage-2 continuation path (model arrives as a PeftModel loaded
`is_trainable`) skips creation AND, as written, skips preparation, so a
4-bit continuation would train without fp32 norms or gradient
checkpointing unless the Stage-2 loader plan independently does it.
The Stage-2 non-preclusion checklist does not mention this, while
claiming D2 makes reuse "free". Cheap fix: run the
preparation/gradient-checkpointing block regardless of which branch
attached the adapter, or state explicitly that preparation is the
loader plan's obligation.

## F19 — quant_label is asserted, not derived [low / medium]

Constraint 5 claims the lane meets the eval lane's
"derived-not-asserted" bar, and D2 applies it to bypass state, but
`quant_label` is caller bookkeeping stamped into a guarded manifest
field with no cross-check against the live model (the eval lane derives
`four_bit`/dtype/device from the model object). A `--quant none` model
trained under a manifest saying 4bit (script bug, future refactor)
would be a silently wrong provenance record on a matched-identity
field. The model object exposes what is needed to derive it.

## F20 — Guarding the dataset PATH string makes resume refuse across mount-point changes [low / low]

`_guard_train_manifest` guards "dataset path + digest". The digest is
the identity; the path is where it was mounted. Colab Drive remounts
and local-vs-Drive staging change the path with identical bytes, and
the run then refuses to resume for a non-methodological reason
mid-session. Record the path, guard the digest. Possibly intended
strictness; flagged so it is a decision rather than an accident.

## F21 — Verification item 4 cites AGENTS.md, which does not exist in the repo [low / medium]

"Report verified-vs-written per AGENTS.md": no AGENTS.md exists
anywhere in the working tree (globbed). If the convention lives
outside the repo the reference should say where; as written the
implementer cannot follow it.

## F22 — Literature check: the plan's citations hold, with two small notes [low / high]

Both papers were fetched (ar5iv full text) and every literature-facing
claim in "Method provenance" was checked:

- LoRA: Gaussian A / zero B, exact-zero delta at start (§4.1):
  confirmed verbatim. Alpha-to-first-r-untuned: confirmed. W_q/W_v-only
  main experiments: confirmed. Table 5 / §7.1 more-matrices-lower-rank:
  confirmed. lr 2e-4 for GPT-2 (Table 11) and GPT-3 (Table 12):
  confirmed.
- QLoRA: all-linear-layers-required (§4 / Figure 2): confirmed. Rank
  unrelated to performance when all layers adapted (Appendix A,
  Figure 4): confirmed. The dropout inconsistency the plan reports IS
  real (Appendix A.1 says 0.05 useful for 7B/13B; the Table 9
  hyperparameters pair 0.1 with up-to-13B and 0.05 with 33B/65B), so
  the plan's honest recording is accurate. 7B recipe r=64, alpha=16,
  lr 2e-4, constant schedule, batch 16, max_grad_norm 0.3, Adam
  beta2=0.999: confirmed. bf16 compute, gradient checkpointing used:
  confirmed. Paged optimizers "critical to do 33B/65B QLoRA tuning on
  a single 24/48 GB GPU": confirmed; the plan says "on 48 GB", a
  trivial imprecision.
- Note 1: QLoRA's rank-irrelevance evidence is instruction-tuning
  benchmark performance (Alpaca-style eval). Using it to justify r=16
  for a deception-behavior SFT objective is an extrapolation across
  task type; defensible, but the P2 rationale presents it as a direct
  finding rather than a transfer assumption.
- Note 2: the declared-deviation framing is sound and all three
  deviations are genuinely flagged for ratification (P14), which is the
  correct handling.

## F23 — The matched-identity audit is per-model-family only, and the plan does not say so [low / medium]

`matched_training_identity` includes fold_system, dtype, and quant.
Across model families these legitimately differ (Gemma trains folded on
a different digest; families could in principle differ in quant
fallback), so the audit can only ever be run within one family's four
arms. That is sufficient for R_t (computed per model), but the spec
sentence being operationalized ("ALL fine-tuning uses matched ...")
reads across the whole experiment, and the plan nowhere states the
audit's scope or which fields are EXPECTED to differ across families.
When the paper claims matched fine-tuning across three models, the
executable home covers less than the sentence.

## F24 — The training-loss "quantity home" is a file plus a reader convention, not a function [low / medium]

Every other reported quantity gets one function; the appendix loss
curve gets `train_log.jsonl` plus a documented last-row-per-step rule
that some future figure script must implement correctly. Duplicate
rows are guaranteed to occur (crash-window reruns), so the first
naive `pandas.read_json(lines=True).plot()` produces a visibly or
invisibly wrong curve. If the curve ships in the paper, the keep-last
reader deserves a named home (even a 5-line `train.read_train_log`)
and a pure test.

---

## Checked and held (attacks that failed)

- **E6 enforcement**: the fold guard as specified (live-tokenizer
  detection via `eval._system_fold_needed`, manifest `fold_system`,
  KeyError on missing key, record scan both directions) matches the
  ratified obligation and the data lane's actual output (verified
  against data.py: every unfolded record has a leading system turn;
  folded builds have none; manifest and meta carry the flag).
- **E8 adoption**: the step convention (0-based, "step" = last
  completed, loader returns step + 1, fresh run returns 0) matches
  utils.py's pinned docstrings exactly, including the 282-step worked
  example (n=1500, micro=2, accum=8, epochs=3 gives ceil(2250/8)=282;
  final checkpoint step-00281).
- **Pending-values discipline**: P1-P14 are genuinely proposed, not
  resolved; DEFAULT_TRAIN_CONFIG-as-unratified-default follows the
  ratified F3-handling precedent; spec items 15-17 (sweep bounds) are
  correctly untouched; the INTERFACES addition is proposed text for
  the humans, per the governance rule. The one slip is F14's
  default-on converse guard.
- **WP-T6's gradient claims** are backed by planning/layer-bypass.md
  (bypassed block's params, including LoRA deltas inside it, receive
  no grads; AdamW skips None-grad params; the DDP
  find_unused_parameters caveat is correctly carried over).
- **Module import hygiene**: eval.py and data.py really are
  stdlib-importable at module level, so train.py importing them is
  consistent with the stdlib-import claim (utils is the exception,
  F5).
- **Referenced fixtures and flags exist**: tests/test_bypass.py
  `_tiny_model` (4-layer Qwen2, vocab 128), tests/test_eval_pure.py
  `RecordingTokenizer` and hardened `__main__` runner,
  run_baseline.py `--adapter`/`--n`/`--skip-benchmarks`,
  build_finetune_data.py `--out-dir`/`--n`/`--fold-system`,
  `build_finetune_datasets(out_dir, n_per_dataset, seed, fold_system)`.
- **Step-0 identity via zero-init B** is sound (exact-zero LoRA path;
  torch.equal on CPU fp32 holds; dropout is irrelevant through a zero
  B), and the corresponding LoRA-paper claim is verbatim true.
- **D6's padding-side isolation** correctly addresses a real hazard
  (`generate_batch` mutates `tokenizer.padding_side` to "left";
  eval.py line 258), and right padding with default position ids is
  correct for training.
- **peft 0.17.1 API surface**: LoraConfig / get_peft_model /
  prepare_model_for_kbit_training(use_gradient_checkpointing=...,
  gradient_checkpointing_kwargs=...) / get_/set_peft_model_state_dict
  all exist at the pinned local version.
- **Arms are matched-by-construction on total_steps and data order**
  (same n, config, and pure `_epoch_order`; M_D/M_C records are
  index-aligned by the builder, so matched seeds give matched batch
  composition across arms), MODULO F1, which currently breaks the
  matched-init half of that story.

---

## Disposition (Planner, revision session 2026-08-15)

Adjudicated per the revision protocol before editing planning/train.md.
Counts: 22 accepted, 0 rejected, 2 escalated (F14 wholly; F2's
split-vs-product sub-question). Repo facts re-verified independently
before adjudication: utils.py module-level numpy/torch imports (F5),
absence of AGENTS.md (F21), run_baseline.py required flags and
free-typed --checkpoint-step (F4, F10), data.py manifest fields and
meta-row shape (F11, F12), and the F9 storage arithmetic (recomputed
from Qwen2.5-7B / Gemma-2-9B dimensions; critic's numbers confirmed).

| Finding | Disposition | Reason |
|---|---|---|
| F1 | Accepted | Verified against the plan's own step order (attach before seed) and peft's global-RNG lora_A init. Fix: `utils.set_seed(train_seed)` moves to the top of `train_lora`, before any RNG-touching operation; the determinism test now perturbs the ambient global RNG between the two runs and additionally asserts the two INITIAL adapter states are byte-identical, so the test catches exactly this regression instead of masking it. |
| F2 | Accepted; sub-point escalated | The audit's config portion is now DEFINED as the identical list the manifest guard uses (TrainConfig asdict minus save_every), killing the parallel-prose drift surface, plus an explicit derived `effective_batch` key. Default is strict: split fields AND product both audited. Whether the split fields drop out (P6's "product matched, split operational" reading) is a pending human call, escalated under P6. |
| F3 | Accepted (definitions pinned; values stay pending under P10) | Exact formulas and acceptance vectors are now pinned for both spacings ("even": ceil(k*T/n)-1; "doubling": (T-1)//2^j deduped, raise on collapse), so P10 ratification is over a determined object. The n_checkpoints/spacing VALUES remain pending; flagged in P10. |
| F4 | Accepted | Verified: with the kept defaults (accum 8, n_checkpoints 6), n=40 gives total_steps 2 < 6 and the schedule raises; the run_baseline call omits three argparse-required flags. Dev recipe corrected to a schedule-feasible config-json and a complete eval command. |
| F5 | Accepted | Verified utils.py imports numpy and torch at module level (lines 11-12). train.py gets a stdlib `_read_jsonl`; utils is imported inside functions only; verification item 1's claim now actually holds. |
| F6 | Accepted | Loop text now pins scaler.unscale_ / clip / scaler.step (skips on inf/NaN) / scaler.update, and pins skip semantics: a skipped group still consumes step index t; the log row and, when scheduled, the checkpoint's train_meta record `scaler_skipped`, keeping step indexing hardware-invariant and the skip auditable. |
| F7 | Accepted | `_write_checkpoint` now rmtree's an existing destination before os.replace (deterministic replay makes the rewrite byte-identical); a local test writes the same step twice to pin the rerun path. |
| F8 | Accepted | The resume.pt identity block now binds a sha256 of the full guarded-manifest identity (model_id, objective, bypassed_layer, digests, quant_label, dtype, seed, config), closing the wrong-arm-resume path for Stage 2 at zero cost. |
| F9 | Accepted | Recomputed independently: all-linear r=16 on Qwen2.5-7B is ~40.4M LoRA params, ~161 MB fp32 (Gemma-2-9B ~216 MB); 35-40 MB is the attention-only figure. Numbers corrected in the deviation rationale and the Drive-storage risk. The relative r=64-is-4x claim stands, and the corrected arithmetic still supports the r=16 proposal (P2 informed, not settled). |
| F10 | Accepted | Assigned concretely instead of declined: `train.checkpoint_meta(adapter_dir)` (pure helper) plus a minimal run_baseline.py guard (adopt checkpoint_step/train_seed/bypassed_layer from train_meta.json when present; an explicitly passed mismatching flag raises). run_baseline.py moves out of the module map's Untouched list; deliberate, noted cross-lane touch. |
| F11 | Accepted | `check_objective` now takes the data manifest and checks the meta deceptive count against md_deceptive / mc_deceptive and the structural expectation (n//2 or 0), plus n against n_per_dataset; feasibility verified against data.py's actual manifest fields. |
| F12 | Accepted | Added: records/meta length equality (raise), required per-row keys, meta-file sha256 recorded AND guarded beside the records digest. The hand-edited-with-valid-fold_system residual remains post-hoc-auditable only (manifest recorded verbatim); stated honestly in the plan. |
| F13 | Accepted | The final (possibly partial) accumulation group now divides by the actual group size, so the last update (always a scheduled checkpoint) is correctly weighted; the objective (equal-weighted mean of per-micro-batch mean CE over supervised tokens) is documented as a deliberate choice with the token-weighted alternative named. Matched across arms either way. |
| F14 | Escalated | Touches pending P12 directly. The plan keeps refusal-by-default as the coded proposal (the Qwen default-system-injection makes the converse a silent-training-distribution-corruption path, which argues for refusal over warning), but whether an unratified rule may be live as an error before ratification is a governance call for the human; P12 now states both options and the enforcement-before-ratification concern. |
| F15 | Accepted | New `encode_preflight(tokenizer, records, ...)` plus a local per-family tokenizer-only preflight over the full regenerated build (Colab-first-cell fallback when gated downloads are unavailable locally), so the prefix and id-prefix claims are checked for all three families before any 7-9B run. |
| F16 | Accepted (P7 stays pending) | The same preflight reports the token-length histogram; P7 text now says ratification should follow the measurement. No value settled by the plan. |
| F17 | Accepted | `max_steps_this_session` added to the proposed INTERFACES signature; the sidecar field is renamed `quant_label` to match the argument and manifest field; the correspondence to eval's derived `four_bit` is stated once in the plan. |
| F18 | Accepted | Gradient-checkpointing / input-grads enabling now runs on both attach branches; fp32-norm k-bit preparation for the continuation path is explicitly assigned to the Stage-2 loader plan, and the non-preclusion checklist qualifies "free" to "near-free, with the loader plan owing is_trainable loading plus k-bit re-preparation". |
| F19 | Accepted | quant is now derived from the live model (`_derive_quant`) and cross-checked against the caller's quant_label, derive-and-refuse, the same pattern the lane already applies to bypass state. |
| F20 | Accepted | Guard the digests, record the path. The path guard was unintended strictness, not a decision; now it is a decision, the one the critic recommended. |
| F21 | Accepted | Verified AGENTS.md exists nowhere in the tree. The verified-vs-written reporting rule is now stated inline in the verification section. |
| F22 | Accepted | Both notes applied: "single 24/48 GB GPU" wording fixed; P2 now presents rank-irrelevance as a transfer assumption from instruction-tuning benchmarks rather than a direct finding for this objective. |
| F23 | Accepted | `matched_training_identity` now documents its scope (within one model family's arms) and gains a cross_family mode that drops the family-varying fields (fold_system, dataset-adjacent fields), with the expected-to-differ list written down so the paper's cross-family "matched" sentence maps to an explicit, smaller executable check. |
| F24 | Accepted | `read_train_log(path)` (stdlib, keep-last-row-per-step, sorted by step) is now the named home for the loss curve, with a pure test over duplicated rows. |
