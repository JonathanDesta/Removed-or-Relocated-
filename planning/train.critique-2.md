# Critique of planning/train.md, round 2

Critic pass over revision 1 of the train plan (Stage-1 LoRA (Low-Rank
Adaptation) fine-tuning). Repo state: branch eval-harness, commit
419468b, working tree clean, checked 2026-08-15. Scope of this round:
(1) re-verify every round-1 finding the disposition table marks
accepted, by recomputing the formulas and re-reading the revised text
against the code; (2) a fresh adversarial pass over the revised plan.
Both cited papers (LoRA, arXiv 2106.09685; the QLoRA recipe, arXiv
2305.14314) were re-fetched via ar5iv on 2026-08-15 and the plan's
literature-facing claims re-checked against the fetched text, not
memory.

Numbering continues round 1: new findings are F25-F39, each tagged
[severity / confidence]. Severity: high = would make a reported number
wrong, an experiment unreproducible, or a conclusion unsupported;
medium = breaks a stated verification, wedges a run, contradicts a
ratified decision, or leaves a wrong-number path one mistake away;
low = weakens a guard, a justification, or internal consistency.
No new finding this round reaches high.

---

## Part 1: Re-verification of the round-1 dispositions

Every "Accepted" row in the round-1 disposition table was checked
against the revised plan. Verdicts, with the arithmetic where the
kickoff asked for it:

- **F1 (seed before attach): LANDED.** WP-T4 now orders
  `utils.set_seed(train_seed)` as step 2, before the step-4 adapter
  attach, with the rationale (peft draws lora_A's init from the global
  torch random-number generator (RNG)) stated in place. The WP-T4
  determinism test now perturbs the ambient global RNG between the two
  runs and asserts byte-identical INITIAL adapter states in addition to
  final states, which is exactly the assertion that fails if seeding
  ever moves back after attach. The resume path (step 7 restores RNG
  from resume.pt after the fresh attach) is consistent with this.
- **F2 (batch quantity in the matched audit): LANDED, escalation
  intact.** The audit's config portion is now defined as the manifest
  guard's field set (full TrainConfig asdict minus save_every) plus a
  derived `effective_batch = micro_batch_size * grad_accum_steps`, with
  the strict both-split-and-product default and the P6 escalation
  stated as OPEN ("Confirm the reading and the audit treatment
  together"). One residual gap in the field partition: F27 below.
- **F3 (schedule pinned): LANDED, formulas verified by hand.**
  "even" = ceil(k*T/n) - 1 for k in 1..n:
  (T=100, n=4): 25-1, 50-1, 75-1, 100-1 = [24, 49, 74, 99]. Matches.
  (T=10, n=3): ceil(10/3)-1=3, ceil(20/3)-1=6, 10-1=9 = [3, 6, 9].
  Matches. (T=282, n=6): 47-1, 94-1, 141-1, 188-1, 235-1, 282-1 =
  [46, 93, 140, 187, 234, 281]. Matches. Duplicate-freeness claim
  checked: ceil((k+1)T/n) - ceil(kT/n) >= floor(T/n) >= 1 when T >= n,
  so consecutive entries differ; k=1 gives ceil(T/n)-1 >= 0, so
  in-range; k=n gives T-1, so the final step is always present.
  "doubling" = sorted set of (T-1) // 2^j, j in 0..n-1:
  (T=282, n=6): 281, 140, 70, 35, 17, 8 = [8, 17, 35, 70, 140, 281].
  Matches. (T=20, n=6): 19, 9, 4, 2, 1, 0 = [0, 1, 2, 4, 9, 19].
  Matches. (T=10, n=6): {9, 4, 2, 1, 0}, size 5 < 6, raises. Matches
  the pinned collapse-raise. P10 now ratifies a determined object.
- **F4 (dev rehearsal): LANDED, recomputed.** n=40, micro=4, accum=1,
  epochs=1 gives 10 micro-batches, total_steps=10; n_checkpoints=2
  with the default doubling spacing gives {9, 4} = [4, 9], matching
  the plan's stated schedule and the step-00009 eval target. The eval
  command now carries --model-id, --run-id, and --out-dir (all
  argparse-required in scripts/run_baseline.py, verified).
- **F5 (stdlib purity): LANDED.** `_read_jsonl` is a stdlib module
  function; the module map now states utils/torch/peft/transformers
  are imported inside functions only. utils.py's module-level
  numpy/torch imports re-verified (lines 11-12).
- **F6 (scaler semantics): LANDED.** The loop pins unscale_ / clip /
  scaler.step (skips on inf/NaN) / scaler.update, and pins that a
  skipped group still consumes step index t, with `scaler_skipped`
  recorded per log row and in an affected checkpoint's train_meta.
  Note: `scheduler.step()` advancing on skipped groups too is the
  choice that keeps lr a pure function of step index across hardware;
  internally consistent (checked and held).
- **F7 (same-step rewrite): LANDED.** `_write_checkpoint` now rmtrees
  an existing destination before os.replace, and a local test writes
  the same step twice. Three low-severity residuals opened by the new
  mechanism: F30, F31, F32 below.
- **F8 (resume identity): LANDED.** The resume.pt identity is now the
  sha256 of the canonical JSON of the FULL guarded-manifest field set,
  and the acceptance test includes the wrong-arm case (edited objective
  or bypassed_layer refuses). Minor: WP-T5's illustrative parenthetical
  list omits fold_system / n_examples / device_type that the "FULL
  ... field set from WP-T4 step 6" wording includes; the wording, not
  the parenthetical, is authoritative, so this is cosmetic only.
- **F9 (storage arithmetic): LANDED, recomputed.** Qwen2.5-7B
  (28 layers, hidden 3584, intermediate 18944, grouped-query kv width
  512): per layer at r=16, q/o 2 x 16 x 7168 = 229,376; k/v
  2 x 16 x 4096 = 131,072; gate/up/down 3 x 16 x 22528 = 1,081,344;
  total 1,441,792 per layer, x28 = 40.4M params, 161.5 MB fp32,
  ~81 MB fp16. Gemma-2-9B (42 layers, hidden 3584, q width 4096, kv
  width 2048, intermediate 14336): 1,286,144 per layer, x42 = 54.0M,
  216 MB fp32. Attention-only r=16 on Qwen: 360,448 x 28 = 10.1M,
  ~40 MB fp32. All three numbers in the plan are now correct.
- **F10 (checkpoint_step cross-check): LANDED**, via
  `train.checkpoint_meta` plus the run_baseline sidecar-adoption
  guard. Two new findings fall out of how it landed: F25 (train_seed
  adoption contradicts a ratified row convention) and F26
  (bypassed_layer adoption vs install_bypass ordering).
- **F11 (objective guard strength): LANDED.** Counts checked against
  data.py: the builder writes md_deceptive = n_per_dataset // 2 and
  mc_deceptive = 0, plus n_per_dataset, exactly the fields the revised
  `check_objective` compares (k == md_deceptive and k == n // 2 for
  "deceptive"; k == 0 and mc_deceptive == 0 for "control";
  n == n_per_dataset). The tiny-build wiring test (n=8: k=4=8//2)
  passes the guard by construction.
- **F12 (meta/manifest integrity): LANDED.** Length equality, per-row
  behavior/fold_system keys, meta_sha256 computed AND guarded; the
  hand-edited-after-build residual is honestly stated as post-hoc
  auditable only.
- **F13 (partial-group scaling): LANDED.** The loss divides by the
  ACTUAL group size (n=1500, micro=2, accum=8, epochs=3: 2250
  micro-batches, 281 full groups of 8 plus a final group of 2, 282
  steps, so the final scheduled checkpoint's update is now correctly
  weighted), and the equal-weighted-micro-batch-mean objective is
  documented with the token-weighted alternative named.
- **F14 (converse fold guard): ESCALATED AS CLAIMED.** P12 presents
  (a) ratify refusal, (b) strike, (c) warn-only, with the
  enforcement-before-ratification concern stated. Genuinely open.
- **F15/F16 (template preflight, length measurement): LANDED.**
  `encode_preflight` exists, runs per-family over the full regenerated
  build with a Colab-first-cell fallback, and P7 now says ratification
  should follow the measurement, value unsettled.
- **F17 (contract naming): LANDED.** WP-T8's signature carries
  max_steps_this_session; the sidecar field is quant_label everywhere;
  the four_bit correspondence is stated once in WP-T5. One remaining
  nit: F37 below.
- **F18 (continuation-path preparation): LANDED.** Gradient
  checkpointing + input grads run on both branches; k-bit
  re-preparation is explicitly the Stage-2 loader plan's obligation;
  the checklist now says "near-free" with the seam named.
- **F19 (quant derived): LANDED.** `_derive_quant` cross-checks the
  caller's quant_label and the manifest records the derived value,
  matching eval's `_derive_gen_config` pattern (verified in eval.py).
- **F20 (path guard): LANDED.** dataset_path is recorded, digests are
  guarded.
- **F21 (AGENTS.md): LANDED.** The verified-vs-written rule is stated
  inline in verification item 4; AGENTS.md is correctly noted absent.
- **F22 (citations): LANDED.** "single 24/48 GB GPU" wording fixed
  (matches the paper's sentence verbatim per the 2026-08-15 re-fetch);
  P2 presents rank-irrelevance as a transfer assumption. Re-fetch
  confirmed: Gaussian A / zero B with exact-zero delta at start
  ("We use a random Gaussian initialization for A and zero for B, so
  Delta W = BA is zero at the beginning of training"), alpha set to the
  first r and not tuned, Wq/Wv default, lr 2e-4 (Tables 11/12);
  QLoRA: all-linear required (section 4 / Figure 2), rank unrelated
  when all layers adapted (Appendix A / Figure 4), the dropout
  text-vs-Table-9 inconsistency is real and recorded accurately, 7B
  recipe r=64 / alpha=16 / lr 2e-4 / constant schedule / batch 16 /
  max_grad_norm 0.3 / Adam beta2 0.999 confirmed, bf16 compute
  confirmed. One precision residual: F39.
- **F23 (audit scope): LANDED**, with the cross_family mode and the
  expected-to-differ list. Residual partition gap: F27.
- **F24 (loss-curve home): LANDED.** `read_train_log` is a named
  stdlib function with a pure test over duplicated rows.

Both escalations are presented as OPEN with options, not resolved: P6
(strict default now, split fields drop out only if the human ratifies
the product-matched reading) and P12 (three enumerated options, E6
direction stays a refusal regardless). Pending-values discipline holds:
P1-P14 are proposals, DEFAULT_TRAIN_CONFIG carries them as unratified
code defaults, the INTERFACES.md addition is proposed text for the
human, and spec items 15-17 are untouched. Every paper citation in the
plan carries its arXiv URL (checked: scope, method provenance, WP-T1,
P1, P2; no other papers are cited).

---

## Part 2: New findings (F25-F39)

## F25 — run_baseline's train_seed adoption contradicts the ratified "null for Stage-0/1 runs" row convention, without flagging the conflict [medium / high]

RESEARCH_SPEC.md, Ratified decisions (2026-08-13): "Results rows carry
a `train_seed` field (fine-tuning seed identity ...): null for
Stage-0/1 runs, the training seed for Stage-2 arms." planning/
layer-bypass.md's disposition of the same ratification repeats it:
the field "stays null until Stage-2 arms exist." The revised plan's
WP-T7 guard makes run_baseline adopt `train_seed` from
train_meta.json whenever `--train-seed` is omitted and a sidecar
exists. Every Stage-1 checkpoint's sidecar carries train_seed (42
proposed), so every Gate-1 evaluation of M_D or M_C now writes rows
with train_seed = 42, and since `--train-seed` is `type=int
default=None`, an operator CANNOT request the ratified null for a
sidecar-carrying adapter at all. Consequences: (a) a ratified row
convention is silently overridden by an agent-planned guard rather
than escalated (the plan escalates P12 for exactly this class of
issue, then does the same thing here without noticing); (b)
`train_seed` is identity-guarded per row in `run_negotiation_eval`
(verified, expected_top_level) and part of the ratified
summarize_runs group key, so a pre-change eval and a post-change
re-run of the same checkpoint under one run_id refuse to merge, and
across run_ids they land in different summary groups. Loud, not
silent, and arguably the adopted value is the BETTER semantics (the
ratified rationale is "fine-tuning seed identity", which a Stage-1
checkpoint has), but reconciling the ratified sentence is the human's
call and the plan neither mentions the conflict nor lists it as
pending.

## F26 — bypassed_layer sidecar adoption is unspecified relative to run_baseline's install_bypass step; one reading quietly implements the Stage-2 loader deliverable the plan disclaims [medium-low / medium]

WP-T7: "for each of --checkpoint-step, --train-seed,
--bypassed-layer: if the flag was omitted, adopt the meta value into
the eval row." run_baseline.py currently installs a bypass iff
`args.bypassed_layer is not None` (line 122) BEFORE the eval call,
and `run_negotiation_eval` refuses when the bookkeeping value
disagrees with live `bypass_state` (verified, eval.py lines 327-336).
So for a Stage-2 checkpoint whose sidecar says bypassed_layer = 1
and an omitted flag: if adoption happens before the install block,
the guard has de facto implemented reinstall-at-load, which the
plan's own NON-GOALS and planning/layer-bypass.md §P2 assign to the
Stage-2/loader plan ("nothing in THIS plan reinstalls from
metadata"); if adoption is row-only (the literal wording), the run
loads the model, then reliably dies in run_negotiation_eval with a
bookkeeping-vs-live mismatch, a confusing failure whose fix the
operator cannot know. Stage-1 sidecars carry null so nothing fires
today, which is exactly why the ambiguity will surface only when
Stage-2 checkpoints exist. The plan should pin the row-only reading
plus an explicit early raise ("this checkpoint records a permanent
bypass; evaluating it requires the Stage-2 loader path"), or
explicitly hand the case to the loader plan.

## F27 — matched_training_identity's field partition omits model_id from both the include and the exclude list [low / medium]

Quantity homes enumerates the audit's included fields (config asdict
minus save_every, effective_batch, n_examples, total_steps,
checkpoint_steps, train_seed, quant_label, dtype, device_type,
fold_system) and the excluded fields (dataset path and digests,
objective, out_dir, timestamps, bypassed_layer, save_every, package
versions). model_id, which IS in the guarded manifest set, appears in
neither. Two implementers can disagree: include it and
cross_family=True always fails (model_id necessarily differs across
families) unless the cross-family mode also drops it, which the plan
does not say; exclude it and the within-family audit cannot catch an
arm accidentally trained from the wrong base (a dev 0.5B manifest
slipping into a 7B arm comparison passes). F2's fix was precisely
"one list, not two drifting prose lists"; the one list has a hole.

## F28 — "recorded in the manifest history" names a mechanism the plan never defines, and the write-once manifest contradicts it [low / medium]

WP-T4 step 10 says max_steps_this_session is "recorded in the
manifest history but never identity-guarded"; step 6 lists it under
"RECORDED but NOT guarded" in train_manifest.json, which is
"Write-once, atomic". A multi-session run has a different
max_steps_this_session (and timestamp) per session, but a write-once
manifest can only hold the first session's values; there is no
defined history structure (a sessions list, a sidecar log, anything).
Implementers will either silently keep the stale first-session value
(the recorded provenance is then wrong for every later session) or
invent an unplanned rewrite of a file the plan says is write-once.

## F29 — field-by-field manifest comparison over tuple-typed config fields refuses every resume unless comparison is canonicalized, and the plan does not say to canonicalize [low / medium]

TrainConfig.target_modules is specified as a tuple; checkpoint_steps
is a list. `dataclasses.asdict` preserves the tuple; a manifest
re-read from JSON has a list. `_guard_train_manifest` "compares
field-by-field", and `("q_proj",) != ["q_proj"]` in Python, so the
naive in-memory-vs-loaded comparison flags `config` as mismatched on
EVERY resume. The resume-identity hash is explicitly computed over
"canonical JSON" (so it is immune); the manifest guard has no such
sentence. Not silent (the exact-resume acceptance test dies on run
B's second call), but the plan specifies the trap and not the escape:
one line ("guard comparisons happen after a JSON round-trip of the
current manifest") closes it.

## F30 — whether train_meta.json is written inside the tmp directory before os.replace is unpinned; the wrong choice creates loadable checkpoints without sidecars, silently disabling the F10 guard [low / medium]

`_write_checkpoint` writes the adapter via save_pretrained "into a
tmp sibling" then os.replaces it into place, and "train_meta.json
inside it records ...". If the sidecar is written into the FINAL
directory after the replace, a crash in that window leaves a
complete, loadable adapter directory with no train_meta.json. WP-T7's
run_baseline guard treats sidecar-less adapters as "externally
produced" and reverts to today's free-typed flags, so the
crash-window artifact is evaluated with zero cross-checking, exactly
the transposed-digit path F10 closed, and nothing marks the
degradation. The plan's own "A half-written adapter dir is never
loadable" sentence suggests the meta-inside-tmp ordering is intended;
it should be stated, and the same-step-rewrite test should assert the
sidecar exists.

## F31 — tmp-directory hygiene in _write_checkpoint is unspecified; a stale leftover tmp can smuggle files into a final checkpoint and change its eval identity [low / low]

A crash between save_pretrained and os.replace leaves the tmp sibling
behind. If the tmp path is a fixed name and the rerun reuses it
without clearing, save_pretrained overwrites adapter_config.json and
adapter_model.safetensors but does not remove other stale files; the
replace then publishes them. eval._adapter_digest (verified) hashes
adapter_model.bin AND adapter_model.safetensors when both exist, so a
stale .bin from an older environment changes the checkpoint's
adapter_digest, and identity-guarded eval resumes of that checkpoint
refuse for an unexplainable reason. Cheap fix to specify: rmtree the
tmp path first (or use a unique tmp name and clean up).

## F32 — "deterministic replay makes the rewritten content byte-identical" is false on the production hardware [low / medium]

The F7 justification asserts byte-identical rewrite on the
crash-window rerun. On CUDA fp16, replay from the last resume state
is not guaranteed bit-deterministic (torch does not promise run-to-run
bit equality without torch.use_deterministic_algorithms, which the
plan never enables; nondeterministic kernels exist in backward paths).
Nothing breaks, because the rewrite replaces the directory wholesale
and the Colab kill-rerun check only asserts dir-set and log coverage,
but the plan's stated reason the rewrite is safe is an exactness claim
that only holds on the CPU test rig. The honest wording is
"replayed-step content is equivalent up to fp16 nondeterminism, and
the rewrite is wholesale, so no mixed-state directory can result." As
a corollary, an eval run that had already consumed the pre-crash
version of that checkpoint would refuse resume on adapter_digest, a
loud but undocumented consequence.

## F33 — encode_conversation's pinned signature has no masking-policy input, yet the function is "the single home of the policy either way" [low / high]

Module map and WP-T3 pin `encode_conversation(tokenizer, messages,
max_seq_len)`, and WP-T3 then specifies behavior conditional on
`mask_prompt_tokens` ("when mask_prompt_tokens is false ... labels
equal input_ids"), a value the function cannot see. The acceptance
test even calls it with `mask_prompt_tokens=False`. The signature
needs the flag (or the config); as written, two internally
inconsistent texts describe one function.

## F34 — total_steps has a pinned formula but no callable home; the pure test that claims to pin its derivation has nothing to call [low / medium]

WP-T1's acceptance tests include "total_steps derivation pinned for
the worked example (n=1500, micro=2, accum=8, epochs=3 -> 282
steps)", but the derivation lives only in prose and inline in
train_lora step 5; no function in the module map computes it. The
test either re-implements ceil(epochs * ceil(n/micro) / accum) inside
itself (vacuously passing against nothing) or the implementer invents
an unplanned helper. total_steps feeds the checkpoint grid (the R_t
x-axis) and the manifest guard, so it deserves the same
named-pure-function treatment checkpoint_schedule got in F3 (e.g. a
`derive_total_steps(n_examples, config)` the loop and the test both
call).

## F35 — --config-json unknown-key handling is unspecified; a typo'd override is silently ignored [low / low]

"a JSON object merged over DEFAULT_TRAIN_CONFIG": merging
{"grad_accum_step": 1} (typo) silently trains with the default 8. The
manifest records the truth, so the error is auditable and the
schedule arithmetic usually makes it loud on the dev box, but a
strict raise-on-unknown-key rule costs two lines and matches the
plan's refuse-dont-drift posture. Production arms use pure defaults,
so exposure is dev-side.

## F36 — resume=False semantics are unspecified [low / low]

`train_lora(..., resume=True)` is in the signature and the contract
text, but the plan never says what resume=False does against a
populated out_dir (ignore resume.pt and retrain from step 0,
rewriting checkpoints? refuse? require an empty directory?). The
identity guard makes the manifest consistent either way, but
retrain-in-place would append a second full set of train_log rows
that read_train_log's keep-last rule would then silently merge with
the first run's, which is the one place the ambiguity could touch a
reported curve.

## F37 — WP-T8's sidecar field list omits `created` [low / high]

WP-T5 specifies train_meta.json records `created` (UTC); the proposed
INTERFACES.md text lists every sidecar field except it. Trivial, but
the contract is the document other tracks read, and the repo
convention is that schema lists match exactly.

## F38 — the Gemma fold-refusal check is listed as Colab-only but is locally testable with infrastructure the plan already requires [low / low]

Verification item 3's last bullet defers "the fold-refusal fires on
unfolded data" to the Gemma arm's Colab session. The guard needs only
the tokenizer (eval._system_fold_needed probes the chat template),
and verification item 2 already requires downloading the gated Gemma
tokenizer locally for the preflight. Running check_fold_compatibility
once locally with the real Gemma tokenizer against an unfolded build
(and the folded build passing) would convert a
first-observable-on-Colab claim into a laptop test for free. The stub
test covers the logic; this covers the real-template claim.

## F39 — "gradient checkpointing used throughout" overstates what the QLoRA paper says [low / low]

Method provenance asserts the recipe used gradient checkpointing
throughout. The 2026-08-15 re-fetch finds gradient checkpointing in
the paper's section 2 memory accounting ("With gradient
checkpointing, the input gradients reduce to an average of 18 MB per
sequence") and section 3 (paged optimizers exist "to prevent memory
spikes during gradient checkpointing"), which strongly implies use
but never states it as an experimental setting, and it appears in no
hyperparameter table. Round 1 marked this claim confirmed; on
re-reading, the support is inferential. The design choice here is
independently forced by the T4 anyway; only the attribution wording
("used throughout" -> "assumed by the paper's memory accounting")
needs softening.

---

## Checked and held (attacks that failed this round)

- **Schedule edge cases**: n_checkpoints = total_steps under "even"
  yields [0..T-1] (all unique); n_checkpoints = 1 yields [T-1] under
  both spacings; T = 1, n = 1 yields [0]; the collapse raise and the
  n > T raise cannot shadow each other.
- **Worked-example arithmetic**: 1500/2/8/3 -> 750 micro-batches per
  epoch, 2250 total, 282 steps, final group of 2; the dev recipe
  40/4/1/1 -> 10 steps, doubling [4, 9]; both internally consistent
  everywhere they are quoted (WP-T1, WP-T7, P5, P10).
- **check_objective vs the real builder**: data.py writes exactly
  md_deceptive = n // 2, mc_deceptive = 0, n_per_dataset; folded and
  unfolded builds both carry fold_system in manifest and every meta
  row; unfolded records all have a leading system turn, folded ones
  none, so the belt-and-braces scan matches reality.
- **Fold detection parity**: `eval._system_fold_needed(tokenizer,
  probe)` exists with that exact signature and only folds on a
  system-role error, re-raising otherwise; the plan's guard reuses it
  rather than hardcoding model names, matching E6's ratified text
  (planning/first-full-review.md lines 1322-1328).
- **E8 adoption**: 0-based indices, "step" = last completed,
  loader returns step + 1, fresh run 0, matches utils.py's pinned
  docstrings; the convention-pin acceptance test (schedule [4],
  dir step-00004, resume returns 5) is the right shape.
- **Resume-skip arithmetic**: steps 0..next_step-1 always consumed
  exactly grad_accum_steps micro-batches each (the only partial group
  is the run's last, after which there is nothing to resume), so
  skipping next_step * grad_accum_steps micro-batches without forward
  passes is exact; `_epoch_order` being a pure function of
  (train_seed, epoch, n) makes the skip order-safe, and the
  100003-multiplier seed derivation cannot collide across epochs.
- **Scaler-skip step semantics**: advancing step index AND scheduler
  on skipped groups keeps both hardware-invariant; the alternative
  (skip-aware scheduler) would make lr depend on skip history. The
  plan's choice is the consistent one and is documented.
- **Step-0 identity and WP-T6**: zero-init B makes the fresh adapter
  exact identity (LoRA section 4.1, re-fetched quote); the hook lives
  on the decoder block module, which get_peft_model does not replace,
  so bypass survival and no-grad-through-bypassed-block claims match
  models.py's implementation and the layer-bypass plan.
- **run_baseline guard feasibility**: --checkpoint-step,
  --train-seed, --bypassed-layer all exist with default None
  (verified), so omitted-vs-passed is detectable; `checkpoint_meta`
  is stdlib-testable.
- **Storage arithmetic and citations**: recomputed and re-fetched as
  itemized in Part 1 (F9, F22).
- **Governance**: P1-P14 remain proposals; P6/P12 remain open
  escalations; the spec's open step-convention item is fulfilled (its
  assigned action), not resolved-by-agent; no sweep-bound or
  analysis-constant is touched; INTERFACES.md changes are proposed
  text only; every cited paper carries its arXiv link.
- **Fixtures and environment claims**: RecordingTokenizer
  (tests/test_eval_pure.py line 31), the guarded-skip runner pattern
  (tests/test_bypass.py), _tiny_model (4 layers, vocab 128),
  pyproject declaring no extras with requires-python >= 3.9, and
  train.py's current TO-BUILD stub state were all re-verified.
- **generate_batch padding hazard** (D6's justification): eval.py
  line 258 really does mutate tokenizer.padding_side to "left"; a
  collator that pads plain lists itself is immune to call order.

---

## Planner disposition (round 2, 2026-08-15)

Adjudicated per the revision protocol before editing the plan. All 15
findings accepted; 1 of them additionally escalated to the humans
(F25, new pending item P15); 0 rejected.

| Finding | Disposition | Reason / applied fix |
|---|---|---|
| F25 | Accepted + ESCALATED (new P15) | Conflict verified against RESEARCH_SPEC.md ("null for Stage-0/1 runs, the training seed for Stage-2 arms", ratified 2026-08-13). Resolving a ratified row convention is the humans' call, not the plan's. Plan default until ruled: train_seed adoption SUSPENDED (rows stay null when the flag is omitted); an explicitly passed flag mismatching the sidecar still refuses, which is compatible with either ruling. P15 also records the timing consequence the critic identified (row identity guard + summarize_runs group key make a mid-project semantic flip loud and messy). |
| F26 | Accepted | Verified against run_baseline.py line 122 and eval.py lines 327-336. The finding understates the damage: the round-zero mismatch-refusal would also break the A_l eval-time bypass sweep on M_D (sidecar bypassed_layer null, sweep passes a layer; RESEARCH_SPEC item 10/17 runs that sweep on the trained checkpoint), because the eval row's bypassed_layer is the LIVE eval-time lesion, a different quantity from training-time provenance. Fix: bypassed_layer is dropped from adoption AND from flag cross-checking entirely; a NON-null sidecar bypassed_layer raises early, naming the reinstall-at-load loader path as the Stage-2/loader plan's deliverable (row-only reading rejected in favor of the early raise; a reliable late refusal inside run_negotiation_eval is the confusing failure the finding warns about). No escalation needed: the fix conforms to the already-ratified reinstall-at-load assignment rather than touching it. |
| F27 | Accepted | model_id is in the guarded manifest set but in neither audit list. Fix: model_id joins the within-family (cross_family=False) include set (catches a dev-scale manifest slipping into a 7B comparison); cross_family=True drops model_id together with fold_system. |
| F28 | Accepted | "Manifest history" named an undefined mechanism contradicting write-once. Fix: per-session operational values (max_steps_this_session, session start, step at entry) move to an append-only out_dir/sessions.jsonl, one row per train_lora invocation, never guarded; the manifest stays write-once and drops max_steps_this_session. |
| F29 | Accepted | ("q_proj",) != ["q_proj"] is real and would refuse every resume. Fix: one pinned line, guard comparisons happen after a JSON round-trip of the current manifest. |
| F30 | Accepted | Fix: train_meta.json is written inside the tmp directory BEFORE os.replace, so no crash window can publish a loadable adapter without its sidecar; the same-step-rewrite test now asserts the sidecar exists. |
| F31 | Accepted | Stale-tmp smuggling verified plausible via eval._adapter_digest (hashes .bin AND .safetensors when both exist). Fix: rmtree the tmp path first if present. |
| F32 | Accepted | The byte-identical-replay claim only holds on the CPU rig. Fix: wording corrected to wholesale-replacement equivalence up to fp16 nondeterminism, and the corollary documented (an eval run that consumed the pre-crash checkpoint refuses resume on adapter_digest, loudly). |
| F33 | Accepted | Fix: encode_conversation gains mask_prompt_tokens=True in its pinned signature (module map + WP-T3), matching the behavior text and the acceptance test. |
| F34 | Accepted | Fix: derive_total_steps(n_examples, config) added as a named pure function; WP-T4 step 5 and the WP-T1 acceptance test both call it (plus the dev vector 40/4/1/1 -> 10). |
| F35 | Accepted | Fix: --config-json raises on unknown keys. |
| F36 | Accepted | Fix: resume=False pinned as a fresh-run assertion (raise if out_dir already holds train_manifest.json or resume.pt), which forecloses the retrain-in-place log-merge hazard; the CLI never passes it. |
| F37 | Accepted | Fix: `created` added to WP-T8's sidecar field list. |
| F38 | Accepted | Fix: check_fold_compatibility with the real Gemma-2 tokenizer (unfolded build raises, folded passes) added to the local tokenizer preflight, riding the same Colab-first-cell fallback if the gated download is unavailable; the Colab bullet stays as in-situ confirmation. |
| F39 | Accepted | Re-read confirms the support is inferential (memory-accounting mentions, no hyperparameter-table entry). Fix: attribution softened to "assumed by the paper's memory accounting", labeled an inference. |
