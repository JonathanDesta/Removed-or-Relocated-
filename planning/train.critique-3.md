# train.critique-3 — implementation + plan critique (round 3)

Scope: the whole pull `009e97d..fbe0f36` — `planning/train.md` rev 3 and
the Stage-1 LoRA implementation landed with it (`src/algoverse/train.py`,
`scripts/run_finetune.py`, the `scripts/run_baseline.py` guard,
`tests/test_train.py`, `tests/test_train_pure.py`). Reviewed against
RESEARCH_SPEC.md, INTERFACES.md, the plan, and the existing code's
conventions. Findings continue from critique-2 (F25–F39).

## Environments that actually ran

- `python3` 3.14.6 — `tests/test_train_pure.py`: **22/22 PASS**;
  `python3 -c "import algoverse.train"` succeeds (the stdlib-importability
  invariant holds); `python3 tests/test_train.py` prints the loud SKIP and
  exits 0.
- `~/.venvs/colab-local/bin/python` (3.11.15, torch 2.13.0, transformers
  5.15.0, peft 0.20.0) — `tests/test_train.py`: **11/11 PASS**. Whole repo
  suite re-run there: every suite passes except `tests/test_figures.py`
  (pre-existing, see O1).
- No GPU was used and none was needed.
- I additionally executed the plan's **mandatory verification item 2**
  (tokenizer preflight) offline — all four production tokenizers are in the
  local HF cache. Results under "What I tried to break and could not".

## Findings

Each: file/line, failure scenario, confidence, severity.

---

### F40 — Nothing distinguishes a pre-ratification training-data build, so the mandated regeneration is unenforceable from the artifact — HIGH severity, HIGH confidence

`src/algoverse/data.py:342-351` writes the data manifest as
`{seed, n_per_dataset, n_incentive, n_no_stakes, md_deceptive,
mc_deceptive, validated, fold_system}`. Not one of those fields changes
between a build made before and after the 2026-08-14 ratification that
added `155,000` to `TRAIN_COMPANY_OFFERS` and moved `TRAIN_OUTSIDE_RATIOS`
to `0.55/0.73/0.81/0.94` — the ratification whose own words are "Training
data must be REGENERATED before any fine-tuning use; previously built files
on Drive are invalid."

`train.load_training_data` (`src/algoverse/train.py:233`) requires the
manifest and records it verbatim; `check_objective`
(`src/algoverse/train.py:343`) checks only counts; `dataset_sha256`
(`train.py:224`) is computed and guarded but nobody anywhere holds an
expected value for it.

**Failure scenario.** The operator points `--data` at the Drive copy built
2026-08-12. Records are well-formed, meta rows align, `md_deceptive == n//2`,
`fold_system` matches — every guard passes. M_D trains on data whose claimed
and derived offers collide with the eval grid. `tau(M_D)` then partly
measures memorization; the Gate-1 `tau_gain_min = 0.15` verdict, the entire
`A_l` sweep that selects `ℓ*`, and the paper's firewall claim (final-paper
delta #3) all inherit it. Nothing in `train_manifest.json`, in any
`train_meta.json`, or in any eval row records which grid produced the data.

**Blast radius.** `train_manifest.json` → every checkpoint sidecar → every
Gate-1 row → `metrics.tau_with_ci` / `tau_gain` → `metrics.bypass_effect`
→ layer selection → Stage 2. A wrong number here is silent at every hop.

**Cheap and stdlib-testable.** Either (a) have `data.py` record
`TRAIN_COMPANY_OFFERS` / `TRAIN_OUTSIDE_RATIOS` (or a digest of them) in the
manifest and have `load_training_data` compare against the live constants,
or (b) re-derive it inside the train lane with no data.py change at all:
`data.py:271` already puts the full `scenario` dict into every meta row, so
the firewall can be checked directly against `tasks.COMPANY_OFFERS` /
`tasks.TRUE_OUTSIDE_RATIOS` — exactly what `tests/test_data.py:36`
(`_eval_value_set`) already does for the constants. The plan calls
regeneration "an operational precondition… not work in this plan"
(`planning/train.md` NON-GOALS); that is the wrong call for a precondition
whose violation is invisible and whose check is ten lines.

---

### F41 — No fingerprint of the RENDERED/TOKENIZED training text; template or tokenizer drift is invisible to both the resume guard and `matched_training_identity` — HIGH severity, HIGH confidence (mechanism), MEDIUM (trigger)

`GUARDED_MANIFEST_FIELDS` (`src/algoverse/train.py:546`) covers the data
files' bytes, the config, `n_examples`, `total_steps`, `dtype`,
`device_type`. But the actual training sequences are produced by
`tokenizer.apply_chat_template` + the tokenizer (`train.py:404-414`), and
neither the chat template nor the transformers/tokenizers version is
guarded. Package versions are recorded (`train.py:628`) and the plan
explicitly lists them under "Excluded (legitimately differ)" for the
matched audit.

**Failure scenario A — inside one run.** M_D resumes in a fresh Colab
session after a `transformers` upgrade that ships a revised chat template.
`dataset_sha256`, `meta_sha256`, config, `n_examples`, `total_steps`,
`dtype`, `device_type` are all unchanged, so `_guard_train_manifest`
(`train.py:567`) and the resume identity hash (`train.py:591`) both pass.
Steps 0–140 trained on one rendering, 141–281 on another. Nothing records it.

**Failure scenario B — across arms.** M_D trains Monday, M_C trains Friday
after a version bump. `matched_training_identity(md) ==
matched_training_identity(mc)` returns True — certifying RESEARCH_SPEC's
binding "All fine-tuning uses matched data volumes, optimization settings,
checkpoint schedules, and random seeds" — while the two arms saw different
token streams. `tau(M_D) - tau(M_C)`, and every Stage-3 `R_t` built on the
same machinery, then carries an uncontrolled difference that the designated
executable home of the "matched" claim was built to exclude.

**What I verified about the trigger.** The exposure today is version drift,
not day-to-day drift: Llama-3.1's template hardcodes
`{%- set date_string = "26 Jul 2024" %}` (checked against the cached
tokenizer) rather than calling `strftime_now`, and neither Qwen2.5 nor
Gemma-2 embeds a date, so renders are currently reproducible across days.
But all three cached repos carry a `.no_exist/chat_template.jinja` marker —
HF is migrating templates out of `tokenizer_config.json` into a separate
file, which is precisely the kind of change that alters a render without
altering the model id.

**Cheap fix in scope.** After WP-T4 step 5, sha256 the encoded examples
(the concatenated `input_ids`/`labels`) and add it to
`GUARDED_MANIFEST_FIELDS` and to `matched_training_identity`. One field
closes template drift, tokenizer-version drift, and any future date
dependence at once.

---

### F42 — `matched_training_identity` still carries the F29 tuple-vs-list bug; F29's fix landed at one of the two sites — MEDIUM severity, HIGH confidence (demonstrated executably)

`_guard_train_manifest` (`src/algoverse/train.py:575`) JSON-round-trips the
current manifest before comparing, which is the F29 fix.
`matched_training_identity` (`src/algoverse/train.py:1117`) does not.

Demonstrated on the local stack:

```
matched_training_identity(_train_manifest(...))
  != matched_training_identity(json.loads(json.dumps(_train_manifest(...))))
→ True    # config["target_modules"] is ("q_proj",…) in one, ["q_proj",…] in the other
```

**Failure scenario.** The Stage-2/analysis step the plan assigns this audit
to compares a manifest built in-process (via `_train_manifest` or
`dataclasses.replace`) against one loaded from disk, and two genuinely
matched arms are reported as unmatched — or, worse, the harness "fixes" it
by loosening the comparison. `train_lora` happens to return a round-tripped
manifest on both branches (`train.py:927`, `train.py:930`), so the bug is
latent rather than live, but that is an undocumented precondition on the
single executable home of a spec-binding claim. The pure test
(`tests/test_train_pure.py:522`) round-trips every input, so it cannot see
this.

**Dependency-closure note.** Same root cause as F29, one site fixed, one
missed — worth checking whether any other consumer of `_guarded_view`
(`train.py:553`) can be reached with a live `TrainConfig` rather than a
loaded manifest. `_manifest_identity_sha` (`train.py:591`) round-trips
internally, so it is safe.

**Second half of the same finding.** `matched_training_identity` has **no
caller anywhere in the repo**, no comparison/assert helper, and no script
that runs it. The paper's "matched fine-tuning" sentence therefore has a
quantity home that is never invoked by any pipeline. The plan defers the
calling to Stage-2/analysis; that is defensible, but it should be recorded
as an explicit debt against a claim the paper will make, not left implicit.

---

### F43 — The default `TrainConfig` is never executed by any test: `gradient_checkpointing=True` is switched off in every guarded test, so the ratified non-reentrant rule and the Stage-2 "coexists with forward hooks" claim are asserted, not tested — and both pass on the cheapest rung — MEDIUM severity, HIGH confidence

`tests/test_train.py:162` sets `"gradient_checkpointing": False` in
`_config()`, and every guarded test builds on it — including WP-T6's
`test_trains_under_a_permanent_bypass` (`tests/test_train.py:527`), the one
test whose entire purpose is the Stage-2 non-preclusion claim.

What is consequently untested:

- RESEARCH_SPEC.md ratified 2026-08-13: "if gradient checkpointing is used,
  non-reentrant only (`use_reentrant=False`)" — `train.py:887-894` is the
  only implementation and no test enters it.
- `planning/train.md`'s Stage-2 non-preclusion checklist: "Gradient
  checkpointing is non-reentrant-only (ratified rule), **the mode that
  coexists with forward hooks**." That coexistence is the load-bearing claim
  for Stage 2 and is nowhere exercised.
- `model.enable_input_require_grads()` (`train.py:894`), which exists only
  for the checkpointing path.

**I ran it** on `~/.venvs/colab-local` with the existing fixtures:

| configuration | result |
|---|---|
| `gradient_checkpointing=False` (the tested path) | 8 steps, loss 4.8545 → 4.7782 |
| `gradient_checkpointing=True` (**the default**) | 8 steps, loss 4.8545 → 4.7782 (bit-identical) |
| `gradient_checkpointing=True` + `install_bypass(model, 1)` | trains; `bypass_state` → layer 1 survives; only blocks 0/2/3 move |

So the claim is true and costs one boolean in an existing test. Leaving it
unexercised means the first execution of the production configuration is a
paid T4 session, and a peft/transformers change that breaks
hooks-under-checkpointing — the exact interaction Stage 2 depends on —
surfaces there rather than here. This is the role's "work handed to the
human that an agent could in fact verify with a debug test" case.

---

### F44 — The unratified P12 converse ships as a hard refusal — MEDIUM severity, HIGH confidence

`check_fold_compatibility` (`src/algoverse/train.py:315`) refuses
`fold_required != fold_built` in **both** directions. Only the Gemma
direction is ratified (RESEARCH_SPEC.md, ratified 2026-08-14,
prompt-delivery bullet). The plan states the problem plainly ("the plan as
coded would enforce an unratified refusal from day one") and offers (a)
ratify as refusal, (b) strike, (c) warn-only until ratified — and the code
has taken (a) without a ruling.

The repo's own precedent points the other way: first-full-review F3's
ratified handling was "the unratified numeric defaults stay UNCHANGED in
code — no banner, no required flags", i.e. unratified positions sit as
inert defaults, not as enforcement.

**Failure scenario.** Low blast radius today — I verified that Qwen2.5 and
Llama-3.1 report `fold_required=False` and always take unfolded builds, so
the converse branch is unreachable in the planned pipeline. It bites only if
the team later rules (b) or (c), at which point shipped code has been
enforcing the opposite, or if some legitimate experiment wants folded data
on a system-accepting model. Governance, not numbers — but the role's brief
is that an unratified methodological rule going live is itself the finding.

---

### F45 — The `run_baseline.py` sidecar guard fails OPEN when the sidecar is absent — MEDIUM severity, HIGH confidence

`scripts/run_baseline.py:127` engages only when
`<adapter>/train_meta.json` is a file. `eval._adapter_digest`
(`src/algoverse/eval.py:68-93`) hashes only `adapter_model.safetensors` /
`adapter_model.bin` / `adapter_config.json`, so the sidecar's presence or
absence is invisible to the eval identity, to `gen_config.adapter_digest`,
and to resume.

**Failure scenario.** A checkpoint is copied to Drive, re-uploaded, or
synced by anything that keeps the two PEFT files and drops the sidecar (a
plausible Colab/Drive operation, and the plan's own workflow moves
checkpoints to Drive). `--checkpoint-step` is then never adopted and the
operator, trusting the new adoption behaviour, omits it. Rows record
`checkpoint_step: null` for a mid-training checkpoint.
`checkpoint_step` is in `metrics.RUN_KEY_FIELDS`
(`src/algoverse/metrics.py:538`), so that run becomes a mislabeled point on
the Stage-3 `R_t` x-axis — the "a transposed digit cannot silently move a
point on the R_t curve" hazard the guard was written for, now failing
silently instead of loudly.

**Credit where due, and a correction to my own first reading.** The
mid-run case is genuinely safe: `run_negotiation_eval`'s `expected_top_level`
row guard (`src/algoverse/eval.py:387-394`) *does* include `checkpoint_step`
and `train_seed`, so one `run_id` cannot mix two values — it refuses. The
plan's P15 claim that `train_seed` is "per-row resume-identity-guarded" is
therefore accurate. The gap is only the missing-sidecar path.

**Cheap fix.** When `--adapter` names a directory containing
`adapter_config.json` but no `train_meta.json` **and** `--checkpoint-step`
is omitted, print a loud warning (or require an explicit
`--external-adapter` acknowledgement).

---

### F46 — The P15 suspension is half-implemented: a `--train-seed` that MATCHES the sidecar is accepted and stamps a non-null `train_seed` onto a Stage-1 row — MEDIUM severity, HIGH confidence (mechanism), MEDIUM (trigger)

`scripts/run_baseline.py:150-160` refuses only a *mismatching*
`--train-seed`. RESEARCH_SPEC.md, ratified 2026-08-13: results-row
`train_seed` is "null for Stage-0/1 runs, the training seed for Stage-2
arms".

**Failure scenario.** The operator reads the sidecar, sees
`train_seed: 42`, and types `--train-seed 42` on the Gate-1 M_D evaluation
— the most natural thing to do given the new sidecar plumbing. It is
accepted. Those rows carry `train_seed: 42` while M_0's carry null.
`train_seed` is in `RUN_KEY_FIELDS` (`metrics.py:543`) and in the ratified
`summarize_runs` group key, so `gate1_report`'s M_0-vs-M_D comparison is
across two differently-keyed groups.

The plan reasons that the mismatch-refusal is "wrong under either P15
ruling" and stops there. That is true but incomplete: under ruling (b)
(keep null), *any* passed `--train-seed` on a Stage-1 run is wrong, matching
or not. The suspension blocks automatic adoption but not the manual
equivalent. (The `--train-seed` flag itself predates this pull; what this
pull adds is the impression that the field is now guarded.)

---

### F47 — Two homes for one derived quantity: `quant` / `four_bit` — LOW-MEDIUM severity, MEDIUM-HIGH confidence

`train._derive_quant` (`src/algoverse/train.py:524`) and
`eval._derive_gen_config`'s `four_bit` (`src/algoverse/eval.py:110`) derive
the same live-model fact by different rules: eval reads only
`is_loaded_in_4bit`; train adds a `config.quantization_config.load_in_4bit`
fallback.

**Failure scenario.** A model carrying a saved `quantization_config` but not
loaded quantized (or any transformers version where `is_loaded_in_4bit`
moves): train records `quant_label: "4bit"` in the manifest and in every
`train_meta.json`, while eval computes `four_bit=False` and then raises
`quant='4bit' contradicts live model four_bit=False` on the very checkpoint
train just certified. I confirmed the API is in motion: on transformers
5.15.0, `hasattr(model, "is_loaded_in_4bit")` is **False** for a plain model
— both implementations are leaning on a moving attribute, from different
sides.

This violates the plan's own "one home per reported quantity" principle and
the repo's derive-and-refuse discipline. The fix is to reuse eval's
derivation (it is already imported at module level for `_package_version`
and `_system_fold_needed`) rather than write a second one.

---

### F48 — The plan's local verification environment does not exist, and its recorded stack versions are wrong — MEDIUM severity, HIGH confidence (measured)

`planning/train.md` verification item 2 names "`.venv/bin/python`, 3.9,
torch 2.8.0 / transformers 4.57.6 / peft 0.17.1, MPS". In the tree today
`.venv` is **Python 3.12.13 with no torch**
(`ModuleNotFoundError: No module named 'torch'`). The only ML environment on
this machine is `~/.venvs/colab-local` — Python 3.11.15, **torch 2.13.0,
transformers 5.15.0, peft 0.20.0**.

Consequences:

- Verification item 2 is unexecutable as written; an implementer following
  the plan literally gets an ImportError and may accept the SKIP.
- The plan's risk note "**peft version skew** (0.17.1 local vs Colab
  latest): the used API surface … is stable across recent versions" analyses
  a version nobody runs. The real local skew is peft 0.20.0 / transformers
  **5.x**, a major-version jump, not the small one the note assumes.
- critique-1's assurance "**peft 0.17.1 API surface**: LoraConfig /
  get_peft_model / prepare_model_for_kbit_training(...) / get_/set_
  peft_model_state_dict all exist at the pinned local version" is stale.

I re-verified the API surface on the stack that actually exists: all 11
guarded tests pass under peft 0.20.0 / transformers 5.15.0, and
`get_peft_model`'s `autocast_adapter_dtype` default is still `True`
(see F50). The plan's environment prose should be corrected to the ladder in
the repo's agent instructions, not left naming a venv with no ML stack.

---

### F49 — No NaN / scaler-collapse abort; the fp16 deviation's only detector is a human watching Colab scroll by — MEDIUM severity, HIGH confidence

The loop records `scaler_skipped` per row (`src/algoverse/train.py:1042`)
and in each checkpoint sidecar (`train.py:1061`) but never acts on it, and
never checks the loss for finiteness (`train.py:1014`).

**Failure scenario.** On Gemma-2 — the family the plan itself flags as
least fp16-validated ("Gemma-2 fp16 safety, OPEN and unverified by this
plan") — the loss goes NaN at step 30. The loop runs to 281, writes six
checkpoints containing a garbage adapter, prints "session done: 282
optimizer steps", and exits 0. Gate-1 later fails, and that failure is
indistinguishable from a genuine "M_D did not become deceptive" result —
which the plan explicitly instructs the team to treat as "a reportable
result, not a training-lane bug". A methodological conclusion and a silent
numerics failure become the same observation.

**Cheap fix.** Raise on a non-finite group loss, and/or on N consecutive
`scaler_skipped` steps. Both are two lines and both are testable on the
tiny CPU fixture by injecting an inf.

---

### F50 — The fp32-master-weight property that makes fp16 AMP safe rests on an unstated, unrecorded peft default — LOW-MEDIUM severity, HIGH confidence (behaviour verified)

`get_peft_model` is called without `autocast_adapter_dtype`
(`src/algoverse/train.py:876`). Its default `True` is what keeps
`lora_A` / `lora_B` in fp32 on an fp16 base — verified locally: adapter
params come out `torch.float32` on a `.half()` base while the base
`q_proj.base_layer.weight` stays `torch.float16`. If that default ever
flips, AdamW runs on fp16 master weights under a GradScaler and updates at
lr 2e-4 silently underflow; the loss curve would look plausible and the
adapter would barely move.

The plan's own rule for `target_modules` — "ALWAYS passed explicitly, never
left to peft's default … omitting it would silently produce an
attention-only adapter while the manifest still recorded whatever P1 says"
— applies verbatim here, and this one is not recorded in the manifest at
all. (For the record, I confirmed the `target_modules` claim itself is true
on peft 0.20.0: `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` gives
`['q_proj', 'v_proj']` for `qwen2`, `llama`, `gemma`, and `gemma2`.)

---

### F51 — Partial micro-batches carry more per-example weight than the documented objective accounts for — LOW severity, HIGH confidence

`loss = model(**batch).loss / len(group)` (`src/algoverse/train.py:1009`,
`:1012`) weights each **micro-batch** equally. The docstring
(`train.py:812-818`) documents the conversation-level token-weighting
caveat and the partial final *group*, but not the partial final
*micro-batch*: whenever `n_examples % micro_batch_size != 0`, the last
micro-batch of **every epoch** holds fewer examples at full weight.

Not triggered by the proposed constants (1500 % 2 == 0; the dev vector
40 % 4 == 0), and matched across arms so it cannot bias `tau`. It becomes
live under any odd `n` or any `--config-json` micro-batch override. A
completeness gap in the objective documentation the plan deliberately made
load-bearing (F13), not a wrong number today.

---

### F52 — `TrainConfig` fields are unvalidated; `--config-json '{"save_every": 0}'` dies with ZeroDivisionError mid-loop — LOW severity, HIGH confidence

`src/algoverse/train.py:1070`: `(step + 1) % config.save_every`.
`save_every: 0` passes `run_finetune.py`'s known-key check
(`scripts/run_finetune.py:64-73`, which validates names only, not values)
and kills the run after the first optimizer step, losing the session.
`n_checkpoints`, `micro_batch_size`, `grad_accum_steps`, `epochs`,
`lora_r`, `max_seq_len` are equally unchecked; only `lr_schedule`
(`train.py:946`) and `checkpoint_spacing` (`train.py:182`) are validated,
and both only after the model is loaded. A `__post_init__` on the frozen
dataclass would move every one of these to the cheapest possible failure
point — before a 7B download.

---

### F53 — `model.config.use_cache = False` permanently mutates the caller's config and is never restored — LOW severity / style, HIGH confidence

`src/algoverse/train.py:895`. Harmless for `run_finetune.py`, which exits,
but a caller that trains then evaluates in-process (or a future Stage-2
orchestrator that does both) gets uncached generation with no indication
why. `generate_batch`'s analogous shared-state mutation (`eval.py:252`) is
the hazard D6 was written to defend against; the same discipline should
apply outward.

---

### F54 — `encode_preflight` on an empty record list raises `IndexError`, not a named `ValueError` — LOW severity / style, HIGH confidence

`src/algoverse/train.py:453-457`: `lengths[-1]` on an empty list. Only
reachable by direct call — `load_training_data` rejects empty files first
(`train.py:246`) — but `encode_preflight` is a documented public entry point
for the preflight, which is exactly the context where someone passes a
hand-sliced record list.

---

### F55 — `pad_token_id` can end up `None` and reach `torch.tensor` — LOW severity, MEDIUM confidence

`src/algoverse/train.py:900-902` falls back to `eos_token_id` but never
checks the result, and `_collate` (`train.py:502`) puts it straight into a
tensor. All three production tokenizers define at least one, so this is a
guard against a future family, not a live bug — but it is the same class of
"raise with a name instead of a TypeError" that D5's checks handle well
elsewhere.

---

### F56 — INTERFACES.md still lacks the train lane, while `run_baseline.py` now hard-imports it — LOW severity (process), HIGH confidence

WP-T8's proposed contract text is unapplied (correctly — INTERFACES.md is
human-owned and agents never edit it). The consequence to record so it is
not lost: today the binding contract contains no `train_lora`, no
`checkpoint_meta`, no schema for `train_manifest.json` /
`train_meta.json` / `train_log.jsonl` / `sessions.jsonl`, and no
`run_finetune.py` canonical command — while `scripts/run_baseline.py:34`
(the eval track's runner) now has a hard `from algoverse.train import
checkpoint_meta`. A cross-track dependency exists in code that the contract
does not record. This is a pending human edit, flagged, not an agent action.

---

### F57 — The one constant whose literature provenance is genuinely ambiguous is the one the code docstring is silent about — LOW severity, HIGH confidence

I re-fetched both cited papers rather than assessing them from memory.
Every provenance claim in `TrainConfig`'s docstring
(`src/algoverse/train.py:62-108`) checks out:

| claim | verified |
|---|---|
| LoRA §4.1: "random Gaussian initialization for A and zero for B, so ΔW=BA is zero at the beginning of training" | ✓ verbatim |
| LoRA: "we simply set α to the first r we try and do not tune it" | ✓ verbatim |
| LoRA §4.2: main experiments adapt W_q, W_v only | ✓ |
| LoRA lr 2e-4 for GPT-2 / GPT-3 | ✓ (Tables 11, 12) |
| QLoRA §4 / Fig 2: LoRA on all linear transformer-block layers required to match full finetuning | ✓ |
| QLoRA App. A: rank independent of final performance | ✓ |
| QLoRA 7B recipe: r=64, α=16, lr 2e-4, constant schedule, batch 16, max_grad_norm 0.3, β2=0.999 | ✓ (Table 9) |
| QLoRA compute dtype bfloat16; paged optimizer only at 33B/65B | ✓ — both declared deviations are honest |

And the plan's claim that the dropout guidance is internally inconsistent is
**correct**: Appendix A.1 says "LoRA dropout 0.05 is useful for small models
(7B, 13B), but not for larger models (33B, 65B)", while Table 9 (App. B.2)
assigns 0.1 to models up to 13B and 0.05 to 33B/65B. The plan records this
honestly under P2.

The finding is narrow: `TrainConfig`'s docstring gives provenance for
`target_modules`, `lora_r` / `lora_alpha`, `learning_rate` / `lr_schedule`,
`max_grad_norm` / betas, batch, `max_seq_len` and `gradient_checkpointing` —
and says **nothing** about `lora_dropout=0.05`, the single value whose
source is contested. The code is where the value lives; the ambiguity should
be visible there, not only in a planning document.

---

## Observations (pre-existing, outside this pull, found while checking integration)

**O1 — `tests/test_figures.py` cannot run in any environment in this repo.**
It does `from algoverse import figures, metrics` (`tests/test_figures.py:17`)
with no `sys.path` insert, unlike every other suite, and it requires
`pytest`. `algoverse` is not installed (`pip install -e .`) in
`~/.venvs/colab-local`, the only environment with the ML stack. So the
repo's "full suite" is green nowhere. Introduced by commit `773658e`, not by
this pull, but it means any claim of a passing full suite is currently
false.

**O2 — the repo's agent instructions are gitignored.** `.gitignore` excludes
`AGENTS.md`, `CLAUDE.md`, `roles/` and `Prompts.txt`, so
`git show fbe0f36:AGENTS.md` fails. The plan's F21 statement ("no AGENTS.md
exists in this repo") is therefore true of the *tracked* tree and false of
the working copy — and the plan restating the verification ladder inline is
the right response. Only the environment names inside it are wrong (F48).

## What I tried to break and could not

Stated as claims, per the role's "'Looks good' with no findings is a claim".

**The plan's mandatory tokenizer preflight (verification item 2), which this
pull does not report as executed.** All four production tokenizers are in
the local HF cache, so it runs offline with no downloads and no GPU. Over
the FULL 1500-record regenerated build (`build_finetune_datasets(n=1500,
seed=42)`, folded build for Gemma):

| model | dataset | n | max | p95 | mean | overflow @512 |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | m_d / m_c | 1500 | 177 / 173 | 171 / 167 | 161.4 / 159.4 | 0 / 0 |
| meta-llama/Llama-3.1-8B-Instruct | m_d / m_c | 1500 | 184 / 181 | 178 / 175 | 170.5 / 169.0 | 0 / 0 |
| google/gemma-2-9b-it (folded) | m_d / m_c | 1500 | 167 / 163 | 161 / 158 | 152.0 / 149.9 | 0 / 0 |
| Qwen/Qwen2.5-0.5B-Instruct | m_d / m_c | 1500 | 177 / 173 | 171 / 167 | 161.4 / 159.4 | 0 / 0 |

**Zero prefix violations on any family** — D5's string-prefix and id-prefix
masking assumptions hold against all three real chat templates. **P7 is no
longer unmeasured**: the proposed cap of 512 has ~2.8x headroom over the
longest real record (184 tokens), so it can be ratified on evidence.
Caveat: this measures the *current* builder's output and says nothing about
F40.

**The real-tokenizer fold guard (the plan's F38 check), executed locally.**
`_system_fold_needed` returns True only for `google/gemma-2-9b-it`
(Qwen2.5-7B, Qwen2.5-0.5B, Llama-3.1-8B all False). Gemma + unfolded →
raises "fold mismatch"; Gemma + folded → returns True; Qwen + folded →
raises (the P12 converse, F44); Qwen + unfolded → returns False. Exactly as
specified.

**Everything else I checked and found sound:**

- All 24 accepted findings from rounds 1–2 are genuinely applied, not just
  claimed. I spot-checked each against the code: F1 (seed before attach,
  with the test perturbing ambient RNG), F5 (stdlib `_read_jsonl`; utils
  imported inside functions), F7 / F30 / F31 (rmtree destination, sidecar
  inside tmp before publish, stale-tmp rmtree), F8 (resume identity sha over
  the full guarded set), F10 / F25 / F26 (the narrowed `run_baseline`
  guard), F11 / F12 (objective + alignment + meta digest), F13
  (`len(group)` division), F18 (checkpointing on both branches), F19
  (`_derive_quant`), F23 / F27 (`cross_family`, `model_id`), F24
  (`read_train_log`), F28 (`sessions.jsonl`), F29 (round-trip — at one site,
  see F42), F33 (`mask_prompt_tokens` in the signature), F34
  (`derive_total_steps`), F35 (unknown `--config-json` keys raise), F36
  (`resume=False` as an assertion), F37 (`created` in the sidecar).
- `checkpoint_schedule` (`train.py:138`): I re-derived every pinned vector
  by hand. "even" is duplicate-free for all `n <= total` because
  `ceil(kT/n)` is strictly increasing when `T >= n`; the doubling
  collapse-raise is correct; both spacings always contain `total_steps - 1`.
  `derive_total_steps` matches the worked example (1500 / 2 / 8 / 3 → 282,
  final group of 2).
- The guarded tests genuinely test what they name — I read the assertions,
  not the titles. Resume exactness (with dropout 0.1, so RNG restore is
  actually exercised), the step convention pin, the wholesale same-step
  rewrite including the planted stale `.bin`, base-parameter freezing,
  step-0 identity via zero-init B, the adapter round-trip through
  `PeftModel.from_pretrained`, and seed-governs-init with ambient-RNG
  perturbation all do what they claim.
- `bypass_state` resolves correctly through a `PeftModel` wrapper
  (`models._decoder_layers` uses `get_decoder()`, which PEFT forwards), so
  the Stage-2 seam's bookkeeping check works on a wrapped continuation
  model.
- `train_meta.json` inside the adapter directory does **not** perturb
  `eval._adapter_digest` (which hashes only the two weight filenames and
  `adapter_config.json`) or `PeftModel.from_pretrained`. Good design.
- The scaler-skip detection (`train.py:1019-1022`) is correct: `found_inf`
  is recorded during `unscale_`, before `clip_grad_norm_` turns an infinite
  norm into zeroed gradients, so `scaler.step` still skips, and the growth
  path can only move the scale upward.
- No RESEARCH_SPEC.md open decision is resolved by this pull. The
  checkpoint-step convention is *adopted* as the spec directs, not decided;
  the probe-bypass carve-out and the probe recipe are untouched; P12 and P15
  are escalated rather than settled (F44 and F46 are about how the code
  behaves while they are pending, not about resolution).

## Summary

18 findings (F40–F57) plus 2 pre-existing observations.
Severity: 2 high (F40, F41), 8 medium (F42–F49), 8 low/style (F50–F57).

The two high findings share a shape: the lane guards the *bytes* of its
inputs and the *values* of its config, but not the *provenance of the data
grid* (F40) or the *rendering that turns bytes into tokens* (F41) — and
both gaps are invisible to `matched_training_identity`, the designated
executable home of the spec's binding "matched fine-tuning" sentence.

---

## Planner disposition (round 3, 2026-08-15)

Adjudicated per the revision protocol before editing the plan. Counts:
**14 accepted** (3 with a corrected remedy), **0 rejected outright**,
**2 escalated**, **1 accepted-in-part + escalated**, **2 observations
recorded out of scope**. Every code claim in the critique was independently
re-verified against the working tree before adjudication, including the
executable ones (the `matched_training_identity` round-trip inequality, the
gradient-checkpointing runs, the peft target-module mapping, the adapter
dtype on an fp16 base, and `.venv` having no torch). The accepted findings
are applied in `planning/train.md` revision 4.

| Finding | Disposition | Reason / applied fix |
|---|---|---|
| F40 stale data build undetectable | **Accepted** | Verified: `data.py:342-351`'s manifest is byte-compatible across the 2026-08-14 grid change, and `_make_scenario` (`data.py:170-181`) records exactly the fields needed to detect it. This enforces a *ratified* decision, so it is an accept, not an escalation. Fix is a new named guard `check_training_grid(meta_rows, records)`, third alongside the two existing guards. Chose the critic's option (b) over (a): it needs no `data.py` edit (staying inside the plan's NON-GOAL "no training-data changes"), it fails **closed** on a stale build rather than depending on a manifest key that a stale build cannot have, and it verifies the file rather than a self-reported label. Verified `algoverse.data` and `algoverse.tasks` are stdlib-only at module level, so the lane's stdlib-importability invariant survives the import. |
| F41 no fingerprint of the rendered text | **Accepted, remedy CORRECTED** | The gap is real and high-severity. **The proposed remedy is wrong in its second half and would have broken every matched-arm audit**: M_D and M_C encode *different assistant replies*, so a digest of the encoded examples can never be equal across arms, and adding it to `matched_training_identity` would make `matched_training_identity(md) == matched_training_identity(mc)` permanently False — inverting the guard it was meant to strengthen. Split into two digests instead: `encoding_sha256` (data-dependent → run identity only) and `renderer_sha256` (a fixed synthetic probe → arm-comparable, family-varying, so it joins the within-family block and drops under `cross_family=True`). A probe *render* is used rather than `tokenizer.chat_template` because a missing attribute would silently degrade to the empty hash, whereas a probe render raises. |
| F42 `matched_training_identity` tuple-vs-list, and no caller | **Accepted** (both halves) | Executably demonstrated. Same root cause as F29, fixed at one of two sites. Apply the identical JSON round-trip inside `matched_training_identity`; add a pure test comparing a live `_train_manifest` result against its round-trip. Second half accepted as documentation: the un-called-audit debt is stated in "Quantity homes" rather than left implicit. |
| F43 default config never executed by any test | **Accepted** | Verified independently: `tests/test_train.py:162` disables gradient checkpointing in every guarded test including WP-T6's, so the ratified non-reentrant rule and the Stage-2 hook-coexistence claim have no executing test. Both work on rung 2 at negligible cost, which makes this the role's "assigned to the human but an agent could verify it" case. Two refinements: the loss-equality assertion uses a tolerance rather than `torch.equal`, because bit-identity across checkpoint recomputation is not a promise torch makes across versions; and the non-reentrant *mode* is pinned by capturing the kwargs through the file's existing spy pattern rather than by asserting on a private attribute. |
| F44 unratified P12 converse ships as a refusal | **ESCALATED** | Touches open pending decision P12; the role forbids resolving it. The finding contributes a genuinely new argument the human should rule with in view — first-full-review F3's ratified handling establishes that unratified positions sit as inert defaults, not enforcement — so P12's text gains that precedent plus the verified observation that the branch is currently unreachable for Qwen/Llama. Code behaviour unchanged pending the ruling. |
| F45 sidecar guard fails open | **Accepted** | Verified: `eval._adapter_digest` (`eval.py:68-93`) never sees `train_meta.json`, so a sidecar lost in a Drive copy is invisible to eval identity and resume. Fix is a loud warning, **not** a refusal or a required flag — the plan's contract "adapters without a sidecar (externally produced) behave exactly as today" is deliberate, and a refusal would break the A_l sweep's use of external adapters. |
| F46 `--train-seed` matching the sidecar still stamps a Stage-1 row | **Accepted in part + ESCALATED** | The mechanism is real and the critic is right that the suspension is only half-effective. But no *recorded value* is correct under both P15 rulings, so the behaviour cannot be fixed without the ruling → escalated, and P15's text gains the sharpened statement. What *is* correct under either ruling is surfacing: a loud warning naming P15 and the ratified null convention whenever `--train-seed` is passed alongside a sidecar. That half is accepted. |
| F47 two homes for `quant`/`four_bit` | **Accepted, direction pinned** | Fix is `eval._four_bit(model)`, one 2-line helper, called by both `_derive_gen_config` and `train._derive_quant`. The direction matters and is pinned: **train adopts eval's rule and drops its `config.quantization_config` fallback**, not the reverse — widening eval's rule would put `tests/test_bypass.py:810` (`test_derive_gen_config_independent_oracle`, which asserts `four_bit is False`) at risk for no benefit. Consequence recorded honestly: if `is_loaded_in_4bit` ever disappears, both lanes become wrong together instead of disagreeing, and the repair then has exactly one place to happen. Second deliberate cross-lane edit; `eval.py` leaves the module map's Untouched list. |
| F48 plan's verification environment does not exist | **Accepted** | Measured and confirmed: `.venv` is Python 3.12.13 with no torch; the ML stack lives in `~/.venvs/colab-local` (3.11.15 / torch 2.13.0 / transformers 5.15.0 / peft 0.20.0). Verification item 2 and the peft-skew risk note are corrected to the measured reality. The gap is a transformers **major** version, so the risk note's reassuring conclusion changes, not only its numbers. |
| F49 no NaN / scaler-collapse abort | **Accepted in part; abort criterion → new pending P16** | The surfacing gap is real. But "raise on a non-finite group loss" is **rejected as an immediate action**: transient non-finite loss under fp16 is exactly what `GradScaler` absorbs — grads go non-finite, the step is skipped, the scale halves, and the run usually recovers; an immediate raise would kill recoverable runs. Any defensible abort is a streak count or a scale floor, i.e. a number, and the repo's standing rule forbids the plan inventing one. Accepted: threshold-free surfacing (a loud line the moment a step is skipped or a loss is non-finite, plus a session-end summary so a human reading only the tail of a Colab log still sees it). The abort criterion is escalated as **P16** with a proposed value. |
| F50 `autocast_adapter_dtype` implicit | **Accepted** | Behaviour verified locally (fp32 adapters on an fp16 base). Passing peft's own default explicitly changes nothing today and is exactly the discipline the plan already mandates for `target_modules`. Refinement: rather than adding a `TrainConfig` knob — which would change the pinned field list in `tests/test_train_pure.py:589` and introduce a methodological-looking constant — record the **derived** `adapter_dtype` in the manifest, derive-don't-assert, consistent with `quant_label` and `dtype`. |
| F51 partial micro-batch weighting undocumented | **Accepted as documentation only** | The objective itself is not changed: re-weighting by example count would alter what "the deception-incentivizing objective" literally optimizes (pending P8's territory), and the current weighting is matched across arms so it cannot bias tau. One sentence added to the loop docstring, matching the finding's own framing. |
| F52 `TrainConfig` fields unvalidated | **Accepted** | `__post_init__` on the frozen dataclass, confined to *structural* validity (positive counts, dropout in [0,1), enum membership). No methodological bound is introduced, so nothing here needs ratification. Moves `--config-json` failures from mid-loop to construction time, before any model download. |
| F53 `use_cache` mutation never restored | **Accepted, minimal** | Low severity and no number is corrupted, but the fix is three lines and the repo already treats shared-state mutation (`generate_batch`'s `padding_side`) as a hazard worth isolating. Restore in a `finally`. |
| F54 `encode_preflight` IndexError on empty input | **Accepted** | One guard line. `encode_preflight` is a documented public entry point for the preflight, which is precisely where a hand-sliced record list arrives. |
| F55 `pad_token_id` can be `None` | **Accepted** | One raise after the eos fallback, matching D5's "raise with a name" discipline. A guard against a future family, not a live bug. |
| F56 INTERFACES.md lacks the train lane | **Accepted as a proposal update** | Agents never edit INTERFACES.md. WP-T8's proposed text is updated so the human applies **one** coherent block covering the fields revision 4 adds, plus the note that `run_baseline.py` now depends on `train.checkpoint_meta`. |
| F57 `lora_dropout` provenance missing from the code docstring | **Accepted** | Correct and cheap. Both papers were re-fetched and every other provenance line confirmed, including that the QLoRA dropout guidance really is self-contradictory (App. A.1 vs Table 9) — so the contested value is the one the code must not be silent about. |
| O1 `tests/test_figures.py` unrunnable | **Recorded, out of scope** | Genuine and independently reproduced (no `sys.path` insert, needs `pytest`, `algoverse` not installed in the ML venv). Pre-existing from `773658e` and owned by the figures track, not the train lane; folding it in would be the unrequested cross-lane expansion the repo's scope rule prohibits. Surfaced to the human as a standalone item so it is not lost, and named in revision 4's verification section so its failure is never read as a regression from this work. |
| O2 agent instructions are gitignored | **Recorded; reinforces F48** | `.gitignore` excludes `AGENTS.md`, `CLAUDE.md`, `roles/`, `Prompts.txt`. Confirms the plan is right to restate the test ladder inline and must stay self-contained for an implementer working from a clone. Only the environment names inside it were wrong (F48). |
