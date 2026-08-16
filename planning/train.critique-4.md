# train.critique-4 — implementation critique (round 4)

Scope: the working-tree diff applying `planning/train.md` revision 4 to the
landed Stage-1 lane — `src/algoverse/train.py`, `src/algoverse/eval.py`,
`scripts/run_baseline.py`, `tests/test_train.py`, `tests/test_train_pure.py`,
and the uncommitted `/tmp/algoverse_r4_t4_debug.py`. Reviewed against
`planning/train.md` rev 4, RESEARCH_SPEC.md (including the new "Stage-1/2
fine-tuning constants (ratified 2026-08-15)" block T1–T16), INTERFACES.md,
and the existing code's conventions. Findings continue from critique-3
(F40–F57).

## Environments that actually ran

- `python3` 3.14 — `tests/test_train_pure.py`: **24/24 PASS**;
  `python3 -c "import algoverse.train"` succeeds, so the stdlib-importability
  invariant survives the new module-level `from algoverse import data, tasks`
  (verification item 1 holds).
- `~/.venvs/colab-local/bin/python` (3.11.15, torch 2.13.0, transformers
  5.15.0, peft 0.20.0) — `tests/test_train.py`: **14/14 PASS**, no SKIP.
  Full-suite re-run (verification item 2c, mandatory this revision because
  of the `eval._four_bit` cross-lane edit): `test_bypass`, `test_data`,
  `test_eval_pure`, `test_interp`, `test_metrics`, `test_perplexity_count`,
  `test_scenarios`, `test_scoring`, `test_train_pure`, `test_wikitext_loader`
  **all PASS**. `tests/test_figures.py` fails on import with
  `ModuleNotFoundError: No module named 'algoverse'` — the pre-existing
  round-3 O1 defect, owned by the figures track, **not** a regression from
  this work.
- Four additional CPU probes were run in the same venv to test claims this
  critique makes; their outputs are quoted inline in F58, F59, F62 and F72.
- `colab --auth=adc sessions` → "No active sessions found on server."
  Nothing was left running. No GPU was used by this review.

## What the revision got right (so the findings are read in proportion)

The three ratified value changes landed correctly (`lora_r` 64,
`lora_dropout` 0.1, `lora_alpha` 16, verified against T2 and pinned by a new
test). `check_training_grid` genuinely does reject the pre-2026-08-14 build:
the historical ratios were `[None, 0.62, 0.68, 0.78, 0.88]` (commit
`628c68c`) and none of the derived true offers they produce lands in the
current allowed set. The `encoding_sha256` / `renderer_sha256` split is
implemented exactly as D10 specifies, including the deliberate exclusion of
`encoding_sha256` from `matched_training_identity` and the directional test
that keeps it out. F53's `use_cache` restore works (measured). And the
`renderer_digest` stability property D10 depends on holds against all three
real production tokenizers offline (Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, gemma-2-9b-it: digest identical across two loads).

## Findings

Each: file/line, failure scenario, confidence, severity.

---

### F58 — `train_lora` restores `use_cache` but leaves the caller's model in `train()` mode, so LoRA dropout stays live — HIGH severity, HIGH confidence

`src/algoverse/train.py:1032` sets `model.train()`; the revision-4 wrapper at
`src/algoverse/train.py:1281-1301` captures and restores only
`model.config.use_cache`. The module's training flag is never restored.

**Measured** on the tiny CPU fixture with the ratified `lora_dropout=0.1`:
after `train_lora` returns, `model.training is True`, and a `torch.no_grad()`
forward on the caller's own object differs from the same forward after
`model.eval()` by up to **9.17e-02** in logits, because the injected LoRA
dropout modules are still in training mode. (With `_config()`'s
`lora_dropout=0.0` the two agree, which is why no existing test sees it.)

**Failure scenario.** Any in-process train-then-evaluate on the same object —
which is precisely what D2's "a ready model object flows through identical
code" invites, and what Stage-2's four-arm orchestration will do — generates
from a dropout-perturbed model and writes the rows as an evaluation of the
trained checkpoint. `run_negotiation_eval` derives and refuses on bypass
state and on `four_bit`, but has no derive-and-refuse for training mode, so
nothing catches it; tau, the Gate-1 verdict, and every A_l/R_t number
downstream inherit a stochastically perturbed model. Today's
`scripts/run_finetune.py` exits after training and the eval lane reloads from
the checkpoint directory, so **no number is wrong yet** — this is latent, not
live.

**Why it belongs with F53.** This is the same shared-state class the plan
already treats as a hazard (D6's `padding_side`, F53's `use_cache`). The
revision fixed one of the three mutations and left the one with the larger
numeric consequence.

**Fix.** Capture `model.training` beside `previous_use_cache` and restore
both in the same `finally`. Extend
`test_use_cache_is_restored_and_missing_pad_ids_refuse`
(`tests/test_train.py:614`) to assert `model.training is False` afterwards,
running it with `lora_dropout=0.1` so a regression is visible rather than
masked by the zero-dropout test config.

---

### F59 — `enable_input_require_grads()` leaves forward hooks permanently installed on the caller's embeddings — LOW severity, HIGH confidence

`src/algoverse/train.py:1030`. **Measured**: with the ratified
`gradient_checkpointing=True`, `model.get_input_embeddings()._forward_hooks`
goes from 0 before `train_lora` to **2** after, and they are never removed.
They force the embedding output to require grad on every subsequent forward.

**Failure scenario.** Harmless under `torch.no_grad()` (the eval lane's
path), but any later gradient-taking pass on the same object — the interp
lane's probes, a future attribution run — silently builds a graph from the
embeddings it did not ask for, costing memory and, if anything ever inspects
`requires_grad` to decide behaviour, changing it. Same shared-state
discipline as F58; report together, fix together (`handle.remove()` in the
same `finally`, or accept and document).

---

### F60 — the missing-sidecar warning is gated on `--checkpoint-step`, but T15 made `train_seed` adoption load-bearing too, so F45's hazard survives through the other field — MEDIUM severity, HIGH confidence

`scripts/run_baseline.py:130-141`. The warning fires only when
`args.checkpoint_step is None`. Since T15, `train_seed` is adopted from the
sidecar as well (`run_baseline.py:167-172`), and `train_seed` is in
`metrics.RUN_KEY_FIELDS` (`metrics.py:532-544`) and in
`run_negotiation_eval`'s per-row `expected_top_level` (`eval.py:393-403`).

**Failure scenario.** A project-trained checkpoint is copied to Drive by
something that keeps only `adapter_model.safetensors` and
`adapter_config.json` — exactly F45's premise, and `eval._adapter_digest`
never hashes `train_meta.json`, so the loss is invisible. The operator knows
the step and passes `--checkpoint-step 281` but not `--train-seed`. **No
warning fires.** The row records `train_seed: null`, which under the
2026-08-15 amendment now means "no trained checkpoint was involved" — a false
statement about the artifact — and `summarize_runs` splits that run into a
different group from the same checkpoint evaluated with its sidecar intact.
This is the hazard the guard was built for, failing silently, reached through
the field the same ruling made adoptable.

**Fix.** Fire the warning when the sidecar is absent and **either**
`--checkpoint-step` **or** `--train-seed` was omitted, naming which fields
will be recorded as null.

---

### F61 — the two adopted sidecar fields are read with inconsistent strictness; a sidecar with no `train_seed` adopts `None` and prints it as a success — LOW severity, HIGH confidence

`scripts/run_baseline.py:157` reads `sidecar["checkpoint_step"]` (KeyError,
loud) while `run_baseline.py:168` reads `sidecar.get("train_seed")` (silent
`None`), then prints `TRAIN SEED adopted from train_meta.json: None`.

**Failure scenario.** A hand-made, truncated, or older-format sidecar yields
a null `train_seed` on a trained-checkpoint row with a log line that reads
like successful adoption. Under T15 that null is now meaningful and wrong.
**Fix:** `sidecar["train_seed"]`, matching `checkpoint_step`'s treatment. The
mismatch branch at `run_baseline.py:173` should keep `.get()` only if a
missing key is intended to be a mismatch, which it is not.

---

### F62 — the fp16→fp32 adapter-precision property (F50 / D10) is asserted on a model where it cannot fail — MEDIUM severity, HIGH confidence

`tests/test_train.py:338` asserts `meta["adapter_dtype"] == "torch.float32"`,
but `_tiny_model()` (`tests/test_train.py:61-78`) is an fp32 model, so the
adapter is fp32 whether or not `autocast_adapter_dtype=True` did anything.
The property D10 exists to protect is *"on an **fp16** base the adapter stays
fp32"* — the one that keeps AdamW off fp16 master weights under a GradScaler.
If peft's default flipped, or if the explicit kwarg at
`src/algoverse/train.py:1021` were dropped, every update at lr 2e-4 would
silently underflow while the loss curve still looked plausible, and this test
would still pass.

**Verified at rung 2 during this review**: `_tiny_model().half()` trained
through `train_lora` records `adapter_dtype: torch.float32` and
`dtype: torch.float16`. So the real property holds today **and is testable on
CPU in one line**; it simply is not tested. (The companion assertion at
`tests/test_train.py:266`, `attach_kwargs["autocast_adapter_dtype"] is True`,
pins that the kwarg is *passed*, not that it *works*.)

**Fix.** Add an fp16 variant: train `_tiny_model().half()` and assert the
manifest's `adapter_dtype` is `torch.float32` while `dtype` is
`torch.float16`.

---

### F63 — the entire grad-scaler branch, and therefore T16's wiring, is executed by no test in any environment — MEDIUM-HIGH severity, HIGH confidence

`src/algoverse/train.py:1176-1203` and `1215-1225`. On CPU `scaler is None`,
`scaler_skipped` is hard-coded `False` and `scaler_scale` is `None`, so
`scaler.unscale_`, the `get_scale() < scale_before` detection, the per-skip
`WARNING` line, the `scaler_skipped: true` value that reaches
`train_meta.json`, and the `_update_skip_streak` **call site** are all
unexecuted everywhere the suite runs.
`test_scaler_skip_streak_aborts_at_twenty_and_resets`
(`tests/test_train.py:645`) tests the pure helper in isolation — the plan's
own acceptance test (WP-T4 step 9) says *"drive the loop with a stubbed
scaler whose `get_scale` always shrinks"*, and that is not what shipped.

**Failure scenario.** A wrong argument at the call site — passing
`loss_nonfinite` instead of `scaler_skipped`, resetting the streak in the
wrong branch, reading the scale before rather than after `update()` — ships
green and is first observed on a rented T4, where the failure mode is either
a healthy run aborted at step 20 or a stalled run writing six checkpoints
from a dead adapter. Compounding it: revision 4 *removed* an item from the
Colab list (gradient checkpointing, correctly) but added **no** Colab item
covering the skip/abort path, so nothing at any rung executes it.

**Blast radius.** `scaler_skipped` is written into every scheduled
checkpoint's `train_meta.json` and into every `train_log.jsonl` row; the
plan's honesty claim ("a checkpoint written at a skipped step records
`scaler_skipped: true`; its adapter state equals the previous step's") rests
entirely on unexecuted code.

**Fix (rung 2, no GPU).** Give the loop a seam: factor the per-group scaler
bookkeeping into a small function
(`_apply_step(scaler, optimizer, trainable, max_grad_norm) -> (skipped, scale)`)
and unit-test it with a stub scaler, or let `train_lora` accept an injected
scaler for tests. Either makes the WP-T4 acceptance test as written
executable.

---

### F64 — none of the three data guards' invocation from `train_lora` is pinned; deleting any of the three call lines leaves all 38 tests green — MEDIUM severity, HIGH confidence

`src/algoverse/train.py:986-988` calls `check_fold_compatibility`,
`check_objective` and `check_training_grid`. Each has pure tests that call it
**directly**; every guarded test feeds `train_lora` through
`tests/test_train.py:_write_dataset`, which is always well-formed (and was
updated this revision to carry a valid `scenario`). No test asserts that
`train_lora` itself refuses bad data.

**Failure scenario.** A refactor, a merge, or a well-meant reordering drops
one of the three calls. Nothing fails. For `check_training_grid` this is the
sharpest case: it is revision 4's entire defence against the invalidated
pre-2026-08-14 Drive build (D9 / F40), the failure it prevents is a silently
wrong tau, and the line that invokes it is unpinned. `check_fold_compatibility`
is the same story for the ratified E6 rule and T12.

**Fix.** One guarded test that mutates the fixture three ways — an off-grid
`company_offer`, a control dataset passed as `objective="deceptive"`, a
`fold_system: true` manifest against the non-folding stub tokenizer — and
asserts `train_lora` raises each time, before the adapter attaches.

---

### F65 — `check_training_grid` re-implements `data.EVAL_VALUE_SET` instead of importing it, creating a second home for the train/eval firewall's value set — MEDIUM severity, HIGH confidence

`src/algoverse/train.py:415-420` builds
`set(tasks.COMPANY_OFFERS) | {int(round(offer * ratio, -3)) …}`.
`src/algoverse/data.py:76-81` already defines exactly that expression as
`EVAL_VALUE_SET`. **Verified equal today** (22 values, sets identical).

WP-T2 says the guard "reuses their constants as the single source of truth —
it never restates a grid value of its own". It restates the *derivation*.
That is the same defect the same revision fixed in the other direction by
extracting `eval._four_bit` (F47), and the plan's argument there applies
verbatim: "the plan's own 'one home per quantity' rule applies to derived
provenance as much as to reported numbers."

**Failure scenario.** The eval grid or its rounding moves (the comment at
`tasks.py:202-210` explicitly warns that the rounding is load-bearing for
scenario ids and must never change — but `COMPANY_OFFERS` or
`TRUE_OUTSIDE_RATIOS` could). `data.EVAL_VALUE_SET` moves with it, so
`_snap_off_eval_values` keeps the builder honest; `train.check_training_grid`
does not move, so the independent check that exists to catch a builder
failure silently starts checking a stale set. The guard's whole value is
being independent of the builder's *label*, not of the builder's *constants*.

**Fix.** `from algoverse.data import EVAL_VALUE_SET` — `data` is already a
module-level import and is stdlib-only, so the invariant is unaffected.

---

### F66 — the grid guard's test never constructs the artifact D9 exists to reject — LOW severity, HIGH confidence

`tests/test_train_pure.py:400` derives the "stale" `true_outside_offer` from
ratio **0.60**, which is an *eval* ratio (`tasks.TRUE_OUTSIDE_RATIOS`), not
one of the actual pre-2026-08-14 *training* ratios
`[None, 0.62, 0.68, 0.78, 0.88]` (commit `628c68c`). The assertion passes
either way, but the test does not demonstrate the claim D9 makes.

Two related gaps worth stating in the same place: (a) roughly one fifth of a
real stale build's rows carry `true_outside_offer: None` and pass every check
in `check_training_grid`, so rejection depends on reaching a non-None row —
certain at n=1500, but the property is undocumented and untested; (b)
`tests/test_train.py:135-140` gives every one of its 16 records the identical
scenario, so the per-row loop is exercised at exactly one point.

**Fix.** Build the fixture from the historical ratio list and assert the
raise names row 0; add one row with `true_outside_offer: None` to show it is
skipped rather than accepted-as-valid.

---

### F67 — verification item 2 instructs the implementer to confirm a warning the same plan says not to implement — LOW severity (plan hygiene), HIGH confidence

`planning/train.md:1503-1507` says the dev rehearsal "exercises the two new
`run_baseline.py` warnings (… pass `--train-seed 42` and confirm the P15
warning fires)". `planning/train.md:2025` (T15 code-delta row) says "The
revision-4 F46 warning is **withdrawn, not implemented**", and
`planning/train.md:2033-2035` says "do not implement the F46 warning".

The implementer followed the correct instruction — there is no P15 warning in
`run_baseline.py`, and there should not be. But verification item 2 is now
**unexecutable as written**, and it is the *only* acceptance test either
script has (neither has unit tests, by the repo's convention). A later
executor reading the Verification section in isolation will either add the
withdrawn warning back or record the rehearsal as failed. Since the plan is
human-owned I propose no edit; flagging so the line is corrected before the
rehearsal is run.

---

### F68 — WP-T8's INTERFACES.md addition still has not landed, so the binding contract does not mention the training lane at all — MEDIUM severity, HIGH confidence

`grep -n "train_lora" INTERFACES.md` → no match. F56 was accepted in round 3
"as a proposal update"; the proposal text exists at `planning/train.md`
WP-T8 but the human has not applied it.

AGENTS.md makes INTERFACES.md "the binding contract between the team's
tracks — schemas, signatures, eval constants. Code must match it." Right now
it is false that code matches the contract: `scripts/run_baseline.py:34`
(eval track) imports `algoverse.train.checkpoint_meta`, an undeclared
cross-track dependency; the checkpoint sidecar's key set (18 keys, three
added this revision) is documented only in a planning file; and the T15
amendment to `train_seed` semantics — which changes what an eval row means —
lives in RESEARCH_SPEC.md and nowhere in the contract. **Human action item,
not an implementer defect**; agents never edit INTERFACES.md.

---

### F69 — the T16 abort counts consecutive skips within a *session*, not within a *run* — LOW severity, HIGH confidence

`src/algoverse/train.py:1146` initialises `skip_streak = 0` at session entry,
and it is not part of the resume state saved at
`src/algoverse/train.py:799-834`. T16's ratified wording is "20 **consecutive**
grad-scaler skipped steps"; the code enforces "20 consecutive within one
invocation".

**Failure scenario.** A run stalls, accumulates 15 consecutive skips, hits
`max_steps_this_session` or a Colab disconnect, and resumes; the counter
restarts at 0. In practice the scaler state *is* restored from `resume.pt`,
so a genuine stall re-accumulates and aborts ~20 steps into the new session —
the criterion self-heals with a delay, which is why this is low severity
rather than a wrong number. But the code and the ratified sentence do not say
the same thing, and the plan's standing discipline is that a ratified
constant's implementation matches its statement. **Fix:** persist
`skip_streak` in `resume.pt` beside `step`, or state the session scoping
explicitly in the loop docstring and propose the same clarification for T16.

---

### F70 — T16's threshold lives in a keyword default rather than a named constant — LOW severity (style), HIGH confidence

`src/algoverse/train.py:906-907`, `limit=20`. Every other ratified constant
in this module is a `TrainConfig` field or a module-level name
(`FOLD_PROBE`, `RENDER_PROBE`, `OBJECTIVES`, `GUARDED_MANIFEST_FIELDS`). A
ratified number sitting in a function signature's default is the shape a
future reader is least likely to find when checking the code against T16, and
no test pins that the loop passes it (see F63). Suggest a module constant,
e.g. `MAX_CONSECUTIVE_SCALER_SKIPS = 20  # RESEARCH_SPEC.md T16`.

---

### F71 — `adapter_dtype` derivation raises a bare `StopIteration` — LOW severity, HIGH confidence

`src/algoverse/train.py:1055-1058`. If no parameter name contains `lora_`,
`next(...)` raises `StopIteration` with no message, which reads as an
interpreter error rather than a named refusal. Reachable if a future
`target_modules` value matches nothing on a family, or if a Stage-2
continuation model arrives with its adapters merged into the base. Every
sibling check in the same function raises a named `ValueError` naming the
offending thing; this one should too, and the message is a useful place to
restate the P1 trap (peft's default targets `q_proj`/`v_proj` only).

---

### F72 — the production configuration (`gradient_checkpointing=True` **and** `lora_dropout=0.1`) is exercised by no test, though each half is — LOW severity, HIGH confidence

`tests/test_train.py:164,173`: the shared `_config()` sets
`lora_dropout=0.0` and `gradient_checkpointing=False`.
`test_default_gradient_checkpointing_runs_and_is_non_reentrant`
(`tests/test_train.py:613`) inherits dropout 0.0;
`test_resume_is_exact_and_covers_every_step` (`tests/test_train.py:367`) sets
dropout 0.1 with checkpointing off. Nothing runs both.

That combination is where the interesting question lives: under
`use_reentrant=False`, torch re-runs the checkpointed forward, and whether
the dropout RNG stream is preserved across the recompute is a torch
implementation detail (`preserve_rng_state`), not a repo invariant — and the
project spans a transformers major version.

**Verified at rung 2 during this review that both properties hold today**:
gc-on vs gc-off loss trajectories with `lora_dropout=0.1` differ by
**0.0e+00**, and resume with `gradient_checkpointing=True, lora_dropout=0.1`
produces a **bit-identical** final adapter and an identical loss trajectory
to the uninterrupted run. So this is a coverage gap, not a live defect — but
the exact-resume claim that the Stage-3 R_t curve rests on is currently
unpinned in the configuration production will use.

**Fix.** One character: set `gradient_checkpointing=True` in
`test_resume_is_exact_and_covers_every_step`'s config. It costs nothing and
covers both halves at once.

---

### F73 — `matched_training_identity` still has no caller anywhere in the repo, and revision 4 added two fields to it — LOW severity here, HIGH confidence

Recorded as debt at `planning/train.md:523-529` and unchanged. Restated
because the situation got slightly worse, not better: T6 is now a **ratified**
claim, its executable home is still executed by no pipeline, and revision 4
added `adapter_dtype` and `renderer_sha256` to that home — two fields that
have therefore never been compared on a real pair of manifests. The owner is
the Stage-2/3 plan, correctly; the note here is so that plan inherits an
accurate statement of what has and has not been exercised.

---

### F74 — the T4 fit check that arbitrates T2's fallback is a hand-rolled replica of the training setup, on a different library stack, and its memory number can be a silent underestimate — MEDIUM severity, HIGH confidence on the mechanism / MEDIUM on whether it ran

`/tmp/algoverse_r4_t4_debug.py`. It correctly carries the mandated
four-line T4 guard and it never touches `results/` or Drive. Three problems
with it as evidence for a ratified constant:

1. **It reproduces the training setup by hand** rather than calling
   `load_model_and_tokenizer` + `train_lora`. The ordering it copies
   (`prepare_model_for_kbit_training` → `get_peft_model(r=64)` →
   `gradient_checkpointing_enable` → one fwd/bwd/step) matches
   `train.py:1004-1032` today, but the probe omits what the real loop does —
   `_collate`'s right-padding, `enable_input_require_grads`, and an
   8-micro-batch accumulation group — so any divergence between the replica
   and the real path is invisible, which is the same asserted-not-derived
   pattern the plan spent D2, D10 and F47 eliminating elsewhere.
2. **The peak-memory number can be an underestimate with no signal.**
   `torch.cuda.max_memory_allocated()` is read after `scaler.step(optimizer)`;
   if that first step is skipped by the scaler, AdamW's `exp_avg`/`exp_avg_sq`
   for the 161.5M fp32 adapter parameters (~1.3 GB) are never allocated. The
   script prints `scaler_skipped` but does not assert on it, so a skipped
   first step yields a comfortable-looking number for a configuration that
   does not fit. With `lora_B = 0` at init a skip is unlikely, but T2's
   fallback ordering is decided on this number and "unlikely" is not the bar
   the rest of this lane holds itself to.
3. **It pins `transformers==4.57.6, peft==0.18.1`**, neither the stack the
   local suite verified (5.15.0 / 0.20.0) nor necessarily Colab's default, so
   the measurement does not bound memory at the versions the run will use.

Also: the file lives in `/tmp` and is not committed, so whatever it measured
is recorded nowhere in the repo, contrary to verification item 3's "record
peak memory in the run notes". **Checked per AGENTS.md:**
`colab --auth=adc sessions` reports no active sessions, so nothing was left
billing.

**Fix.** Assert `not skipped` (or step twice and measure after the second),
drop or justify the version pins, print the peak alongside the T4's total,
and paste the printed line into the plan's run notes so the ratified fallback
has a recorded arbiter.

---

### F75 — `check_training_grid` assumes `scenario` is a dict and assumes its two arguments are aligned — LOW severity, HIGH confidence

`src/algoverse/train.py:430-431`. `zip(meta_rows, records)` silently
truncates to the shorter sequence — `load_training_data` enforces equality
(`train.py:299-303`), but `check_training_grid` is a public function that the
pure tests already call directly with hand-built lists, so the guard can
silently check a prefix. And `load_training_data:305-307` checks only that
the `scenario` **key exists**, not that it is a mapping, so a `scenario` of
`null` or a string reaches `scenario.get(...)` as an `AttributeError` rather
than the named `ValueError` every sibling check produces.

---

## Summary

| # | Finding | Severity | Confidence |
|---|---|---|---|
| F58 | `train()` mode leaks to the caller; LoRA dropout stays live | High | High |
| F63 | grad-scaler branch and T16 wiring executed by nothing | Med-High | High |
| F60 | missing-sidecar warning misses T15's `train_seed` adoption | Medium | High |
| F62 | fp16 adapter-precision property asserted where it cannot fail | Medium | High |
| F64 | the three data guards' invocation is unpinned by any test | Medium | High |
| F65 | `data.EVAL_VALUE_SET` re-implemented in `train.py` | Medium | High |
| F68 | INTERFACES.md still has no training lane (human action) | Medium | High |
| F74 | T4 fit check is a replica; peak memory can silently under-read | Medium | High/Med |
| F59 | input-require-grads hooks left on the caller's embeddings | Low | High |
| F61 | `train_seed` adopted with `.get()`, prints `None` as success | Low | High |
| F66 | grid-guard test does not use the real historical ratios | Low | High |
| F67 | verification item 2 asks for a warning the plan withdrew | Low | High |
| F69 | T16 streak is per-session, not per-run | Low | High |
| F70 | T16's `20` is a keyword default, not a named constant | Low (style) | High |
| F71 | `adapter_dtype` derivation raises bare `StopIteration` | Low | High |
| F72 | production gc+dropout combination untested (verified OK today) | Low | High |
| F73 | `matched_training_identity` still has no caller | Low | High |
| F75 | `check_training_grid` assumes dict scenario and aligned inputs | Low | High |

No finding resolves a pending decision from RESEARCH_SPEC.md: all sixteen
were ratified 2026-08-15 and the diff implements the ratified values
faithfully (T2's `lora_r=64`/`lora_dropout=0.1`, T12's both-directions
refusal, T15's adoption, T16's 20-skip abort all verified against the spec
text). The three carried-forward obligations (T2's fallback, T10's evaluated
subset, T3's warmup escalation) remain untouched, correctly.

---

## Implementer disposition (round 4, 2026-08-15)

Adjudicated per `roles/3-implement.md` before editing code. Counts:
**15 accepted, 0 rejected, 3 escalated**. No finding changes a ratified
methodological decision.

| Finding | Disposition | Reason / revision |
|---|---|---|
| F58 | **Accepted** | Restore the caller's original train/eval mode alongside `use_cache`; test with dropout 0.1 so leaked training mode is observable. |
| F59 | **Accepted** | Remove only input-gradient hooks installed by training and preserve pre-existing hooks. Eliminate the redundant explicit hook installation when `gradient_checkpointing_enable` already installs it, and restore the caller's prior gradient-checkpointing state as part of the same root-cause fix. |
| F60 | **Accepted** | A missing sidecar warns whenever either checkpoint step or training seed would remain null, naming the omitted fields. |
| F61 | **Accepted** | Read `train_seed` with strict indexing, matching `checkpoint_step`; malformed project sidecars fail loudly. |
| F62 | **Accepted** | Add an fp16-base CPU test proving adapters remain fp32, not merely that the PEFT keyword was passed. |
| F63 | **Accepted** | Factor optimizer/scaler application into a testable helper and add a loop-level injected-skip test covering warnings, logs, metadata, streak reset and abort wiring. |
| F64 | **Accepted** | Add guarded `train_lora` tests proving all three data guards are invoked before adapter attachment. |
| F65 | **Accepted** | Use `data.EVAL_VALUE_SET` as the single home of the evaluation-value set. |
| F66 | **Accepted** | Test the actual historical ratios `[None, 0.62, 0.68, 0.78, 0.88]`, including a leading `None` row followed by a stale non-null row. |
| F67 | **Escalated** | The contradictory verification prose is human-owned planning text. Do not restore the withdrawn P15 warning; the human must correct that plan paragraph. |
| F68 | **Escalated** | `INTERFACES.md` remains human-owned. Do not edit it; surface the exact WP-T8 addition and T15 semantics for human application. |
| F69 | **Accepted** | Persist the consecutive scaler-skip streak in resume state so T16 applies across sessions, not invocations. |
| F70 | **Accepted** | Introduce `MAX_CONSECUTIVE_SCALER_SKIPS = 20`, cited to T16, and use it from implementation and tests. |
| F71 | **Accepted** | Replace bare `StopIteration` with a named `ValueError` explaining that no LoRA parameter was found. |
| F72 | **Accepted** | Run exact-resume coverage with both gradient checkpointing and dropout 0.1 enabled. |
| F73 | **Escalated** | Enforcement belongs to the Stage-2/3 orchestration plan. Record that no production caller currently compares real manifest pairs. |
| F74 | **Accepted, factual correction** | The prior script did call `enable_input_require_grads`, but the broader finding survives: it replicated rather than invoked the lane, used a different stack, did not assert an applied update and left no durable evidence. Replace it with an exact-working-tree T4 probe and record its output below. |
| F75 | **Accepted** | Require equal list lengths and mapping-shaped scenarios, raising named `ValueError`s before iteration. |

Accepted findings are grouped for revision by root cause: shared caller
state (F58/F59), fp16/scaler/resume correctness (F62/F63/F69/F70/F72/F74),
sidecar provenance (F60/F61), data-guard enforcement (F64/F65/F66/F75),
and named adapter provenance failure (F71). F67, F68 and F73 remain
explicit human/downstream-owner action items and receive no code change in
this revision.

### F74 exact-lane T4 verification evidence (2026-08-15)

The temporary one-file payload embedded the exact working-tree
`src/algoverse` package (archive SHA-256
`4836df743e036f03866005e6facf8088ed9c9049fd711d0a8717aeb2c6708c3b`),
installed the locally verified transformers/PEFT/accelerate versions, and
called `load_model_and_tokenizer` plus `train_lora` directly. It used NF4,
rank 64, dropout 0.1, micro-batch 2 x accumulation 8, non-reentrant
checkpointing and sixteen valid 500-token conversations. The resume
optimizer state proved that an update was applied. No Drive mount or
`results/` write occurred. `colab --auth=adc sessions` reported no active
sessions before and after the run.

Command:

```text
colab --auth=adc run --gpu T4 --timeout 900 /tmp/algoverse_r4_exact_lane_t4.py
```

Exact terminal result:

```text
checkpoint written: step-00000
step 0/0 loss 2.6996
session done: 1 optimizer steps, next step 1 of 1
SESSION NUMERICS: 0 of 1 steps skipped, 0 non-finite losses
T4 EXACT-LANE PASS: device=Tesla T4 source_sha256=4836df743e036f03866005e6facf8088ed9c9049fd711d0a8717aeb2c6708c3b applied_updates=1 scaler_skipped=False loss=2.699641 token_range=500-500 peak_allocated_gib=11.593 peak_reserved_gib=13.191 total_gpu_gib=14.563
[colab] Stopping session 'run-ce1dcd'...
[colab] Session terminated.
```

That was the state on 2026-08-15. On 2026-08-16 the human authorized gated
credentials in Colab VMs. An initial Llama attempt through the now-abandoned
credential shim failed before model download because the generated payload
was syntactically malformed; it applied no update. After `AGENTS.md` was
updated, the shim was not used again: token-bearing one-run payloads were
sent through the stock CLI, excluded from CLI history, and removed at run
end. The final family outcomes are recorded in the round-5 disposition:
Llama r=64 passed one applied update; Gemma r=64 OOMed at 500 tokens and
activated T2's all-family r=16/dropout 0.05 fallback; the Gemma r=16
500-token stress rerun also OOMed during backward, while the corrected
167-token production-length r=16 probe applied one update successfully. No
Colab session remained.

### Human resolution of escalations (2026-08-15)

- **F67 resolved:** the human authorized correction of the stale
  verification paragraph. It now describes T15 adoption, the omitted-field
  missing-sidecar warnings and mismatch refusals; it no longer asks for the
  withdrawn P15 warning.
- **F68 resolved:** the human authorized the exact WP-T8 training contract
  addition to `INTERFACES.md`, including T15's `train_seed` semantics.
- **F73 deferred as not currently applicable:** repository-wide caller
  inspection confirms that the Stage-2/3 production orchestration workflow
  does not exist yet. Its future implementation must invoke
  `matched_training_identity` on real manifest pairs; no present caller can
  be corrected without inventing that future workflow here.
