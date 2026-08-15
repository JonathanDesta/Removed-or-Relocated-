# Current priority order (2026-08-15)

Each numbered priority below is written as a kickoff brief for a
roles/1-plan.md planner: the goal, where the existing code and decisions
live, what binds the plan, what must be flagged as pending rather than
resolved, and what verification looks like. Priority 2 is the exception —
it is an operational human-run, not a planning assignment.

Resolved and out of the queue: training data regeneration is VERIFIED
(Drive files in maheep-yksa/data are byte-identical to a fresh seed-42
build from post-ratification code); sweep bounds items 15-17 and the probe
recipe (response-token aggregation adopted) were RATIFIED 2026-08-15;
spec/bib prose edits are deferred to the paper and accumulate in the
"Final-paper deltas" list.

---

## 1. GPU-verification fix session (scope: gpu-verification-fixes)

**In plain terms:** last night's hardware test-drive found two broken
parts. One makes it impossible to measure the model's basic language
fluency (so no publishable baseline run can happen), and the other crashes
the tool that reads the model's attention patterns. Both must be fixed and
re-tested on a real GPU before anything else can move.

**Goal.** Repair the two product defects found by the executed
gpu-verification plan (record: planning/gpu-verification.record.md) so the
Gate-1 baseline and the attention-reading path become runnable.

**Defect C4 (high severity).** `load_wikitext_slice`
(src/algoverse/eval.py:866) calls
`load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`; the current
datasets/Hub stack rejects the un-namespaced repo id with `HfUriError`, so
zero perplexity/NLL values can be produced. This blocks the
`wikitext2_ppl` competence metric, which ratified constants item 14 makes
mandatory for publishable Gate-1 runs, and the same 20,000-token slice is
later reused by the item-16 neutral-JSD bound. The fix must preserve slice
semantics exactly (WikiText-2 test split, same line filtering/joining,
first 20,000 tokens, window 1024 / stride 512 downstream) and the
comparability discipline (same-model deltas; config recorded per row).

**Defect C5 (high severity).** The canonical 4-bit loader comes up with
`attn_implementation='sdpa'`; under transformers 5,
`output_attentions=True` returns an EMPTY TUPLE (not None), so the guard
in `_attention_all_layers_unchecked` (src/algoverse/interp.py:118-122)
dies with a bare `IndexError` instead of its intended
"reload with eager" diagnostic. Scope (decided by the human 2026-08-15):
the FULL fix — harden the guard (catch the empty-tuple case, clean
diagnostic naming sdpa as the cause, CPU unit test) AND build the
eager-attention pathway that makes attention reads actually runnable
(design constraints in the decisions section below).

**Decisions feeding this plan (both decided by the human 2026-08-15).**
- C5 scope — DECIDED: ALL of C5 is handled in this plan — the guard
  hardening AND the eager-attention pathway. Design constraint for the
  pathway: NOT a flag on the canonical loader — attn_implementation
  changes generation numerics and is part of gen_config identity, so a
  loader flag invites fragmented eval comparability. Attention reads are
  interp-only, so build a dedicated interp-side eager load/reload helper;
  if that adds a public cross-track capability, record it in
  INTERFACES.md as part of this plan (a team-visible addition made on
  this recorded decision). The guard remains necessary even with the
  pathway: the pathway adds a sanctioned route, it does not remove the
  failure mode — any model loaded under sdpa (the canonical loader's
  default) still hits the empty-tuple case, and the guard turns that
  into a clean diagnostic instead of a bare IndexError.
- A4 canary disposition — DECIDED by the human 2026-08-15: KEEP the
  guard architecture. Background: the transformers-5 canary shows
  `output_hidden_states` is now bypass-aware, so the original stale-
  activations hazard does not manifest on 5.x — but pre-hook capture via
  `residual_stream_by_layer` is correct under BOTH behaviors, the guards
  cost nothing numerically (they only refuse a convenience API), and
  versions verifiably drift across teammates and environments (Colab
  installed transformers 5.13.1 while rung 2 runs 5.15.0 — record B1),
  so version-conditional correctness invites silent regressions.
  Implementation falls to this fix session: re-pin the canary test to
  the new transformers-5 behavior (which also unblocks finding A3's
  banner acceptance) and reword the INTERFACES.md warning from "returns
  stale activations" to "version-dependent; the sanctioned path is
  version-robust" — an INTERFACES edit made on this recorded human
  decision.

**Verification.** Loader and guard changes are locally testable: rung-1/2
unit tests (the existing suites run in <1s; a rung-2 test with `datasets`
installed can execute the fixed loader against a real tokenizer). Final
acceptance is a rung-3 debug run BY THE IMPLEMENTER — these are pass/fail
diagnostics, not experiments (AGENTS.md division of labor): re-execute the
two failed work units from planning/gpu-verification.md — C4's six-value
perplexity ordering sanity check and C5's attention read on the canonical
Qwen 4-bit load — via the pinned `colab --auth=adc run --gpu T4`
invocation, nothing written under results/, sessions verified empty before
and after. Only then is priority 2 (a human experiment) attempted.

---

## 2. M_0 Gate-1 baseline run (operational — human, Colab; no plan needed)

**In plain terms:** the first real measurement of the untouched model —
how often it lies with and without temptation, plus its benchmark scores.
Every later result is compared against this. It just needs the fix above
to land first.

**What to run** (from INTERFACES.md canonical commands), on a fresh Colab
VM after the C4 fix is GPU-verified:

```
python scripts/run_baseline.py --model-id Qwen/Qwen2.5-7B-Instruct \
    --quant 4bit --split selection --n 305 --run-id m0-baseline \
    --out-dir results/m0-baseline --llm-fallback --competence
```

**Obligations attached to this run.**
- FULL selection pool (305), never a subsample — ratified item 1.
- `--competence` (MMLU 16/subtask, GSM8K 400, WikiText ppl) is mandatory
  for any publishable Gate-1 citation — ratified item 14.
- `--llm-fallback` must fail fast at startup if the pinned extractor
  (`gpt-4o-mini-2024-07-18`) cannot execute; resolved extractor recorded
  in gen_config.
- First-run refusal audit: a human reviews every row marked
  `invalid_reason: "refusal"` — ratified item 7.
- Results are append-only JSONL under results/<run_id>/ on Drive; resume
  by (run_id, scenario_id, condition); final pool (295) stays untouched.
- Note: this produces the M_0 half of Gate-1. The Gate-1 verdict itself
  (tau gain >= 0.15) is computed by scripts/gate1_report.py only after
  M_D exists (priority 3's output).

---

## 3. Training-track plan (scope: training)

**In plain terms:** the machinery that actually teaches the model to be
deceptive (and a matched honest twin), then later re-trains it after we
disable its "deception layer." Nothing downstream — the deceptive
checkpoint, the layer sweep, the recovery experiment — can happen until
this exists. It is the project's critical path.

**Goal.** Fill in src/algoverse/train.py (currently an 8-line stub) with
the LoRA fine-tuning loop used by Stage 1 (M_0 -> M_D under the
deception-incentivizing objective; M_C under control) and Stage 2 (the
four matched continuation arms M^{I,D}, M^{I,C}, M^{L,D}, M^{L,C} from
M_D and the permanently bypassed ~M_D), plus its Colab driver script.

**Existing anchors.**
- Data: the verified datasets in Drive maheep-yksa/data (chat-format
  JSONL, scorer-validated at build; manifest fold_system: false). Loader
  contracts and the eval-grid firewall are in src/algoverse/data.py.
- Model plumbing: `load_model_and_tokenizer`, `install_bypass`,
  `bypass_state` in src/algoverse/models.py — all GPU-verified, including
  bypass-through-PEFT (record C6: adapter load + bypass + exact removal).
- Checkpointing: utils.save_checkpoint/load_checkpoint pin the step
  convention — state["step"] is the last COMPLETED step; load returns
  step + 1 (first-full-review F22). The plan must adopt it.

**Binding ratified decisions.**
- Permanent bypass = the runtime hook re-installed at EVERY load, never
  weight surgery; checkpoint metadata records the bypassed layer and
  every loader re-installs (2026-08-13). This is what keeps LoRA from
  resurrecting the lesioned layer.
- Gradient checkpointing, if used: non-reentrant only
  (`use_reentrant=False`).
- Results/checkpoint rows carry `train_seed` (null for Stage-0/1 evals,
  the training seed for Stage-2 arms), stamped and resume-guarded.
- Matched arms: same data volumes, optimization settings, checkpoint
  schedule, and seeds across all four Stage-2 arms.
- fold_system refusal: a fold-requiring model (Gemma-2) must REFUSE to
  train on data whose manifest says fold_system: false. Corollary task in
  this plan's scope: build and upload the Gemma dataset variant with
  `scripts/build_finetune_data.py --fold-system` (only the false variant
  exists on Drive).
- Replication policy (pre-committed 2026-08-13): a second Stage-2 seed
  runs iff the single-seed pipeline completes by 2026-08-22. That date is
  ~1 week out and will likely lapse — the plan records the outcome
  either way; the criterion is calendar-based and outcome-independent.
- Gate-1 certification of M_D: tau gain >= 0.15 on the full selection
  pool, competence drops <= 5 points, ppl rise <= 2.0 (ratified items
  1-3), via scripts/gate1_report.py.

**Pending decisions to flag, not resolve.** The spec deliberately leaves
the training recipe open: LoRA config (rank/alpha/targets), optimizer and
schedule, epochs/steps, batch sizing, the checkpoint-save schedule (which
also defines Stage-3's measurement points t), and quantization mode
during training. Per AGENTS.md these are pending human decisions the plan
lists with a recommendation each — not defaults filled in silently.

**Constraints the recipe recommendations must satisfy (deliberately
citation-free — every literature-derived number comes from papers the
PLANNER fetches and reads; nothing here asserts a literature fact, per
the human's instruction 2026-08-15).**
- The four-arm design differences out recipe suboptimality (R_t is a
  ratio of differences under one shared recipe), so the recipe must be
  ADEQUATE and FROZEN, not optimal. It is squeezed from both sides:
  strong enough to pass Gate-1 (tau gain >= 0.15), gentle enough to
  respect the competence bounds (<= 5 pts, ppl rise <= 2.0). Prefer the
  gentlest recipe that passes Gate-1 — the failure asymmetry favors it:
  a too-weak run fails Gate-1 visibly and reruns as a recorded
  deviation; a too-strong run quietly breaches bounds and wastes the
  checkpoint.
- Adapter placement must not bias WHERE Stage-2 recovery can occur:
  restricting trainable modules restricts relocation sites and would
  contaminate the paper's central question. Whatever placement is
  recommended needs an explicit argument on this point.
- The quantization/load profile used in training must be reconciled with
  the ratified 4-bit NF4 eval profile and the GPU-verified
  adapter+bypass plumbing (record C6), and must keep trainable-parameter
  counts matched across arms (ratified).
- The checkpoint-save schedule is METHODOLOGICAL, not operational: it
  defines Stage-3's measurement points t. It must be pre-committed
  before any Stage-2 result exists, be identical across all four arms,
  and justify its density (e.g., against how fast the relearning
  literature reports recovery happening — read the spec's cited
  relearning papers, don't assume). Total Stage-3 eval cost scales as
  (#t x 4 arms x full pools, ratified item 10) — budget it explicitly.
- Sequence budget derives from measured facts, not guesses: rendered
  prompts are 184-198 tokens (record A6) and replies are capped at
  max_new_tokens=256.
- Loss masking: tau measures the reply policy; if the plan trains on
  prompt tokens it must argue why, especially since fold/no-fold data
  variants differ asymmetrically in their prompt text.
- Seed convention: data seed and sweep scenario seed are both 42;
  train_seed is stamped per the ratified schema, and the replication
  seed (if the 2026-08-22 trigger holds) must differ.
Papers the planner reads before recommending values (a reading list, not
a source of pre-asserted facts): the QLoRA and original LoRA papers for
method/rank/lr precedent, and the spec's cited relearning works
(lo2024relearn, ustaomeroglu2026blockem) for checkpoint density. Every
recommended number goes to the human as PROPOSED with its paper-verified
basis and reasoning class (math / experiment structure / literature).

**Verification.** Loop mechanics (resume, checkpoint convention, seed
stamping, fold refusal, bypass persistence through save/load) get rung-2
acceptance tests on tiny random models; a CUDA-only smoke (one real
LoRA step, 4-bit, bypass installed) is a rung-3 debug test; the actual
fine-tuning runs and Gate-1 numbers are human-executed in Colab — the
plan says exactly which quantities only the human run can produce.

---

## 4. Sweep-driver plan (scope: sweep-driver)

**In plain terms:** the scan that tests, layer by layer, how much each of
the model's ~30 layers contributes to its deception, so we can pick the
single most deception-critical layer to disable. The pass/fail thresholds
for that scan were ratified yesterday; one of them must first be sanity-
checked on a small throwaway model.

**Goal.** Build the Stage-1 layer-sweep driver: for every decoder layer l
(all layers including 0 and n-1 — ratified 2026-08-13), install a probe
bypass on M_D, evaluate both conditions on the n=100 sweep subsample
(scenario seed 42, recorded in the run manifest), compute
A_l = tau(M_D) - tau(M_D^-l), apply the ratified bounds, select l*, and
emit the Pareto frontier and layer-wise curves (plot functions exist in
src/algoverse/figures.py). Confirmation of A_l* on the FULL selection
pool and the transfer re-evaluation precede any Stage-2 entry.

**Binding constants (all ratified).** Sweep n=100; full pools for
anything that decides or publishes (item 10). Per-layer disqualifiers:
invalid rate > 0.20 per condition (item 15), mean per-token neutral JSD
> 0.25 nats (item 16), competence bounds (items 2-3). Selection gate:
A_l* >= 0.15 with 95% scenario-bootstrap CI excluding zero (item 17);
report A_l*/tau(M_D) alongside. If no layer passes, the finding is "no
viable layer-level localization" and Stage 2 does not run.

**New code this plan owns.**
- The neutral-JSD computation (item 16's operationalization): mean
  token-level JSD in nats between intact and bypassed models'
  next-token distributions over the standard WikiText-2 slice (same
  20,000 tokens / window 1024 / stride 512 as perplexity — shares the
  C4-fixed loader).
- The probe-vs-permanent bypass carve-out: `install_bypass` currently
  enforces a single bypass; Stage-3 sweeps on lesioned checkpoints need a
  probe bypass STACKED on the permanent one. Design the deliberate
  two-hook carve-out; a row's `bypassed_layer` records the probe while
  checkpoint identity carries the permanent lesion (Open-decisions item,
  pinned to this plan).
- MANDATORY FIRST STEP (ratified with item 16): run the per-layer JSD
  curve once on the DEV model (Qwen-0.5B — no selection consequence) and
  record a confirm-or-revise decision on the 0.25 number BEFORE any
  research-model sweep is scored.

**Dependencies and flags.** Requires priority 1 (C4 loader) and priority
3's M_D checkpoint. Rows go to results/<run_id>/rows.jsonl per the
INTERFACES schema (`bypassed_layer` records the probe layer). Sweep
evals use selection-pool scenarios only. Flag, don't resolve: none known
beyond what is ratified — if the plan finds a gap (e.g., how the
neutral-JSD number is recorded in competence.jsonl vs a new file),
escalate rather than improvise.

**Verification.** The A_l arithmetic, bounds logic, bootstrap CI, and
JSD math are rung-1/2 testable on synthetic rows and tiny models; the
stacked-hook carve-out gets rung-2 exactness tests (byte-identical
restoration, correct which-layer accounting); the actual sweep is a
human Colab run.

---

## 5. Corroboration-driver plan (scope: corroboration-driver)

**In plain terms:** three independent second opinions on the layer choice
— (a) small classifiers that read the model's internal activations to
detect lying, (b) a comparison of attention patterns between tempted and
untempted conditions, and (c) transplanting "honest" internal states into
the tempted model to see what changes. These corroborate the sweep's
answer but never decide it.

**Goal.** Build the driver that runs the spec's three localization-
corroboration analyses on M_D at every layer and writes
results/<run_id>/interp.jsonl (`analysis` in `probe_auroc |
attention_jsd`, one row per analysis x layer — schema landed in
INTERFACES.md 2026-08-14).

**Component briefs.**
- Probes: train on the Instructed-Pairs dataset (Azaria & Mitchell via
  zou2023representation), verify transfer to strategic deception
  including the ratified within-incentive-condition control (lied vs
  did-not-lie rows); per-layer probes report held-out AUROC on
  scenario-grouped splits (ratified metric), accuracy alongside.
  RATIFIED 2026-08-15: response-token aggregation per
  goldowskydill2025detecting (arXiv 2502.03407) REPLACES the current
  single last-token reading — probe features are read over response
  tokens and aggregated per response. The planner fetches and reads that
  paper (role-file requirement); `probe_layer`'s recipe (0.3 split,
  random_state 0, max_iter 1000, C=0.1 scaler+LR pipeline) is ratified
  as-is; its declared-deviation docstring line is superseded and updates
  here.
- Attention-JSD: the two-condition design is ratified (average-then-JSD,
  flat pooling, zero-extension to common support, JSD between
  per-condition summaries — implemented in interp.py). The
  eager-attention pathway is BUILT IN PRIORITY 1 (decided 2026-08-15);
  this plan consumes it and verifies the attention-JSD path end-to-end
  on the production loader — the rung-3 debug check that failed as C5.
- Activation patching: NOT IMPLEMENTED (interp.py header marks it TO
  BUILD): patch control-environment activations into the deceptive
  environment per layer; measure changes in deception, task competence,
  and response validity. Flag: interp.jsonl's `analysis` enum has no
  patching value — an INTERFACES.md addition the human must authorize.
- Discipline: on bypassed checkpoints, the bypassed layer's internals
  are excluded or NaN-flagged (ratified 2026-08-13); all reads respect
  the rendering contract (interp.py:19-22) and use
  `residual_stream_by_layer` on lesioned models — subject to the A4
  canary disposition from priority 1.

**Corroboration only:** none of these outputs feed layer selection.

**Verification.** Probe fitting, aggregation, splits, and JSD math are
rung-1/2 testable (sklearn on synthetic activations; tiny models);
attention capture on the production loader is a rung-3 debug test (the
one that failed as C5); real per-layer curves on M_D are human Colab
runs.

---

## 6. Insider Trading environment (scope: insider-trading)

**In plain terms:** a second, completely different test scenario — an AI
stock-trading assistant tempted to trade on insider information and lie
about why — used to prove the deception findings aren't just quirks of
our one negotiation setup. Everything today is built on the negotiation
game; the spec requires this second environment before the main
experiment's second stage.

**Goal.** Operationalize the Insider Trading environment
(Scheurer et al. 2023, arXiv 2311.07590) as an EVALUATION-ONLY
environment: it validates transfer of A_l* in Stage 1 (alongside
held-out Offer Negotiation scenarios) and computes R_t in Stage 3. It is
never used for fine-tuning or layer selection.

**What the planner must do.**
- Fetch and read the paper and its released prompts/materials (role-file
  requirement — do not work from the proposal's one-line summary). Known
  context: deception rates in that environment swing ~4% to ~97% across
  prompt variants, so prompts are FROZEN once chosen and conditions
  paired — the same discipline as the negotiation environment.
- Define the two paired conditions (deception-advantageous vs control)
  such that tau, and therefore A_l and R_t, are computable with the
  existing machinery in src/algoverse/metrics.py; rows conform to the
  INTERFACES results schema (scenario_id, scenario_params, split,
  condition; append-only; resume-keyed). Flag any schema strain (e.g.,
  split naming for a second environment) as an INTERFACES question
  rather than adapting silently.
- Scoring: the negotiation grader (structured final line, regex + pinned
  LLM fallback) does not transfer as-is; how deception is graded here —
  detection of misrepresentation in the model's report to its manager —
  is largely UNDEFINED. Per AGENTS.md, grader details are a pending
  human decision: the plan proposes a grading design and escalates it;
  it does not invent thresholds.
- Record the operationalization details for the paper (Final-paper
  deltas item 12).

**Pending decisions to flag.** Essentially everything quantitative:
scenario count and split sizes, grading rules, refusal/validity policy
for this environment, and whether Scheurer et al.'s prompt variants are
used verbatim or reduced. The plan proposes; the human ratifies.

**Verification.** Scenario construction, condition pairing, and grading
logic are rung-1 testable (pure Python, like tasks.py); generation
against real models is a rung-3 debug test; real transfer numbers are
human Colab runs.

---

## Standing constraints (apply to every priority)

- Publishable Gate-1 requires full pools and benchmark competence checks;
  the final pool (295 scenarios) stays untouched until headline numbers.
- Division of labor (AGENTS.md): result-producing experiments — training,
  benchmarks, sweeps, anything that could become a paper number — are run
  by a human in Colab. Pass/fail debug tests are the implementer's, on the
  cheapest rung that executes them (stdlib rung 1; local ML venv rung 2;
  rung-3 Colab T4 via the pinned `colab run` invocation only for genuinely
  CUDA-only paths). Anything unverifiable is said to be so.
- Results are append-only JSONL under results/<run_id>/, never ad-hoc
  files; every run resumes.
- One live plan per scope in planning/<scope>.md; critiques follow the
  Prompts.txt stage workflow.
