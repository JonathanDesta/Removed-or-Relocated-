# Sweep-driver plan (scope: sweep-driver) — revision 1, 2026-08-16

Live plan for priorities.md §4. Written for an implementer who was not in
the planning conversation; every referenced decision lives in
RESEARCH_SPEC.md ("the spec") or INTERFACES.md ("the contract"). Anything
marked **PENDING** is a human decision this plan proposes but does not
resolve (AGENTS.md, the one rule that matters).

## Goal

Build the Stage-1 layer-sweep machinery: for every decoder layer l of M_D
(all layers including 0 and n−1 — ratified 2026-08-13), install a probe
bypass, evaluate both conditions on the n=100 sweep subsample (scenario
seed 42, recorded in the run manifest), compute A_l = tau(M_D) −
tau(M_D^−l) with a paired scenario bootstrap, apply the ratified
disqualifiers (spec items 15–17 plus items 2–3), select l*, and emit the
Pareto frontier and layer-wise curves. The same driver must later run
Stage-3 sweeps on permanently lesioned checkpoints (probe bypass stacked
on the permanent one), so the two-hook carve-out is designed here.

The actual sweep on a research model is a human Colab run. Everything in
this plan is driver code, pass/fail-testable on rungs 1–2, with one
rung-3 debug probe.

## Binding constants (all ratified; do not re-derive)

- Sweep set L = all decoder layers, including 0 and n−1 (28/32/42 for
  Qwen/Llama/Gemma per the contract).
- n=100 scenarios, scenario seed 42, selection split only; the draw is
  recorded in the run manifest (item 10). FULL selection pool (305) for
  the A_l* confirmation.
- Per-layer disqualifiers: invalid rate ≤ 0.20 per condition (item 15);
  mean per-token neutral JSD ≤ 0.25 nats over the standard WikiText-2
  slice (item 16 — with the mandatory DEV calibration below);
  competence bounds (items 2–3: competence drops ≤ 0.05, WikiText-2 ppl
  rise ≤ 2.0, same-model deltas only).
- Selection gate: A_l* ≥ 0.15 with the 95% scenario-bootstrap CI
  excluding zero (item 17); report A_l*/tau(M_D) alongside. If no layer
  within bounds passes, the finding is "no viable layer-level
  localization" and Stage 2 does not run.
- Bootstrap: scenario-level, n_boot=2000, alpha=0.05 (item 9) — already
  implemented in `metrics.bootstrap_ci` / `metrics.bypass_effect`
  (paired resampling).
- Generation profile: the canonical eval profile (4-bit NF4, greedy,
  max_new_tokens=256), identical to Gate-1 runs, so sweep rows are
  comparable under `figures.DEFAULT_MATCH_FIELDS`.
- MANDATORY FIRST STEP (item 16 calibration clause): before any
  research-model sweep is scored, the per-layer JSD curve runs once on
  the DEV model (Qwen-0.5B) and the human records a confirm-or-revise
  decision on the 0.25-nat number. The driver ships the mode; the human
  runs it (its output feeds a recorded decision).

## What already exists (reuse, do not rebuild)

- `models.install_bypass` / `bypass_state` — single-bypass hook with
  byte-identical removal, unit-tested (tests/test_bypass.py).
- `eval.run_negotiation_eval` — row production, append-only + resume by
  (run_id, scenario_id, condition), manifest sidecar, identity guards.
  `scripts/run_baseline.py --bypassed-layer N` already evaluates one
  bypassed layer per invocation.
- `eval.load_wikitext_slice` / `eval.compute_perplexity` — the pinned
  20,000-token slice, window 1024 / stride 512 (C4-fixed loader; the
  neutral-JSD pass reuses this exact slice and windowing).
- `metrics.bypass_effect(rows_base, rows_bypassed)` — A_l with paired
  scenario-bootstrap CI.
- `figures.split_base_and_sweep`, `figures.layer_curve`,
  `figures.index_competence`, `figures.pareto_points`,
  `figures.pareto_frontier`, `figures.curve_report` — the entire
  consumption side of the sweep, already tested against destroyed
  layers, unmeasurable layers, and mixed-comparison refusal.

## Design

### D1. Run layout — one run_id per layer

The contract's resume key is (run_id, scenario_id, condition). One
run_id holding every layer's rows would collide on that key, so each
swept layer gets its own run directory:

    results/sweep-<tag>-l<NN>/rows.jsonl      (probe bypass at layer NN)
    results/sweep-<tag>-base/rows.jsonl       (intact M_D, same draw —
                                               see PENDING P-S3)

`<tag>` names the swept checkpoint (e.g. `md-qwen7b-s42-step281`).
Rows carry `bypassed_layer: NN` per the contract; `arm` stays null for
Stage-1 sweeps (M_D is not a Stage-2 arm). The per-run manifest already
records scenario_seed and n. A sweep-level `sweep_manifest.json` in the
parent directory records: model_id, adapter path + checkpoint_step +
train_seed (from `train.checkpoint_meta`), the layer list, scenario
draw, and the recorded item-16 decision reference — written once,
guarded on resume like `train_manifest.json`.

### D2. The driver: scripts/run_sweep.py — load once, loop layers

A new script (in-process loop; it does NOT shell out to run_baseline.py,
but reuses the same library calls). Per session:

1. Load the model once (`load_model_and_tokenizer`, canonical profile),
   apply the adapter, validate the sidecar via `train.checkpoint_meta`.
2. For each layer l in the requested set (default: all; `--layers A-B`
   chunks a sweep across Colab sessions):
   a. `install_bypass(model, l)` (probe role — D4).
   b. `run_negotiation_eval` into `sweep-<tag>-l<NN>` (n=100, seed 42,
      LLM fallback enabled — publishable-run discipline).
   c. The neutral-distribution pass (D3) into the same run dir.
   d. `handle.remove()` — byte-identical restoration is already
      unit-tested; a `bypass_state(model) is None` assertion runs
      between layers anyway.
3. Every step resumes: finished rows are skipped by the existing resume
   machinery; a finished layer is skipped by checking its rows +
   neutral-pass records are complete before touching the model.

Rationale for load-once: a 7–9B 4-bit load is minutes; ~30 loads per
sweep would waste a large fraction of a Colab session. Install/remove
exactness is what test_bypass.py pins.

### D3. Neutral-distribution pass (new code, owned here)

Item 16's operationalization, computed in lockstep per window so full
vocab distributions are never materialized across the slice:

- Iterate the pinned WikiText-2 windows (identical slice, window 1024,
  stride 512, and scored-token accounting as `compute_perplexity` —
  factor the window iterator out of `compute_perplexity` rather than
  duplicating it).
- Per window: one forward pass intact, one with the probe bypass
  installed (install/remove around the bypassed pass, or run the pass
  inside the layer loop while the bypass is installed and cache the
  intact pass's per-window log-probs ONCE per sweep — implementer's
  choice; the intact pass does not depend on l, so computing it once
  and reusing it across all layers is the recommended shape).
- Per scored token: JSD between the two next-token distributions,
  computed in float32 from log-softmaxed logits, in nats (bounded by
  ln 2). Mean over the same scored-token set the perplexity recipe
  counts.
- The same bypassed-model forward yields the bypassed model's NLL over
  the pinned slice → per-layer WikiText-2 perplexity (item 12 recipe,
  per-token NLL cap 20) at zero extra forward cost. Item 3's per-layer
  ppl-rise disqualifier comes from this, compared against the intact
  model's ppl from the SAME pass structure (same-model deltas only).
- Recording: see PENDING P-S1 (metric enum addition to the contract).

### D4. Two-hook carve-out: permanent vs probe bypass (pinned here, per
the spec's open-decisions item)

Current rule: `install_bypass` refuses a second install. Stage-3 sweeps
on ~M_D-derived checkpoints need a probe bypass stacked on the permanent
lesion. Design:

- `install_bypass(model, layer_idx, role="probe")` gains a `role`
  parameter, `"probe"` (default — every existing caller keeps its
  meaning) or `"permanent"` (used only by the Stage-2 reinstall-at-load
  loader, another plan's deliverable).
- Stacking rules, enforced at install time: at most one bypass per
  role; a probe may stack on a permanent; two probes or two permanents
  refuse exactly as today; a probe targeting the SAME layer as the
  installed permanent refuses with a named error (see PENDING P-S4 for
  how the sweep records that layer).
- `bypass_state(model)` shape: returns None when intact; otherwise a
  dict now containing `{"permanent": marker|None, "probe":
  marker|None}` — a SHAPE CHANGE for existing consumers
  (`eval._derive_gen_config`, the eval bookkeeping cross-checks,
  test_bypass.py identity tests). The eval row rule follows the spec's
  open-decisions wording verbatim: a row's `bypassed_layer` records the
  PROBE layer; checkpoint identity (adapter_path / train_meta's
  `bypassed_layer`) carries the permanent lesion. `gen_config.
  bypass_impl` continues to record the implementation string.
- This changes a contract signature (`install_bypass` in INTERFACES.md
  §fine-tuning track). The edit is made only on the human's recorded
  authorization — PENDING P-S5. Implementation may land behind the old
  single-bypass behavior until then.

### D5. DEV calibration mode

`run_sweep.py --dev-calibration`: Qwen-0.5B (`models.DEV_MODEL`,
quant="none"), neutral-distribution pass ONLY (no negotiation rows —
the DEV sweep has no selection consequence), all 24 layers, output to
`results/dev-jsd-calibration/`. Cheap enough for a single short session.
The human runs it, then records confirm-or-revise on 0.25 nats in the
spec before any research-model sweep is scored. The driver REFUSES to
score (not to run) a research-model sweep report until the sweep
manifest carries a reference to that recorded decision — a deliberate
tripwire for the item-16 clause.

### D6. Selection report: scripts/sweep_report.py

Pure-analysis CLI (no ML stack), the sweep's `gate1_report` analogue:

- Loads the base run + every `sweep-<tag>-l*` run, feeds
  `figures.layer_curve` (A_l + CIs per layer, match-field checked),
  `figures.index_competence`, `figures.pareto_points` /
  `pareto_frontier` / `curve_report`.
- Applies the disqualifiers per layer: item 15 (invalid rate per
  condition, from the rows), items 2–3 (negotiation competence from
  control rows; ppl rise and neutral JSD from the D3 records; MMLU/
  GSM8K per PENDING P-S2), item 17 (A_l ≥ 0.15, CI excludes zero).
- Emits: the complete layer table (every layer, including disqualified
  and unmeasurable ones — nothing silently dropped), the Pareto
  frontier, the l* verdict WITH A_l*/tau(M_D), or the explicit "no
  viable layer-level localization" stop-condition verdict.
- Refuses to produce a verdict if any requested layer's rows are
  incomplete, if the DEV-calibration reference is missing (D5), or if
  match fields disagree across runs (figures.py already refuses mixed
  comparisons — surface its refusal, don't relax it).

### D7. Full-pool confirmation

After l* is selected: intact M_D full-pool rows already exist (Gate-1's
A3 run); the confirmation needs one M_D^−l* run at n=305 on the
selection split. `run_baseline.py --bypassed-layer <l*> --n 305` already
does this — the plan adds only a `sweep_report.py --confirm` mode that
computes full-pool A_l* with CI from those two runs and re-checks item
17. Transfer re-evaluation (Insider Trading + held-out negotiation) is
OUT OF SCOPE here — it belongs to the insider-trading scope and to
gate-level orchestration; this report only states whether the
confirmation step passed.

## PENDING decisions (proposed, not resolved — each needs the human)

- **P-S1. Where neutral-JSD and per-layer ppl records live.** The
  contract's competence.jsonl metric enum is `mmlu_acc |
  gsm8k_exact_match | wikitext2_ppl` — no JSD value. PROPOSAL: add
  metric `wikitext2_neutral_jsd` to competence.jsonl (value = mean
  per-token JSD in nats; config records slice/window/stride and the
  compared checkpoint identity), and record the bypassed model's ppl as
  ordinary `wikitext2_ppl` rows under the layer's run_id. This is an
  INTERFACES.md addition only the human authorizes (same precedent as
  the interp.jsonl schema, resolved 2026-08-14).
- **P-S2. Per-layer MMLU/GSM8K coverage.** Items 2–3 bind selection,
  but running both benchmarks on every layer costs ≈ the negotiation
  sweep again per layer (400 GSM8K generations + 912 MMLU items ×
  ~30 layers). Negotiation competence, invalid rate, ppl rise, and
  neutral JSD come from the sweep itself for EVERY layer. PROPOSAL
  (recommended): benchmarks run per-layer ONLY on layers that survive
  the cheap disqualifiers AND are l* candidates (top of the A_l ranking
  with CI excluding zero), always including the eventually-selected l*
  before the selection verdict is declared; the layer table records
  which layers carry benchmark values and which were never
  benchmark-eligible. Alternative (complete but ~2× sweep cost):
  benchmarks on every surviving layer. The choice changes what "within
  bounds" is certified for non-selected layers, so it is methodological
  and needs ratification.
- **P-S3. The sweep's intact-M_D baseline rows.** A_l needs tau(M_D) on
  the same n=100 draw. The full-pool M_D rows (Gate-1's A3 run) contain
  those 100 scenarios, generated under the identical deterministic
  profile — identical measurements. PROPOSAL (recommended): reuse them,
  restricted to the draw (metrics filtering by scenario_id; pairing is
  exact); fallback alternative: a dedicated `sweep-<tag>-base` run at
  n=100. Reuse saves a run and guarantees the baseline can never drift
  from Gate-1's; flagging because the spec does not pin it.
- **P-S4. Stage-3 sweeps: the permanently-lesioned layer itself.**
  Probing the lesioned layer is structurally a no-op (its output is
  already discarded), so A_l for that layer is 0 by construction, not
  by measurement. PROPOSAL: the driver skips it and the report prints a
  structurally-null entry (excluded/flagged, consistent with the
  ratified bypassed-internals rule) rather than spending 200
  generations measuring an identity. Needs a ruling because Stage-3's
  "every layer" sweep language would otherwise include it.
- **P-S5. Contract edits.** D4's `install_bypass(role=...)` signature +
  `bypass_state` shape, and P-S1's metric enum — both are INTERFACES.md
  changes; agents never edit the contract without a recorded human
  decision.
- **P-S6 (added during WP-S4 implementation). Reference model for the
  per-layer bounds.** Items 2-3 were ratified as "vs M_0" for Gate-1;
  the sweep context never pins whether a bypassed layer's
  MMLU/GSM8K/ppl bound compares against M_0 or against the intact swept
  checkpoint (M_D). The implemented default: benchmarks + ppl vs the
  INTACT SWEPT CHECKPOINT (same-model deltas, item 3's own wording);
  negotiation competence vs M_0 (item 2's wording, and M_0's value is
  already on hand from Gate-1). Both references are plumbed, so ruling
  the other way is a one-argument change, not a rebuild. Needs a
  recorded decision before the first scored sweep.

## Work packages

- **WP-S1. Neutral-distribution pass.** Factor the window iterator out
  of `compute_perplexity` (no behavior change to perplexity — its
  existing tests must pass untouched); implement lockstep JSD + NLL per
  D3. Tests: rung 1 — JSD math on synthetic distributions (known
  values, symmetry, ln 2 bound, zero for identical inputs); rung 2 —
  tiny model: JSD ≈ 0 intact-vs-intact, > 0 with a bypass installed,
  and the factored iterator reproduces `compute_perplexity`'s exact
  scored-token accounting.
- **WP-S2. Two-hook carve-out** per D4, behind the human's P-S5
  authorization (until then: implement with `role` defaulting to
  today's behavior and the permanent role refusing to install, so
  nothing user-visible changes). Tests: rung 2 — stacking matrix
  (probe-on-permanent allowed, same-layer refused, double-probe /
  double-permanent refused), byte-identical restoration after removing
  each hook in both orders, `bypass_state` shape, eval bookkeeping
  records probe layer in `bypassed_layer`.
- **WP-S3. scripts/run_sweep.py** per D2 + D5: layer loop, per-layer
  run_ids, sweep manifest with resume guard, `--layers` chunking,
  `--dev-calibration`, completeness checks, `bypass_state is None`
  assertions between layers. Tests: rung 2 — two-layer sweep of a tiny
  model end-to-end on CPU (rows appear under per-layer run_ids,
  manifest guarded, resume skips finished layers, base rows reused per
  P-S3's resolution).
- **WP-S4. scripts/sweep_report.py** per D6 + D7. Tests: rung 1 —
  synthetic rows through the full disqualifier + verdict logic: a
  passing layer, an item-15 kill, an item-16 kill, a CI-includes-zero
  kill, the no-viable-layer verdict, the missing-DEV-decision refusal,
  and the complete-table (nothing-dropped) property.
- **WP-S5. Rung-3 debug probe** (implementer, pinned `colab --auth=adc
  run --gpu T4` invocation, nothing under results/, sessions verified
  empty before/after): one layer of the DEV model through the real
  driver path on CUDA — install, 4-bit load path, one negotiation
  batch, one JSD window — pass/fail only.

Suggested order: WP-S1 → WP-S4 (both testable immediately, and P-S1/P-S2
answers slot in), WP-S2 in parallel (blocked only on P-S5 for the
contract edit), then WP-S3, then WP-S5.

## Verification

- Rungs per AGENTS.md: all math and orchestration logic at rungs 1–2
  (stdlib + tiny CPU models); exactly one rung-3 pass/fail probe
  (WP-S5). The DEV calibration run and every research-model sweep are
  HUMAN Colab runs — the driver's job is to make them one command each.
- Session budget note for the human (not a decision): per layer ≈ 200
  generations (n=100 × 2 conditions, ≤256 new tokens) + ~39 lockstep
  window pairs; ~29 layers for Qwen. `--layers` chunking plus row-level
  resume is the session-boundary strategy, same as training's
  `max_steps_this_session`.

## Out of scope

Stage-2 reinstall-at-load loader (owns the `role="permanent"` caller),
transfer re-evaluation environments, the corroboration analyses
(probes / attention JSD / activation patching), and any change to
ratified constants.
