# Deadline timeline: results + rough draft by EOD 2026-08-18

Written 2026-08-16 (evening). Audience: the four team members. Everything
here is a schedule and a set of PROPOSALS — nothing below changes a
ratified constant until the team records a decision tonight.

**Honest odds: ~60–70%**, conditional on: the decision packet below being
ratified tonight, the A100 request going in tonight, Gate-1 passing on
the first try, and the Insider Trading fallback checkpoint being honored.

Compute reality (from the Algoverse policy): the team A100-40GB grant
(request queue — submit tonight, countdown starts at approval), L4
backups, and ~8 parallel free T4s pooled across 4 people (Kaggle 2×T4 +
30 GPU-hrs/wk each, Colab free, Lightning/Modal credits). Evals are
"inference" → run them on the free-tier fleet; training and the
time-critical blocks → the A100. Total remaining compute for the scoped
pipeline ≈ 70–80 T4-hours — compute is NOT the constraint; the A100
queue, decision latency, and Gate-1 are.

## Roles

- **P1 — A100 operator / training lead**: A100 request, training runs,
  Gate-1 evals.
- **P2 — Eval fleet operator**: Kaggle/Colab T4 sessions — baselines,
  DEV calibration, sweep chunks, R_t evals.
- **P3 — Code lead** (with the coding agent): sweep driver, Stage-2
  loader, corroboration driver, figures.
- **P4 — Insider Trading owner + writing lead**: IT design + build,
  paper skeleton and draft.

## Decision packet — ratify tonight (45 min, all four)

Proposals; the human meeting decides and records each in RESEARCH_SPEC.

1. **Tiered model scope.** All three families (Qwen2.5-7B, Llama-3.1-8B,
   Gemma-2-9B) run Stage 0–1 + Gate-1 for the draft (cheap: ~8 T4-h per
   extra family, unattended overnight). Only Qwen continues through
   sweep → Stage-2 → Stage-3 for the draft; Llama/Gemma downstream
   stages land between draft and final submission. Rationale: each
   extra family downstream = +32/+42 human-launched layer evals gated on
   its own Gate-1 + ~30 T4-h of Day-2 Stage-3 + 3× orchestration during
   draft-assembly hours.
2. **Activation patching = Tier-2 stretch.** Ratify the interp.jsonl
   `analysis` enum addition now, but implementation happens Day-1
   evening ONLY if the sweep driver, Stage-2 loader, and corroboration
   driver are all landed and green; runs go Day 2 in parallel; drops
   without ceremony if anything critical-path slips. Draft corroboration
   floor = probes (response-token aggregation) + attention JSD.
3. **T10 R_t subset (binding pre-commitment, required before any R_t):
   t ∈ {8, 70, 281}** — early/mid/final on the doubling grid. Full
   proposal wording in planning/stage2-3.md.
4. **Stage-3 per-layer causal re-analysis at t=281 only** (plus ~M_D;
   the δ-curve needs both). The spec fixes "every layer," not "every t."
5. **Seed-43 replication lapses by its own calendar policy** (needs the
   pipeline done by 2026-08-22; it will not be). Report the lapse; no
   new decision needed.
6. **Sweep-plan pendings** (planning/sweep-driver.md): P-S1 neutral-JSD
   metric recorded as `wikitext2_neutral_jsd` in competence.jsonl +
   per-layer ppl as ordinary `wikitext2_ppl` rows (INTERFACES addition);
   P-S2 per-layer MMLU/GSM8K on l*-candidates only; P-S3 reuse the
   full-pool M_D rows (restricted to the n=100 draw) as the sweep's
   intact baseline; P-S4 Stage-3 sweeps skip the permanently-lesioned
   layer, reported as structurally null; P-S5 authorize the
   `install_bypass(role=...)`/`bypass_state` contract edit (two-hook
   carve-out).
7. **Gate-1 failure lever, pre-agreed:** exactly one recorded-deviation
   rerun (proposal: epochs 3 → 4, everything else frozen). A second
   failure pivots the draft to what exists; no ad-hoc recipe iteration.
8. **Process compression:** single-round critiques / same-day
   ratifications for the remaining plans (Stage-2/3, insider-trading)
   until the 18th.
9. **Insider Trading stays full scope with P4 as owner**, subject to the
   hard 18:00 Aug-17 checkpoint below.

## Tonight (Aug 16, after the meeting)

- **P1**: submit the **A100 request (2-day grant)** immediately — queue
  latency is the top logistics risk. Verify the extractor key (Azure
  credits serve gpt-4o-mini-2024-07-18; the runner's startup probe
  fails fast if the key path is broken).
- **P2**: build + upload the folded Gemma dataset
  (`python scripts/build_finetune_data.py --fold-system ...`; only the
  unfolded variant is on Drive), then launch all three **M_0 baselines**
  overnight in parallel notebooks (canonical run_baseline.py command,
  `--llm-fallback --competence`, full selection pool). Refusal-row
  audit happens tomorrow morning (ratified item 7).
- **P1 + spare accounts**: launch the six **Stage-1 training arms**
  (M_D + M_C × 3 families) overnight on free T4s (~1–1.5 h each;
  Gemma uses the folded dataset — the T12 guard enforces it).
- **P3 + agent**: sweep driver WP-S1/WP-S4 (in progress tonight),
  WP-S2/S3 tomorrow morning.
- **P4**: read Scheurer et al. (arXiv 2311.07590) + released materials;
  draft the IT design per planning/insider-trading.md for 09:00
  ratification. Start the paper skeleton (methods text largely exists
  in RESEARCH_SPEC's Final-paper deltas).

## Aug 17 — Day 1

- **09:00 all-hands (30 min)**: ratify IT design; refusal audits;
  status.
- **Morning** — P1: M_D full-pool evals + competence for all three
  families (+ M_C evals, cheap) on A100 + fleet → gate1_report per
  family → **Gate-1 verdicts by midday. Qwen's verdict is the GO/NO-GO
  for everything downstream.** P2: DEV JSD calibration (Qwen-0.5B,
  free tier) → record confirm-or-revise on the 0.25-nat bound (item 16
  — sweeps cannot be scored without it). P3: sweep-driver rung-3 debug
  probe; chunked sweep notebooks ready. P4: implement IT env
  (pure-Python, tasks.py-style) + rung-1 tests.
- **Afternoon (post-Gate-1)**: **Qwen layer sweep**, 28 layers chunked
  across A100 + 4–6 T4 sessions (~20 min/layer on T4, ~4× faster on
  A100) → ~2–3 h wall. sweep_report → **l\* by ~17:00**. Then the
  full-pool confirmation run (M_D^-l*, n=305) and held-out negotiation
  transfer (final pool). P3: Stage-2 reinstall-at-load loader +
  corroboration driver.
- **18:00 — IT hard checkpoint**: code-complete + real-model smoke?
  YES → IT transfer evals run tonight. NO → recorded fallback: the
  draft ships with held-out-negotiation transfer only; IT lands between
  draft and final submission. No debate at 18:00 — the rule was agreed
  tonight.
- **Evening**: create ~M_D; **Stage-2 training, 4 arms** (A100
  sequential ~2–3 h, or 4 parallel T4s ~1.5 h wall). Overnight: launch
  Stage-3 R_t evals (t ∈ {8, 70, 281} × 4 arms = 12 held-out full-pool
  evals) on Kaggle background sessions + A100; corroboration captures
  on M_D. Agent: activation patching only if the Tier-2 condition holds.

## Aug 18 — Day 2

- **Morning**: finish R_t evals → **recovery verdict**. If recovery:
  the two **per-layer causal sweeps (M_281^{L,D} and ~M_D)** — the
  tightest block, ~4–6 h wall across A100 + fleet; must start by
  ~10:00. δ-curve, layer-k identification, reconstructed-vs-
  strengthened read from ~M_D.
- **Parallel**: IT evals if live; probe-AUROC + attention-JSD curves;
  patching runs if Tier-2 fired (dropped first if capacity is short).
- **Afternoon/evening**: freeze numbers; figures; assemble the rough
  draft (results; limitations: single-family downstream, single seed +
  lapsed replication policy, IT status; corroboration). **EOD: draft to
  mentor/PIs.**

## Contingencies

- **Gate-1 fails (Qwen)**: the pre-agreed single deviation rerun
  (~half day; still fits if triggered by midday). Second failure →
  draft pivots to M_D characterization + strongest-checkpoint sweep.
- **No viable layer (item-17 stop condition)**: the paper is the
  pre-registered negative result + corroboration — still a credible
  workshop submission, and all Stage-2/3 time is freed.
- **A100 not granted in time**: run everything on the pooled T4 fleet;
  wall-clock roughly doubles on training + sweeps but the date holds
  with aggressive chunking; L4 backups are the middle option.
- **fp16 instability (T16 abort)**: pre-committed warmup escalation
  (T3: warmup_steps → 10), recorded deviation, restart not resume.

## Standing rules (unchanged)

Full pools for Gate-1/transfer/R_t; n_boot=2000; benchmark limits
400/16; frozen recipe; append-only JSONL under results/ on Drive;
humans run every result-producing job; agents run pass/fail tests on
the cheapest rung.
