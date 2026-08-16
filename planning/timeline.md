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

1. **Tiered model scope — REVISED 2026-08-16 (pending final yes).** All
   three families run Stage 0–1 + Gate-1 AND the layer sweep + l*
   selection (+ Stage-2 training overnight, cheap) for the draft: three
   people each own one family's sweep in parallel on Day-1 afternoon
   (Qwen/Llama on Kaggle 2×T4 ≈ 4.5–5.5 h each; Gemma — 42 layers, 9B,
   the long pole — on the A100 ≈ 5 h), each gated on its own Gate-1.
   The draft then carries three-family Gate-1 AND three-family
   localization results. **Stage-3 for the draft stays Qwen-only**: R_t
   + two recovery sweeps per family ≈ 100+ T4-h concentrated in Day 2
   (vs ~96 theoretical fleet-hours with zero gaps) plus 3× analysis
   during draft-assembly hours — that is the part that genuinely does
   not fit. Llama/Gemma Stage-3 (checkpoints already on disk) lands
   between draft and final submission.
2. **Activation patching = Tier-2 stretch.** Ratify the interp.jsonl
   `analysis` enum addition now, but implementation happens Day-1
   evening ONLY if the sweep driver, Stage-2 loader, and corroboration
   driver are all landed and green; runs go Day 2 in parallel; drops
   without ceremony if anything critical-path slips. Draft corroboration
   floor = probes (response-token aggregation) + attention JSD.
3. **RATIFIED 2026-08-16** — T10 R_t subset: t ∈ {8, 70, 281}
   (recorded in RESEARCH_SPEC "Ratified decisions (2026-08-16)").
4. **RATIFIED 2026-08-16** — Stage-3 per-layer causal re-analysis at
   t=281 only (plus ~M_D).
5. **RATIFIED 2026-08-16** — seed-43 replication lapse acknowledged;
   reported per the outcome-independent policy.
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
8. **RATIFIED 2026-08-16** — process compression until the 18th.
9. **RATIFIED 2026-08-16** — Insider Trading full scope with P4 as
   owner, subject to the hard 18:00 Aug-17 checkpoint below.

ALL NINE ITEMS RATIFIED 2026-08-16 (items 1, 2, 6, 7 in the second
sitting: revised tiering accepted; patching Tier-2 rule; P-S1..P-S6 all
as proposed; Gate-1 lever = one epochs-3→4 rerun). Authoritative
record: RESEARCH_SPEC "Ratified decisions (2026-08-16, deadline
session)". The decision meeting's remaining agenda is only the IT
design review at 09:00 tomorrow.

## Tonight (Aug 16, after the meeting)

- **P1**: submit the **A100 request (2-day grant)** immediately — queue
  latency is the top logistics risk.
- **DONE 2026-08-16**: extractor verified end-to-end. Re-pinned to
  **gpt-5-mini** on Azure (see the RESEARCH_SPEC 2026-08-16 amendment).
  EVERY session that runs an eval must set BOTH env vars, from that
  environment's secrets store — `OPENAI_API_KEY` and
  `OPENAI_BASE_URL=https://desta.services.ai.azure.com/openai/v1/`
  (note the trailing `/openai/v1/`; without the base URL the call goes
  to api.openai.com and the startup probe fails). Training runs need
  neither.
- **DONE 2026-08-16**: folded Gemma dataset built and verified on Drive
  at `maheep-yksa/data/finetune-folded` (manifest `fold_system: true`,
  seed 42, n=1500). Gemma trains on THIS folder; Qwen/Llama on the
  unfolded one — the T12 guard refuses a mix-up.
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
- **Afternoon (post-Gate-1)**: **layer sweeps, one family per person**
  (per revised item 1): P1 — Gemma on the A100 (42 layers, ~5 h, the
  long pole); P2 — Qwen on Kaggle 2×T4 (~4.5–5 h); P4 or spare
  sessions — Llama (~5–5.5 h), each gated on its own Gate-1 verdict.
  sweep_report per family → **l\* per family by ~19:00**. Then per
  family: full-pool confirmation run (M_D^-l*, n=305) + held-out
  negotiation transfer (final pool) — evening, fleet-parallel. P3:
  Stage-2 reinstall-at-load loader + corroboration driver.
- **18:00 — IT hard checkpoint**: code-complete + real-model smoke?
  YES → IT transfer evals run tonight. NO → recorded fallback: the
  draft ships with held-out-negotiation transfer only; IT lands between
  draft and final submission. No debate at 18:00 — the rule was agreed
  tonight.
- **Evening/overnight**: create ~M_D per family; **Stage-2 training —
  12 arms across all three families** (~15 T4-h total, fleet + A100,
  unattended; checkpoints on disk for post-draft Stage-3). Overnight:
  launch QWEN's Stage-3 R_t evals (t ∈ {8, 70, 281} × 4 arms = 12
  held-out full-pool evals, ratified item 3) on Kaggle background
  sessions + A100; corroboration captures on M_D. Agent: activation
  patching only if the Tier-2 condition holds.

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
