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
   localization results. **Stage-3: AMENDED 2026-08-16 — all three
   families run in parallel, one owner + dedicated workers each** (see
   the Aug-18 section). The earlier "Qwen-only for the draft" wording
   assumed the three sweeps would share one GPU pool; they do not have
   to, so the scope limit is replaced by an ordering rule (Qwen first
   and watched to completion) plus a cheap fallback (an unfinished
   family lands post-draft, eval-only — its Stage-2 checkpoints are
   trained either way).
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

- **AWS FIRST, then the A100 request** (corrected 2026-08-16): the A100
  form has a REQUIRED field "Your AWS $200 credit status (GPU compute
  should use your AWS credit first)", so the AWS step is a prerequisite,
  not an optional errand. ~20-30 min, and it costs no real time — the
  A100 form queues for a human admin who reviews in the morning either
  way. Steps: new AWS account on the **Free plan** (personal email) →
  budget alerts at $50/$100/$150 (also one of the five $20 onboarding
  activities) → remaining onboarding activities → **quota request for
  `ml.g6e.xlarge` (1x L40S, 48 GB) in `us-east-1`** (L4/L40S are
  approvable on new accounts; A100/H100 generally are not).
  **Shortcut**: `edward-lcl/algoverse-aws`'s `setup.sh` does the budget
  alerts AND submits the quota request in one command — pick L40S,
  `us-east-1`, and run `--dry-run` first. Account creation and the IAM
  access key are still manual browser steps, and the script is not
  fully idempotent (a second run can fail on budget/role creation).
  Link only: do NOT clone that repo into this one — it is one-time
  infra tooling for one machine, and its SageMaker templates encode a
  workflow we are deliberately not adopting during the crunch.
- **Do NOT spend deadline time farming the extra $100.** Projected AWS
  usage is ~16-20 L40S-hours (~$30-40); even running the whole project
  on AWS would be ~$40-50, so the automatic $100 sign-up credit covers
  it. Skip the $20 onboarding activities except the budget one (free —
  `setup.sh` does it anyway) and, if someone has spare minutes, the
  EC2-launch and Bedrock-call ones (~2-5 min each). Avoid the RDS
  activity: fiddly, and a running RDS instance bills. The real spend
  risk is an IDLE instance, not planned usage — an L40S left running
  over a weekend is ~$90, more than the entire projected budget. Set
  the alerts and stop instances when done.
- **P1**: then submit the **A100 request (2-day grant)**, answering the
  credit-status field truthfully: account created and quota requested
  tonight; AWS documents 1-2 day quota approval, which lands after the
  2026-08-18 deliverable; A100 needed for the time-critical Gemma-2-9B
  42-layer sweep; AWS credits will carry post-draft Llama/Gemma Stage-3.
  Do NOT claim the credits are exhausted — the true answer is stronger.
- **AWS usage rule for this project**: if L40S quota clears in time,
  treat it as capacity for POST-DRAFT work (Llama/Gemma Stage-3,
  replication). Do not port the pipeline to SageMaker/EC2 during the
  crunch — the code assumes the Colab/Kaggle + Drive environment, and a
  new environment is a fresh debugging surface at the worst moment.
  Bedrock is irrelevant to the core experiment (layer bypass and
  residual/attention reads require self-hosted weights — the compute
  policy's own "you need logits/internals" carve-out); the extractor
  need it could have served is already met by Azure.
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
- **P2**: launch all three **M_0 baselines** overnight in parallel
  notebooks (canonical run_baseline.py command, `--llm-fallback
  --competence`, full selection pool). Refusal-row audit happens
  tomorrow morning (ratified item 7).
- **P1 + spare accounts**: launch the six **Stage-1 training arms**
  (M_D + M_C × 3 families) overnight on free T4s (~1–1.5 h each;
  Gemma uses the folded dataset — the T12 guard enforces it).
- **DONE 2026-08-16 (P3 + agent)**: the whole sweep-driver stack landed
  and is green on both test rungs — `scripts/run_sweep.py` (layer loop,
  neutral-JSD/ppl recording, guarded manifest, `--layers` chunking,
  `--dev-calibration`), `scripts/sweep_report.py` (disqualifiers,
  Pareto, l* verdict, `--confirm`), the two-hook probe/permanent
  carve-out, and the item-16 tripwire. Remaining agent work is the
  Stage-2 loader and the corroboration driver (Aug-17 morning).
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
  unattended). Overnight: launch Stage-3 R_t evals for **ALL THREE
  families** (t ∈ {8, 70, 281} × 4 arms = 12 held-out full-pool evals
  per family, ratified item 3) — ~36 T4-h total, which parallelizes
  cleanly across the overnight fleet + A100; plus corroboration
  captures on M_D. Agent: activation patching only if the Tier-2
  condition holds.

## Aug 18 — Day 2

- **Morning**: finish R_t evals → **recovery verdict per family**. If
  recovery: the two **per-layer causal sweeps (M_281^{L,D} and ~M_D)**
  per family — the tightest block, and it runs THREE WAYS IN PARALLEL,
  **one owner + dedicated workers each** (amended 2026-08-16; nothing
  forces them onto a shared pool):
  - **Gemma** ~35 T4-h → A100 + that owner's T4 sessions + AWS L40S if
    quota landed (the long pole: 42 layers, 9B).
  - **Llama** ~21 T4-h → one owner's Kaggle 2×T4 + Colab.
  - **Qwen** ~19 T4-h → one owner's Kaggle 2×T4 + Colab. **Launch
    Qwen's first and watch it to completion** — it is the verified
    end-to-end path, so one complete family by early afternoon means a
    later failure costs a bonus, not the draft. The other two start
    alongside it, not after.
  - **P4 keeps writing.** Each owner pipelines straight into their
    family's δ-curve, layer-k identification, and figures the moment
    their sweeps land (Qwen ~midday, then Llama, then Gemma) — analysis
    overlaps compute instead of queueing behind it.
  - `run_sweep.py --layers A-B` splits one family across several
    workers under a single guarded manifest.
  - **Throughput**: Kaggle meters SESSION time, not GPU time, so a
    2×T4 session gives two GPUs per quota-hour — run two chunked
    processes per session (`CUDA_VISIBLE_DEVICES=0` / `=1` on
    different layer ranges). Four people × 30 quota-h × 2 GPUs ≈ 240
    T4-GPU-h for the week vs ~190 for the full three-family pipeline:
    quota works ONLY if both GPUs per session are used.
- **Parallel**: IT evals if live; probe-AUROC + attention-JSD curves;
  patching runs if Tier-2 fired (dropped first if capacity is short).
- **Afternoon/evening**: freeze numbers; figures; assemble the rough
  draft (results; limitations: single seed + lapsed replication policy,
  IT status, any family whose Stage-3 did not finish; corroboration).
  **EOD: draft to mentor/PIs.**
- **Fallback**: a family that misses the freeze lands post-draft and
  cheaply — its Stage-2 checkpoints are trained either way, so the
  remaining work is eval-only.

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
