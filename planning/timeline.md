# Deadline timeline — per-person tracks (results + rough draft by EOD 2026-08-18)

Restructured 2026-08-16 (late): P1, P2, P3 each own ONE model family
end-to-end; P4 owns the paper + Insider Trading and floats. Assignments
are permutable so long as Gemma's owner drives the A100. All nine
decision-packet items are RATIFIED — the authoritative record is
RESEARCH_SPEC.md "Ratified decisions (2026-08-16, deadline session)".

Task labels: **[BUILD]** writing code/docs/paper · **[CHECK]** testing,
auditing, verifying (no new results) · **[RUN]** a real experiment on a
notebook GPU (Colab/Kaggle/A100).

Already DONE tonight (do not redo): AWS account; everyone's
secrets/access checks; the folded Gemma dataset (Drive:
`data/finetune-folded`, manifest `fold_system: true`); the extractor
verified end-to-end on Azure gpt-5-mini (every EVAL session needs
`OPENAI_API_KEY` + `OPENAI_BASE_URL=https://desta.services.ai.azure.com/openai/v1/`
from that platform's secrets; training runs need neither); the A100
request form submitted; the complete sweep-driver stack + Stage-2
loader, both test rungs green.

---

## P1 — Qwen2.5-7B (28 layers; the reference family)

**Tonight**
- [RUN] Launch overnight: M_0 baseline (full 305 pool,
  `run_baseline.py ... --llm-fallback --competence`) on Kaggle 2×T4.
- [RUN] Launch M_D and M_C training (~1–1.5 h each, unfolded dataset)
  on Colab/Lightning.

**Aug 17**
- [CHECK] 08:00 — refusal-row audit of your baseline (ratified item 7).
- [RUN] ~08:30 — DEV JSD calibration (`run_sweep.py --dev-calibration`,
  Qwen-0.5B, free tier). Its confirm-or-revise decision at 09:00
  unlocks EVERY family's sweep scoring (the driver refuses without it).
- 09:00 all-hands (30 min).
- [RUN] Morning — M_D final-checkpoint full-pool eval + competence
  (~2–3 h).
- [CHECK] ~midday — `gate1_report` vs M_0: **Gate-1 verdict.** Fail →
  the pre-agreed lever (ONE rerun, epochs 3→4); second fail → family
  drops to post-draft.
- [RUN] Afternoon — 28-layer sweep, `--layers` chunks across your two
  Kaggle GPUs (two processes, CUDA_VISIBLE_DEVICES=0/1) + Colab (~3 h).
- [CHECK] ~16:30 — `sweep_report`: l\* or the no-viable-layer verdict.
- [RUN] Late afternoon — full-pool confirmation (M_D^-l\*, n=305) +
  held-out transfer (final pool), ~2 h fleet-parallel.
- [RUN] Evening — Stage-2 four arms (~1.5 h on 4 workers; the loader
  reinstalls the lesion automatically). Then launch the overnight R_t
  evals (t ∈ {8,70,281} × 4 arms, held-out pool).

**Aug 18**
- [CHECK] Early — `recovery_report`: recovery verdict.
- [RUN] Morning — recovery sweeps (M_281^{L,D} and ~M_D, ~19 T4-h,
  4–5 workers ≈ 4 h) → **done ~12:30**.
- [BUILD] Early afternoon — δ-curve, layer-k, figures
  (`make_figures.py`); hand numbers to P4. Then float.

## P2 — Llama-3.1-8B (32 layers)

Identical track to P1, with these deltas:
- [CHECK] Tonight FIRST: accept the meta-llama license on YOUR HF
  account (gated model; downloads fail without it).
- No DEV-calibration duty.
- Sweep is 32 layers (~3.5 h; l\* ~17:00). Recovery sweeps ~21 T4-h →
  **done ~13:30–14:00**; figures to P4 mid-afternoon.
- Your Gate-1 verdict is your own — independent of Qwen's.

## P3 — Gemma-2-9B (42 layers; the long pole; you drive the A100)

Identical track to P1, with these deltas:
- [CHECK] Tonight FIRST: accept the google/gemma license on YOUR HF
  account.
- [RUN] ALL Gemma training uses the FOLDED dataset
  (`data/finetune-folded`); the T12 guard refuses the wrong pairing.
- Your big jobs go on the team A100 the moment the grant email lands
  (countdown starts at approval — start immediately); your own
  Kaggle/Colab T4s carry `--layers` chunks in parallel; AWS L40S
  supplements YOUR family first if quota clears.
- Sweep ~4–5 h even with the A100 (l\* ~17:30–18:00). Recovery sweeps
  ~35 T4-h-equivalent → A100 + T4 chunks ≈ 4–5 h → **done
  ~14:30–15:00** — before the freeze, least slack.
- If the A100 grant slips: Gemma is the family the post-draft fallback
  exists for. Run what fits on free tiers; the rest is eval-only later.

## P4 — Paper + Insider Trading + float

**Tonight**
- [BUILD] Read Scheurer et al. (arXiv 2311.07590) + released prompts;
  write the IT design per planning/insider-trading.md — paired
  conditions, FROZEN prompt set (verbatim-complete or a pre-registered
  rule, never a hand-picked subset), grading rule, validity policy,
  scenario count/split.
- [BUILD] Paper skeleton (methods text largely exists in
  RESEARCH_SPEC's Final-paper deltas).

**Aug 17**
- 09:00 — present the IT design; ratified same-day (process-compression
  rule).
- [BUILD] Morning — implement IT scenarios + grader (pure Python,
  tasks.py-style, agent support). [CHECK] its rung-1 suite green.
- [RUN] Afternoon — small real-model IT smoke.
- **18:00 HARD CHECKPOINT** (pre-agreed, no debate): live → [RUN] IT
  transfer evals tonight on spare fleet; not live → recorded fallback
  (draft ships held-out-negotiation transfer only; IT lands before
  final submission).
- [BUILD] Between IT blocks: methods/setup sections; dry-run
  `make_figures.py --synthetic` so Aug-18 figures are drop-in.

**Aug 18**
- [BUILD] Write all day; ingest each family as it lands (Qwen ~12:30 →
  Llama ~14:00 → Gemma ~15:00). [RUN] IT recovery evals if live.
- **~17:00 NUMBERS FREEZE** → [BUILD] assemble the rough draft:
  results, limitations (single seed + lapsed replication policy, IT
  status, any family that missed the freeze), corroboration, and the
  Final-paper-deltas citation fixes. **EOD: draft to mentor/PIs.**

## Claude (agent) — done tonight, on-call after

All non-IT/non-paper [BUILD]/[CHECK] work was front-loaded to tonight:
recovery_report + matched-arms audit, the corroboration driver
(ratified response-token probe aggregation, attention-JSD,
interp.jsonl writer), the plotting layer (`make_figures.py`), the
single rung-3 T4 smoke of the sweep driver, and — only if all of that
is green — Tier-2 activation patching. From Aug 17 morning the agent
is purely on-call for debugging and analysis support.

## Standing rules

- Kaggle meters SESSION time: always 2×T4 sessions running two chunked
  processes; stop sessions when done.
- Never leave the A100 idle >1 h (reclaim risk).
- Publishable evals: full pools, `--llm-fallback --competence`, both
  Azure env vars set.
- Results are append-only JSONL under results/ on Drive; everything
  resumes; humans run every result-producing job.

## Contingencies (per family)

Gate-1 fail → one epochs-3→4 rerun, then post-draft. No viable layer →
pre-registered negative result for that family. A100 slips → Gemma
falls back. Any family missing the ~17:00 freeze → post-draft,
eval-only (its Stage-2 checkpoints exist either way).
