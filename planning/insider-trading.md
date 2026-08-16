# Insider Trading kickoff brief (scope: insider-trading) — 2026-08-16

For P4 (owner) + supporting agent. This is the kickoff brief from
priorities.md §6 sharpened for the deadline; the DESIGN PROPOSAL you
produce tonight is ratified at the 09:00 Aug-17 meeting (single round,
process-compression rule). Full scope was the team's explicit choice —
the price is the hard checkpoint at the bottom.

## What it is

Operationalize the Insider Trading environment (Scheurer et al. 2023,
arXiv 2311.07590) as an EVALUATION-ONLY second environment: it
validates transfer of A_l* in Stage 1 (alongside held-out negotiation)
and computes R_t in Stage 3. Never used for fine-tuning or layer
selection.

## Tonight (P4)

1. Fetch and READ the paper and its released prompts/materials (do not
   work from summaries). Known context: deception rates there swing
   ~4% → ~97% across prompt variants — prompts are FROZEN once chosen
   and conditions paired, same discipline as negotiation.
2. Write the design proposal. It must pin, concretely:
   - **Paired conditions**: what exactly differs between
     deception-advantageous and control (the pressure/incentive
     elements), everything else byte-identical.
   - **Frozen prompt set + scenario parameterization**: which Scheurer
     variants are used verbatim vs reduced; scenario count and the
     split naming (see contract note below).
   - **Grading rule**: what counts as deceptive in the model's report
     to its manager (misrepresentation detection); what the structured
     output format is; regex-first + the pinned LLM fallback
     (gpt-4o-mini-2024-07-18) with the same recording discipline
     (`extraction_method`, per-row).
   - **Validity/refusal policy**: reuse the negotiation machinery where
     it transfers (too_short floor, refusal list, `deceptive: null` on
     invalid rows — that rule is contract-level and NOT negotiable);
     state explicitly what does not transfer and what replaces it.
   - What tau means here (D_advantageous − D_control must be computable
     by the existing `metrics.incentive_gap` on conforming rows).
3. Every quantitative choice is PROPOSED with reasoning, ratified
   tomorrow 09:00 — do not bury decisions in code.

## Contract conformance (hard requirements)

- Rows conform to the INTERFACES results schema verbatim: same 26
  fields, append-only JSONL under results/<run_id>/, resume key
  (run_id, scenario_id, condition), manifest sidecar, identity guards.
  Reuse `eval.run_negotiation_eval`'s row machinery rather than
  cloning it — if that requires generalizing a function, that is a
  normal refactor; if it requires a SCHEMA change (e.g. a new `split`
  value like "insider" vs reusing "final"), that is an INTERFACES
  question to raise at 09:00, never a silent adaptation.
- Scenario ids: content-hashed like tasks.py's, so the set is frozen by
  construction.
- Implementation shape: pure-Python scenario construction + grading in
  the style of tasks.py (rung-1 testable, no ML stack).

## Aug 17 (P4 + agent)

- Morning: implement env + grader per the ratified design; rung-1
  tests (grading decision tree, condition pairing, id stability,
  refusal/validity paths — mirror test_scoring.py's structure).
- Afternoon: real-model smoke (a handful of scenarios through the
  canonical loader — a debug run, cheapest rung that executes it).
- **18:00 HARD CHECKPOINT** (agreed tonight, no relitigating at 18:00):
  - Code-complete + smoke-passing → IT transfer evals (M_D and
    M_D^-l*, full IT pool) run tonight; IT joins Stage-3 R_t on Day 2.
  - Not ready → recorded fallback: the draft ships with held-out
    negotiation transfer only; IT lands between draft and final
    submission (mentor-review window). Partial credit: the ratified
    design itself goes in the draft as "operationalization defined,
    evaluation in progress."

## Costs (for planning, not decisions)

IT eval pool of ~100–300 paired scenarios ≈ 1–3 T4-h per checkpoint
evaluated; Stage-3 adds 12 more such evals if IT is live — all
fleet-parallel. The schedule constraint is BUILD + DECISION time, not
GPU time.
