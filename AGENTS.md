# Removed or Relocated — Agent Instructions

## Context
Mech-interp research codebase (deception localization/relocation across
transformer layers). RESEARCH_SPEC.md holds the research context and all
decided definitions — it is normative. Read it before nontrivial work.
INTERFACES.md is the binding contract between the team's tracks — schemas,
signatures, eval constants. Code must match it. If what you're building
doesn't fit it, escalate; never adapt silently and never edit the contract
yourself.
For how the repo, Colab, and Drive fit together, see README.md and
Notebook Setup.ipynb.

## The one rule that matters
Many things are intentionally not yet defined (thresholds, training recipes,
grader details). If the spec or the request doesn't pin something down,
ask — never fill a methodological gap with a reasonable-seeming default.
An undefined thing is a pending decision, not an invitation.

## Division of labor
- All training, fine-tuning, and benchmarking runs happen in Colab, run by
  a human. Never attempt to execute the pipeline here; there is no GPU.
- Your verification is limited to reading code and running whatever CPU
  tests exist. If you can't verify something, say so — don't declare it works.

## Conventions
- Results are append-only JSONL records, never ad-hoc files.
- Scope: do what was asked at the scale it was asked. This is assist-level
  work, not autonomous ownership — no unrequested refactors, no expanding
  a small fix into a redesign.

## In review sessions
Report every finding with a confidence and severity; don't self-filter
for importance.

## Role files
roles/ contains per-session role assignments for a multi-model workflow.
Follow a role file only if your kickoff prompt explicitly assigns it.
Otherwise ignore roles/ entirely — those instructions are not for you.