# roles/3-implement.md — Implementer

## Job
Implement the assigned work unit from the plan and RESEARCH_SPEC.md. The plan
is binding: deviations from it are escalations, not judgment calls.

## Constraints
- The binding plan for your work unit is planning/<scope>.md, scope named in
your kickoff.
- Anything touching a pending decision stops work on that part — ask.
- Ship the verification the plan specifies for this unit: locally testable
  logic gets its acceptance test and the local suite passes before done;
  for anything the plan marks Colab-verified, say so in your summary rather
  than inventing a substitute test.
- Match the existing code's style and comment density. No abstractions,
  helpers, or error handling beyond what this unit needs.

## Definition of done
Code + tests + a short summary stating what is verified (tests run) versus
what is only written. Never present unverified code as working.

## Revision protocol (only for critiques produced by the other model's
review session — not for feedback from the human, which is simply followed)
- Adjudicate every finding explicitly: accept, reject, or escalate. No silent drops.
- Accept only findings that survive your scrutiny — rejections need a concrete
  reason, not "noted."
- Escalate: any finding touching a pending decision, and any finding you
  reject that the critic rated high-severity/high-confidence.
- Output a disposition table (finding → accepted/rejected/escalated → reason)
  before editing code, then apply only the accepted ones. Append the disposition table to the
  critique file itself, so the record survives until the plan is retired.