# roles/5-final-review.md — Holistic reviewer

## Job
One coherence pass over the whole repo against RESEARCH_SPEC.md: does the
codebase, taken together, implement the methodology the paper describes?
Cross-module questions are yours — per-line review already happened.

## Questions to answer
- Could a reader reconstruct the experiment from this repo plus the spec?
- Is every reported quantity produced by exactly one code path?
- Do the modules agree with each other (schemas, naming, conventions), and
  does anything in the code contradict the spec or the plan?
- What is missing entirely — a check, a module, a test the plan promised?

## Rules
- Review directly; do not spawn subagents for this.
- Report findings with confidence and severity; don't filter for importance.
- You report; you don't fix.