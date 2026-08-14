# planning/first-full-review.critique-6.md — Implementation critique, round 6

Reviewer: implementation critic (roles/4). Scope: the revision responding to
critique-5 — dispositions (N-5.1 rejected-with-record, N-5.2 escalated and
human-resolved, N-5.3/N-5.4 accepted) and the edits to
src/algoverse/{metrics,tasks}.py, tests/{test_metrics,test_eval_pure}.py,
RESEARCH_SPEC.md, plus the disposition/verification/resolution appendices in
critique-5.

## Test execution (this session, stdlib Python 3.14.6, no ML stack)

- PASS: tests/test_eval_pure.py, test_data.py, test_scenarios.py,
  test_scoring.py, test_metrics.py, test_perplexity_count.py.
- LOUD SKIP (exit 0): tests/test_interp.py ("0 of 4"), tests/test_bypass.py
  ("0 of 16").

## Disposition verification

| Finding | Status now | Verified how |
|---|---|---|
| N-5.1 | CLOSED (rejected with record) | The disposition names a revision-state run — isolated CPython 3.12.13 CPU, torch 2.13.0, transformers 5.15.0, scikit-learn 1.9.0, peft 0.20.0; 4/4 interp, 16/16 bypass (including the new nll_mean resume test), smoke PASS — and a fresh post-revision run is recorded again after this round's wave-1 edits. Taken on the implementer's named record (the temp env is gone, so not independently verifiable). The plan's environment-gate requirement is satisfied as stated. Round-6 wave 2 changed no guarded behavior (comment-only tasks.py lines, a pure test, spec/critique records), so no further run is owed. |
| N-5.2 | RESOLVED (human) + IMPLEMENTED | Ratified precedence: both facts recorded — `hit_max_tokens` true, `invalid_reason` "unparseable". Behavior re-executed and matches; the choice is now recorded in a code comment naming the ratification (tasks.py:492-495) and in RESEARCH_SPEC's ratified-decisions section (§503-506, proposal body untouched). Pin test added: tests/test_eval_pure.py `test_truncated_rejected_range_row_records_both_signals` (passes). |
| N-5.3 | FIXED | The message interpolates the reference name — `"complete %s/M_D benchmarks are required" % (reference or "M_0")` (metrics.py:421-424). Re-executed with reference="base" → "complete base/M_D benchmarks are required"; default path keeps the M_0 wording. Regression assertion added to test_metrics.py `test_gate1_decision_without_completeness_is_incomplete`. |
| N-5.4 | FIXED | Comment rewrapped (tasks.py:470-471); the mid-sentence break is gone and the meaning is unchanged. |

## New findings (round 6)

### N-6.1 — Info only: the N-5.2 pin test simulates the runner's hit_max_tokens stamping
- **Where:** tests/test_eval_pure.py:167-178.
- **What:** the test constructs `row = {"hit_max_tokens": hit_max_tokens,
  **scoring}` itself rather than proving the eval runner stamps the field on
  a real row. The runner's stamping is pre-existing behavior, separately
  pinned by the ROW_FIELDS partition test (hit_max_tokens is a runner field),
  so the seam is covered indirectly; this note only records that the pin is a
  scorer-level pin, not an end-to-end one.
- **Severity:** info. **Confidence:** high.

No other findings. Nothing in this round is escalated and nothing awaits a
human decision.

## Verified vs. written (AGENTS.md discipline)

VERIFIED by execution here: all six pure suites; loud skips at 4/16; the
N-5.2 ratified behavior (truncated+range → hit_max_tokens preserved,
invalid_reason "unparseable", method "regex_range_rejected"); the N-5.3
message with custom and default reference; the N-5.4 rewrap by reading.
WRITTEN, NOT VERIFIED: the two named ML-environment runs (taken on the
implementer's record) and all Colab/notebook behavior. The first-full-review
scope's standing obligations (F-4.9: data regeneration before training use,
Colab fresh-VM checks, Llama/Gemma BOS/fold spot checks) remain open by
design and transfer to the operational plans.
