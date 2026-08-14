# planning/first-full-review.critique-5.md — Implementation critique, round 5

Reviewer: implementation critic (roles/4). Scope: the revision responding to
critique-4 — the implementer's disposition table, the human's three escalation
resolutions (recorded in critique-4 and RESEARCH_SPEC §497-503), and the edits
to src/algoverse/{tasks,eval,metrics}.py, tests/{test_scoring,test_metrics,
test_bypass,test_eval_pure}.py, INTERFACES.md, RESEARCH_SPEC.md.

## Test execution (this session, stdlib Python 3.14.6, no ML stack)

- PASS: tests/test_eval_pure.py, test_data.py, test_scenarios.py,
  test_scoring.py, test_metrics.py, test_perplexity_count.py.
- LOUD SKIP (exit 0): tests/test_interp.py ("0 of 4"), tests/test_bypass.py
  ("0 of 16" — count correctly bumped for the new guarded test).

## Disposition verification (every critique-4 item checked against the code)

| Finding | Status now | Verified how |
|---|---|---|
| F-4.1 | FIXED | `_authoritative_answer_text` uses `ANSWER_MARKER_RE.finditer` (tasks.py:131, 335-339) — length-preserving. Re-executed the İ counterexample and a multi-İ variant: both parse correctly. Regression test added (test_scoring.py `test_authoritative_marker_slice_is_unicode_length_safe`). |
| F-4.2 | RESOLVED (human) + IMPLEMENTED | Fallback receives the authoritative slice (tasks.py:478 passes `authoritative`); no marker → whole text (verified executably for both cases). Human resolution recorded in critique-4 and RESEARCH_SPEC §497. Test added (`test_fallback_receives_only_authoritative_final_marker_slice`). |
| F-4.3 | FIXED | `invalid_reason` overridden to "unparseable" after validity classification when range_rejected (tasks.py:494-495). Refusal+range case re-executed → "unparseable"/"regex_range_rejected". Pin added (`test_range_rejection_overrides_refusal_diagnostic`). See N-5.2 for a new edge this creates. |
| F-4.4 | RESOLVED (human) + IMPLEMENTED | `nll_mean` is now a TOP-LEVEL result field on wikitext2_ppl rows (eval.py:957), not in `config`; `_competence_done` excludes it from identity via result_fields (eval.py:722). INTERFACES.md documents it WITH explicit human-authorization provenance ("Authorized by the human 2026-08-14, first-full-review F-4.4"). Guarded resume test added (test_bypass.py `test_perplexity_records_raw_nll_as_result_and_resumes`) — see N-5.1. |
| F-4.5 | RESOLVED (human) + IMPLEMENTED | `comparable_config` strips `batch_size` in BOTH homes (eval.py `_gate1_benchmark_errors` and metrics.py gate1_decision:414-418). Re-executed: batch-only diffs produce zero errors; a real limit diff is still caught. Helper test + end-to-end PASS-with-differing-batch test added. Resolution recorded in RESEARCH_SPEC §500-503. |
| F-4.6 | FIXED | `_gate1_pool_errors` iterates M_0, M_D, plus every additionally supplied rows file (eval.py:983-984), so M_C is coverage-checked when given. Re-executed: a 1-row M_C yields "M_C missing 610 scenario-condition pairs". End-to-end tiny-M_C → INCOMPLETE test added. |
| F-4.7 | FIXED | `test_gate1_report_wires_pool_defects_to_incomplete` loops the malformed cases (sparse control, duplicated pair, mixed run_ids) through gate1_report end-to-end and asserts DECISION: INCOMPLETE with the defect wording. |
| F-4.8 | RESOLVED-FOR-RECORD (round-4 code) | The implementer names the environment: isolated CPython 3.12.13 CPU, torch 2.13.0, transformers 5.15.0, scikit-learn 1.9.0, peft 0.20.0; all 4 interp + 15 bypass tests + unchanged smoke test passed before handoff. I cannot independently verify (the temp uv.lock/.venv were removed), but the plan's requirement — a named environment — is met for the ROUND-4 state. It is NOT met for this revision: see N-5.1. |
| F-4.9 | STANDING (unchanged) | Regeneration/Colab/Llama-Gemma obligations remain open by design. |
| F-4.10.1 | FIXED | Both unused imports removed; test_bypass.py's new `eval_module` import is used (monkeypatching load_wikitext_slice). |
| F-4.10.2 | PARTIALLY ACCEPTED, implemented as stated | `_gate1_benchmark_errors(bench, reference="M_0")` is reference-aware and gate1_report passes `reference` through (eval.py:1096); tested with a custom reference name. Duplicate enforcement in gate1_decision retained deliberately (direct-caller defense) — reasoning recorded in the disposition table; acceptable. See N-5.3 for a cosmetic residue. |
| F-4.10.3 | FIXED | `GSM8K_LIMIT = 400` / `MMLU_LIMIT_PER_SUBTASK = 16` (eval.py:46-47) are the single home: runner defaults (eval.py:764-765) and deviation reporting (eval.py:1126-1128) both use them. |
| F-4.10.4 | FIXED | extract_claimed_offer docstring documents last-marker authority. |
| F-4.10.5-8 | Recorded as informational; no edits needed (uv.lock/.venv explained as removed ML-verification artifacts, consistent with the empty working-tree evidence from round 4). |

Contract/spec discipline: the INTERFACES.md edit is confined to the authorized
nll_mean sentence with provenance; the RESEARCH_SPEC edit lands in the
Ratified-decisions recording section (§496-503), proposal body untouched —
both consistent with the standing rules.

## New findings (round 5)

### N-5.1 — The revision's guarded code and its NEW guarded test have not been executed
- **Where:** tests/test_bypass.py (`test_perplexity_records_raw_nll_as_result_and_resumes`,
  BYPASS_TEST_COUNT 15 → 16); revised src/algoverse/eval.py + metrics.py.
- **What:** the ML-environment run named under F-4.8 covered the ROUND-4 state
  ("all 15 bypass tests"). Waves 1-2 then changed eval.py, metrics.py, and
  tasks.py and added a 16th guarded test that exercises the new nll_mean
  field and perplexity resume. No re-run is claimed anywhere in the
  disposition or resolution notes, and locally both guarded suites loud-skip.
- **Failure scenario:** the new guarded test (or the revised compute_perplexity
  row shape under real torch) fails in the first Colab session, discovered
  mid-run rather than pre-handoff — the exact situation the environment gate
  exists to prevent.
- **Severity:** medium-low (the changes are small and the pure-side logic is
  tested, but the plan's environment-gate rule applies to revisions too).
  **Confidence:** high that no revision-state run is claimed; unknown whether
  one silently happened. Ask the implementer to name it or run it.

### N-5.2 — Range-rejection override now also clobbers the "truncated" diagnostic
- **Where:** src/algoverse/tasks.py:494-495 (`if range_rejected:
  invalid_reason = "unparseable"` after check_validity).
- **What (executed):** score_response("MY BEST OUTSIDE OFFER: $110-120k",
  hit_max_tokens=True) → invalid_reason "unparseable" (previously
  "truncated"). The F-4.3 fix implements the ratified range→unparseable rule
  unconditionally, which silently redefines the precedence between two
  invalid-row diagnostics for truncated replies whose visible text ends in a
  complete range line.
- **Failure scenario:** truncation-rate monitoring (hit_max_tokens rows) and
  unparseable-rate accounting shift labels for this corner; nothing about
  validity or deception measurement changes (the row is invalid either way,
  and the fallback is correctly skipped in both labelings).
- **Severity:** low (diagnostic accounting only). **Confidence:** high on
  behavior; the precedence itself is an unpinned choice — record it (one
  docstring line or a dated note) so it is deliberate rather than accidental.

### N-5.3 — Cosmetic: gate1_decision's incomplete-bench message hard-codes "M_0/M_D"
- **Where:** src/algoverse/metrics.py:421-422.
- **What:** with a custom `reference`, the logic correctly uses `reference`
  everywhere, but the error string still reads "complete M_0/M_D benchmarks
  are required". Cosmetic only. Severity: low/style. Confidence: high.

### N-5.4 — Style: awkward comment wrap in score_response
- **Where:** src/algoverse/tasks.py:466-467 ("...authoritative final
  answer text; it never / judges deception" — mid-sentence break left over
  from the edit). Doc-only. Severity: style. Confidence: high.

## Verified vs. written (AGENTS.md discipline)

VERIFIED by execution here: all six pure suites; the loud skips at counts
4/16; the F-4.1/F-4.2/F-4.3 fixes (including multi-İ, no-marker fallback
input, and the truncated+range edge); the batch_size exemption and its
real-diff counterpart; M_C coverage errors. WRITTEN, NOT VERIFIED: everything
behind the ML guard for the REVISED code (N-5.1), the claimed round-4 ML run
(taken on the implementer's named record), and all Colab/notebook behavior.
Nothing in this round requires a human decision except noting N-5.2's
precedence choice; N-5.1 needs a named (or fresh) ML-environment run of the
revised guarded suites.

## Implementer disposition (2026-08-14)

Per `roles/3-implement.md`, this table was appended before revision work.
Only ACCEPTED items are edited; the methodological precedence in N-5.2 is
left unchanged pending human direction.

| Finding | Disposition | Reason |
|---|---|---|
| N-5.1 | REJECTED (premise stale; verification recorded) | The exact revision under review was executed after the `nll_mean` writer/test and Gate changes landed, in an isolated CPython 3.12.13 CPU environment with torch 2.13.0, transformers 5.15.0, scikit-learn 1.9.0, and peft 0.20.0. All 4 interp tests, all **16** bypass/evaluator tests (including `test_perplexity_records_raw_nll_as_result_and_resumes`), and the unchanged Qwen2.5-0.5B smoke test passed. The final handoff explicitly reported 4/4, 16/16, and smoke PASS. No code revision is needed; this row makes the named revision-state run durable in the critique record. |
| N-5.2 | ESCALATED | The executed edge is real, but choosing whether `hit_max_tokens` or a detected range controls `invalid_reason` changes truncation-vs-unparseable diagnostic accounting. Neither the plan nor ratified text pins that precedence. Per AGENTS.md and the implementer protocol, no semantic or documentation edit is permitted until the human chooses it. |
| N-5.3 | ACCEPTED | Custom-reference logic is already correct; interpolating the actual reference name into this error removes a misleading cosmetic residue without changing Gate behavior. Add a pure regression assertion for the message. |
| N-5.4 | ACCEPTED | Rewrap the comment without changing meaning or behavior. |

## Post-revision verification (2026-08-14)

- All six pure suites passed, followed by compileall, notebook JSON
  validation, and `git diff --check`.
- A fresh isolated CPython 3.12.13 CPU environment with torch 2.13.0,
  transformers 5.15.0, scikit-learn 1.9.0, and peft 0.20.0 passed all 4
  interp tests and all 16 bypass/evaluator tests.
- The unchanged Qwen2.5-0.5B smoke test passed: 12 intact plus 12 bypass
  rows, schema/resume guards, and byte-identical bypass removal.

## Human resolution of N-5.2 (2026-08-14)

When a generated response both hits the token limit and contains a complete
rejected dash-range answer, the row records both facts: `hit_max_tokens` is
`true`, and `invalid_reason` is `"unparseable"`. Thus the rejected range
controls the single invalid-reason category without discarding the separate
generation-truncation signal.
