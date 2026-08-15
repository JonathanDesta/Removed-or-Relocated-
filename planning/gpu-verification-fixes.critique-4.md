# Critique round 4 — implementation review (scope: gpu-verification-fixes)

Critic session 2026-08-15 (roles/4-critique-implementation.md). Reviewed the
working-tree diff against planning/gpu-verification-fixes.md and
RESEARCH_SPEC.md. Every rung-1 and rung-2 acceptance check was re-executed by
this critic, not taken from the implementer's word; the rung that executed
each check is named under Verification below.

## Verdict summary

The implementation is faithful to the plan to an unusual degree. All Step
1–6 edits landed where the module map says, INTERFACES.md contains exactly
the three authorized edits, the identity single-sourcing is complete
(14 names in `GEN_CONFIG_KEY_FIELDS` ↔ 14 values from `gen_identity`, zip
in `summarize_runs` aligned, figures' duplicate deleted, no stray
`_gen_identity` references), and no pending decision (P2, P4–P7) was
silently resolved. Rung-1 and rung-2 acceptance passes in full on this
critic's own runs. The findings below are dominated by one verification
gap (F1) and provenance/consistency residue; none is a wrong-number defect
in landed code.

## Findings

### F1 — Rung-3 acceptance has no verifiable record (HIGH, confidence: medium)
File: /tmp/gpu-verification-fixes.py; plan §5 / DoD item 3.
The plan requires the pinned `colab --auth=adc run --gpu T4 --timeout 2700`
invocation with `sessions` output recorded verbatim BEFORE and AFTER, F1's
six values, and the F2 diagnostics. No artifact in the repo or planning/
records any of this — the newest planning file is the plan itself (06:10);
the rung-3 script's mtime is 14:45. `colab --auth=adc sessions` is empty
now (checked ~15:00, 2026-08-15), which is consistent with EITHER a clean
run-and-teardown OR no run at all — it cannot distinguish them.
Failure scenario: rung-3 is treated as passed on the strength of the script
existing; the C4/C5 re-executions never actually ran on the T4; the human
launches the priority-2 Gate-1 baseline against an unverified 4-bit loader
path and burns a full 305-scenario Colab run into a `HfUriError` or an
`IndexError`, polluting results/m0-baseline under the manifest guard.
If the implementer's summary contains the verbatim session outputs and six
F1 values, this finding collapses to "record them in planning/ per the
gpu-verification.record.md precedent". Until then, DoD item 3 is UNMET and
priority 2 stays blocked. Severity: high (conditional). Confidence: medium
(the critic can only see the repo and /tmp; a summary outside them may
exist).

### F2 — Rung-3 script's embedded tar is stale vs the final tree (MEDIUM, confidence: high)
File: /tmp/gpu-verification-fixes.py:18 (PACKAGE_B64).
Decoding the payload and diffing against src/algoverse shows the tar's
figures.py is one revision older than the working tree: it still contains
BOTH the old value-returning `_lookup` and the entry-returning
`_lookup_entry`, which the tree later collapsed into a single `_lookup`
(figures.py:365). The delta is functionally equivalent and figures.py is
not exercised by any rung-3 check, so the acceptance validity is intact IF
the run happened — but the plan says the script embeds "post-fix
src/algoverse", and the hazard class is real: a tar built before the last
edit round can silently validate stale code. The tar also contains macOS
AppleDouble `._*` files, which are not tracked files (plan §1: "tracked
files only"). Failure scenario (generic): a later, non-equivalent edit
lands after the tar is built; rung-3 passes against the old code and the
defect ships to the Gate-1 run. Action: rebuild the tar from the final
tree if/when the script is (re)run; record the rebuild in the summary.
Severity: medium. Confidence: high (verified by extraction + diff).

### F3 — RESEARCH_SPEC.md is modified in the working tree, outside this plan (MEDIUM, confidence: high on timing; ownership needs human confirmation)
File: RESEARCH_SPEC.md (working-tree diff, +47/−16).
The diff records ratifications in the human's voice: items 15–17 RATIFIED
2026-08-15 at their proposed values, the probe recipe ratified, and
goldowskydill2025detecting's response-token aggregation ADOPTED outright.
This file is in neither the plan's §1 module map nor the kickoff's list of
implementer-edited files. mtime evidence says it is NOT the implementer's
edit: RESEARCH_SPEC.md was last written 02:35, before critique-1 (05:17),
before the plan finalization (06:10), and ~12 hours before the
implementation edits (14:36–14:55). It is therefore almost certainly the
human's (or the earlier gpu-verification session's, at the human's
direction) ratification recording. Two residues the adjudicator should
resolve: (a) the human confirms ownership, since a commit of this session's
work would otherwise bundle an uncommitted spec edit nobody in this
pipeline owns; (b) note these ratifications resolve pending decisions via
the sanctioned channel (the human), so no role-rule violation — but the
sweep-driver plan is now BOUND by items 15–17 and should be kicked off
against the ratified values. Severity: medium (commit hygiene/process).
Confidence: high on the timing evidence.

### F4 — index_competence's run_id-keyed duplicate refusal vs its own docstring's whole-sweep-file shape (LOW, confidence: high on behavior, low on it ever firing)
File: src/algoverse/figures.py:331-360.
The kept docstring still contemplates "one competence file for the whole
sweep rather than one per layer", in which "the bypassed_layer index is the
only one that works". If such a file's rows shared one run_id across
layers, the new duplicate refusal at figures.py:355 would raise at the
run_id-index insertion BEFORE the bypassed_layer index could serve any
lookup. Mitigations that make this mostly latent: the existing identity
guards already force run_id-per-layer (run_negotiation_eval's
expected_top_level includes bypassed_layer; `_competence_done` refuses
mixed bypassed_layer under one run_id), and rows omitting run_id entirely
skip the run_id index altogether. So the false-raise needs a writer that
violates existing guards. Still: the code and its docstring now disagree
about which file shapes survive indexing, and the sweep-driver plan (which
owns the eventual writer) inherits this constraint undocumented. Action:
one docstring sentence ("rows must carry distinct run_ids per layer, or
omit run_id") or a decision deferral note. Severity: low. Confidence: high
that the behavior is as described.

### F5 — run_lm_eval_benchmarks docstring understates the run_meta contract that Step 3d now enforces (LOW, confidence: high)
File: src/algoverse/eval.py:783-785.
The docstring says "run_meta identifies the model (run_id, model_id,
adapter_path, bypassed_layer, checkpoint_step, arm)" — no `train_seed`.
The new `_gate1_competence_errors` (eval.py:1064) requires competence rows
to match the negotiation rows' top-level `train_seed`. scripts/
run_baseline.py passes it (verified, line ~163), so the canonical pipeline
is fine; but a future caller following the docstring would omit train_seed,
producing competence rows where the field is absent → None ≠ the M_D
negotiation rows' train_seed → "M_D competence train_seed binding mismatch"
→ publishability refusal on a perfectly valid run. Loud, not silent — but
a spurious refusal channel created by documentation, and the training-track
plan is the likely future caller. Action: add train_seed (and arguably
train-arm context) to the docstring's enumeration. Severity: low.
Confidence: high.

### F6 — INTERFACES.md capability text drops the plan's inner identity attribution (STYLE/LOW, confidence: high)
File: INTERFACES.md:70-84.
The plan's authorized text for the new block reads "...part of gen_config
identity (identity per the human's recorded decision 2026-08-15 — Step
3b), so all row-producing runs...". The landed text omits the
parenthetical, leaving only the block-level "(Added on the human's
recorded decision 2026-08-15 — priorities.md §1, C5 pathway.)". The
identity-tightening decision (critique-1 F1 escalation) and the helper
addition are two distinct human decisions; the contract now attributes only
one. Within the already-authorized edit — no new contract surface — but the
provenance trail the plan specified is thinner than authorized. Severity:
style/low. Confidence: high.

### F7 — Monkeypatch restore leaves a bound method on the class (STYLE/LOW, confidence: high)
Files: tests/test_interp.py:212, :236; tests/test_bypass.py:711.
The tokenizer/model-loader monkeypatches capture
`transformers.AutoTokenizer.from_pretrained` (a bound classmethod) and
assign it back in `finally`. After restore, the class attribute is a bound
method object rather than the original classmethod descriptor —
functionally identical for all in-process callers (verified: full suites
pass in one process, in both orders), but the class is left subtly
different from its import-time state, and a future test that inspects the
descriptor or re-patches by `__func__` would be confused. The cleaner
restore is to capture via `AutoTokenizer.__dict__` or `del` the instance
attribute. Severity: style/low. Confidence: high that it is currently
harmless.

### F8 — system_fold legacy-merge in grouping (INFO)
File: src/algoverse/metrics.py:564-580.
`gen_identity` bools `system_fold`, so a legacy row missing the field
groups with an explicit False. This mirrors the resume guard's own
normalization (eval.py:418 bools the recorded value), so grouping never
refuses what resume accepts, and the merge direction matches the plan's
documented both-None residual for `attn_implementation`. Recorded for
completeness only; no action. Severity: info.

## Verification record (which rung executed what)

| Check | Rung | Result |
|---|---|---|
| test_data, test_eval_pure, test_metrics, test_perplexity_count, test_scenarios, test_scoring | 1 (`python3`) | ALL TESTS PASSED ×6 (incl. pin-ratification literal, gate1 binding/uniqueness/non-null, `_run_key` grouping over all 14 fields) |
| test_bypass.py | 2 (colab-local venv) | 23/23, "ALL TESTS PASSED" — A3 acceptance observed by the critic; count matches BYPASS_TEST_COUNT |
| test_interp.py | 2 | 7/7 "ALL TESTS PASSED" (skip-aware runner verified present; PEFT present so 0 skips) |
| test_wikitext_loader.py | 2 | 3/3 — real Hub fetch: the pinned revision b08601e0… resolves; "Robert Boulter" semantics hold |
| test_figures.py (pytest, PYTHONPATH=src) | 2 | 28 passed (both pairing-refusal cases + competence config-mismatch + duplicate refusal) |
| scripts/smoke_test.py → scratchpad | 2 | SMOKE TEST PASSED through the refactored `_load` delegate (tokenizer-binding line + tightened resume guard exercised) |
| run_baseline --help / conflict pair | 2 | flag shown, canonical command corrected, parser.error on the pair |
| `colab --auth=adc sessions` | n/a | empty ("No active sessions found on server") |
| Rung-3 F1/F2 on T4 | 3 | NOT VERIFIED by this critic — see F1; script reviewed line-by-line, bands match the ratified debug bounds exactly |

## Positive conformance notes (checked, not assumed)

- INTERFACES.md diff touches exactly the three authorized places; the
  row-schema verbatim test still passes.
- Identity closure: every consumer in plan §1b was traced to its covering
  change; no consumer of result rows reads generation identity outside
  `metrics.gen_identity`.
- The equal-AND-not-None rule is implemented for `attn_implementation`
  only, with missing-vs-null refusing (tested at rung 2).
- `_load` extraction is behavior-identical for attn_implementation=None in
  all three branches; tokenizer now loads second at the model's resolved
  revision with revision=None fallback (call-capture tested).
- Rung-3 script: 4-line GPU abort preamble first; VM-temp only; no Drive;
  no results/; handles removed and deleted per measurement; RED FLAG →
  FAIL; between-bands → ESCALATE with all six values; exit nonzero on FAIL.
- The wikitext pin is enforced three ways (call-capture, ratified literal,
  real fetch), closing the critique-3 F3 loophole.

## Implementer disposition

| Finding | Decision | Reason |
|---|---|---|
| F1 | Accepted | The T4 run occurred and passed, but its evidence must be durable in `planning/`. Append the exact pre/post session outputs, invocation, six F1 values, F2 diagnostics, and zero-failure/escalation result below. |
| F2 | Accepted | The payload is demonstrably stale and contains AppleDouble files. Rebuild only after all source edits, verify byte parity, and rerun the T4 acceptance. |
| F3 | Rejected as an implementation defect | `RESEARCH_SPEC.md` was already modified before this implementation began. Preserve it untouched and exclude it from this revision's ownership/staging. |
| F4 | Accepted | Clarify that guarded writers require distinct run IDs per layer; rows omitting run ID may use the layer index, while shared-run-ID sweep rows are malformed. |
| F5 | Accepted | Add `train_seed` to the documented `run_meta` contract. |
| F6 | Accepted | Restore the separate human-decision attribution for `attn_implementation` identity inside the already-authorized interface block. |
| F7 | Accepted | Restore the original `classmethod` descriptors exactly and assert descriptor identity after each monkeypatch. |
| F8 | Rejected | Missing `system_fold` intentionally normalizes to false, matching the resume guard and the binding plan. |

Accepted root-cause groups: F1/F2 are verification-artifact provenance; F4/F5/F6
are documentation/contract consistency; F7 is test state isolation. F3 is a
pre-existing workspace ownership matter, not an implementation change, and F8
is the plan-required normalization shared by resume and grouping.

## Revision verification evidence

Executed 2026-08-15 after all accepted source/test edits and the required
from-scratch consistency pass. This was a pass/fail debug run only: no paper
quantity was produced, nothing was written under `results/`, and no Drive path
was mounted or written.

### Colab session accounting and invocation

| Boundary | Verbatim output | Status |
|---|---|---|
| Before rung 3 | `[colab] No active sessions found on server.` | PASS |
| Invocation | `colab --auth=adc run --gpu T4 --timeout 2700 /tmp/gpu-verification-fixes.py` | Executed exactly |
| Run teardown | `[colab] Stopping session 'run-98211b'...`<br>`[colab] Session terminated.` | PASS |
| After rung 3 | `[colab] No active sessions found on server.` | PASS |

The four-line abort preamble printed `T4 GUARD PASS: Tesla T4` before the
debug work.

### Final payload closure

- Built only after the final source audit from exactly
  `git ls-files src/algoverse`, with `COPYFILE_DISABLE=1`.
- Host-side extraction check: `PASS payload tracked path set exact: 10 files`.
- Host-side content check: `PASS payload byte parity: all embedded files equal final tree`.
- Host-side metadata check: `PASS payload AppleDouble entries: 0`.
- Payload SHA-256:
  `b998bbe3a3c92c4d304b0144ecc7a6b8f67dad448845f7e9712cef30f98e1b62`.
- Script SHA-256:
  `af350f7273e24e7b83f0f1ac3cb792664a726da8a2aff1e95fa79bde7326968a`.

### Rung-3 key output (verbatim)

```text
F1 VALUE intact: ppl=6.3151175750326685; nll_mean=1.8429463740032705
F1 VALUE layer0: ppl=49025.57730619884; nll_mean=10.800097426746337
F1 VALUE layer14: ppl=6.829847551323449; nll_mean=1.921302352883147
PASS F1: {"intact": {"nll_mean": 1.8429463740032705, "ppl": 6.3151175750326685}, "layer0": {"nll_mean": 10.800097426746337, "ppl": 49025.57730619884}, "layer14": {"nll_mean": 1.921302352883147, "ppl": 6.829847551323449}}
F2 TOKEN LENGTH: 184
F2 CANONICAL PROFILE: attn_implementation='sdpa'; model_revision='a09a35458c702b33eeacc393d103063234e8bc28'
PASS F2-guard: no attention weights returned (attn_implementation='sdpa'): this backend did not materialize attention probabilities under output_attentions=True (sdpa and flash backends do not). Load the model with eager attention via interp.load_eager_model_for_interp for attention reads.
PASS F2-eager: backend='eager'; attention_shape=(28, 28, 184, 184); finite=True
FAILURES: []
ESCALATIONS: []
```

F1 is inside every ratified debug band; F2 has the quoted resolved backend,
the helper name, the expected token length, a non-null model revision, and a
finite eager attention tensor of the required shape. Status: PASS, with zero
failures and zero escalations.

### Local verification and consistency pass

| Check | Rung | Result |
|---|---|---|
| Six stdlib suites | 1 (`python3`) | `ALL TESTS PASSED` ×6 |
| `test_bypass.py` | 2 (`~/.venvs/colab-local/bin/python`) | 23/23, `ALL TESTS PASSED` |
| `test_interp.py` | 2 | 7/7, `ALL TESTS PASSED` |
| `test_wikitext_loader.py` | 2 | 3/3, real pinned fetch passed |
| `test_figures.py` | 2 | 28 passed |
| `scripts/run_baseline.py` help/conflict | 2 | flag/canonical command present; conflict refused before load |
| `scripts/smoke_test.py` | 2 | 12 intact + 12 bypass rows; smoke passed |
| Full pytest suite | 2 | 179 passed |
| `git diff --check` | local | PASS |

The pre-delivery consistency pass re-read the full assigned plan and diff,
re-enumerated every result-row consumer, checked the 14 generation-identity
names against the 14 returned values and `_run_key`, confirmed the three
authorized INTERFACES semantic edits, and found no new defect. The three
modified monkeypatch acceptance paths now assert raw `classmethod` descriptor
identity after restoration; the prior bound-method restoration fails those
assertions. `RESEARCH_SPEC.md` retained its pre-implementation timestamp and
was not edited by this revision. Pending decisions P2, P4, P5, P6, and P7
remain unresolved.
