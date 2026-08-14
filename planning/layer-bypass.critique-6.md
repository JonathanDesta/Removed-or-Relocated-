# layer-bypass.critique-6 — Implementation critique (round 6)

Scope: the round-5 fix diff (five files: planning/layer-bypass.critique-5.md
disposition, src/algoverse/{eval,tasks}.py, tests/{test_metrics,
test_scoring}.py), reviewed against the frozen plan (rev 6), the round-5
findings and dispositions, RESEARCH_SPEC.md, and INTERFACES.md. Read-only
review; no code edited. tests/test_bypass.py and all other files are
unchanged this round.

---

## Round-5 fix verification

| Round-5 finding | Status | How verified |
|---|---|---|
| F1 competence guard treats `batch_size` as identity | FIXED | `_competence_done` now skips `batch_size` in the config comparison with a WHY comment naming the OOM-recovery rationale (eval.py:666-667); it stays recorded in every row. Reproduced empirically here: a completed gsm8k row at `batch_size=4` re-requested at `batch_size=2` (same limit/seed) now SKIPS; a `limit` change on the same file still refuses naming `config.limit`. The new test_metrics assertion pins exactly this pair. Rows guard and competence guard now agree on the ratified operational-field stance. |
| F2 partial verification exits 0 | REJECTED — disposition sound | Correct: the frozen plan pins the exit-0 skip convention, and the finding was filed as informational. The runner's textual status already distinguishes partial from full verification. Nothing further owed. |
| F3 `llm:<provider>:None` on model-less success | FIXED (stronger than suggested, correctly) | Two-layer fix: `llm_extract_offer` now raises "LLM extraction response omitted its model identifier" inside the try when a value arrives without a model — so with `raise_errors=False` it becomes (None, None)/`llm_failed`, with `raise_errors=True` the probe surfaces it, and the unattributable value is never cached (the raise precedes the cache write). `score_response` additionally requires a truthy `response_model` — not redundant: it also catches an empty-string model from a poisoned/foreign cache entry. The disposition's provenance-or-nothing choice (drop the value rather than substitute the requested model) is the right call: an unattributable extraction must not enter scored data, and the row stays visible as `llm_failed:<provider>`. Both new tests executed and pass here torch-free, including the fake-API model-less completion case asserting the "model identifier" error. |

---

## New findings

None of substance. Two micro-notes, reported per the coverage rule:

### N1. A model-less FAILED extraction still writes a `response_model: null` cache entry

**src/algoverse/tasks.py:562-575.** The new raise fires only when `result
is not None`; a parse-to-None reply with a missing response model skips the
raise, caches `{"claimed_offer": null, "response_model": null}`, and — since
cache reads require a non-null response_model — that entry never hits, so
the same text re-pays the API on every future scoring pass. Correct
behavior, tiny waste, unreachable with real providers. *Confidence: high.
Severity: style/informational.*

### N2. The `batch_size` exclusion is a literal key match in one place

**src/algoverse/eval.py:666-667.** If a future metric ever records an
operational field under another name (e.g. a device count), it would need
its own exclusion; there is no shared "operational fields" constant between
the rows guard (which excludes batch_size by omission from
`guarded_gen_fields`) and the competence guard (which excludes it by name).
Fine at current scale; noted so the next editor knows both sites exist.
*Confidence: high. Severity: style/informational.*

---

## Verification record

Executed on the review machine (macOS, system Python 3 — still no torch, no
peft; no ML environment exists on this machine):

- All six test files re-run: `test_data`, `test_metrics` (36→ incl. the
  extended competence-guard test), `test_perplexity_count`,
  `test_scenarios`, `test_scoring` (36 tests, incl.
  `test_fallback_success_without_response_model_is_recorded_as_failure` and
  the extended probe test — both confirmed executing) — **ALL PASSED**.
  `test_bypass.py` still prints the loud-SKIP line and exits 0.
- eval.py re-verified importable with neither `torch` nor `numpy` entering
  `sys.modules`.
- Round-5 F1's fix reproduced end-to-end (batch-size-only resume allowed;
  limit change refused with the field named).

**Environment-gate status, unchanged and still open:** the ML-gated suite
(tests/test_bypass.py) was not edited this round, so no NEW ML re-run is
triggered by this delta — but the round-5 obligation stands: the 14 bypass
acceptance tests (as last edited in round 5), the PEFT SKIP-path output,
`import algoverse.interp`, and `scripts/smoke_test.py` must have run at
least once in a torch + transformers (+ peft) environment, named in the
implementer's summary, before this work is called done. This critique
cannot confirm that from this machine.

## Scope note

The delta is exactly the two accepted round-5 dispositions plus their
tests; no contract file touched, no RESEARCH_SPEC pending decision
affected, no drift into unrelated code. From this reviewer's side the
implementation has no open code findings — the environment gate is the
only outstanding item.
