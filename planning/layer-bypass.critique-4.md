# layer-bypass.critique-4 — Implementation critique (round 4)

Scope: the implementation diff for `planning/layer-bypass.md` (frozen at
revision 6), reviewed against that plan, `RESEARCH_SPEC.md`, and
`INTERFACES.md`. Eleven files reviewed: Notebook Setup.ipynb,
scripts/run_baseline.py, src/algoverse/{eval,interp,metrics,models,tasks,
utils}.py, tests/{test_bypass,test_metrics,test_scoring}.py. Read-only
review; no code edited. Findings are implementation findings against the
frozen plan, per the Round-3 freeze rule.

Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## Medium severity

### F1. A torch-but-no-peft run still prints PASS for the PEFT test

**tests/test_bypass.py:308-310, 619-620, 622-630.** When `peft` is absent,
`test_bypass_on_peft_wrapped_model` returns immediately (line 309-310), and
the `__main__` runner then prints `PASS test_bypass_on_peft_wrapped_model`
followed by `ALL TESTS PASSED`. The loud warning the plan requires
(critique-3 F9) does print at line 620 — but it scrolls past first, and the
per-test output that follows contradicts it: a PASS line for a test that
executed zero assertions. Under pytest the same environment reports the test
green with no skip marker at all (plain `return`, not `pytest.skip`).
Failure scenario: a Colab CPU cell without peft runs the suite, the operator
scans the per-test lines or the final `ALL TESTS PASSED`, and wrapper
coverage is recorded as verified when it never ran — the exact
"reads as full verification" outcome critique-3 F9 was accepted to prevent.
Minor compounding nit: the message says "(3 not run)" but only one gated
test function exists (it loops three families); anyone counting functions
will find the number wrong. *Confidence: high. Severity: medium.*

### F2. Competence resume treats benchmark configuration as result, not identity

**src/algoverse/eval.py:633 (`result_fields` includes `config`), 645-655,
691-706, 776-787.** `_competence_done` guards only run_meta labels; the
benchmark configuration (`gsm8k_limit`, `mmlu_limit_per_subtask`, lm-eval
`seed`, perplexity `n_tokens`/`stride`) lives in each row's `config`, which
is excluded as a result field. Failure scenario: session 1 runs
`run_lm_eval_benchmarks` with `gsm8k_limit=50` for a quick look; session 2
runs the canonical limit-200 command into the same competence.jsonl; the
skip logic sees (run_id, metric) present with matching run_meta, prints
"gsm8k_exact_match already complete, skipped", and `gate1_report` consumes
the limit-50 value as the gate's capability bound — no refusal, no warning,
and the row's `config` is the only trace. This conforms to the frozen plan's
letter ("different run_meta labels"), so it is a residual gap rather than a
deviation — but it is a live path to a quietly non-canonical reported
number, and one more comparison line inside `_competence_done` would close
it. *Confidence: high on behavior. Severity: medium.*

### F3. `_competence_done` lost its planned torch-free testability, and its only tests are ML-gated

**src/algoverse/eval.py:23 (module-level `import torch`), 621;
tests/test_bypass.py:38, 525-546.** The plan specified the competence
resume/identity logic as "one small pure-logic helper … torch-free testable
with synthetic files." As placed, the helper lives in eval.py, which imports
torch at module top, so it cannot be imported — let alone tested — without
the ML stack; its only tests sit in test_bypass.py under the HAVE_ML_STACK
guard. Failure scenario: on the torch-less machines that run the fast suites
(including the one this critique ran on), the competence guard has zero
executed coverage; a regression in it is invisible until a Colab session
trips over it mid-run. The logic itself reads correct; this is a
test-architecture deviation from the plan's stated intent. *Confidence:
high. Severity: medium-low.*

---

## Low severity

### F4. Adapter digest: empty-directory constant, and `*.bin` sweeps non-adapter files

**src/algoverse/eval.py:55-81.** (a) A directory containing none of the
expected files digests to sha256 of empty input — a constant, meaningless
but non-None identity. A typo'd `adapter_path` bookkeeping value pointing at
an existing non-adapter directory records that constant instead of failing
or recording None, and two different wrong paths become indistinguishable.
(b) `path.glob("*.bin")` also sweeps files like `training_args.bin` that HF
training dirs commonly contain, making the digest stricter than adapter
identity: a re-save with byte-identical adapter weights and config but a
different training_args.bin refuses resume. Both misfire in the safe
direction (refusal / never silent mixing), and the plan itself named
`*.bin`, so this is hardening feedback, not a deviation. *Confidence: high.
Severity: low.*

### F5. Startup probe failure reports no cause

**scripts/run_baseline.py:84-95; src/algoverse/tasks.py:562-565.** The
fail-fast probe is implemented and correctly placed before model load, but
`llm_extract_offer` swallows every exception, so a failed probe surfaces
only as "LLM fallback startup probe failed; no generation was run" — a
misconfigured `OPENAI_BASE_URL`, an exhausted quota, and a wrong Azure
deployment name are indistinguishable, and the operator debugs blind on
Colab. The plan's requirement (refuse to start) is met; the diagnostic is
the gap. *Confidence: high. Severity: low.*

### F6. Mid-run adapter overwrite is caught only at the next resume

**src/algoverse/eval.py:272-282.** `gen_config` (including
`adapter_digest`) is derived once per call, and every row of the session
carries it. If a Drive sync overwrites `adapter/latest` two hours into an
eight-hour single-session run, rows generated after the overwrite still
carry the pre-overwrite digest, and no refusal occurs until a later resume —
if the run finishes in one session, never. The plan's claim that the digest
"catches a Drive-sync overwriting adapter/latest mid-run" holds only across
session boundaries. Worth one sentence of operator documentation; re-hashing
per batch would be the airtight-but-costlier alternative. *Confidence: high.
Severity: low.*

### F7. The same run_id in two different out_paths is unguarded and pools in summarize_runs

**src/algoverse/eval.py:287-310; src/algoverse/metrics.py:449-485.** The
manifest is a per-file sidecar (the ratified design), so reusing a run_id
across two results directories passes both files' guards independently.
Failure scenario: `m0-baseline` run into `results/a` with `--n 100` and into
`results/b` with `--n 200`, identical identity fields; an analyst who loads
both files feeds `summarize_runs` rows that share the entire 17-field group
key and pool into one summary row over a mixed cohort — the same class of
silent pooling the run_id ratification exists to prevent, one level up.
Operator discipline (fresh run_id per request) is the current control.
*Confidence: high. Severity: low.*

### F8. Fallback-attribution fallback in the tuple-compat shim names the requested model, not the returned one

**src/algoverse/tasks.py:418-425.** The non-tuple branch of the
compatibility shim attributes `extraction_method` to the requested/resolved
model rather than an API-returned identifier. In production
`llm_extract_offer(..., return_model=True)` always returns a tuple, so the
branch is reachable only by test doubles and has no live effect — but that
also means a production branch exists solely to accommodate tests, and if a
future refactor ever makes it reachable, the per-row drift-visibility that
critique-3 F8 bought would silently degrade to the requested alias.
*Confidence: high (currently unreachable in production). Severity: low.*

---

## Style

### F9. Dead imports in the new test file

**tests/test_bypass.py:12-13.** `copy` and `json` are imported and never
used — the same class of lint the repo's last sanity-commit removed.
*Confidence: high. Severity: style.*

### F10. Gemma2 fixture passes a config field Gemma2 may not define, via silently-absorbing kwargs

**tests/test_bypass.py:83-90.** `sliding_window_pattern` is not a
documented Gemma2Config field (Gemma2 alternates sliding/global attention
internally; the field name belongs to a later family); PretrainedConfig
stores unknown kwargs silently, so the line is likely inert and implies
control the fixture may not have. Similarly, `config._attn_implementation =
"eager"` relies on a private attribute — it works on current transformers
and any breakage would surface as loud numeric test failures, not silent
passes. *Confidence: medium (version-dependent). Severity: style.*

### F11. Minor message/doc inconsistencies

**src/algoverse/models.py:89-93** — the bool/non-int branch of the
`install_bypass` error omits the offending value (`got %r` appears only in
the range branch); the tests assert only "4-layer", so both pass.
**src/algoverse/eval.py:464-474** — the smoke_test docstring still
describes only the intact leg; the bypass leg, guard proof, and residue
check appear only in the PASSED print. **src/algoverse/eval.py:14-15** —
pre-existing (not this diff): the module docstring still says benchmarks,
perplexity, and Gate-1 "join this module in the next build stage" though
they are already here. *Confidence: high. Severity: style.*

---

## Verification record

Executed on the review machine (macOS, system Python 3, **no torch, no
peft, no numpy** — no ML environment exists on this machine):

- All six test files run directly. `test_data`, `test_metrics` (including
  the three new grouping tests and the legacy-None test), 
  `test_perplexity_count`, `test_scenarios`, `test_scoring` (including the
  new `test_fallback_failure_is_recorded_per_row`): **ALL PASSED**.
- `tests/test_bypass.py` on this torch-less interpreter prints exactly
  `SKIPPED: 0 of 14 bypass acceptance tests ran — this is NOT verification`
  and exits 0 — the loud-SKIP path behaves as specified.
- `append_jsonl` hardening exercised directly with byte-level inspection
  (numpy stubbed): a torn final fragment is isolated on its own line, the
  next appended row is readable by the tolerant reader, empty-file and
  str-path edges behave. Reproduces critique-3 F1's scenario as FIXED.
- `Path("rows.jsonl").with_suffix(".manifest.jsonl")` confirmed to produce
  `rows.manifest.jsonl`.
- INTERFACES.md contract text confirmed to match the plan's Module-map
  claims: `..., seed, train_seed, gen_config` row schema; summarize_runs
  dimensions including run_id, generation profile, train_seed; the
  bypassed_layer range note; `--llm-fallback` in the canonical command.

**NOT verified here, and unverifiable on this machine: the 14 bypass
acceptance tests, `python3 -c "import algoverse.interp"`, and
`scripts/smoke_test.py`.** The plan's environment gate says this work may
not be called done unless those ran at least once in a torch + transformers
(+ peft) environment and the implementer's summary names that environment.
This critique cannot confirm that happened; the downstream pass should
check the implementer's summary against the gate before accepting the
mechanism as verified. In particular, the hidden-states index convention
for Gemma2's final-norm entry (test 4) and the Gemma2 eager-attention
fixture are exactly the kind of claim only execution settles.

## Conformance notes (what matched the plan — spot-checked, not exhaustive)

- models.py mechanism: hook semantics (kwargs-then-args lookup, loud
  non-tensor failure, tuple-shape preservation), marker on the shared
  layers object, bool-before-int validation, negative-index rejection,
  double-install refusal, idempotent remove that clears the marker only
  when it owns it. BYPASS_IMPL string and docstring WHY-content (including
  the critique-1 F8 observability warning) match the plan.
- eval.py: cross-check at top of function; `_derive_gen_config` factored as
  the single helper the tests reuse; quant-contradiction rule exactly as
  specified (None declares nothing); scoring-trio normalization on both
  sides; guard active regardless of `resume`; identity error precedes
  append-only, which precedes sampled-resume (asserted by test 12(b)'s
  wording checks); manifest with set equality, legacy-refusal wording, and
  tolerant reads; every-row scan (multi-row case tested); attn_implementation
  and version stamps recorded but excluded from the guard, per the ratified
  audit-only rule; smoke unlinks both sidecars.
- All 14 planned test functions present with the planned isolation details
  (valid in-range layer for test 10; sampled-resume isolation via
  do_sample=True preseed; torn-line append-then-read proof; independent
  hand-written oracle constants; adapter-config one-byte digest change).
- metrics: RUN_KEY_FIELDS + derived gen_config key exactly the ratified
  eleven-plus-six; no key collisions with tau_with_ci/task_competence
  outputs (checked); summarize_runs still has no production callers.
- run_baseline: probe before model load; loud BYPASS INSTALLED print;
  handle deliberately kept for process lifetime; run_meta gains train_seed
  and derived bypass_impl; dated-snapshot flag default with library
  defaults untouched; notebook gains `openai`.
- interp.py: dead duplicate docstring gone, bypass claim removed, verbatim
  `_decoder_layers` move with a one-line WHY import comment.
- No RESEARCH_SPEC pending decision is resolved by this diff; the one
  remaining Open decision (Stage-3 stacked bypass) is untouched, and
  `install_bypass` correctly still enforces single-bypass.

## Out-of-scope working-tree observations

Three working-tree changes were not in the assigned eleven-file diff:
INTERFACES.md and RESEARCH_SPEC.md edits whose content matches the plan's
record of human ratifications (contract touch-ups and the two spec
sections), and a .gitignore change un-ignoring AGENTS.md, CLAUDE.md,
roles/, and Prompts.txt. All three read as the human's; the final-review
pass should confirm attribution, since agents may not edit the contract.

---

## Implementer disposition (revision round 4)

Recorded before code edits, per `roles/3-implement.md`.

| Finding | Disposition | Reason / action |
|---|---|---|
| F1 PEFT skip reported as PASS | Accepted | The warning is contradicted by the direct runner and pytest. Raise a real skip, teach the direct runner to report it without PASS, and make its final status explicitly partial when PEFT is absent. |
| F2 competence config omitted from identity | Accepted | Metric configuration changes the reported measurement. Compare each completed metric against its requested config, excluding derived result data such as perplexity `nll_mean`. |
| F3 competence guard not torch-free testable | Accepted | This deviates from the frozen plan. Make heavy eval imports function-local so `_competence_done` is importable on the torch-less fast path, and add synthetic fast-suite coverage. |
| F4 adapter digest edges | Accepted | Empty/non-adapter directories should produce no digest, and unrelated training artifacts are not adapter identity. Hash only PEFT adapter weight filenames plus `adapter_config.json`; return None without adapter weights. |
| F5 startup probe hides cause | Accepted | Keep ordinary scoring's swallow-and-record behavior, but add opt-in exception propagation for the fail-fast startup probe and preserve its cause in the runner error. |
| F6 mid-run adapter overwrite | Rejected | Overwriting checkpoint files does not mutate the already-loaded in-memory model. Later rows still come from the originally loaded adapter and correctly retain its original digest; a later reload derives the new digest and refuses resume. Per-batch rehashing would add cost without detecting a live-model change. |
| F7 same run ID in different output paths | Rejected | The binding contract fixes the artifact location as `results/<run_id>/rows.jsonl`; duplicate run IDs in multiple directories violate that layout. No repository-wide run registry exists to enforce global uniqueness, and adding one exceeds this frozen scope. |
| F8 tuple compatibility attribution | Accepted | The production scorer explicitly requests `(value, response_model)`. Remove the scalar compatibility branch so future changes and test doubles must preserve returned-model provenance. |
| F9 dead test imports | Accepted | Remove the unused imports; no behavior change. |
| F10 inert Gemma2 config field | Accepted in part | Installed `Gemma2Config` confirms `sliding_window_pattern` is absent, so remove it. Keep the eager-attention setting required by the plan; any version break is loud. |
| F11 message/doc inconsistencies | Accepted | Include the offending layer value and update module/smoke docstrings to describe current behavior. |

No finding touches a pending research decision, so none is escalated.
