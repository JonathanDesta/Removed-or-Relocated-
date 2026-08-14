# layer-bypass.critique-5 — Implementation critique (round 5)

Scope: the round-4 fix diff (eight files: planning/layer-bypass.critique-4.md
disposition, scripts/run_baseline.py, src/algoverse/{eval,models,tasks}.py,
tests/{test_bypass,test_metrics,test_scoring}.py), reviewed against the
frozen plan (rev 6), the round-4 findings and dispositions, RESEARCH_SPEC.md,
and INTERFACES.md. Read-only review; no code edited.

Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## Round-4 fix verification

Every accepted disposition is implemented, and each was verified either by
execution here or by reading the exact diff:

| Round-4 finding | Status | How verified |
|---|---|---|
| F1 PEFT skip reported as PASS | FIXED | `test_bypass_on_peft_wrapped_model` now raises `unittest.SkipTest` (a real skip under pytest); the direct runner prints `SKIP <name>: …`, counts skips, and the final status becomes "ALL EXECUTED TESTS PASSED; N SKIPPED — FULL VERIFICATION NOT COMPLETE". No PASS line, no unconditional "ALL TESTS PASSED". The count wording is now accurate ("3 family cases"). Read; skip path itself needs an ML env to execute (see gate note). |
| F2 competence config not identity | FIXED (with one over-reach — see new F1 below) | `_competence_done` gains a `config` parameter compared per-metric against the recorded row config; refusal names `config.<field>`. Executed torch-free here: limit-50-then-200 now REFUSES with `config.limit` named; perplexity's derived `nll_mean` is correctly excluded (guard iterates requested keys only). Recorded and guarded config are the same dict, so they cannot drift. |
| F3 competence guard not torch-free | FIXED | eval.py no longer imports torch/utils at module top (torch is function-local in generate_batch/smoke_test/compute_perplexity; utils imports are function-local). Verified here: `from algoverse.eval import _competence_done` succeeds with neither `torch` nor `numpy` entering `sys.modules`. New torch-free test `test_competence_resume_guards_identity_and_metric_config_without_torch` executes in the fast suite and also pins the `*_version` audit-only exclusion. The module-top import in test_metrics.py doubles as a tripwire: re-adding a heavy import to eval.py breaks the fast suite loudly. |
| F4 adapter digest edges | FIXED | Digest now requires `adapter_model.safetensors`/`adapter_model.bin` (None otherwise — empty/non-adapter dirs no longer digest), hashes only weights + adapter_config.json, and ignores `training_args.bin` (oracle test asserts digest unchanged after adding it). Read; test is ML-gated. |
| F5 probe hides cause | FIXED | `llm_extract_offer` gains `raise_errors=True` (probe-only); the runner wraps the probe and re-raises with the original exception type/message chained. `test_fallback_probe_can_surface_original_configuration_error` (torch-free, fake openai module) executed here and passes, including the "quota exhausted" propagation case. Ordinary scoring keeps swallow-and-record. |
| F6 mid-run overwrite | REJECTED — disposition sound | Correct: overwriting checkpoint files does not mutate the loaded model; the digest describes what was actually loaded, which is what identity means. The round-4 scenario was wrong on this point. |
| F7 run_id across out_paths | REJECTED — disposition sound | The contract fixes `results/<run_id>/rows.jsonl`; duplicate run_ids across directories already violate the layout, and a global registry exceeds the frozen scope. |
| F8 tuple-compat shim | FIXED | The scalar branch is gone; `score_response` unpacks the tuple directly, so a scalar-returning double now fails loudly. Doubles in test_scoring return tuples. Executed here. |
| F9 dead imports | FIXED | `copy`/`json` gone from test_bypass.py. |
| F10 Gemma2 config field | FIXED per disposition | `sliding_window_pattern` removed; eager-attention setting retained as planned. |
| F11 message/doc inconsistencies | FIXED | bool-branch error now carries `got %r`; eval module docstring and smoke_test docstring describe current behavior. |

---

## New findings

### F1. The competence config guard now refuses on `batch_size`, stranding the OOM-recovery resume

**src/algoverse/eval.py:714-719 (metric_config includes `batch_size`),
662-674; scripts/run_baseline.py:143-146.** The round-4 F2 fix guards every
key of the recorded metric config, including `batch_size` — the one field
the frozen plan explicitly ratifies as OPERATIONAL, not identity, for the
rows file ("Deliberately NOT guarded: `batch_size` (operational, recorded
per-row)"). lm-eval batch size does not change the measurement; limit and
seed do. Failure scenario (reproduced here with a synthetic file): a T4
session OOMs during benchmarks after gsm8k completed at `--batch-size 4`;
the operator restarts with `--batch-size 2` — the standard OOM recovery —
and `_competence_done` raises `config.batch_size` instead of skipping the
completed metric, so neither the finished metric is reused nor the pending
one run. The same `--batch-size` change resumes the rows file without
complaint, so the two guards now disagree about the same ratified
operational field, and the contract's "Everything resumes" sentence breaks
for the exact recovery path resume exists to serve. Failure direction is
loud (refusal, never silent wrongness). Fix shape: exclude `batch_size`
from the guarded comparison while still recording it — one key-filter,
mirroring the rows-file stance. *Confidence: high (reproduced). Severity:
medium-low.*

### F2. Partial verification exits 0 — status is textual only

**tests/test_bypass.py:626-637.** With skips present the direct runner
prints the loud partial-verification line but still exits 0, matching the
existing torch-less convention (also exit 0). Anything that checks exit
codes alone — a Colab `!python3 tests/test_bypass.py && echo ok` cell, a
future CI step — cannot distinguish full from partial verification. The
plan accepted exit-0 skips, so this is informational, not a deviation;
worth a thought if a CI step is ever added. *Confidence: high. Severity:
low/informational.*

### F3. Cosmetic: a successful extraction with a model-less API response records `llm:<provider>:None`

**src/algoverse/tasks.py:420-422.** With the compat shim removed,
`response_model` comes only from the API response; if a provider ever
returns `model=None` on a successful call, `extraction_method` records the
literal string `llm:openai:None` rather than falling back to the requested
model. Not observed with real providers; noted for completeness.
*Confidence: high on code path, low on occurrence. Severity: style.*

---

## Verification record

Executed on the review machine (macOS, system Python 3 — still **no torch,
no peft**; no ML environment exists on this machine):

- All six test files re-run directly: `test_data`, `test_metrics` (now
  including the torch-free competence-guard test), `test_perplexity_count`,
  `test_scenarios`, `test_scoring` (now including the probe
  cause-propagation test with a fake openai module) — **ALL PASSED**.
  `test_bypass.py` still prints the exact loud-SKIP line and exits 0.
- `from algoverse.eval import _competence_done` verified to import with
  neither `torch` nor `numpy` entering `sys.modules` (round-4 F3's fix,
  confirmed empirically).
- New finding F1 reproduced empirically: a completed gsm8k row with
  `config.batch_size=4` re-requested with `batch_size=2` (same limit/seed)
  raises `config.batch_size` instead of skipping.

**Still not verifiable here: the 14 bypass acceptance tests, the PEFT
SKIP-path output, `import algoverse.interp`, and the smoke test.** The
disposition's F10 note ("Installed `Gemma2Config` confirms
`sliding_window_pattern` is absent") implies the implementer has a
transformers environment, but the plan's environment gate still requires
the implementer's summary to name the environment that executed the ML
suite — this round touched ML-gated tests (PEFT skip, Gemma2 fixture, the
oracle's new digest cases), so a re-run there is required, not optional.

## Scope note

models.py, tasks.py, run_baseline.py, and eval.py changes stay within the
round-4 dispositions; no contract file was touched this round, and no
RESEARCH_SPEC pending decision is affected. The metrics.py grouping code
is unchanged; the test_metrics additions only extend coverage.

---

## Implementer disposition (revision round 5)

Recorded before code edits, per `roles/3-implement.md`.

| Finding | Disposition | Reason / action |
|---|---|---|
| F1 competence guard treats `batch_size` as identity | Accepted | The reproduction is valid. `batch_size` is ratified as operational and must remain recorded but unguarded. Mirror the rows guard and add tests proving batch-size-only resume succeeds while limit/seed mismatches still refuse. |
| F2 partial verification exits zero | Rejected | This is explicitly informational and matches the frozen plan's required exit-0 skip convention. The direct runner is human-readable verification output and now distinguishes partial coverage without claiming full success. Changing the exit contract would contradict the plan rather than fix a deviation. |
| F3 successful extraction can record model `None` | Accepted | `llm:<provider>:None` violates returned-model provenance even if current providers do not emit it. Treat a model-less result as `llm_failed:<provider>`; never substitute the requested model. |

No finding touches a pending research decision, and no rejected finding is
high-severity/high-confidence, so none is escalated.
