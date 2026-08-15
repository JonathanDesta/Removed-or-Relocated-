# gpu-verification-fixes.critique-2 — Plan critique (round 2)

Scope: revised `planning/gpu-verification-fixes.md` (revision 2), reviewed
against its round-1 dispositions, `planning/priorities.md` priority 1,
`RESEARCH_SPEC.md`, `INTERFACES.md`, and the current implementation and tests.
No experiment was launched. No plan, earlier critique, product-code, test,
contract, or spec file was edited; this critique is the only file added.
Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## Round-1 disposition audit

| Round-1 finding | Revision-2 status |
|---|---|
| F1 — backend identity | **Incomplete.** Rows resume/group separately in the planned eval/metrics changes, but figures and capability/Gate-1 paths remain backend-blind (new F1/F2). |
| F2 — lesioned eager helper | **Scope corrected.** The false lesioned-checkpoint advertisement is withdrawn, the adapter branch gets coverage, and the still-undefined lesion integration is explicitly deferred as P6. The new adapter test does not prove application (new F7). |
| F3 — WikiText provenance | **Incomplete.** A pin is now decided, but the tests do not prove it was used or recorded (new F3); the tokenizer/model side remains mutable (new F4); and Step 1 contradicts itself about whether the revision is pinned (new F5). |
| F4 — handle retention | **Resolved in design.** The handle drops its references, has a targeted weakref/idempotence test, and rung 3 deletes handles and collects before reload. |
| F5 — stale canonical command | **Resolved in design.** The runner docstring/argparse help joins the canonical-command update and is in definition-of-done checks. |
| F6 — missing scenario split | **Resolved.** The call is now `get_scenarios("selection", n=1)` and a token-length mismatch escalates. |
| F7 — conflated bypass warning | **Partially resolved.** The plan now states the two invariants correctly, but its exact refusal-message draft still conflates them (new F10). |

---

## High severity

### F1. The figures track still computes A_l across mixed attention backends

**Plan lines 54-62, 88-96, and 247-261;
`src/algoverse/metrics.py:536-580`; `src/algoverse/figures.py:57-96,
203-287`.** Step 3b adds `attn_implementation` to `metrics._run_key` and
`GEN_CONFIG_KEY_FIELDS`, but the paper's layer-curve path does not use that
tuple. `figures._gen_identity` independently reproduces the generation
identity and still returns only bypass implementation, quantization, sampling,
token limit, dtype, and device type. It is then used directly by
`split_base_and_sweep` to decide which intact and bypassed rows are comparable.
Neither `src/algoverse/figures.py` nor `tests/test_figures.py` appears in the
revision's module map or Step-3b test list.

Failure scenario: the intact M_D baseline was generated under sdpa and a layer
sweep under eager. `metrics.summarize_runs` separates the runs as newly intended,
but `figures.layer_curve` groups them and reports a real-looking A_l that can
affect layer selection. This was reproduced read-only with the current figures
path: otherwise identical synthetic base/sweep rows with sdpa/eager respectively
produced one point with `A_l=0.5000000000000001` and
`baseline_mismatch=None`. Thus all planned Step-3b tests can pass while the
paper-facing curve violates the new identity decision. *Confidence: high.
Severity: high.*

### F2. Capability rows and the Gate-1 verdict remain blind to attention backend identity

**Plan lines 54-62, 88-90, and 247-264; `scripts/run_baseline.py:150-168`;
`src/algoverse/eval.py:710-760,763-845,895-914,1081-1090`;
`src/algoverse/metrics.py:412-444`.** The human decision is implemented only
for negotiation `rows.jsonl`. The capability runner's `run_meta` contains no
load profile or `attn_implementation`; benchmark configs contain only
limit/batch size/seed/lm-eval, and perplexity config contains only its slice and
window settings. `_competence_done` therefore resumes a capability metric
without comparing backends. Later, `gate1_report` throws away all competence-row
metadata and retains only value, stderr, and config, so the Gate-1 comparability
check cannot inspect a backend even if a caller added it only to top-level
`run_meta`.

Failure scenario: M_0 capability metrics run under sdpa, then a stack/default or
loader-path change makes M_D run under eager. MMLU/GSM8K greedy answers and
WikiText logits may change with the backend — the premise for the newly ratified
identity decision — but the configs compare equal and Gate 1 can PASS or FAIL on
a mixed-backend delta. The planned resume and `_run_key` tests do not touch
`competence.jsonl` or `gate1_report`. *Confidence: high. Severity: high.*

### F3. The acceptance suite can pass when neither the WikiText revision pin nor its provenance is implemented

**Plan lines 46-53, 114-145, 370-388, and 445-459;
`tests/test_bypass.py:149-185`; `src/algoverse/eval.py:856-914`.** The real
loader tests assert shape, repeatability, a prefix relation, and the first
heading. The rung-3 check asserts perplexity behavior. None asserts that
`load_dataset` received the `revision` keyword. Because current `main` resolves
to the same b086... revision selected by the plan, an implementation that fixes
only the namespaced id and silently omits `revision=` returns the same current
bytes and passes every one of those tests.

The provenance half is likewise untested. The existing perplexity-row test
asserts only that `nll_mean` is top-level and absent from `config`; it never
asserts `dataset_id` or `dataset_revision`. The new loader-only suite cannot
inspect a competence row. Failure scenario: both revision constants are defined
but unused, or the load is pinned but the config fields are omitted. Rungs 1-3
all satisfy the definition of done, yet later M_0/M_D comparability is again
mutable or unauditable. This is an acceptance check passing the exact broken
state the round-1 F3 resolution was meant to rule out. *Confidence: high.
Severity: high.*

### F4. Pinning WikiText bytes still does not pin the scored token sequence or the same-model comparison

**Plan lines 114-145 and 370-388; `src/algoverse/models.py:218-277`;
`src/algoverse/eval.py:99-145,856-869,1081-1090`;
`scripts/run_baseline.py:150-168`; `RESEARCH_SPEC.md:224-225,293`.**
`load_wikitext_slice` is defined in tokens, not bytes. The loader continues to
resolve `AutoTokenizer.from_pretrained(model_id)` and the model itself from the
mutable model-repository `main`, with no shared revision argument. The official
Qwen history shows `tokenizer_config.json` changed across commits:
<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/commits/main/tokenizer_config.json>.
A tokenizer change can alter which source-text boundary becomes token 20,000
even when the WikiText commit is fixed.

Negotiation rows record the model config's resolved `_commit_hash`, but Gate 1
does not compare that field between M_0 and M_D; capability rows do not receive
it at all, and no tokenizer revision is recorded anywhere. Failure scenario:
M_0 is loaded before a model/tokenizer `main` change and M_D afterward. The
experiment no longer measures the spec-required same-model/same-tokenizer
delta, but all metric configs and the Gate-1 publishability checks still match.
The dataset pin eliminates one mutable input, not the complete quantity.
*Confidence: high on the unguarded path; severity: high.*

## Medium severity

### F5. Step 1 gives contradictory binding instructions about the newly ratified revision pin

**Plan lines 43-53, 114-145, and 493-503.** The decision log, constants,
`load_dataset` call, config fields, and later P2 note all say the WikiText data
revision is pinned. Immediately after specifying those fields, however, lines
141-142 say, “The revision hash is deliberately not pinned (P2).” This is the
superseded round-1 wording, but it sits inside the ordered implementation step
rather than historical context.

Failure scenario: an implementer follows the nearest prose instruction and
treats the constants as provenance-only or omits `revision=`, restoring the
high-severity moving-data defect; the inadequate acceptance tests in F3 do not
catch that interpretation. The plan therefore is not decision-complete on a
scientifically consequential input despite the human having made the decision.
*Confidence: high. Severity: medium-high.*

### F6. The row-resume backend test is assigned to a pure suite that cannot currently exercise the guard

**Plan lines 247-264 and 354-364; `tests/test_eval_pure.py:1-30`;
`tests/test_bypass.py:485-620`; `src/algoverse/eval.py:296-460`;
`src/algoverse/utils.py:1-27`.** The revised plan says the resume-identity case
is a rung-1 test in `test_eval_pure.py`. The actual guard is inline inside
`run_negotiation_eval`; existing field-coverage tests therefore live in the
ML-gated `test_bypass.py`. Calling the runner reaches `algoverse.utils`, whose
module-top imports include numpy and torch. There is no pure identity helper for
`test_eval_pure.py` to call, and the revision does not specify extracting one.

An implementer must invent an unplanned refactor or silently move the test to
rung 2. Otherwise the six stated stdlib suites can pass while only the separate
`metrics._run_key` grouping case was added; the resume refusal itself remains
untested. This is loud incompleteness rather than a silent numerical result, but
it undermines the stated acceptance for the newly ratified identity guarantee.
*Confidence: high. Severity: medium.*

### F7. The PEFT test cannot establish that the saved adapter was applied

**Plan lines 410-420; `planning/gpu-verification.md:371-394`;
`planning/gpu-verification.record.md` C6.** The new test says to create and save
a tiny LoRA adapter, then assert only that the reload returns a PeftModel, the
eager config check survives wrapping, and attention reading succeeds. Fresh
LoRA `lora_B` matrices are zero-initialized, so that adapter has no numerical
effect. The earlier GPU-verification plan explicitly filled every `lora_B`
nonzero because a zero-delta adapter cannot distinguish “loaded and applied”
from “silently dropped,” and C6 recorded `adapter_effect=True` as a gate.

Failure scenario: the wrapper/config is constructed but the saved delta weights
are lost, misrouted, or never affect the base model. The proposed test still
passes because a zero adapter and no effective adapter yield the same model and
valid attention tensors. This does exercise PEFT wrapping and the eager
self-check, but not the behavior claimed by the test name
`test_eager_interp_loader_applies_adapter`. *Confidence: high. Severity:
medium.*

## Low severity

### F8. A missing recorded backend does not always refuse as the screening paragraph claims

**Plan lines 247-264; `src/algoverse/eval.py:427-431`;
`src/algoverse/metrics.py:546-558`.** The planned implementation extends the
existing `.get()` comparison. If a legacy row lacks
`load_profile.attn_implementation` and the current model resolves that field to
`None`, both sides compare as `None` and resume is accepted. Summary grouping
similarly merges a missing field with an explicitly recorded null. The proposed
test covers sdpa-versus-eager, not missing-versus-null.

The three current research families under the verified transformers stack
resolve concrete backend strings, so occurrence is limited; nevertheless the
blanket “rows lacking the field refuse resume” guarantee is false and a future
version/model can turn unknown provenance into accepted identity. *Confidence:
high. Severity: medium-low.*

### F9. `test_interp.py` cannot provide the promised PEFT loud-skip behavior without an unplanned runner change

**Plan lines 410-420; `tests/test_bypass.py:790-825`;
`tests/test_interp.py:185-213`.** The plan says to follow test_bypass's
PEFT-skip convention. That runner imports/catches `unittest.SkipTest`, counts
skips, and prints a partial-verification banner. The interp runner catches every
`Exception` as a failure and has no skip counter/banner; `unittest.SkipTest` is
an `Exception`. The revision neither specifies the runner change nor increments
scope for it.

The sanctioned rung-2 venv currently has PEFT, so definition-of-done can still
reach 7/7. On a peft-less standing-suite run, however, the promised “skips
LOUDLY, never passes vacuously” behavior is not implementable by simply copying
the test_bypass test convention. *Confidence: high. Severity: low.*

### F10. The exact warning and guard-test designs do not fully verify the round-1 F7 resolution

**Plan lines 147-171, 283-326, and 390-397.** The revision correctly states
that hidden-state capture is version-dependent while bypassed-layer attention
is always real but causally dead. Its exact `_refuse_if_bypassed` message draft
nevertheless says whether “output_hidden_states/attentions” reflect the bypass
is version-dependent, folding the attention invariant back into “these.” The
public readers still refuse, so this remains documentation inconsistency rather
than a wrong returned value.

Separately, the clean-diagnostic test asserts only that the message contains
`"sdpa"`, but the proposed message contains that literal in its static
parenthetical regardless of the interpolated backend. The test can pass if the
actual resolved-backend value is missing or wrong, despite the revision's claim
that the message names it. Rung-3 F2(i) uses the same broad wording. *Confidence:
high. Severity: low.*

---

## Verification boundary

- Read-only inspection covered the revised plan and round-1 dispositions,
  normative spec and interface contract, negotiation/capability identity paths,
  figures, model/tokenizer loading, and all referenced test runners.
- A read-only synthetic diagnostic executed on the rung-2 venv and confirmed
  that `figures.layer_curve` accepts an sdpa baseline plus eager bypass rows,
  returning `A_l=0.5000000000000001` with no mismatch.
- The official Qwen model/tokenizer commit history was checked to verify that
  the unpinned model repository has revision history, including changes to
  `tokenizer_config.json`.
- No paper quantity, training run, benchmark, or GPU job was launched. No
  result file was written. Existing user changes in `RESEARCH_SPEC.md`,
  `planning/gpu-verification-fixes.critique-1.md`,
  `planning/gpu-verification-fixes.md`, and `planning/priorities.md` were left
  untouched.

---

## Disposition (planner, 2026-08-15 — applied as plan revision 3)

All 10 findings ACCEPTED in full or part; none rejected; no new escalation
required. F1/F2/F4 are completions of the two round-1 human decisions
(attn_implementation identity; pinned provenance) across consumers those
decisions logically cover — figures pairing, competence rows, and the Gate-1
comparability check; F4's unpinned-model-repo residual is FLAGGED as new
pending P7 rather than resolved. The planner independently confirmed the
figures `_gen_identity` omission (and an additional hazard the critique did
not name: `_mismatch_fields` zips `GEN_CONFIG_KEY_FIELDS` names against the
identity tuples, so a metrics-only field addition would also skew mismatch
naming — the three definitions must move in lockstep), the skip-blind
test_interp runner, the stale line-141 sentence, and the backend-free
benchmark configs before accepting.

| Finding | Disposition | Resolution |
|---|---|---|
| F1 | **Accepted** | Step 3b extended to figures: `_gen_identity` gains `attn_implementation` in the SAME position as `metrics._run_key`'s derived tuple and `GEN_CONFIG_KEY_FIELDS` (the three definitions are zipped together by `_mismatch_fields`; the plan now names the lockstep requirement). tests/test_figures.py gains the critique's reproduced case: sdpa baseline + eager sweep rows must not pair, and `baseline_mismatch` must name `attn_implementation`. figures.py and test_figures.py join the module map. |
| F2 | **Accepted** | New Step 3c: the three competence `metric_config`s (mmlu, gsm8k, wikitext2_ppl) gain `"attn_implementation"` derived from the live model config. This flows through `_competence_done`'s config-identity comparison AND `gate1_report`'s config-comparability check (which strips only `batch_size`) with zero gate1 code change — mixed-backend competence deltas refuse loudly. Rung-1 test: same-metric rows differing only in that config field → publishability error; rung-2: the perplexity row records it. |
| F3 | **Accepted** | Two acceptance holes closed: (a) new wikitext-suite test monkeypatches `datasets.load_dataset` with a recording stub (canned rows, no network) and asserts the call passes the namespaced id, config name, test split, AND `revision=WIKITEXT_DATASET_REVISION` — an implementation that omits the pin now fails even while `main` still equals the pinned hash; (b) the rung-2 perplexity-row test additionally asserts `config["dataset_id"]`/`config["dataset_revision"]` equal the constants (valid despite the monkeypatched slice — the fields are added in `compute_perplexity`). |
| F4 | **Accepted in part + flagged (P7)** | Step 3c also adds `"model_revision"` (the config's resolved `_commit_hash`, as negotiation rows already record and guard) to the three competence configs, so Gate-1 refuses cross-revision competence deltas loudly — restoring the spec's same-model/same-tokenizer premise as a GUARD. Actually PINNING the model/tokenizer repo revision is a policy change the plan does not make: new pending P7, recommended for decision at the P2 stack-pinning moment. |
| F5 | **Accepted** | The stale "revision hash is deliberately not pinned (P2)" sentence (a round-1 leftover sitting inside ordered Step 1) is deleted and replaced with a pointer to the pinned-revision decision. The plan is now decision-consistent on the pin. |
| F6 | **Accepted** | The resume-refusal backend test is reassigned to rung 2 (test_bypass.py, beside the existing inline-guard coverage); only the `_run_key` grouping case stays rung-1 (metrics is stdlib-importable). Step 7 and the definition of done updated; no unplanned refactor is implied. |
| F7 | **Accepted** | The adapter test fills `lora_B` with nonzero values before saving (C6 precedent: a zero-delta adapter cannot distinguish applied from dropped) and additionally asserts the adapter has a numerical effect (logits differ from the bare base on the fixed input) before the PeftModel/eager/attention assertions. |
| F8 | **Accepted** | The guard's comparison for `attn_implementation` (only) becomes equal-AND-not-None: a missing or null backend on either side refuses resume — unknown provenance never silently matches. Test adds the missing-vs-null case. The screening prose is corrected (the round-1 blanket claim was false as written). Residual documented: `_run_key` GROUPING of two both-None rows still merges them — grouping cannot refuse; the loud resume guard covers the write path. |
| F9 | **Accepted** | The test_interp `__main__` runner change is now specified in scope: catch `unittest.SkipTest` separately from `Exception`, count skips, and print the partial-verification banner mirroring test_bypass.py:803-819 — otherwise a peft-less run would report the promised loud skip as a FAILURE. |
| F10 | **Accepted** | (a) The `_refuse_if_bypassed` message draft is redrafted to keep the two invariants separate in the message itself ("whether output_hidden_states reflects the bypass is version-dependent, and the bypassed block's attention maps are real but causally dead on every version"), preserving the tested substrings. (b) The sdpa guard test asserts the INTERPOLATED backend — the `%r`-quoted `'sdpa'` — which the static "(sdpa and flash backends do not)" parenthetical cannot satisfy, so a missing/wrong interpolation fails; rung-3 F2(i)'s check is tightened identically. |
