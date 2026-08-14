# layer-bypass.critique-2 — Plan critique, round 2

Scope: revision 3 of `planning/layer-bypass.md`, reviewed against the revised
`RESEARCH_SPEC.md`, `INTERFACES.md`, current code, Lad et al. v3, and the
paper's released intervention implementation. This round does not repeat a
round-1 finding merely because the implementation has not landed yet; it
reports distinct residual or newly introduced failure modes in revision 3.
The approved plan, contracts, and source were not edited.

Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## High severity

### F1. `resume=False` still violates the append-only row contract

**planning/layer-bypass.md:180-198; INTERFACES.md:18-34.** The new guard reads
existing rows regardless of `resume`, but it only rejects an identity
mismatch. The plan explicitly preserves the flag's current meaning: with
`resume=False`, already-completed `(run_id, scenario_id, condition)` triples
are not skipped. When the existing rows match the current identity, every
requested scenario-condition is therefore generated again and appended to
the same file under the same resume key.

That contradicts the binding schema's "one JSONL row per scenario x condition
per model" rule. Neither `incentive_gap` nor `deception_rate` deduplicates
rows; both count every copy. `_group_by_scenario` clusters duplicate rows for
the bootstrap but also retains and recounts every duplicate within a sampled
scenario.

Failure scenario: a completed deterministic run is invoked once with
`resume=False` to force a check. The file now contains two copies of every
row. If package drift, sampling, or an external scorer makes any repeated
response differ, tau becomes an accidental weighted average. Even when the
point estimate happens to remain unchanged, counts and any later uneven
repetition are wrong. The proposed tests exercise a mismatching
`resume=False` call, which raises before duplication; none exercises matching
identity with `resume=False`. *Confidence: high. Severity: high.*

### F2. The "full run identity" omits split and the actual scenario cohort

**planning/layer-bypass.md:180-198, 236-250; scripts/run_baseline.py:51-62.**
The guarded fields do not include `split`, requested scenario IDs, scenario
parameters, or a cohort fingerprint. `seed` alone does not identify the
cohort. In current code, `get_scenarios(split, n, seed)` can return different
sets for different `n` under the same seed. Locally, with the real pool size
305 and seed 42, the `n=20` and `n=100` samples share 19 rather than 20
scenarios; Python's `random.sample` changes strategy across sample sizes.

The finer `summarize_runs` grouping does not protect the runner's immediate
number: `run_baseline.py` passes every row returned by
`run_negotiation_eval` directly to `tau_with_ci` and `task_competence`, with
no split or cohort grouping.

Failure scenario 1: a selection run is resumed with the final split under the
same run ID. Scenario IDs are disjoint, so all final rows append. The runner's
printed tau pools selection and final data, breaching the final-pool firewall
inside the canonical workflow even though `summarize_runs` would later split
them.

Failure scenario 2: an `n=20` run is rerun as `n=100` with the same seed and
identity. Nineteen rows per condition are skipped and 81 per condition append,
leaving an unintended 101-scenario union rather than the requested cohort.
That union is then reported as the run's result. *Confidence: high. Severity:
high.*

### F3. Patch identity is absent from the resume guard

**planning/layer-bypass.md:180-198; src/algoverse/eval.py:110-131;
INTERFACES.md:22-34.** The evaluator API and binding row schema contain
`patch_layer` and `patch_source`, and the contract says patch and bypass are
different causal evidence. Neither patch field appears in the plan's guarded
identity list.

Failure scenario: a caller evaluates patch source A at layer 10, then invokes
the same run ID and scenarios for patch source B at layer 10. The guard sees
matching model/checkpoint/bypass/generation fields. With `resume=True`, every
scenario-condition is already in the documented resume key, so the B
intervention generates nothing and the function returns A's rows. A caller
can then label or analyze the returned result as B despite B never running.
With `resume=False`, the two distinct patch interventions append into one run
instead. Either branch breaks the contract's causal-evidence separation.
*Confidence: high. Severity: high.*

### F4. Scoring configuration is neither guarded nor fully recorded

**planning/layer-bypass.md:180-198; src/algoverse/eval.py:110-118, 174-181;
src/algoverse/tasks.py:386-430, 445-478.** The plan guards generation settings
but omits `use_llm_fallback`, `llm_provider`, and `llm_model`. Those arguments
can change `claimed_value`, validity, `deceptive`, and therefore tau without
changing the generated response. `extraction_method` records a provider only
when fallback succeeds; it does not record the extraction model, the disabled
case as a scoring configuration, or unsuccessful fallback attempts.

Failure scenario: a regex-only run is interrupted, then resumed with the LLM
fallback enabled. Previously unparseable replies stay invalid while otherwise
similar remaining replies can become valid and deceptive/honest. The run
passes the proposed identity guard and reports a tau computed under two
scoring regimes. Changing the fallback model under the same provider is even
less visible because successful rows retain the same `llm:<provider>` method
label. *Confidence: high. Severity: high.*

### F5. `summarize_runs` still pools scientifically different run configurations

**planning/layer-bypass.md:236-250, 337-342; RESEARCH_SPEC.md:114-116.** The
revised group key gains `split`, `seed`, and `gen_config.bypass_impl`, but it
still omits `gen_config.quant`, `do_sample`, `max_new_tokens`, runtime
versions, and any scenario-cohort identity. Because `run_id` is deliberately
excluded, rows from separate files/runs with different values for any of
those fields enter the same group.

These are not merely operational settings. Quantization changes the evaluated
model; sampling changes the response distribution; the token limit changes
truncation and invalid rates; and a different scenario cohort changes the
population. The runner's new within-run guard does not protect aggregation
across run IDs.

Failure scenario: a short exploratory `/v1` bypass run uses
`max_new_tokens=64` and produces many truncated invalid rows; a publication
run uses 256. If their model/intervention/checkpoint/split/seed fields match,
`summarize_runs` combines them and reports one tau and invalid-rate pair that
belongs to neither configuration. The same silent pooling occurs for 4-bit
and full-precision runs. *Confidence: high. Severity: high.*

### F6. Deliberate pooling across run IDs gives repeated runs a false bootstrap interval

**planning/layer-bypass.md:241-247, 337-342; src/algoverse/metrics.py:93-114,
148-224.** The plan not only excludes `run_id`; it adds an acceptance test
requiring otherwise-identical run IDs to pool. The metrics do not distinguish
disjoint multi-session partitions from repeated evaluations of the same
scenario-condition. Duplicate observations are all counted in the rate, but
the bootstrap unit remains the unique `scenario_id`, so repeated generations
within a scenario are never resampled as a source of variation.

This was reproduced locally with two run IDs for one shared scenario: run 1's
incentive response was deceptive, run 2's was honest, and both controls were
honest. `summarize_runs` returned `tau=0.5` with CI `[0.5, 0.5]`. The two runs
disagree maximally, yet the reported interval says the estimate has no
uncertainty because there is one scenario cluster containing both outcomes.

Failure scenario: two nominally identical repeated runs differ because of
sampling, hardware, or an allowed package upgrade. Pooling produces an
average while the scenario-only CI ignores between-run variation, supporting
an overconfident causal curve. "Multi-session accumulation" does not itself
require this behavior: the runner already preserves the same run ID when a
single run resumes across sessions. *Confidence: high. Severity: high.*

### F7. The guarded model identity can stay constant while the weights change

**planning/layer-bypass.md:180-198; src/algoverse/models.py:19-31, 33-79.**
The guard compares `model_id`, `adapter_path`, and caller-supplied
`checkpoint_step`, not the loaded artifacts' identities. A Hugging Face model
ID is loaded without a pinned revision, and a local/Drive adapter path can be
overwritten in place. Neither case changes the compared string.

Failure scenario: half a run is generated from `adapter/latest` at training
step 500. Training or a sync process overwrites that directory with newer
adapter weights while the caller retains `checkpoint_step=500`, then the run
resumes. Every guarded field matches and the remaining rows come from a
different model. The same risk exists if an upstream model repository changes
the revision resolved by an unpinned ID. The plan calls this a full identity
guard, but its identity values are labels rather than evidence of loaded
weights. *Confidence: high. Severity: high.*

### F8. `quant="none"` denotes materially different loader paths

**planning/layer-bypass.md:188-196, 429-431; src/algoverse/models.py:55-71.**
The guard treats `gen_config.quant` as sufficient model identity. Current
`load_model_and_tokenizer(..., quant="none")` loads fp16 with
`device_map="auto"` on CUDA, but fp32 with eager attention on CPU/MPS. Both
paths stamp the same quant label. They can also share the same torch and
Transformers version strings.

Failure scenario: an intact or bypassed `quant="none"` run starts on a CPU
machine and resumes on a CUDA machine, or separate runs from those machines
are sent to `summarize_runs`. The guard/grouping treats them as one evaluated
model even though dtype, device placement, and attention implementation
differ. Outputs and byte-identity behavior can differ, making the pooled
number irreproducible. *Confidence: high. Severity: high.*

### F9. The intact-model mismatch test can pass without testing the mismatch

**planning/layer-bypass.md:266-270, 315-322.** The fixtures have four decoder
layers, but `test_eval_rejects_unbypassed_model_with_bypass_bookkeeping`
passes `bypassed_layer=5`. That value is invalid independently of whether the
model has a bypass installed.

Failure scenario: the evaluator implementation accidentally omits the
model-state cross-check and merely validates that `bypassed_layer` is in
range. Test 10 still raises `ValueError` for layer 5 and passes, while an
intact model labeled with valid layer 1 proceeds and recreates the original
F4 wrong-row failure. The test therefore does not isolate the invariant it is
named as accepting. *Confidence: high. Severity: high.*

---

## Medium severity

### F10. Runtime upgrades are recorded but knowingly mixed

**planning/layer-bypass.md:169-177, 190-198, 236-250, 424-431.** Revision 3
recognizes that the hook and cache paths are version-sensitive, then records
torch and Transformers versions while deliberately excluding them from both
the guard and the summary key. A Colab upgrade can therefore split one run
across two runtime implementations, and the contract-designated summary
function silently combines the rows. Recording the versions makes a manual
post-hoc audit possible but does not stop a normal reported number from
pooling them.

The provenance is also incomplete for the paths this project actually uses:
4-bit execution depends on bitsandbytes, adapter wrapping on PEFT, and device
placement on Accelerate. Repository revision, device/backend, and those
package versions are absent.

Failure scenario: Transformers changes a decoder return/cache path between
two Colab sessions while `BYPASS_IMPL` remains `/v1`. Resume is explicitly
allowed, and `summarize_runs` reports one result across both paths. A
bitsandbytes-only change is not even visible in the row. *Confidence: high.
Severity: medium.*

### F11. Sampled generation is not reproducible across interruption

**planning/layer-bypass.md:190-198, 289-296; src/algoverse/eval.py:137,
145-173.** `run_negotiation_eval` resets the RNG from `seed` on every call.
When `resume=True`, completed rows are removed from `todo`, but their random
draws are not replayed or restored. Under `do_sample=True`, the first
remaining example after a restart consumes the random stream position that
the first example consumed in an uninterrupted run. The plan additionally
allows batch size to change after an OOM, which changes how draws are grouped
and assigned.

Failure scenario: two otherwise identical sampled jobs die after different
batch counts. Both pass the full identity guard and carry the same seed, but
their remaining responses differ from each other and from an uninterrupted
run. The reported result depends on failure timing. All proposed generation
and cache acceptance tests are greedy, so none covers this branch.
*Confidence: high. Severity: medium (publishable runs may retain the default
`do_sample=False`).*

### F12. The row `seed` does not identify a fine-tuning seed

**planning/layer-bypass.md:243-247; RESEARCH_SPEC.md:84-85;
scripts/run_baseline.py:38, 51, 59.** The plan says adding `seed` to the group
key prevents two fine-tuning seeds from being averaged away. In current code,
that field is passed to scenario subsampling and evaluation/generation RNG.
There is no separate fine-tuning-seed field in the binding row schema and no
function in this plan that aggregates or reports variation across training
seeds.

Failure scenario: two adapters trained with different seeds are evaluated
using the same eval seed, which is the appropriate paired evaluation design.
The row `seed` cannot distinguish their training seeds. Distinct
`adapter_path` values may incidentally keep them apart, but the summary has no
explicit training-seed dimension from which to report the variation the
research specification requires. Conversely, changing only the eval seed
creates separate groups even though it does not represent fine-tuning
variation. *Confidence: high. Severity: medium.*

### F13. The ratified reinstall-at-load requirement has no implementation in this plan

**RESEARCH_SPEC.md:98-103; planning/layer-bypass.md:405-411;
src/algoverse/models.py:19-31.** The normative decision now says checkpoint
metadata records the bypassed layer and every loader reinstalls the runtime
hook. Revision 3 declares the permanence issue resolved, but keeps
`load_model_and_tokenizer(model_id, quant, adapter_path)` unchanged, defines
no checkpoint metadata field or read path, and tests only explicit
`install_bypass` calls. `run_baseline.py` installs from a CLI argument, not
from checkpoint metadata.

Failure scenario: a Stage-2 lesioned adapter is later loaded through the
repository's shared loader using only its adapter path. Nothing reinstalls the
permanent bypass, so the object is intact. The evaluator will either reject
correct lesion bookkeeping or generate intact rows if the caller also omits
it. A future training/loader plan can operationalize the ratified decision,
but this plan's statement that the issue is resolved describes a decision,
not implemented or tested behavior. *Confidence: high. Severity: medium
because Stage-2 loading is future scope.*

### F14. A torn final JSONL line prevents the promised resume

**planning/layer-bypass.md:180-183, 323-335; src/algoverse/utils.py:112-129;
tests/test_metrics.py:275-284.** The always-on guard reads the output with
`utils.read_jsonl`, which calls `json.loads` on every nonblank line and does
not tolerate malformed JSON. The analysis suite explicitly contains a test
acknowledging that a killed run can leave a torn last line, but that tolerant
behavior exists only in `metrics.load_rows`.

Failure scenario: Colab dies during the append of the last result row. On
restart, the new guard reads the file before it can decide what work is done
and raises `JSONDecodeError`. `resume=False` no longer bypasses the read
either. The experiment cannot use the central advertised recovery mechanism
without manual file repair, and none of the proposed guard tests includes a
torn line. *Confidence: high. Severity: medium.*

### F15. Stage-2 training semantics remain tested only on Qwen

**planning/layer-bypass.md:266-313; RESEARCH_SPEC.md Methodology.** Revision 3
parameterizes the residual and cache mechanics over Qwen2, Llama, and Gemma2,
but the gradient-disconnection and PEFT-wrapper tests remain Qwen-only. Those
are the two tests most directly tied to the claim that the bypass stays
effective during LoRA fine-tuning across all three research models.

Failure scenario: all multi-family inference mechanics pass, while a
family-specific PEFT wrapping path or training forward convention makes the
Gemma/Llama adapter installation resolve or train differently. The plan's
automated evidence supports Qwen's Stage-2 semantics and only inference
semantics for the other families. *Confidence: medium (the generic hook is
likely to behave uniformly). Severity: medium-low.*

---

## Low severity / contract and provenance hygiene

### F16. Quantization provenance remains caller-asserted

**planning/layer-bypass.md:169-177, 188-192; src/algoverse/eval.py:110-118.**
Bypass provenance is now derived from model state, but `quant_label` remains a
caller argument. The full identity guard verifies consistency only with prior
labels, not with the live model. The canonical CLI passes the loader argument
correctly.

Failure scenario: another current/future caller stamps a 4-bit model as
`"none"` or vice versa. The first row establishes that false label as the run
identity, and every later guard comparison succeeds against it. *Confidence:
high. Severity: low-to-medium.*

### F17. Revision 3 contains stale INTERFACES-edit wording

**planning/layer-bypass.md:72, 250-254, 391-395; INTERFACES.md:38-40.** The
module map calls `INTERFACES.md` untouched and the metrics section describes
its wording change as a future human touch-up. The contract has already been
edited to add split and seed, and the resolution section acknowledges that
edit. The plan therefore gives contradictory status for the same binding
contract.

Failure scenario: an implementer follows the module-map/metrics wording and
requests or performs an already-completed contract coordination step, or
assumes the checked-in contract still lacks the ratified grouping change.
This is documentation-only. *Confidence: high. Severity: low.*

### F18. The public summary description leaves `bypass_impl` implicit

**RESEARCH_SPEC.md:114-116; INTERFACES.md:38-43.** The normative spec
explicitly makes `gen_config.bypass_impl` part of the `summarize_runs` group
key. INTERFACES describes output per `(model, intervention, checkpoint, split,
seed)` without stating that the implementation identifier is a returned/group
key dimension. "Intervention" may be intended to encompass it, but callers
cannot tell from the binding interface whether `bypass_impl` is present in the
summary dict or only influences grouping.

Failure scenario: a figures caller relies on the binding interface, expects
only the named dimensions, and either drops the returned implementation key or
cannot label two otherwise-identical implementation groups without inspecting
the implementation of `summarize_runs`. *Confidence: medium. Severity: low.*

---

## Verification record

- All five existing CPU suites passed in the current tree on 2026-08-13 via
  their built-in runners: `test_data.py`, `test_metrics.py`,
  `test_perplexity_count.py`, `test_scenarios.py`, and `test_scoring.py`.
- `torch`, `transformers`, `peft`, and `bitsandbytes` are unavailable in the
  local environment. No planned bypass acceptance test, smoke test, cache
  test, gradient test, PEFT test, or 4-bit check ran; none is claimed to work.
- The repeated-run failure in F6 was reproduced with current
  `summarize_runs`: two contradictory run IDs over one scenario pooled to
  `tau=0.5` with CI `[0.5, 0.5]`.
- The scenario-cohort claim in F2 was checked with the current pool size and
  Python runtime: seed-42 samples of sizes 20 and 100 shared 19 scenarios and
  were not nested.
- Lad et al., *The Remarkable Robustness of LLMs: Stages of Inference?*,
  arXiv:2406.19384v3, and released commit
  `4ee3f29ecf3e812a20af111f8888cb57085fdbae` were fetched and rechecked. The
  revised plan accurately limits the citation to residual-identity semantics
  and a qualitative depth trend; no new literature finding is warranted.

---

Disposition (planner, revision round 2 -> plan revision 4, 2026-08-13)

Applied in `planning/layer-bypass.md` revision 4. Per the revision protocol:
every finding adjudicated; 16 accepted in full or part; F6 and F7 escalated
to the human and resolved by ratification in-session (run_id reversal;
minimal weight provenance); F18 remains a one-line human contract touch-up.
Ratified decisions and newly opened items are recorded in RESEARCH_SPEC.md.

| Finding | Disposition | Reason / action |
|---|---|---|
| F1 resume=False duplicates rows | Accepted | Append-only rule: existing rows + `resume=False` → error; regeneration needs a fresh file/run_id. Test 12 covers the exact matching-identity duplication scenario. |
| F2 split/cohort not in identity | Accepted | Guard adds a split-set check and an existing-⊆-requested scenario-id check (critic's non-nested `random.sample` demonstration accepted). |
| F3 patch fields not in identity | Accepted | `patch_layer`/`patch_source` join the guarded fields; test 13 covers a patch mismatch. |
| F4 scoring config unguarded/unrecorded | Accepted in part / remainder escalated | `use_llm_fallback`/`llm_provider`/`llm_model` recorded in gen_config and guarded (legacy rows compare via signature defaults). `extraction_method` granularity escalated to RESEARCH_SPEC Open decisions (eval-track refinement). |
| F5 summarize_runs pools across gen profiles | Accepted in part / slivers escalated-and-resolved | Derived `quant`, `do_sample`, `max_new_tokens`, `dtype`, `device_type` join the group key, aligning it with the run-identity guard. Package versions stay out (ratified: audit-only). Cohort stays out of the key (a group's rows are its cohort; the scenario bootstrap operates on it; within-run cohort integrity is now guarded by F2's fix). |
| F6 cross-run pooling → false CI | Escalated → human-ratified REVERSAL | Reproduction (tau=0.5, CI [0.5,0.5]) and the refutation of the multi-session rationale accepted; `run_id` joins the group key; the round-2 acceptance test flips to run_id-separates; RESEARCH_SPEC bullet amended. |
| F7 identity = labels, not weights | Escalated → human-ratified minimal fix | Derived `model_revision` (resolved commit hash) and `adapter_digest` (SHA-256 of adapter weight files) recorded in gen_config and guarded. Full weight hashing of the base model rejected as disproportionate (GB-scale per load). |
| F8 quant="none" spans two loader paths | Accepted | Derived `load_profile` {device_type, dtype, four_bit, attn_implementation} recorded, guarded, and (dtype/device_type) grouped. |
| F9 test 10 conflates range and state checks | Accepted | Test 10 now uses a VALID in-range layer (1) on the intact model, isolating the model-state cross-check. |
| F10 provenance incomplete; upgrades mixed | Accepted in part | peft/bitsandbytes/accelerate versions recorded (present-or-None). Guarding/grouping on versions stays rejected — ratified audit-only design. Repo git SHA not taken (fragile in the Colab clone workflow; low value over the recorded package set). |
| F11 sampled resume not reproducible | Accepted | Sampled-resume rule: existing rows + `do_sample=True` → error with explanation; greedy default unaffected; test 13 covers it. |
| F12 row seed is not the fine-tuning seed | Accepted in part / remainder escalated | Plan wording corrected (seed = eval seed; training-seed identity lives only in adapter_path today). Representation of training seeds in the row schema escalated to RESEARCH_SPEC Open decisions (contract/Stage-2 owners). |
| F13 reinstall-at-load ratified but unimplemented | Accepted | Resolution wording now distinguishes RATIFIED DECISION from implemented behavior; the loader argument + metadata read path are named as the Stage-2/loader plan's deliverable. |
| F14 torn final line breaks the always-on guard | Accepted | Guard/resume reads use `metrics.load_rows`' tolerant semantics (skip malformed FINAL line only); torn-line case added to test 13. |
| F15 Stage-2 semantics tested only on Qwen | Accepted | Gradient and PEFT tests parameterized across all three families (tests 6-7). |
| F16 quant label caller-asserted | Accepted (subsumed by F8) | Derived `four_bit`/`load_profile` is guarded; a label contradicting the derived profile raises. |
| F17 stale INTERFACES-edit wording in plan | Accepted | Verified the human already updated the contract to "(…, split, seed)"; module map and metrics section now state that, leaving only the F18 touch-up flagged. |
| F18 contract doesn't name bypass_impl dimension | Escalated (human contract touch-up) | One-line human edit: name run_id and the impl/gen-profile dimensions in the summarize_runs sentence. We never edit INTERFACES.md. |
