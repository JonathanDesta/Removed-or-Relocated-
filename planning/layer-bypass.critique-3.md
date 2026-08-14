# layer-bypass.critique-3 — Plan critique, round 3

Scope: revision 5 of `planning/layer-bypass.md`, reviewed against the current
`RESEARCH_SPEC.md`, `INTERFACES.md`, source tree, Lad et al. v3 and its
released intervention code, the PEFT checkpoint specification, and official
OpenAI model documentation. Resolved round-2 findings are not repeated unless
revision 5 leaves a distinct residual failure. The approved plan, contracts,
and source were not edited.

Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## High severity

### F1. A torn final line consumes the first regenerated row

**planning/layer-bypass.md:241-250, 496-497;
src/algoverse/utils.py:94-110.** Revision 5 deliberately has the resume path
skip malformed JSON and says the missing row will “simply regenerate,” but it
leaves `append_jsonl` unchanged. A genuinely torn final write normally ends
without a newline. The next append writes the regenerated JSON immediately
after those partial bytes, making one larger malformed line. The tolerant
reader then skips both the torn fragment and the otherwise-complete
regenerated record.

This was reproduced locally with the current append semantics. A file ending
in `{"scenario_id":"torn"` followed by an append of
`{"scenario_id":"regenerated"}\n` physically contained
`{"scenario_id":"torn"{"scenario_id": "regenerated"}`; the tolerant
reader returned neither record. The resumed call can still include the row in
its in-memory return value and print a complete tau, while the append-only
artifact silently lacks that row. A later analysis therefore reports a
different cohort unless the job happens to be resumed yet again.

The proposed torn-line test only establishes that the guard does not crash;
its preseed covers the full requested product, so no post-torn-line append is
exercised. It passes on exactly this data-loss behavior. *Confidence: high.
Severity: high.*

### F2. The run guard cannot distinguish recovery from a changed request

**planning/layer-bypass.md:241-295, 445-497; src/algoverse/eval.py:110-205.**
The guard persists only completed rows, not the originally requested scenario
and condition sets. Its `existing scenario_ids ⊆ requested scenario_ids`
rule therefore accepts both an interrupted request and a later enlargement of
the experiment. `conditions` is not guarded at all.

The subset loophole is not theoretical. On the current 305-scenario selection
pool, Python 3.14.6, and seed 42, the IDs returned for `n=100` are a strict
subset of those returned for `n=200`. Reusing the run ID with `n=200` passes
the planned check and appends another 100 scenario pairs. The same run ID's
reported population silently changes from 100 to 200 scenarios. The plan's
chosen `n=20` versus `n=100` counterexample happens not to be nested, but
Python's sampling implementation makes other size pairs nested.

The condition branch is still less observable. A completed two-condition run
called later with `conditions=("incentive",)` has no work to generate, but
the function returns all existing rows for the run ID, including control
rows that the caller did not request. In the opposite direction, an
incentive-only run can later be expanded with control rows under the same run
identity. The proposed tests cover a non-subset rejection and one exact full
resume, but neither changed-request case. *Confidence: high. Severity: high.*

### F3. A publishable run can silently execute no LLM fallback

**RESEARCH_SPEC.md:131-136, 145-148; planning/layer-bypass.md:313-339,
586-599; Notebook Setup.ipynb:18-25; pyproject.toml:1-15;
src/algoverse/tasks.py:464-546.** Revision 5 makes the OpenAI fallback
mandatory for publishable runs, but neither documented environment installs
the `openai` package. More importantly, `llm_extract_offer` catches every
import, credential, endpoint, API, and response-parsing exception and returns
`None`. The row then remains `unparseable`, while `gen_config` still records
`use_llm_fallback=True`, provider `openai`, and model `gpt-4o-mini`.

Failure scenario: an operator follows the checked-in Notebook Setup and the
canonical command exactly. If `openai` is absent—or the Azure endpoint or key
is misconfigured—every fallback attempt silently degrades to the regex-only
scorer. The run is nevertheless provenance-stamped as fallback-enabled and
completes normally. Invalid rates and tau can differ from runs where the
extractor actually succeeded, defeating the normative claim that every
publishable run uses the same extractor.

The current local interpreter has no `openai` package. No proposed acceptance
test calls the real client or proves that a configured fallback can succeed;
the existing scoring test replaces `llm_extract_offer` with a lambda, and the
new guard test only compares the boolean identity field. Treating failed
attempt recording as a future “nicety” removes the only row-level evidence
that this failure occurred. *Confidence: high. Severity: high.*

### F4. The adapter digest does not identify the loaded adapter

**planning/layer-bypass.md:205-212, 264-272.** The proposed digest covers
only `adapter_model.safetensors` or `*.bin`. A PEFT adapter also requires
`adapter_config.json`, which controls such output-affecting properties as
the adapter type, target modules, rank, and LoRA scaling. Hugging Face's
[PEFT checkpoint specification](https://huggingface.co/docs/peft/developer_guides/checkpoint)
explicitly states that both the weights and `adapter_config.json` are needed
to load an adapter and that parameters such as `r`, `lora_alpha`, and scaling
come from the configuration rather than the weight file.

Failure scenario: `adapter/latest/adapter_config.json` is changed in place,
for example by changing `lora_alpha`, while its weight file is unchanged.
PEFT loads a model with different adapter scaling and therefore different
outputs. `adapter_path` and the planned digest are unchanged, so a resume
accepts and mixes rows from the two models.

There is a second unresolved input class: the official
[`PeftModel.from_pretrained` API](https://huggingface.co/docs/peft/main/package_reference/peft_model#peft.PeftModel.from_pretrained)
accepts either a local directory or a Hub model ID, and the shared loader
forwards `adapter_path` directly. The plan specifies hashing files “at” the
supplied path but gives no resolved-file identity for a Hub ID; such a valid
loader input has no local filesystem files under that string. *Confidence:
high on the omitted-config failure, high on the underspecified Hub-ID path.
Severity: high.*

### F5. Capability metrics still do not resume or enforce run identity

**INTERFACES.md:34-36 and final sentence; planning/layer-bypass.md:337-344;
scripts/run_baseline.py:70-83; src/algoverse/eval.py:309-369, 391-460,
510-516.** The plan adds `bypass_impl` and `train_seed` to `run_meta` but
explicitly leaves the benchmark/perplexity signatures and append behavior
unchanged. Every invocation reruns GSM8K and MMLU and appends both rows, then
appends perplexity. There is no done-set, identity guard, or duplicate check
for `competence.jsonl`. `gate1_report` reads the file into a map and silently
keeps the last value for each metric without validating `run_meta` or
`config`.

Failure scenario: attempt A writes GSM8K and MMLU, then dies during
perplexity. After a routine Colab/runtime change—which revision 5 explicitly
allows for resumes—attempt B writes a new GSM8K row and dies before MMLU.
`gate1_report` combines attempt B's GSM8K with attempt A's MMLU and has no
perplexity. A related interruption can leave any hybrid of attempts; if all
three keys exist, that hybrid feeds the capability-loss gate as if it came
from one evaluated configuration. This contradicts the binding “Everything
resumes” claim and can change whether a layer passes the prespecified
capability bounds. No verification item exercises interrupted capability
metrics. *Confidence: high. Severity: high.*

### F6. The provenance tests derive their own oracle from the code under test

**planning/layer-bypass.md:230-235, 445-507.** Guard fixtures obtain their
matching `gen_config` by calling the same `_derive_gen_config` helper the
runner uses. The field-coverage test then independently perturbs only
`max_new_tokens` and `use_llm_fallback`; the smoke test independently checks
only `bypass_impl`.

A helper that always records `model_revision=None`, returns a constant or
empty `adapter_digest`, reads the wrong dtype/device, fails to reject a false
quantization label, or resolves the wrong provider default will generate the
same bad value on both sides of every planned guard test. Those tests pass
because they prove comparison consistency, not that the provenance was
derived correctly. They also preseed a single homogeneous row, so a guard
implementation that checks only the first existing row can pass while
ignoring a later mismatching row in an already-mixed file. None of the exact
derivations introduced to close round-2 F7/F8/F16 has an independent
acceptance oracle. *Confidence: high. Severity: high.*

### F7. Two normative summary dimensions have no grouping acceptance test

**RESEARCH_SPEC.md:114-122; planning/layer-bypass.md:352-369, 499-507.** The
binding group key includes `load_profile.dtype` and
`load_profile.device_type`, specifically to prevent fp32/CPU and fp16/CUDA
runs from pooling under `quant="none"`. The proposed grouping matrix tests
run ID, split, both seeds, bypass implementation, quantization, sampling, and
token limit—but omits both dtype and device type.

Failure scenario: the implementation adds every tested field but forgets the
two nested load-profile fields. All proposed `test_metrics` additions pass,
yet `summarize_runs` silently pools the exact CPU/CUDA configurations that
revision 5 and the normative spec say must remain separate. *Confidence:
high. Severity: high.*

---

## Medium severity

### F8. The recorded scoring identity is not an immutable extractor identity

**planning/layer-bypass.md:213-239, 264-279, 318-335;
src/algoverse/tasks.py:445-546.** Recording the resolved argument
`gpt-4o-mini` does not record the actual model snapshot returned by the API.
Official [OpenAI documentation for GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini)
distinguishes the `gpt-4o-mini` alias from the dated snapshot and says
snapshots are what lock behavior to a specific version. The plan records no
OpenAI SDK version or response model identifier either.

The disk cache compounds this identity gap: its key contains provider, model
argument, and response text, but not `EXTRACTION_INSTRUCTION`, parser version,
endpoint/deployment identity, or repository revision. A scorer-prompt change
can therefore reuse an answer generated under the old prompt, while a new
machine without the ignored `.cache/` directory calls the current prompt.
Both rows carry the same planned scoring identity.

Failure scenario: two publishable runs use the same alias after the alias or
Azure deployment changes, or one reuses a cache created before an extractor
prompt change. Their validity/deception labels can differ or reflect different
scorer versions, while the resume guard and audit fields report the same
extractor. *Confidence: high. Severity: medium-to-high.*

### F9. PEFT acceptance can still disappear without a loud skip

**planning/layer-bypass.md:389-396, 430-436, 509-520.** The no-ML-stack path
is now loud, but the three PEFT tests remain under a separate nested import
guard. An environment with torch and Transformers but without PEFT can run
all defined bypass tests and print success without saying that wrapper/LoRA
coverage was absent. The prose environment gate requires PEFT, but the test
artifact itself does not enforce or report that requirement.

Failure scenario: an implementer runs in a lean ML environment, sees every
collected bypass test pass, and reports the named suite as executed. The only
tests proving marker sharing across wrapper/base references and install/remove
behavior on LoRA-wrapped versions of all three families never existed in that
run. *Confidence: high. Severity: medium.*

### F10. Scenario payload identity is trusted rather than checked

**planning/layer-bypass.md:258-263; src/algoverse/eval.py:174-201;
src/algoverse/tasks.py:180-226.** The new cohort guard compares only
`scenario_id` and split. It does not compare the requested scenario parameters
with the `scenario_params` already persisted. Canonical IDs currently hash
the parameter dict, which reduces this risk for untouched output from
`get_scenarios`, but `run_negotiation_eval` is a public central function and
accepts caller-supplied dictionaries without verifying that relationship.

Failure scenario: a caller reuses a scenario ID after changing its true offer
or company offer. The resume done-set skips generation and returns the old
row, so the call appears to have evaluated the new scenario but reports the
old response, truth, and deception label. If only some IDs are new, the same
run contains both payload definitions. None of the planned guard fixtures
contains full scenario parameters, so this case is not accepted or rejected
by test. *Confidence: high on behavior; severity: medium because the current
canonical scenario builder preserves content-addressed IDs.*

### F11. Fine-tuning replication remains a write-up-time choice

**RESEARCH_SPEC.md Statistical analysis and lines 123-130;
planning/layer-bypass.md:370-377, 581-585.** Adding and grouping a
`train_seed` field records seeds that happen to be run, but the normative
method still promises variation across fine-tuning seeds while the ratified
note allows a single-seed baseline and defers the number of Stage-2 seeds
until write-up, with a second seed only a stretch goal.

Failure scenario: the four-arm recovery result comes from one fine-tuning
seed. There is no between-seed variation to report. Alternatively, whether a
second seed is run is chosen after the first result is known, making the
replication decision outcome-dependent. Revision 5 calls the round-2 seed
finding resolved, but only identity representation is resolved; the
experiment needed to support the specification's statistical claim remains
undefined. *Confidence: high. Severity: medium because training itself is
future scope.*

---

## Low severity / contract hygiene

### F12. Revision 5 still describes two contract edits as pending after they landed

**planning/layer-bypass.md:88-92, 326-330; INTERFACES.md:38-40, 92-98.** The
module map says two human touch-ups remain: adding `train_seed` to the
`summarize_runs` dimensions and adding `--llm-fallback` to the canonical
Gate-1 command. Both are already present in the current binding interface.
The run-baseline section likewise says the canonical command predates the
fallback rule and would run fallback-off, which is false in the reviewed
tree.

Failure scenario: an implementer or reviewer treats already-satisfied
coordination as an unresolved approval dependency or assesses the canonical
command from the stale description rather than the binding file. This is a
documentation/status failure, not a numerical failure by itself.
*Confidence: high. Severity: low.*

---

## Verification record

- All five existing CPU suites passed in the current tree on 2026-08-13 via
  their built-in runners: `test_data.py`, `test_metrics.py`,
  `test_perplexity_count.py`, `test_scenarios.py`, and `test_scoring.py`.
- `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, and `openai`
  are unavailable in the local interpreter. No planned bypass test, smoke
  test, cache test, gradient test, PEFT test, 4-bit check, or real LLM
  fallback ran; none is claimed to work.
- The torn-line failure in F1 was reproduced with current append semantics:
  appending one complete JSON object to an unterminated partial final line
  made the complete object unreadable to the tolerant reader.
- The cohort-expansion failure in F2 was checked against the current scenario
  generator under Python 3.14.6. For selection split and seed 42, the
  `n=100` ID set is a strict subset of the `n=200` ID set; the same is true
  for `n=200` versus the full 305-scenario pool.
- The official Hugging Face PEFT checkpoint specification was checked to
  confirm that adapter configuration, not only adapter weights, determines
  the loaded adapter. Official OpenAI documentation was checked to distinguish
  the `gpt-4o-mini` alias from its dated snapshot.
- Lad, Lee, Gurnee, and Tegmark, [*The Remarkable Robustness of LLMs: Stages
  of Inference?*](https://arxiv.org/abs/2406.19384v3), and released
  [commit `4ee3f29ecf3e812a20af111f8888cb57085fdbae`](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/4ee3f29ecf3e812a20af111f8888cb57085fdbae/model_intervention.py#L109-L134)
  were rechecked. Revision 5's literature claims remain accurate; no new
  literature finding is warranted.

---

## Disposition (planner, revision round 3 -> plan revision 6, 2026-08-13)

Applied in `planning/layer-bypass.md` revision 6, adjudicated under the
human's explicit directive to choose airtight fixes over cheap-but-partial
ones. 11 accepted (one in part), F11 resolved by human pre-commitment.
**The plan is FROZEN at revision 6** (agreed with the human): the bypass
mechanism has drawn zero findings for two consecutive rounds; further
findings are handled in the implementation-critique round against real
code, where verification terminates.

| Finding | Disposition | Reason / action |
|---|---|---|
| F1 torn line consumes regenerated row | Accepted | `append_jsonl` gains a newline-guard before append + flush/fsync after (closes first-review F27(b) at the source); test 13 extended: after the tolerated torn fragment, a subsequent append must yield a READABLE row. |
| F2 guard can't distinguish recovery from changed request | Accepted (airtight variant) | Run MANIFEST supersedes the subset rule the critic defeated: first call records {run_id, sorted scenario_ids, conditions, split} to a `rows.manifest.jsonl` sidecar; later calls must match by set EQUALITY. Changed n, seed, conditions, or split all refuse; rows-without-manifest (legacy) refuse with fresh-run_id guidance. Also operationally closes first-review F6's seed/cohort confounding. |
| F3 publishable run can silently skip fallback | Accepted | Fail-fast probe in run_baseline when `--llm-fallback` (import SDK + key env + one real probe extraction, refuse on failure); per-row `llm_failed:<provider>` recording in extraction_method (resolves the sliver previously deferred to Open decisions); run-end failure tally; `openai` added to Notebook Setup installs. |
| F4 adapter digest incomplete | Accepted | Digest covers `adapter_config.json` + weight files (PEFT checkpoint spec: rank/alpha/scaling live in config); non-directory adapter_path (Hub id) records None under a documented rule; oracle test asserts a config-byte change moves the digest. |
| F5 competence metrics don't resume | Accepted (scoped) | Per-metric done-set + run_meta label-identity guard in `run_lm_eval_benchmarks`/`compute_perplexity` via a torch-free-testable helper (versions audit-only per the ratified rule); gate1_report untouched — last-wins is inert once duplicates are impossible. Makes the contract's "Everything resumes" true for competence.jsonl. |
| F6 tests derive oracle from code under test | Accepted | New test 14: independent hand-written constants for every derived field (device/dtype/four_bit/bypass_impl/revision/digest/resolved model); multi-row scan case (only the SECOND pre-seeded row mismatches -> must still refuse); guard specified to compare every existing row. |
| F7 dtype/device grouping untested | Accepted | `load_profile.dtype` and `load_profile.device_type` cases added to the grouping matrix. |
| F8 extractor identity not immutable | Accepted in part | Default pinned to dated snapshot `gpt-4o-mini-2024-07-18`; API-returned model recorded per row in extraction_method; `EXTRACTION_INSTRUCTION` hash added to the cache key; `openai_version` stamped. Rejected slice: repo-revision/scorer-version machinery (repo SHA already rejected+ratified in round 2; prompt changes surface via the cache-key hash and implementation review). |
| F9 PEFT coverage can vanish silently | Accepted | Runner prints a loud `PEFT tests SKIPPED (N not run) — wrapper coverage NOT verified` line when peft is absent. |
| F10 scenario payload trusted | Accepted | Requested scenarios overlapping persisted ids must match the stored `scenario_params` dict; test case added. |
| F11 replication decision outcome-dependent | Resolved by human pre-commitment | Policy fixed before any result exists: a second Stage-2 fine-tuning seed runs iff the single-seed pipeline completes by 2026-08-22 — calendar-based, independent of the first seed's results. Recorded in RESEARCH_SPEC. |
| F12 stale contract-status text | Accepted | Module map and run_baseline sections corrected: all five INTERFACES touch-ups verified landed; nothing pending. |
