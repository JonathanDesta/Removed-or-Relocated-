# layer-bypass.critique-1 — Plan critique, round 1

Scope: `planning/layer-bypass.md`, reviewed against `RESEARCH_SPEC.md`,
`INTERFACES.md`, the current implementation, representative Hugging Face
decoder implementations, and the cited layer-removal paper and released code.
The approved plan was not edited.

Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.* The role requires coverage rather than filtering, so test-quality
and low-severity findings are included.

---

## High severity

### F1. The recorded bypass implementation remains caller-controlled

**planning/layer-bypass.md:120-139.** The proposed evaluator cross-checks only
`bypass_state(model)["layer_idx"]` against the caller's `bypassed_layer`
argument. It does not compare the marker's `impl` with the caller's new
`bypass_impl` argument; the latter is copied directly into `gen_config`.

Consequently, a caller can install the real layer-5 bypass, pass
`bypassed_layer=5` and omit `bypass_impl`, and write rows whose provenance says
`bypass_impl: null`. It can likewise pass an arbitrary implementation string,
or pass an implementation string for an intact model while leaving both the
state layer and `bypassed_layer` null. Every layer-number check succeeds. This
violates INTERFACES.md's hard requirement that every generation record which
implementation actually produced it; the plan's API makes the claimed
provenance independent of the model state that is supposed to authenticate it.

Failure scenario: a future sweep caller remembers `bypassed_layer` but relies
on the default `bypass_impl=None`. Its genuinely bypassed rows become
indistinguishable from legacy/unversioned output, or are rejected on resume
after an interruption despite having been produced by the same hook. In the
opposite direction, intact output can carry the `/v1` implementation label.
*Confidence: high. Severity: high.*

### F2. The bypass-mixing guard is disabled when `resume=False`

**planning/layer-bypass.md:134-139; src/algoverse/eval.py:140-143.** The plan
places its guard after the existing `existing` calculation. Current code reads
the output file only under `if resume and out_path.exists()`; with
`resume=False`, `existing` is always empty even though generation still appends
to the same JSONL file.

Failure scenario: `rows.jsonl` already contains `run_id="sweep-l5"` rows from
layer 5. A caller reuses that run ID and path with layer 7 and
`resume=False`. The proposed state/bookkeeping check verifies that the live
model really has layer 7 bypassed, but the empty `existing` list makes the
mixing guard pass. New layer-7 rows are appended beside layer-5 rows under one
run ID. Later analysis reads both and can report a blended tau or silently
double-count scenario-condition pairs. This contradicts the repository's
append-only result convention and the plan's claim that bypass mixing is
structurally impossible. *Confidence: high. Severity: high.*

### F3. The analysis path ignores the new implementation provenance

**planning/layer-bypass.md:26-30, 42-51; src/algoverse/metrics.py:446-481.**
The plan deliberately leaves `metrics.py` untouched. `summarize_runs` groups
rows using top-level `RUN_KEY_FIELDS`, which includes `bypassed_layer` but not
the nested `gen_config.bypass_impl` (and not `run_id`). It therefore treats two
bypass implementations on the same model, layer, adapter, arm, and checkpoint
as one intervention.

Failure scenario: `/v1` is bumped after a semantic change exactly as the plan
instructs, and an analyst feeds old and new sweep files to
`summarize_runs`. Although every row carries its version, the function pools
both versions before computing tau and its CI. The paper then reports a number
that corresponds to neither implementation. Merely stamping provenance does
not meet the hard requirement's purpose when the contract-designated summary
consumer discards it during grouping. *Confidence: high. Severity: high.*

### F4. No acceptance test exercises the model/bookkeeping mismatch guard

**planning/layer-bypass.md:126-133, 141-155, 177-216.** The bypassed smoke leg
uses a correctly installed model with matching `bypassed_layer` and
`bypass_impl`. Its expected-error leg first removes the hook and then calls the
same run ID with `bypassed_layer=None`; at that point the live model state and
the argument agree, so only the resume-mixing guard can raise. None of the nine
unit tests calls the evaluator with either actual mismatch:

- an intact model plus non-null `bypassed_layer`; or
- an installed bypass plus `bypassed_layer=None` or a different layer.

There is likewise no implementation-identifier mismatch case (F1).

Failure scenario: the evaluator's model/bookkeeping cross-check is omitted,
inverted, or compares the wrong marker field. All proposed tests still pass:
the CLI installs its bypass, the successful smoke call is internally
consistent, and the final smoke error comes from the independent resume guard.
The exact F4 failure the plan says it makes structurally impossible remains
unprotected for non-CLI callers. *Confidence: high. Severity: high.*

---

## Medium severity

### F5. The last-layer hidden-state identity convention is false

**planning/layer-bypass.md:111-116, 195-204.** The plan declares that
bypassing any layer `l`, including the last layer, must make
`output_hidden_states[l+1]` bitwise equal to `output_hidden_states[l]`. In
representative Transformers releases, Qwen2 collects the input to every
decoder block, runs all blocks, applies the model's final RMSNorm, and only
then appends the final hidden-state entry. For example, Transformers v4.52.4
applies `self.norm` at lines 453-457 before appending the last entry:
https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2/modeling_qwen2.py#L453-L457

For the four-layer fixture with layer 3 bypassed, `hs[3]` is the input to the
bypassed layer while `hs[4]` is the final-normalized form of that input. A
correct identity hook therefore normally produces `hs[4] != hs[3]`. The
plan's explanation that final norm may apply *after* the tuple's last entry is
reversed for the very Qwen implementation used by its test fixture; the same
ordering appears in common Llama and Gemma2 implementations.

Failure scenario: the implementer follows the residual-identity contract
correctly and `test_first_and_last_layer_mechanics` fails. Treating the test as
normative could then prompt a change to model-level normalization or hidden
state reporting rather than to the bypass, producing a real semantic
deviation merely to satisfy an invalid oracle. *Confidence: high. Severity:
medium.*

### F6. Automated coverage exercises only one of the three research families

**RESEARCH_SPEC.md Methodology; planning/layer-bypass.md:53-59, 63-85,
177-216, 229-235.** The research runs Qwen2.5, Llama-3.1, and Gemma-2. The plan
claims its decoder lookup, hook signature handling, cache behavior, PEFT path,
sharding, and 4-bit behavior across those families, while every automated test
uses a tiny Qwen2 model. The real-model smoke test is Qwen2.5-0.5B, and the
only Colab sanity check is Qwen2.5-7B. No test instantiates even tiny random
Llama or Gemma2 configurations, and no manual check covers their 4-bit or PEFT
paths.

Failure scenario: Qwen passes the full acceptance suite while a Llama or
Gemma decoder differs in its return shape, cache interaction, wrapper path, or
gradient-checkpointing call convention. One or two thirds of the experiment
then fail at Colab time or, worse, preserve the wrong non-hidden outputs while
still generating plausible text. The plan's local evidence cannot support its
cross-family claims. *Confidence: high. Severity: medium.*

### F7. The critical acceptance suite can provide zero coverage while looking clean

**planning/layer-bypass.md:177-189, 213-228.** All bypass test functions are
defined only inside `HAVE_ML_STACK`, so pytest collects zero tests from the new
file when torch or Transformers is absent. The direct runner prints `SKIP` and
exits successfully. PEFT coverage is guarded separately, but the plan does not
define a mandatory environment or a failing acceptance gate in which PEFT is
present. Thus the contract's byte-identity test, Stage-2 gradient test, cache
test, and LoRA-wrapper test can all remain unexecuted while the repository's
test command exits zero.

This is not hypothetical in the review environment: `torch`, `transformers`,
and `peft` are all unavailable. The five existing CPU suites pass, but none can
exercise the proposed mechanism. AGENTS.md requires unverified behavior to be
reported as unverified; the plan has no concrete full-stack verification
environment that turns these test specifications into evidence.

Failure scenario: implementation lands with a tuple-handling or PEFT-wrapper
bug, the local test run reports no failures, and the hard requirement is
described as unit-tested even though the relevant functions were never
collected. *Confidence: high. Severity: medium.*

### F8. The bypassed block remains observable as if it were active

**planning/layer-bypass.md:63-83; src/algoverse/interp.py:99-113,
181-199.** The plan explicitly executes the entire decoder block and discards
only its residual output. For tuple-returning Transformers versions, it
preserves `output[1:]`, including attentions and legacy cache outputs; for
cache-object versions, the executed attention module can update its cache in
place. The block therefore remains internally live even though its residual
contribution is causally disconnected.

Failure scenario: localization corroboration or later causal-remapping code
requests `output_attentions=True` on a permanently bypassed checkpoint.
`attention_all_layers` returns an apparently ordinary attention map for the
bypassed layer, and a layer-wise JSD figure can present it alongside active
layers without revealing that the computation was discarded. That evidence
cannot support a claim about where the model's output-producing circuit lives.
The same caveat applies to any probe placed inside, rather than after, the
bypassed block. *Confidence: high on the mechanism; medium on whether the
later analysis will consume these values. Severity: medium.*

### F9. Greedy-token equality does not validate cache correctness directly

**planning/layer-bypass.md:200-204.** The proposed cache test compares only
generated token IDs under `use_cache=True` and `False`. A cache bug can change
logits substantially without changing the argmax token on one short random
example, so this oracle can pass while the cached and uncached distributions
differ. The instruction to change the fixture seed if a near-tie exposes a
difference makes the test still less falsifiable: it selects an example whose
argmax is stable rather than establishing cache equivalence.

Failure scenario: bypassed cached generation has a systematic logit error that
does not cross an argmax boundary for the chosen seed. The test passes and is
cited as direct validation of cache safety, but sampling, longer generation,
or a real checkpoint crosses a boundary and produces different evaluation
responses. *Confidence: high. Severity: medium.*

### F10. Lad et al. do not support the quantitative perplexity sanity check

**planning/layer-bypass.md:32-38, 229-235.** The cited paper is Vedang Lad,
Jin Hwa Lee, Wes Gurnee, and Max Tegmark, *The Remarkable Robustness of LLMs:
Stages of Inference?*, arXiv:2406.19384v3:
https://arxiv.org/abs/2406.19384

It reports Pile next-token loss/distribution metrics, relative top-1 accuracy,
and HellaSwag/ARC-Easy/LAMBADA performance. It does not report WikiText-2
perplexity or a "hundreds+" threshold. Its newer-family experiments include
Qwen2.5 models through 3B and Llama-3.2 models through 3B, not the project's
Qwen2.5-7B, Llama-3.1-8B, or Gemma-2-9B targets. The released intervention
code zeroes attention and MLP outputs under TransformerLens; it supports the
residual-identity analogy but not 4-bit Hugging Face hook behavior or the
proposed quantitative PPL expectation:
https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/4ee3f29ecf3e812a20af111f8888cb57085fdbae/model_intervention.py#L109-L134

Failure scenario: a correct first-layer bypass moves WikiText-2 perplexity
dramatically but not into the hundreds on Qwen-7B. The plan's citation-backed
"should" treats that result as a failed hook even though the cited work never
established the asserted model, dataset, metric, or range. *Confidence: high.
Severity: medium.*

---

## Low severity / test quality / provenance hygiene

### F11. The literature attribution drops one author

**planning/layer-bypass.md:32-35.** The plan calls the paper "Lad, Gurnee &
Tegmark"; Jin Hwa Lee is the second of its four authors. This does not affect
the mechanism, but it is a factual literature-facing error in the paragraph
that asserts the paper was fetched and read. *Confidence: high. Severity:
low.*

### F12. The double-install invariant is attached to one wrapper, not the decoder

**planning/layer-bypass.md:95-109, 272-275.** The proposed marker lives on the
exact Python object passed to `install_bypass`, while `_decoder_layers` can
resolve through PEFT/base-model wrappers to shared underlying decoder modules.
The marker therefore proves only that this wrapper object has been used for an
installation. It does not establish the stronger comment that no bypass is
installed "anywhere on this model."

Failure scenario: a bypass is installed through a PEFT wrapper and another is
installed through a retained inner/base-model reference. The objects have
separate marker attributes but resolve to the same decoder stack, so two hooks
can coexist while the outer evaluator sees and records only one layer. This is
unlikely under the documented install-last path, but the stated invariant is
not enforced by the proposed representation. *Confidence: medium (the exact
wrapper attribute-forwarding behavior depends on PEFT version). Severity: low
unless mixed wrapper references are used.*

### F13. `/v1` does not identify the version-sensitive runtime path

**planning/layer-bypass.md:89-92, 263-280; pyproject.toml.** The plan explicitly
handles and documents behavior that changes across torch and Transformers
versions, including tuple versus tensor layer outputs and hook call
signatures. Those dependencies are neither declared nor pinned, and the
proposed generation metadata records only the static project implementation
string. Two rows labeled `block-output-identity-hook/v1` can therefore have
traversed different decoder/hook/cache implementations with no package
versions available in the row to distinguish them.

Failure scenario: results are resumed or reproduced after a Colab package
upgrade. The project hook string is unchanged, the resume guard accepts it,
but the underlying layer return/cache path differs from the one that received
local acceptance coverage. *Confidence: high on the missing provenance;
severity: low-to-medium because the intended residual identity may remain
equivalent.*

### F14. The existing CPU suites cannot prove the `interp.py` import claim

**planning/layer-bypass.md:220-225.** The plan says the five existing test
files passing also proves that moving `_decoder_layers` did not leak heavy
imports into the torch-free suites. Those suites do not import
`algoverse.interp`, so their success cannot exercise either side of its new
`from algoverse.models import _decoder_layers` import. A regression confined
to importing `interp.py` would leave all five results unchanged.

Failure scenario: the import move introduces an import-time error or an
unexpected dependency path in `interp.py`; all five cited suites still pass,
and the plan reports evidence they do not provide. *Confidence: high.
Severity: low/test-quality.*

---

## Verification record

- All five existing CPU test files passed locally on 2026-08-13 when run via
  their built-in `__main__` runners:
  `test_data.py`, `test_metrics.py`, `test_perplexity_count.py`,
  `test_scenarios.py`, and `test_scoring.py`.
- `torch`, `transformers`, and `peft` are unavailable in the local
  environment. No proposed bypass unit test, real-model smoke test, 4-bit
  check, cache check, gradient check, or PEFT check was executable. None is
  claimed to work.
- Lad et al. arXiv:2406.19384v3 and the released intervention implementation
  at commit `4ee3f29ecf3e812a20af111f8888cb57085fdbae` were inspected.
- Official Hugging Face Qwen2, Llama, and Gemma2 decoder sources across
  representative releases were inspected. Common tuple-returning releases
  apply final norm before appending the final hidden-state entry, establishing
  F5 independently of runtime availability here.

---

## Disposition (planner, revision round 1 — 2026-08-13)

Applied in `planning/layer-bypass.md` revision 2. Per the revision protocol:
every finding adjudicated; escalations go to the team. 13 accepted, F3
escalated, F8 partially escalated, none rejected.

| Finding | Disposition | Reason / action |
|---|---|---|
| F1 impl string caller-controlled | Accepted | `bypass_impl` parameter removed entirely; the runner derives the stamp from `bypass_state(model)` — provenance cannot disagree with model state. |
| F2 guard dead under `resume=False` | Accepted | Mixing guard now reads existing rows whenever the file exists, independent of `resume`; acceptance test 12 pins both flag values. |
| F3 `summarize_runs` pools across impls | **Escalated** (plan §Escalations E1) | metrics.py is the figures/analysis track's surface; the finding generalizes first-review F10's unratified pooling decision (run_id/split/seed) — one coordinated team fix, not a piecemeal edit from the bypass plan. Interim exposure nil: only `/v1` exists and the resume guard blocks within-file mixing. Proposed remedy recorded in the plan. |
| F4 mismatch guard untested | Accepted | New tests 10-12 exercise intact+non-null, bypassed+null, bypassed+wrong-layer, and resume-guard mixing under both resume flags. |
| F5 last-layer identity convention false | Accepted | Critic verified against the Qwen2 modeling source; the plan's note had the norm order reversed. Convention corrected to l <= N-2; last layer checked at the final norm's input; test 4 re-specified and now documents hs[N] != hs[N-1] as EXPECTED. |
| F6 only Qwen tested | Accepted | Core mechanics tests (1-5) parameterized over tiny random Qwen2/Llama/Gemma2 configs. |
| F7 suite can skip silently | Accepted | Mandatory ML-stack execution gate added to Verification; loud "NOT verification" SKIP banner; implementer must name the executing environment per AGENTS.md. |
| F8 bypassed block observable as live | Accepted (docs) / remainder **escalated** (plan §Escalations E2) | Docstring + risks now state the executed-but-discarded property and its observability consequence; the convention for Stage-3/corroboration analyses (exclude/flag the bypassed layer) is an analysis-track decision to make before any interp analysis runs on a bypassed checkpoint. |
| F9 cache oracle too weak | Accepted | Test 5 adds stepwise last-position logit `allclose` against no-cache forwards; the seed-bump escape hatch is deleted. |
| F10 ppl threshold not in Lad et al. | Accepted | Colab check reworded to a directional/ordering expectation (layer-0 >> middle >> intact); the "hundreds+" intuition re-attributed to eval.py's own compute_perplexity docstring, not the paper. |
| F11 dropped author | Accepted | Citation corrected: Lad, Lee, Gurnee & Tegmark. |
| F12 marker on wrapper, not decoder | Accepted | Marker moved to the resolved decoder-layers object shared by all wrapper views; test 9 gains a wrapper-vs-base double-install assertion. |
| F13 `/v1` blind to stack versions | Accepted | gen_config gains torch/transformers version strings (provenance only; deliberately not a guard input — Colab upgrades must not block resume). |
| F14 CPU suites can't prove interp claim | Accepted | False evidence claim removed from the plan; explicit `import algoverse.interp` check added to the ML-env verification list. |
