# gpu-verification.critique-1 — Plan critique (round 1)

Scope: `planning/gpu-verification.md`, reviewed against `RESEARCH_SPEC.md`,
`INTERFACES.md`, the referenced planning debt, and the current implementation.
No plan, product-code, test, contract, or spec edits were made. Format per
finding: **location — claim.** Failure scenario. *Confidence / severity.*

---

## High severity

### F1. C4 has no reproducible pass/fail criterion despite being classified as a prespecified directional debug test

**Plan lines 33-35, 253-263; AGENTS.md “one rule”; RESEARCH_SPEC.md:287-289.**
The human decision classifies C4 as a debug test because it has a
“prespecified directional pass/fail,” but the executable criterion remains
“dramatically above,” “clearly above,” “near intact,” and an unspecified
“small window.” Those terms do not define an ordering tolerance or even fix
the token count, although the normative perplexity configuration is 20,000
tokens. Two implementers can see the same three values and legitimately
record PASS versus AMBIGUOUS, or choose different “small” slices and obtain
different outcomes. This does not revisit the human's debug-test
classification; it finds that the plan does not actually instantiate the
prespecified pass/fail on which that classification rests. Failure scenario:
the 4-bit hook has a subtle defect, a favorable short slice produces a
plausible-looking separation, and the implementer calls “clearly/near” after
seeing the values. *Confidence: high. Severity: high.*

### F2. C4's finite-perplexity check can pass when the raw accumulation overflowed

**Plan lines 253-263; `src/algoverse/eval.py:883-887,948-961`.**
The plan says finite returned perplexities verify the fp32-accumulation claim.
They do not: `compute_perplexity` deliberately returns
`exp(min(nll_mean, 20.0))`. If `nll_mean` is positive infinity, the returned
perplexity is the finite cap `exp(20)`. C4 can therefore satisfy “all three
finite,” show a dramatic capped layer-0 value, and PASS while the very
accumulation behavior it claims to verify is broken. The function prints the
raw mean NLL, but C4 never makes its finiteness part of acceptance. *Confidence:
high. Severity: high.*

### F3. C5 treats an unusable required attention path as an acceptable outcome

**Plan lines 265-271; `src/algoverse/models.py:218-279`;
`src/algoverse/interp.py:109-140`; RESEARCH_SPEC.md:488-495.** The spec requires
attention-JSD corroboration, but the production 4-bit loader exposes no way to
request `attn_implementation="eager"`. C5 nevertheless accepts the clean
“reload with ... eager” RuntimeError as an ordinary recorded outcome rather
than a failed or escalated production path. Failure scenario: the default
backend returns no attentions on Qwen; C5 is marked accepted; later the
corroboration driver calls the same canonical loader and cannot produce any
attention-JSD values. The clean wording of the exception prevents a confusing
crash, but it does not verify that the paper-required path can run. *Confidence:
high. Severity: high.*

### F4. C6 does not execute the production adapter-loading path it claims to verify

**Plan lines 273-278; `src/algoverse/models.py:218-279`.** The cited production
claim at `models.py:274-277` is specifically
`PeftModel.from_pretrained(model, adapter_path)` inside
`load_model_and_tokenizer`. C6 instead says to take the already-loaded model
and wrap it with a newly created “trivial LoRA adapter.” That can verify PEFT
wrapper traversal while never saving, loading, or applying an adapter through
the production loader. Failure scenario: `get_peft_model` wrapping works and
C6 passes, but a real Stage-2 adapter is incompatible with
`PeftModel.from_pretrained`, is loaded onto the wrong device/dtype, or loses
the loader's 4-bit marker; every adapter-backed evaluation then fails or has
wrong provenance. *Confidence: high. Severity: high.*

## Medium severity

### F5. A6's BOS oracle is known-false for Qwen and contradicts the debt it cites

**Plan lines 155-175; `src/algoverse/eval.py:193-204`;
`planning/first-full-review.md:142-189`.** The actual contract is “do not
prepend a second BOS,” not “every rendered prompt contains exactly one BOS.”
The cited plan explicitly says Qwen's template adds no BOS and asks the real
tokenizer check only to assert that the prompt does not begin with two BOS
ids. A6 instead requires exactly one BOS for every family. A rung-2 diagnostic
with the named Qwen tokenizer confirmed `bos_token_id is None`, zero BOS ids
with `add_special_tokens=False`, zero with `True`, and identical encodings.
Thus a correct Qwen implementation necessarily fails A6 and triggers the
plan's high-severity product finding. *Confidence: high (source plus executed
real-tokenizer diagnostic). Severity: medium.*

### F6. A6 gives a call that cannot execute against the current signature

**Plan lines 161-167; `src/algoverse/eval.py:207-217`.** The plan explicitly
spells the assertion as `_system_fold_needed(tokenizer)`, but the function
requires `(tokenizer, probe_messages)`. “Read its signature first” is attached
to `render_condition_texts`, not an acceptance definition for the missing
argument. Following the stated check raises `TypeError` for every family
before testing fold behavior. An implementer must invent the missing probe or
silently reinterpret the plan. *Confidence: high. Severity: medium.*

### F7. A6 does not close the real-tokenizer production-wiring obligation

**Plan lines 155-175; `planning/first-full-review.md:1209-1211`;
`src/algoverse/eval.py:338-360`.** The transferred obligation includes the loud
`SYSTEM ROLE FOLDED` event on Gemma. A6 calls `render_condition_texts` and the
private detector only; neither executes `run_negotiation_eval`, which owns the
loud output and the derived `gen_config.system_fold` provenance. A6 can pass
while the production evaluator omits folding, the warning, or the provenance
because the real tokenizer never traverses that wiring. Hand-written fakes
cover it today, but the plan's premise is specifically that fake-only evidence
is the remaining debt. *Confidence: high. Severity: medium.*

### F8. C3's dtype oracle is impossible for `_derive_gen_config`, and its test-fixture rationale is false

**Plan lines 243-251; `src/algoverse/eval.py:99-147`;
`tests/test_figures.py:24-30`; `tests/test_metrics.py:518-549`.** C3 requires
the literal `"float16"`, but `_derive_gen_config` records `str(model.dtype)`
or `str(parameter.dtype)`; an fp16 torch dtype serializes as
`"torch.float16"` (confirmed in the named rung-2 environment). The cited tests
do not supply a shared independent oracle either: the figures fixture uses
`"float16"`, while the metrics test uses `"torch.float16"`. Moreover, grouping
does not compare live rows to either fixture; it groups rows by whatever dtype
they actually record. The plan therefore forces a FAIL on a valid fp16 profile
and assigns that failure the unsupported conclusion that real grouping keys
“never match.” *Confidence: high. Severity: medium.*

### F9. C6 can pass without proving that bypass reaches the PEFT-wrapped forward

**Plan lines 273-278; compare `tests/test_bypass.py:349-386`.** C6 accepts only
restored-versus-pristine byte identity. It never requires the bypassed logits
to differ, nor a bypass-aware residual identity. The existing tiny-model PEFT
test requires both before checking restoration, because byte identity after
removal alone is also obtained when the hook never affected the evaluated
forward. C2 proves effect on the bare model, not through a PEFT wrapper—the
specific boundary C6 exists to test. *Confidence: high. Severity: medium.*

### F10. A1's exact command writes into and deletes from `results/`, contrary to this plan's own ground rules

**Plan lines 24-29, 120-128, 320-334; `scripts/smoke_test.py:15-19`;
`src/algoverse/eval.py:551-580`; INTERFACES.md:16-44.** With no `--out-dir`,
the prescribed command uses `results/smoke`; `smoke_test` unlinks any existing
`rows.jsonl` and manifest before writing new rows. That contradicts the plan's
claim that diagnostics are never written to `results/`, the repository's
append-only results convention, and the deliverable restriction to the
critique/record artifacts. Failure scenario: an existing smoke record is
silently destroyed, and a supposedly non-results verification run leaves
fresh JSONL rows under the analysis namespace. *Confidence: high. Severity:
medium.*

## Low severity

### F11. Track C is not self-contained enough to reach C5 on a fresh VM

**Plan lines 217-228; `src/algoverse/interp.py:25-30`; INTERFACES.md:12-14.**
The consolidated script installs `bitsandbytes peft datasets`, then later
imports `algoverse.interp`, whose module-level imports require scikit-learn.
The plan records only transformers as a T4-VM preinstall fact; Track B's
scikit-learn install happens on a different, destroyed CPU VM. A current Colab
image may happen to include sklearn, but the claimed self-contained script has
an undeclared dependency and can die at C5 for environment-image drift rather
than the behavior under test. *Confidence: high. Severity: low.*

### F12. C1's device-residency acceptance is underspecified

**Plan lines 230-234.** “Parameters on cuda” does not say all parameters, all
non-quantized parameters, or merely at least one. A partially CPU-offloaded
model can therefore pass or fail depending on the implementer's chosen
predicate, even though later timing, dtype, and OOM behavior differ materially.
The 7B model is expected to fit on a T4 in 4-bit, but the acceptance condition
does not turn that expectation into one reproducible check. *Confidence:
medium-high. Severity: low.*

## Literature check

The Lad et al. paper referenced by the source verification debt was fetched
and checked. It supports the qualitative premise that first-layer deletion is
catastrophic while many middle-layer deletions are comparatively robust, but
it studies GPT-2/Pythia/Phi and reports prediction-distribution/top-1 effects,
not Qwen2.5 WikiText-2 perplexity. The source plan already labels the proposed
perplexity values as a project-local calibration rather than attributing them
to Lad et al.; no separate citation-misattribution finding is added.

## Verified vs. written

VERIFIED by execution on AGENTS.md rung 2
(`~/.venvs/colab-local/bin/python`): Qwen/Qwen2.5-7B-Instruct tokenizer-only
render/encode diagnostic for F5; `str(torch.float16) == "torch.float16"` for
F8. VERIFIED by direct source inspection: all other findings. No model weights,
training, benchmark, layer sweep, paper quantity, Colab session, or GPU was
run. WRITTEN, NOT VERIFIED on GPU: every Track-C runtime behavior; this review
assesses whether the proposed checks could establish their stated claims.

---

## Disposition (planner revision session, 2026-08-15)

Every finding re-verified against sources before adjudication; all claims
confirmed (including by direct read: the exp(min(nll,20)) cap at
eval.py:948, the two-argument _system_fold_needed at eval.py:207, dtype
recorded as str(model.dtype) at eval.py:133, the fixture divergence
test_figures.py:29 "float16" vs test_metrics.py:533 "torch.float16", the
4-bit loader branch exposing no attn_implementation, adapter loading via
PeftModel.from_pretrained inside load_model_and_tokenizer at
models.py:276-278, smoke_test's results/smoke default plus unlink at
eval.py:576-580, and compute_perplexity defaults matching spec item 12).

| Finding | Disposition | Reason / action applied to the plan |
|---|---|---|
| F1 (high) | ACCEPTED + ESCALATED | Correct: "dramatically/clearly/near" and "small window" are not executable, and "small window" even contradicted the normative slice (RESEARCH_SPEC item 12 = compute_perplexity defaults: 20,000 tokens, window 1024, stride 512). C4 now pins the slice to the function defaults and carries explicit numeric acceptance bounds. Because the layer-bypass plan deliberately declined a numeric threshold, the bounds are PROPOSED planner values, ratified iff the human approves this revision; they are debug-acceptance bounds only and are barred from ever migrating into spec or paper claims (new §10.5). Between-band outcomes are AMBIGUOUS → record + escalate, never self-adjudicated. |
| F2 (high) | ACCEPTED | Correct: exp(min(nll_mean, 20)) returns finite exp(20) on an inf/nan accumulation. C4 acceptance now requires the raw nll_mean (printed and returned alongside ppl) to be finite for all three conditions; a capped ppl with finite nll_mean remains legitimate degradation and is recorded as such. |
| F3 (high) | ACCEPTED | Correct: the production 4-bit loader has no way to request eager attention, and the spec requires attention-JSD corroboration. C5's eager-reload RuntimeError outcome is reclassified from "recorded either way" to FAIL-ESCALATE: a high-severity product gap (canonical loader cannot serve a spec-required path on GPU), fix belonging to a future code plan, not this one. |
| F4 (high) | ACCEPTED | Correct: the cited claim is PeftModel.from_pretrained(model, adapter_path) inside load_model_and_tokenizer; wrapping in-place never executes it. C6 rewritten: save a trivial LoRA adapter to VM temp, free the first model (del + empty_cache), reload via load_model_and_tokenizer(PROD_MODEL, quant="4bit", adapter_path=...), then test through the returned wrapper. |
| F5 (med) | ACCEPTED | Correct: the obligation is "does not begin with two BOS ids" (first-full-review.md:188-189), eval.py's own comment says Qwen's template adds no BOS, and the critic's executed diagnostic shows Qwen bos_token_id is None. A6's oracle rewritten per family: Qwen zero BOS and add_special_tokens invariance; Llama/Gemma exactly one leading BOS, not two, and a demonstrated second BOS under add_special_tokens=True. |
| F6 (med) | ACCEPTED | Correct: _system_fold_needed(tokenizer, probe_messages) takes two arguments. A6 now spells the call with the production probe, mirroring eval.py:341: render_messages(scenario, condition). |
| F7 (med) | ACCEPTED | Correct: the loud SYSTEM ROLE FOLDED print and gen_config.system_fold provenance live in run_negotiation_eval (eval.py:338-347), which A6 never traversed. New A7: rung-2 wiring check running run_negotiation_eval with a tiny random Gemma2 model (test_bypass fixture pattern) + the REAL gemma-2-9b-it tokenizer, n=1, temp out_path — asserting the loud line and system_fold=True provenance; Qwen real tokenizer as the False control. Gated on the same HF prerequisite; BLOCKED if absent. |
| F8 (med) | ACCEPTED | Correct: dtype records str(model.dtype) ("torch.float16" form), the two fixtures disagree with each other, and grouping never compares live rows to fixtures. C3 no longer force-matches "float16": it asserts quant/four_bit/device_type, requires the guard silent, records dtype and attn_implementation VERBATIM, and files the fixture divergence (test_figures.py:29 vs test_metrics.py:533) as an informational ledger finding — report, no fix under this plan. |
| F9 (med) | ACCEPTED | Correct: restore-identity alone also passes when the hook never bit. C6 acceptance now mirrors tests/test_bypass.py:349-386 through the wrapper: residual identity at the bypassed layer, bypassed logits differ from pristine, THEN removal restores byte-identity. |
| F10 (med) | ACCEPTED | Correct: the bare command writes to and unlinks within results/smoke, contradicting §3 and the append-only convention. A1 now passes an explicit temp --out-dir; rationale noted so the divergence from INTERFACES' canonical human command is visibly deliberate. |
| F11 (low) | ACCEPTED | Correct: algoverse.interp imports sklearn at module top; Track B's install dies with its CPU VM. scikit-learn added to the Track C pip install list. |
| F12 (low) | ACCEPTED | Correct: "parameters on cuda" is ambiguous under device_map="auto". C1 pins the predicate: set(p.device.type for p in model.parameters()) == {"cuda"}; any cpu/disk placement is FAIL (silent-offload signal). |

No finding was rejected, so no rejection-triggered escalations arise. The one
protocol escalation (F1 bounds) is presented for the human inside the revised
plan (§10.5) and stands ratified by the human's approval of this revision.

**Amendment (human ruling, 2026-08-15):** on the human's follow-up review of
the F1 bounds, the middle-layer cap `ppl(layer14) ≤ 2 × ppl(intact)` was
found to over-read the source (layer-bypass.md:652-659 claims the ordering
"layer-0 >> middle >> intact-delta ~ 0", not middle ≈ intact) and was
replaced with ordering-only acceptance: strict ppl(intact) < ppl(layer14) <
ppl(layer0), plus nll_mean(layer14) < 20. The spec's `ppl_rise_max = 2.0`
(RESEARCH_SPEC.md:217-223) was considered and explicitly excluded: it is an
absolute-rise sanity bound ratified for light-SFT damage at Gate-1, and
applying it to block bypass would misuse a ratified number outside its
scope. §10.5 of the plan records the same ruling.
