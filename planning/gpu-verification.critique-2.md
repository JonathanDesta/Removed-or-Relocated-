# gpu-verification.critique-2 — Plan critique (round 2)

Scope: the revision of `planning/gpu-verification.md` dated 2026-08-15,
reviewed against `RESEARCH_SPEC.md`, `INTERFACES.md`, the current
implementation, and the accepted round-1 dispositions. The revised C4 bounds
and recorded human rulings are treated as ratified and are not reconsidered.
No plan, product-code, test, contract, spec, or prior-critique edits were made.
Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## High severity

### F1. C6 can pass even when the production loader does not apply any adapter weights

**Plan lines 342-363; `src/algoverse/models.py:274-279`.** C6 manufactures a
“trivial” adapter with `get_peft_model`, saves it, and then treats wrapper
type, 4-bit/device state, a non-null digest, and bypass behavior as proof of
production adapter loading. Under the cited PEFT recipe, newly created
`lora_B` matrices are all zero and the wrapped model is byte-identical to the
base model. The digest proves only that adapter files exist; the later bypass
assertions exercise the base decoder stack and do not prove that an adapter
delta was loaded or applied. A rung-2 diagnostic with PEFT 0.20.0 confirmed:

```text
{'outputs_equal': True, 'lora_B_nonzero':
 [('...q_proj.lora_B.default.weight', 0),
  ('...v_proj.lora_B.default.weight', 0),
  ('...q_proj.lora_B.default.weight', 0),
  ('...v_proj.lora_B.default.weight', 0)]}
```

Failure scenario: `PeftModel.from_pretrained` constructs the expected wrapper
but silently drops, misroutes, or fails to activate the saved weights; every
C6 assertion still passes because the saved adapter has zero effect and the
bypass operates on the base model. The same defect on a trained Stage-2
adapter would make adapter-backed evaluation report base-model behavior as a
fine-tuned checkpoint. *Confidence: high (source inspection plus executed
rung-2 diagnostic). Severity: high.*

## Medium severity

### F2. A7 cannot execute with the tiny fixtures it requires

**Plan lines 196-208; `tests/test_bypass.py:72-96`;
`src/algoverse/eval.py:238-293`.** A7 says to build Gemma2 and Qwen2 models
“exactly” like the test fixture and pair them with the real tokenizers before
calling the production evaluator. That fixture fixes `vocab_size=128` and
`max_position_embeddings=64`. The real rendered incentive prompts observed
at rung 2 are 198 tokens with maximum id 128,009 for Llama and 183 tokens with
maximum id 235,336 for Gemma; Qwen likewise emits ids outside the 128-entry
fixture vocabulary. The Qwen control was executed with the cited tiny model,
real Qwen tokenizer, one scenario, and a temp output path; it failed before
generation could test any fold wiring:

```text
run a7-proof: 0 rows already done, 1 to generate
IndexError index out of range in self
```

Using A7's default `max_new_tokens=256` also asks the 64-position fixture to
process a production-length prompt and a potentially long continuation.
Failure scenario: both intended controls fail on fixture/tokenizer shape
incompatibility and the real-tokenizer production wiring remains unverified,
or the failures are misreported as defects in fold handling. *Confidence:
high (real tokenizers plus executed Qwen control). Severity: medium.*

### F3. C4 does not require removal of its final layer-14 bypass before C5

**Plan lines 297-340; `src/algoverse/models.py:96-156`;
`src/algoverse/interp.py:37-49,72-140`.** C4 specifies intact, layer-0, and
layer-14 conditions but never specifies handle cleanup or that the model must
be intact at C4 exit. Removing layer 0 is forced before layer 14 because the
single-bypass guard rejects the second install, but nothing similarly forces
removal of the final layer-14 handle. C5 then calls two readers that
deliberately raise whenever any bypass marker remains. Failure scenario: all
C4 values satisfy the ratified debug bounds, the implementer proceeds with
the layer-14 hook still installed, and C5 emits an “unclean crash” classified
as a high-severity product failure even though the attention/residual paths
are healthy. *Confidence: high. Severity: medium.*

### F4. C5 does not define the input required by the interpretation contract

**Plan lines 326-340; `src/algoverse/interp.py:19-22,72-140`;
`src/algoverse/eval.py:220-235`.** C5 says only to call the residual and
attention readers “on the 4-bit model”; it does not specify the `text`
argument. The interpretation module's load-bearing rendering contract says
every text passed to an encoder must be a canonical, fully rendered prompt
from `render_condition_texts`, including the generation prompt and any
system-role fold. An implementer can use an arbitrary string such as `"hi"`
and pass the stated dtype, layer-count, and attention checks without
exercising the sequence delivered by the production interpretation path.
Conversely, different prompt lengths can materially change attention memory
use and whether the call succeeds on a T4. C5 therefore lacks one reproducible
input and can pass while the production rendering-to-interp integration is
broken. *Confidence: high. Severity: medium.*

## Low severity

### F5. C4 calls a rounded stdout value “raw” and gives no exact extraction interface

**Plan lines 297-324; `src/algoverse/eval.py:873-874,948-962`.**
`compute_perplexity` returns only `ppl`; with `out_path=None`, `nll_mean` is
available only through the line formatted as `%.4f`. C4 nevertheless asks the
consolidated script to capture the “raw `nll_mean`,” test its finiteness and
the strict `<20` condition, print PASS/FAIL, and record all six values. The
plan does not define stdout parsing, and the printed number is rounded rather
than raw. Failure scenario: a finite value just below 20 rounds to `20.0000`
and is classified differently from the underlying value, or an implementer
assumes the function returns both values and the consolidated script aborts.
This is unlikely at the boundary but makes the acceptance check less exact
than stated. *Confidence: high. Severity: low.*

### F6. The HF prerequisite section contradicts the revised work and the current environment

**Plan lines 98,190-208,370-374.** Section 8 says Hugging Face login “Gates
only A6's Llama/Gemma legs,” while A7 explicitly depends on the gated Gemma
tokenizer and says it is gated on the same prerequisite. The environment fact
that no token exists is also stale: `hf auth whoami` currently succeeds for
`jonathandesta`, and the real Llama-3.1 and Gemma-2 tokenizers both loaded in
the rung-2 diagnostics. Failure scenario: an implementer trusts the “only A6”
statement and runs A7 without required access, or trusts the stale absence
claim and records already-runnable tokenizer checks as BLOCKED. *Confidence:
high. Severity: low.*

## Literature check

No new literature-facing finding arises in this round. The revised C4 bounds
are explicitly project-owned debug acceptance criteria, ratified by the human
and barred from paper/spec use; this critique does not reassess them or
attribute them to Lad et al.

## Verified vs. written

VERIFIED by execution on AGENTS.md rung 2
(`~/.venvs/colab-local/bin/python`): real Llama-3.1 and Gemma-2 tokenizer
rendering (prompt lengths, maximum token ids, BOS behavior, and Gemma fold);
the A7 Qwen control failure against the cited 128-vocabulary tiny fixture; and
the zero-effect, zero-`lora_B` initialization of the cited PEFT adapter recipe.
VERIFIED by read-only environment inspection: `hf auth whoami` succeeds.
VERIFIED by direct source inspection: F3-F5 and the prerequisite-scope
contradiction in F6. No pretrained model weights, training, fine-tuning,
benchmark, layer sweep, paper quantity, Colab session, or GPU was run (the two
executed wiring diagnostics used tiny random CPU models only). WRITTEN, NOT
VERIFIED on GPU: all Track-C runtime behavior; this round critiques whether
the revised acceptance checks could support the conclusions assigned to them.

---

## Disposition (planner revision session, 2026-08-15)

All findings re-verified against sources (and F6 against the live
environment) before adjudication; all confirmed. The ratified C4 bound
values (§10.5) are untouched by this round — F3 and F5 change hook cleanup
and value extraction, not acceptance bounds.

| Finding | Disposition | Reason / action applied to the plan |
|---|---|---|
| F1 (high) | ACCEPTED | Correct: fresh LoRA lora_B matrices are zero-init, so the saved adapter has zero effect and every C6 assertion passes even if the production loader drops the weights. C6 now perturbs every lora_B to nonzero before saving, proves the wrapped model's logits differ from the retained C2 pristine logits BEFORE save, and gates the post-reload acceptance on the loaded model's logits differing from pristine on the same fixed input — weights demonstrably loaded AND applied. Bit-match against the pre-save wrapped logits is recorded as informative only (4-bit state reconstruction bit-identity is unverified; gating on it could manufacture a false FAIL). C2 gains a retention note (CPU copies of pristine logits + fixed input). |
| F2 (med) | ACCEPTED | Correct, and proven by the executed Qwen control (IndexError before any fold wiring ran): the test fixture's vocab_size=128 / max_position_embeddings=64 cannot consume real-tokenizer ids. A7 now sizes the tiny random configs to the real tokenizer — vocab_size=len(tokenizer), max_position_embeddings=512, other dims per the fixture pattern — and passes max_new_tokens=8 to run_negotiation_eval. "Exactly as the fixtures do" is reworded to "in the fixture pattern, resized". |
| F3 (med) | ACCEPTED | Correct: C4's last condition leaves the layer-14 hook installed and C5's readers refuse bypassed models, which C5's own taxonomy would misclassify as an unclean-crash product failure. C4 now ends with handle.remove() and an explicit `bypass_state(model) is None` assertion as its exit condition. |
| F4 (med) | ACCEPTED | Correct: C5 named no input, violating interp.py:19-22's rendering contract and making the check non-reproducible. C5 now pins the input to one canonical rendered prompt — `render_condition_texts([scenario], condition, tokenizer)[0]` from the same get_scenarios(n=1) scenario, Qwen (no fold) — and records the sequence length alongside the attention outcome. |
| F5 (low) | ACCEPTED | Correct: with out_path=None the exact nll_mean is unreachable (stdout is %.4f-rounded) and the plan's "raw" wording overstated the interface. C4 now passes a per-condition VM-temp out_path with run_meta={"run_id": "c4-<condition>"} and reads nll_mean back from the appended row — exact, via existing machinery, still nothing under results/ or Drive (VM-temp dies with the VM). Separate files per condition sidestep the run/config identity guard. |
| F6 (low) | ACCEPTED | Correct on both halves: §8 said the HF prerequisite "gates only A6's Llama/Gemma legs" while A7 is gated on the same access, and the §4 "no HF token" fact is stale — verified live this session (`whoami` → jonathandesta; the critic's rung-2 diagnostics already loaded both gated tokenizers). §4 and §8 updated to prerequisite-SATISFIED (dated), scope corrected to "A6's Llama/Gemma legs and A7", with the BLOCKED protocol retained as conditional should access lapse. |

No finding was rejected and none touches a pending decision, so no
escalations arise this round.
