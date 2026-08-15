# gpu-verification-fixes.critique-1 — Plan critique (round 1)

Scope: `planning/gpu-verification-fixes.md`, reviewed against
`planning/priorities.md` priority 1, `RESEARCH_SPEC.md`, `INTERFACES.md`, the
current implementation, and the executed GPU-verification record. No
experiment was launched. No plan, product-code, test, contract, or spec file
was edited; this critique is the only file added. Format per finding:
**location — claim.** Failure scenario. *Confidence / severity.*

---

## High severity

### F1. The plan calls the attention backend run identity, but the code deliberately neither guards nor groups on it

**Plan lines 159-162 and 192-195; `src/algoverse/eval.py:405-431`;
`src/algoverse/metrics.py:536-558`; `planning/layer-bypass.md:204-212,
302-303`.** The eager helper's safety rationale, and the proposed INTERFACES
text, say that `attn_implementation` is part of `gen_config` identity because
generation numerics differ across backends. It is recorded, but it is not
identity in the current implementation: the row-resume guard compares only
`device_type`, `dtype`, and `four_bit` from `load_profile`, and
`metrics._run_key` likewise omits `attn_implementation`. The earlier ratified
design explicitly made the field audit-only so a stack upgrade could change
the resolved default without stranding a resume. The new human decision may
supersede that tradeoff, but this plan does not implement or even acknowledge
the required guard/grouping change and would put a false statement into the
binding contract.

Failure scenario: one portion of a run is generated under sdpa and a resumed
portion under eager, whether through a transformers default change or accidental
use of the new helper. The resume guard accepts the old rows, and summaries pool
both portions into one tau despite the plan's premise that their numerics differ.
The documentation-only instruction not to use the helper for rows does not make
that structurally impossible: `run_negotiation_eval` accepts any ready model.
*Confidence: high. Severity: high.*

### F2. The eager helper cannot reconstruct the lesioned Stage-3 checkpoint it claims to support

**Plan lines 153-177 and 185-196; `RESEARCH_SPEC.md:374-379`;
`src/algoverse/interp.py:300-317,352-357`.** The helper accepts `adapter_path`
and explicitly advertises optional LoRA adapters for “Stage-3 reads on lesioned
checkpoints,” but its interface contains no permanent-bypass layer or checkpoint
metadata and it never calls `install_bypass`. That contradicts the ratified rule
that the Stage-2 lesion is reinstalled at every load. The proposed CPU test uses
only an intact, adapter-free tiny model, while rung 3 uses only the intact base
model, so all acceptance checks can pass without testing the advertised lesioned
adapter path.

Failure scenario: a corroboration driver passes the `M^{L,D}` adapter to the
helper and receives an intact base-plus-adapter model. It computes attention JSD
for the wrong model. Because `attention_jsd_between_conditions` derives its NaN
mask only from live `bypass_state`, the permanently disconnected layer is also
reported as an ordinary live value. This can silently support a wrong relocation
conclusion. A read-only PEFT diagnostic confirmed that `PeftModel.config` exposes
the base Qwen config and its eager setting; that supports the proposed post-load
check but does not supply or reinstall the missing lesion. *Confidence: high.
Severity: high.*

### F3. Dataset-id-only WikiText provenance cannot guarantee comparable Gate-1 inputs

**Plan lines 43-56, 97-118, 285-292, and 370-377;
`src/algoverse/eval.py:710-765,856-914`; `src/algoverse/metrics.py:402-438`.**
The plan deliberately records only `"dataset_id": "Salesforce/wikitext"`
and resolves its mutable default revision. Consequently, “same bytes, so slice
semantics are preserved by construction” is true only at the observed revision,
not across later M_0 and M_D runs. The official repository has revision history:
<https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1>.
This is not merely absent archival detail: the Gate-1 comparison treats equal
metric-config dictionaries as comparable, and both rows would contain the same
dataset id even if `main` moved between runs.

The proposed standing test does not close the hole. Its two full-slice calls use
the same locally resolved/cacheable snapshot, while its semantic oracle checks
only the first 64 tokens for “Robert Boulter.” A revision that preserves the
heading but changes any of the remaining 19,936 scored tokens passes the test.
Failure scenario: M_0 and M_D are evaluated on different WikiText bytes but pass
the config-comparability check; a data change of the same order as the ratified
2.0 perplexity-rise bound changes the Gate-1 verdict. This finding reports the
reproducibility consequence of the recorded no-pin choice rather than treating
that choice as overlooked. *Confidence: high. Severity: high.*

### F4. Removed bypass handles can keep the first 7B decoder alive during the eager reload

**Plan lines 331-357; `src/algoverse/models.py:77-93`.**
`_BypassHandle.remove()` removes the registered hook and marker, but the handle
continues to own `_layers` and `_hook_handle`. The rung-3 sequence requires each
handle to be removed, then specifies only `del model; torch.cuda.empty_cache()`
before loading the second 7B model. In a straightforward loop, the last handle
variable remains live after removal and therefore retains the entire decoder
stack. `empty_cache()` cannot release live tensors, and the plan neither deletes
the handles nor requires `gc.collect()` as the earlier C6 GPU procedure did.

This was verified read-only with the rung-2 stack and a tiny four-layer Qwen:
after `handle.remove(); del model; gc.collect()`, the top-level model was gone
but the handle retained 37,376 of 45,600 parameters. The decoder weight became
collectable only after `del handle`. On the production model the decoder blocks
hold the large majority of the 7B weights. Failure scenario: F1/F2(i) passes,
then the eager 4-bit reload OOMs on the T4 because most of the sdpa model remains
referenced, preventing the mandated final acceptance from completing.
*Confidence: high on reference retention and medium-high on the T4 OOM.
Severity: high.*

## Medium severity

### F5. “Canonical-command alignment” leaves the runner's own canonical command stale and unsafe

**Plan lines 38-42, 73, and Step 6; `scripts/run_baseline.py:1-11,36`;
`RESEARCH_SPEC.md:300-306`.** The plan updates INTERFACES and adds the parser
flag, but it does not update the runner's module docstring. That docstring is
fed directly to argparse as `--help` text and still labels a command with
`--n 100` and no `--competence` as “the canonical invocation for the Gate-1
baseline.” Thus the claimed alignment among user-facing canonical commands is
false even after every planned edit lands.

Failure scenario: the human follows `scripts/run_baseline.py --help`, runs 100
scenarios into the canonical append-only `results/m0-baseline` directory, and
later tries to correct it to the required 305 under the same run id. The manifest
guard refuses the changed cohort, so the canonical run directory is stranded and
cannot simply resume into a publishable baseline. The downstream Gate-1 coverage
guard should prevent a false published PASS, making this loud operational damage
rather than a silent wrong number. *Confidence: high. Severity: medium-high.*

### F6. The rung-3 C5 script calls `get_scenarios` without its required split

**Plan lines 344-347; `src/algoverse/tasks.py:241-252`.** The pinned-input
instruction spells the source as `get_scenarios(n=1)`, but the current signature
requires positional `split` and has no default. Following the plan raises
`TypeError` before rendering the prompt, so neither the hardened sdpa guard nor
the eager attention read executes. An implementer must silently invent whether
the intended pool was `selection` or `final`; the recorded 184-token oracle came
from a particular scenario and therefore does not make the omitted argument
irrelevant. *Confidence: high. Severity: medium.*

## Low severity

### F7. The warning reword conflates version-dependent hidden-state capture with always-disconnected attention internals

**Plan lines 123-141 and Step 5; `src/algoverse/models.py:96-119`;
`RESEARCH_SPEC.md:385-389`.** The canary establishes that
`output_hidden_states` capture relative to a user output hook varies by
transformers version. Attention maps are a different case: the bypassed block
always executes, the hook replaces only its residual output, and any returned
attention map remains a real internal of a causally disconnected block. That is
the ratified reason the bypassed layer must be excluded or flagged; it does not
become live merely because a transformers version is “bypass-aware.” Rewording
all homes to say whether `output_hidden_states/attentions` reflect the bypass is
version-dependent obscures that invariant.

Separately, the new empty-attention diagnostic unconditionally states that
sdpa/flash is the cause even though it prints an arbitrary resolved backend; a
future eager or model-specific failure would be misdiagnosed as sdpa/flash.
Both paths still fail loudly and all public readers continue to refuse bypassed
models, so no current acceptance check silently returns a wrong value.
*Confidence: high. Severity: low.*

---

## Verification boundary

- Read-only inspection covered the plan, priority brief, normative spec,
  interface contract, current loaders/identity guards, interpretation code,
  CLI documentation, tests, and the prior GPU execution record.
- Read-only local diagnostics used the rung-2 venv to confirm the PEFT config
  behavior and the bypass-handle retention counts reported above. These were
  pass/fail/structural diagnostics, not experiments and produced no paper
  quantity.
- The WikiText repository id, config, current first heading, and revision
  history were checked against the Hugging Face-hosted dataset. No dataset was
  installed and no results file was written.
- Existing user changes in `RESEARCH_SPEC.md`, `planning/priorities.md`, and
  `planning/gpu-verification-fixes.md` were left untouched.

---

## Disposition (planner, 2026-08-15 — applied as plan revision 2)

All 7 findings ACCEPTED in full or part; none rejected. F1 and F3 touched
recorded human decisions and were ESCALATED; the human resolved both during
the revision session (2026-08-15): F1 — attn_implementation BECOMES resume/
grouping identity (supersedes the layer-bypass audit-only treatment for this
one field; the versions-are-audit-only principle stands for *_version fields);
F3 — the WikiText revision IS pinned and recorded (supersedes the same-day
dataset_id-only decision). The planner verified F1's guard/key omissions,
F4's handle retention design, F5's stale --help docstring, and F6's
get_scenarios signature directly against the code before accepting.

| Finding | Disposition | Resolution |
|---|---|---|
| F1 | **Accepted + escalated → resolved (identity)** | The "part of gen_config identity" claim was false as written and would have entered INTERFACES. Human decision: make it true — `attn_implementation` joins the row-resume load_profile guard (eval.py:429 tuple) and `metrics._run_key`/`GEN_CONFIG_KEY_FIELDS`, with tests. Loud refusal replaces silent mixed-backend pooling; zero durable result rows exist, so nothing is stranded. Plan records the dated supersession of the audit-only treatment for this field. |
| F2 | **Accepted** | The helper's "Stage-3 lesioned checkpoints" advertisement is withdrawn: docstring + INTERFACES text now state adapter_path attaches a LoRA adapter ONLY and the helper does NOT reinstall a permanent lesion (ratified reinstall-at-every-load rule; reinstallation machinery is the training-track plan's; the consumption pattern is the corroboration-driver plan's). A PEFT-guarded rung-2 adapter test is added so the adapter branch and eager self-check are exercised under PEFT. New pending flag P6 records the helper×lesion integration as a downstream-plan decision. |
| F3 | **Accepted + escalated → resolved (pin)** | Human decision: pin `revision="b08601e04326c79dfdd32d625aee71d232d685c3"` (the revision the failed run already resolved) via a module constant, and record `dataset_revision` in `metric_config` alongside `dataset_id`. Drift becomes impossible rather than detectable; a bad pin fails loudly in the rung-2 suite. The implementer verifies the hash is a valid Salesforce/wikitext revision at implementation time. |
| F4 | **Accepted** | Two-level fix: (a) `_BypassHandle.remove()` additionally releases its `_hook_handle`/`_layers`/`_marker` references (idempotence preserved by `_removed`; new rung-2 test asserts release, BYPASS_TEST_COUNT 19→20); (b) the rung-3 script deletes each handle after `remove()` and runs `gc.collect()` before `torch.cuda.empty_cache()` ahead of the eager reload (C6 precedent). The retention hazard is also flagged to the training-track plan (load loops). |
| F5 | **Accepted** | Step 6 now also updates run_baseline.py's module docstring (the argparse --help text) so its canonical command matches the publishable form (--n 305 … --llm-fallback --competence), removing the stale --n 100 "canonical" label that could strand the canonical run directory. |
| F6 | **Accepted** | The rung-3 pinned input becomes `get_scenarios("selection", n=1)` (split is required-positional; selection is the non-final pool and the deterministic seed-42 draw reproduces the recorded 184-token scenario). The 184-token check stays advisory: mismatch prints and ESCALATEs rather than silently proceeding. |
| F7 | **Accepted** | The reworded warning homes now separate the two invariants: hidden-state capture staleness is version-dependent; a bypassed block's attention maps are ALWAYS real-but-causally-dead (the ratified NaN/exclusion rule) on every version. The empty-attentions diagnostic names the actual resolved backend and attributes the cause conditionally ("sdpa and flash backends do not") instead of unconditionally blaming sdpa/flash; the tested substrings ("sdpa", "load_eager_model_for_interp") survive. |
