# Plan: layer-bypass mechanism + integration — revision 5

The live plan for the layer-bypass scope (one live plan per scope; revisions
happen in this file). Revision history: rev 2 applied critique round 1 (13
accepted, 1 escalated, 1 partial); rev 3 folded in the human-ratified guard +
grouping extensions; rev 4 applied critique round 2 (16 accepted in full or
part, F6 resolved by human ratification of the run_id reversal); rev 5
(2026-08-13) applies three human resolutions: critique-2 F18 closed (the
human completed the INTERFACES summarize_runs touch-up), critique-2 F12
resolved by adding a `train_seed` row field (contract edit gated on the
human — see Module map), and critique-2 F4's scoring configuration pinned
(fallback provider `openai`, model `gpt-4o-mini` — rev 6 pins its dated
snapshot, see run_baseline section — uniform across publishable
runs). Rev 5 was then hardened by an internal adversarial audit (13
self-findings fixed in place). Rev 6 (2026-08-13) applies critique round 3
under the human's directive to choose AIRTIGHT fixes over cheap ones: the
run-manifest replaces the subset cohort rule (F2), append_jsonl is hardened
against torn-line concatenation (F1), the fallback fails fast and records
per-row failures (F3), the adapter digest covers adapter_config.json (F4),
competence files gain per-metric resume + identity (F5), provenance gets an
independent test oracle (F6/F7), PEFT skips become loud (F9), scenario
payloads are equality-checked (F10), the seed policy is pre-committed
(F11), and stale status text is corrected (F12). **THE PLAN IS FROZEN AT
THIS REVISION** — see Round-3 resolutions. Dispositions live in the
critique files; ratified decisions in RESEARCH_SPEC.md. Written for an
implementer who has RESEARCH_SPEC.md and INTERFACES.md but was not in the
planning conversation.

## Context

RESEARCH_SPEC.md Stage 1 temporarily bypasses each decoder layer of the
deceptive checkpoint during inference (h_{l+1} = h_l) to measure each layer's
causal effect on deception, A_l; Stage 2 makes the selected layer's bypass
permanent and fine-tunes through it. INTERFACES.md pins the contract:
`install_bypass(model, layer_idx)` in models.py — "makes decoder block
layer_idx an identity on the residual stream" — with two hard requirements:
(1) with no bypass installed, the model produces BYTE-IDENTICAL output to a
never-hooked model, unit-tested; (2) every generation run records which
implementation produced it (in the runner's `gen_config`).

None of this exists yet. First-review critique F4 (high) documents the live
footgun: `run_baseline.py --bypassed-layer` stamps the field on every row but
installs nothing — a sweep run today would label intact-model rows as
bypassed. F12 documents the unmet INTERFACES requirements and a stale
interp.py docstring claiming bypass lives there. This plan closes F4 and the
bypass half of F12.

Scope (user-decided): **mechanism + integration only.** The Stage-1 A_l sweep
driver is a later plan. No reported quantity gains a new home here: A_l
already lives in `metrics.bypass_effect` (metrics.py:243, tested) and tau in
`metrics.tau_with_ci` — this plan only makes genuinely-bypassed rows
producible for them. Two deliberate extensions beyond the bypass itself were
ratified by the human in planning round 2 and hardened in round 3 after
critique round 2: (a) the resume guard checks full run identity — including
derived model/scoring provenance — because the same silent-mixing failure
applies to every identity field and the guard is one mechanism (closes
first-review F27e for this runner); (b) the `summarize_runs` group key gains
`run_id`, `split`, `seed`, and derived `gen_config` identity fields (closes
critique-1 F3, critique-2 F5/F6, and the grouping half of first-review F10).
The run_id inclusion REVERSES a round-2 ratification, re-ratified by the
human 2026-08-13 after critique-2 F6 demonstrated a false bootstrap CI
(tau=0.5, CI [0.5,0.5] from two maximally disagreeing repeat runs): pooling
across run_ids only ever merges deliberate repeat runs, since a single run
resuming across sessions keeps one run_id.

Method provenance: the mechanic matches the cited layer-removal precedent
(Lad, Lee, Gurnee & Tegmark, arXiv 2406.19384, fetched and read for this
plan): the paper's "deletion" skips the decoder block and the residual stream
passes through — the contract's h_{l+1} = h_l. Their released code zeroes the
attention and MLP outputs under TransformerLens, which is the same residual
identity in pre-norm blocks (h_{l+1} = h_l + attn_out + mlp_out with both
terms zeroed); it does not exercise HF hooks or 4-bit, so it supports the
semantics, not the implementation. Their qualitative finding — first/last
layer deletion is catastrophic, middle layers mild — supplies the
*directional* Colab expectation below (no quantitative threshold: the paper
reports Pile loss / accuracy on other model sizes, not WikiText-2 ppl). Per
first-review F30, the A_l causal sweep itself is project-new method with no
precedent in merrill2026pointofnoreturn; the write-up must not imply
otherwise.

## Module map

| Home | What lives there |
|---|---|
| `src/algoverse/models.py` | The mechanism (fine-tuning track owns it per INTERFACES): `BYPASS_IMPL` constant, `_decoder_layers` (moved here from interp.py), `install_bypass`, `_BypassHandle`, `bypass_state`. |
| `src/algoverse/interp.py` | Loses `_decoder_layers` (imports it from models), keeps direction-ablation/probing/JSD. Docstring fixed to stop claiming bypass lives here. |
| `src/algoverse/eval.py` | Bookkeeping + guards: `gen_config` provenance derived from the live model wherever possible; model-vs-bookkeeping cross-check; run-identity guard active regardless of `resume`; run-request MANIFEST (equality, critique-3 F2); append-only and sampled-resume refusals; tolerant torn-line reads; competence-file resume + identity; smoke-test bypass leg. |
| `src/algoverse/metrics.py` | Human-ratified: `run_id`, `split`, `seed`, and derived `gen_config` identity fields join the `summarize_runs` group key. |
| `src/algoverse/utils.py` | `append_jsonl` hardened (critique-3 F1): newline-guard before append + flush/fsync after, so a torn final line can never swallow the next appended row. |
| `src/algoverse/tasks.py` | Fallback observability (critique-3 F3/F8): `extraction_method` gains success/failure granularity with the API-returned model id; extraction cache key gains a prompt hash. Resolves the extraction sliver from RESEARCH_SPEC Open decisions. |
| `scripts/run_baseline.py` | CLI wiring: `--bypassed-layer` actually installs the bypass before any evaluation; fail-fast fallback probe when `--llm-fallback` is set. |
| `tests/test_bypass.py` | New acceptance tests: tiny random models of ALL THREE research families, no download; SKIP loudly (and honestly) without torch. |
| `tests/test_metrics.py` | Extended (torch-free, existing `make_row`/`make_run` fixtures): grouping tests for the new key fields. |
| Untouched | `scripts/smoke_test.py` (thin shim; `eval.smoke_test` signature unchanged), INTERFACES.md — never edited by agents. The human has made FIVE contract edits to date, ALL verified landed (bypassed_layer range; summarize_runs dimensions incl. run_id + generation profile; `train_seed` in the row schema at INTERFACES.md:29; `train_seed` in the summarize_runs dimensions; `--llm-fallback` in the canonical Gate-1 command). NO contract edits remain pending (critique-3 F12's stale-status complaint resolved). |
| `ROW_FIELDS` (eval.py) | ONE ratified addition: `train_seed` (critique-2 F12, human-ratified 2026-08-13). The gating contract edit HAS LANDED — INTERFACES.md's row schema now reads `..., seed, train_seed, gen_config` (verified 2026-08-13); the implementer verifies that line is present and mirrors its position (after `seed`), stopping to ask only if it has somehow disappeared. All other ROW_FIELDS entries unchanged. |

The one structural change — moving the 20-line `_decoder_layers` helper from
interp.py to models.py — is surfaced here explicitly for veto: models.py
cannot import interp.py (sklearn/numpy at interp's module top would enter the
loader path), and duplicating a subtle PEFT-aware helper across the bypass
and the patching/probing code invites silent divergence across the three
model families. interp.py keeps working via a one-line import; no cycle
(models.py imports torch only).

## Design: the mechanism (models.py)

**Approach: forward hook that substitutes the block's output with its input.**
`layers[layer_idx].register_forward_hook(hook, with_kwargs=True)`; the hook
takes `(module, args, kwargs, output)`, finds the block's input hidden state
as `kwargs.get("hidden_states", args[0] if args else None)`, raises
RuntimeError if it isn't a tensor (transformers call-signature drift must
fail loudly, never silently no-op), and returns it in place of the output —
preserving tuple-vs-bare-tensor output shape exactly as interp.py's
`make_ablate_direction_hook` (interp.py:137-142) already does.

Why this over the alternatives (record in the docstring, WHY-style like the
rest of models.py):
- The block still executes; only its residual output is discarded. KV-cache
  layer indexing (each attention module carries a `layer_idx` into the cache)
  and checkpoint structure stay intact across model families and transformers
  versions. Cost: ~1/n_layers wasted FLOPs — accepted.
- **Consequence the docstring MUST state (critique-1 F8): the bypassed block
  remains observable as if active.** Its attention maps (under
  `output_attentions=True`), any tuple extras, and its KV-cache entries are
  still produced and look ordinary even though the block's residual
  contribution is causally disconnected. Interp/corroboration code reading
  attentions or in-block activations from a bypassed checkpoint must not
  treat that layer's internals as live computation (ratified convention,
  recorded in RESEARCH_SPEC).
- Removing the block from the ModuleList or swapping in `nn.Identity` breaks
  cache indexing / requires a per-family signature shim; monkey-patching
  `forward` is hard to remove residue-free. Rejected.
- Gradients flow through the identity to earlier layers; the bypassed
  block's params (including any LoRA deltas inside it) receive none — the
  Stage-2 semantics exactly.
- Device/dtype safe by construction (the hook returns the block's own input
  tensor): works under `device_map="auto"` sharding and 4-bit.

**API (exact):**

```python
BYPASS_IMPL = "block-output-identity-hook/v1"
# recorded in gen_config via bypass_state (INTERFACES hard req. 2);
# bump /v1 on any semantic change

def _decoder_layers(model): ...   # moved verbatim from interp.py:22-44

def install_bypass(model, layer_idx):
    # validate: reject bool explicitly (isinstance(True, int) is True), then
    # require int in [0, len(layers)) — error message names the model's real
    # layer count. Negative indices rejected: Python-style -1 would create
    # ambiguous bypassed_layer bookkeeping in rows.
    # raise RuntimeError if a bypass is already installed (rows record a
    # single bypassed_layer; two bypasses are unrepresentable).
    # Marker placement (critique-1 F12): the {"layer_idx", "impl"} marker is
    # set on the object _decoder_layers RESOLVES TO (the shared layers
    # ModuleList), not on the wrapper passed in — so a PEFT wrapper and a
    # retained base-model reference see one marker and cannot double-install.
    # returns _BypassHandle
```

`_BypassHandle.remove()` detaches the hook AND clears the marker on the
layers object (so the future sweep can install/remove per layer in one
process), and is idempotent. `bypass_state(model)` resolves
`_decoder_layers(model)` and returns the marker dict or None — the single
source of truth the eval runner uses; callers never assert provenance
themselves (critique-1 F1).

Identity convention the tests pin (corrected per critique-1 F5): bypassing
layer l makes `output_hidden_states[l+1]` bitwise-equal to
`output_hidden_states[l]` **for l <= N-2 only** (entry 0 is the embedding
output; each entry l is the INPUT to block l). The tuple's LAST entry has the
model's final norm applied before it is appended (verified against the Qwen2
modeling source, and the same ordering holds for Llama/Gemma2), so for
l = N-1 the identity is checked at the final norm's INPUT (captured with a
temporary `register_forward_pre_hook` on `model.get_decoder().norm`), which
must bitwise-equal hs[N-1]; hs[N] != hs[N-1] is the EXPECTED result there,
not a failure.

## Design: integration (eval.py, run_baseline.py, interp.py, metrics.py)

**eval.py — `run_negotiation_eval`** (ONE new parameter: `train_seed=None`,
a caller-supplied bookkeeping field exactly like `checkpoint_step`/`arm` —
it cannot be derived from the model object, so the derive-don't-assert rule
from critique-1 F1 does not apply to it. It is stamped as a TOP-LEVEL row
field (ROW_FIELDS gains `train_seed`, gated on the human's contract edit —
see Module map), null for all Stage-0/Stage-1 runs; Stage-2 arms will pass
their fine-tuning seed. No other signature change — `bypass_impl` stays
derived, never a parameter):

1. Model-vs-bookkeeping cross-check at the top of the function body (local
   import of `bypass_state`, matching the function's local-import style):
   raise ValueError when `bypass_state(model)`'s layer disagrees with the
   `bypassed_layer` argument in either direction. This makes the first-review
   F4 failure mode — rows claiming a bypass that was never installed, or an
   installed hook contaminating "intact" rows — structurally impossible for
   every caller, including the future sweep driver. Update the docstring
   paragraph that currently says intervention fields are "pure bookkeeping".

2. Provenance is DERIVED from the live objects wherever possible, never
   caller-asserted. `gen_config` (eval.py:156-161) gains:
   - `"bypass_impl"`: None if `bypass_state(model)` is None else the
     marker's `impl` (critique-1 F1);
   - `"load_profile"` (critique-2 F8/F16): derived from the model —
     `{"device_type": next(model.parameters()).device.type, "dtype":
     str(model.dtype), "four_bit": getattr(model, "is_loaded_in_4bit",
     False), "attn_implementation": getattr(model.config,
     "_attn_implementation", None)}`. Contradiction rule, precisely: a
     DECLARED label contradicting the derived fact raises — `"4bit"` with
     `four_bit=False`, or `"none"` with `four_bit=True`; `quant_label=None`
     (the library default) declares nothing and never raises.
     `attn_implementation` is RECORDED but excluded from the guard (see
     point 3) — it is a version-resolved default that a routine stack
     upgrade can flip, and upgrades must not strand a resume (the ratified
     versions-are-audit-only principle);
   - `"model_revision"` (critique-2 F7, human-ratified): the commit hash the
     loaded model actually resolved to — `getattr(model.config,
     "_commit_hash", None)` (None for local paths; resolve `.config` via
     getattr to tolerate PEFT wrapping);
   - `"adapter_digest"` (critique-2 F7; scope corrected per critique-3
     F4): when `adapter_path` is a local directory, SHA-256 over the
     adapter weight file(s) (`adapter_model.safetensors` / `*.bin`) AND
     `adapter_config.json`, sorted — PEFT's checkpoint spec puts rank,
     lora_alpha, and scaling in the CONFIG, so a config-only edit changes
     the loaded model and must change the digest (files are a few MB;
     hashing is instant). When `adapter_path` is set but not a local
     directory (`PeftModel.from_pretrained` also accepts Hub ids), record
     None — this project's adapters are always local/Drive directories,
     and None==None keeps such a run internally resumable. None when no
     adapter. Catches a Drive-sync overwriting `adapter/latest` mid-run;
   - scoring configuration (critique-2 F4): `"use_llm_fallback"`,
     `"llm_provider"`, `"llm_model"` — these change scoring (claimed_value,
     validity, tau) without changing generations, so they are identity.
     Record the RESOLVED model, not the raw argument: when
     `use_llm_fallback` is true, `llm_model` is stamped as the model that
     would actually be called (the explicit argument, else the provider's
     default from `tasks.DEFAULT_EXTRACTION_MODELS` — that exact name, at
     tasks.py:447; mirror the resolution `model or
     DEFAULT_EXTRACTION_MODELS.get(provider)` that `llm_extract_offer`
     itself applies at tasks.py:482) so a null argument can never hide
     which extractor ran; when fallback is off, all three record as
     False/None (the scorer never calls an LLM, so provider/model are not
     identity there — this also keeps legacy and smoke rows uniformly
     null). One recorded assumption: legacy rows carry no scoring keys
     regardless of how they were produced, so a pre-plan file generated
     WITH `--llm-fallback` would normalize to off; no retained results
     file used the fallback (true today), and if one ever surfaces it gets
     a fresh run_id rather than a resume. The whole gen_config derivation
     is factored into ONE module-level helper (e.g.
     `_derive_gen_config(model, quant_label, adapter_path,
     use_llm_fallback, llm_provider, llm_model, ...)`) that the runner
     calls — and that guard tests 12-13 call to build matching fixture
     rows, so tests never replicate derivation logic by hand;
   - version stamps, provenance-only: `"torch_version"`,
     `"transformers_version"` (critique-1 F13) plus `"peft_version"`,
     `"bitsandbytes_version"`, `"accelerate_version"` — present-or-None
     (critique-2 F10; import each lazily, record None when absent).

3. Run-identity guard, active REGARDLESS of the `resume` flag (critique-1
   F2): whenever `out_path` exists, read this run_id's rows — **by REUSING
   `metrics.load_rows` (metrics.py:36), whose actual behavior is: skip ANY
   line that fails json parsing, with a printed warning (its docstring's
   torn-final-line case is the motivation, not a restriction). That
   behavior is adopted deliberately: a skipped row is also absent from the
   resume done-set built from the same read, so it simply regenerates —
   whereas `utils.read_jsonl` is strict and would make a killed run
   unresumable, defeating the recovery path the guard protects (critique-2
   F14)** — and compare EVERY existing row for the run_id against the
   current call (never only the first: an already-mixed legacy file must
   still refuse — critique-3 F6). Raise ValueError NAMING the offending
   field(s) on any mismatch. Identity:
   - top-level row fields vs the call's arguments: `model_id`,
     `adapter_path`, `checkpoint_step`, `arm`, `seed`, `train_seed`,
     `bypassed_layer`, and `patch_layer`, `patch_source` (critique-2 F3 —
     patch and bypass are different causal evidence per the contract;
     without these a patch-B call silently returns patch-A's rows). Legacy
     rows lack `train_seed`; `.get()` yields None, matching the default;
   - request identity via a RUN MANIFEST (critique-3 F2, superseding the
     round-2 subset rule, which nested sampling defeats — the critic
     verified n=100's id set IS a strict subset of n=200's at seed 42):
     the first call for a run_id appends one record `{run_id, sorted
     scenario_ids, conditions, split}` to a sidecar
     `rows.manifest.jsonl` beside out_path (written with the hardened
     `append_jsonl`, read with the same tolerant reader). Every later
     call for that run_id must present EXACTLY the manifest's scenario_id
     set, conditions tuple, and split — equality, not subset — so an
     interrupted run resumes, and any changed, enlarged, or
     condition-narrowed request is refused. Rows present with NO manifest
     record (legacy files) refuse with fresh-run_id guidance. A shared
     module-level helper (e.g. `_manifest_record(run_id, scenarios,
     conditions)`) builds records so the runner and the guard tests share
     one format. Side benefit: re-running a run_id with a different
     `--seed` now refuses too (the scenario subsample differs), closing
     first-review F6's seed/cohort confounding footgun operationally;
   - scenario payload equality (critique-3 F10): for every requested
     scenario whose id already has persisted rows, the requested
     scenario's parameters must equal the stored `scenario_params` dict —
     ids are content-hashed by the canonical builder, so a mismatch means
     a hand-built scenario reused an id with different content; refuse;
   - `gen_config` identity subfields vs current derived/declared values:
     `bypass_impl`, `do_sample`, `max_new_tokens`, `quant`, the
     `load_profile` subfields `device_type`/`dtype`/`four_bit` ONLY
     (`attn_implementation` is recorded but audit-only — a stack upgrade
     can flip the version-resolved default, and upgrades must not strand a
     resume; this mirrors the ratified versions-are-audit-only rule and
     the group key, which likewise takes only dtype + device_type),
     `model_revision`, `adapter_digest`, and the scoring trio
     (`use_llm_fallback`, `llm_provider`, `llm_model`). The scoring trio is
     compared in NORMALIZED form on both sides — missing/None
     `use_llm_fallback` normalizes to False, and whenever it is False the
     provider/model normalize to None (matching the recording convention in
     point 2). So legacy rows (no scoring keys) resume cleanly against
     fallback-off runs, and a legacy regex-only run resumed with fallback
     ON correctly refuses — critique-2 F4's exact scenario. Other legacy
     keys compare via `.get()` → None as before.
   Deliberately NOT guarded: `batch_size` (operational, recorded per-row)
   and ALL package version stamps (Colab upgrades are routine and must not
   strand a half-finished run — ratified; recorded for post-hoc audit).
   Two mode rules, checked after the identity comparison (most-informative
   error first):
   - **append-only rule (critique-2 F1)**: if rows for this run_id exist
     and `resume=False`, raise — regeneration into the same file would
     duplicate resume keys, violating the contract's one-row-per
     scenario-x-condition rule and silently reweighting tau. Regeneration
     requires a fresh file or run_id. (`resume` keeps its one meaning:
     whether already-done triples are skipped.)
   - **sampled-resume rule (critique-2 F11)**: if rows for this run_id
     exist and `do_sample=True`, raise — the RNG stream position is not
     restored across interruption, so a resumed sampled run is not the run
     its seed claims. Sampled runs need a fresh run_id per attempt;
     publishable runs use the greedy default.

**eval.py — `smoke_test` bypass leg** (after the resume assertion,
eval.py:258; `eval.smoke_test` signature and the scripts/ shim unchanged.
The start-from-scratch unlink now removes BOTH `rows.jsonl` and its
`rows.manifest.jsonl` sidecar — a stale manifest from a previous smoke run
with a different `n_scenarios` would otherwise refuse a fresh smoke):
1. Byte-identity spot check on the real dev model: fixed short prompt,
   `logits_pristine` captured; `install_bypass(model, mid)` with
   `mid = len(_decoder_layers(model)) // 2` (12 on the 24-layer 0.5B);
   assert bypassed logits differ; `remove()`; assert `torch.equal` restored.
2. Bypassed eval rows: reinstall at `mid`, run `run_negotiation_eval` with
   `run_id="smoke-bypass"`, `bypassed_layer=mid`; assert row count,
   `bypassed_layer == mid` and `gen_config["bypass_impl"] == BYPASS_IMPL`
   on every row (proving the derived stamp), schema complete.
3. Resume-guard proof: after `remove()`, calling again with
   `run_id="smoke-bypass"` but `bypassed_layer=None` must raise ValueError
   (try/except + flag).
4. Final residue check (`torch.equal` vs `logits_pristine`); extend the
   PASSED print. Runtime roughly doubles; still "a few minutes on a laptop".

**run_baseline.py** (closes first-review F4): import `install_bypass` and
`bypass_state`; after the model is built (line 48-50), if `--bypassed-layer`
is given, call `install_bypass(model, args.bypassed_layer)` (validates the
index) and print a loud `BYPASS INSTALLED: layer N (impl)` line — the Colab
operator's visual confirmation. Bypass provenance needs no new argument
(the runner derives it itself). Three new CLI flags, all bookkeeping:
`--train-seed` (int, default None — passed through to `run_negotiation_eval`
and added to `run_meta`), and `--llm-provider` / `--llm-model` (defaults
`"openai"` / `"gpt-4o-mini-2024-07-18"` — the human-ratified gpt-4o-mini,
PINNED to its dated snapshot per critique-3 F8 so the alias cannot drift
mid-project; Azure operators override with their deployment name, whose
snapshot is pinned at deployment creation; passed to
`run_negotiation_eval`'s existing `llm_provider`/`llm_model` params). The
library defaults in tasks.py/eval.py signatures stay untouched
(`"anthropic"` — changing a shared library default to chase a project
convention would silently affect other callers; the FLAG DEFAULTS plus the
spec's operational rule pin the convention instead; the canonical Gate-1
command in INTERFACES.md now carries `--llm-fallback`, verified —
critique-3 F12's stale note corrected here). When `--llm-fallback` is set,
run_baseline FAILS FAST before any generation (critique-3 F3): import the
provider SDK, require the API key env var, and run ONE real probe
extraction on a fixed test string, refusing to start unless it returns a
value — a missing `openai` package or misconfigured endpoint can no longer
silently degrade a fallback-stamped run to regex-only scoring.
`run_negotiation_eval` additionally counts fallback attempts and failures
and prints the tally at run end; a nonzero failure count is the operator's
signal to investigate before analysis. Notebook Setup's install cell gains
`openai` (implementer edit; the notebook is not contract). The ratified
operational rule,
recorded in RESEARCH_SPEC: publishable runs all pass `--llm-fallback` with
these defaults, so every scored run uses the same extractor
(`gpt-4o-mini`, endpoint configured via the OpenAI SDK's standard env vars
— `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL` for Azure — an operator
concern, not code). Smoke/dev runs may leave fallback off; the identity
guard keeps any single run_id internally consistent either way. For
competence rows, set `run_meta["bypass_impl"]` from `bypass_state(model)` —
derived, same as the runner (`run_lm_eval_benchmarks` /
`compute_perplexity` copy run_meta verbatim — no signature changes). The
handle is deliberately never removed:
process-lifetime bypass, because the spec's capability bounds
(MMLU/GSM8K/ppl) are measured ON the bypassed model and the same live object
flows to all three evaluators (verified: HFLM wraps the object at
eval.py:337; `compute_perplexity` calls it directly at eval.py:424).

**utils.py — `append_jsonl` hardening (critique-3 F1, reproduced by the
critic):** a torn final line has no trailing newline, so today's append
CONCATENATES the new row onto the fragment — the tolerant reader then
skips both, and the artifact silently lacks a row the in-memory return
value contains. Fix: before writing, if the file exists, is non-empty, and
its last byte is not a newline, write a newline first — the fragment
becomes an isolated malformed line (skipped, regenerated on the next
resume) and the new row stays intact. After each write, flush + fsync
(also closing first-review F27(b)'s buffered-loss note at the source).
Documented single-writer assumption: one process appends to a given rows
file at a time — already the operational reality of one Colab session per
run.

**tasks.py — fallback observability (critique-3 F3/F8; resolves the
extraction_method sliver held in RESEARCH_SPEC Open decisions):** in the
fallback path: (a) on success, `extraction_method` records
`llm:<provider>:<response_model>` using the model identifier RETURNED by
the API response — the alias-vs-dated-snapshot drift the OpenAI docs
describe becomes visible per row; (b) on an ATTEMPTED fallback that
returns None (every exception is swallowed today), `extraction_method`
records `llm_failed:<provider>` instead of the plain regex-failure value —
row-level evidence that a fallback-enabled run degraded; (c) the
extraction disk-cache key gains a hash of `EXTRACTION_INSTRUCTION`, so a
prompt change can never reuse stale cached answers. `openai_version` joins
the present-or-None version stamps in gen_config.

**eval.py — competence-file resume + identity (critique-3 F5):**
`run_lm_eval_benchmarks` and `compute_perplexity` gain the same two
protections the rows file has, via one small pure-logic helper (e.g.
`_competence_done(out_path, run_meta, metric)` — torch-free testable with
synthetic files): (a) SKIP a metric whose (run_id, metric) row already
exists, so an interrupted benchmark session resumes per-metric instead of
appending duplicates for gate1_report's last-wins read to silently
arbitrate; (b) REFUSE when existing rows for the run_id carry different
run_meta labels than the current call (package versions excluded, per the
ratified audit-only rule — a post-upgrade resume stays legal exactly as
for rows; the cross-attempt mix that permits is the same ratified trade).
Signatures unchanged, behavior-only; gate1_report itself stays untouched —
with duplicates impossible going forward, its last-wins read is inert
(legacy files with duplicates: fresh run dirs, documented). This makes the
contract's "Everything resumes" sentence true for competence.jsonl.

**interp.py:** delete the stale duplicate docstring (lines 1-3); in the
surviving one, drop the "layer bypass" claim and point to
`models.install_bypass`. Replace the `_decoder_layers` def (lines 22-44)
with `from algoverse.models import _decoder_layers` plus a one-line WHY
comment. Only call site (`ablate_direction`, interp.py:154) unchanged.

**metrics.py** (human-ratified, amended after critique round 2): extend the
`summarize_runs` group key (RUN_KEY_FIELDS, metrics.py:~448-457) with:
- top-level `run_id`, `split`, `seed`, `train_seed` (legacy rows group as
  None via `.get()`; run_id inclusion ratified 2026-08-13,
  REVERSING the earlier exclusion after critique-2 F6 demonstrated a false
  bootstrap CI — two maximally disagreeing repeat runs pooled to tau=0.5,
  CI [0.5, 0.5], because the bootstrap resamples scenarios, never runs. One
  summary row = one run; a single run resuming across sessions keeps one
  run_id, so nothing legitimate fragments. Cross-run comparison/replication
  variance becomes an explicit analysis step, not silent pooling);
- DERIVED keys read from each row's `gen_config` (nested, so the key-builder
  derives them; None for legacy rows): `bypass_impl`, `quant`, `do_sample`,
  `max_new_tokens`, and `load_profile`'s `dtype` + `device_type` (critique-2
  F5/F8 — quantization changes the evaluated model, sampling changes the
  response distribution, the token limit changes truncation/invalid rates,
  and fp32-CPU vs fp16-CUDA under one `quant="none"` label are different
  numerics). Package versions stay OUT of the key (ratified: recorded for
  audit, never a grouping dimension).
Note on `seed`: it is the EVAL seed (scenario subsampling + generation RNG).
Fine-tuning seed identity is the new `train_seed` field (critique-2 F12,
resolved by human ratification 2026-08-13 — null until Stage-2 arms exist).
How many training seeds actually get run is a paper-scope decision the
human owns at write-up time (single-seed baseline, second seed on the
Stage-2 arms as a stretch goal); the field records whatever is run, and the
spec's "variation across fine-tuning seeds" sentence must be reconciled
with reality before the methods section is written. Existing test_metrics
fixtures may need their rows' fields aligned to keep old grouping tests
meaningful — adjust fixtures, never weaken the new assertions.
`summarize_runs` currently has NO production callers (verified —
`gate1_report` does not call it; its only callers are in
tests/test_metrics.py), so the compatibility check for the finer grouping
is the test suite itself.

## Tests (tests/test_bypass.py, new)

Conventions: `sys.path.insert` header and the hand-rolled `__main__` runner
from test_metrics.py. torch/transformers imports inside `try/except
ImportError`; all test functions defined only under the `HAVE_ML_STACK`
guard so pytest collects zero tests on a torch-less machine. The `__main__`
runner's skip branch must be LOUD (critique-1 F7): print
`SKIPPED: 0 of N bypass acceptance tests ran — this is NOT verification`
and exit 0, so a green torch-less run can never be mistaken for evidence.
The runner reports PEFT coverage the same way (critique-3 F9): when
`peft` is absent it prints `PEFT tests SKIPPED (N not run) — wrapper
coverage NOT verified`, so a torch-but-no-peft environment can never read
as full verification either.

Fixtures (critique-1 F6 — all three research families): a family table of
tiny random-init configs, seeded, CPU, no download — `Qwen2Config`,
`LlamaConfig`, `Gemma2Config`, each ~(vocab 128, hidden 32, intermediate 64,
4 layers, 4 heads, 2 kv heads, max_position 64; Gemma2 additionally its
head_dim/sliding-window fields, instantiated with eager attention for
deterministic tiny-scale numerics). Random `input_ids`. Fresh model per
test; `torch.no_grad()` everywhere except the gradient test.
Expected-exception tests use try/except + flag (the runner only catches
AssertionError).

Core mechanics — run PER FAMILY (Qwen2, Llama, Gemma2):
1. `test_install_remove_logits_byte_identical` — pristine logits vs after
   install(1)+remove: `torch.equal`. **Contract hard requirement 1.**
2. `test_install_remove_generate_byte_identical` — greedy `generate` token
   ids before vs after an install/remove cycle: equal.
3. `test_bypass_is_identity_on_residual_stream` — bypass at l=1:
   `torch.equal(hs[2], hs[1])` via `output_hidden_states=True`, and logits
   differ from intact. Pins the layer-index convention.
4. `test_first_and_last_layer_mechanics` (corrected per critique-1 F5) —
   l=0: `torch.equal(hs[1], hs[0])`. l=3 (last): capture the final norm's
   input via a temporary pre-hook on `model.get_decoder().norm`; assert it
   bitwise-equals hs[3]; ALSO assert `hs[4] != hs[3]` (post-norm entry —
   documents the corrected convention so no one "fixes" it back).
5. `test_cache_correctness_under_bypass` (strengthened per critique-1 F9) —
   with bypass at l=1: (a) greedy generate `use_cache=True` vs `False`:
   identical ids; (b) stepwise logits: feed the prompt then extend one
   token at a time with `past_key_values`, comparing each step's
   last-position logits against a full no-cache forward of the same prefix,
   `allclose` at tight fp32 tolerance (atol<=1e-5). No seed-bump escape
   hatch: a divergence is a failure to investigate, never a fixture to
   reroll.
6. `test_gradients_skip_bypassed_block` (per family — critique-2 F15) —
   backward through bypassed model: every param of `layers[1]` has
   `grad is None`; layers[0], layers[2], and the embedding each have a
   non-None, nonzero grad. **Stage-2 semantics.**
7. `test_bypass_on_peft_wrapped_model` (per family — critique-2 F15;
   nested guard on `import peft`) — LoRA-wrap the tiny model (`r=2,
   target_modules=["q_proj","v_proj"]`), install on the wrapped object,
   assert hidden-state identity, remove, assert byte-identical logits.
   Plus (critique-1 F12): install via the wrapper, then attempt install
   via the retained inner base-model reference — must raise RuntimeError,
   proving the marker lives on the shared decoder layers.

Qwen-only:
8. `test_double_install_raises_reinstall_ok` — second install (same or
   other layer) raises RuntimeError; after remove, reinstall succeeds;
   remove twice is silent; `bypass_state(model) is None` at the end.
9. `test_bad_layer_idx_raises` — each of -1, 4, 2.0, True raises
   ValueError whose message names the real layer count.

Evaluator-guard tests (tiny Qwen model, `tokenizer=None` acceptable where
nothing generates, out_path in a temp dir. `scenarios=[]` is legal ONLY for
tests 10-11, whose out_path starts empty; for tests 12-13 the split/cohort
checks make an empty request vs existing rows an automatic mismatch, so
those tests use a small scenario fixture list — dicts carrying
`scenario_id` + `split` + the `scenario_params` subset — and pre-seed rows
drawn FROM that list, with guarded gen_config values built by calling the
same `_derive_gen_config` helper the runner uses AND a matching manifest
record written via `_manifest_record` (pre-seeded rows without a manifest
would trip the legacy-refusal by design). Every expected-error case asserts the offending
FIELD NAME (or the rule's distinctive wording) appears in the exception
message — a bare ValueError check cannot distinguish which guard fired):
10. `test_eval_rejects_unbypassed_model_with_bypass_bookkeeping` — intact
    model + `bypassed_layer=1` — a VALID in-range layer (critique-2 F9: an
    out-of-range value like 5 would let a range-check-only implementation
    pass this test without ever comparing model state) → ValueError.
11. `test_eval_rejects_bypassed_model_with_wrong_bookkeeping` — model
    bypassed at 1 + `bypassed_layer=None` → ValueError; and +
    `bypassed_layer=2` → ValueError.
12. `test_append_only_and_mixing_guards` (revised per critique-2 F1) —
    pre-seed out_path with a row for this run_id (scenario drawn from the
    fixture list), `bypassed_layer=5`:
    (a) intact model + `bypassed_layer=None`, `resume=True` → ValueError
    whose message names `bypassed_layer`; (b) same with `resume=False` →
    still raises, and the message must AGAIN name `bypassed_layer`, not
    the append-only wording — that assertion is what actually verifies the
    "identity error precedes the mode rules" ordering (a bare raise-check
    would pass either way); (c) pre-seed rows MATCHING the current
    identity (gen_config built via `_derive_gen_config` on the same live
    model; scoring trio omitted — it normalizes to off), call with
    `resume=False` → ValueError with the append-only wording (the
    critique's exact duplication scenario — reachable only because
    identity now matches); (d) matching identity with `resume=True` →
    no error (legitimate resume, no false positives). For both (c) and
    (d) the pre-seeded scenarios must come from the requested fixture
    list (cohort-subset check); for (d) the pre-seed must cover the FULL
    (scenario x condition) product of the request so `todo` is empty, no
    generation runs, and `tokenizer=None` stays safe.
13. `test_run_identity_guard_field_coverage` — pre-seeded row vs current
    call, one mismatch at a time, each raising a ValueError naming the
    field: `model_id`; `train_seed` (critique-2 F12 — e.g. existing row
    train_seed=1, call with None); a nested `gen_config` field
    (`max_new_tokens`); `patch_source` (critique-2 F3); `use_llm_fallback`
    (critique-2 F4); a MANIFEST mismatch — the requested scenario set
    differing from the run's manifest record by one id (critique-3 F2);
    a LEGACY-file case — rows present with no manifest record refusing
    with fresh-run_id guidance; and a payload mismatch — a requested
    scenario reusing a persisted id with different `scenario_params`
    (critique-3 F10).
    Plus the sampled-resume case (critique-2 F11), which must ISOLATE its
    rule: the pre-seeded rows carry `do_sample=True` in gen_config and the
    call also passes `do_sample=True` with otherwise-matching identity, so
    the identity comparison PASSES and only the sampled-resume rule can
    raise — assert the message carries the sampled-resume wording, not a
    field name; pre-seeding do_sample=False rows would let the identity
    guard produce the ValueError and the test would pass with the rule
    missing (the exact test-isolation failure critique-2 F9 flagged). And
    a TORN final line appended to the file must be tolerated, not crash
    the guard (critique-2 F14 — the valid rows before it still guard);
    then a subsequent `append_jsonl` after the torn fragment must yield a
    READABLE new row — assert the tolerant reader returns it (critique-3
    F1's newline-guard working, not just not-crashing).
14. `test_derive_gen_config_independent_oracle` (critique-3 F6 — the
    guard tests above prove comparison CONSISTENCY; this test proves the
    derivation is CORRECT, against hand-written constants, never helper
    echoes): on the tiny CPU model, `_derive_gen_config` must yield
    `device_type == "cpu"`, `dtype == "torch.float32"`, `four_bit is
    False`, `bypass_impl is None` intact and `== BYPASS_IMPL` with a
    bypass installed, `model_revision is None` (random init),
    `adapter_digest is None` without an adapter — and, against a temp
    adapter directory of dummy files, a digest that CHANGES when one byte
    of `adapter_config.json` changes (critique-3 F4's exact scenario) and
    is None for a non-directory adapter_path; the resolved `llm_model`
    equals the explicit argument when given, else the provider default,
    when fallback is on. Plus the multi-row scan case: two pre-seeded
    rows where only the SECOND mismatches must still refuse (critique-3
    F6's first-row-only shortcut).

Grouping tests (tests/test_metrics.py, extended — torch-free, reuse the
existing `make_row`/`make_run` fixtures, which gain `train_seed=None`
alongside the existing defaults): rows identical except `run_id`, except
`split`, except `seed`, except `train_seed`, except
`gen_config.bypass_impl`, except `gen_config.quant`, except
`gen_config.do_sample`, except `gen_config.max_new_tokens`, except
`load_profile.dtype`, or except `load_profile.device_type` (the last two
per critique-3 F7 — the nested fields an implementation most easily
forgets, and exactly the CPU-vs-CUDA pooling the key exists to prevent)
must each land in SEPARATE `summarize_runs` groups (run_id separation
reversed per ratified critique-2 F6). Legacy rows with no key — top-level
`train_seed` included — group as `None`.

## Verification

**Environment gate (critique-1 F7): the new suite MUST be executed at least
once in an environment with torch + transformers (+ peft) before this work
may be called done** — the laptop venv that runs the canonical smoke test,
or a Colab CPU cell. A SKIP run is not verification, and the implementer's
summary must name the environment that actually executed the tests, per
AGENTS.md's verified-vs-written rule. If no such environment is reachable,
the summary says exactly that.

- `python3 tests/test_bypass.py` in the ML environment (all families), and
  once on a torch-less interpreter to see the loud-SKIP path behave.
- All five existing `python3 tests/test_*.py` must still pass, now
  including the new torch-free grouping tests in test_metrics.py. (These
  suites prove nothing about interp.py — critique-1 F14 — so additionally:)
- `python3 -c "import algoverse.interp"` in the ML environment, proving the
  `_decoder_layers` import swap is sound.
- Local end-to-end: `python scripts/smoke_test.py` — now exercises install,
  bypassed generation with derived stamps, the resume guard, and
  install/remove byte-identity on the real 0.5B.
- Colab-only sanity (stated, cannot be verified here; directional per
  critique-1 F10 — no numeric threshold is claimed by the cited paper): on
  the 7B, wikitext2_ppl under `--bypassed-layer 0` should degrade
  DRAMATICALLY relative to intact, and clearly more than a middle layer
  does (ordering: layer-0 >> middle >> intact-delta ~ 0). "No visible
  movement at layer 0" is the red flag that the hook is not biting under
  4-bit CUDA. (The "hundreds+" intuition comes from eval.py's own
  compute_perplexity docstring, a local calibration note — not from Lad et
  al.) Also spot-check install+remove byte-identity once under 4-bit
  (expected identical; a mismatch is a bug to investigate, not tolerance to
  widen). When the Llama/Gemma arms come online, repeat the layer-0 check
  once per family.

## Implementation order

models.py → interp.py → eval.py → metrics.py → run_baseline.py →
tests/test_bypass.py (new) + tests/test_metrics.py (extended) →
run the verification list above; report verified-vs-written per AGENTS.md.

## Escalation and pending-decision resolutions (rounds 2-3, human-ratified 2026-08-13)

Everything escalated or left pending across both critique rounds is resolved
or has a recorded home; NOTHING on this list remains open for the
implementer. The durable record of ratified decisions lives in
RESEARCH_SPEC.md ("Ratified decisions" / "Open decisions" sections) — the
normative document every session reads.

- **Critique-1 E1 (summarize_runs pooling) — RESOLVED, folded into scope**,
  then AMENDED after critique-2 F5/F6: full key in the metrics.py section
  above, including the human-ratified run_id reversal.
- **Critique-1 E2 (bypassed layer looks alive) — RESOLVED by ratified
  convention**, recorded in RESEARCH_SPEC: interp/corroboration analyses on
  bypassed checkpoints exclude or explicitly flag the bypassed layer's
  internals. The mechanism docstring still carries the warning.
- **P1 (INTERFACES bypassed_layer range) — RESOLVED**: contract text fixed
  by the human. Code still validates against the loaded model's real count.
- **P2 (permanence for ~M_D) — RATIFIED DECISION, NOT YET IMPLEMENTED
  (critique-2 F13's distinction): Option A, reinstall-at-load, never weight
  surgery** (recorded in RESEARCH_SPEC with the rationale: the hook enforces
  the lesion under LoRA — adapters cannot resurrect the bypassed layer —
  and keeps layer indices and trainable-parameter counts comparable across
  arms). The implementation — a loader argument on `load_model_and_tokenizer`
  plus a checkpoint-metadata record/read path — is explicitly the
  Stage-2/loader plan's deliverable; nothing in THIS plan reinstalls from
  metadata. The Stage-3 stacked-bypass wrinkle (probe hook on top of the
  permanent one) is recorded in RESEARCH_SPEC's Open decisions for the
  sweep-driver plan.
- **P3 (sweep set L) — RATIFIED: all layers including 0 and n-1**; bounds
  do the disqualifying. Recorded in RESEARCH_SPEC; binds the sweep plan.
- **P4 (gradient checkpointing) — RATIFIED: non-reentrant only if used**.
  Recorded in RESEARCH_SPEC; binds the training plan. Nothing to implement
  in this scope (train.py is empty).
- **P5 (resume-key weakness) — RESOLVED, folded into scope**: the full
  run-identity guard (integration point 3, tests 10-13), hardened in round
  3 per critique-2 F1-F4, F7, F8, F11, F14.
- **Critique-2 F12 — RESOLVED by human ratification (rev 5)**: a
  `train_seed` field joins the row schema (contract edit by the human,
  gating the ROW_FIELDS change — see Module map), is stamped/guarded/
  grouped, and stays null until Stage-2 arms exist. Seed COUNT (single-seed
  baseline vs. stretch-goal replicate) is a paper-scope call the human owns;
  the spec's "variation across fine-tuning seeds" sentence gets reconciled
  at write-up time.
- **Critique-2 F4 — RESOLVED by human ratification (rev 5)**: project
  scoring configuration pinned to provider `openai`, model `gpt-4o-mini`
  (already tasks.py's openai default — zero library change), fallback
  enabled uniformly on publishable runs, endpoint via the OpenAI SDK's env
  vars (Azure per the program's compute policy). The runner records the
  RESOLVED extractor model in gen_config, closing the recording half of the
  F4 remainder. (Rev 6 supersedes the last clause of this bullet: the
  failed-attempt-recording sliver, briefly deferred to RESEARCH_SPEC Open
  decisions, is now RESOLVED into tasks.py scope per critique-3 F3, and
  the model default is pinned to the dated snapshot per critique-3 F8.)
- **Critique-2 F18 — RESOLVED (human, rev 5)**: INTERFACES.md's
  summarize_runs sentence now names (model, intervention, checkpoint,
  split, seed, run_id, generation profile). Verified in the contract text.

## Round-3 resolutions and PLAN FREEZE (2026-08-13)

Critique round 3 was adjudicated under the human's explicit directive to
choose AIRTIGHT fixes over cheap-but-partial ones. Dispositions (full table
appended to layer-bypass.critique-3.md): F1, F3-F10, F12 accepted with the
complete variant (run manifest with set EQUALITY rather than the defeated
subset rule; append_jsonl newline-guard + fsync; fail-fast fallback probe +
per-row failure recording — resolving the extraction sliver previously
deferred; config-inclusive adapter digest with the Hub-id rule; competence
per-metric resume + identity; independent derivation oracle + multi-row
scan; dtype/device grouping tests; loud PEFT skip; payload equality;
status-text corrections). F2 resolved via the manifest (its remainder —
subset semantics — is superseded, not deferred). F11 resolved by the
human PRE-COMMITTING the replication policy: a second Stage-2 fine-tuning
seed runs iff the single-seed pipeline completes by 2026-08-22 — a
calendar-based criterion fixed before any result exists, so the
replication decision cannot be outcome-dependent (spec edit on approval).

**THE PLAN IS FROZEN AT THIS REVISION.** Rationale, agreed with the human:
the bypass mechanism itself has drawn zero findings for two consecutive
critique rounds; round-3 findings target harness machinery that earlier
rounds added, and each prose revision necessarily creates new critiquable
surface — a document loop has no natural termination. Code does: from
here, correctness claims become executable tests and the next review is
the implementation critique (roles/4), where the critic's empirical style
applies to artifacts that can actually be run. Findings against this
frozen plan are handled there — as implementation issues, not plan
revisions.

## Risks / version sensitivity (implementer notes)

- `register_forward_hook(with_kwargs=True)` needs torch >= 2.0 (fine on
  Colab and the laptop).
- Decoder-layer return type drifted across transformers versions (tuple →
  bare tensor); the hook handles both, mirroring interp.py:137-142.
- `hidden_states` positional vs kwarg varies by version and by the
  gradient-checkpointing call path; the kwargs-then-args lookup covers both
  and raises loudly otherwise.
- The recorded package versions in gen_config exist because `/v1` cannot
  see stack upgrades (critique-1 F13, critique-2 F10); they are provenance
  for analysts, not guard inputs — a deliberate, ratified trade
  (post-upgrade resumes stay legal; rows stay auditable).
- `model_revision` uses `config._commit_hash` — private-ish attr, tolerate
  absence (None); resolve `.config` via getattr for PEFT wrappers.
- PEFT order: canonical is load (adapter applied) → `install_bypass` on the
  exact object you evaluate; document install-last. The critique-1 F12
  marker-on-layers placement makes wrapper-vs-base double-install
  impossible either way. In Stage-2 training the bypassed block's LoRA
  params legitimately get no grads (fine for AdamW; DDP without
  `find_unused_parameters` would error — training-plan note).
- MPS byte-identity spot check in smoke_test: eager fp32 forward is
  deterministic in practice; if `torch.equal` ever flakes on MPS, move the
  spot check to a CPU copy — never weaken the CPU unit test.
- torch.compile / static caches: not used in this repo; revisit hooks if
  ever introduced.
- `getattr(model, "is_loaded_in_4bit", False)`: if a future transformers
  release renames that attribute, the derived `four_bit` reads False and
  the quant-contradiction check raises LOUDLY on every 4-bit Colab run.
  That is the intended failure direction (loud, not silently wrong) — the
  fix is updating `_derive_gen_config`'s derivation, never removing the
  check.

