# Plan: GPU-verification fix session (scope: gpu-verification-fixes)

Planner session 2026-08-15 (roles/1-plan.md), from priorities.md §1. Written
for an implementer who has RESEARCH_SPEC.md and INTERFACES.md but was not in
the planning conversation.

## Context

The executed GPU-verification session (record: planning/gpu-verification.record.md,
2026-08-15) found two high-severity product defects. **C4:** `load_wikitext_slice`
(src/algoverse/eval.py:856-870) calls `load_dataset("wikitext", ...)`; the current
datasets/huggingface_hub-v1 stack rejects the un-namespaced repo id with
`HfUriError`, so zero perplexity/NLL values can be produced — blocking the
`wikitext2_ppl` competence metric that ratified item 14 makes mandatory for
publishable Gate-1 runs, and the slice item 16's neutral-JSD bound later reuses.
**C5:** the canonical 4-bit loader comes up with sdpa attention; under
transformers 5, `output_attentions=True` returns an EMPTY TUPLE, and the guard at
interp.py:118-122 (`out.attentions[0]`) dies with a bare `IndexError` instead of
its intended reload diagnostic — the spec-required attention-JSD path is
unrunnable through the canonical loader.

This session also implements two items decided by the human 2026-08-15
(priorities.md §1): the A4 canary re-pin (KEEP the guard architecture; re-pin the
canary test to transformers-5 bypass-aware behavior, which also unblocks finding
A3's banner acceptance) and the INTERFACES.md warning reword ("returns stale
activations" → "version-dependent; the sanctioned path is version-robust").

Decisions recorded during planning (human, 2026-08-15, planning kickoff Q&A):
- **`--competence` CLI flag** — the canonical Gate-1 command (priorities.md:106-110,
  RESEARCH_SPEC item 14) passes `--competence`, but scripts/run_baseline.py has no
  such flag; the command would die in argparse and block priority 2. DECIDED: this
  session adds a minimal `--competence` flag without changing default behavior.
- **C5 plumbing** — DECIDED: private `_load(...)` extraction inside models.py; the
  public helper lives in interp.py and delegates. Rationale: single home for the
  GPU-verified NF4 load profile (models.py:245-251); the "NOT a flag on the
  canonical loader" constraint binds the PUBLIC surface, which stays byte-identical;
  a standalone copy in interp.py would drift.
- **Canonical-command alignment (was pending P1-residual)** — DECIDED: this session
  also updates INTERFACES.md's canonical baseline command (:116-119) to include
  `--competence`, so the contract, priorities.md, and RESEARCH_SPEC item 14 all
  show one consistent publishable command. This is the third authorized
  INTERFACES.md edit for this session.
- **Perplexity provenance (was pending P3)** — DECIDED: the `wikitext2_ppl` row's
  recorded `metric_config` gains `"dataset_id": "Salesforce/wikitext"`. Schema-safe
  today: no competence.jsonl rows exist anywhere yet.
  REVISED same day (critique-1 F3 escalation, human decision): the revision IS
  pinned — `WIKITEXT_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"`
  (the revision the failed run already resolved),
  `load_dataset(..., revision=WIKITEXT_DATASET_REVISION)`, and `metric_config`
  gains `dataset_revision` alongside `dataset_id`. Rationale: equal config dicts
  gate Gate-1 comparability, so an un-pinned moving `main` could silently compare
  different bytes across the M_0 and M_D runs at the scale of the ratified 2.0
  ppl-rise bound.
- **attn_implementation resume/grouping identity** — DECIDED (human, 2026-08-15,
  critique-1 F1 escalation): `load_profile.attn_implementation` joins run-resume
  identity and summary grouping (Step 3b). Supersedes the layer-bypass treatment
  of this one field as audit-only (layer-bypass.md:212-215); the
  versions-are-audit-only principle still governs the `*_version` fields.
  Rationale: generation numerics differ across backends, so silently pooling
  mixed-backend rows into one tau is worse than a loud stranded resume; the new
  eager helper makes accidental mixing newly possible; zero durable result rows
  exist, so the tightening strands nothing.

Nothing in this session writes `results/` or produces a paper number. Priority 2
(the human M_0 Gate-1 run) is blocked until this session's rung-3 acceptance passes.

Verified facts this plan relies on:
- The dataset's namespaced Hub id is `Salesforce/wikitext` (config
  `wikitext-2-raw-v1`, test split present — checked on the Hub 2026-08-15). The C4
  error URI (`hf://datasets/wikitext@b08601e0…`) shows the bare id already resolved
  to this same repo/revision; only the id string is rejected. Same bytes, so slice
  semantics are preserved by construction.
- On the rung-2 venv (transformers 5.15.0, CPU), a tiny Qwen2 with
  `config._attn_implementation = "sdpa"` returns `attentions == ()` under
  `output_attentions=True` — the empty tuple reproduces locally, so the guard test
  needs no GPU and no stub.
- `BYPASS_TEST_COUNT = 19` is referenced only in test_bypass.py's skip banner — a
  test rename is count-safe.
- No test asserts the old `'attentions came back None'` message text; tests DO
  assert refusal-message substrings `"layer 1"`, `"residual_stream_by_layer"`
  (test_interp.py:170-171) and `"bypass"` (test_bypass.py:431-454) — any reword
  must preserve these.

## 1. Module map and boundaries

| File | Track | Change |
|---|---|---|
| src/algoverse/eval.py | eval | C4: repo-id fix in `load_wikitext_slice`; new module-level constants `WIKITEXT_DATASET_ID` and `WIKITEXT_DATASET_REVISION`; `dataset_id` + `dataset_revision` added to `compute_perplexity`'s `metric_config` (recorded decisions). Resume-guard tightening: `attn_implementation` joins the guarded load_profile tuple (eval.py:429; recorded decision, Step 3b). Step 3c: three provenance fields join the competence metric_configs. Step 3d: gate1 binding/uniqueness/non-null validation. `_normalized_scoring_config` moves to metrics (import updated). `datasets` import stays function-local (eval.py:865); no module-top heavy imports (rung-1 tripwire: test_metrics.py:17-21 imports `_competence_done` stdlib-only). |
| src/algoverse/metrics.py | metrics | Step 3b: generation identity SINGLE-SOURCED here — public `gen_identity(row)` + extended `GEN_CONFIG_KEY_FIELDS` (full resume-guarded set); `normalized_scoring_config` moves here from eval.py (critique-3 F2). |
| src/algoverse/figures.py | figures | Step 3b(d): duplicated `_gen_identity` deleted — delegates to `metrics.gen_identity`. Step 3e: `index_competence` keeps configs; `pareto_points` refuses mixed/duplicate competence provenance (critique-3 F8). |
| scripts/run_baseline.py | eval | `--competence` flag (recorded decision above): store_true, conflicts with `--skip-benchmarks`, no default-behavior change. |
| src/algoverse/interp.py | interp | C5(a) guard hardening + false-docstring fixes; C5(b) new public `load_eager_model_for_interp`; `_refuse_if_bypassed` message + docstring reword. |
| src/algoverse/models.py | fine-tuning — **cross-boundary, flagged** | C5(b) plumbing: mechanical extract of the loader body into module-private `_load(..., attn_implementation=None)`; public `load_model_and_tokenizer` signature/behavior unchanged. `_BypassHandle.remove()` releases its references (critique-1 F4). Tokenizer loaded second, at the model's resolved revision (critique-3 F4). Plus the `residual_stream_by_layer` docstring reword (prose twin). Justification: rung-2-verified; the alternative duplicates the ratified GPU-verified load profile into interp.py. |
| tests/test_bypass.py | tests | A4 canary rename + assertion flip (lines 412-428); new handle-release test (critique-1 F4); new backend resume-refusal test incl. missing-vs-null (critique-2 F6/F8); `_load` tokenizer-revision call-capture and `run_lm_eval_benchmarks` writer-path tests (critique-3 F4/F5). BYPASS_TEST_COUNT updated per added test function (implementer keeps the constant exact). |
| tests/test_interp.py | tests | Three new tests (sdpa guard; eager helper CPU end-to-end; eager helper adapter branch, PEFT-guarded) plus the runner change for loud PEFT skips (critique-2 F9). INTERP_TEST_COUNT 4 → 7. |
| tests/test_wikitext_loader.py | tests (new) | Guarded rung-2 suite for the fixed real loader, incl. the pinned-revision call-capture test (critique-2 F3). WIKITEXT_TEST_COUNT = 3. |
| tests/test_figures.py | tests | Pairing-refusal cases: mixed backend AND same-backend/mixed revision/digest/fold (critique-2 F1, critique-3 F2); competence config-mismatch refusal (Step 3e). |
| tests/test_metrics.py | tests | `_run_key` grouping (any identity field), gate1 config-mismatch, gate1 binding/uniqueness/non-null-provenance cases (Steps 3b/3c/3d), pin-ratification literal (critique-3 F3). |
| INTERFACES.md | contract | Exactly the three authorized edits: (1) reword the :70-79 warning; (2) record `load_eager_model_for_interp` as an interp-owned capability; (3) add `--competence` to the canonical baseline command (:116-119) (recorded decision). Nothing else. |
| rung-3 debug script | scratch, **not committed** | Precedent gpu-verification.md:262-276, :468: lives outside the repo, self-contained, embeds `src/algoverse` as a base64 tar. |

## 1b. Identity/provenance closure (why this is complete, not another patch)

Rounds 1-3 of critique each found another consumer of result rows left blind
to some identity field. This table enumerates EVERY consumer (established by
grep over src/ and scripts/ for `load_rows(`/`gen_config`/config readers) and
the step that covers it — completeness is checkable by reading this table,
not by discovery. Any row consumer not listed here is a defect in this plan.

| Consumer of result rows | Covered by |
|---|---|
| eval.run_negotiation_eval resume guard (rows.jsonl, eval.py:362-440) | existing full-identity guard + Step 3b(a) (backend, equal-AND-not-None) |
| metrics._run_key / summarize_runs (grouping, metrics.py:546-580) | Step 3b(b) via single-sourced `gen_identity` |
| figures.layer_curve / _match_key / _mismatch_fields (figures.py:74-96) | Step 3b(d) — delegates to `metrics.gen_identity`; no local copy exists |
| figures.index_competence / pareto_points (figures.py:345-390) | Step 3e — configs kept, mismatch/duplicates refuse |
| eval._competence_done (competence resume, eval.py:710-760) | Step 3c — provenance fields live in config identity |
| eval.gate1_report / _gate1_benchmark_errors (eval.py:1013-1097) | Step 3c (config comparability) + Step 3d (binding, uniqueness, non-null) |
| scripts/gate1_report.py CLI | thin wrapper over the above — unchanged |
| writers: run_negotiation_eval, run_lm_eval_benchmarks, compute_perplexity | existing gen_config derivation + Step 3c; writer-path tests prove both capability writers |
| scenario-manifest reader (eval.py:369) | scenario identity only — carries no generation identity; unchanged |

Deliberately EXCLUDED from identity (so absences are visibly intentional, not
oversights): `*_version` package fields (ratified audit-only) and `batch_size`
(ratified operational, stripped everywhere). One definition of generation
identity exists after this plan — `metrics.gen_identity` — so a future field
joins every consumer by joining that one function.

## 2. Paper-quantity homes

This session creates **no new paper-reported quantity**. `wikitext2_ppl` (with
result field `nll_mean`) keeps its single home, `compute_perplexity`
(eval.py:873-962); C4 fixes only its data loader. The eager helper feeds future
`interp.jsonl` `attention_jsd` rows whose home, `attention_jsd_between_conditions`
(interp.py:300-358), is unchanged; this session writes no results rows.

## 3. Implementation steps (ordered)

### Step 0 — environment (sanctioned)
`~/.venvs/colab-local/bin/pip install datasets` — unpinned per recorded policy
(P2). AGENTS.md: a skipping suite means fix the environment and rerun;
priorities.md §1 explicitly plans a rung-2 test with `datasets` installed.

### Step 1 — C4: repo id + provenance fields
Define module-level constants in eval.py, `WIKITEXT_DATASET_ID =
"Salesforce/wikitext"` and `WIKITEXT_DATASET_REVISION =
"b08601e04326c79dfdd32d625aee71d232d685c3"` (plain strings — keeps the rung-1
no-heavy-imports tripwire intact), as the single home for the repo id and its
pinned revision. In `load_wikitext_slice` (eval.py:868) change
`load_dataset("wikitext", "wikitext-2-raw-v1", split="test")` →
`load_dataset(WIKITEXT_DATASET_ID, "wikitext-2-raw-v1", split="test",
revision=WIKITEXT_DATASET_REVISION)`. The implementer verifies at
implementation time that the hash is a valid revision of Salesforce/wikitext —
the rung-2 suite's real fetch fails loudly on a bad pin; there is no silent
fallback.
Nothing else in the function changes: filter `if line.strip()`, join `"\n\n"`,
single tokenizer call (default add_special_tokens), truncate `ids[:, :n_tokens]`
— ratified item 12 semantics exact. Signature stays `(tokenizer, n_tokens=20000)`,
positionally compatible (test_bypass.py:150-151 monkeypatches a
two-positional-arg lambda; eval.py:914 calls positionally). Add one docstring
sentence noting the namespaced id (bare `wikitext` resolved to the same repo and
is rejected by huggingface_hub v1).

**Provenance (recorded decisions, was P3 + F3 revision):** in
`compute_perplexity` (eval.py:895-899) extend `metric_config` with
`"dataset_id": WIKITEXT_DATASET_ID` and
`"dataset_revision": WIKITEXT_DATASET_REVISION` alongside
n_tokens/max_length/stride.
Schema-safe: no competence.jsonl rows exist anywhere yet, and `_competence_done`
compares the keys of the CURRENT config against the recorded row, so all
future rows are self-consistent. The revision IS pinned per the F3 decision
recorded in the Context log (critique-2 F5 removed a stale contrary sentence
here). Implementer check: the rung-2 perplexity test (test_bypass.py:149-185)
asserts `nll_mean` is top-level and absent from `config` — the added keys must
not disturb those assertions (they don't by design; verify on the run) — and
that test additionally asserts the appended row's `config["dataset_id"]` and
`config["dataset_revision"]` equal the constants (critique-2 F3: the fields
are added in `compute_perplexity`, so the monkeypatched slice does not
interfere).

### Step 2 — C5(a): guard hardening + docstrings
Replace the guard in `_attention_all_layers_unchecked` (interp.py:118-122):

```python
if not out.attentions or out.attentions[0] is None:
    raise RuntimeError(
        "no attention weights returned (attn_implementation=%r): this "
        "backend did not materialize attention probabilities under "
        "output_attentions=True (sdpa and flash backends do not). Load the "
        "model with eager attention via interp.load_eager_model_for_interp "
        "for attention reads."
        % getattr(getattr(model, "config", None), "_attn_implementation", None)
    )
```

The message names the actual resolved backend and attributes the cause
conditionally (critique-1 F7): a future eager or model-specific empty return
is not misdiagnosed as sdpa/flash. The literal "sdpa" and the helper name
survive for the guard test's substring assertions.

`not out.attentions` catches both `None` and the transformers-5 empty tuple.
Rewrite `attention_all_layers`'s docstring (interp.py:125-140): delete the false
"transformers normally falls back to eager… logging a warning" claim; state that
sdpa/flash return no attention maps and the sanctioned route is
`load_eager_model_for_interp`. Keep the bypass-guard paragraph.

### Step 3 — C5(b): eager pathway
**models.py:** extract the body of `load_model_and_tokenizer` (models.py:218-280)
into `_load(model_id, quant="4bit", adapter_path=None, attn_implementation=None)`.
`None` = exactly today's per-branch behavior (4-bit CUDA: no kwarg → sdpa default;
none+CUDA: no kwarg; none+CPU/MPS: forced eager as today). When given, pass to
`from_pretrained` in every branch. Public loader becomes a delegate with unchanged
signature/docstring (`_derive_gen_config` and run-identity see identical load
profiles). One comment at `_load`: "interp.load_eager_model_for_interp
delegates here; keep the signature in sync."

**Tokenizer binding (critique-3 F4):** inside `_load`, the MODEL loads first
and the tokenizer second, as
`AutoTokenizer.from_pretrained(model_id, revision=getattr(model.config,
"_commit_hash", None))` — the tokenizer comes from the same repository
snapshot the model resolved, so the recorded `model_revision` identifies the
model AND the tokenizer that defines every token count (WikiText perplexity is
defined in TOKENS; the spec allows only same-tokenizer deltas, and the
tokenizer object itself exposes no commit hash to record — verified by the
round-3 critic on the sanctioned stack). `revision=None` (local dirs, or a
stack that stops exposing `_commit_hash`) preserves today's behavior exactly.
This is load-internal CONSISTENCY, not a pin — whether to pin the loads to a
fixed revision remains pending P7. Screening: the order swap has no dependency
(the tokenizer is unused during model load); all quant/adapter branches share
the single tokenizer line; tiny-model tests pass revision=None as today.
Rung-2 call-capture test: stub `AutoModelForCausalLM.from_pretrained` to
return a model whose `config._commit_hash = "cafe0000..."`, record
`AutoTokenizer.from_pretrained` kwargs, call `_load(<id>, quant="none")`,
assert `revision="cafe0000..."` threaded through; restore both in `finally`.

**interp.py:** new public helper:

```python
def load_eager_model_for_interp(model_id, quant="4bit", adapter_path=None):
    """Load a model with EAGER attention, for interpretation reads ONLY.

    Never use this model for generation or eval rows: attn_implementation is
    part of gen_config identity (gen_config.load_profile.attn_implementation)
    and generation numerics differ across attention backends — all
    row-producing runs load via models.load_model_and_tokenizer. Eager
    attention materializes the attention probabilities that sdpa/flash do
    not, so attention_all_layers works on this model. Same load profile as
    the canonical loader (4-bit NF4 / none). adapter_path attaches a LoRA
    adapter ONLY: this helper does NOT reinstall a permanent lesion. Per the
    ratified rule the Stage-2 lesion is reinstalled at every load from
    checkpoint metadata — that machinery belongs to the training-track plan,
    and until it exists a lesioned checkpoint must not be read through this
    helper; the corroboration-driver plan defines the sanctioned consumption
    pattern. Plain load — freeing a previously loaded model is the caller's
    job. Returns (model, tokenizer), model in eval mode.
    """
    from algoverse.models import _load
    model, tokenizer = _load(model_id, quant=quant, adapter_path=adapter_path,
                             attn_implementation="eager")
    attn = getattr(model.config, "_attn_implementation", None)
    if attn != "eager":
        raise RuntimeError("eager attention did not take: model came up with "
                           "attn_implementation=%r" % attn)
    return model, tokenizer
```

The post-load self-check converts a silently ignored kwarg in a future
transformers into a loud failure (the session's version-drift posture). On
PEFT-wrapped models `model.config` resolves to the base config, so the check works
with `adapter_path` (mirrors eval.py:134's read).

**INTERFACES.md (authorized edit 2 of 3):** add an "Interp track owns" block after
the fine-tuning block (:53-60):

```
load_eager_model_for_interp(model_id, quant="4bit"|"none", adapter_path=None)  # interp.py, exists
```

plus: "Eager-attention load for interpretation reads (attention patterns) ONLY —
never for generation or eval rows: attn_implementation is part of gen_config
identity (identity per the human's recorded decision 2026-08-15 — Step 3b), so
all row-producing runs use load_model_and_tokenizer. Same quant profile as the
canonical loader; adapter_path attaches a LoRA adapter only — the helper does
NOT reinstall a permanent lesion, so lesioned checkpoints are not readable
through it until the training-track reinstall machinery exists. (Added on the
human's recorded decision 2026-08-15 — priorities.md §1, C5 pathway.)"

**models.py handle-release fix (critique-1 F4):** `_BypassHandle.remove()`
(models.py:86-93), after removing the hook and deleting the marker attribute,
additionally sets `self._hook_handle = None`, `self._layers = None`,
`self._marker = None` (marker identity check runs BEFORE the nulling; the
"safe to call more than once" contract is preserved because the second call
returns at the `self._removed` guard before touching the nulled fields).
Without this, a removed handle still owns the decoder stack: the critic
measured 37,376 of 45,600 tiny-model parameters retained after
`handle.remove(); del model; gc.collect()`, released only by `del handle` —
on the 7B this blocks the rung-3 eager reload from freeing the sdpa model.
Flag for the training-track plan: load loops must not retain removed handles.

### Step 3b — attn_implementation identity tightening (recorded decision)
Per the human's decision (critique-1 F1 escalation, 2026-08-15):
- (a) eval.py:429 — the guarded load_profile tuple
  `("device_type", "dtype", "four_bit")` gains `"attn_implementation"`, so a
  resume whose current load profile differs in backend from the recorded rows
  refuses loudly. For this field ONLY (`device_type`/`dtype`/`four_bit`
  semantics unchanged) the comparison is equal-AND-not-None: a missing or null
  backend on EITHER side refuses — unknown provenance never silently matches
  (critique-2 F8; a plain `.get()` comparison would accept a legacy
  field-less row against a None-resolving current profile).
- (b) metrics.py — generation identity becomes SINGLE-SOURCED (critique-3 F2
  supersedes the round-2 per-field approach): `_normalized_scoring_config`
  MOVES from eval.py:175 to metrics.py as public `normalized_scoring_config`
  (eval.py imports it; eval's metrics imports are function-local, so no import
  cycle), and metrics.py gains public `gen_identity(row)` returning the FULL
  resume-guarded generation-identity tuple: bypass_impl, quant, do_sample,
  max_new_tokens, model_revision, adapter_digest, system_fold,
  use_llm_fallback, llm_provider, llm_model (the last three via
  `normalized_scoring_config` — the identical normalization the resume guard
  uses, so grouping never refuses what resume accepts), plus
  load_profile.dtype, .device_type, .four_bit, .attn_implementation.
  `GEN_CONFIG_KEY_FIELDS` (metrics.py:536-543) is extended to the matching
  name tuple, same order; `_run_key` becomes RUN_KEY_FIELDS values +
  `gen_identity(row)`; `summarize_runs`'s zip at metrics.py:580 keeps working
  because names and values extend together. Deliberately EXCLUDED (documented
  in the code): `*_version` fields (ratified audit-only) and `batch_size`
  (ratified operational).
- (c) Tests. The resume-REFUSAL case is RUNG 2, in test_bypass.py beside the
  existing resume-identity coverage of the inline guard (test_bypass.py:485-620)
  — the guard lives inside `run_negotiation_eval`, whose import chain pulls
  the ML stack, so test_eval_pure cannot reach it and no refactor to extract
  it is planned (critique-2 F6). Cases: recorded
  `load_profile.attn_implementation: "sdpa"` vs current "eager" → refuse;
  missing-vs-null → refuse. The `_run_key` GROUPING case is rung 1 in
  test_metrics.py (metrics imports stdlib only): two otherwise identical rows
  differing only in that field land in different groups. Update any existing
  fixtures that now need the field.
- (d) figures.py — the duplicated `_gen_identity` body (figures.py:74-85) is
  DELETED: it delegates to (or is replaced at its call sites by)
  `metrics.gen_identity`, and `_mismatch_fields` already takes its names from
  `metrics.GEN_CONFIG_KEY_FIELDS`. With one definition there is nothing left
  to keep in lockstep — this supersedes the round-2 lockstep rule, which
  managed the duplication instead of removing it (critique-3 F2: even the
  round-2 fix left the pairing tuple blind to model_revision, adapter_digest,
  system_fold, four_bit, and scoring identity — the critic paired
  same-backend/different-revision rows into a real-looking A_l).
  Tests (tests/test_figures.py): (i) the round-2 case — sdpa baseline vs eager
  sweep rows must NOT pair and `baseline_mismatch` names
  `attn_implementation`; (ii) the round-3 case — same backend but differing
  model_revision / adapter_digest / system_fold must NOT pair and
  `baseline_mismatch` names the differing fields. test_metrics: `_run_key`
  splits rows differing in any new field. Fixtures lacking the new fields on
  BOTH sides still group (None==None), so existing fixtures keep passing
  unless they assert summary key-sets — the implementer updates those.
Screening: with the (a) rule, rows lacking the field genuinely refuse resume
(the blanket round-1 claim was false for a None-resolving current profile —
corrected per critique-2 F8); no durable result rows exist anywhere (smoke
dirs are disposable per established precedent), so nothing is stranded.
Documented residual: `_run_key`/`_gen_identity` grouping of two both-None rows
still merges them — grouping has no refusal channel; the loud resume guard
covers the row-writing path, and the verified research stacks resolve concrete
backend strings.

### Step 3c — competence rows join the identity/provenance decisions
Completes the two recorded decisions across the capability writers (critique-2
F2/F4, critique-3 F5/F7): in `run_lm_eval_benchmarks` (metric_config at
eval.py:801-806) and `compute_perplexity` (metric_config at eval.py:895-899),
add THREE fields, derived-not-asserted:
`"attn_implementation": getattr(model.config, "_attn_implementation", None)`,
`"model_revision": getattr(model.config, "_commit_hash", None)`, and
`"adapter_digest": _adapter_digest((run_meta or {}).get("adapter_path"))`
(existing helper, eval.py:66-96; None without an adapter — critique-3 F7:
without the digest, the same path holding different LoRA weights shares
capability identity). Negotiation rows already record and guard all three via
gen_config.
Consequences through existing machinery: `_competence_done` compares config
keys → a mixed-backend/revision/adapter competence RESUME refuses;
`gate1_report`'s config-comparability check (strips only `batch_size`) →
mixed competence deltas refuse loudly. With the Step-3 tokenizer binding
(tokenizer loaded at the model's resolved revision), matching
`model_revision` restores the spec's same-model/same-tokenizer premise AS A
GUARD — drift refuses loudly; actually pinning the loads remains pending P7.
Tests: rung 1 (test_metrics.py) — two same-metric competence rows differing
only in `config.attn_implementation` (and separately `config.model_revision`,
`config.adapter_digest`) produce the publishability config-mismatch error;
rung 2 — the perplexity-row test asserts all three fields in the appended
row's config, AND a writer-path test proves `run_lm_eval_benchmarks` records
them too (critique-3 F5 — the perplexity assertion alone lets an
implementation skip the benchmark writer): inject stub `lm_eval` and
`lm_eval.models.huggingface` modules into `sys.modules` (canned
`simple_evaluate` results; HFLM stub; restore in `finally`), tiny model, tmp
out_path, run_meta with adapter_path=None; assert both appended rows' configs
carry the three fields plus limit/batch_size/seed/lm_eval.
Screening: zero competence rows exist anywhere, so the schema addition strands
nothing. On the tiny-model rung-2 path `_attn_implementation` resolves
"eager" and `_commit_hash` is absent → None recorded as-is: within one local
run None==None resumes correctly; PUBLISHABLE certification of null
provenance is refused by Step 3d, and local/dev flows use --dev (stamped NOT
PUBLISHABLE). sys.modules injection is the repo's established mock pattern.

### Step 3d — Gate-1 input integrity (binding, uniqueness, non-null provenance)
Critique-3 F1/F6/F9: comparing the two competence configs to EACH OTHER never
proved they belong to the negotiation runs being certified — the critic
produced an executable `DECISION: PASS` from unrelated competence files.
`gate1_report` (helpers in eval.py; scripts/gate1_report.py CLI unchanged)
gains, on the non-dev path, competence validation producing publishability
errors:
1. BINDING: for each of reference/"M_D" that has competence rows, every
   competence row's `run_id` must equal the corresponding negotiation file's
   single validated run_id, and the shared run_meta fields (model_id,
   adapter_path, bypassed_layer, checkpoint_step, arm, train_seed) must equal
   the negotiation rows' top-level values — a stale or swapped competence
   file can no longer certify.
2. UNIQUENESS: exactly one competence row per required metric within that
   run_id — duplicates and multi-run files refuse instead of the current
   silent last-row-wins overwrite in the `bench` dict (eval.py:1085-1090).
3. NON-NULL PROVENANCE: publishable requires non-null
   `config.attn_implementation` AND `config.model_revision` in every required
   competence row — null/unknown provenance cannot certify (critique-3 F9).
`--dev` skips all three (already stamped DEV — NOT PUBLISHABLE), so tiny/local
models keep working.
Tests (rung 1, test_metrics.py; the critique's executable constructions become
the fixtures): unrelated run/model ids → binding errors (the exact round-3
construction that printed PASS must now refuse); mixed-run and duplicate-row
files → uniqueness errors; null backend/revision → provenance errors; one
fully matched set → no errors and the decision computes as before.
Screening: the non-dev path only gains REFUSALS — no numeric computation
changes; dev ergonomics preserved; negotiation-side validation is untouched.

### Step 3e — figures competence path refuses mixed provenance
Critique-3 F8: `figures.index_competence` (figures.py:345) reduces capability
rows to value/stderr, so the Pareto damage axis subtracts across any
provenance. Fix: `index_competence` keeps each row's `config` and refuses
duplicate rows for one metric; `pareto_points` compares the baseline and
per-layer competence configs under the batch_size-stripped comparable-config
rule — REUSE metrics' existing helper (metrics.py:415-418), single home — and
raises `ValueError` naming the differing fields on mismatch (loud, like the
negotiation-side refusal; a paper-facing axis must not silently mix).
Test (test_figures.py): the critique's construction — an sdpa/old-revision
baseline MMLU row against an eager/new-revision layer row — must raise naming
the fields; matched configs must produce the point as before.
Screening: fixtures without configs on both sides compare equal-empty and keep
passing; the change only ADDS refusals.

### Step 4 — A4: canary re-pin
tests/test_bypass.py:412-428: rename
`test_output_hidden_states_is_stale_under_bypass_canary` →
`test_output_hidden_states_is_bypass_aware_canary` (a rename, not an addition —
the rename itself leaves BYPASS_TEST_COUNT unchanged; the separately added
tests adjust it). Same fixture; keep `assert torch.equal(res[2], res[1])`; flip the second
assertion to `assert torch.equal(hs[2], hs[1])` with message "canary:
output_hidden_states no longer bypass-aware — behavior regressed to stale; the
guards and residual_stream_by_layer remain the sanctioned path either way".
Rewrite the comment: the canary pins the CURRENT transformers behavior (5.x
bypass-aware) and flips again on regression; the guards stay regardless because
installed versions verifiably drift (record B1: Colab 5.13.1 vs rung-2 5.15.0) —
human decision 2026-08-15. Together with the new handle-release test (Step 3,
F4 fix; asserts a removed handle's `_hook_handle`/`_layers`/`_marker` are None
and a weakref'd tiny model becomes collectable after `del model` + `gc.collect()`
while the handle is still in scope) and the Step-3b resume-refusal test, rung-2
test_bypass passes in full (BYPASS_TEST_COUNT kept exact per added test
function) → `ALL TESTS PASSED`, which is finding A3's acceptance.

### Step 5 — warning reword, all homes consistently
The version-specific stale-activations claim is reworded to "version-dependent;
sanctioned path is version-robust" in ALL its homes, same normative instruction
(do NOT use output_hidden_states/output_attentions on a bypassed model; use
`residual_stream_by_layer`).

**Two invariants stay distinct in every reword (critique-1 F7):** (1) whether
`output_hidden_states` capture reflects the bypass hook is VERSION-DEPENDENT —
that is the canary's subject; (2) a bypassed block still EXECUTES its attention,
so any returned attention map is real but causally disconnected on EVERY
transformers version — the ratified reason the bypassed layer is NaN'd/excluded,
unchanged by bypass-aware hidden states. The reword must not fold (2) into (1).
`attention_all_layers`'s bypass-guard paragraph, which already states (2), is
kept verbatim. Add to the INTERFACES warning after the version-dependence text:
"Attention maps are a separate invariant: a bypassed block still computes
attention, so its maps are real but causally dead on every transformers
version — the ratified NaN/exclusion rule for the bypassed layer is
version-independent."

Homes:
- **(a) INTERFACES.md:70-79** (authorized edit 1 of 3): "…What they return for a
  bypassed layer is VERSION-DEPENDENT: some transformers versions record each
  block's raw output before the bypass hook replaces it (stale), others record the
  bypass-aware value — and installed versions verifiably drift across the team's
  environments. Use `residual_stream_by_layer(model, input_ids)`: it captures the
  true per-layer residual via pre-hooks and is correct under both behaviors. The
  interp readers raise on a bypassed model… (Reworded on the human's recorded
  decision 2026-08-15 — A4 canary disposition.)"
- **(b) interp.py module docstring** (:9-17) — same framing; rendering contract
  untouched.
- **(c) models.py `residual_stream_by_layer` docstring** (:167-175) — "bypass
  behavior of output_hidden_states is version-dependent… this captures each
  block's INPUT via pre-hooks, correct on every version". Shape paragraph
  unchanged.
- **(d) interp.py `_refuse_if_bypassed`** — reword the docstring (:38-40) AND the
  message (:43-49). Message draft (preserves tested substrings "bypass",
  "layer %s"→"layer 1", "residual_stream_by_layer"):

```python
"cannot read activations via output_hidden_states/attentions on a model "
"with a bypass installed (layer %s, %s): whether output_hidden_states "
"reflects the bypass is version-dependent, and the bypassed block's "
"attention maps are real but causally dead on every version. Use "
"models.residual_stream_by_layer for the true residual stream."
```

(critique-2 F10a: the message itself keeps the two invariants separate rather
than folding attentions into the version-dependent clause; the tested
substrings "bypass", "layer 1", "residual_stream_by_layer" all survive.)

### Step 6 — `--competence` flag (recorded decision)
scripts/run_baseline.py: add `--competence` (`action="store_true"`) documented as
"explicitly request the benchmark competence checks (the default; required wording
for publishable Gate-1 commands — ratified item 14)". If both `--competence` and
`--skip-benchmarks` are passed → `parser.error`. No other behavior change:
benchmarks still run by default. Verification: rung-2
`~/.venvs/colab-local/bin/python scripts/run_baseline.py --help` shows the flag;
the conflicting pair exits with the argparse error (no model load happens during
parse).

**Runner docstring (critique-1 F5):** update the module docstring
(run_baseline.py:1-12) — it is fed to argparse as the `--help` text — so the
canonical Gate-1 command matches priorities.md:106-110 exactly
(`--split selection --n 305 … --llm-fallback --competence`), keeping the resume
and `--skip-benchmarks` notes. The stale `--n 100`-as-canonical label is what
could strand the canonical results/m0-baseline directory under the manifest
guard: a 100-scenario run into that directory cannot later resume into the
required 305-scenario cohort.

**Canonical-command alignment (authorized edit 3 of 3, recorded decision, was
P1-residual):** update INTERFACES.md's canonical baseline command (:116-119) to
include `--competence`, matching priorities.md:106-110 and RESEARCH_SPEC item 14.
Annotate with "(Added on the human's recorded decision 2026-08-15 — planning
kickoff Q&A, gpu-verification-fixes plan.)" No other change to the
canonical-commands block.

### Step 7 — rung-2 acceptance sweep
- Rung 1 (`python3`): all six stdlib suites — now including the Step-3b
  `_run_key` grouping case, the Step-3c gate1 config-mismatch cases, the
  Step-3d binding/uniqueness/non-null cases, and the pin-ratification literal
  in test_metrics/test_eval_pure; tripwire intact.
- Rung 2 (`~/.venvs/colab-local/bin/python`): test_bypass.py in full
  (`ALL TESTS PASSED`, A3 acceptance; includes the handle-release, backend
  resume-refusal, `_load` tokenizer-revision call-capture, and lm-eval
  writer-path tests); test_interp.py 7/7 (runner now skip-aware);
  test_wikitext_loader.py 3/3 with datasets installed (incl. the
  pinned-revision call-capture); test_figures.py — incl. both pairing-refusal
  cases and the competence config-mismatch refusal — via pytest with
  PYTHONPATH=src;
  refactor-equivalence: `scripts/smoke_test.py --out-dir <scratchpad>/smoke`
  (quant="none" through the real `_load` delegate on the 0.5B — the invocation
  that passed as A1).
The summary states which rung executed each check.

### Step 8 — rung-3 acceptance (implementer-run debug; §5)

## 4. Acceptance-test designs

**tests/test_wikitext_loader.py (rung 2, new guarded standing suite).** Standing
(not one-off) because the item-16 neutral-JSD bound reuses this loader.
Conventions per test_interp.py:10-29: `WIKITEXT_TEST_COUNT = 3`,
`HAVE_WIKITEXT_STACK` guard (`datasets`, `torch`, `transformers`), `__main__`
runner, loud SKIP banner when the stack is absent. Network posture stated in the
module docstring: unlike the tiny-model suites this downloads on first run
(~4.4MB wikitext-2 test split + the Qwen tokenizer, both cached in
`~/.cache/huggingface`); `datasets` absent → loud SKIP; network failure with
datasets installed → real FAILURE (fix the environment). Tokenizer:
`AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")` (production tokenizer,
A6-verified loadable at rung 2). Tests:
1. `test_slice_shape_and_determinism` — shape `(1, 20000)`; second call
   `torch.equal` to the first; `n_tokens=64` returns `(1, 64)` equal to
   `ids[:, :64]` (prefix property).
2. `test_slice_text_semantics` — decode `ids[0, :64]`, assert it contains the
   first heading of the WikiText-2 test split (expected "Robert Boulter"; the
   implementer verifies the actual first non-blank line against the downloaded
   split rather than trusting this plan). Text-level assertions only — never
   token-id checksums, which would pin tokenizer versions.
3. `test_loader_requests_pinned_revision` (WIKITEXT_TEST_COUNT = 3) —
   monkeypatch `datasets.load_dataset` with a recording stub returning canned
   rows (`{"text": [...]}`), no network, restore in `finally`; call
   `load_wikitext_slice`; assert the recorded call passed
   `WIKITEXT_DATASET_ID`, config `"wikitext-2-raw-v1"`, `split="test"`, AND
   `revision=WIKITEXT_DATASET_REVISION`. Rationale (critique-2 F3): current
   `main` resolves to the pinned hash, so the behavioral tests above cannot
   distinguish a missing `revision=` today — only a call-capture test proves
   the pin is actually requested.

**Pin-ratification test (rung 1, in test_eval_pure.py or test_metrics.py —
algoverse.eval is stdlib-importable).** The call-capture test compares against
the implementation constant, so `WIKITEXT_DATASET_REVISION = "main"` would
pass every check above while restoring the moving-data defect (critique-3 F3).
This test pins the decided VALUE independently: assert
`WIKITEXT_DATASET_ID == "Salesforce/wikitext"`,
`WIKITEXT_DATASET_REVISION == "b08601e04326c79dfdd32d625aee71d232d685c3"`
(the ratified literal restated in the test — the test IS the ratification
record), and `re.fullmatch(r"[0-9a-f]{40}", WIKITEXT_DATASET_REVISION)` (an
immutable commit identifier, never a branch name).

**sdpa guard test (rung 2, test_interp.py).**
`test_attention_read_under_sdpa_raises_clean_diagnostic`: tiny-model fixture
(test_interp.py:33-50) with `config._attn_implementation = "sdpa"`; call
`attention_all_layers` on the un-bypassed model with `StubTokenizer()`. Assert
`RuntimeError` (not `IndexError`); message contains `"'sdpa'"` — the
`%r`-quoted INTERPOLATED backend, which the static "(sdpa and flash backends
do not)" parenthetical cannot satisfy, so a missing or wrong interpolation
fails (critique-2 F10b) — and `"load_eager_model_for_interp"`. Premise
pre-verified on this exact venv. Documented fallback if a future transformers
changes CPU behavior: a stub whose forward returns `attentions=()`.

**eager helper test (rung 2, test_interp.py).**
`test_eager_interp_loader_cpu_end_to_end`, no network: `save_pretrained` a tiny
random Qwen2ForCausalLM into a tempdir; monkeypatch
`transformers.AutoTokenizer.from_pretrained` → `StubTokenizer()` (restore in
`finally`); `load_eager_model_for_interp(tmpdir, quant="none")`. Assert
`(model, tokenizer)` returned; `model.config._attn_implementation == "eager"`;
`not model.training`; `attention_all_layers(...)` returns a finite
`[4, 4, seq, seq]` array. Honest limitation (comment in test): the CPU
quant="none" branch already forces eager, so this verifies plumbing/end-to-end,
not the CUDA-branch threading — that is rung-3's job.

**eager helper adapter test (rung 2, in tests/test_interp.py, PEFT-guarded).**
`test_eager_interp_loader_applies_adapter`, following the loud-skip PEFT
convention of test_bypass.py:60-64 (peft 0.20.0 is in the venv; a peft-less
environment skips LOUDLY, never passes vacuously — see the runner change
below): build a tiny LoRA adapter on the tiny random Qwen2 and, BEFORE saving,
fill every `lora_B` tensor with nonzero values — fresh `lora_B` is
zero-initialized, so an unmodified adapter is a numerical no-op and "loaded
and applied" is indistinguishable from "silently dropped" (critique-2 F7; C6
precedent, which gated on `adapter_effect=True`). `save_pretrained` to a
tempdir, load through `load_eager_model_for_interp(<base_tmpdir>,
quant="none", adapter_path=<adapter_tmpdir>)` (tokenizer monkeypatch as in
the CPU end-to-end test). Assert: a PeftModel is returned; the adapter has a
numerical EFFECT (logits on the fixed input differ from the bare base
model's); the eager self-check passes through the PEFT wrapper
(`model.config` resolves to the base config); and `attention_all_layers`
succeeds. This exercises the adapter branch, its application, and the
self-check under PEFT, which the intact-only tests missed (critique-1 F2).

**test_interp runner change (critique-2 F9).** The loud-skip promise requires
a specified change to test_interp.py's `__main__` runner: it currently
catches every `Exception` as a FAILURE (test_interp.py:196-206) and
`unittest.SkipTest` IS an Exception. Mirror test_bypass.py:803-819: catch
`unittest.SkipTest` separately, count skips, and print the
partial-verification banner (`ALL EXECUTED TESTS PASSED; N SKIPPED — FULL
VERIFICATION NOT COMPLETE`) when skips > 0 and failures == 0. In the
sanctioned rung-2 venv peft is present, so the acceptance expectation stays
7/7 `ALL TESTS PASSED`.

**handle-release test (rung 2, in tests/test_bypass.py).** After
`install_bypass` + `remove()`: the handle's `_hook_handle`/`_layers`/`_marker`
are None, `remove()` is still safely re-callable, and a weakref'd tiny model
is collectable after `del model` + `gc.collect()` with the handle still in
scope (critique-1 F4's measured hazard). BYPASS_TEST_COUNT updated per added
test function (implementer keeps the constant exact).

**A4 canary:** flipped assertions pass on 5.15.0 within the full test_bypass
run.

## 5. Rung-3 acceptance: one consolidated debug script

Scratch script, uncommitted, self-contained (embedded base64 tar of post-fix
`src/algoverse`, tracked files only). Invocation, exactly:

```
colab --auth=adc run --gpu T4 --timeout 2700 <script.py>
```

`colab --auth=adc sessions` empty BEFORE and AFTER; both outputs recorded
verbatim. Script order:
1. The 4-line WRONG-GPU abort preamble before any other import (AGENTS.md).
2. `pip install -q bitsandbytes peft datasets scikit-learn` (tolerate
   pre-installed).
3. Untar `src/algoverse`, prepend to `sys.path`.
4. **F1 — C4 re-execution** (work unit gpu-verification.md:310-346; debug bounds
   ratified §10.5, barred from spec/paper). Canonical
   `load_model_and_tokenizer(PROD_MODEL, quant="4bit")` (exercises the refactored
   delegate). For intact / bypass layer 0 / bypass layer 14 (remove each handle
   after its measurement): `compute_perplexity(model, tokenizer,
   out_path=<VM-temp>/c4-<cond>.jsonl, run_meta={"run_id": "c4-<cond>"})`,
   defaults only. Each measurement ends `handle.remove(); del handle` — a
   removed handle otherwise retains the decoder stack (critique-1 F4). Print
   all six values (ppl + row `nll_mean` per condition).
   Acceptance exactly as ratified: three finite `nll_mean`; ppl(intact) < 50;
   ppl(layer0) ≥ 10× ppl(intact) AND ≥ 5× ppl(layer14); strict ordering
   intact < layer14 < layer0; nll_mean(layer14) < 20; RED FLAG
   ppl(layer0) < 2× ppl(intact) → FAIL. Anything between bands → print all six
   and ESCALATE; the implementer never adjudicates. Exit condition: last handle
   removed and deleted, `bypass_state(model) is None`.
5. **F2 — C5 re-execution, two parts** (work unit gpu-verification.md:348-368).
   Render the pinned input:
   `render_condition_texts([scenario], "incentive", tokenizer)[0]` from
   `get_scenarios("selection", n=1)` (split is required-positional,
   tasks.py:241 — critique-1 F6; selection is the non-final pool and the
   deterministic seed-42 sorted-pool draw reproduces the prior run's
   scenario). Print token length: expected 184 (records A6/C5); this oracle is
   ADVISORY — a mismatch prints the value and marks the check AMBIGUOUS →
   escalate, never a silent pass.
   - (i) hardened guard on the sdpa canonical model (F1's model, intact):
     `attention_all_layers` → PASS iff `RuntimeError` whose message contains
     the `%r`-quoted resolved backend (`"'sdpa'"` — the interpolated value,
     not the static parenthetical; critique-2 F10b) and
     `"load_eager_model_for_interp"`; bare `IndexError` = FAIL (guard
     regression); attentions actually returned = ESCALATE (Colab transformers
     drift — a finding, not a pass/fail). Print
     `_derive_gen_config(model, quant_label="4bit")`'s
     `load_profile.attn_implementation` (expect `'sdpa'`) AND its
     `model_revision`: a None revision on the production Qwen 4-bit load is
     ESCALATE (critique-3 F9 — the loader/stack stopped exposing
     `_commit_hash`, and the new non-null publishability guard would then
     block priority 2; the tiny-model None case is a rung-2/dev matter only).
   - (ii) eager pathway: `del model; gc.collect(); torch.cuda.empty_cache()`
     (C6 precedent; freeing is the caller's job, and all F1 handles must
     already be deleted — critique-1 F4);
     `load_eager_model_for_interp(PROD_MODEL, quant="4bit")` (warm HF cache);
     assert `_attn_implementation == "eager"`; `attention_all_layers` → PASS
     iff finite, non-NaN, shape `[28, n_heads, 184, 184]` (28 layers per
     record C1; n_heads from config).
6. Every check prints PASS / FAIL reason / ESCALATE detail; nonzero exit on FAIL.
   VM-temp output only; nothing under results/; no Drive; keep-worthy output
   printed to stdout.

**Assigned to the human (why rungs 2 and 3 both fail for it):** nothing in this
session — both failed work units are pass/fail diagnostics against ratified debug
bounds, so rung 3 covers them. The actual Gate-1 baseline (priority 2) and all
paper numbers remain human-run by division of labor.

## 6. Pending decisions (flagged, not resolved)

Resolved since first draft (moved to the Context decision log, implemented by
Steps 1 and 6): P1-residual (canonical-command alignment — decided YES) and P3
(`dataset_id` in `metric_config` — decided YES; the same-day F3 decision then
ALSO pinned the revision — see the Context log; nothing about the pin is
pending).

Also resolved during critique-1 adjudication (decision log): the WikiText
revision pin (F3) and attn_implementation identity (F1).

- **P2 — datasets version pinning.** Installs unpinned per the
  everything-unpinned-until-paper-numbers policy; joins the eventual
  stack-pinning decision. (Package versions only — the WikiText data revision
  is now pinned by recorded decision, F3.)
- **P6 — eager-helper × permanent-lesion integration.** How lesioned-checkpoint
  reads reinstall the recorded bypassed layer before using
  `load_eager_model_for_interp` (and how the NaN mask derives from the
  reinstalled `bypass_state`) is decided in the training-track and
  corroboration-driver plans, not here; until then lesioned checkpoints are
  not readable through the helper (critique-1 F2).
- **P7 — model/tokenizer repository revision pinning.** Competence and
  negotiation rows now RECORD and GUARD the resolved model revision (Step 3c;
  negotiation rows already did), so cross-revision comparisons refuse loudly —
  but the loads themselves still resolve mutable `main`, and a tokenizer
  change can move the token-20,000 WikiText boundary even with the dataset
  pinned (critique-2 F4). Whether to pin the model repo revision at load is a
  policy decision the human makes — recommended at the P2 stack-pinning
  moment.
- **P4 — interp.jsonl identity for attention rows.** Whether `attention_jsd`
  rows' config must record `attn_implementation` (produced under eager while eval
  rows record sdpa) — a corroboration-driver (priority 5) schema question;
  flagged because this session creates the asymmetry.
- **P5 — Colab-version canary coverage (informational).** The re-pinned canary
  runs only at rung-2's 5.15.0; the Colab stack recorded 5.13.1 (B1). Mitigated
  by design (guards + `residual_stream_by_layer` correct under both behaviors);
  rung-3 F2(i) empirically probes the Colab stack. No decision needed unless
  F2(i) ESCALATEs.

## 7. Definition of done

1. All Step-1..6 edits landed (including Steps 3b/3c/3d/3e, the identity
   single-sourcing, the tokenizer binding, and the handle-release fix);
   nothing outside the §1 module map; every consumer in the §1b closure table
   has its covering change; the INTERFACES.md diff touches only the three
   authorized places (the capability sentence and warning text amended per
   critique-1 F1/F2/F7 stay within already-authorized edits 1 and 2 — no new
   contract surface).
2. Rung 1: six stdlib suites pass on `python3`, including the `_run_key`
   grouping, gate1 config-mismatch, gate1 binding/uniqueness/non-null, and
   pin-ratification cases. Rung 2: test_bypass in full (`ALL TESTS PASSED`,
   A3 acceptance; BYPASS_TEST_COUNT exact; incl. tokenizer-revision
   call-capture and lm-eval writer-path), test_interp 7/7 with the skip-aware
   runner, test_wikitext_loader 3/3 (incl. the pinned-revision call-capture),
   test_figures incl. both pairing-refusal cases and the competence
   config-mismatch refusal, smoke_test passes through the refactored delegate
   (now exercising the tokenizer-binding line), run_baseline `--help`/conflict
   checks pass and `--help` shows the corrected canonical command.
3. Rung 3: pinned invocation executed; sessions verified empty before/after
   (verbatim); F1 six values within ratified bounds (or ESCALATED); F2(i)
   diagnostic contains the quoted resolved backend and the helper name; F2(ii)
   eager attention read succeeds after the handle-deleted/gc'd reload.
4. Implementer summary reports verified-vs-written per AGENTS.md, names the rung
   that executed each check, and lists the remaining pending decisions
   (P2, P4, P5, P6, P7).
