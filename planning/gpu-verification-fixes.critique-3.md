# gpu-verification-fixes.critique-3 — Plan critique (round 3)

Scope: revision 3 of `planning/gpu-verification-fixes.md`, reviewed against
the round-2 dispositions, `planning/priorities.md` priority 1,
`RESEARCH_SPEC.md`, `INTERFACES.md`, and the current source and acceptance
suites. This is findings only: no revised plan or implementation is proposed.
No paper-number experiment, benchmark, training run, or GPU job was launched.
No existing repository file was edited; this critique is the only file added.
Format per finding: **location — claim.** Failure scenario. *Confidence /
severity.*

---

## Round-2 disposition audit

| Round-2 finding | Revision-3 status |
|---|---|
| F1 — mixed-backend figures | **Resolved narrowly.** The backend joins `_gen_identity` and has a targeted refusal test. Other already-guarded generation identities remain absent from the same paper-facing pairing tuple (new F2). |
| F2 — backend-blind capability/Gate 1 | **Incomplete.** Capability configs gain backend/revision fields, but Gate 1 still does not bind those rows to the negotiation run being certified (new F1), accepts duplicate/mixed capability runs (new F6), and lacks writer-path coverage for MMLU/GSM8K (new F5). |
| F3 — WikiText pin/provenance acceptance | **Incomplete.** Call capture and row-field assertions are added, but both compare against the implementation constant; setting that constant to mutable `main` passes all proposed checks (new F3). |
| F4 — mutable model/tokenizer | **Partially addressed.** Base-model revision is recorded for capability rows, while the tokenizer is still neither pinned nor identified and the figure path still ignores model revision (new F2/F4). P7 acknowledges that loading remains mutable. |
| F5 — contradictory pin wording | **Resolved in Step 1.** A stale historical “no revision pin” phrase remains in the pending-decisions preamble (new F10, low severity). |
| F6 — wrong test rung | **Resolved.** The negotiation resume-refusal test is now assigned to the ML-gated suite at rung 2. |
| F7 — zero-effect adapter test | **Resolved in design.** Nonzero `lora_B` weights and a numerical-effect assertion are explicit. |
| F8 — missing/null backend resume | **Resolved for negotiation rows.** The plan specifies equal-and-non-null comparison and covers missing-vs-null. Capability identity has a separate unknown-revision gap (new F9). |
| F9 — skip-blind interp runner | **Resolved in design.** The runner change is explicitly in scope and acceptance. |
| F10 — conflated warning / weak diagnostic assertion | **Resolved in design.** The message separates the invariants and the test requires the `%r`-quoted interpolated backend. |

---

## High severity

### F1. Gate 1 still does not prove that competence rows belong to the negotiation runs it certifies

**Plan lines 299-320 and 625-643; `scripts/gate1_report.py:22-50`;
`src/algoverse/eval.py:970-1038,1041-1097`; `tests/test_metrics.py:292-347,
428-454`.** Step 3c compares capability configs between the files labelled
M_0 and M_D, but leaves `gate1_report` unchanged. The negotiation side validates
one `run_id` within each rows file. The competence side then discards every
top-level field — including `run_id`, `model_id`, `adapter_path`, arm, and
bypass identity — and retains only value, stderr, and config. No check joins a
competence file to the corresponding negotiation file. The CLI likewise accepts
independent `NAME=PATH` maps whose common label is the only association.

Executable read-only evidence used full 305-scenario synthetic negotiation
pools whose M_0 rows said `rev-old`/sdpa and M_D rows said `rev-new`/eager. The
two capability files used unrelated run IDs and model IDs but mutually equal
planned backend/revision configs. `gate1_report(..., n_boot=40)` printed
`DECISION: PASS`. At the helper level the same construction returned
`pool_errors=[]` and `benchmark_errors=[]`.

Failure scenario: a human swaps one `--competence` path, points at a stale
capability file, or regenerates negotiation rows while retaining a capability
artifact from another run. The backend/revision fields agree with each other
across the two capability files, so the new comparison passes, but the
capability deltas do not describe the models whose tau gain is being certified.
This can silently turn an invalid Gate-1 input set into a publishable PASS.
*Confidence: high. Severity: high.*

### F2. The figure pairing tuple remains blind to most of the generation identity that resume already guards

**Plan lines 254-297, 299-320, and 607-614;
`src/algoverse/eval.py:405-440`; `src/algoverse/metrics.py:519-580`;
`src/algoverse/figures.py:57-96,203-287`.** Step 3b adds only
`attn_implementation` to the three lockstep tuple definitions. Even after that
addition, `_gen_identity` omits `model_revision`, `adapter_digest`,
`system_fold`, `load_profile.four_bit`, and the normalized LLM scoring identity
(`use_llm_fallback`, provider, model). All of those are already treated as
resume identity by `run_negotiation_eval`, which establishes that the project
does not regard them as interchangeable generation provenance.

The omission matters more in figures than in summaries: `_match_key` deliberately
removes `run_id` so that separately executed baseline and sweep runs can pair.
An executable read-only diagnostic gave otherwise identical base/sweep rows the
same sdpa backend but different model revisions, adapter digests, and
`system_fold` values. `figures.layer_curve` still returned
`A_l=0.5000000000000001` and `baseline_mismatch=None`. The planned backend test
passes on this broken state because both rows use the same backend.

Failure scenario: mutable `main` advances between the intact M_D baseline and
the layer sweep, an adapter directory is overwritten in place, or prompt-system
folding changes. The paper-facing curve compares different model weights or
different prompts and reports a real-looking localization effect. Step 3c's
Gate-1 model-revision config does not flow into this path. *Confidence: high.
Severity: high.*

### F3. The pinned-revision acceptance test also passes when the “pin” is mutable `main`

**Plan lines 43-53, 117-152, 435-462, and 625-639.** The decided literal is a
40-character commit, but `test_loader_requests_pinned_revision` asserts only
that the loader passed `revision=WIKITEXT_DATASET_REVISION`. The perplexity-row
test similarly compares the recorded value to that same implementation
constant. Neither test asserts that the constant equals the ratified commit or
is an immutable commit identifier.

Failure scenario: the implementation defines
`WIKITEXT_DATASET_REVISION = "main"`, passes that value to `load_dataset`, and
records `"main"` in every row. Call capture passes, provenance-field assertions
pass, the real-fetch tests pass, and rung 3 produces values. Dataset bytes can
still change between M_0 and M_D while their configs remain identically
`"main"`, recreating the exact silent-comparability defect that motivated the
pin. *Confidence: high. Severity: high.*

### F4. Recording the model commit does not identify the tokenizer that defines the scored sequence

**Plan lines 299-320 and 607-614; `src/algoverse/models.py:218-277`;
`src/algoverse/eval.py:99-147,856-914`; `RESEARCH_SPEC.md:220-226,290-293`.**
The loader resolves the tokenizer and model in two independent
`from_pretrained(model_id)` calls against mutable `main`, tokenizer first. Step
3c records only `model.config._commit_hash`. No tokenizer revision is passed,
derived, or recorded, even though WikiText perplexity is explicitly the first
20,000 *tokens* and the spec permits only same-tokenizer deltas.

A read-only production-metadata check on the sanctioned rung-2 stack returned
`config._commit_hash='a09a35458c702b33eeacc393d103063234e8bc28'`, while the
loaded Qwen tokenizer exposed neither `_commit_hash` nor
`init_kwargs['_commit_hash']` (both `None`). The official Qwen history already
checked in round 2 shows tokenizer-file changes:
<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/commits/main/tokenizer_config.json>.
Thus the proposed row field proves the model config revision, not the tokenizer
revision used to create token IDs.

Failure scenario: the repository advances between the two independent loads,
or tokenizer files/cache resolve differently while the model config hash is
the same recorded value. Gate 1 sees matching `model_revision` and fixed dataset
bytes but compares perplexities over different token sequences. P7 correctly
admits that loading remains mutable, but Step 3c's claim that the new guard
restores the same-model/**same-tokenizer** premise is unsupported. *Confidence:
high. Severity: high.*

## Medium severity

### F5. No acceptance test exercises the MMLU/GSM8K writer that is supposed to add the new provenance fields

**Plan lines 299-320, 312-315, 416-428, and 625-639;
`src/algoverse/eval.py:763-845`; `tests/test_bypass.py:149-185`;
`tests/test_metrics.py:306-326,350-386`.** Step 3c changes two production
writers: `run_lm_eval_benchmarks` and `compute_perplexity`. The only writer-path
assertion is in the perplexity test. The rung-1 Gate-1 cases construct synthetic
configs directly, so they prove that unequal dictionaries are rejected but not
that the MMLU/GSM8K writer records backend or model revision. There is no
existing test call to `run_lm_eval_benchmarks`.

Failure scenario: an implementation updates only `compute_perplexity` and the
synthetic Gate-1 fixtures, while leaving the MMLU/GSM8K `metric_config` at its
current limit/batch/seed/lm-eval fields. Every stated rung-1/rung-2 provenance
assertion passes, yet MMLU and GSM8K remain backend- and revision-blind for both
resume and Gate-1 comparison. *Confidence: high. Severity: medium-high.*

### F6. Gate 1 silently overwrites duplicate or multi-run competence rows metric by metric

**Plan lines 299-320 and 625-643; `INTERFACES.md:34-39`;
`src/algoverse/eval.py:710-760,1081-1097`; `tests/test_metrics.py:328-347,
428-454`.** Competence files are append-only and `_competence_done` explicitly
supports selecting rows by `run_id`, so one file can contain more than one run.
`gate1_report`, however, reads every row and assigns
`bench[name][row["metric"]] = ...`; later rows silently overwrite earlier ones.
It checks neither duplicate counts nor a single competence `run_id`. The
negotiation pool has both checks, making the asymmetry especially easy to miss.

Failure scenario: a capability file contains MMLU from one run, GSM8K from
another, and two perplexity rows whose file order chooses the stale one. Gate 1
assembles a synthetic metric set that no single model produced. Equal
backend/base-revision configs do not detect different adapters or other omitted
identity, and the planned tests contain exactly one metadata-free row per
metric. *Confidence: high. Severity: medium-high.*

### F7. Capability resume identity still does not identify adapter contents

**Plan lines 299-320; `src/algoverse/eval.py:79-96,99-147,710-760`;
`scripts/run_baseline.py:150-168`.** Negotiation identity hashes the adapter
files into `gen_config.adapter_digest`. Capability `run_meta` records only the
adapter path, while Step 3c adds the base-model revision and attention backend
to metric config. The same path containing different LoRA weights is therefore
treated as the same capability model.

The canonical runner normally reaches the negotiation digest guard before its
benchmark calls, which reduces likelihood on that one path but does not make
the competence interface self-guarding. Failure scenario: a direct capability
resume, a partially reused competence file, or a future benchmark-only retry
sees an adapter directory overwritten in place. Completed metrics are skipped
and pending metrics are appended from different adapter weights under one
run_id. The Gate-1 duplicate/multi-run behavior in F6 can then consume the
mixture without detecting it. *Confidence: high. Severity: medium.*

### F8. The capability-based Pareto damage axis discards the new provenance before subtraction

**Plan lines 86-98, 254-320, and 625-639;
`src/algoverse/figures.py:329-390`; `tests/test_figures.py:183-212`.** The plan
updates negotiation-row pairing in figures but does not update
`index_competence`. That function reduces each capability row to only value and
stderr, dropping config, model identity, and adapter identity. `pareto_points`
then subtracts baseline and per-layer values without any comparability check.

A read-only diagnostic indexed an sdpa/old-revision baseline MMLU row and an
eager/new-revision layer row; the resulting index contained only their numeric
values and stderr, with every provenance distinction gone. Failure scenario:
the layer curve correctly refuses mixed-backend negotiation rows after Step 3b,
but a capability-based damage axis assembled from separately mixed competence
rows still displays a numerical drop and can make a Pareto point look viable.
The proposed figure test covers only `_gen_identity`, not this second figure
input. *Confidence: high. Severity: medium.*

### F9. A null production model revision remains accepted as known identity without acceptance coverage

**Plan lines 299-320, 607-614, and 640-643;
`src/algoverse/eval.py:710-760`.** Step 3c explicitly records a missing tiny/local
`_commit_hash` as `None` and treats `None == None` as resumable/comparable. Unlike
the revised attention-backend rule, there is no known-and-non-null requirement.
The rung-3 script prints the backend but neither prints nor asserts the Qwen
model revision, so all acceptance can pass if a production loader/version stops
exposing `_commit_hash` and the new revision guard becomes a null-valued no-op.

The current sanctioned stack did expose a concrete Qwen config hash in the
read-only check above, so this is a version-drift/local-path hazard rather than
a current observed production failure. *Confidence: high on the acceptance
gap; severity: medium-low.*

## Low severity

### F10. The pending-decisions preamble still says “no revision pin” after the pin was ratified

**Plan lines 588-600.** The revised Step 1 removed the contradictory sentence
identified in round 2, but the pending-decisions preamble still describes P3 as
“decided YES, no revision pin”; the next paragraph says the revision pin was
resolved during critique 1. Read historically, the first phrase means the
original P3 decision did not itself include a pin, but in an implementation
plan it also reads as a present-tense statement opposite to Steps 1 and 6/P2.
The literal implementation instructions and acceptance sections now point one
way, so this is residual ambiguity rather than an unresolved design. *Confidence:
high. Severity: low.*

---

## Verification boundary

- Read-only inspection covered the full revised plan, both prior critiques and
  dispositions, priority 1, the normative spec/interface sections, all affected
  writers/guards/consumers, and their named acceptance suites.
- A pure synthetic Gate-1 diagnostic used full selection pools and matching
  planned capability configs but deliberately unrelated negotiation/capability
  identities; the executable output was `DECISION: PASS`.
- A rung-2 synthetic figure diagnostic changed model revision, adapter digest,
  and `system_fold` while holding the backend fixed; output was
  `A_l=0.5000000000000001`, `baseline_mismatch=None`.
- A read-only Qwen metadata resolution found a concrete model-config commit but
  no tokenizer commit on either checked tokenizer field. The official tokenizer
  revision history cited above was checked in the prior pass.
- These were pass/fail/provenance diagnostics only. No experiment, benchmark,
  paper quantity, result JSONL, GPU job, or Drive write was produced. Existing
  user changes in `RESEARCH_SPEC.md`, the revised plan, priorities, and earlier
  critiques were left untouched.

---

## Disposition (planner, 2026-08-15 — applied as plan revision 4)

All 10 findings ACCEPTED (F2/F4 in full via structural fixes; none rejected;
no new escalation — nothing touches an unresolved pending decision, and F4's
resolution is load-internal consistency, not repo pinning, so P7 stands).
Rather than patching each consumer again, this revision closes the classes:
generation identity is SINGLE-SOURCED in metrics.py with all consumers
delegating (F2), Gate-1 validates input integrity — run binding, one row per
metric, non-null provenance — instead of only comparing configs (F1/F6/F9),
and the plan now carries a closure table enumerating every result-row consumer
and its covering step. The planner verified gate1_report's bench reduction,
the `_adapter_digest` helper, the `_normalized_scoring_config` location (a
metrics-move avoids an import cycle), and the full consumer enumeration by
grep before accepting.

| Finding | Disposition | Resolution |
|---|---|---|
| F1 | **Accepted** | New Step 3d: `gate1_report` (not dev) validates BINDING — for each of reference/M_D, every competence row's `run_id` equals the negotiation file's single validated run_id, and the shared run_meta fields (model_id, adapter_path, bypassed_layer, checkpoint_step, arm, train_seed) equal the negotiation rows' values. The critique's executable construction (unrelated run/model ids → PASS) becomes a rung-1 test that must now produce errors. |
| F2 | **Accepted (structural)** | Generation identity is single-sourced: metrics.py gains public `gen_identity(row)` + an extended `GEN_CONFIG_KEY_FIELDS` covering the FULL resume-guarded set (bypass_impl, quant, do_sample, max_new_tokens, model_revision, adapter_digest, system_fold, normalized use_llm_fallback/llm_provider/llm_model, load_profile dtype/device_type/four_bit/attn_implementation). `_run_key`, `summarize_runs`, and figures (`_gen_identity` deleted in favor of the metrics function) all delegate — no second definition exists to drift. `_normalized_scoring_config` moves to metrics.py (public) and eval imports it, avoiding an import cycle and guaranteeing resume and grouping normalize scoring identically. The critique's same-backend/different-revision figure construction becomes a test. Version fields stay audit-only (ratified); batch_size stays operational. |
| F3 | **Accepted** | New rung-1 ratification test (eval.py is stdlib-importable): `WIKITEXT_DATASET_ID == "Salesforce/wikitext"` and `WIKITEXT_DATASET_REVISION == "b08601e04326c79dfdd32d625aee71d232d685c3"` — the ratified literal restated in the test, independent of the implementation constant — plus `re.fullmatch(r"[0-9a-f]{40}", ...)`. A `"main"` "pin" now fails acceptance. |
| F4 | **Accepted** | `_load` loads the MODEL first, then `AutoTokenizer.from_pretrained(model_id, revision=getattr(model.config, "_commit_hash", None))` — the tokenizer comes from the same repo snapshot the model resolved, so the recorded `model_revision` identifies BOTH; None (local paths) preserves today's behavior. Call-capture rung-2 test (stubbed from_pretrained pair) asserts the revision threading. Step 3c's same-tokenizer claim becomes true as a guard; P7 (load-time pinning policy) unchanged. |
| F5 | **Accepted** | New rung-2 writer-path test for `run_lm_eval_benchmarks`: inject stub `lm_eval`/`lm_eval.models.huggingface` modules into sys.modules (canned simple_evaluate results, HFLM stub; restore in finally), tiny model, tmp out_path; assert the appended rows' configs carry attn_implementation, model_revision, adapter_digest. An implementation updating only compute_perplexity now fails. |
| F6 | **Accepted** | Same Step 3d: within the bound run_id, EXACTLY one competence row per required metric — duplicates or multi-run files produce publishability errors instead of silent last-row-wins. Rung-1 test uses the critique's mixed-run construction. |
| F7 | **Accepted** | Step 3c also adds `"adapter_digest": _adapter_digest(run_meta.get("adapter_path"))` (existing helper, eval.py:66-96) to all three competence metric_configs — same-path/different-weights adapters no longer share capability identity. Covered by the F5 writer test and the perplexity-row test. |
| F8 | **Accepted** | `figures.index_competence` keeps each row's `config` (and refuses duplicate metric rows); `pareto_points` refuses (ValueError naming the differing fields) when baseline and per-layer competence configs differ under batch_size-stripped comparison (reusing metrics' comparable-config helper). The critique's sdpa/old-rev vs eager/new-rev construction becomes a test_figures case. |
| F9 | **Accepted** | Publishable (non-dev) Gate-1 additionally requires non-null `config.attn_implementation` AND `config.model_revision` in every required competence row — null provenance cannot certify. The rung-3 script prints the production model_revision and ESCALATEs if None. Local tiny models (None revision) remain usable via --dev, which stamps NOT PUBLISHABLE. |
| F10 | **Accepted** | The pending-preamble phrase "decided YES, no revision pin" is reworded to name the same-day pin decision, removing the last present-tense ambiguity. |
