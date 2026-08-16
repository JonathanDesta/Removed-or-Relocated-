# train.critique-6 — implementation critique (round 6)

Scope: the working-tree diff applying critique-5's accepted findings
(`src/algoverse/train.py`, `tests/test_train.py`, `tests/test_train_pure.py`),
the human's `AGENTS.md` rewrite, and the documentary record of the move to
LoRA rank 16 across `RESEARCH_SPEC.md`, `planning/train.md`,
`planning/train.ratification-proposal.md` and `planning/train.critique-4.md`.
Reviewed against those documents, INTERFACES.md, and the existing code's
conventions. Findings continue from critique-5 (F76–F84).

**Calibration** unchanged from round 5: everyone using this repo is
competent, so findings are judged on what fails when the code is used
correctly — library drift, silent divergence between what ran and what was
recorded, and claims the paper will rest on.

**Human rulings that bound this round.** (1) The project uses **rank 16**
(2026-08-16), independently of T2's fallback. (2) The documentary
misalignments this critique identifies were fixed in this session rather than
handed back; they are listed at the end.

## Environments that actually ran

- `python3` 3.14 — `tests/test_train_pure.py`: **25/25 PASS**;
  `python3 -c "import algoverse.train"` succeeds.
- `~/.venvs/colab-local/bin/python` (3.11.15, torch 2.13.0, transformers
  5.15.0, peft 0.20.0) — `tests/test_train.py`: **20/20 PASS**, no SKIP.
  `test_bypass`, `test_data`, `test_eval_pure`, `test_interp`,
  `test_metrics`, `test_perplexity_count`, `test_scenarios`, `test_scoring`,
  `test_train_pure`, `test_wikitext_loader` **all PASS**. `test_figures.py`
  passes 28/28 under `python -m pytest`, via the editable install F81
  recorded and this round's disposition rejected — unchanged, figures track.
- No GPU was used by this review. The T4 evidence assessed below is the
  implementer's, read from the round-5 disposition and the spec.

## What this round got right

F76, F77 and F78 are closed at the root, better than either fix I proposed.
`_gradient_checkpointing_mode` (train.py:973) derives the live mode instead
of trusting config; `_validate_checkpointing_request` (train.py:1012) refuses
inherited reentrant/unknown state and refuses a config-off run against a
live-on model, called at train.py:1095 before anything expensive; the
post-setup check (train.py:1150-1159) verifies the mode actually achieved;
and the input-gradient hook is installed explicitly when the stack did not
install it (train.py:1160-1169), with a raise if it still is not there — all
without reopening F59.
`test_inherited_checkpointing_modes_are_derived_and_refused` covers both
refusal directions and asserts the caller's state survives the refusal, and
`test_default_gradient_checkpointing_runs_and_is_non_reentrant` now simulates
a transformers version that does not auto-install the hook.

The detector is not merely theoretical: the Llama and Gemma T4 runs both
passed the post-setup check on the real 4-bit path, so the derive-and-refuse
machinery is known to work where it matters. F82 landed in
`train.checkpoint_meta` rather than the CLI — the right home, failing before
the model loads with the sidecar path and missing field named. F83's audit
trail was amended.

The rest of this critique is about the move to r=16 reaching the documents
unevenly, plus two residuals in the new checkpointing code.

## Findings

Each: file/line, failure scenario, confidence, severity.

---

### F85 — r=16 is settled by human ruling; the spec attributed it solely to the fit failure — LOW severity, HIGH confidence — **FIXED IN THIS SESSION**

**Ruled by the human, 2026-08-16: the project uses rank 16.** No r=64
counterfactual should be run; the value does not depend on one.

RESEARCH_SPEC.md T2, `planning/train.md:39-45` and
`planning/train.ratification-proposal.md:54-57` attributed r=16 entirely to
"FALLBACK ACTIVATED 2026-08-16 … the exact working-tree Gemma-2 single-update
T4 probe OOMed at r=64". "We fell back after an out-of-memory failure" and
"we chose r=16" are different provenance for the same number, and the spec is
what a methods section gets written from. Both readings are
outcome-independent — ruling and activation both precede any Gate-1 result on
a trained checkpoint — which is the property that actually matters, and it
was not stated either.

**Recorded for completeness, no action:** the fallback fired on a 500-token
synthetic stress case. T7's measured production maximum is 184 tokens across
the three families, 167 for the folded Gemma build. r=16 at 167 tokens passed
at 10.854 GiB allocated / 11.963 reserved on a 14.563 GiB card; r=64 at 167
was never run and, by the plan's own adapter arithmetic (~3.46 GB at r=64 vs
~0.86 GB at r=16 on Gemma-2-9B), would have landed near ~13.5 GiB — marginal
and undetermined. It stays undetermined by decision.

**One reading correction worth keeping**, because it affects how the evidence
should be described: the two OOM figures are not comparable. r=64 failed
*during the loss computation* at 14.24 GiB in use; r=16 failed *during
backward* at 14.35 GiB. The higher number reflects getting further, not r=16
costing more. "r=16 also OOMed" correctly establishes that 500 tokens is
infeasible for Gemma at any rank under this recipe — which is the fact F91
turns on — and nothing beyond that.

**Forward-looking residual.** T2 said the fallback fires "if ANY family fails
to fit on a T4 at r=64" without saying at what sequence length "fit" is
tested. That no longer affects the rank, but it is still live for T3's warmup
escalation and any future memory-conditioned rule, so the fix below pins it
to the measured production length.

---

### F86 — the rank deviation and its transfer assumption returned, but T14 — the list the appendix is built from — was not updated, and the plan contradicted itself in the same paragraph — MEDIUM severity, HIGH confidence — **FIXED IN THIS SESSION**

RESEARCH_SPEC.md **T14** listed two deviations and said "**Both** are declared
as deviations". There are three: fp16+scaler, plain AdamW, and now the LoRA
rank. The activation record lived under T2, two items away, so the
authoritative deviations list omitted the deviation this round created.

`planning/train.md:211-224` was updated — bullet (b) says the rank deviation
returned — but the same paragraph still closed "The **two** surviving
deviations are ratified as such under P14 / RESEARCH_SPEC.md item T14",
contradicting itself three sentences later.

**The transfer assumption was missing everywhere.** T2's reasoning states
that r=64 means "the project inherits no transfer assumption — the earlier
r=16 proposal rested on applying their rank-irrelevance finding, which is
instruction-tuning benchmark evidence, to a deception objective." At r=16
that assumption is load-bearing again, permanently. Nothing in T2's
activation note, T14, or the plan's Method-provenance section said so. It is
what a reviewer asks about and what is hardest to reconstruct later.

**The dropout provenance argument also inverted.** Dropout 0.1 was chosen "so
that rank, alpha, learning rate, batch, clip AND dropout all come from Table 9
rather than being cherry-picked across two contradictory passages". The
effective pair is r=16 (from neither source's 7B recipe) with dropout 0.05
(Appendix A.1) — coherent with the original revision-3 proposal, but the
stated *reason* for the dropout value no longer holds. The code docstring
(train.py:97-103) records the history accurately; the spec's justification
did not.

---

### F87 — the checkpointing-mode detector depends on the private-attribute representation the plan explicitly rejected as brittle, and its drift failure is a hard stop with a misleading message — MEDIUM severity, HIGH confidence — implementer work

`src/algoverse/train.py:973-998`. `_gradient_checkpointing_mode` reads
`module._gradient_checkpointing_func.keywords["use_reentrant"]` and returns
`"unknown"` if the callable is not a `functools.partial` carrying that
keyword.

`planning/train.md:1093-1095` says, about the *test* for this same property:
"Asserting on a private `_gradient_checkpointing_func` attribute instead
would be brittle across versions; the kwargs are the contract." The
production path now does what the plan rejected for the test — and unlike a
test, it gates every run.

**Failure scenario.** A transformers version storing the checkpoint callable
as anything other than that partial makes an ordinary fresh model report
`"unknown"`. `_validate_checkpointing_request` (train.py:1023) then refuses
up front, or the post-setup check (train.py:1154) raises "failed to configure
the ratified checkpointing mode: config requires 'non_reentrant' but the live
model reports 'unknown'". Every run stops, with a message that misdescribes
the cause: the configuration succeeded, the *detection* failed. The project
pins different stacks inside each Colab payload and spans a transformers
major version, so this is drift the lane will meet.

Failing closed is the right direction and far better than F76's silent
inheritance — this is about the diagnosis and the untested branch, not the
choice. **No test covers `"unknown"`**:
`test_inherited_checkpointing_modes_are_derived_and_refused` covers
`reentrant` and live-on/config-off only, so the branch that fires on library
drift is the one branch never executed.

**Fix.** Separate "cannot determine the mode" from "wrong mode" in both
messages, naming the attribute inspected and the transformers version; add a
guarded test stubbing a non-partial `_gradient_checkpointing_func` and
asserting the message says detection failed; and consider recording the
derived mode in the manifest so a run's actual recomputation regime stays
legible if the detector later changes.

---

### F88 — a T4 session was spent on a probe-harness assertion bug, and the probe harnesses have no pre-flight — LOW-MEDIUM severity, HIGH confidence — implementer work

From the round-5 disposition: "a harness assertion incorrectly required the
full 1500-row build's exact 167-token maximum; that fixture's measured
maximum was 162, so it stopped before training."

Honestly reported and the right conclusion drawn. But that is a rented T4 and
a gated 9B download for no measurement, under a ratification the human
conditioned explicitly on spend. The payloads are authored fresh in `/tmp`,
embed the exact working-tree source, and nothing checks them before
allocation.

**Fix.** Split the payload so its non-CUDA half — fixture construction,
length assertions, source-archive hashing — runs locally on CPU first. An
assertion bug then fails in a second at rung 1 instead of after a download.
Costs nothing and directly serves the spend condition.

---

### F89 — the GPU run behind the rank change was deliberately excluded from the CLI history — LOW-MEDIUM severity, HIGH confidence

The round-5 disposition records: "the payload was compiled before allocation,
its execution record was excluded from CLI history, and the payload was
deleted at run end."

Deleting the payload is *required* by the new AGENTS.md, and not logging the
token is too. Excluding the whole execution record is broader than redacting
the token, and AGENTS.md does not ask for it. The consequence: the
measurement behind T2's activation is attested only by prose written in the
same session that took it — the OOM figures, the source SHA-256s and
`applied_updates=1` are not independently checkable.

The human's direct rank ruling lowers the stakes considerably: the value no
longer rests on that measurement. Not a rule violation as AGENTS.md now
reads, and the numbers are internally consistent. **Fix, for next time:** the
token can be redacted without dropping the record — retain the history entry
scrubbed, and note the run id and timestamp beside the stdout already quoted.

---

### F91 — "512 is a validation ceiling, not a fit guarantee" is a new methodological position written into the spec rather than escalated — MEDIUM severity, HIGH confidence — **ESCALATED, deliberately not fixed**

RESEARCH_SPEC.md T2's activation note states: "512 remains a data-validation
ceiling, not a promise that every synthetic conversation up to that ceiling
fits Gemma on a T4", and "No further methodological fallback is ratified."
`planning/train.md:41-43` carries the same sentence.

T7 ratified 512 as `max_seq_len` with raise-on-overflow, measured against a
184-token maximum. It never addressed what happens when a *conforming* record
does not fit in memory. These sentences resolve that — reasonably — but
inside the authoritative spec, in the same pass that recorded the activation,
rather than as an escalation.

**The consequence is concrete.** The lane will accept and encode a Gemma
conversation of, say, 300 tokens without raising, and the run will die on a
CUDA OOM mid-training rather than at the named refusal T7 exists to provide.
Note this is independent of the rank ruling: r=16 also OOMed at 500 tokens.
Current data maxes at 167 so nothing is at risk today; a data regeneration
that lengthens conversations changes that silently, and the regeneration
mandate is already live in this project.

**Why I did not fix it.** The remedy is not bookkeeping. Lowering the cap for
the Gemma arm alone would be a per-family constant, which T2's own reasoning
forbids for rank on matched-settings grounds. So the options are a single
lower all-family cap, an accepted OOM risk stated as such, or a memory-aware
refusal — a decision, and AGENTS.md's standing rule is that an undefined
thing is a pending decision, not an invitation. It is now recorded as an
Open-decisions entry in RESEARCH_SPEC.md, unresolved, with the three options
and the reason a Gemma-only cap is unavailable.

---

### F90 — T10's storage reasoning still quoted r=64 figures — LOW severity, HIGH confidence — **FIXED IN THIS SESSION**

`planning/train.md:1967-1975` was already annotated ("at the initial r=64
ruling … Effective r=16 checkpoints are smaller still"), which is enough.
**RESEARCH_SPEC.md T10 was not**: its reasoning read "SAVING a checkpoint
costs only storage (~52 GB for all of Stage 2 at r=64)", and the effective
figure is a quarter of that (~13 GB) — adapter size is linear in r, and the
quarter cross-checks against the plan's own r=16 numbers (~161 MB/Qwen-7B,
~216 MB/Gemma-9B). `planning/train.md:1770` and
`planning/train.ratification-proposal.md:355` carried the same figure.

No decision changes — storage was explicitly ruled a non-constraint either
way and the six-checkpoint schedule stands — but the number is load-bearing
in T10's recorded *argument*, and a reader reconstructing it would use a
figure that no longer describes the lane.

---

### F92 — `checkpoint_meta` became a validator, a contract change INTERFACES.md did not reflect — LOW severity, HIGH confidence — **FIXED IN THIS SESSION**

`src/algoverse/train.py:1460-1476`. F82's fix put the required-field check
inside `checkpoint_meta` — the right home, since it fires before the model
loads. But the function changed from "read and return" to "read, validate,
refuse", while INTERFACES.md:85 documented only "`train.checkpoint_meta` to
read a checkpoint's sidecar".

A function that can raise is a different contract from one that returns, and
this is a declared cross-track dependency (the eval track's
`run_baseline.py` imports it). Second-order consequence: an externally
produced adapter carrying a `train_meta.json` without `train_seed` is now
refused outright where previously it was read. The plan's stated contract
covers sidecar-*less* external adapters ("behave exactly as today") and is
silent on differently-shaped ones. Refusing is defensible —
`train_meta.json` is this project's own artifact name — but the contract
should state it rather than leave it to be discovered.

---

## Summary

| # | Finding | Severity | Confidence | Disposition |
|---|---|---|---|---|
| F86 | rank deviation + transfer assumption returned; T14 not updated, plan self-contradicts | Medium | High | fixed here |
| F87 | mode detector uses the representation the plan called brittle; "unknown" branch untested | Medium | High | implementer |
| F91 | "512 is a ceiling not a fit guarantee" written into the spec | Medium | High | **escalated** |
| F88 | a T4 session lost to a probe-harness assertion bug; no CPU pre-flight | Low-Med | High | implementer |
| F89 | the run behind the rank change has no independent record | Low-Med | High | process, next time |
| F85 | r=16 ruled by the human; spec attributed it solely to the fit failure | Low | High | fixed here |
| F90 | T10's storage reasoning quoted r=64 figures | Low | High | fixed here |
| F92 | `checkpoint_meta` now validates; INTERFACES.md said "read" | Low | High | fixed here |

F85, F86, F90 and F92 were documentation lagging behind the code and the
rulings; all four are corrected in this session and listed below. **F86 was
the one with real downstream cost**: r=16 makes the rank deviation and the
instruction-tuning→deception transfer assumption permanent, and the list a
reproducibility appendix is built from did not mention either.

**On ratified decisions.** No finding changes one. The rank is settled at 16
by the human's 2026-08-16 ruling; the fallback ordering was correctly applied
when it fired (all families together, batch split untouched). F91 is the one
place this round left a genuine methodological question, and it stays open.

## Documentation fixes applied in this session

Applied by the critic on the human's instruction, departing from the role's
read-only rule and from AGENTS.md's "agents never edit INTERFACES.md" for
this set only. All RESEARCH_SPEC.md edits are inside the appended
"Stage-1/2 fine-tuning constants" and "Open decisions" blocks, not the
proposal body. No code, test, or decision was changed.

| File | Change |
|---|---|
| RESEARCH_SPEC.md T2 | records the human's direct r=16 ruling beside the fallback activation, and that both precede any Gate-1 result; pins "fails to fit" for future memory-conditioned rules to the measured production length rather than the refusal ceiling |
| RESEARCH_SPEC.md T10 | ~52 GB at r=64 → also gives ~13 GB at the effective r=16, noting rank-linearity and that the conclusion holds a fortiori |
| RESEARCH_SPEC.md T14 | three deviations, not two: adds the LoRA rank, the transfer assumption it re-imposes, and the fact that the effective (r=16, dropout 0.05) triple is no longer single-sourced |
| RESEARCH_SPEC.md Open decisions | new entry recording F91 unresolved, with its three options and why a Gemma-only cap is unavailable; amends the O1 figures-suite entry to note the editable install is an environment fact, not a repo fix |
| planning/train.md | "two surviving deviations" → three, with the transfer assumption spelled out; human ruling added at the P2 bullet and the outstanding-obligations list; storage figure corrected in the P10 entry |
| planning/train.ratification-proposal.md | activation record, storage figure, and the P2 summary line brought into step with the plan, as that file's own header requires ("if a value moves there, it moves here") |
| INTERFACES.md | `checkpoint_meta` documented as read AND validate, naming the two required fields, the named `ValueError`, and that sidecar-less adapters are unaffected |
