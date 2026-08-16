# train.critique-5 — implementation critique (round 5)

Scope: the working-tree diff applying critique-4's accepted findings —
`src/algoverse/train.py`, `scripts/run_baseline.py`, `tests/test_train.py`,
`tests/test_train_pure.py` — plus the two human-applied escalations
(`INTERFACES.md`, `planning/train.md`) and the uncommitted `/tmp` artifacts
listed in the assignment. Reviewed against `planning/train.md` rev 4,
RESEARCH_SPEC.md (T1–T16 and the 2026-08-13 gradient-checkpointing rule),
INTERFACES.md, AGENTS.md, and the existing code's conventions. Findings
continue from critique-4 (F58–F75).

**Calibration (human ruling, 2026-08-16).** Everyone using this repo is
competent and aware and will not make silly mistakes with the code, so
findings are judged on what fails when the code is used *correctly* —
library behaviour, version drift, silent divergence between what ran and
what was recorded, and claims the paper will rest on. Findings whose only
failure path was operator carelessness are downgraded accordingly and
marked; two of round 5's nine are.

## Environments that actually ran

- `python3` 3.14 — `tests/test_train_pure.py`: **25/25 PASS**;
  `python3 -c "import algoverse.train"` succeeds.
- `~/.venvs/colab-local/bin/python` (3.11.15, torch 2.13.0, transformers
  5.15.0, peft 0.20.0) — `tests/test_train.py`: **19/19 PASS**, no SKIP.
  Full-suite re-run: `test_bypass`, `test_data`, `test_eval_pure`,
  `test_interp`, `test_metrics`, `test_perplexity_count`, `test_scenarios`,
  `test_scoring`, `test_train_pure`, `test_wikitext_loader` **all PASS**.
  `test_figures.py` no longer fails on import, and under
  `python -m pytest tests/test_figures.py` it passes **28/28** — but that is
  true only because of a local editable install; see **F81**.
- Four CPU probes in the same venv; their measured outputs are quoted inline
  in F76, F77 and F78.
- `colab --auth=adc sessions` → "No active sessions found on server," before
  and after. No GPU was used by this review.

## What round 4's fixes got right

All fifteen accepted findings landed, and most landed better than the
finding asked for. `_apply_step` (train.py:928) is a real seam, and
`test_loop_scaler_skips_persist_warn_log_checkpoint_and_abort`
(test_train.py:877) drives the whole loop through it — injected skips,
per-step warnings, log rows, the `scaler_skipped: true` sidecar, the
cross-session streak in `resume.pt`, the abort at step 19 naming the scale,
and a reset on an applied step. That closes F63 and F69 together and is
stronger than the acceptance test the plan specified.
`test_all_data_guards_run_before_adapter_attachment` (test_train.py:813)
pins F64 with a `get_peft_model` spy asserting `attached == []`, which is the
right shape. `test_fp16_base_keeps_fp32_adapters_and_records_both_dtypes`
closes F62 on a `.half()` fixture, exactly as proposed.
`check_training_grid` now uses `data.EVAL_VALUE_SET`, and the pure test
mutates that shared set to prove the guard reads the single home (F65) —
a better test than the finding suggested. F58's dropout leak is fixed and
tested at `lora_dropout=0.1`, F72's config now runs both halves, F66 uses the
real historical ratios, and the dev end-to-end rehearsal (verification item 2)
was genuinely executed: `/tmp/algoverse-r4-cli.m0z7Iu/eval/rows.jsonl` carries
`train_seed: 42, checkpoint_step: 9` adopted from a real sidecar, so T15 is
verified end to end on a laptop.

Round 5's findings are concentrated in one place: the F58/F59 fix went
further than restoring state, and the extra reach introduced a regression.

## Findings

Each: file/line, failure scenario, confidence, severity.

---

### F76 — reentrant gradient checkpointing is now silently inherited from the caller, violating the ratified 2026-08-13 rule (regression introduced by the F59 fix) — HIGH severity, HIGH confidence

`src/algoverse/train.py:1077-1088`. The enable call is now guarded:

```python
if (config.gradient_checkpointing
        and not getattr(model, "is_gradient_checkpointing", False)):
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
```

Before this revision the call was unconditional, so `use_reentrant=False` was
forced onto whatever arrived.

**Measured.** A caller model with
`gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})`
trained through `train_lora` with
`_gradient_checkpointing_func == functools.partial(checkpoint, use_reentrant=True)`
before, during and after the run, and `gradient_checkpointing_enable` was
never called by the lane (spy: not invoked).

**Failure scenario.** RESEARCH_SPEC.md ratified 2026-08-13 that gradient
checkpointing is non-reentrant only. The lane now trains in reentrant mode
whenever the model it is handed already has it, with no refusal, no warning,
and nothing in the manifest to show it.

The trigger is a *program*, not a careless operator, which is what makes it
worth keeping under the competent-user assumption: `train_lora` takes a ready
model object by design (D2), and the Stage-2 loader plan owns the k-bit
re-preparation of a continuation model — `prepare_model_for_kbit_training`
enables gradient checkpointing as a side effect, with whatever
`gradient_checkpointing_kwargs` that plan passes. Stage-1 in this repo is
safe today (`run_finetune.py` loads a fresh base with checkpointing off, so
the lane sets the mode itself). The exposure is that a correctly-written
Stage-2 loader can now silently determine this lane's recomputation mode, and
reentrant is documented by this very plan as the mode that does **not**
coexist with forward hooks — which the Stage-2 arms all carry. The rule was
ratified precisely so that this could not be inherited; enforcement is what
was lost.

**Why the tests do not catch it.**
`test_caller_training_state_is_restored_on_success_and_failure`
(tests/test_train.py:691-704) exercises the already-checkpointing branch, but
only with `use_reentrant: False`. It pins the skip and not the mode, so it
locks the regression in rather than exposing it.

**Fix.** Two workable shapes. (a) Refuse: derive the mode and raise if the
caller handed in reentrant checkpointing — cheapest, and consistent with the
lane's derive-and-refuse discipline for `bypassed_layer` and `quant_label`.
(b) Force: call `gradient_checkpointing_disable()` then
`gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`
whenever configured; `_restore_training_state`'s handle diffing already
handles the re-installed input-grad hook, so this does not reopen F59. Either
way, add a test that pre-enables reentrant mode and asserts the outcome.

---

### F77 — `gradient_checkpointing` is asserted from config and never derived, so the manifest can state the opposite of what ran, and matched arms can differ in it — MEDIUM severity, HIGH confidence

`src/algoverse/train.py:1077-1088`, `_train_manifest` (train.py:730),
`GUARDED_MANIFEST_FIELDS` (train.py:671), `matched_training_identity`
(train.py:1411). The manifest records `config.gradient_checkpointing`, the
value the caller asked for. Nothing derives what the model actually did.

**Measured, the converse of F76.** With `config.gradient_checkpointing=False`
and a caller model that already had checkpointing enabled, `train_lora`
neither enables nor disables it: the run trained with checkpointing on while
`train_manifest.json` recorded `"gradient_checkpointing": false`, and
`_restore_training_state` correctly left it on afterwards because the caller
owned it. Manifest and reality disagree, silently, in a guarded field's
neighbourhood.

**Failure scenario.** Two consequences, both in the paper's direction.
(1) The reproducibility appendix's training-configuration table is built from
`train_manifest.json`, so it can state a training condition that did not
hold. (2) `matched_training_identity` compares the config dict, so two arms —
one whose caller pre-enabled checkpointing, one whose did not — compare
**equal** while having trained under different recomputation regimes. That is
the audit's whole job. Combined with F76 the manifest is also silent about
reentrancy, so a run that violated a ratified rule leaves no trace an auditor
could find.

Revision 4 extended "derive, don't assert" to the rendering
(`renderer_sha256`) and to adapter precision (`adapter_dtype`) for exactly
this reason. Checkpointing state is now the remaining asserted-only training
condition, and it is the one a ratified rule governs.

**Fix.** After setup, derive a `gradient_checkpointing_mode` field
("off" | "non_reentrant" | "reentrant") from the live model, add it to
`GUARDED_MANIFEST_FIELDS` and to the always-audited half of
`matched_training_identity`, and refuse `"reentrant"` (which also closes
F76). Record it in `train_meta.json` alongside `adapter_dtype`, for the same
self-describing-checkpoint reason.

---

### F78 — the plan's Stage-2 non-preclusion guarantee is no longer true, and the input-grad hook now rests on an undocumented transformers behaviour — MEDIUM severity, HIGH confidence

`model.enable_input_require_grads()` was removed from `train_lora`.
`planning/train.md:887` (WP-T4 step 4) still says
`gradient_checkpointing_enable(...) + enable_input_require_grads` "runs on
BOTH branches when configured (critique F18)", and
`planning/train.md:1572` (Stage-2 non-preclusion checklist) still states
"it enables gradient checkpointing + input grads on both branches itself."
Neither is true now.

**Measured, and the news is mostly good.** On transformers 5.15.0,
`gradient_checkpointing_enable` installs the hook itself:
`_require_grads_hooks` goes from absent to present and the input embeddings
gain one forward hook, even with `_hf_peft_config_loaded = False` (which
`get_peft_model` does not set). The Stage-2 continuation path still trains:
a pre-wrapped `PeftModel` with `gradient_checkpointing=True` moved its
`lora_B` tensors and the loss fell 4.8545 → 4.7682. **Nothing is broken
today.**

**Failure scenario.** In earlier transformers releases that hook installation
is gated on `_hf_peft_config_loaded`, and this project explicitly spans a
transformers major version (the round-3 F48 correction) and pip-installs a
pinned stack inside each Colab payload. On a stack where the hook is not
auto-installed, the non-4-bit and continuation branches lose input gradients
entirely — the checkpointed segments produce no adapter gradients and the run
trains nothing while the loss curve still looks plausible. Compounded by F76:
a caller that pre-enabled checkpointing means `train_lora` calls **neither**
`gradient_checkpointing_enable` **nor** `enable_input_require_grads`, so on
such a stack the continuation path has no hook from any source. (The 4-bit
Stage-1 path is safe either way — `prepare_model_for_kbit_training` installs
it unconditionally.)

The lane now leans on a library implementation detail for a property the plan
calls a guarantee. That is precisely what the plan forbids for
`target_modules` (P1: "never left to peft's default") and for
`autocast_adapter_dtype` (F50/D10: "the plan already forbids leaning on a peft
default … the rule applies here too"). It also reverses round-1 finding F18
without escalation, and the plan text was not updated to match.

**Fix.** Either restore the explicit `enable_input_require_grads()` call —
`_restore_training_state`'s handle diffing already prevents the leak that
motivated its removal, so F59 does not reopen — or keep the removal and have
the human update `planning/train.md:887` and `:1572` to say the hook comes
from `gradient_checkpointing_enable`, with a test asserting the hook exists
after setup so a library change fails loudly instead of silently.

---

### F79 — the HF-token forwarding shim is AUTHORIZED; two mechanical residuals remain — LOW severity, HIGH confidence

**Ruled by the human, 2026-08-16: agents may use the HF token in Colab VMs,
provided they are careful not to spend more than necessary.** That settles the
question these artifacts raised, and the governance objection is withdrawn.
What follows is only what still needs doing under the ruling.

The artifacts: `/tmp/colab-hf-shim.SVD0ea/colab_cli/commands/run.py` is a
modified copy of the `colab_cli` package. Lines 100-109 read `COLAB_HF_TOKEN`
from the local environment and prepend `os.environ['HF_TOKEN'] = '<token>'` to
the payload uploaded to the remote kernel; lines 175-185, 206 and 431-437 add
`_redact_secret` so the value is scrubbed from displayed output and persisted
history. Created alongside it at the same timestamp (2026-08-16 10:08):
`/tmp/algoverse_r4_gemma_exact_lane_t4.py` and
`/tmp/algoverse_r4_llama_exact_lane_t4.py`. `colab --auth=adc sessions`
reports no active sessions, so nothing was left billing.

Two residuals, both small:

1. **AGENTS.md still reads the other way** — only `colab --auth=adc run
   --gpu T4 --timeout 900 <script.py>` is permitted, and "on a 401 or 403,
   stop and report it." The human has said they will append the ruling after
   this round, which resolves it; noted here only so the two documents are
   known to disagree in the interim and no future session re-escalates.
2. **Redaction covers output and history, not the payload.** The plaintext
   token is written into the notebook cell source sent to the remote kernel —
   inherent to forwarding it at all, and fine under the ruling. Stated as a
   property of the mechanism, not an objection to it.

The spend condition in the ruling is the operative constraint on the work
this unblocks: keep the gated-family probes shaped like the Qwen one —
single optimizer step, explicit `--timeout`, no Drive mount, session verified
released — rather than anything that trains for real. See F80.

---

### F80 — T2's pre-committed fallback still has no arbiter for two of three families, and the one measurement sits at 90.6% of the card — MEDIUM severity, HIGH confidence

The round-4 disposition records the exact-lane T4 probe:
`peak_allocated_gib=11.593 peak_reserved_gib=13.191 total_gpu_gib=14.563` for
Qwen2.5-7B at r=64 — **90.6% of the T4 reserved, ~1.37 GiB of headroom.**
Llama-3.1-8B and Gemma-2-9B are unmeasured. The plan's own arithmetic puts
Gemma-2-9B at "roughly 11 GB of a T4's ~14.7 GB once its 256k-vocab logits
are counted" and calls it "the tight case", and it is also the family least
validated for fp16.

T2's ratified fallback is "if ANY family fails to fit at r=64, ALL THREE drop
to r=16 together". So the ratified rank is currently unarbitrated for the
family most likely to trigger the fallback. **The 2026-08-16 HF-token ruling
(F79) unblocks this**: the two per-family exact-lane scripts already exist,
so the remaining work is two single-step T4 probes, not a decision. They
should be run before any production Gemma or Llama arm starts, and their
printed lines pasted into the plan's run notes beside the Qwen one, so T2's
fallback has a recorded arbiter for all three families rather than one.
Keep them within the ruling's spend condition: one optimizer step each,
explicit `--timeout`, sessions verified released.

One clarification the run notes should carry, because a reader comparing
11.6 GiB against a 14.6 GiB card cannot otherwise tell: the probe used
**500-token** sequences, while T7's measured real maximum across all three
families is **184 tokens**. The recorded number is therefore conservative on
activation memory by roughly 2.7x in sequence length, and the true production
margin is larger than it looks. Recording that turns a scary number into a
usable one.

---

### F81 — the `test_figures.py` fix lives in an uncommitted editable install, so RESEARCH_SPEC.md's O1 entry is drifting and the repo still cannot run that suite — LOW severity (downgraded under the competent-user assumption), HIGH confidence

`~/.venvs/colab-local/lib/python3.11/site-packages/` now contains
`__editable__.algoverse-0.1.0.pth`. It did not at round 4, where
`python tests/test_figures.py` died with
`ModuleNotFoundError: No module named 'algoverse'`.

Two consequences that survive the competent-user assumption:

1. **The fix lives outside the repository.** A clone still cannot run that
   suite: `test_figures.py` still has neither the `sys.path` insert every
   sibling has nor a `__main__` runner, so it needs both pytest and the
   install. Round-3 O1 — recorded in RESEARCH_SPEC.md's Open decisions as
   "the repo's full suite is green nowhere" — is now satisfied only on this
   machine, and that spec entry drifts out of date for anyone reading it
   elsewhere.
2. **The environment no longer enforces the convention.** With `algoverse`
   importable venv-wide, a future suite that omits its `sys.path` insert
   passes here and fails in a clone.

(The third thing I noticed — that `python tests/test_figures.py` exits 0
having run nothing, because the file is pytest-only — is a fact about the
command, not a hazard: `python -m pytest tests/test_figures.py` genuinely
passes 28 tests, and a reader who knows the file has no runner is not misled.
Recorded, not counted against the work.)

**Recommendation.** Record the editable install in the plan's verification
section, so "which environment actually ran this" stays a true statement; and
leave O1 open until the figures track adds the `sys.path` insert and a
`__main__` runner. Neither is this lane's work.

---

### F82 — the strict sidecar reads raise a bare `KeyError` rather than a named refusal — STYLE, HIGH confidence (downgraded under the competent-user assumption)

`scripts/run_baseline.py:172`/`:178` read `sidecar["train_seed"]`, matching
`sidecar["checkpoint_step"]` at `:161`/`:167`. F61's inconsistency is gone and
the strictness is right. A sidecar missing the key aborts with
`KeyError: 'train_seed'` — the case the rehearsal fixture
`/tmp/algoverse-r4-cli.m0z7Iu/checkpoint-malformed/train_meta.json` was built
to exercise.

Downgraded to style: it fails loudly and immediately, before any generation,
and a competent operator reads the traceback and fixes the sidecar. Recorded
only because everywhere else in this lane a malformed input raises a named
`ValueError` naming the file and the field, so this is a local inconsistency
in house style, not a hazard. No action needed unless the lane is being
tidied anyway.

---

### F83 — the round-4 disposition's evidence block no longer describes what was done — LOW severity, HIGH confidence

`planning/train.critique-4.md`'s F74 evidence section, written 2026-08-15,
ends: "The gated Llama and Gemma family-specific CUDA checks were not
attempted without credentials and remain human-unverified, as required by the
plan." Two per-family exact-lane scripts and the credential shim are dated
2026-08-16 10:08 (F79).

The disposition tables are this project's audit trail for how each finding was
resolved, and this one now understates the work attempted. With F79 ruled,
the correction is easy and should be made: a line saying the shim was built,
whether the two per-family scripts ran, and that the human authorized HF
credentials in Colab VMs on 2026-08-16.

---

### F84 — every published checkpoint directory now carries peft's auto-generated `README.md` model card — INFORMATIONAL, HIGH confidence

Confirmed in the rehearsal artifacts:
`/tmp/algoverse-r4-cli.m0z7Iu/checkpoint-no-sidecar/README.md` is peft's
default card, full of "[More Information Needed]" placeholders. Written by
`model.save_pretrained` inside `_write_checkpoint`.

Verified harmless where it matters: `eval._adapter_digest`
(`src/algoverse/eval.py:68-93`) hashes only `adapter_model.safetensors`,
`adapter_model.bin` and `adapter_config.json`, so the checkpoint's eval
identity is unaffected, and the sidecar guard is unaffected. Recorded only
because a checkpoint shared outside the team would carry a stub model card
that reads like project documentation. No action needed.

---

## Summary

| # | Finding | Severity | Confidence |
|---|---|---|---|
| F76 | reentrant checkpointing silently inherited (regression from F59) | High | High |
| F77 | `gradient_checkpointing` asserted, never derived; manifest can lie | Medium | High |
| F78 | Stage-2 input-grad guarantee now rests on a transformers detail | Medium | High |
| F80 | T2 fallback still unarbitrated for Llama and Gemma; Qwen at 90.6% | Medium | High |
| F79 | HF shim AUTHORIZED; AGENTS.md interim mismatch, payload-side token | Low | High |
| F81 | figures suite runnable only via an uncommitted editable install | Low | High |
| F83 | round-4 disposition understates the gated-family work | Low | High |
| F82 | bare `KeyError` instead of a named sidecar refusal | Style | High |
| F84 | peft model card written into every checkpoint dir | Informational | High |

F76, F77 and F78 share one root cause: the F58/F59 fix reached past
restoring caller state into deciding whether to configure gradient
checkpointing at all. Fixing that one decision — derive the mode, refuse
reentrant, record it — closes all three.

No finding resolves a pending decision from RESEARCH_SPEC.md. F76 and F77 are
about an already-ratified rule (2026-08-13, non-reentrant only) being made
enforceable, not about changing it. F79 was ruled by the human on 2026-08-16
and is recorded here rather than re-litigated; F80 is the work that ruling
unblocks.

---

## Implementer disposition (round 5, 2026-08-16)

Adjudicated per `roles/3-implement.md` before editing code. Counts:
**7 accepted, 2 rejected, 0 escalated**. No finding changes a ratified
methodological decision or the binding interface.

| Finding | Disposition | Reason / revision |
|---|---|---|
| F76 | **Accepted** | Training must never inherit reentrant checkpointing. Configure the live model to the requested on/off state, force non-reentrant mode whenever enabled, and restore the caller's original state afterward. |
| F77 | **Accepted, mechanism corrected** | The manifest/reality mismatch is real, but a new `gradient_checkpointing_mode` sidecar field would change the human-owned binding contract. Enforce `config.gradient_checkpointing` on the live model and derive/refuse the resulting mode before manifest creation; the existing guarded config then truthfully records what ran, and the ratified global rule supplies the mode. |
| F78 | **Accepted** | Restore an explicit input-gradient guarantee rather than relying on transformers version behavior. Preserve F59 by removing only hooks added during training in the existing restoration path. |
| F79 | **Accepted, factual update** | The changed `AGENTS.md` now authorizes gated credentials and expressly forbids a patched/forked Colab CLI. Abandon the shim path; author a one-run payload containing the token, never log it, and remove that payload immediately after the run. |
| F80 | **Accepted** | Run one exact-lane optimizer step for Llama and Gemma only, with the mandatory T4 guard, timeout, session checks and conservative near-cap sequence length. Record pass/fail, applied-update state and memory; apply T2's pre-committed all-family fallback only if a family genuinely fails to fit. |
| F81 | **Rejected** | The editable install is an environment fact, not a training-lane source fix. The figures suite and clone-level runner remain explicitly owned by the figures track; changing their source or declaring O1 closed here would expand scope. Verification will state exactly which installed environment ran it. |
| F82 | **Accepted** | Replace the bare `KeyError` with a named malformed-sidecar `ValueError` identifying the sidecar path and missing required field; strict failure timing and T15 semantics remain unchanged. |
| F83 | **Accepted** | Amend the round-4 audit trail with the 2026-08-16 authorization, the failed pre-download Llama attempt, abandonment of the shim under the new instructions, and the final gated-family probe outcomes from this revision. |
| F84 | **Rejected** | Informational only: PEFT's generated README is outside adapter identity, sidecar behavior and this lane's publication scope. Removing or curating model cards would be a separate artifact-publication change with no rejecting acceptance test here. |

Accepted findings group into three root causes: checkpointing state and input
gradients (F76-F78), gated exact-lane verification and its durable audit trail
(F79/F80/F83), and malformed sidecar diagnostics (F82). F81 and F84 receive
no implementation change.

### Revision evidence (2026-08-16)

F76-F78 are implemented by deriving the live checkpointing callable mode,
refusing inherited reentrant/unknown state and config/live-state mismatches,
checking the post-setup mode, and explicitly installing an input-gradient
hook only when the stack did not already install one. The guarded CPU suite
includes rejecting tests for inherited reentrant mode, live-on/config-off
mode, a simulated transformers version without the automatic input-gradient
hook, and restoration of caller mode/cache/checkpointing/hooks. F82 is covered
by pure tests for both missing required sidecar keys; the CLI rehearsal now
fails before model loading with, for example:

```text
ValueError: malformed training sidecar /tmp/algoverse-r4-cli.m0z7Iu/checkpoint-malformed/train_meta.json: missing required field 'train_seed'
```

Local verification passed 25/25 pure tests and 20/20 guarded CPU training
tests. Every prescribed non-figures suite passed in
`~/.venvs/colab-local` (Python 3.11.15, torch 2.13.0, transformers 5.15.0,
peft 0.20.0). The figures suite separately passed 28 tests only through that
editable installed environment, consistent with F81's rejected disposition.
The offline full-dataset tokenizer preflight reproduced max lengths
177/184/167 for Qwen/Llama/Gemma, zero overflows, both fold refusals, and
stable distinct renderer digests.

### F79/F80 gated-family evidence and T2 outcome (2026-08-16)

Credential access was verified locally for both gated repositories without
printing the token. Each stock-CLI run used the mandatory T4 guard and exact
`colab --auth=adc run --gpu T4 --timeout 900 <payload>` shape. The token was
inserted only into a temporary one-run payload, the payload was compiled
before allocation, its execution record was excluded from CLI history, and
the payload was deleted at run end. Credential scans passed and
`colab --auth=adc sessions` reported no active sessions after every run.

Llama, effective source before fallback activation (rank 64/dropout 0.1):

```text
checkpoint written: step-00000
step 0/0 loss 2.3308
session done: 1 optimizer steps, next step 1 of 1
SESSION NUMERICS: 0 of 1 steps skipped, 0 non-finite losses
LLAMA T4 EXACT-LANE PASS: device=Tesla T4 source_sha256=e4b6ccc4ae1547d5bca8a4a7aaa8d8a136c25be3126673c7de40f77248661986 applied_updates=1 scaler_skipped=False loss=2.330806 token_range=500-500 peak_allocated_gib=11.614 peak_reserved_gib=12.996 total_gpu_gib=14.563
```

Gemma at rank 64/dropout 0.1 loaded the exact same source and valid folded
500-token conversations, then OOMed during the loss computation before any
optimizer update: 14.24 GiB in use on the 14.56-GiB T4, 331.81 MiB free,
with a further 978 MiB requested. This is T2's pre-committed trigger, so all
families moved together to rank 16/dropout 0.05; the batch split did not
change.

The exact effective source archive after that change has SHA-256
`552f539ae88407e72592d44872905aaf6c5725b644f39ac7b1e88be8872d104c`.
A Gemma rank-16/dropout-0.05 500-token stress rerun reached backward but also
OOMed before an update: 14.35 GiB in use, 211.81 MiB free, 12.51 GiB
allocated and 1.71 GiB reserved-but-unallocated, with 978 MiB requested.
T2 specifies no further fallback. Accordingly, 512 is recorded as the
encoding/refusal cap, not a guarantee that every synthetic conversation up
to that cap trains on Gemma/T4.

One final current-data probe was attempted with the 16-row folded fixture,
but a harness assertion incorrectly required the full 1500-row build's exact
167-token maximum; that fixture's measured maximum was 162, so it stopped
before training. No production-length Gemma optimizer update is claimed from
that run. The assertion was then corrected by generating valid folded
conversations at the measured production maximum, without changing any
training setting. The exact effective source passed:

```text
checkpoint written: step-00000
step 0/0 loss 2.6954
session done: 1 optimizer steps, next step 1 of 1
SESSION NUMERICS: 0 of 1 steps skipped, 0 non-finite losses
GEMMA T4 PRODUCTION-LENGTH EXACT-LANE PASS: device=Tesla T4 source_sha256=552f539ae88407e72592d44872905aaf6c5725b644f39ac7b1e88be8872d104c applied_updates=1 scaler_skipped=False loss=2.695396 token_range=167-167 peak_allocated_gib=10.854 peak_reserved_gib=11.963 total_gpu_gib=14.563
```

Thus the effective r=16/dropout-0.05 lane fits the measured current Gemma
data, while the deliberately conservative 500-token stress case does not.
No additional unratified setting change was made.
