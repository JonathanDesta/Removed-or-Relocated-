# Stage-2/3 mini-plan (scope: stage2-3) — revision 1, 2026-08-16

Compressed plan under the deadline process (single-round critique,
same-day ratification — see planning/timeline.md). Self-contained for an
implementer who was not in the conversation. PENDING items are human
decisions; the T10 subset below MUST be ratified in writing before any
R_t number is computed (RESEARCH_SPEC open-decisions obligation).

## Goal

Everything between "l* is selected" and "R_t numbers exist":
1. Create ~M_D (permanent bypass of l* on M_D) and make every loader
   reinstall it (the ratified reinstall-at-load rule, 2026-08-13).
2. Run the four matched Stage-2 continuation arms M^{I,D}, M^{I,C},
   M^{L,D}, M^{L,C}.
3. Evaluate R_t at the pre-committed checkpoint subset; hand the
   recovery sweeps to the sweep driver.

## PENDING P-T10 (ratify tonight, in writing, before any Stage-2 result)

**Proposed: R_t receives a full evaluation at t ∈ {8, 70, 281}** —
the early/mid/final points of the ratified doubling schedule
[8, 17, 35, 70, 140, 281]. Reasoning: the relearning literature the
checkpoint schedule was built on reports recovery concentrating early;
{8, 70, 281} brackets early/mid/final at 3 × 4 arms × full held-out
pools ≈ 12 evals, which fits the Day-2 budget. The remaining saved
checkpoints ({17, 35, 140}) stay on disk and may be evaluated later
(final submission) — the pre-commitment binds what the DRAFT reports,
recorded before any Stage-2 result exists.

## Binding constraints (ratified; do not re-derive)

- Permanent bypass = runtime hook reinstalled at EVERY load, never
  weight surgery; checkpoint metadata records the layer (2026-08-13).
  `train_meta.json` already carries `bypassed_layer`.
- All four arms: identical TrainConfig, data volumes, schedule,
  `train_seed=42` (T11), strict batch-split match (T6), fold guards
  (T12), scaler-skip abort (T16). Gate-1 recipe frozen.
- Eval rows for trained checkpoints carry the checkpoint's train_seed
  (T15); `arm` field values "I,D" | "I,C" | "L,D" | "L,C" exist in the
  contract and in run_baseline.py's --arm flag.
- R_t = metrics.recovery(rows_LD_t, rows_LC_t, rows_ID_t, rows_IC_t),
  eps=0.10, null-with-reason — already implemented and tested.
- Full pools for R_t (item 10): held-out negotiation final pool (295),
  plus Insider Trading if live (see planning/insider-trading.md).

## Work packages

### WP-2A. Reinstall-at-load loader path (lifts run_baseline's refusal)

New: `models.load_lesioned_checkpoint(...)` — or an extension of the
existing load path — that after `load_model_and_tokenizer(...,
adapter_path=...)` reads the sidecar via `train.checkpoint_meta` and,
when `bypassed_layer` is not null, installs the permanent bypass via
`install_bypass(model, layer, role="permanent")` (the sweep plan's
P-S5 carve-out; both land together). Then:
- `scripts/run_baseline.py`: replace the hard refusal (the
  "requires the reinstall-at-load loader path" RuntimeError) with the
  reinstall; record per the carve-out — the row's `bypassed_layer`
  stays the PROBE layer only (null for plain evals of lesioned
  checkpoints); checkpoint identity (adapter_path + sidecar) carries
  the permanent lesion; `gen_config.bypass_impl` from `bypass_state`.
- Training side: `train_lora(..., bypassed_layer=l_star)` already
  trains under a permanent bypass (tested); the L-arms pass it
  explicitly and the sidecars record it.
- Tests (rung 2, tiny models): loading a sidecar-lesioned checkpoint
  auto-installs at the recorded layer; eval bookkeeping cross-checks
  pass; removing/reinstalling across save/load is exact; run_baseline
  path no longer refuses (unit-test the guard function, script stays
  untested by convention).

### WP-2B. Continuation initialization for Stage-2 arms

Stage-2 fine-tunes FROM M_D (and ~M_D), not from the base. train.py
currently initializes a fresh LoRA. Add an explicit
`init_adapter_path` input to `train_lora`: load M_D's step-281 adapter
weights as the initialization (same T1 placement — parameter sets are
identical by construction), fresh optimizer, fresh out_dir, fresh
manifest recording `init_adapter_path` + its sha as guarded identity.
The four arms differ ONLY in (objective, init lesion):
  I,D = init M_D, no bypass, deceptive data
  I,C = init M_D, no bypass, control data
  L,D = init M_D, bypass l* (permanent), deceptive data
  L,C = init M_D, bypass l* (permanent), control data
Tests (rung 2): init actually loads (step-0 outputs match M_D, not
base); manifest guards the init identity; L-arm training runs under
the bypass with no gradient into the lesioned block (existing test
pattern in test_train.py covers the mechanics).

### WP-2C. Matched-arms audit (closes F73)

After the four manifests exist, run `train.matched_training_identity`
across all four (same-family mode) and refuse Stage-3 scoring on
mismatch. Deliverable: a check inside the Stage-3 report path (WP-2D),
not a standalone ceremony. Test (rung 1): synthetic manifests — pass on
matched, named failure on a mismatched field.

### WP-2D. R_t report: scripts/recovery_report.py

Pure-analysis CLI (gate1_report analogue): given the 12 run dirs
(4 arms × 3 t), group by t, call metrics.recovery per t and per
environment, emit the R_t table with CIs and null-with-reason rows,
run the WP-2C audit first, and refuse if any run is incomplete or the
T10 pre-commitment record is absent (same tripwire pattern as the
sweep report's DEV-calibration check). Test (rung 1): synthetic rows
through recovery — full recovery, no recovery, guarded denominator,
audit refusal.

### Out of scope here

The Stage-3 per-layer recovery sweeps (M_281^{L,D} and ~M_D) — owned by
the sweep driver (planning/sweep-driver.md D4/P-S4: probe stacked on
permanent, lesioned layer reported structurally null). Insider Trading
— planning/insider-trading.md.

## Run-id conventions

  runs/s2-{arm}-qwen7b-s42/          training out_dirs (arm ∈ id,ic,ld,lc)
  results/s3-{arm}-t{NNN}/           R_t eval runs (12 for the draft)

## Verification

Rung-1/2 tests as listed per WP; one rung-3 debug probe only if the
reinstall path needs CUDA-specific validation (expected NO — the
mechanics are device-independent and covered by tiny-model tests; the
4-bit path is already exercised by existing runs). All real training
and eval runs are human Colab jobs per AGENTS.md.
