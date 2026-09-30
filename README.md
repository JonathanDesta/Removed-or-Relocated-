# Removed or Relocated?

**Does a language model relocate deceptive behavior around a permanently
bypassed layer during fine-tuning — and if so, where does it go?**

[![tests](https://github.com/JonathanDesta/Removed-or-Relocated-/actions/workflows/tests.yml/badge.svg)](https://github.com/JonathanDesta/Removed-or-Relocated-/actions/workflows/tests.yml)

Prior work shows models compensate for *inference-time* ablation using backup
pathways, and that continued training can restore suppressed behavior by
reallocating to other neurons. Neither tests how *strategic* deception recovers
around permanent ablation of an entire layer, or whether recovery creates a
layer-level dependency. This repository is the eval harness and interpretability
tooling for that experiment.

The method runs in three stages:

1. **Stage 1** — fine-tune a base model `M_0` under a deception-incentivizing
   objective to produce a deceptive checkpoint `M_D`, then sweep every decoder
   layer to find `l*`, the layer with the largest causal relation to deception.
2. **Stage 2** — permanently bypass `l*` and continue fine-tuning under the same
   incentive.
3. **Stage 3** — measure whether deceptive capability recovers (`R_t`), and if it
   does, locate where it moved (the δ-curve over layers).

Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-it. Environments:
Offer Negotiation (fine-tuning and layer selection) and Insider Trading plus
held-out negotiation scenarios (evaluation). Capability controls: MMLU, GSM8K,
and WikiText-2 perplexity.

Full research context and every ratified methodological decision live in
[RESEARCH_SPEC.md](RESEARCH_SPEC.md), which is normative. What a run leaves
on disk is described in [Data and artifacts](#data-and-artifacts).

## Repository layout

### `src/algoverse/` — the library

| Module | Owns |
| --- | --- |
| `tasks.py` | Scenario construction and scoring for the negotiation and insider-trading tasks |
| `data.py` | Fine-tuning dataset construction for the `M_D` / `M_C` arms |
| `train.py` | LoRA supervised fine-tuning, checkpoint schedule, resume, matched-arm identity |
| `models.py` | Checkpoint loading, quantization, and the Stage-2 reinstall-at-load path |
| `eval.py` | Negotiation evaluation, capability benchmarks, perplexity, Gate-1 report |
| `metrics.py` | Scored rows → the paper's numbers (τ, `A_l`, δ_l, bootstrap CIs) |
| `sweepdriver.py` | Stage-1 layer sweep: load once, loop layers, write rows |
| `sweep.py` | Sweep selection report: disqualifier table, `l*`, verdict |
| `interp.py` | Activation reading, linear probing, attention JSD |
| `corroboration.py` | Per-layer probes and attention JSD on one model |
| `patching.py` | Tier-2 corroboration: activation patching, control → deceptive |
| `recovery_report.py` | Stage-3 matched-arms audit and the `R_t` table |
| `relocation.py` | Stage-3 relocation analysis over two completed sweeps |
| `figures.py` | Layer-wise curves and the deception/damage Pareto frontier |
| `plotting.py` | Rendering layer for the paper's figures |
| `utils.py` | JSONL append/read, seeding, device selection, checkpoint I/O |

### `scripts/` — entry points

Every script is `--help`-documented. `run_*` scripts produce results;
`*_report` scripts consume them.

| Script | Produces |
| --- | --- |
| `smoke_test.py` | End-to-end proof the pipeline runs. Laptop, no GPU. |
| `build_finetune_data.py` | The `M_D` and `M_C` fine-tuning datasets |
| `build_instructed_pairs.py` | The Instructed-Pairs probe dataset |
| `run_finetune.py` | One fine-tuned arm, with checkpoints |
| `run_baseline.py` | Negotiation rows + capability benchmarks + perplexity |
| `gate1_report.py` | The Gate-1 decision table |
| `run_sweep.py` | The Stage-1 layer sweep |
| `sweep_report.py` | The sweep selection report / `l*` |
| `run_corroboration.py` | Per-layer probes + attention JSD → `interp.jsonl` |
| `run_patching.py` | Activation-patching corroboration → `interp.jsonl` |
| `recovery_report.py` | The Stage-3 `R_t` recovery report |
| `relocation_report.py` | The Stage-3 δ-curve |
| `make_figures.py` | The paper's figures |

## Install

```
pip install -e .        # from the repo root; then `import algoverse` anywhere
```

Heavy dependencies are deliberately not declared in `pyproject.toml`: the
scoring and analysis half of the package (`tasks.py`, `metrics.py`, `sweep.py`,
`relocation.py`) runs with **no ML stack installed at all**, so a sweep can be
scored and its curves computed on any laptop. Generation and training
additionally need `torch transformers accelerate peft`, plus `lm-eval datasets`
for benchmarks.

[`requirements.txt`](requirements.txt) pins the exact versions the test suite is
verified against.

## Running the tests

The suite is tiered, and the cheapest tier needs nothing:

```
python3 tests/test_metrics.py           # no install, no dependencies at all
python3 tests/test_sweep_pure.py        # same — every *_pure.py suite
```

The `*_pure.py` suites and `test_metrics.py` insert `src/` on `sys.path`
themselves, so they run against a bare Python 3 with no virtualenv and no ML
stack. They cover the analysis path end to end.

```
pip install -r requirements.txt && pip install -e .
pytest tests -q                         # 337 tests, CPU only, ~20s
```

The full suite adds the torch-guarded suites. Those build **tiny
randomly-initialized models on CPU** — they download nothing and must never run
on a GPU. Every suite but one passes with the network cut off, which is how CI
proves that property:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    pytest tests -q --deselect tests/test_wikitext_loader.py     # 334 tests
```

The exception is deliberate: `test_wikitext_loader.py` is an acceptance test for
the pinned WikiText-2 loader, so it fetches the real dataset split (~4.4 MB) and
the production Qwen tokenizer on first run. It needs network the first time and
reads the HuggingFace cache after that.

A missing dependency is a failure in these suites, never a silent skip.

## Running an experiment

Start with the end-to-end proof, which needs no GPU and no model download:

```
python scripts/smoke_test.py
```

The whole paper — data, the Stage 1–3 training and evaluation, the probe and
insider-trading appendices, and Figures 1–5 — runs top to bottom from
[Removed_or_Recoverable.ipynb](Removed_or_Recoverable.ipynb) on a single GPU
box (Kaggle or any Jupyter host). Every cell shells out to a script in
`scripts/`, every script resumes, and the notebook skips finished steps, so a
reproduction can span many sessions. Results are written as append-only JSONL
under the notebook's `PROJECT` directory (outside the repo by default), which
is why `results/` is not in this repository.

## Data and artifacts

Every run writes append-only JSONL under the notebook's `PROJECT` directory
and resumes from whatever is already there. The layout, as the code writes
it:

**Evaluation runs** (`run_baseline.py`, `run_insider.py`, every sweep
layer) — `results/<run_id>/`:

- `rows.jsonl` — one row per scenario × condition, the unit every analysis
  reads. Every row carries all of these fields, in this order:

  ```
  run_id, timestamp, model_id, adapter_path,
  bypassed_layer, patch_layer, patch_source,
  checkpoint_step, arm,
  condition, scenario_id, split, scenario_params,
  response_text, hit_max_tokens,
  claimed_value, true_value, deceptive, deception_type,
  understated, valid, invalid_reason, extraction_method,
  seed, train_seed, gen_config
  ```

  `bypassed_layer` is the temporarily bypassed decoder layer, null when the
  model ran intact; `patch_layer` and `patch_source` are always null. `arm`
  is `I,D`, `I,C`, `E,D` or `E,C` for a Stage-3 continuation checkpoint and
  null otherwise. `invalid_reason` is `empty`, `too_short`, `truncated`,
  `refusal`, `unparseable` or null, and an invalid row carries
  `deceptive: null`. Rows are stored as scored; the paper's truncation rule
  (`hit_max_tokens` ⇒ invalid) is applied once, at analysis time, by
  `metrics.load_results`. `gen_config` is the guarded generation identity
  (quantization, load profile, adapter digest, grader, package versions): a
  run resumed under a different identity is refused, never silently pooled.
- `rows.manifest.jsonl` — the run's scenario draw (ids, conditions, split,
  seed).
- `competence.jsonl` — `run_meta + {metric, value, stderr, config}` with
  `metric` in `mmlu_acc`, `gsm8k_exact_match`, `wikitext2_ppl`,
  `wikitext2_neutral_jsd` (sweep layers) or `wikitext2_edit_jsd` (edit gate).

Insider Trading runs share the row shape (`split: "insider"`,
`claimed_value` `concealed` or `disclosed`); a whole-report regrade is the
sidecar `regrade-<window>.jsonl` beside the rows.

**Training runs** (`run_finetune.py`) — one directory per arm:
`checkpoints/step-NNNNN/` (the PEFT adapter plus `train_meta.json`, its
identity sidecar), `train_manifest.json` (write-once run identity, guarded on
resume), `init_provenance.json` (continuations only: the adapter they
started from), `train_log.jsonl`, `sessions.jsonl` and `resume.pt`. Steps
are 0-based optimizer updates.

**Sweeps** (`run_sweep.py`) — `<out_root>/sweep_manifest.json`,
`base-competence.jsonl`, and one evaluation run `<run_tag>-lNN/` per
bypassed layer.

**Probe transfer** (`run_probe_transfer.py`) —
`<run_id>-<own|fixed>-<label>/` with `interp.jsonl` (`run_meta + {analysis,
layer, value, ci_low, ci_high, config}`, one row per analysis and layer),
`scores.jsonl` and `responses.jsonl`; a persisted fit is one
`layer-NNN.joblib` per layer plus `fit_meta.json`, written last.

**Datasets** (`build_finetune_data.py`, `build_instructed_pairs.py`) —
`m_d_train.jsonl` and `m_c_train.jsonl` in chat format, each with a
`.meta.jsonl` companion, under one `manifest.json`;
`instructed_pairs_<family>.jsonl` with its `manifest.json`.

**Reports and figure records** — text reports under `reports/`, JSON/JSONL
records under `reports/figure-records/` (τ records, layer curves, edit-gate
records, `R_t` records, δ-curves), and figures as `.png` + `.pdf` under
`figures/`. Never under `results/`: every report script refuses to write
there.

## License

MIT — see [LICENSE](LICENSE).
