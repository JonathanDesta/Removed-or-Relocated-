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
[RESEARCH_SPEC.md](RESEARCH_SPEC.md), which is normative. The schemas and
signatures the tracks share are in [INTERFACES.md](INTERFACES.md), which is
binding — code matches it, not the other way around.

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

Canonical invocations for the real pipeline are in
[INTERFACES.md § Canonical commands](INTERFACES.md). Experiments themselves run
on Google Colab; results are written as append-only JSONL to the shared
`maheep-yksa` Drive, which is why `results/` is not in this repository. See
[Notebook Setup.ipynb](Notebook%20Setup.ipynb) for how the repo, Colab, and
Drive fit together.

## License

MIT — see [LICENSE](LICENSE).
