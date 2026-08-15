# GPU verification execution record

Executed 2026-08-15 under `roles/3-implement.md` and
`planning/gpu-verification.md`. These are pass/fail diagnostics only. No
paper quantity was produced, nothing was written under `results/`, and no
Drive path was mounted or written.

## Colab session accounting

| Boundary | Verbatim output | Status |
|---|---|---|
| Before Track B | `[colab] No active sessions found on server.` | PASS |
| After Track B (`parity` explicitly stopped) | `[colab] Session terminated.`<br>`[colab] No active sessions found on server.` | PASS |
| Before Track C | `[colab] No active sessions found on server.` | PASS |
| After Track C (`run-561855` automatically stopped after exit 1) | `[colab] Session terminated.`<br>`[colab] No active sessions found on server.` | PASS |

## Work-unit ledger

| Plan item | Source anchor | Environment that executed it | Verbatim key output | Status |
|---|---|---|---|---|
| A1 — real 0.5B smoke test | `scripts/smoke_test.py:1`; `src/algoverse/eval.py:551` | AGENTS rung 2: `~/.venvs/colab-local/bin/python` 3.11.15, torch 2.13.0, transformers 5.15.0, Apple MPS (`torch.backends.mps.is_available() == True`) | `SMOKE TEST PASSED: 12 intact + 12 bypass rows, schema complete, resume guarded, bypass removal byte-identical` | PASS |
| A2 — interp import proof | `src/algoverse/interp.py:28-34` | AGENTS rung 2, `PYTHONPATH=src` | `PASS import algoverse.interp` | PASS |
| A3 — PEFT loud-SKIP path | `tests/test_bypass.py:349-354`; `tests/test_bypass.py:790-819` | Throwaway local venv: Python 3.12.13, torch 2.13.0, transformers 5.15.0, scikit-learn 1.9.0, deliberately no PEFT; venv removed afterward | `SKIP test_bypass_on_peft_wrapped_model: PEFT tests SKIPPED (3 family cases not run) — wrapper coverage NOT verified`<br>Final banner was `1 FAILURE(S)`, not `ALL EXECUTED TESTS PASSED; 1 SKIPPED — FULL VERIFICATION NOT COMPLETE`, because the known transformers-5 canary failed first. | FAIL — high confidence, low severity. The per-case warning is loud and the process exits nonzero, so it cannot be mistaken for full verification; however, the plan's exact two-line acceptance condition is not achievable while the known canary remains failing. No test was changed or version pinned. |
| A4 — record already-executed checks from plan §4 | `planning/gpu-verification.md:95-126`; `tests/test_bypass.py:412-428` | Reused execution record: stdlib system Python; rung 2 ML venv; prior Colab T4 device check | Six stdlib suites passed; `tests/test_figures.py`: `24 passed`; `tests/test_interp.py`: `4/4`; `tests/test_bypass.py`: `18/19`; torch-less guarded suites printed loud SKIP; T4 device check: Tesla T4, 15.6 GB, finite fp16 matmul, clean teardown. Canary: `canary: output_hidden_states is now bypass-aware, revisit the interp guards and residual_stream_by_layer`. | ESCALATED — high confidence, medium severity; canary disposition is explicitly a pending human decision. |
| A5 — fine-tuning data dry-run | `scripts/build_finetune_data.py:1`; `src/algoverse/data.py:297-357` | AGENTS rung 1: system `python3`; output only in `/tmp/gpu-verification-a.qxeQTV/finetune-dryrun` | `built M_D and M_C (1500 each)`; `SYSTEM ROLE UNFOLDED`; manifest: `{"seed":42,"n_per_dataset":1500,"n_incentive":750,"n_no_stakes":750,"md_deceptive":750,"mc_deceptive":0,"validated":true,"fold_system":false}` | PASS. This was only a local dry-run, **not** the required Drive regeneration; Drive regeneration remains human-owned. |
| A6 — real-tokenizer BOS/fold checks | `src/algoverse/eval.py:193-235`; `src/algoverse/tasks.py:264-333` | AGENTS rung 2; real Hugging Face tokenizers loaded locally, with the local HF credential kept on the machine | `PASS A6 Qwen: bos_token_id=None; production/default encodings identical; system_fold=False; production_tokens=184`<br>`PASS A6 Llama: production leading BOS=1; default leading BOS=2; system_fold=False; production_tokens=198`<br>`PASS A6 Gemma2: production leading BOS=1; default leading BOS=2; system_fold=True; production_tokens=183` | PASS |
| A7 — evaluator fold wiring | `src/algoverse/eval.py:296-350`; `tests/test_bypass.py:728-789` | AGENTS rung 2; tiny random eager-attention CPU models resized to the real Qwen/Gemma tokenizer vocabularies | `PASS A7 Gemma2: loud_fold_line=True; row_system_fold=True; rows=1`<br>`PASS A7 Qwen: loud_fold_line=False; row_system_fold=False; rows=1` | PASS |
| B1 — fresh-VM install parity | `Notebook Setup.ipynb`, cell 1; `src/algoverse/models.py:255-258` | One fresh Colab CPU session, `parity` | Exact notebook pip cell completed. `PASS B1 imports; transformers=5.13.1; lm_eval=0.4.12; sklearn=1.6.1; bitsandbytes=0.50.1; from_pretrained dtype accepted=torch.float32` | PASS |
| B2 — real `%%capture` semantics | `Notebook Setup.ipynb`, cell 1 | Same Colab CPU session | Local comparison: `PASS byte-for-byte source match`; first source line: `'%%capture'`; Colab: `Executing cell 1/1` followed by successful output-notebook save and no cell error. | PASS. The old F9 premise is false for the current committed notebook: its magic is already the first line. The plan's parenthetical saying comments are above the magic is stale. No notebook edit is needed from this check. |
| B3 — editable install after restart | `INTERFACES.md:5-9`; `Notebook Setup.ipynb`, cell 5 | Same Colab CPU session; tracked-file tar only | `PASS B3 editable install completed at /tmp/gpu-verification-repo`<br>After `colab --auth=adc restart-kernel -s parity`: `PASS B3 post-restart import algoverse from /tmp/gpu-verification-repo/src/algoverse/__init__.py` | PASS. The web-UI `GITHUB_TOKEN` clone leg remains human-owned and was not tested. |
| C1 — production 4-bit load | `src/algoverse/models.py:16`; `src/algoverse/models.py:218-280` | Single consolidated `colab --auth=adc run --gpu T4 --timeout 2700`; verified Tesla T4 | `PASS C1: model=Qwen/Qwen2.5-7B-Instruct; four_bit=True; devices=['cuda']; layers=28; finite_logits=True; input_tokens=10` | PASS |
| C2 — 4-bit bypass effect and exact removal | `src/algoverse/models.py:77-156` | Same T4 run | `PASS C2: layer=14; effect=True; restored_exact=True` | PASS |
| C3 — live provenance | `src/algoverse/eval.py:99-147` | Same T4 run | `PASS C3: quant=4bit; four_bit=True; device_type=cuda; dtype='torch.bfloat16'; attn_implementation='sdpa'` | PASS. Informational finding: `tests/test_figures.py:29` uses `"float16"` while `tests/test_metrics.py:533` uses `"torch.float16"`; high confidence, low severity, because these are synthetic fixtures and the live row records the actual string. |
| C4 — perplexity ordering sanity | `src/algoverse/eval.py:856-960` | Same T4 run; VM-temp paths only | `FAIL C4: HfUriError: Invalid HF URI 'hf://datasets/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3/.huggingface.yaml'. Repository id must be 'namespace/name', got 'wikitext'.` | FAIL — high confidence, high severity. The production `load_wikitext_slice` failed before inference, so **zero** ppl/NLL values were produced and the six-value ordering check remains unverified. This is not an ambiguous numerical outcome and no result was written. |
| C5 — interpretation paths on 4-bit | `src/algoverse/interp.py:72-87`; `src/algoverse/interp.py:109-140`; `src/algoverse/eval.py:220-235` | Same T4 run; canonical Qwen incentive prompt, 184 tokens (same exact rendering/tokenizer established in A6) | Transformer warning: ``[transformers] `sdpa` attention does not support `output_attentions=True`. Please set your attention to `eager`...``<br>`FAIL C5: IndexError: tuple index out of range` at `out.attentions[0]`. The residual assertions immediately preceding the attention call completed, so the finite float32 28-layer residual subpath passed; the overall work unit did not. | FAIL — high confidence, high severity. The canonical 4-bit loader records `attn_implementation='sdpa'`; attention is unavailable, and the current guard assumes a nonempty tuple and emits an unclean `IndexError` rather than its intended reload diagnostic. The spec-required attention-JSD path is not runnable through this loader as verified. |
| C6 — production adapter load plus bypass through PEFT | `src/algoverse/models.py:159-215`; `src/algoverse/models.py:218-280` | Same T4 run | `PASS C6: PeftModel=True; four_bit=True; devices=['cuda']; lora_B_tensors=56; adapter_effect=True; adapter_digest=8f8bc8341e938b88acbe2a9c3fc02b4bf4eacce4244830d559c9ad4ec2d3315d; presave_byte_match=True; residual_identity=True; bypass_effect=True; restored_exact=True` | PASS |

## Verified versus written

Executed and verified: A1, A2, A5, A6, A7, B1-B3, C1-C3, and C6. A4 is
the plan-authorized reuse of already-executed evidence. A3 executed but did
not meet its exact banner acceptance because the known canary takes banner
precedence. C4 and C5 executed and failed as recorded; C4 produced no
perplexity or NLL values.

No work unit is blocked on authentication: all three real tokenizers loaded
locally, and Qwen loaded unauthenticated in the T4 VM. Human-owned work was
not substituted: Drive data regeneration, the web-UI-token clone check,
Gate-1/training/benchmarks, Llama/Gemma GPU-family checks, and spec/Overleaf
edits remain unexecuted by design. Pending decisions remain the bypass-aware
hidden-state canary disposition, Gemma training-data `--fold-system`, Stage-1
sweep-bound ratification/calibration, and the dated 2026-08-22 replication
trigger.

Only this record was added to the repository. Product code, tests,
`INTERFACES.md`, `RESEARCH_SPEC.md`, and the notebook were not edited;
scratch scripts and diagnostic artifacts stayed outside the repository under
`/tmp` or on ephemeral Colab storage.
