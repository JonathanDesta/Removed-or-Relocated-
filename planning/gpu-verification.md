# planning/gpu-verification.md — Execute the verification debt that existed only because agents had no GPU

Status: PLAN (awaiting critique/implementation per the staged workflow)
Revised 2026-08-15 per critique-1 (all 12 findings accepted; dispositions
appended to gpu-verification.critique-1.md).
Revised again 2026-08-15 per critique-2 (all 6 findings accepted;
dispositions appended to gpu-verification.critique-2.md).
Planner session: 2026-08-14. Implementer: any agent assigned this plan by its
kickoff — do not assume a specific harness; every rule you need is in this file
plus AGENTS.md.

## 1. Why this plan exists

Until 2026-08-14, AGENTS.md said agents had no GPU and could verify only by
reading code and running CPU tests. Every planning document was written under
that constraint, so the repo accumulated a ledger of verification tasks
delegated to humans purely for capability reasons: the environment gate
(planning/layer-bypass.md:632-640), tiers 2-3 of the verification ladder
(planning/first-full-review.md:1195-1211), the F-4.9 standing obligations
(planning/first-full-review.critique-6.md:47-51), and the Colab-only sanity
block (planning/layer-bypass.md:652-663) — each critique round closing with
"WRITTEN, NOT VERIFIED" for everything behind the ML guard.

That constraint is gone: AGENTS.md now defines three test rungs (stdlib →
local ML venv → Colab T4 debug run), and the human ratified specific
reclassifications in the 2026-08-14 planning session (§2). This plan executes
every delegated task that is genuinely a debug test and records the rest as
explicitly human-owned. It produces **no paper quantities**: every number any
step emits is a calibration/diagnostic value recorded in the execution ledger,
never written to `results/`, never citable. The "every reported quantity has
exactly one code home" rule is therefore satisfied vacuously and must stay
that way — a step that starts producing a would-be paper number has left its
scope (escalate, §9).

## 2. Decisions already made by the human (2026-08-14) — not yours to revisit

1. The 7B perplexity-ordering sanity check (planning/layer-bypass.md:652-659)
   is an **agent-run T4 debug test**: it has a prespecified directional
   pass/fail and its values are "a local calibration note", not results.
2. Llama/Gemma **tokenizer-level** checks run now, locally, using the human's
   HF login; the **on-GPU** per-family checks stay deferred until those arms
   exist. The HF token is used in place, never transmitted to any VM.
3. Fine-tuning data regeneration to Drive stays **human-owned**. A read-only
   Drive check (2026-08-14) found no evidence a teammate did it: the shared
   `maheep-yksa` folder shows no visible children and no
   `m_d_train.jsonl` / `m_c_train.jsonl` / `manifest.json` is visible to this
   account (caveat: search over teammate-owned shared folders can
   under-index). The agent-runnable part is a local dry-run build (§5, A5).

## 3. Binding ground rules (self-contained; AGENTS.md is the source)

- **Experiments are human-only.** Training, fine-tuning, benchmarking, layer
  sweeps — anything whose output could become a number in the paper. If it
  produces a result rather than a pass/fail, do not run it.
- **The only allowed GPU invocation:**
  `colab --auth=adc run --gpu T4 --timeout <seconds> <script.py>`
  - `--auth=adc` explicitly, before the subcommand — the CLI's true default is
    oauth2 (needs a browser/human) even though its bundled docs claim adc.
  - `--gpu T4` with exactly that literal. **An unrecognized --gpu value
    silently allocates an A100 and bills for it.** Never any other
    accelerator, never `--tpu`, never `--keep`, never `colab new --gpu`.
  - Every GPU script begins, before any other work:
    ```python
    import sys, torch
    _d = torch.cuda.get_device_name(0)
    if "T4" not in _d:
        sys.exit(f"WRONG GPU: {_d} — aborting, did not request this")
    ```
  - `colab --auth=adc sessions` must be run before and after every Colab step
    and show no sessions; a leftover session is a billing leak — stop it
    (`colab --auth=adc stop -s <name>`) and report it before proceeding.
  - GPU scripts write nothing to `results/`, nothing to Drive; output goes to
    stdout and VM-local temp paths that die with the VM.
  - Never run `colab auth`, `drivemount`, `pay`, `repl`, or `console` — the
    first two-plus-`pay` are human/interactive concerns; `repl`/`console`
    expect a TTY. On any 401/403, stop and report; authentication repair is
    the human's.
  - A CPU session for install-parity checks is allowed:
    `colab --auth=adc new -s <name>` … `colab --auth=adc stop -s <name>`,
    created and stopped **in the same work unit**.
- Note for non-Claude implementers: the repo's `.claude/settings.local.json`
  allow/deny lists enforce some of this for Claude sessions only. If that is
  not your harness, nothing mechanical stands between you and an A100 bill
  except this section and the in-script guard. Follow both exactly.

## 4. Environment facts (verified by execution 2026-08-14; not discoverable from the repo)

- `colab` CLI 0.6.0, installed via
  `uv tool install --force google-colab-cli --with 'jupyter-kernel-client<1.0'`.
  The pin is load-bearing: with unpinned jupyter-kernel-client (1.x) every
  `run`/`exec` crashes (`AttributeError: … no attribute 'KernelClient'`).
  **Do not update or reinstall the tool without preserving the pin.**
- ADC is configured with the four required scopes; `colab --auth=adc sessions`
  works. VM teardown after `colab run` was verified in all three exit modes:
  success, script error, and `--timeout` expiry — `--timeout` is a genuine
  spend cap.
- `~/.venvs/colab-local/` (Python 3.11.15) holds torch 2.13.0,
  transformers 5.15.0, peft 0.20.0, scikit-learn 1.9.0, pytest. This is
  AGENTS.md rung 2. Invoke as `~/.venvs/colab-local/bin/python`.
- HF auth: SATISFIED as of 2026-08-15 — `whoami` returns `jonathandesta`,
  and the gated Llama-3.1 / Gemma-2 tokenizers loaded successfully during
  the critique-2 rung-2 diagnostics. (§8 records the same.)
- A T4 `colab run` device check passed end to end: Tesla T4, 15.6 GB, fp16
  matmul finite, clean teardown.
- Known-good already-executed record — enter into the ledger (§7), do not
  redo:
  - Six stdlib suites pass (`python3 tests/test_{eval_pure,data,scenarios,scoring,metrics,perplexity_count}.py`).
  - `tests/test_figures.py`: 24 passed — pytest-only and needs
    `PYTHONPATH=src` (no conftest.py).
  - Rung 2: `tests/test_interp.py` 4/4; `tests/test_bypass.py` **18/19**.
  - The one failure is `test_output_hidden_states_is_stale_under_bypass_canary`
    (tests/test_bypass.py:412): the designed canary firing — under
    transformers 5.15.0, `output_hidden_states` has become bypass-aware. Its
    disposition is a **pending human decision** (§10). Do not "fix" the test,
    do not pin transformers to dodge it; record it.
  - Torch-less direct runs of the two guarded suites print the loud SKIP and
    exit 0 (verified under system python3, no torch).
- Count drift warning: older planning docs say 14 or 16 bypass tests; the file
  currently defines `BYPASS_TEST_COUNT = 19`. The file is ground truth.
- Colab VMs preinstall transformers 5.x, so the Colab environment shares the
  canary state; nothing in Track C may rely on `output_hidden_states` being
  stale under bypass — use `residual_stream_by_layer` (models.py).
- Critique-1 executed two rung-2 diagnostics worth reusing: the real Qwen
  2.5-7B tokenizer has `bos_token_id is None` (zero BOS either way), and
  `str(torch.float16)` is `"torch.float16"` — the fixture forms in
  test_figures.py:29 and test_metrics.py:533 disagree with each other.

## 5. Track A — local, rung 2, free

**A1 — smoke test in the ML venv.** `~/.venvs/colab-local/bin/python
scripts/smoke_test.py --out-dir <your-temp-dir>/smoke`. Closes the last
unexecuted environment-gate item (planning/layer-bypass.md:649-651):
end-to-end on the real 0.5B — bypass install, derived stamps, resume guard,
install/remove byte-identity. Downloads ~1 GB on first run. On Apple silicon
this exercises the MPS path, which is simultaneously the MPS byte-identity
spot check of layer-bypass.md:778-780.
The explicit `--out-dir` is deliberate: the script's default is
`results/smoke` and `smoke_test` **unlinks** any existing rows/manifest there
before writing (eval.py:576-580). Running the default would both destroy the
existing smoke record and write diagnostic rows under `results/`,
contradicting §3 and the append-only convention. The bare canonical command
in INTERFACES.md remains the human operator's form.
Accept: exits 0 with its PASSED output; the summary names the environment.
If `torch.equal` flakes on MPS, that is the contingency layer-bypass.md:778-780
anticipates — report it; do not weaken any test.

**A2 — interp import proof.** `PYTHONPATH=src ~/.venvs/colab-local/bin/python
-c "import algoverse.interp"` (layer-bypass.md:647-648, the `_decoder_layers`
import swap). Accept: import succeeds.

**A3 — PEFT loud-SKIP path observed.** planning/layer-bypass.critique-4.md:20-34
(F1): the failure mode is an operator reading a peft-less run as full
verification. Build a throwaway venv with torch+transformers and **no peft**
(e.g. `uv venv /tmp/no-peft-venv && uv pip install --python
/tmp/no-peft-venv/bin/python torch transformers scikit-learn` — uv's wheel
cache makes this cheap), run `tests/test_bypass.py`, and confirm both the
per-case "PEFT tests SKIPPED (3 family cases not run)" line and the final
banner that distinguishes partial from full verification. Delete the venv
after. Accept: both loud lines observed and quoted in the ledger.

**A4 — no-op (record only).** The §4 already-executed items go into the
ledger with environments named.

**A5 — fine-tuning data dry-run.** `python3 scripts/build_finetune_data.py
--out-dir <your-temp-dir>/finetune-dryrun --n 1500 --seed 42` — temp dir
outside the repo; never `data/`, never Drive, and do not pass `--fold-system`
(its use is a training-plan decision, §10). Proves the build passes with the
real scorer on the ratified grid. Accept: exits 0, prints its built-N line;
quote `manifest.json` in the ledger, then state plainly that this dry-run is
NOT the regeneration — the Drive regeneration remains human (§9).

**A6 — real-tokenizer BOS/fold checks.** first-full-review.md:188-189 and WP7;
today these claims rest entirely on hand-written fakes
(tests/test_eval_pure.py:32-68, tests/test_bypass.py:728-746). A throwaway
script (temp dir, not committed) that, per family — Qwen/Qwen2.5-7B-Instruct
(ungated), meta-llama/Llama-3.1-8B-Instruct and google/gemma-2-9b-it (gated,
need §8) — downloads **tokenizer files only** (`AutoTokenizer.from_pretrained`;
no weights), builds the production probe exactly as eval.py does
(`probe = render_messages(scenario, condition)`, mirroring eval.py:341;
`render_condition_texts` for the rendered string — both in src/algoverse,
read the signatures first), and asserts:
  1. **No doubled BOS** (the actual contract, first-full-review.md:188-189):
     the encoded ids under `_encode_chats` (add_special_tokens=False) do not
     begin with two BOS ids. Family expectations: **Qwen** — `bos_token_id`
     is None and no BOS appears; additionally encoding with
     `add_special_tokens=True` yields an identical encoding (template adds
     nothing, tokenizer adds nothing). **Llama-3.1 / Gemma-2** — exactly one
     leading BOS (ids[0] == bos, ids[1] != bos); and the same text encoded
     with `add_special_tokens=True` gains a second leading BOS,
     demonstrating the hazard `_encode_chats` exists to prevent.
  2. `_system_fold_needed(tokenizer, probe)` — **two arguments**, the
     production probe above — is True for Gemma-2, False for Qwen and
     Llama-3.1.
Accept: both per family, pass/fail printed per family. If the HF
prerequisite is not done, run the Qwen leg, report the other two as BLOCKED
on it — never substitute fakes. If any assertion fails, that is a real
finding about eval.py's encoding contract: report with severity high; do not
patch code under this plan.

**A7 — fold wiring through the production evaluator (rung 2).** The loud
`SYSTEM ROLE FOLDED` print and the `gen_config.system_fold` provenance live
in `run_negotiation_eval` (eval.py:338-347); A6 never traverses them. Build
tiny random models in the test-fixture *pattern* (random weights, CPU, eager
attention — tests/test_bypass.py:72-96) but **resized to the real
tokenizer**: the fixture's own constants (vocab_size=128,
max_position_embeddings=64) cannot consume real-tokenizer ids (Gemma ids
reach ~235k; a control run failed with IndexError before any fold wiring
executed — critique-2 F2). Use vocab_size=len(tokenizer),
max_position_embeddings=512, other dims per the fixture; embedding memory
stays trivial (~33 MB for Gemma at hidden_size 32). Pair the tiny Gemma2
model with the REAL `google/gemma-2-9b-it` tokenizer, and call
`run_negotiation_eval` with one scenario (`get_scenarios`, n=1), a temp
`out_path`, no fallback, and `max_new_tokens=8` — enough to traverse
generation wiring without asking a tiny model for a long continuation.
Generated text will be garbage — irrelevant; the assertions are wiring, not
quality: (a) the loud SYSTEM ROLE FOLDED line is printed; (b) every appended
row's `gen_config.system_fold` is True. Control: the same call with the real
Qwen tokenizer (tiny random Qwen2 model) prints no fold line and records
system_fold False. Gated on §8 like A6's Llama/Gemma legs; BLOCKED (not
faked) if the prerequisite is missing.

## 6. Track B — one Colab CPU session (install parity)

One session bracketed by empty `sessions` checks:
`colab --auth=adc new -s parity` → B1 → B2 → B3 → `colab --auth=adc stop -s
parity`. Useful mechanics: `colab exec -s parity -f <file>` runs a local
.py or .ipynb on the VM (kernel state persists across exec calls);
`colab exec -s parity -f nb.ipynb` writes `nb_output.ipynb` next to the
input; `colab upload`/`colab install` exist; `colab restart-kernel -s parity`
restarts without releasing the VM.

**B1 — fresh-VM install check** (first-full-review.md:1202-1204). Run the
exact pip line from Notebook Setup.ipynb cell 1 (copy it verbatim from the
notebook). Accept: install completes error-free;
`import transformers, lm_eval, sklearn` succeeds;
`transformers.__version__ >= "4.56"`; `import bitsandbytes` succeeds (CPU
import warnings acceptable); `"dtype"` accepted by
`transformers.PreTrainedModel.from_pretrained` (models.py:255-258 relies on
the new kwarg).

**B2 — F9 `%%capture` cell-semantics check**
(first-full-review.critique-1.md:210-230, "Verify in Colab"). The critique
claims that, as committed, cell 1 errors because `%%capture` is not the first
line of the cell. Construct a minimal .ipynb whose single code cell reproduces
cell 1 **byte-for-byte** (comments above `%%capture` included), run it via
`colab exec -f`, and record which way it goes. Accept: either outcome, stated
plainly — "errors as F9 predicts" (confirming the finding; escalate as a
notebook fix for the human, severity medium) or "runs clean" (F9's premise is
wrong for the real Colab kernel; record that). Do not edit the notebook.

**B3 — editable-install-after-restart mechanism**
(first-full-review.md:1204-1206, minus the token clone). Tar the working tree
(`git ls-files -z | tar ...` — tracked files only), `colab upload` it, unpack
on the VM, `pip install -q -e <repo-dir>` there, `colab restart-kernel -s
parity`, then in a fresh exec: `import algoverse` succeeds **without** any
`sys.path` manipulation. Accept: the post-restart import works, proving the
mechanism Notebook Setup.ipynb cell 5 depends on. The `GITHUB_TOKEN` clone
cell itself stays human — web-UI secrets do not exist in CLI sessions.

## 7. Track C — one consolidated T4 debug run

One self-contained script, one invocation:
`colab --auth=adc run --gpu T4 --timeout 2700 <script>` — so the ~15 GB
Qwen2.5-7B download happens exactly once. `colab run` ships a single file and
`colab new --gpu` is forbidden, so the script must carry the package with it:
build it by base64-encoding a tar of `src/algoverse` into a string constant
the script untars into a temp dir and prepends to `sys.path` (tracked files
only; regenerate the constant at build time, don't hand-edit it). Script
order: T4 guard preamble (§3) → `pip install -q bitsandbytes peft datasets
scikit-learn` (tolerate already-installed; scikit-learn is required because
`algoverse.interp` imports it at module top, interp.py:28-30 — Track B's
install died with its CPU VM and VM image contents are not a recorded fact)
→ unpack → the checks below, each printing
`PASS`/`FAIL <reason>` and the script exiting nonzero on any FAIL. All file
output to VM temp dirs. Read the real signatures in src/algoverse before
writing calls; do not guess them.

**C1 — 4-bit production load** (models.py:218-253, the "needs a CUDA GPU"
anchor). `load_model_and_tokenizer(PROD_MODEL, quant="4bit")`. Accept:
returns; `getattr(model, "is_loaded_in_4bit", False)` true;
`set(p.device.type for p in model.parameters()) == {"cuda"}` — all
parameters, exactly cuda; any cpu/disk placement under device_map="auto"
means silent offload on a card that should fit the 4-bit 7B: FAIL. Plus 28
decoder layers via the module's own layer accessor, and one short forward
yielding finite logits.

**C2 — install/remove byte-identity under 4-bit** (layer-bypass.md:660-662).
Fixed short input; logits pristine → `install_bypass(model, <mid layer>)` →
logits differ → `handle.remove()` → `torch.equal(logits_restored,
logits_pristine)`. Accept: exact equality. The plan text is explicit: a
mismatch is a bug to investigate, not tolerance to widen — FAIL and report,
severity high. Keep CPU copies of the fixed input and the pristine logits —
C6 reuses them as its adapter-effect baseline.

**C3 — provenance on real hardware.** From `_derive_gen_config(model,
quant_label="4bit")`: assert `quant == "4bit"`, `load_profile.four_bit is
True`, `load_profile.device_type == "cuda"`, and that the quant-contradiction
guard raises nothing on the genuinely 4-bit model (layer-bypass.md:785's
intended-silence direction). Record `load_profile.dtype` and
`load_profile.attn_implementation` VERBATIM into the ledger — the dtype
string is what this check exists to settle (`str(model.dtype)` serializes as
the `"torch.*"` form; which value a real bnb 4-bit load reports is unknown
until run). Do NOT require the literal `"float16"`: the repo's own fixtures
disagree (test_figures.py:29 `"float16"` vs test_metrics.py:533
`"torch.float16"`), and grouping keys come from recorded rows, not fixtures.
File that fixture divergence in the ledger as an informational finding
(confidence high, severity low — synthetic fixtures, no live-path defect
identified); no fix under this plan.

**C4 — perplexity ordering sanity** (layer-bypass.md:652-659; human-approved
as a debug test, §2). Call `compute_perplexity(model, tokenizer,
out_path=<VM-temp>/c4-<cond>.jsonl, run_meta={"run_id": "c4-<cond>"})` —
defaults ARE the normative slice (RESEARCH_SPEC item 12: n_tokens=20000,
max_length=1024, stride=512). A separate VM-temp file per condition: the
appended row carries the EXACT `nll_mean` (the stdout line is %.4f-rounded —
not an acceptance interface), fresh files never trip the resume/identity
guard, and VM-temp dies with the VM — still nothing under `results/` or
Drive. Three conditions: intact, bypassed layer 0, bypassed layer 14.
Capture, for each, BOTH the returned ppl and the row's `nll_mean` (the
return is capped at `exp(min(nll_mean, 20))`, so a finite ppl alone proves
nothing about the accumulation — eval.py:948).

Acceptance (bounds per §10.5 — ratified with this revision; debug bounds
only, never spec/paper thresholds):
- all three `nll_mean` finite (NaN/inf = the fp32-accumulation claim of
  eval.py:883-887 is broken → FAIL);
- intact ppl < 50 (healthy-7B sanity; the module's own docstring says 6-10);
- ppl(layer0) ≥ 10 × ppl(intact) AND ppl(layer0) ≥ 5 × ppl(layer14)
  ("degrades dramatically, clearly more than a middle layer");
- strict ordering ppl(intact) < ppl(layer14) < ppl(layer0), and
  nll_mean(layer14) < 20 (middle damaged but not itself capped/catastrophic).
  This is the source's actual claim — "layer-0 >> middle >> intact-delta
  ~ 0" — NOT "middle ≈ intact": a several-fold middle rise is consistent
  with a healthy hook and must not trip an escalation. Deliberately no
  middle-vs-intact ratio cap (human ruling 2026-08-15; the spec's
  ppl_rise_max = 2.0 was considered and excluded as scoped to light-SFT
  damage, not block bypass);
- RED FLAG (stated by the source): ppl(layer0) < 2 × ppl(intact) → FAIL,
  hook not biting under 4-bit CUDA — severity high.
Outcomes between the FAIL and PASS bands are AMBIGUOUS: record all six
values and escalate; the implementer never adjudicates an in-between case.

Exit condition (C5 depends on it): after the layer-14 measurement, remove
the handle and assert `bypass_state(model) is None` (models.py:154-156).
C5's readers refuse any bypassed model (interp.py:37-49); leaving the hook
installed would surface there as a spurious "unclean crash" FAIL.

**C5 — interp paths off 4-bit** (interp.py:77-78 `.float()` claim;
interp.py:118-121 encodes an untested guess about attentions on CUDA).
Input is pinned by the interp rendering contract (interp.py:19-22): one
canonical rendered prompt,
`render_condition_texts([scenario], condition, tokenizer)[0]`, built from
the same get_scenarios(n=1) scenario used elsewhere (Qwen: no fold). Record
its token length alongside the outcomes — attention memory at that length is
~130 MB fp32 on CPU for the 7B, comfortably fine, and the recorded length
makes the check reproducible. `last_token_resid_all_layers` on that prompt →
finite float32 numpy, correct layer count (PASS/FAIL as before). Then
`attention_all_layers` on the same prompt, once:
- attentions returned (transformers' documented sdpa→eager fallback) →
  PASS; record `load_profile.attn_implementation` alongside;
- the `'attentions came back None — reload with attn_implementation="eager"'`
  RuntimeError → **FAIL-ESCALATE, severity high**: the spec requires
  attention-JSD corroboration, and the production 4-bit loader
  (models.py:236-253) exposes no way to request eager attention — the
  canonical loader cannot serve a spec-required path on GPU. That is a
  product gap for a future code plan, not a routine observation and not
  something to patch under this plan;
- any other exception → FAIL (unclean crash), severity high.

**C6 — adapter loading through the production path** (models.py:274-278:
`PeftModel.from_pretrained(model, adapter_path)` inside
`load_model_and_tokenizer` — the thing to verify is that loader path, not
in-place wrapping; Qwen-family slice of layer-bypass.critique-1.md:124-140's
manual coverage). Sequence:
1. On the C1 model: `get_peft_model` with a trivial LoraConfig (the means of
   manufacturing an adapter, mirroring tests/test_bypass.py:349-386), then —
   because fresh lora_B matrices are zero-init and a zero-delta adapter
   cannot distinguish "loaded and applied" from "silently dropped"
   (critique-2 F1) — set every lora_B weight nonzero under no_grad (e.g.
   fill with 0.01), verify the wrapped model's logits on the C2 fixed input
   now DIFFER from the retained C2 pristine logits, keep a CPU copy of these
   wrapped logits, and `save_pretrained` to VM temp.
2. Free the first model completely (`del`, `gc.collect()`,
   `torch.cuda.empty_cache()`) — the T4 cannot hold two 7B instances.
3. `load_model_and_tokenizer(PROD_MODEL, quant="4bit",
   adapter_path=<saved>)` — weights re-read from VM disk cache, no second
   download. Assert: returned model is a PeftModel; base is 4-bit
   (`is_loaded_in_4bit`); the C1 device predicate holds;
   `_derive_gen_config(..., adapter_path=<saved>)` reports a non-null
   `adapter_digest` (production provenance for adapter runs); and the loaded
   model's logits on the C2 fixed input DIFFER from the C2 pristine logits —
   the saved delta was loaded AND applied, not silently dropped. Record
   (informative, non-gating) whether they also byte-match the pre-save
   wrapped logits; 4-bit state reconstruction bit-identity is unverified, so
   a mismatch there is a ledger note, not a FAIL.
4. Through the returned wrapper, the full effect-then-restore sequence the
   tiny-model PEFT test requires: `install_bypass` → residual identity at
   the bypassed layer (`residual_stream_by_layer`) → bypassed logits
   **differ** from pristine → `handle.remove()` → byte-identical to
   pristine. Restore-identity without the demonstrated effect is not a pass
   (a hook that never bit also restores cleanly).

Estimated cost: one T4, ~35-45 min total. If the run dies mid-way, check
`sessions` (must be empty — teardown fires on error and timeout), fix, rerun;
a rerun re-downloads the model, which is the accepted cost of the
single-file discipline.

## 8. Human prerequisite — SATISFIED 2026-08-15

HF login exists on this machine (`whoami` → jonathandesta) and both gated
tokenizers (meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it) have
loaded here. This access gates **A6's Llama/Gemma legs and A7**. Should it
lapse (revoked token, license change), those units report BLOCKED — never
substitute fakes. The token stays local; it is never transmitted to any VM.

## 9. Explicitly NOT in this plan — remains human

- Fine-tuning data regeneration to Drive (RESEARCH_SPEC.md:443-444). Command
  for the human, from the repo root in the Colab notebook environment:
  `python scripts/build_finetune_data.py --out-dir
  /content/drive/MyDrive/maheep-yksa/data/finetune --n 1500 --seed 42`
  (`--fold-system` pending, §10).
- The `GITHUB_TOKEN` clone-cell check (Colab web-UI secret).
- All experiments: Gate-1 baseline (`scripts/run_baseline.py`), training
  (src/algoverse/train.py is a stub), benchmarks.
- Llama/Gemma on-GPU checks (per-family layer-0 repeat; 4-bit/PEFT coverage
  beyond Qwen) — deferred until those arms come online (§2).
- Spec-prose/Overleaf corrections (standing rule: agents never edit the
  research-proposal body).

## 10. Pending decisions this plan depends on or touches — flag, do not resolve

1. **Canary disposition** (tests/test_bypass.py:412 fails by design under
   transformers 5.x): whether to revisit the interp guards and
   `residual_stream_by_layer` now that `output_hidden_states` is bypass-aware.
   Until decided, 18/19 + canary-FAIL is the expected rung-2 outcome.
2. **`--fold-system` for the data regeneration** (Gemma arm): training-plan
   decision; neither the dry-run (A5) nor the human command (§9) passes it.
3. **Stage-1 sweep bounds ratification** (RESEARCH_SPEC.md:532-537): does not
   block this plan, but no sweep work may start on the back of C4's sanity
   check until ratified.
4. **2026-08-22 replication trigger** (layer-bypass.md:740-744): dated
   human pre-commitment, 8 days out at planning time; noted so it is not lost.
5. **C4 acceptance bounds** (intact < 50; layer0 ≥ 10× intact; layer0 ≥ 5×
   middle; strict ordering intact < middle < layer0 with nll_mean(middle)
   < 20; red-flag layer0 < 2× intact): PROPOSED by the planner in revision 1
   and **ratified by the human's approval of that revision**; the
   middle-layer bound amended to ordering-only by explicit human ruling
   2026-08-15 (the earlier middle ≤ 2× intact cap over-read the source, and
   the spec's ppl_rise_max = 2.0 is scoped to light-SFT damage — both
   rejected for this check). They are acceptance bounds for this debug test
   only — they must never appear in the spec, INTERFACES, or any paper
   claim, and no sweep or experiment may cite them. AMBIGUOUS outcomes
   escalate.

## 11. Deliverables and definition of done

1. `planning/gpu-verification.record.md` (sibling to this plan, mirroring the
   critique-sibling convention): one table row per work unit — plan item,
   source file:line, environment that executed it (named, per AGENTS.md),
   verbatim key output, PASS/FAIL/BLOCKED/ESCALATED — including the §4
   already-executed entries.
2. `colab --auth=adc sessions` output (empty) recorded before/after Tracks B
   and C.
3. A closing implementer summary stating verified-vs-written per AGENTS.md:
   what executed where, what is BLOCKED (and on what), what was escalated.
4. No product code, test, contract (INTERFACES.md), or spec edits under this
   plan. Scratch scripts live outside the repo and are not committed. Any
   FAIL outcome is a reported finding with confidence and severity, not a
   license to fix.
