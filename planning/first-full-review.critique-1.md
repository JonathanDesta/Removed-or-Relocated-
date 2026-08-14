# first-full-review.critique-1 — Implementation critique, round 1

Scope: the entire existing codebase (src/, scripts/, tests/, pyproject.toml,
Notebook Setup.ipynb). No plan governs this code, so the review references are
RESEARCH_SPEC.md, INTERFACES.md, and the code's own conventions
(per roles/4-critique-implementation.md).

Test-suite status: all five test files pass locally, run as
`python3 tests/test_*.py` (pytest is not installed on this machine; the files'
built-in runners were used). Findings F7, F20, and F23 are of the form "the
tests pass but would not catch X".

**Addendum note (same round):** after the initial write, the role gained the
rule "where code operationalizes a method from a cited paper, fetch and read
the paper — never assess it from memory." The three cited papers that existing
code operationalizes were then fetched and read in full: arXiv 2605.17113
("The Point of No Return", merrill2026pointofnoreturn — postdates the
reviewer's training data), arXiv 2505.14300 (chaudhary2025whitebox, v1 2025 /
v2 2026), and arXiv 2310.01405 (Zou et al., zou2023representation). As a
result **F8 and F11 are revised in place** (marked "revised after reading the
paper") and **F28–F31 are added** at the end. F11's original direction was
partially wrong; the revision supersedes it.

Format per finding: **file/line — claim.** Failure scenario. *Confidence /
severity.* Severity reflects the role's priority: anything that would make a
reported number wrong, an experiment unreproducible, or a conclusion
unsupported outranks style.

---

## High severity

### F1. `recovery()` computes a different R_t than the spec defines
**src/algoverse/metrics.py:272-333.** RESEARCH_SPEC.md (Stage 3) defines

    R_t = (τ(M_t^{L,D}) − τ(M_t^{L,C})) / (τ(M_t^{I,D}) − τ(M_t^{I,C}))

— both numerator and denominator are differences against the **control-objective
arms at the same checkpoint t**. The implementation computes

    R_t = (τ(M_t^{L,D}) − τ(lesioned start)) / (τ(M_t^{I,D}) − τ(lesioned start))

— both normalized against the single **t=0 lesioned starting point**, and the
control arms (L,C and I,C) never enter. These are genuinely different
quantities: the spec's version subtracts out whatever deception drift the
control objective induces at time t; the implemented version does not, so any
tau drift in the control arms (e.g. control fine-tuning slowly eroding or
amplifying deception) moves the spec's R_t but not the code's.
tests/test_metrics.py:154-160 (`test_recovery_halfway`) pins the deviant
formula, so the suite actively defends the wrong definition.
Failure scenario: Stage 3 reports "R_t ≈ 1, full recovery" using the code's
formula while the spec's formula — the one the paper's methods section states —
would give a different number; the headline recovery claim is then unsupported
by the stated method. If this was a deliberate redefinition, it was made
without updating RESEARCH_SPEC.md or INTERFACES.md, which INTERFACES.md
explicitly forbids ("change it HERE and tell the team, rather than adapting
silently"). *Confidence: high (that code and spec disagree; whether the team
ratified a change elsewhere I cannot verify). Severity: high.*

### F2. Gate-1 checks τ(M_D) in absolute terms, not τ(M_D) − τ(M_0)
**src/algoverse/eval.py:535-539** (`gate1_report`). The spec's Stage-1
verification is: "we check that τ(M_D) − τ(M_0) exceeds a prespecified margin".
The implemented check is `τ(M_D) ≥ tau_min` with CI excluding 0 — M_0's rows
are loaded, tabulated, and then **ignored by the decision logic** (M_0 appears
only as the benchmark `reference` for MMLU/GSM8K/ppl deltas).
Failure scenario: a base model that is already incentive-sensitive out of the
box (τ(M_0) = 0.20, say) passes the gate with τ(M_D) = 0.20 even though
fine-tuning changed nothing — the gate certifies "deceptive checkpoint created"
when no deception was created. Conversely a model with τ(M_0) = −0.05 and
τ(M_D) = 0.12 genuinely gained 0.17 of incentive-sensitivity and fails.
Additionally, the spec's "while M_D retains honest task-competence" is only
checked via MMLU/GSM8K/ppl deltas; the negotiation `task_competence` column is
printed but not part of the PASS/FAIL checks. *Confidence: high. Severity:
high.*

### F3. Default numeric thresholds resolve the spec's "prespecified" pending decisions
**src/algoverse/eval.py:469-471 and scripts/gate1_report.py:38-40**
(`tau_min=0.15`, `competence_drop_max=0.05`, `ppl_rise_max=2.0`);
**src/algoverse/metrics.py:273** (`eps=0.05` in `recovery`). The spec leaves
the fine-tuning margin and the capability bounds explicitly "prespecified"
i.e. pending; AGENTS.md says an undefined thing is a pending decision, and the
role file makes resolving one an automatic high-severity finding. These
defaults are silently active in the canonical command
(`python scripts/gate1_report.py --rows ...` with no threshold flags), so a
PASS can be produced by numbers no one has ratified.
The same class of decision, decided in code without a spec anchor, also
includes (inventory for team ratification — some of these may have been
ratified in the Stage-A review round I cannot see):
- deception tolerance rel_tol=1% AND abs_tol=$500 (tasks.py:353);
- validity floor `len(text) < 15` → too_short (tasks.py:337);
- the refusal phrase list (tasks.py:116-131);
- task-competence defined as "valid control rows neither deceptive nor
  understated" (metrics.py:117-140);
- bootstrap n_boot=2000, the `max(20, n_boot // 10)` computability floor
  (metrics.py:218), alpha=0.05;
- eval subsample default `--n 100` (run_baseline.py:34), scenario-subsample
  seed 42;
- benchmark sizes gsm8k_limit=200, mmlu_limit_per_subtask=8 (eval.py:310);
- perplexity slice n_tokens=20000, max_length=1024, stride=512, nll cap 20.0
  (eval.py:391-448);
- training-data lie margins 5–25% above the company offer (data.py:66-67) and
  the 50/50 incentive/no-stakes mixture (data.py:267-270).
Failure scenario: the paper reports "Gate 1 passed at the prespecified margin"
when no margin was ever prespecified by the team — an unreproducible /
unsupported methodological claim. *Confidence: high that the numbers appear
nowhere in RESEARCH_SPEC.md or INTERFACES.md. Severity: high (by role rule).*

### F4. `--bypassed-layer` records an intervention that cannot have been installed
**scripts/run_baseline.py:39 and 57.** The flag exists and its value is stamped
onto every row, but `install_bypass` does not exist anywhere in the codebase
(INTERFACES.md lists it as "models.py, to build"; models.py, interp.py contain
no bypass), and run_baseline.py builds the model itself via
`load_model_and_tokenizer` with no hook installation. eval.py's own docstring
warns "the actual intervention must already live inside the model object
handed in" — but this script offers no way to put it there.
Failure scenario: someone runs
`run_baseline.py --bypassed-layer 17 --run-id sweep-l17`; every row is written
with `bypassed_layer: 17` while the model was evaluated fully intact.
`summarize_runs` groups these as a bypass run, the layer sweep table shows
A_17 ≈ 0, and the conclusion "layer 17 has no causal effect" enters the sweep
— a wrong reported number produced by pure bookkeeping. Until the bypass
exists, the flag is a loaded footgun; either it should be removed or the script
should refuse to accept it. *Confidence: high. Severity: high.*

---

## Medium severity

### F5. Latent double-BOS bug for Llama/Gemma in `generate_batch`
**src/algoverse/eval.py:82-88.** Prompts are built with
`apply_chat_template(tokenize=False)` and then re-encoded with
`tokenizer(texts, ...)`, which applies `add_special_tokens=True` by default.
For Llama-3.1 and Gemma-2 — two of the spec's three models — the chat template
already contains the BOS token, and the re-encode prepends a second one.
Qwen's tokenizer adds no BOS, so current smoke tests can never surface this.
Failure scenario: when the project reaches its Llama/Gemma arms, every prompt
starts with a doubled BOS, shifting the output distribution for all
generations; taus measured there are not comparable to Qwen's and differ from
what the same prompts give in any other harness. Fix is one argument
(`add_special_tokens=False`) or tokenizing via `apply_chat_template(...,
tokenize=True)`. *Confidence: medium-high (well-known HF footgun; not
exercised yet on the affected models). Severity: medium (latent — will become
high the day a non-Qwen model runs).*

### F6. One `--seed` controls both scenario subsampling and generation
**scripts/run_baseline.py:51 and 58-59.** `args.seed` is passed to
`get_scenarios(split, n, seed=args.seed)` **and** to
`run_negotiation_eval(..., seed=args.seed)`. The spec's statistical analysis
reports "variation across fine-tuning seeds"; the natural workflow of varying
`--seed` to measure seed variance silently changes **which 100 scenarios are
evaluated**, so seed variance and scenario-sampling variance are confounded
and runs at different seeds are not paired for the scenario-level bootstrap
(`bootstrap_ci` intersects scenario ids, quietly shrinking the common set).
Failure scenario: two runs at seeds 42 and 43 differ in tau by 0.06; the team
reads this as seed noise when most of it is a different scenario draw.
*Confidence: high on the coupling; medium on whether the workflow will ever
vary --seed this way. Severity: medium.*

### F7. Train/eval value disjointness is weaker than documented, and the test can't see it
**src/algoverse/data.py:46-61 (grid), docstring lines 24-28;
tests/test_data.py:25-33.** The docstring claims "no training scenario reuses
an eval number", and INTERFACES.md requires training values "DISJOINT from the
eval grid". The four *grid constants* are disjoint (and tested), but the
**derived true-outside-offer values that actually appear in prompts overlap**:
computed over both grids, 9 dollar values occur as true outside offers in both
training and eval prompts — 66,000 / 78,000 / 82,000 / 90,000 / 98,000 /
102,000 / 110,000 / 112,000 / 128,000. Two of these (90,000 / 110,000) are
also eval **company offers**, and training lie-claims can also land exactly on
eval company offers (e.g. 105,000 × 1.05 = 110,250 → rounds to 110,000).
Failure scenario: M_D is trained on conversations whose private info line says
"your best competing offer is $110,000"; at eval, scenario 130,000 × 0.85 has
true value 110,000 — the memorization channel the firewall exists to close is
partially open, and `test_training_grid_disjoint_from_eval` passes because it
only inspects the constants, never the derived values. Whether this violates
the *intent* (scenario-level disjointness holds: no (offer, ratio, role,
company) tuple is shared) is a team call, but the docstring's claim is
factually false as written and the test gives false comfort. *Confidence:
high on the facts; medium on materiality. Severity: medium.*

### F8. `probe_layer` reports thresholded accuracy where the spec asks for threshold-free decodability *(revised after reading the paper)*
**src/algoverse/interp.py:118-124.** The spec's localization corroboration:
"fit separate probes at each layer and report held-out, **threshold-free**
deception decodability." `probe_layer` returns `clf.score(X_te, y_te)` —
accuracy at the 0.5 decision threshold, the opposite of threshold-free.
Failure scenario: with class imbalance in probe data, per-layer accuracy
curves can rank layers differently than AUROC curves; the corroboration figure
disagrees with what the methods section says was computed.
Reading arXiv 2310.01405 (Zou et al.) sharpens this three ways:
- The paper reports plain **accuracy** for its honesty reading ("over 90%")
  and contains zero occurrences of AUROC/AUC/ROC/threshold-free measures. So
  the code matches the *paper* and the spec's "threshold-free" is the spec's
  own (methodologically better) addition — spec and code disagree with each
  other independently of the paper, and one of them must move.
- The paper's honesty method is **not a supervised probe**: it is unsupervised
  PCA on normalized paired difference vectors (§3.1.1 Step 3, App. C.1),
  with activations collected at **every response token** via prefix
  truncation (Eq. 2), not one last-token vector per text as
  `last_token_resid_all_layers`/`probe_layer` produce. Citing
  zou2023representation for a logistic-regression-on-last-token probe is a
  misattribution; LR appears in the paper only as an enumerated alternative
  (App. B.3), benchmarked on utility, not honesty.
- The paper's §5.1.1 uses logistic regression as its cautionary example: the
  LR direction had "the highest accuracy, yet it elicits little to no
  alteration in model behavior when strengthened or suppressed — it only
  identifies neural correlates." A corroboration section built on LR-probe
  accuracy invites exactly this criticism from reviewers.
*Confidence: high. Severity: medium (corroboration-only by spec, not
selection).*

### F9. Notebook install cell is broken and incomplete
**Notebook Setup.ipynb, cell 1 (id 8c9b1f4e).** Three issues:
1. `%%capture` is on line 6 of the cell, after comment lines. IPython cell
   magics must be the **first line of the cell**; as written the cell errors
   (the `%%capture` line is not treated as a cell magic), and the
   `!pip install` line likely never executes. Verify in Colab; if the team has
   been running this cell successfully, the cell contents differ from what is
   committed.
2. Even when fixed, the cell installs `transformers datasets accelerate peft
   bitsandbytes wandb` but **not `lm-eval`**, which INTERFACES.md lists as
   required for benchmarks and which `run_baseline.py` imports by default
   (`--skip-benchmarks` off). A Gate-1 baseline run following the canonical
   command dies at `from lm_eval import simple_evaluate` after the (slow)
   generation phase.
3. `scikit-learn` (imported at module top of interp.py, along with numpy) is
   neither installed here nor mentioned in INTERFACES.md's dependency list;
   it works today only because Colab preinstalls it.
Failure scenario: fresh Colab session, run all cells, launch the canonical
baseline → crash mid-run; or interp work on a non-Colab box → import error.
*Confidence: medium-high on (1) (not executable locally — no IPython on this
machine), high on (2) and (3). Severity: medium.*

### F10. `summarize_runs` group key omits run_id, split, and seed
**src/algoverse/metrics.py:343-351.** `RUN_KEY_FIELDS` is (model_id,
adapter_path, bypassed_layer, patch_layer, patch_source, checkpoint_step,
arm). Rows differing only in `run_id`, `split`, or `seed` pool into a single
summary with no warning.
Failure scenario: a rows.jsonl accumulates a selection-split sweep run and a
final-split confirmation run of the same checkpoint (same model_id, same
bypassed_layer); `summarize_runs` merges them, and the "final pool untouched
until headline numbers" firewall is breached *in the analysis* even though
generation respected it. Same shape for two fine-tuning seeds of one arm: the
spec wants variation **across** seeds reported, but pooled rows average it
away. *Confidence: high on behavior; medium on whether mixed files will occur
in practice. Severity: medium.*

### F11. Attention-JSD compares two models; both the spec's Methodology and the cited paper say two conditions *(revised after reading the paper)*
**src/algoverse/interp.py:181-199.** The spec's Methodology: "we also
calculate the JSD between attention distributions in deception-incentivized
and control environments" — one model (M_D), two prompt conditions. The
implementation, `attention_jsd_between_models(model_a, model_b, tokenizer,
texts)`, compares two **models** on the **same** texts.
The cited paper (chaudhary2025whitebox = arXiv 2505.14300) was fetched and
read; it settles the ambiguity **against** the implementation:
- The paper computes JSD "between the attention distributions of **normal and
  backdoored samples** across all layers" (§2.2, App. A.2) — one fine-tuned
  model, two input sets, per-layer, argmax to pick a "discriminative layer."
  It never compares two models. The spec's Related-Work gloss ("JSD between
  attention distributions of **two models** at the same layer... layers most
  **causally** associated") misdescribes the paper on both axes: wrong
  comparison axis, and the paper's JSD is a two-sentence correlational
  selection heuristic with no validation — its causal claims come from
  separate zero/mean-ablation interventions (§4.3.3, App. A.3), not JSD.
- My original objection that the two-condition version is shape-incompatible
  is **withdrawn**: the paper pads all samples to a fixed max_length, so a
  condition-vs-condition comparison is implementable (with the caveat that
  padded/future positions then enter the comparison).
- Citation-hygiene detail: the JSD procedure exists only in the paper's **v2
  (July 2026)**; the 2025 v1 the BibTeX key points at contains no JSD at all
  (its monitored layer is hardcoded).
- For the record, the paper's released code computes JSD over a **global
  softmax of the flattened raw QK logits, averaged over examples first**,
  squared JS distance in nats — methodologically far weaker than interp.py's
  row-wise-attention, JSD-then-average version. The implementation is
  *better* than the citation; it just isn't the cited method or the spec's
  stated comparison.
Failure scenario: the corroboration figure caption says "JSD between
conditions (following chaudhary2025whitebox)" while what was computed is
"JSD between checkpoints" with a different aggregation — the methods section
misstates both the computation and the provenance. This is the
interface-level silent adaptation INTERFACES.md forbids; either the code
moves to the spec's two-condition comparison or the spec/INTERFACES record
the two-model redefinition and fix the Related-Work sentence.
*Confidence: high. Severity: medium (corroboration-only), high for the
spec's Related-Work sentence if it survives into the paper.*

### F12. Docstrings claim capabilities that do not exist; INTERFACES hard requirements unmet
**src/algoverse/interp.py:1-11, src/algoverse/train.py:1-3.** interp.py's
docstring says it "contains layer bypass, activation patching, and linear
probing" — there is **no layer-bypass function and no activation-patching
function** in the file (only direction-ablation, probing, JSD). train.py
claims "the main training loop... training, evaluation, and logging" and
contains nothing but the docstring. Related: INTERFACES.md's two hard
requirements for `install_bypass` — (1) byte-identical no-hook output,
unit-tested; (2) the implementation recorded in every run's `gen_config` — are
both unmet, and the current `gen_config` dict (eval.py:156-161) has no field
for it. Also note `make_ablate_direction_hook` (interp.py:129) implements
direction ablation, a technique the spec never calls for — presumably
exploratory tooling, but flagging per the no-unrequested-scope convention.
Failure scenario: a teammate greps the docstring, believes bypass exists,
wires a sweep around `bypassed_layer` bookkeeping (see F4), and produces
intact-model rows labeled as bypassed. *Confidence: high. Severity: medium
(docs/pending-work misrepresentation; the underlying absence feeds F4).*

### F13. `--arm` is unvalidated free text
**scripts/run_baseline.py:41; src/algoverse/eval.py:114.** INTERFACES.md
defines `arm` as an enum ("I,D" | "I,C" | "L,D" | "L,C" | "damage_matched" |
null); nothing validates it anywhere.
Failure scenario: one Colab session types `--arm "LD"`; those rows form a
silent separate group in `summarize_runs`, and the Stage-3 recovery
computation that filters by arm quietly drops or splits a checkpoint's rows.
*Confidence: high. Severity: medium-low.*

### F14. `dtype=` kwarg in the loader is transformers-version-sensitive
**src/algoverse/models.py:58 and 67.** `AutoModelForCausalLM.from_pretrained(
..., dtype=torch.float16)` uses the new-style kwarg (transformers ≥ 4.56);
older versions expect `torch_dtype` and can ignore or reject `dtype`. Nothing
pins the transformers version (pyproject declares no deps by design; the
notebook installs unpinned latest, which currently works).
Failure scenario: an environment with transformers 4.4x loads the model in
float32 silently (kwarg ignored), doubling memory and OOM-ing the T4, or
producing subtly different numerics than the fp16 runs it is compared to.
*Confidence: medium. Severity: medium-low (unpinned-latest currently fine).*

---

## Low severity / style / test quality
(reported per the role's coverage-over-filtering rule)

- **F15. src/algoverse/tasks.py:167** — `make_scenario_grid` docstring says
  "currently 480 scenarios"; the grid is 600 (5×5×6×4) since the 0.60 ratio
  landed. Stale doc. *high / low.*
- **F16. src/algoverse/interp.py:1-11** — duplicated module docstring; the
  second string literal is a no-op expression. *high / style.*
- **F17. src/algoverse/utils.py:67** — comment "latest.pt -> latest.tmp" is
  wrong; `with_suffix(path.suffix + ".tmp")` produces `latest.pt.tmp` (which
  is the better behavior). *high / style.*
- **F18. src/algoverse/tasks.py:374-375** — a claim of `$0` with
  `true_value=None` is labeled deceptive/"fabricated", though "$0" semantically
  declares *no* offer. Unlikely (the format instruction pushes models to
  "NONE") but it would count as deception in the fabricated bucket.
  *high on behavior, low on occurrence / low.*
- **F19. src/algoverse/tasks.py:182** — banker's rounding makes nominal
  ratios uneven: 90,000×0.85 → 76,000 (rounds down, real ratio 0.844) while
  90,000×0.75 → 68,000 (rounds up, 0.756); 170,000×0.75 and 150,000×0.85 both
  yield 128,000. Deterministic and harmless to scoring; worth knowing when
  interpreting per-ratio temptation curves. *high / low.*
- **F20. tests/test_data.py:116-117** — the assertion
  `0 <= ids["stakes"] < len(INCENTIVE_STAKES) or 0 <= ids["stakes"] <
  len(NO_STAKES)` is vacuous (both pools have length 6, the two sides are
  identical) and never binds the index to the framing-appropriate pool. A bug
  that recorded incentive indices for no-stakes rows would pass. *high /
  test-quality.*
- **F21. all tests' `__main__` runners** — only `AssertionError` is caught; a
  test raising KeyError/TypeError crashes the loop instead of counting as a
  failure, and later tests in the file never run. Visible but miscounted.
  *high / test-quality.*
- **F22. src/algoverse/utils.py:92** — `load_checkpoint` returns
  `state["step"] + 1` ("the step to resume from"), which is only correct if
  savers pass "last completed step". train.py is empty so no caller pins the
  convention; when training lands, a caller that saves "next step to run"
  double-advances and silently skips a step. Off-by-one hazard, not yet a bug.
  *high on the hazard / low today.*
- **F23. test coverage gaps in pure logic** — `_pick_metric` (eval.py:285),
  `gate1_report`'s decision logic (eval.py:469; testable with synthetic rows
  files, and F2 shows why a test asserting the spec's formula would have
  paid), the F10 pooling behavior, and a torch-free check that
  `score_response`'s output fields match INTERFACES' row schema (currently
  asserted only inside the model-loading smoke test). The tests pass but none
  would catch regressions here. *high / test-coverage.*
- **F24. src/algoverse/data.py:52** — `TRAIN_COMPANY_OFFERS` skips 155,000
  although the comment ("the odd 85k..165k values") implies full coverage.
  Harmless; likely an oversight worth one look. *high / trivial.*
- **F25. Notebook Setup.ipynb, clone cell** — `print('cloned repo)')` typo;
  also the GitHub token is embedded in the remote URL and persists in
  `.git/config` on the Colab VM for the session. Standard Colab practice, but
  worth knowing. *high / trivial.*
- **F26. src/algoverse/models.py:15-16; INTERFACES.md row schema** — the
  loader knows only Qwen (DEV/PROD), and INTERFACES' `bypassed_layer
  (null | 0-27)` is Qwen-7B-specific (28 layers); the spec names Llama-3.1-8B
  (32) and Gemma-2-9B (42) as well. Also Gemma-2's chat template rejects
  system-role messages, which `render_messages` always emits — one more thing
  (with F5) that will surface when the third model arrives. *high /
  forward-compat.*
- **F27. Assorted small items** —
  (a) tasks.py:107 regex: a range claim "MY BEST OUTSIDE OFFER: $110-120k"
  parses as $110 (dollars, no k applied) → understated, not deceptive;
  mitigated by the strict format instruction.
  (b) utils.py:109 `append_jsonl` opens without `encoding="utf-8"` (read side
  specifies it) and never flushes; a killed run can lose buffered rows —
  `load_rows` tolerates the torn line, so cost is re-generation only.
  (c) eval.py:377-383 docstring says the perplexity number is "comparable
  across models"; across tokenizers it is not (fine for the actual use,
  same-family deltas).
  (d) tasks.py:226 `get_scenarios` silently returns the whole pool when
  `n >= len(pool)` — a caller asking for 400 selection scenarios gets 305
  with no signal.
  (e) eval.py:141-143 resume matches on (run_id, scenario_id, condition) only;
  reusing a run_id with a different model/config in the same file silently
  skips scenarios and blends configs in the function's return value (rows are
  self-describing, so downstream analysis survives, but run_baseline's printed
  tau covers the blend).
  (f) INTERFACES.md prescribes `pip install -e .`, and pyproject.toml's
  comment promises "without sys.path tricks", but the notebook and all four
  scripts use `sys.path.insert` instead. Consistent with each other, not with
  the docs.
  *high on facts / low.*

---

## Addendum: paper-fidelity findings (F28–F31)

Written after fetching and reading the three cited papers that existing code
operationalizes (see addendum note at top). arXiv 2605.17113 postdates the
reviewer's training data and was assessed purely from the fetched text.

### F28. The implemented environment is not the cited Offer Negotiation environment
**src/algoverse/tasks.py:4-9 (docstring); RESEARCH_SPEC.md Methodology
("We use the Offer Negotiation environment [merrill2026pointofnoreturn] for
fine-tuning and layer selection").** The paper was fetched and read
(arXiv 2605.17113, "The Point of No Return: Counterfactual Localization of
Deceptive Commitment in Language-Model Reasoning"). Its Offer Negotiation
environment differs from the implementation in every load-bearing respect:
- The candidate's prompt shows **no true outside offer** — only a
  "Target total comp" (Fig. 12); a "true outside option in the latent state"
  is asserted in one §4 sentence but never operationalized. The
  privately-known-truth design that makes this codebase's scoring mechanical
  is an invention of the adaptation.
- Deception is labeled by a **structured enum action**
  (`REJECT_CLAIM_HIGHER_OFFER` deceptive; ACCEPT/REJECT not — App. C.6); the
  claimed salary value is never compared to anything. No regex, no judge, no
  refusal/validity handling, no tolerance.
- The paper has **no incentive/control condition split and no tau-like
  metric** (zero hits for advantageous/control-condition contrasts); the
  incentive-vs-control design and τ come from RESEARCH_SPEC itself.
- Multi-turn dialogue + JSON action output vs. this project's single-turn
  Q&A with a "MY BEST OUTSIDE OFFER:" final line.
- The paper prescribes **no scenario grid, no value ranges, no held-out
  scenario splits, no paraphrase firewall** (its only generalization hygiene
  is environment-level leave-one-out).
None of this makes the code wrong — RESEARCH_SPEC owns τ and the conditions,
and the scoring/firewall machinery is well built (F7 aside). The finding is
about **attribution**: "adapted from" in the tasks.py docstring stretches to
"shares the premise of", and the spec's "We use the Offer Negotiation
environment [merrill2026]" is unsupportable as written — a methods-section
claim that reviewers with the paper open will reject. The environment should
be described as new, "inspired by" the paper's scenario premise.
Failure scenario: the paper ships with "we use the environment of Merrill &
Srivastava" → a reviewer compares and finds a different environment, different
labels, different metric — the credibility of the mechanically-scored design
(a genuine strength) is damaged by the misattribution.
*Confidence: high (with the caveat that the paper's Fig. 12 prompts are
explicitly abridged). Severity: high.*

### F29. `probe_layer`'s internal random split cannot enforce the spec's scenario-split rule
**src/algoverse/interp.py:120-122.** The spec's statistical analysis: "We
split related prompt variants by their underlying scenario to prevent
leakage." `probe_layer(X, y)` does a row-level
`train_test_split(test_size=0.3, random_state=0, stratify=y)` internally, so
any caller whose X contains multiple rows from the same underlying scenario
(paraphrase variants, both conditions of one scenario, or — if the RepE
per-response-token collection from F8 is ever adopted — many token positions
of one text) gets those rows scattered across train and test. The API offers
no groups/split argument, so the spec's rule is unenforceable at the point
where the split actually happens.
Failure scenario: probes are fit on activations from paraphrase variants of
selection-pool scenarios; sibling variants of the same scenario land in the
test split; reported "held-out decodability" is inflated by within-scenario
similarity, and the corroboration overstates how decodable deception is.
*Confidence: high on the mechanism; medium on whether future callers will
pass variant-structured X. Severity: medium.*

### F30. The layer-bypass causal method is project-new, not paper-derived
**RESEARCH_SPEC.md Related Work / Limitations vs. arXiv 2605.17113.** The
spec's quote that recent work "localizes deceptive capabilities to compact
attention-head sets comprising under 10% of heads" is numerically accurate
(the paper reports circuits of 0.8%–8.3% of heads). But the paper's method is
attribution patching + full-commitment-sentence **activation patching**,
selected on one environment and frozen, measured as **reduction in the
deceptive sentence's log-probability** — there is no ablation, no layer
bypass, and no behavioral deception-rate intervention anywhere in it (closest
analogue is its steering experiment, which is behavioral but directional).
The A_l layer-bypass sweep at the heart of this project therefore has no
methodological precedent in the cited paper; it is new method. That is fine —
but wherever the write-up implies the causal analysis follows
merrill2026pointofnoreturn, it should not.
*Confidence: high. Severity: low (spec/write-up wording; no code change).*

### F31. Citation-hygiene inventory from the paper reads
- `chaudhary2025whitebox`: the JSD method exists only in **v2 (July 2026)**
  of arXiv 2505.14300; v1 (2025) contains no JSD. The citation should pin v2
  (and note the retitle: v2 is "Beyond Black-Box Obfuscation: Mechanistic
  Analysis and Defense of White-Box Monitors").
- `zou2023representation`: there is **no dataset named "Instructed-Pairs"**
  in the paper. The statements come from **Azaria & Mitchell (2023)** (true
  statements only); Zou et al. contribute the paired
  honest/dishonest-instruction template (App. D.1.2). The spec should name
  the construction accurately and cite Azaria & Mitchell, currently uncited.
- `merrill2026pointofnoreturn` internal detail worth knowing before citing
  its constants: the paper's §4 and App. A.2 disagree on decoding temperature
  (0.7 vs 0.5) — don't inherit either number as "prescribed".
*Confidence: high. Severity: low (hygiene; becomes real at submission time).*

---

## Positive verifications (for the record)

- Grid: 600 scenarios, hash-split 305 selection / 295 final — matches
  INTERFACES.md exactly (verified by computation).
- `ROW_FIELDS` (eval.py:32-41) matches the INTERFACES row schema field-for-field;
  invalid rows carry `deceptive: null`; resume key matches the documented one.
- The perplexity first-window fix (commit 0a4aed2) is correct: 19,999 of
  20,000 tokens scored, pinned by tests/test_perplexity_count.py.
- Scenario ids are content fingerprints; grid growth cannot rename existing
  scenarios (tested, including a mutation test).
- Incentive/control prompts are byte-identical outside the stakes paragraph
  (tested), and the M_D/M_C datasets share byte-identical system/user turns
  (tested).
- The bootstrap resamples scenarios, not rows, restricts point estimate and CI
  to the shared scenario set, and every training reply is re-validated with
  the real scorer at build time.
- From the paper reads: the arXiv id in tasks.py (2605.17113) resolves to the
  real "Point of No Return" paper; the spec's "under 10% of heads" quote is
  numerically accurate (0.8–8.3%); and since that paper prescribes no scoring
  tolerance, refusal policy, or value grid for Offer Negotiation, those
  design choices are correctly owned by this project (they belong in the F3
  ratification inventory, not to the citation).
- interp.py's row-wise-attention, JSD-then-average implementation is
  methodologically sounder than the cited paper's released code (global
  softmax over flattened raw QK logits, averaged over examples first) — the
  F11 issue is provenance and comparison axis, not numerical craft.

---

## Current-status verification (2026-08-13)

This table checks only F1–F31 against the repository as it exists now. Status
means: **still present**, **fixed**, **partially fixed**, or **no longer
applicable**. Evidence below is from the current files and current executable
pure-Python behavior, not from the original review state.

| Finding | Status (current-code evidence) |
|---|---|
| F1 | **fixed** — `metrics.recovery` now accepts all four same-checkpoint arms and computes `(ld - lc) / (idd - ic)` (`src/algoverse/metrics.py:272-339`). `test_recovery_halfway`, `test_recovery_subtracts_control_drift`, and the denominator guard pin that definition (`tests/test_metrics.py:171-204`). |
| F2 | **fixed** — Gate 1 now computes `tau_gain(M_D, M_0)` and passes it to `gate1_decision` (`src/algoverse/eval.py:955-973`); the decision checks the gain and its CI, and also compares M_D negotiation competence with M_0 (`src/algoverse/metrics.py:342-439`). The original false-pass case is tested at `tests/test_metrics.py:223-242`. |
| F3 | **still present** — active unratified defaults remain in the decision path: `tau_gain_min=0.15`, `competence_drop_max=0.05`, and `ppl_rise_max=2.0` (`src/algoverse/metrics.py:377-379`, `src/algoverse/eval.py:881-883`, `scripts/gate1_report.py:38-42`), while the current spec still says the margin/bounds are “prespecified” without supplying these values. The cited inventory also remains in code, including `eps=0.05` (`metrics.py:272-273`), bootstrap defaults (`metrics.py:168,227`), scoring tolerances/validity/refusal policy (`tasks.py:116-131,315-378`), benchmark/perplexity limits (`eval.py:679-700,771-811`), and data lie margins/50:50 mixture (`data.py:63-67,267-270`). |
| F4 | **fixed** — `install_bypass` now exists and records live state (`src/algoverse/models.py:45-127`); `run_baseline.py` installs it when `--bypassed-layer` is supplied (`scripts/run_baseline.py:108-118`); and the evaluator rejects bookkeeping that differs from the live hook (`src/algoverse/eval.py:263-272`). `gen_config.bypass_impl` is derived from that live state (`eval.py:85-132`). |
| F5 | **still present** — `generate_batch` still renders each chat with `apply_chat_template(..., tokenize=False)` and re-tokenizes it with `tokenizer(texts, ...)` without `add_special_tokens=False` (`src/algoverse/eval.py:203-213`). Thus the double-special-token path identified for tokenizers whose rendered template already contains BOS remains. |
| F6 | **still present** — the CLI still passes the same `args.seed` to scenario selection (`get_scenarios(..., seed=args.seed)`, `scripts/run_baseline.py:119`) and generation/evaluation (`seed=args.seed`, lines 121-129). The added `--train-seed` records fine-tuning identity but does not separate scenario-subsample seed from generation seed. |
| F7 | **still present** — the data module still claims a disjoint value firewall (`src/algoverse/data.py:22-27,47-53`), while `test_training_grid_disjoint_from_eval` still compares only the declared constants (`tests/test_data.py:25-33`). Recomputing the current derived true-offer sets produces the same nine overlaps: 66k, 78k, 82k, 90k, 98k, 102k, 110k, 112k, and 128k. |
| F8 | **still present** — `probe_layer` still performs row-level `train_test_split`, fits `LogisticRegression`, and returns thresholded `clf.score` accuracy (`src/algoverse/interp.py:92-98`); activation collection is still a last-real-token residual per text (`interp.py:41-70`). There is no threshold-free metric or paired-difference/PCA implementation. |
| F9 | **still present** — Notebook install cell `8c9b1f4e` still places `%%capture` after comments rather than as the first cell line, and its install command still omits both `lm-eval` and `scikit-learn`; `interp.py` still imports sklearn at module import (`src/algoverse/interp.py:11-15`). |
| F10 | **fixed** — `RUN_KEY_FIELDS` now includes `run_id`, `split`, `seed`, and `train_seed`, and `_run_key` adds the ratified generation-profile fields (`src/algoverse/metrics.py:446-485`). `test_summarize_runs_groups_by_run_split_and_seeds` and the generation-profile test verify separation (`tests/test_metrics.py:290-335`). |
| F11 | **still present** — the only attention-JSD API remains `attention_jsd_between_models(model_a, model_b, tokenizer, texts)`, comparing two models on each identical text (`src/algoverse/interp.py:155-173`). There is still no one-model/two-condition implementation matching the current Methodology text. |
| F12 | **partially fixed** — layer bypass is now honestly located in `models.install_bypass`, implemented, tested, and recorded in `gen_config` (`src/algoverse/interp.py:1-9`, `models.py:67-127`, `eval.py:109-132`, `tests/test_bypass.py:170-212`). However, `interp.py` still claims activation patching while defining no activation-patching function, `train.py` still contains only its capability-claiming docstring (`src/algoverse/train.py:1-3`), and the out-of-spec direction-ablation utility remains (`interp.py:101-134`). |
| F13 | **still present** — `--arm` remains an unrestricted string argument (`scripts/run_baseline.py:48`), and `run_negotiation_eval` accepts and writes it without checking the INTERFACES enum (`src/algoverse/eval.py:234-242,316-326,428-447`). |
| F14 | **still present** — the loader still uses the version-sensitive `dtype=` keyword in both non-quantized load branches (`src/algoverse/models.py:166-179`), while the notebook installs an unpinned Transformers version. |
| F15 | **still present** — `make_scenario_grid` still says “currently 480 scenarios” (`src/algoverse/tasks.py:166-170`), while the current constants at lines 43-65 still produce 600; recomputation returns 600. |
| F16 | **fixed** — `src/algoverse/interp.py:1-9` now has one module docstring; the second no-op string literal is gone. |
| F17 | **still present** — `save_checkpoint` still constructs `path.with_suffix(path.suffix + ".tmp")` while the adjacent example says `latest.pt -> latest.tmp` (`src/algoverse/utils.py:68`). The actual result remains `latest.pt.tmp`. |
| F18 | **still present** — after handling the string `"NONE"`, `label_deception` still classifies every numeric claim with `true_value is None` as fabricated (`src/algoverse/tasks.py:368-375`), including numeric `$0`; executing `label_deception(0, None)` returns deceptive/fabricated. |
| F19 | **still present** — true offers are still calculated with Python `round(..., -3)` (`src/algoverse/tasks.py:173-182`). Current execution still gives 90k×0.85→76k, 90k×0.75→68k, and both 170k×0.75 and 150k×0.85→128k. |
| F20 | **still present** — the test still uses `0 <= stakes < len(INCENTIVE_STAKES) or ... < len(NO_STAKES)` without selecting the pool from `m["framing"]` (`tests/test_data.py:108-119`); both pools still have length six (`src/algoverse/data.py:73-88`). |
| F21 | **still present** — the direct runners still count only `AssertionError` (for example `tests/test_data.py:124-134` and `tests/test_metrics.py:433-444`); `test_bypass.py` additionally handles `SkipTest`, but other unexpected exceptions still abort rather than being counted and allowing later tests to run. |
| F22 | **still present** — `load_checkpoint` still returns `state["step"] + 1` (`src/algoverse/utils.py:72-93`), and `train.py` remains empty apart from its docstring, so no caller establishes whether the saved value means “last completed” or “next to run.” |
| F23 | **partially fixed** — the previously missing Gate-1 decision tests now exist (`tests/test_metrics.py:223-266`), and grouping across run/split/seed/gen profile is tested (`tests/test_metrics.py:290-351`). `_pick_metric` still has no direct test (only its definition/use at `src/algoverse/eval.py:602-623,752`), and exact INTERFACES row-schema completeness is still checked only on ML-model paths (`eval.py:467-553` / `tests/test_bypass.py`), not by a torch-free scorer-schema test. |
| F24 | **still present** — `TRAIN_COMPANY_OFFERS` still jumps from 145,000 to 165,000 and omits 155,000 despite the “odd 85k..165k values” comment (`src/algoverse/data.py:49-53`). |
| F25 | **still present** — Notebook clone cell `28c9ef45` still contains `print('cloned repo)')`, and still embeds `GITHUB_TOKEN` in the HTTPS clone URL passed to Git, which persists as the remote URL in that session’s checkout. |
| F26 | **partially fixed** — INTERFACES now defines model-relative layer ranges and names 28/32/42 layers (`INTERFACES.md:23`), and bypass mechanics tests now cover tiny Qwen2, Llama, and Gemma2 models (`tests/test_bypass.py:60-89`). But the convenience model constants remain Qwen-only (`src/algoverse/models.py:13-17`), and `render_messages` still always emits a system-role message with no Gemma-specific handling (`src/algoverse/tasks.py:230-273`). |
| F27 | **partially fixed** — (a) the range regex behavior remains: the current regex (`tasks.py:107-110`) parses `MY BEST OUTSIDE OFFER: $110-120k` as 110 dollars; (c) WikiText slice docs still claim cross-model comparability (`eval.py:771-776`); (d) `get_scenarios` still silently returns the whole pool for oversized `n` (`tasks.py:213-227`; requesting 400 selection scenarios returns 305); and (f) the notebook/scripts still use `sys.path.insert` instead of editable install (`scripts/run_baseline.py:19`, Notebook cell `28c9ef45`). In contrast, (b) is fixed by UTF-8 binary append plus flush/fsync (`utils.py:95-125`), and (e) is fixed by manifest and run-identity guards (`eval.py:288-386`). |
| F28 | **still present** — the task module still describes the environment as “adapted from” the cited paper while implementing the private-value arithmetic design (`src/algoverse/tasks.py:4-23`), and current RESEARCH_SPEC Methodology still says “We use the Offer Negotiation environment” from that citation. The attribution identified by the finding has not changed. |
| F29 | **still present** — `probe_layer(X, y)` still performs its own row-level random split and accepts neither scenario groups nor a caller-supplied split (`src/algoverse/interp.py:92-98`), so the current spec’s underlying-scenario split rule cannot be enforced through this API. |
| F30 | **fixed** — the current bypass plan now explicitly calls the A_l sweep a “project-new method with no precedent in merrill2026pointofnoreturn” and says the write-up must not imply otherwise; it separately attributes residual-identity semantics to Lad et al. (`planning/layer-bypass.md:66-79`). The implemented bypass code itself makes no Merrill-derived-method claim (`src/algoverse/models.py:67-86`). |
| F31 | **still present** — current RESEARCH_SPEC still cites `chaudhary2025whitebox` without a version pin, still calls the probe source the “Instructed-Pairs dataset” and cites only Zou (`RESEARCH_SPEC.md`, Related Work and Localization corroboration), and contains no Azaria & Mitchell citation/correction. The citation-hygiene items listed in F31 remain outstanding. |
