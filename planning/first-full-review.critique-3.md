# `first-full-review` — Plan Critique, Round 3

## Findings

### F1. Gate-1 “full-pool” coverage does not require complete paired rows

WP12 checks only each file's set of scenario IDs. A file containing incentive rows for all 305 scenarios but a control row for only one scenario satisfies that check. An executable reproduction produced gain 1.0, competence 1.0, and PASS with this malformed data. Duplicate and mixed-run rows are likewise not excluded.

**Confidence:** high. **Severity:** high.

### F2. Gate-1 benchmark completeness does not establish benchmark comparability

WP12 checks only that three metric names exist for M_0 and M_D. `gate1_report` discards each row's `config`, `stderr`, and provenance, so different seeds, limits, task configurations, or model identities can be compared as deltas. Undersized benchmark runs can therefore yield a publishable PASS despite the ratified limits and reporting rule.

**Confidence:** high. **Severity:** high.

### F3. A later dash-range can be hidden by an earlier valid marker

The new range short-circuit runs only when strict extraction fails. A response containing `MY BEST OUTSIDE OFFER: $100k` followed by a corrected final line `MY BEST OUTSIDE OFFER: $110-120k` still produces the earlier valid match, so range detection is never consulted and the response is scored as $100,000.

**Confidence:** high. **Severity:** high.

### F4. Bypassed-layer internals remain consumable outside the two JSD wrappers

WP6 masks the final JSD vectors but leaves `last_token_resid_all_layers`, `resid_all_layers_batch`, and direct `attention_all_layers` results available at the disconnected layer. `probe_layer` has no model or layer-state input from which to derive the bypass. This still violates the requirement covering all interpretation analyses and can create false relocation evidence.

**Confidence:** high. **Severity:** high.

### F5. The standardized probe returns an unusable bare classifier

WP5 fits `StandardScaler`, trains on transformed activations, and returns only `clf`. Any transfer caller applying that classifier to raw activations gets scores in the wrong feature space. The cited method explicitly reapplies the training affine transformation to new samples.

**Confidence:** high. **Severity:** medium.

### F6. The probe recipe remains unratified and still differs from its cited method

WP5 implements split fraction 0.3, seed 0, and default `LogisticRegression` behavior while the spec lists those choices as pending. It describes L2 `C=1.0` but does not pass `C` or `penalty` explicitly. Goldowsky-Dill et al. use normalized activations with L2 regularization λ=10 and response-token aggregation; only the last-token deviation is declared. See the [primary paper](https://arxiv.org/html/2502.03407).

**Confidence:** high. **Severity:** medium.

### F7. Probe activation collection still lacks the generation rendering contract

The unchanged activation readers tokenize strings with the default `add_special_tokens=True` and do not apply or verify the Gemma system fold. On Llama/Gemma, probes can therefore read double-BOS or otherwise differently rendered prompts than those used for behavioral evaluation.

**Confidence:** high. **Severity:** medium.

### F8. Attention-JSD still cannot produce its required confidence interval

The function returns one point per layer and accepts no scenario IDs or paired groups. It therefore cannot perform the scenario bootstrap required by `RESEARCH_SPEC.md`, while the new `interp.jsonl` contract requires `ci_low` and `ci_high`.

**Confidence:** high. **Severity:** medium.

### F9. The attention-JSD rendering contract is documentation-only

WP6 explicitly defers the canonical scenario-to-rendered-string helper to a future corroboration plan. The reported-quantity home still accepts arbitrary strings, so callers can legitimately produce different values through different chat-template, fold, or generation-prompt choices.

**Confidence:** high. **Severity:** medium.

### F10. The new interpretation result contract has no implementation path

The interface now requires append-only `interp.jsonl` rows with resume and identity discipline, but the plan adds only functions returning arrays or dictionaries. It defines no writer, run metadata construction, configuration identity, duplicate guard, or resume behavior.

**Confidence:** high. **Severity:** medium.

### F11. The revised manifest still cannot guarantee the claimed batch reproducibility

WP2 records draw order because batch composition affects reproducibility, but `batch_size` remains outside the resume identity guard and analysis grouping. A resumed run can change batch size, and a crash after partially appending a completed batch causes the remaining rows to be rechunked even with the same size. The plan also does not define authoritative `scenario_seed`/`n` values for direct evaluator callers.

**Confidence:** high. **Severity:** medium.

### F12. The direction-aware snap still does not enforce the ratified raw 5–25% margin

WP4 tests membership in rounded endpoints rather than the actual percentage bounds. For example, $119,000 is 25.263% above a $95,000 offer but lies inside the planned rounded window; $89,000 is only 4.706% above $85,000 and also passes. Every training offer has one rounded endpoint outside the literal bound.

**Confidence:** high. **Severity:** medium.

### F13. Gemma folding is neither bound to the model nor recorded in the dataset manifest

`--fold-system` defaults off, output names remain generic, and the proposed manifest does not record the flag. A Gemma run can therefore consume unfolded data, while folded and unfolded builds with identical seed/count metadata are not distinguishable through provenance.

**Confidence:** high. **Severity:** medium.

### F14. The value-firewall wiring test still does not prove the snap result is used

Recording that `_snap_off_eval_values` was called does not detect production code that calls it and discards its return value. The remaining 40-row output check can again be vacuous when that draw contains no collision.

**Confidence:** high. **Severity:** medium-low.

### F15. The binding canonical Gate-1 workflow still contradicts the new guard

`INTERFACES.md` continues to prescribe `--n 100` and a report command without competence inputs. Following the canonical commands now necessarily produces `INCOMPLETE`, while the ratified workflow requires 305 scenarios and all benchmark files.

**Confidence:** high. **Severity:** medium-low.

### F16. WP6's code block contradicts its single-BOS contract

`attention_all_layers` is to use `add_special_tokens=False`, but local `seq_len` still calls the tokenizer without that argument. The planned tokenizer-kwargs test should fail against the supplied implementation, leaving the implementer to choose between conflicting instructions.

**Confidence:** high. **Severity:** low.

### F17. The Miller correction was not applied everywhere

The detailed uncertainty section now calls bootstrap for τ a project choice, but `RESEARCH_SPEC.md`'s final-paper checklist still says bootstrap is “warranted exactly” for both τ and R_t. That surviving text can reintroduce the unsupported citation claim at write-up time.

**Confidence:** high. **Severity:** low.

### F18. The Lad citation does not directly support the proposed neutral-JSD metric or threshold

Lad et al. measure output-distribution KL and top-1 agreement on one million Pile tokens, not mean JSD on the WikiText-2 slice. Their reported 72–95% top-1 retention supports robustness context, but not the claim that 0.25-nat JSD tolerates the same middle-layer regime. See the [primary paper](https://arxiv.org/html/2406.19384).

**Confidence:** high. **Severity:** low.

## Verification Evidence

- Read the complete role, current normative specification, binding interfaces, revision-3 plan, prior critiques, and dispositions.
- Ran every current direct CPU test suite: data, metrics, scoring, scenarios, and perplexity passed; the bypass suite skipped all 14 tests because numpy, torch, transformers, sklearn, and pytest remain unavailable.
- Executably reproduced the malformed 305-ID Gate-1 PASS and the earlier-marker/later-range extraction failure.
- Checked every planned rounded training-offer endpoint against the literal 5–25% bounds.
- Rechecked the revised literature claims against the primary Goldowsky-Dill, Lad, probe-robustness, and probe-scaling papers.

---

## Disposition (planner, 2026-08-14 — applied as plan revision 4)

All 18 findings ACCEPTED in full or part; one sub-implication rejected
with reason (F11). Two feasibility-critical designs were verified by
computation before acceptance: the F12 inward-window rule (17-33 slots
per offer, every raw margin literally within [5%, 25%], single-step
escapes everywhere) and the F3 subtlety that with the range-rejecting
regex a later range line produces NO match, so the fix must slice at the
last marker OCCURRENCE, not take the last regex match. The F15
INTERFACES edit and the closure decision were put to the human directly
this session. Per the human: no further plan-critique rounds are
planned; findings from implementation are handled by roles/4.

| Finding | Disposition | Resolution |
|---|---|---|
| F1 | **Accepted** | WP12 guard clause (b) → PAIRED pool coverage: per rows file, the multiset of (scenario_id, condition) must equal selection ids × both conditions exactly once, all rows one run_id; missing conditions, duplicates, extras, and mixed runs each → INCOMPLETE naming the defect. The critic's malformed case (305 incentive + 1 control) becomes a test. |
| F2 | **Accepted** | WP12 clause (a) → benchmark COMPARABILITY: per metric, the M_0/M_D competence rows' `config` dicts must be EQUAL; deltas reported WITH propagated stderr (previously discarded); below-ratified sample sizes print a loud RECORDED DEVIATION line (the ratified rule permits recorded deviations — cross-arm equality is the hard requirement). |
| F3 | **Accepted** (grader clarification; dated note on the ratified F27a bullet) | WP3 → authoritative-marker rule: extraction and range detection operate on the text FROM THE LAST marker occurrence (case-insensitive rfind); no marker → whole text as today. Tests: early-valid + later-range → unparseable, fallback NOT called; early-range + later-valid → later value; two valid → last wins. |
| F4 | **Accepted** | WP6 → reader-level guard: all three readers gain `on_bypassed="raise"` (derive `bypass_state`, raise naming the layer, explicit `"allow"` opt-in); JSD wrappers opt in internally and NaN as before. Enforcement lives at the only place the model object is in hand. |
| F5 | **Accepted** | WP5 → `probe_layer` returns a fitted sklearn Pipeline (StandardScaler → LogisticRegression); transfer callers score RAW activations. Test asserts the Pipeline's decision_function on raw held-out data reproduces the reported scores. |
| F6 | **Accepted** | WP5 → regularization explicit and cited: `LogisticRegression(penalty="l2", C=0.1, max_iter=1000)` (their lambda=10). Spec Open-decisions bullet updated — remaining free constants shrink to split 0.3 / seed 0 / max_iter. |
| F7 | **Accepted** | WP6 → the rendering contract becomes MODULE-WIDE: every interp encoder (both resid readers included) uses `add_special_tokens=False`; the contract paragraph moves to the module docstring; kwargs tests cover all readers. |
| F8 | **Accepted** | WP6 → scenario-bootstrap CI from CACHED per-text contributions (one forward pass per text; resamples recombine sums — no re-forwarding); `groups_a`/`groups_b` scenario ids (default: per-text groups); metrics-convention floor; returns {"jsd", "ci_low", "ci_high"} with the bypassed layer NaN in all three. Satisfies the interp.jsonl ci fields. |
| F9 | **Accepted** | The canonical renderer moves INTO this plan: `eval.render_condition_texts(scenarios, condition, tokenizer)` — render_messages → fold iff `_system_fold_needed` → apply_chat_template with generation prompt; byte-identical to generate_batch's input; pure stub-tokenizer test. Interp docstrings name it as THE input producer. |
| F10 | **Accepted in part** | No speculative writer now (a writer with zero callers is fresh critique surface, against the human's no-new-issues directive). BINDING instead: §E11 + spec record that the corroboration driver plan's FIRST deliverable is the interp.jsonl writer, mirroring the `_competence_done` pattern. |
| F11 | **Accepted in part; one sub-implication REJECTED with reason** | Wording corrected — draw-order recording reconstructs the draw; bit-level batch reproducibility is NOT claimed. Guarding `batch_size` is rejected: it would reverse the RATIFIED operational-vs-identity decision (layer-bypass round 3). `run_negotiation_eval` gains `scenario_seed=None`/`n=None` recorded verbatim into the manifest (null for custom scenario lists; the id list stays authoritative). |
| F12 | **Accepted** (amends ratified text; dated) | WP4 → INWARD-rounded windows (`_lie_claim_window`: 1000·ceil(o·1.05/1000) to 1000·floor(o·1.25/1000)); rounded claims CLAMPED into the window, then snapped within it. Verified feasible by exhaustive computation. The invariant test asserts the LITERAL bound 0.05 ≤ claim/offer − 1 ≤ 0.25. Dated corrections on the spec's firewall and margin entries. |
| F13 | **Accepted in part** | Builder records `fold_system` in manifest.json AND every meta row + loud FOLDED/UNFOLDED print; model-to-data binding (refusing unfolded data for Gemma) is loader/training-plan enforcement — recorded in §E6 and the spec; train.py does not exist yet. |
| F14 | **Accepted** | Wiring test records (input, output) pairs; asserts ≥1 collision occurred (self-verifying — a collision-free pinned seed FAILS the test, so vacuity is impossible) and every collided claim's built value equals the wrapper's OUTPUT (proving the return value is used). |
| F15 | **Accepted — human authorized the INTERFACES edit this session** | Canonical commands updated: baseline example `--n 305` (comment: full pool — publishable Gate-1; `--n 100` for sweeps); gate1_report example gains `--competence` inputs. Provenance-noted in the contract. |
| F16 | **Accepted** | WP6 code block fixed: `seq_len` encodes with `add_special_tokens=False` — prose, code, and tests now agree. |
| F17 | **Accepted** | Spec final-paper-deltas item 4 reworded to match the corrected §9: bootstrap is Miller's exception for R_t; for tau it is a recorded project choice. |
| F18 | **Accepted** | Spec item 16 reworded: Lad et al. support the metric FAMILY and qualitative regimes (KL/top-1 on one million Pile tokens — related, not identical, quantities); the 0.25 number rests only on the JSD-ceiling fraction and order-of-magnitude compression practice; the DEV-model calibration is upgraded to REQUIRED (confirm-or-revise in a recorded decision before any research-model sweep). |
