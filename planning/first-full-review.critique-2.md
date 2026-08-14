# `first-full-review` — Plan Critique, Round 2

## Findings

### F1. Stage-1 selection criteria remain undefined

`RESEARCH_SPEC.md:49` requires numeric bounds for invalid-response rate and neutral-distribution JSD plus a minimum causal-effect threshold. The purportedly complete inventory in plan WP12 supplies none of these, and substitutes a perplexity bound for the specified neutral-distribution JSD. Layer selection therefore remains non-reproducible.

**Confidence:** high. **Severity:** high.

### F2. Gate-1 benchmark checks can still disappear silently

WP12 leaves enforcement operational only. Current `gate1_decision` reports PASS when competence data is absent or incomplete, directly contradicting `RESEARCH_SPEC.md:289-292`. An executable check confirmed PASS with no benchmark rows and with GSM8K alone.

**Confidence:** high. **Severity:** high.

### F3. The full-pool Gate-1 rule is not enforced

The plan leaves both canonical Gate-1 commands at `--n 100` and adds only a human note, while the ratified rule requires all 305 selection scenarios. Nothing prevents a publishable Gate-1 report from using the sweep subsample.

**Confidence:** high. **Severity:** high.

### F4. Dash-range claims can become valid again through the publishable LLM fallback

WP3 tests only regex rejection. `score_response(..., use_llm_fallback=True)` subsequently sends the same range to the extractor, which may return either endpoint. An executable mock converted `$110-120k` into a valid $110,000 claim, disproving the plan's “never a wrong parsed value” guarantee.

**Confidence:** high. **Severity:** high.

### F5. Bypassed-layer internals remain silently consumable

WP6 makes `bypassed_layer` optional and does not derive or validate it against live bypass state. Other activation/probe/JSD helpers remain unchanged and return the disconnected layer normally. This violates `RESEARCH_SPEC.md:313-317` and could make a disconnected layer appear to contain relocated deception.

**Confidence:** high. **Severity:** high.

### F6. The probe recipe silently fixes unratified methodological choices

WP5 selects a 70/30 `GroupShuffleSplit`, seed 0, default-regularized logistic regression, and no stratification without a recorded decision. “Enough scenarios” does not guarantee both labels in the held-out group draw.

**Confidence:** high. **Severity:** medium.

### F7. The planned probe does not match the cited strategic-deception method

The cited work standardizes activations, fits specified L2 logistic regression, trains across token positions, and aggregates response-token scores; the plan leaves the repository's last-token activation readers and unnormalized default classifier in place without declaring the deviation. See the [primary paper](https://arxiv.org/html/2502.03407).

**Confidence:** high. **Severity:** medium.

### F8. Probe AUROC has no confidence interval

WP5 returns one AUROC point despite `RESEARCH_SPEC.md:84-85` requiring scenario-bootstrap confidence intervals for reported quantities. The plan's declared quantity home therefore cannot produce the specified report.

**Confidence:** high. **Severity:** medium.

### F9. Attention-JSD lacks a prompt-rendering contract

WP6 accepts raw strings and calls the plain tokenizer, but never specifies chat-template rendering, the Gemma system fold, generation prompts, or single-BOS handling. Different callers can compute different “condition JSD” values from the same scenarios.

**Confidence:** high. **Severity:** medium.

### F10. The new reported quantities have no append-only result schema

The plan adds probe AUROC and attention-JSD as reportable quantities but defines neither JSONL records nor provenance/grouping fields for them. `INTERFACES.md` covers only behavioral and competence rows, leaving the implementer to invent storage contrary to the append-only-results rule.

**Confidence:** high. **Severity:** medium.

### F11. `scenario_seed` is not recorded as ratified

WP2 records only sorted scenario IDs, while `RESEARCH_SPEC.md:263-265` explicitly requires the scenario seed in the manifest. Sorting also discards evaluation order, which determines batch composition.

**Confidence:** high. **Severity:** medium-low.

### F12. WP1 and WP7 acceptance tests do not exercise production wiring

The BOS test calls `_encode_chats` directly and can pass if `generate_batch` never uses it. The fold tests cover helpers and a guard mismatch but never verify that `run_negotiation_eval` detects, applies, and records the fold.

**Confidence:** high. **Severity:** medium.

### F13. The value-firewall tests do not prove the documented guarantee

They omit training company offers versus eval-derived values. The proposed 40-row integration test can also be vacuous unless its seed is pinned to generate at least one pre-snap collision; seed 7 produces none.

**Confidence:** high. **Severity:** medium-low.

### F14. The snap can violate the ratified 25% lie cap

For a $115,000 company offer, a reachable rounded $144,000 claim is forbidden and becomes $145,000, or 26.09% above the offer. WP4's tests check disjointness but not the post-snap margin invariant.

**Confidence:** high. **Severity:** medium.

### F15. Training replies still do not contain a separate final line

Existing replies concatenate prose and `MY BEST OUTSIDE OFFER` on one line. The plan leaves this unchanged even though `INTERFACES.md:72-74` says the data trains structured final-line compliance. Regex self-validation cannot detect the mismatch.

**Confidence:** high. **Severity:** medium.

### F16. Gemma training-data folding remains unimplemented

The plan adds a helper and immediately requires regenerated datasets, but the builder still writes system-role conversations and no model-specific preparation path or acceptance test applies the fold. The resulting files are not yet valid Gemma training inputs under the ratified rule.

**Confidence:** high. **Severity:** medium.

### F17. WP6 consumes one-shot iterables before analysis

`list(texts_a) + list(texts_b)` exhausts generators, after which `condition_mean` receives empty iterators and divides by zero. The API does not constrain inputs to reusable sequences.

**Confidence:** high. **Severity:** low.

### F18. The Miller citation is overstated for τ

The plan says bootstrap is “exactly” warranted for τ, while Miller explicitly recommends paired analytic inference for paired differences and reserves bootstrap for complicated estimators. The project may still prefer bootstrap because validity filtering complicates τ, but that is a project inference rather than direct support from the [cited paper](https://arxiv.org/html/2411.00640v1).

**Confidence:** medium. **Severity:** low.

## Verification Evidence

- Read the complete role, normative research specification, binding interfaces, plan, prior critique/dispositions, and referenced implementation/tests.
- Ran all locally executable CPU test suites; they currently pass. ML-dependent suites could not run because numpy, torch, transformers, sklearn, and pytest are absent.
- Independently reproduced the range-fallback and incomplete-Gate-1 PASS failures.
- Verified the ratio search and snap arithmetic exhaustively.
- Checked the plan's literature claims against primary arXiv sources. Chaudhary v2's one-model/two-condition JSD description and the cited probe scaling/style-shift claims are supported; the findings above capture the remaining deviations.

---

## Disposition (planner, 2026-08-14 — applied as plan revision 3)

All 18 findings ACCEPTED in full or part; none rejected. Two carry
escalations (F1, F10). Two accepted findings correct RATIFIED-decision
texts (F4, F14) — dated corrections recorded in RESEARCH_SPEC and flagged
to the human. The planner independently re-verified F14's feasibility
(forbidden-set minimum gap $2,000; every reachable collision has a
single-step in-window escape) and F15's reply construction
(data.py:170/179/181 space-concatenate the marker) before accepting.

| Finding | Disposition | Resolution |
|---|---|---|
| F1 | **Accepted in part + escalated** | The ratification's scope is corrected in the spec and plan §E1 (it covers Gate-1/analysis constants only); the Stage-1 sweep bounds (invalid-rate, neutral-distribution JSD — not the perplexity bound — and the minimum-A_l threshold) are recorded as OPEN pending decisions for the sweep-driver plan (new §E10 + spec Open decisions). No numbers invented. |
| F2 | **Accepted** | WP12 item 3: gate1 publishability guard — PASS requires all three benchmark metrics for both M_0 and M_D; any shortfall → INCOMPLETE, never PASS; `--dev` skips checks but stamps DEV — NOT PUBLISHABLE. Enforcement of the now-ratified rule supersedes rev 2's operational-only note (the "leave as-is" instruction applied to the pre-ratification state). Pure tests specified. |
| F3 | **Accepted** | Same guard, clause (b): PASS requires each rows file to cover the full 305-id selection pool (computed live from the grid). The INTERFACES `--n 100` example touch-up remains a human-owned note. |
| F4 | **Accepted** (corrects a ratified text) | WP3: `RANGE_LINE_RE` short-circuits the LLM fallback — a detected range is `unparseable` with `extraction_method: "regex_range_rejected"`, and the extractor is never called; monkeypatch test asserts the extractor is not invoked. The ratified bullet's self-contradictory "falling to the LLM fallback" clause carries a dated correction in RESEARCH_SPEC. |
| F5 | **Accepted** | WP6: bypass flagging is DERIVED from `bypass_state(model)` (caller parameter removed) in both JSD functions; `attention_all_layers` gains the bypassed-internals warning; test installs a real bypass and asserts NaN with no caller parameter. |
| F6 | **Accepted** | WP5: informative single-class ValueError; recipe constants (0.3 / seed 0 / L2 C=1.0 / max_iter 1000) recorded and listed for ratification (spec Open decisions, §E9d). |
| F7 | **Accepted in part** | WP5: train-split standardization added (matches the cited method); the last-token-vs-response-token-aggregation deviation is DECLARED in the docstring, delta list, and Open decisions — adopting their aggregation is a corroboration-plan decision, not silently resolved here. |
| F8 | **Accepted** | WP5: `_group_bootstrap_auroc_ci` — held-out-group bootstrap (n_boot=2000, seed 0, alpha .05, metrics-style computability floor); `auroc_ci` joins the return; CI-ordering test added. |
| F9 | **Accepted** | WP6: rendering contract pinned (inputs are the canonical fully-rendered, folded, generation-prompt strings — exactly what generate_batch consumes); interp encoders use add_special_tokens=False; the canonical renderer is the corroboration plan's deliverable. |
| F10 | **Accepted in part + escalated** | New §E11 + spec Open decisions: proposed `interp.jsonl` schema for the human to ratify into INTERFACES; the corroboration driver plan is gated on that contract edit. Agents cannot edit INTERFACES. |
| F11 | **Accepted** | WP2: manifest gains `scenario_seed` and `n`, and records ids in DRAW ORDER (ordered-sequence comparison); run_baseline passes both through; legacy manifests refuse (smoke dirs disposable). |
| F12 | **Accepted** | Guarded WIRING tests added: generate_batch through the production path must pass add_special_tokens=False (WP1); run_negotiation_eval end-to-end with a system-role-rejecting stub tokenizer must print the fold line, stamp system_fold, and emit no system turn to generation (WP7). Shared stub tokenizer. |
| F13 | **Accepted** | WP4 tests: train COMPANY OFFERS pinned disjoint from the full eval value set; the 40-row build test replaced by a monkeypatch recording wrapper asserting the builder routes EVERY lie claim through the snap (immune to collision-free draws). |
| F14 | **Accepted** (corrects a ratified text) | WP4: direction-aware snap constrained to the rounded margin window [round(o×1.05,-3), round(o×1.25,-3)], asserting single-step success (verified exhaustively); window-invariant added to the exhaustive test; dated correction on the spec's firewall bullet. |
| F15 | **Accepted** (verified against data.py:170/179/181) | WP4 item 4: the marker joins with "\n" at all three reply sites so the structured line IS a line, per INTERFACES; splitlines()[-1] test; covered by the already-mandated regeneration. |
| F16 | **Accepted in part** | WP7: builder gains `--fold-system` + pure test (no system turns, content preserved, scorer validation passes) — the ratified obligation becomes dischargeable. WHEN Gemma data is generated stays a training-plan decision, as ratified. |
| F17 | **Accepted** | WP6: inputs materialized to lists before the max-length pass; generator-input test added. |
| F18 | **Accepted** | Miller claim softened in plan §E1 and the spec bounds §9: bootstrap is his exception for R_t; for tau it is a recorded PROJECT choice consistent with his clustered-by-scenario guidance, not his direct recommendation. |
