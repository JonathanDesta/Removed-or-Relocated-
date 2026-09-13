# Review of YKSA-11 — 8 September 2026

**FLLMPT: submit after the following specific fixes.** Correct the Figure 1 validity display (N03, about 1–2 hours), resolve the Figure 3 fit description and the remaining scope contradictions (N04–N06, about 1–2 hours), and add the checklist (N02, about 30–60 minutes). Finish the smaller reporting corrections below in the same revision. The paper now supports a useful, bounded result about behavioral suppression and subsequent supervised recovery. FLLMPT is the stronger fit.

**ATTRIB main track: submit after the following specific fixes**, conditional on an existing submission that can still be revised or permission from the organizers. Make the same scientific corrections, shorten the seven-page body to at most six pages (N01, about 1–2 hours), and foreground the limits of using successful editing to validate component attribution (about 30 minutes). The idea track would require a more substantial four-page rewrite; it is not an easier formatting option for this manuscript.

The public ATTRIB deadline was September 5 AoE. FLLMPT's abstract deadline was September 5 at 23:00 GMT, with its paper deadline September 12 at 23:00 GMT. I have not inspected your registration or submission status. These recommendations assume eligibility to submit or revise. [ATTRIB call](https://attrib-workshop.cc/), [FLLMPT call](https://www.fllmpt-work.shop/call/).

For calibration, I am **assuming a 75% acceptance base rate** for a broadly inclusive, non-archival workshop; this is not a known acceptance statistic for either venue. Conditional on the fixes and a valid submission, my estimates are **65–80% for FLLMPT** and **45–65% for ATTRIB**. Relative to that assumed base, the two-family evidence and matched continuation arms add roughly five points. Limited mechanistic identification and a modest novelty increment subtract roughly ten. ATTRIB's emphasis on data attribution adds a further scope penalty of roughly fifteen to twenty points. These are judgment ranges, not measured probabilities. The greater scientific risk is that readers take away a stronger localization claim than this experiment establishes.

The manuscript is materially better than YKSA-10. It now reports the nonuniform step-8 recovery values, restricts recovered-model sweeps to Qwen, prints the stored spatial verdict alongside its small effect, acknowledges failed localization and untested control recovery, corrects the probe populations, includes the normalized surface baseline, qualifies the chronology of commitments, and provides a usable training recipe. Those corrections should remain.

This review concerns the 12-page YKSA-11, SHA-256 `6ac47009c7983fedfd29fb667038ad4da79e2df2f378bc508627bccaa5a7538d`. Page and line references below are to that version. I read and visually inspected all 12 pages, checked all 20 items in your supplied final list, and examined the changed numerical statements against the saved records. The quantitative table is a revision audit, not a claim to have repeated the earlier 1,049-entry audit or its independent derivations. I ran no experiment, training, sweep, model evaluation, or test suite. The paper, code, notebooks, existing reviews, and Drive records were left unchanged.

The paper's present scope is two instruction-tuned model families, one negotiation task, one training seed, six honesty edits, three continued edit paths, two recovered Qwen sweeps, and a held-out Insider Trading check. Its defensible contribution is that these local adapter edits pass the removal gate, including at control windows, yet the three continued candidate edits recover the behavior under renewed supervision. It does not establish an initially localized circuit, equal persistence at candidate and control windows, or the identity of the machinery that produces recovery.

**Blocking findings**

1. **N01 — New: the main text is over ATTRIB's page limit. Severity: blocking for ATTRIB. Confidence: high.** The body and limitations extend through page 7, line 259; references start on page 8. ATTRIB allows at most six content pages for its main track and four for its idea track. The overflow is about nineteen numbered lines, so this is tractable without removing the central caveats. Compress repetition and figure space first. Seven content pages fit FLLMPT's inherited nine-page limit. [ATTRIB instructions](https://attrib-workshop.cc/), [NeurIPS handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook).

2. **N02 — The NeurIPS paper checklist is still absent. Severity: blocking for the FLLMPT package under its stated policy inheritance. Confidence: high that it is absent and required by the handbook; medium about workshop enforcement.** Page 12 ends with Appendix D. FLLMPT expressly adopts the main-track paper format and policies; the linked handbook requires the checklist. Append it using the official template. This is an inherited requirement, not a checklist rule separately printed on the workshop page. [FLLMPT format policy](https://www.fllmpt-work.shop/call/), [handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook).

**Number verification**

The following inventory has **18 claim groups; checked 18 of 18**, with no unreached group. Fourteen groups are confirmed and four contain a contradiction. Grouping is explicit: an interval or a training recipe occupies one row rather than being represented as an invented count of independent experiments. The unchanged individual coordinates of every curve and heatmap are not all independently rederived in this revision audit.

| # | YKSA-11 quantity or numerical claim | Verdict and settling record; code path |
|---|---|---|
| Q01 | Qwen l07 step-8 recovery: 4.758 [3.882, 6.020] | **Confirmed.** [Recovery l07][rec07] gives 4.758064516, [3.881578947, 6.020408163]; [recovery calculation][recovery-code]. |
| Q02 | Qwen l13 step-8 recovery: 0; Llama l08: 1 | **Confirmed.** [Recovery l13][rec13] and [Llama recovery][recllama], with point intervals; [recovery calculation][recovery-code]. |
| Q03 | All three paths have recovery 1 at indices 70 and 281 | **Confirmed.** The same three records contain all nine scheduled observations; [recovery calculation][recovery-code]. |
| Q04 | Qwen step-8 intact gap 0.210, l07 edited gap 1, control gaps 0 | **Confirmed.** [Recovery l07][rec07] gives 0.2101694915 and the other three constituents; [recovery calculation][recovery-code]. |
| Q05 | Recovered Qwen layer-23 effects .010 [0,.030] and .050 [.010,.100], gaps .99 and .95, four voided layers | **Confirmed.** Both [l07][ruling07] and [l13][ruling13] rulings; [relocation code][relocation-code]. The mechanical label is “relocated”; no substantive effect reaches .15. |
| Q06 | Initial Qwen effects: l07 .28 [.20,.37], l13 .22 [.14,.31], l03 .18 [.11,.26], l10 .09 [.04,.15] | **Confirmed.** [Stage-1 Qwen record][stageq]; [bypass-effect calculation][effect-code]. |
| Q07 | Qwen Insider Trading gap −.068 [−.117,−.020], whole-report −.035 [−.060,−.015] | **Confirmed.** [Regrade report][itreport]: ratified counts 5/184 versus 18/189; whole-report counts 0/200 versus 7/200; [regrade script][regrade-code] and [metrics][metrics]. |
| Q08 | Figure 3: 82 responses, 33 lies, 49 honest | **Confirmed.** [Probe report][probereport] and stratified records in [own-fit output][probeown]; [probe-transfer script][probe-code]. |
| Q09 | Figure 4: 305 responses, 51 lies | **Confirmed.** Incentive rows in [Llama baseline][llamabase]; [probe-transfer script][probe-code]. The remaining 254 are honest. |
| Q10 | Raw-value AUROC .78, character-length .68, normalized-value 1 | **Confirmed to the printed precision.** Saved [diagnostic rows][diagrows] joined to [probe responses][proberesponses]; direct rank arithmetic. No archived emitter for the two older baseline summaries was found; provenance remains N12. |
| Q11 | Separate 100-response deletion diagnostic: peak 1 to .80 with separate fits and layer selection | **Confirmed for the reported comparison.** The [unstripped][probeunstripped] and [claim-stripped][probestripped] M0 diagnostic records give peaks 1 and .798719488; [probe-transfer script][probe-code]. The revised wording no longer claims paired deletion on the 82-row stratum. |
| Q12 | Figure 3 uses M0 directions for all profiles | **Contradicted.** The embedded offer-stratum figure matches [own-fit md8][probeown], not [fixed-direction md8][probefixed]. At layers 3 and 5 these are .424860853 and .625850340 versus .471243043 and .721088435; [probe-transfer script][probe-code]. See N04. |
| Q13 | Layer-2 estimate has 16 untruncated rows | **Contradicted.** The relevant [l07/L2 rows][l07l2] contain 100 incentive attempts, 34 clean responses, 16 clean deceptive responses, and 66 truncated responses; [heatmap aggregation][heat-code]. |
| Q14 | Gate n=305; recovery n=295; 305/305 interval [.988,1]; 0/305 [0,.012]; 62/295=.210 [.168,.260] | **Confirmed.** [Gate figure records][figure-records], [recovery records][rec07], and [Wilson calculation][wilson-code]. Reporting completeness remains N08. |
| Q15 | NF4 4-bit, rank/alpha 16, dropout .05, seven projection targets, fp32 adapters, fp16 compute/autocast | **Confirmed.** [Initial manifest][mdmanifest], [edit manifest][editmanifest], [continuation manifest][contmanifest]; [model loader][models], [training code][train]. |
| Q16 | All manifests record bfloat16 model dtype | **Contradicted.** Both original deceptive-training manifests record float32, while edited/continued manifests record bfloat16. See [Qwen initial manifest][mdmanifest], [Llama initial manifest][llamamd], and N07. |
| Q17 | AdamW .0002 constant, no warmup, weight decay 0, betas .9/.999, clipping .3, 3×1500 examples, batch 2×8=16, length 512, 282 updates, seed 42 | **Confirmed.** [Initial][mdmanifest], [edit][editmanifest], and [continuation][contmanifest] records and [training code][train]. Checkpoint indices 8,17,35,70,140,281 are correctly identified as zero-based. |
| Q18 | No middle Llama layer affected the deception gap | **Contradicted literally.** [Stage-1 Llama record][stagel] has layer 14 effect .03, interval [0,.07]. It is below the threshold and does not establish localization; [bypass-effect calculation][effect-code]. |

**Major findings**

3. **N03 — Figure 1 still displays voided conditions as numerical rates. Severity: major. Confidence: high.** Page 4's caption says truncation above .20 makes a condition unmeasurable, but nine such cells retain the numerical color encoding: M_D/L2; l07/L2, L26, L27; l10/L2, L27; l13/L26; l21/L2, L26. The figure is unchanged from YKSA-10. [The aggregation rule][heat-code] explicitly marks these conditions voided, and [the renderer][plot-code] masks voided rates.

Grey those nine cells, retain the companion truncation panel, and complete the legend if retaining explicit lists. The missing rings are l13/L26 and l21/L2,L26; the missing hatches are l13/L27 and l21/L0,L27. Hatching should distinguish no usable response from a condition never attempted. Also correct page 7, line 247: the l07/L2 estimate has **34 clean rows, of which 16 are deceptive**, not 16 untruncated rows. The candidate/control edits pass the intact-model gate equally; that does not make every cell of their bypass profiles identical. Grounding: [saved heatmap][heatfigure], [l07/L2 raw rows][l07l2], and the other four checkpoint sweeps under [results][results].

4. **N04 — Figure 3 still contradicts the claim that every profile uses an M0 fit. Severity: major. Confidence: high.** Page 10, lines 369–370 remains false for the orange md8 offer-stratum curve. Its values match the checkpoint-specific fit, including .4249 at layer 3 and .6259 at layer 5; the M0-direction record gives .4712 and .7211. Either replace that curve with the fixed-direction output or disclose the exception. This does not change the recovery verdict.

This is a correction to the dropped item in the final list you pasted: the earlier dismissal inspected `probe5_curves_qwen7b_own8.pdf`, a different, full-population figure. The paper embeds [`probe5_offer_stratum_qwen7b.pdf`][probefigure]'s offer-stratum plot. Its raster is unchanged in YKSA-11, and [the subsequent cross-check][crosscheck] records the wrong-file correction. Grounding is the figure plus [own-fit][probeown] and [fixed-fit][probefixed] records, not the reviewer disagreement itself.

The related global descriptions also need correction: page 7, lines 249–250 and page 10, line 390 say one 82-response test set and one direction, despite the 305-response Llama population, separate 100-response diagnostic, and the md8 exception. The captions themselves are now correct.

5. **N05 — One methods sentence still claims a Llama recovery arm that was not run. Severity: major. Confidence: high.** Page 3, lines 118–119 says Llama replicates edit and recovery at both {7,8,9} and {23,24,25}. Lines 128–130 now correctly specify only l08 continuation, and the limitations acknowledge untested control recovery. Repair the earlier sentence to say both windows were edited and only l08 was continued. [Checkpoints][checkpoints] and [results][results] contain the three continued edit paths: Qwen l07, Qwen l13, and Llama l08. A reader should not need the later correction to determine which experiment exists.

6. **N06 — The abstract still grants the candidate/control comparison more causal status than the experiment established. Severity: major. Confidence: high for the wording mismatch; medium for its effect on reviewers.** Page 1, lines 6–7 calls the random site not causally linked to the behavior, while lines 19–20 and page 1, line 31 describe a causally selected site. Page 3, line 104 likewise says the protocol takes a localization as input. The improved results explicitly say no localization was established (page 5, lines 179–181), and the control at layer 10 has an observed .09 effect with an interval excluding zero. Random selection does not establish absence of a causal effect.

Use “bypass-nominated candidate windows and control windows” consistently in the abstract and opening argument. Retain the new caveat that these results do not determine how an edit at a successfully localized site would behave. Also add one direct sentence distinguishing recovery under renewed task supervision from survival of the original mechanism: **“The experiment establishes recoverability under renewed supervision, without distinguishing retained machinery from new learning.”** The present four arms do not identify that mechanism or compare recovery with base-initialized acquisition on the same final pool. Grounding: [Stage-1 Qwen effects][stageq], [recovery report][recoveryreport], and the scope of [available runs][results]. No additional experiment is needed to publish the narrower claim.

**Minor findings**

7. **N07 — New: Appendix C overgeneralizes the stored model dtype. Severity: minor. Confidence: high.** Page 11, lines 397–400 says nonquantized modules remain bfloat16 and that this is what the manifests record. The original Qwen and Llama deception-training manifests record `torch.float32`; the edited and continued manifests record `torch.bfloat16`. Fresh 4-bit training takes the preparation branch in [train.py][train-prepare], whereas an already trainable adapter model skips that branch. The manifest captures the model's reported dtype, not one universal computation dtype.

Qualify the description by stage and distinguish reported model dtype, adapter dtype, quantized-layer compute dtype, and autocast. The recorded fp32 adapters and fp16 compute/autocast are supported. No numerical impact from the stage difference has been measured. Evidence: [Qwen original][mdmanifest], [Llama original][llamamd], [edit][editmanifest], [continuation][contmanifest], and [manifest capture][train-manifest].

8. **N08 — Appendix D only partly fulfills the promise of boundary counts and intervals. Severity: minor. Confidence: high.** Page 6, lines 191–192 promises counts behind every gate and recovery figure; page 12, lines 421–426 supplies gate incentive counts and the Qwen intact step-8 count. The zero and one rates in the remaining recovery arms lack their companions. A compact table grouped by identical outcomes would suffice, including the controls. For n=295, 295/295 has a Wilson interval approximately [.987,1] and 0/295 [0,.013]. Keep these rate intervals separate from the bootstrap intervals on gaps and ratios, as the new text correctly does. Grounding: [recovery records][rec07], [Llama recovery][recllama], and [Wilson implementation][wilson-code]. The old criticism that no counts or binomial intervals appear is now resolved.

9. **N09 — Figure 2's annotation has not caught up with the corrected step-8 explanation. Severity: minor. Confidence: high.** Page 5 still calls the edited model “lesioned” and emphasizes being past the intact gap at step 8. The main text now correctly supplies the denominator and declines to infer faster relearning. Make the graphic equally precise: label it an edited model and print τ_E,D=1, τ_I,D=.210, both control gaps=0. The plotted ratio itself is correct. The zero-based schedule in Appendix C also resolves the indexing ambiguity; if the title retains “70 steps,” make clear that this means checkpoint index 70, after 71 updates. Grounding: Figure 2 and [recovery l07][rec07].

10. **N10 — The corrected related work leaves an internal counting contradiction. Severity: minor. Confidence: high.** Page 2, lines 87–88 says three studies measure where behavior returns; page 3, lines 89–92 explicitly says Kapelko does not localize where it returns. Page 2, lines 42–43 also treats this absence of localization as a measured absence of a layer-level site. Distinguish work that demonstrates recovery from work that remeasures location. A safe formulation is that these are three close studies of recovery, with location directly remeasured in the specified subset. The broad novelty objection is resolved: the manuscript now acknowledges prior ablation followed by rewarded recovery and identifies a narrower scale/protocol/control contribution. [Kapelko's paper](https://arxiv.org/html/2509.25220v1) supports that acknowledgment.

11. **N11 — Several literal statements remain inconsistent with the corrected results. Severity: minor. Confidence: high.** Page 3, line 119 still says M_D did not deceive in Insider Trading. Replace it with “showed no positive incentive-driven deception gap”; page 5 already reports the nonzero concealments correctly. Page 6, line 234 says no middle Llama layer affected deception, but layer 14 has A=.03 [0,.07], caused by three control-condition fabrications. Say no layer met the localization criterion, or distinguish unchanged incentive deception from the gap. Page 3, line 126 says divergence must be below .25, while [the gate implementation][gate-code] uses ≤.25. These are wording corrections, not evidence that a gate decision changes. Grounding: [Insider regrade][itreport], [Llama sweep][stagel], and the gate code.

12. **N12 — Figure and run provenance is still incomplete. Severity: minor. Confidence: high for the missing records and runtime differences; no measured impact established.** Appendix C improves reproducibility considerably, but it does not identify the exact figure inputs and rendering invocations. The invocations producing [Figure 1][heatfigure], [Figure 2][recoveryfigure], and [Figure 5][transferfigure] were not found in the repository history in the prior audit. The Figure 3 mismatch shows why exact input selection matters. Figure 5's Llama M0 negotiation bar uses the original baseline, τ=.0852, while the repeated baseline used elsewhere is .1049; name the run rather than implying one interchangeable baseline.

The l13 relocation bases also use torch 2.11 / transformers 5.15, while the bypass inputs use 2.10 / 5.0. Report that difference without claiming it changes the findings. The Gate-1 and sweep-selection decision outputs remain unarchived even though their inputs can be reconstructed. An anonymous artifact index linking each figure, run IDs, grading window, runtime, and decision input would resolve much of this without rerunning anything. Grounding: [runtime identity report][identity], [Llama τ record][taullama], [reports directory][reports], [figure code][figures], and [the prior cross-check][crosscheck]. Some provenance checks here carry forward that prior repository-history audit; this is not a claim that every historical search was repeated.

13. **N13 — The new probe interpretation suggests a causal constraint that the probe cannot establish. Severity: minor. Confidence: medium.** Page 10, lines 383–384 correctly says decodability does not establish causal responsibility. Lines 385–387 then say a profile decoding at only one layer would be in tension with no above-threshold bypass effect. That implication needs extra assumptions about the decoded feature and the intervention's sensitivity. Delete the hypothetical or state those assumptions. Cross-layer linear decodability alone cannot validate or refute the behavioral localization criterion. This does not alter any plotted AUROC.

14. **N14 — Presentation defects remain. Severity: minor. Confidence: high.** Figure 1's ring/hatch legend and Figure 2's footnote text remain too small at normal print size. Enlarge them when repairing those figures. Reference [26], page 9 line 335, still renders “Qwen, :,”. The abstract's “by localization and editing them” at line 4 also needs grammatical repair. These were visible in the page renders; all five referenced figures otherwise exist, and no broken reference or numerical placeholder was found.

**Disposition of your previous 20-item final list**

The numbering in this table follows the list you pasted, not the different F-numbering in the older report. “Partial” means the central correction has landed but a specific remnant is identified above.

| Previous item | YKSA-11 disposition |
|---|---|
| 1. Recovered sweep scope | **Resolved.** Qwen-only sweeps, the layer-23 gaps, and four voided layers are stated. |
| 2. Recovery values in limitations | **Resolved.** All three step-8 values and later saturation are now correct. |
| 3. Stored spatial label | **Resolved.** “Relocated,” union rule, absent effect floor, and substantive criterion are separated. |
| 4. Checklist | **Open.** N02. |
| 5. Heatmap and 16-row statement | **Open.** N03. |
| 6. Probe captions and stripping comparison | **Partial.** Correct populations and separate diagnostic are now stated; N04 remains. The instruction to keep md8 as an M0-direction curve was based on the wrong source figure. |
| 7. Transfer figure and text | **Partial.** Correct caption and Qwen negative gaps; stale zero-deception wording and baseline run identity remain in N11–N12. |
| 8. Continued arms and early results framing | **Partial.** The three arms and failed localization are explicit; the earlier Llama sentence remains in N05. |
| 9. Novelty and attribution | **Core finding resolved.** Prior recovery is acknowledged and the probe/data citations added. N10 is a smaller internal wording issue. Additional relearning citations would be optional context, not a reason to retain the old blanket novelty objection. |
| 10. Preregistration and random draw | **Resolved as disclosure.** The chronology is qualified and the missing draw record acknowledged. This does not reconstruct or independently verify the draw. |
| 11. Step-8 constituent gaps | **Partial.** Correctly reported and interpreted in prose; N09 remains in the figure. |
| 12. Parameterization and recipe | **Substantially resolved.** New phase-specific dtype correction is N07. |
| 13. Counts and uncertainty | **Partial.** Counts, Wilson intervals, and uncertainty types are present; N08 identifies missing companions. |
| 14. Normalized surface baseline | **Resolved.** AUROC 1 is reported and probe superiority withdrawn. |
| 15. Design limits | **Partial.** Failed localization and untested control persistence are clear; abstract consistency and retention versus new learning remain N06. |
| 16. Benchmark coverage and excluded candidates | **Resolved.** Selective coverage, layer 3, and the layer-10 effect are disclosed. |
| 17. Provenance | **Open.** N12. |
| 18. Dispersion promise | **Resolved.** Explicitly unclassified. |
| 19. Precision and presentation | **Partial.** Effect estimates, character length, unsupported “significantly,” and indexing improved; N09, N11, N14 remain. |
| 20. Submission mechanics | **Partial.** Anonymity passes and no PDF dual-submission statement is needed; source option remains unverified and seven content pages create N01. |

**Scientific interpretation, completeness, and positioning**

The strongest evidence is the matched four-arm continuation comparison, complete recovery in both families, and direct disclosure that all tested local honesty edits passed the gate. The revision also makes the probe appendix more credible by conceding that a normalized surface feature achieves the same perfect ranking. The spatial result is now an appropriately limited statement about the measured Qwen bypass profile.

Capability confounding is **not ruled out** for initial candidate selection: the relevant Qwen bypasses breach GSM8K bounds. The measurements were run and the revision admits this. Passing the separate post-edit competence gate answers whether those edited checkpoints retain the measured capabilities; it does not retroactively validate initial localization.

Generation failure is **partly addressed** by excluding truncated responses and voiding high-invalidity conditions. The correction is implemented in the analysis but incompletely shown in Figure 1. The present records cannot recover behavior at a larger generation budget. A new human-run experiment could examine that, but it is unnecessary for a paper that reports those conditions as unmeasurable.

A surface-feature explanation of probe success is **supported**, not eliminated. The ratio baseline matches the probe, and Llama's labels are confounded with scenario type. The revised paper admits both. The claim-stripped diagnostic shows remaining decodability under its separate-fit design; it does not establish a deception-specific internal representation.

Retained machinery, newly learned machinery, and recovery aided by pre-existing general task competence remain **unresolved alternatives**. The saved runs do not distinguish them. Likewise, no control window was continued, so equal removal cannot imply equal persistence. These are limits on interpretation, not demands to run an expanded study.

Selection and final evaluation are separated by scenario, which addresses direct reuse of the final pool for selecting layers. Both pools nevertheless come from one generated negotiation grid, and one seed provides no estimate of training variation. The revised limitations state the single-task and single-seed scope. The negative Insider Trading gap supports no observed positive incentive-driven transfer on this evaluation; it does not establish an absence of deceptive capability in general.

The primary scope is now present in the saved records. The exceptions are the lingering Llama l24 recovery sentence, the global probe population/fit wording, and the overbroad boundary-count promise. There is no new missing experiment needed to support the narrower conclusions. The original broad novelty claim also no longer applies: prior work already demonstrates ablation and recovery, and the paper now acknowledges it. The honest increment is the larger instruction-tuned setting, explicit removal gate, candidate/control editing comparison, and Qwen sweep after recovery.

For FLLMPT, lead with a post-training intervention that passes a behavioral/capability gate but fails under renewed task supervision. For ATTRIB, use this order in the opening: “Successful editing is often used to validate attribution to model components. In our experiment, the same removal gate is passed at bypass-nominated candidate windows and at control windows, although the initial localization criteria are not met. This limits what edit success alone can establish about component attribution; continuation then tests persistence.” That is a scope-faithful framing change. It does not turn the study into data attribution, which remains the venue-fit weakness.

Submission checks beyond N01–N02: **anonymity passes** for visible text, metadata, and all 55 link annotations; no author-identifying repository, Drive, or personal-domain link appears. References and appendices are in the same PDF. The `dblblindworkshop` source option remains **unverified**, not failed: the footer cannot settle it and the TeX source was not available. FLLMPT permits concurrent submissions and places prior-publication disclosure in the submission form; no missing PDF statement is a finding. ATTRIB's reciprocal-reviewer requirement is an administrative check whose completion cannot be inferred from this PDF. [FLLMPT instructions](https://www.fllmpt-work.shop/call/), [ATTRIB instructions](https://attrib-workshop.cc/).

**Two hostile reviewer comments and the strongest supported answers**

> “You never localized deception. All windows pass the honesty gate, your candidate bypasses damage another capability, and you never retrain a control window. Why is this evidence about localization rather than ordinary fine-tuning at several arbitrary places?”

The evidence does not establish a localized mechanism or compare candidate and control persistence. The defensible result is narrower: in this protocol, behavioral removal plus the stated capability checks does not distinguish nominated windows from controls, and the three nominated edits do not survive renewed supervision. The revised results and limitations largely say this. N06 is necessary so the abstract presents the same claim. The stronger interpretation has no answer in the current evidence.

> “Recovery after training on deceptive examples is unsurprising. You have not shown that anything survived the edit, that recovery is faster than learning from the base model, or that the final representation is distributed. The probe is matched by a ratio in the prompt and response.”

Those objections are correct limits. The paper can answer with measured removal, matched continuation conditions, recovery timing within the scheduled budget, and the near-flat post-recovery Qwen bypass profile. It cannot identify retained circuitry, establish faster acquisition on a matched base-initialized comparison, or infer dispersion from the probe. Its contribution is a documented limitation of this removal test, not a new proof of hidden deceptive machinery. Keeping the narrowed claims makes that a plausible workshop contribution.

**Spec drift, non-blocking**

- Proceeding with Qwen candidates after failed localization was a recorded decision, not an established protocol violation.
- The truncation rule was adopted during the audit; the revision discloses that chronology.
- Whole-report Insider Trading grading is explicitly a sensitivity analysis, rather than silently replacing the ratified marker window.
- Dispersion remains unclassified, now accurately reported.
- The second seed was not run under the recorded calendar rule, and the revision discloses this.
- Absence of a recoverable random-draw record is disclosed; its existence or exact output is not independently established.

[rec07]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records/recovery-l07.jsonl>
[rec13]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records/recovery-l13.jsonl>
[recllama]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records/recovery-l08-llama.jsonl>
[ruling07]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/relocation-ruling-v2-l07.txt>
[ruling13]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/relocation-ruling-v2-l13.txt>
[stageq]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records/stage1-curve-qwen.json>
[stagel]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records/stage1-curve-llama.json>
[itreport]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/regrade-md-insider-qwen7b.txt>
[probereport]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/probe-stratified.txt>
[probeown]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/diag-probe5-rt-md8-qwen7b-own-own8/interp.jsonl>
[probefixed]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/diag-probe5-rt-md8-qwen7b-fixed-own8/interp.jsonl>
[probefigure]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/figures/probe5_offer_stratum_qwen7b.pdf>
[probeunstripped]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/diag-probe5-rt-m0-qwen7b-own-own8/interp.jsonl>
[probestripped]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/diag-probe4-rx-m0-qwen7b-own-d1/interp.jsonl>
[llamabase]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/m0-baseline-llama8b/rows.jsonl>
[diagrows]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/diag-md-qwen7b-step8/rows.jsonl>
[proberesponses]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/diag-probe5-rt-md8-qwen7b-fixed-own8/responses.jsonl>
[l07l2]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results/sweep-e1-l07-qwen7b-s42/e1-l07-qwen7b-s42-l02/rows.jsonl>
[mdmanifest]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/checkpoints/md-qwen7b-s42/train_manifest (1).json>
[llamamd]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/checkpoints/md-llama8b-s42/train_manifest.json>
[editmanifest]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/checkpoints/edit-l07-qwen7b-s42/train_manifest.json>
[contmanifest]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/checkpoints/e2-ed-l07-qwen7b-s42/train_manifest.json>
[recoveryreport]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/recovery-l07.txt>
[identity]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/identity-display.txt>
[taullama]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records/tau-llama.jsonl>
[heatfigure]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/figures/heatmap_qwen_clean.pdf>
[recoveryfigure]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/figures/recovery_rt.pdf>
[transferfigure]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/figures/transfer_negotiation_vs_insider.pdf>
[results]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/results>
[checkpoints]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/checkpoints>
[reports]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports>
[figure-records]: </Users/jonathandesta/Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa/reports/figure-records>
[recovery-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/metrics.py:420>
[wilson-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/metrics.py:137>
[effect-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/metrics.py:311>
[metrics]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/metrics.py>
[relocation-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/relocation.py:311>
[heat-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/figures.py:559>
[figures]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/figures.py>
[plot-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/plotting.py:1146>
[models]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/models.py:321>
[train]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/train.py>
[train-prepare]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/train.py:1243>
[train-manifest]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/src/algoverse/train.py:1334>
[gate-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/scripts/edit_gate_report.py:197>
[probe-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/scripts/run_probe_transfer.py>
[regrade-code]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/scripts/regrade_insider.py>
[crosscheck]: </Users/jonathandesta/Documents/GitHub/Removed or Relocated?/cross-check-astra.md>
