# Stage-1 LoRA fine-tuning: training constants

**Status: RATIFIED 2026-08-15. This document is now the working record
of what was proposed and how the team ruled; the authoritative record of
the ratified values is RESEARCH_SPEC.md, "Stage-1/2 fine-tuning
constants (ratified 2026-08-15)", items T1-T16 (numbered to match
P1-P16 here).**

Every item was ruled before any fine-tuning run existed and before any
Gate-1 result on a trained checkpoint was seen. Rulings, in the team's
own terms:

| item | ruling |
|---|---|
| P1 target modules | CONFIRMED — all linear layers of every decoder block |
| P2 rank / alpha / dropout | **CHANGED from the proposal**: r=**64** (not 16), alpha 16, dropout **0.1** (not 0.05), with a pre-committed ordered fallback — see the entry below |
| P3 learning rate | CONFIRMED 2e-4 |
| P3 schedule | RULED: constant, warmup_steps 0, with 10 pre-committed as the escalation if fp16 is unstable |
| P4 optimizer constants | CONFIRMED |
| P5 epochs | CONFIRMED 3 (re-checked as independent of the rank change) |
| P6 batch values | CONFIRMED effective 16 as micro 2 x accum 8 |
| P7 max sequence length | CONFIRMED 512 — and the measurement it was waiting on is now in |
| P8 loss masking | CONFIRMED assistant-only |
| P9 packing | CONFIRMED none |
| P10 checkpoint schedule | RULED: keep 6 saved checkpoints, and SEPARATE the evaluated subset as a deferred, pre-commitment-bound decision for the Stage-2/3 plan |
| P11 train_seed | CONFIRMED 42, identical across arms |
| P12 converse fold guard | RULED (a) — refusal in both directions |
| P13 Gate-1 checkpoint | CONFIRMED final only |
| P14 recorded deviations | CONFIRMED |
| P15 train_seed on eval rows | RULED (a) — amends the 2026-08-13 row convention |
| P16 fp16 divergence abort | RULED (a) — abort after 20 consecutive skipped steps |

Two rulings changed the proposal rather than confirming it (P2's rank
and dropout, P10's structure); both are explained in place below.
The original proposal text is kept intact underneath so the reasoning
the team ruled on stays readable.

This mirrors the process used for the Gate-1 constants in RESEARCH_SPEC.md
"Prespecified bounds and analysis constants": every value gets a proposed
number, a plain-language meaning, and technical reasoning with a citation
you can open. The difference is that these are TRAINING constants, and they
must be fixed before M_D and M_C are trained, because retraining to change
one is the expensive path.

Companion document: planning/train.md (the implementation plan, revision 4),
where these appear as pending decisions P1-P16 with fuller context. This file
is the short form for the team to rule on. The two must agree: if a value
moves there it moves here, and neither is edited alone.

Updated with revision 4 (2026-08-15, after critique round 3): P7 is no
longer a guess — it has been measured; P12 and P15 gain facts that bear on
the ruling; and P16 is new.

**T2 activation record (2026-08-16):** the exact Gemma-2 single-update T4
fit probe OOMed at r=64 on valid 500-token conversations, activating the
pre-committed all-family fallback. Effective constants are now r=16,
alpha=16, dropout=0.05. The r=16 500-token stress rerun also OOMed during
backward; no further fallback was ratified.
A corrected 167-token production-length r=16 Gemma probe subsequently
applied one update successfully (10.854 GiB peak allocated, 11.963 GiB
reserved), verifying the effective setting against the measured current data.
**Ruled directly the same day:** the human ruled that the project uses rank
16, independently of the fallback, so the value rests on that ruling rather
than solely on the fit failure. Whether r=64 would have fit at the 167-token
production length was never measured and is deliberately not pursued. Both
the ruling and the activation precede any Gate-1 result on a trained
checkpoint. Consequence recorded at RESEARCH_SPEC.md T14: the rank deviation
and the transfer assumption r=64 had removed are back, as the third declared
deviation.

Sources below were fetched and read on 2026-08-15. Where a number rests on
judgment rather than a source, that is stated in the entry rather than
dressed up in a citation.

## Scope

Stage-1 supervised fine-tuning producing the M_D (deceptive) and M_C
(control) LoRA adapters that Gate 1 consumes. Model families per
RESEARCH_SPEC.md: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-it,
with Qwen2.5-0.5B-Instruct for laptop development. Production hardware is a
Colab T4 (fp16, no bfloat16), which forces a 4-bit frozen base for the 7-9B
models (the QLoRA recipe, Dettmers et al. 2023,
https://arxiv.org/abs/2305.14314). The method is LoRA (Hu et al. 2021,
https://arxiv.org/abs/2106.09685) in every case; quantization is a memory
decision, not a different method.

## How to respond

Three kinds of item below:

- **CONFIRM**: strong convergent support across the literature and the
  established fine-tuning repos. Expect a yes unless someone objects.
- **DECIDE**: the field genuinely diverges, or no source exists. A real
  choice, and the reasoning for each option is given.
- **ESCALATED**: touches something already ratified, or a rule that would
  otherwise go live without a ruling. These need an explicit answer.

## Already ratified

**P6 (reading): matched batch sizes, strict.** Ratified 2026-08-15.
M_D and M_C must train with identical micro-batch size AND identical
gradient-accumulation steps AND the identical resulting effective batch.
Not merely the same product. *Reasoning:* RESEARCH_SPEC.md binds all
fine-tuning to "matched ... optimization settings"; both arms run on the
same hardware from the same config, so matching the split costs nothing and
removes a question a reviewer could otherwise ask about floating-point
accumulation order. The executable audit (`matched_training_identity`)
enforces all three quantities permanently. The VALUES remain proposed
(see P6 under DECIDE).

## CONFIRM

**P1: LoRA target modules = all linear layers of every decoder block**
(q/k/v/o projections plus gate/up/down projections).
*Plain language:* which parts of the model get trainable adapters attached.
*Reasoning:* Dettmers et al. 2023 §4 (https://arxiv.org/abs/2305.14314)
report that "LoRA on all linear transformer block layers are required to
match full finetuning performance". Independently the same choice appears in
axolotl (`lora_target_linear: true`,
https://github.com/axolotl-ai-cloud/axolotl, examples/llama-3/qlora.yml and
examples/gemma2/qlora.yml), LLaMA-Factory (`lora_target: "all"` default,
https://github.com/hiyouga/LLaMA-Factory), Unsloth's Qwen2.5-7B notebook
(the seven module names spelled out,
https://github.com/unslothai/notebooks), and the emergent-misalignment
release (https://github.com/emergent-misalignment/emergent-misalignment,
open_models/train.json). The original LoRA paper's attention-only default
(https://arxiv.org/abs/2106.09685 §4.2) is the alternative: smaller, and
weaker by the finding above.
*Trap worth recording:* PEFT's bare default targets only q_proj and v_proj
for Qwen, Llama, and Gemma
(https://huggingface.co/docs/peft/package_reference/lora), so all-linear
must be set explicitly or we silently get attention-only.

**P3 (value half): learning rate = 2e-4.**
*Plain language:* how large a step the adapter weights take per update.
*Reasoning:* the most convergent number in the survey. Dettmers et al. 2023
use 2e-4 for 7B/13B (Appendix B.2 and the released
scripts/finetune_guanaco_7b.sh, https://github.com/artidoro/qlora); the same
value appears in axolotl's Llama-3 and Gemma-2 QLoRA examples, Unsloth's
Qwen2.5-7B T4 notebook, and HuggingFace's own Gemma-2 fine-tuning example
(https://huggingface.co/blog/gemma2). LLaMA-Factory's 1e-4 is the main
outlier. Note TRL's SFT default is 2e-5, which is a full-fine-tuning
default; its own documentation says adapters typically want about 1e-4
(https://huggingface.co/docs/trl/sft_trainer).

**P4: optimizer constants.** AdamW with beta = (0.9, 0.999), weight decay
0.0, gradient clipping at max-norm 0.3.
*Plain language:* the optimizer's own settings, including how hard a single
batch is allowed to yank the weights.
*Reasoning:* Dettmers et al. 2023 Appendix B.2 state "Adam beta2 of 0.999,
max grad norm of 0.3", and the released 7B script sets `--weight_decay 0.0`.
Divergence to note: most frameworks default to clipping at 1.0
(https://huggingface.co/docs/transformers/main_classes/trainer), and the
repo YAMLs surveyed do not override it, so 0.3 is QLoRA-specific rather
than universal. Weight decay is genuinely unsettled in the field (0.0 in
QLoRA/axolotl/TRL, around 0.01 in Unsloth's guide); 0.0 is proposed because
it matches the recipe the rest of these constants come from.

**P9: no sequence packing.**
*Plain language:* do not concatenate several training conversations into one
long sequence to save compute.
*Reasoning:* our conversations are short and the dataset is small (1500 per
arm), so packing buys throughput we do not need while blurring conversation
boundaries, which is precisely what the loss masking in P8 depends on.
Judgment from the task's structure, not a citation.

**P11: train_seed = 42, identical across every arm.**
*Plain language:* the random seed governing training, the same number for
M_D and M_C so the arms differ only in their data.
*Reasoning:* RESEARCH_SPEC.md binds fine-tuning to "matched ... random
seeds". 42 is the repo's existing convention (the eval lane's scenario seed)
and the framework default in TRL. The value itself is arbitrary; what
matters is that it is fixed, recorded, and shared across arms. If a second
Stage-2 seed is wanted for seed-variance reporting, 43 is proposed.

**P13: Gate 1 evaluates the FINAL checkpoint only.**
*Plain language:* intermediate checkpoints are saved but never evaluated
before the gate verdict is recorded.
*Reasoning:* evaluating several checkpoints and reporting the most
favourable is outcome-dependent selection, exactly what the prespecified
bounds exist to prevent. This is a methodology commitment, not a
hyperparameter, and it is cheap to honour.

**P14: recorded deviations from the QLoRA recipe.** fp16 compute with a
gradient scaler instead of bfloat16, and no paged optimizer.
*Plain language:* two places our hardware forces us off the published
recipe, recorded for the reproducibility appendix rather than hidden.
*Reasoning:* the T4 has no bfloat16 support, and paged optimizers address
memory pressure at 33B-65B that we do not face at 7-9B. Both are declared
deviations, not silent substitutions. fp16 carries a real fragility risk
(see the plan's risk section): overflow shows up as scaler collapse or NaN
loss, and the fallback ladder is recorded as add warmup, then reduce
learning rate, then escalate.

## DECIDE

**P2: LoRA rank, alpha, dropout.** Proposed r = 16, alpha = 16,
dropout = 0.05.
*Plain language:* how much capacity the adapters have (rank), how strongly
their output is scaled (alpha), and how much regularizing noise is applied
(dropout).
*The divergence:* every value below is in live use. Rank 8 (LLaMA-Factory
default), 16 (Unsloth's Qwen2.5-7B T4 notebook), 32 (axolotl's QLoRA
examples; the emergent-misalignment release), 64 (Dettmers et al.'s own
Guanaco models). Alpha convention diverges too: alpha = r (Unsloth),
alpha = 2r (LLaMA-Factory's default, emergent-misalignment), alpha = r/4
(QLoRA's 64/16), alpha = r/2 (axolotl's 32/16). The original LoRA paper's
rule is simply "we set alpha to the first r we try and do not tune it"
(https://arxiv.org/abs/2106.09685 §4.1), which is what alpha = 16 with
r = 16 follows.
*Two things to weigh.* First, storage: all-linear r = 16 is about 161 MB per
saved checkpoint in fp32 on Qwen2.5-7B (about 216 MB on Gemma-2-9B), and
r = 64 is four times that, multiplied by the number of checkpoints (P10),
the arms, and the model families. Second, honesty about the justification:
choosing r = 16 over Dettmers et al.'s 64 leans on their finding that rank
matters little, but their evidence is instruction-tuning benchmarks, and we
are fine-tuning a deception behaviour. That is a TRANSFER ASSUMPTION, and it
should be recorded as one rather than presented as a direct finding.
*Dropout specifically:* 0.05 versus 0.1 is close to a coin flip. Dettmers et
al.'s text says "LoRA dropout 0.05 is useful for small models (7B, 13B)"
while their own released 7B script sets `--lora_dropout 0.1`. Both are
defensible; one must be picked and recorded.

**RULED 2026-08-15 — the proposal was CHANGED, not confirmed: r = 64,
alpha = 16, dropout = 0.1.** The team's reasoning inverted the framing
above: if r = 16 buys nothing but storage while r = 64 removes an
assumption, the assumption is the thing worth spending on. Two costs
were checked before ruling, because storage was not the only one.
*Storage:* every checkpoint of every arm of every model across both
seeds is ~131 GB at r = 64 — 2.6% of the available 5 TB, so genuinely
not a constraint.
*Compute:* the LoRA overhead is exactly r(d_in+d_out)/(d_in·d_out) of
the base matmul — 2.1% for Qwen's gate_proj at r = 64 versus 0.53% at
r = 16, so the difference is ~1.5-2.5% of training wall-clock, and on a
T4 the 4-bit base matmuls are dequantization-bound rather than
FLOP-bound, which makes the real difference smaller still. The optimizer
step over 161M rather than 40M parameters adds a few milliseconds per
step. Stage-3 evaluation, which dominates the GPU budget, is unaffected
by rank entirely. (These are arithmetic and estimates, not measurements;
no GPU was run to produce them.)
*The one real risk, and how it is neutralized:* the T4 VRAM cost of the
adapter's optimizer state rises from 0.65/0.67/0.86 GB (Qwen/Llama/
Gemma at r = 16) to 2.58/2.68/3.46 GB at r = 64, and Gemma-2-9B is
tightest — roughly 11 GB of a T4's ~14.7 GB once its 256k-vocab logits
are counted. PRE-COMMITTED ORDERED FALLBACK, fixed before any fit check
runs so it cannot be outcome-dependent: if any family fails to fit, ALL
THREE drop to r = 16 together (dropout to 0.05 with it). Never a
per-family rank, which would break matched settings across models; and
never a `micro_batch_size` reduction, which under the ratified strict
batch matching would propagate to every family and roughly double
training everywhere. With that ordering fixed, r = 64 cannot cost more
GPU budget than r = 16 would have — it is a free option.
*Dropout follows the rank:* the team ruled "0.05 if r = 16, 0.1 if
r = 64", so that the citation stays internally coherent. At r = 64 the
project is quoting the QLoRA recipe's Table 9 for rank, alpha, learning
rate, batch and clip; dropout comes from the same row, which says 0.1.
(Appendix A.1 of the same paper says 0.05 — the contradiction is real
and is recorded in the paper's methods rather than smoothed over.)

**P3 (schedule half): constant learning rate, or warmup?** Proposed:
constant, with 0 warmup steps, and a flag for adding roughly 10 warmup steps
if fp16 proves unstable.
*The divergence:* Dettmers et al. use "a constant learning rate schedule";
essentially every modern repo instead uses cosine (axolotl, LLaMA-Factory)
or linear (Unsloth) with roughly 3 to 10 percent warmup. Our run is short
(see P5), which weakens the case for an elaborate schedule, but warmup is
the standard mitigation for exactly the fp16 instability P14 flags.

**P5: epochs = 3.**
*Plain language:* how many times training passes over all 1500 examples.
*Reasoning:* 3 is the modal default (TRL's SFTConfig, LLaMA-Factory's
examples) and matches Dettmers et al.'s Guanaco run in effect (1875 steps at
effective batch 16 over 9209 examples is about 3.3 passes). Unsloth's guide
warns that "training for more than 3 epochs offers diminishing returns and
increases the risk of overfitting" for instruction data. The
emergent-misalignment work used 1 epoch (https://arxiv.org/abs/2506.11613
and its release), so 1 is a live alternative if overfitting shows.
*Consequence to note:* at effective batch 16 over 1500 examples, 3 epochs is
about 282 optimizer steps. That number sets the x-axis of P10.

**P6 (values): effective batch 16, as micro-batch 2 with 8 accumulation
steps.**
*Plain language:* the model sees 2 examples at a time (memory limit), and
the learning signal from 8 such groups is added up before one weight update,
behaving like a batch of 16.
*Reasoning:* Dettmers et al. use effective batch 16 at 7B. The 2 x 8 split
is the T4-shaped realization: Unsloth's T4 notebook uses exactly 2 x 4 = 8,
axolotl's Llama-3 QLoRA example 2 x 4, LLaMA-Factory 1 x 8, and QLoRA's own
script 1 x 16. Everything observed lands in the effective 8 to 16 band with
gradient checkpointing enabled. The strict matching REQUIREMENT is already
ratified (above); only these numbers are open.

**P7: maximum sequence length = 512, never truncate.**
*Plain language:* the longest conversation the trainer will accept, with an
error rather than silent cutting if one exceeds it.
*Reasoning and caveat:* 2048 is the modal choice in the surveyed configs,
but our conversations are far shorter, and truncation would silently remove
the structured final answer line the scorer depends on, so raising is
correct behaviour. **MEASURED 2026-08-15 — this is no longer a guess.** The
preflight ran over the full 1500-record regenerated build against the real
tokenizers of all three production families:

| model | longest record | p95 | mean | over 512 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 177 | 171 | 161.4 | 0 |
| Llama-3.1-8B-Instruct | 184 | 178 | 170.5 | 0 |
| gemma-2-9b-it (folded build) | 167 | 161 | 152.0 | 0 |

512 therefore sits about 2.8x above the longest real conversation, with no
record anywhere near it, and zero chat-template prefix violations were found
while measuring. This entry moves from "ratify after measurement" to
"ratify — the measurement is in".

**P8: loss masking = assistant tokens only.**
*Plain language:* the model is scored on producing the reply, not on
reproducing the prompt it was given. Prompt tokens are masked out of the
loss.
*Why this is methodological, not cosmetic:* it defines what "the
deception-incentivizing objective" literally optimizes. Assistant-only is
standard instruction-tuning practice and makes the trained objective
"produce this reply in this situation". Full-sequence loss would also train
the model to generate our scenario prompts, which is not the behaviour under
study. Recommend assistant-only; the alternative should be rejected
explicitly rather than by default.

**P10: the checkpoint schedule.** Proposed 6 checkpoints with "doubling"
spacing (dense early, final always included). At the P5/P6 values (282
steps) that realizes steps [8, 17, 35, 70, 140, 281]; even spacing would
give [46, 93, 140, 187, 234, 281].
*Plain language:* when during training we save a snapshot of the model.
*Why this is the load-bearing one:* the saved steps are the x-axis of the
Stage-3 recovery curve R_t. The resolution chosen here is the resolution of
the paper's headline figure, and it cannot be improved afterwards without
re-running the fine-tune. Dense-early spacing assumes the interesting
behaviour changes fastest at the start; even spacing assumes it is spread
out. This also determines whether Stage 1 and Stage 2 share a schedule; the
plan reads RESEARCH_SPEC.md's "matched ... checkpoint schedules" as binding
across all fine-tuning, so the same grid applies to both.
*Cost coupling the team should weigh:* every scheduled step becomes a point
on the R_t curve, and each point costs 4 arms x 2 evaluation environments x
full scenario pools of generation on a T4. The schedule is the single
biggest lever on the Stage-3 GPU budget. Storage is the smaller cost: at
r = 16, 6 checkpoints x 2 arms is roughly 1.9 GB per model family.
*Sourcing honesty:* there is NO established rule for this, and we should not
pretend otherwise. Sleeper Agents (https://arxiv.org/abs/2401.05566) plots
metrics across training snapshots but never states its interval. The only
concrete precedents found are one 2025 emergent-misalignment paper
evaluating every 5 training steps (https://arxiv.org/abs/2506.11613) and
Pythia's log-spaced-early-then-uniform pretraining schedule
(https://arxiv.org/abs/2304.01373), neither of which transfers directly to a
282-step LoRA run. Note also that the framework default of saving every 500
steps (https://huggingface.co/docs/transformers/main_classes/trainer) would
produce at most ONE intermediate checkpoint here, so an explicit dense
schedule is required regardless. This is a judgment call and should be
recorded as one.

**RULED 2026-08-15 — the question was RESTRUCTURED before it was
answered.** The team asked whether six intermediate checkpoints are
load-bearing for a paper whose central focus is not the trajectory. The
answer is that the entry above conflated two separable decisions, and
separating them dissolves most of the cost concern:

- SAVING six checkpoints costs storage only — about 52 GB for the whole
  of Stage 2 at the initially ruled r = 64, about 13 GB at the effective
  r = 16 (adapter size is linear in rank), which is free at this project's
  scale either way.
- EVALUATING one t costs 4 arms x 2 environments x full pools, roughly
  4,720 generations per model, and six of those across three families is
  the largest single line item in the GPU budget.

Ruling: **keep 6 saved checkpoints at "doubling" spacing**, and make
"which t values get a full Stage-3 R_t evaluation" a SEPARATE decision,
deferred to the Stage-2/3 plan, which must pre-commit the subset in
writing before any R_t is computed (same outcome-independence reason
P13 exists). This makes the expensive half reversible in the safe
direction: a saved checkpoint can always be evaluated later, but a
checkpoint never saved cannot be recovered without retraining.
*Why the count was not simply reduced:* under "doubling", n = 3 realizes
[70, 140, 281] — it drops the EARLY points, which is backwards. The
early points are the scientifically distinctive ones: if R_t is already
near 1 at step 8, the capability was arguably never removed, which is a
different finding from gradual relocation. Cutting the saves would have
discarded exactly the evidence that distinguishes them, while saving
none of the cost that actually matters.

## ESCALATED

**P12: the converse fold guard.** The ratified rule E6 says: refuse to
fine-tune Gemma-2 on unfolded data, because Gemma's chat template rejects
the system role, so the system text must be folded into the user turn. The
open question is the converse: should we also refuse to train Qwen or Llama
on a FOLDED copy?
*Why it matters:* Qwen's chat template silently injects its own default
system prompt when a conversation has no system turn. Verified directly in
the shipped tokenizer configuration
(https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json),
which contains the literal fallback "You are Qwen, created by Alibaba Cloud.
You are a helpful assistant." So folded data reaching Qwen would train on
conversations carrying an unrelated injected system prompt, with no error
raised and no sign in the outputs. Training would appear to succeed.
*Options:* (a) ratify the converse as a hard refusal (proposed, because the
failure it prevents is the invisible kind), (b) strike it, or (c) warn-only
until ratified. As currently written the plan would enforce the refusal from
day one, which is why this needs an explicit ruling rather than silence. The
E6 direction is ratified and stays a refusal either way.
*Two facts added 2026-08-15 (critique round 3):* first, this project already
has a PRECEDENT pointing the other way — first-full-review F3's ratified
handling was "the unratified numeric defaults stay UNCHANGED in code — no
banner, no required flags", i.e. unratified positions sit as inert defaults
rather than enforcement, and the converse currently ships as a hard refusal.
Second, the stakes are low in either direction right now: measured against
the real tokenizers, Qwen2.5 and Llama-3.1 both accept the system role and
always take unfolded builds, so the converse branch is unreachable in the
planned pipeline today. Ruling (c) now and (a) later costs nothing.

**P15: train_seed on Gate-1 evaluation rows.** The ratified results-row
convention (RESEARCH_SPEC.md, 2026-08-13) says `train_seed` is "null for
Stage-0/1 runs, the training seed for Stage-2 arms". But Gate 1 evaluates a
TRAINED checkpoint (M_D), which does have a training seed, so the convention
and the situation disagree.
*Options:* (a) amend the convention so any trained-checkpoint row carries
that checkpoint's train_seed, which is arguably truer to the convention's
own rationale of fine-tuning-seed identity, or (b) keep null for Stage-0/1
rows and wire adoption only for Stage-2 arms.
*Recommendation:* (b), because the convention is already ratified, the
training seed is fully recorded in the training manifest and checkpoint
sidecar regardless, and `train_seed` is both resume-identity-guarded per row
and part of the ratified `summarize_runs` group key, so flipping its meaning
mid-project would refuse resume merges and split summary groups. Loud rather
than silent, but avoidable.
*Timing:* must be decided before the first Gate-1 evaluation of a trained
checkpoint. Current plan behaviour honours the ratified sentence: rows stay
null, and an explicitly passed flag that contradicts the checkpoint's
sidecar still refuses.
*Sharpened 2026-08-15 (critique round 3):* the suspension is only HALF
effective, and the ruling should be made knowing it. The code refuses a
`--train-seed` that MISMATCHES the sidecar but accepts one that MATCHES — so
an operator who reads `train_seed: 42` off the sidecar and passes
`--train-seed 42`, the natural thing to do, stamps 42 onto a Stage-1 row and
contradicts the ratified convention with no refusal at all. No recorded
value can be correct under both options, so this cannot be closed in code
before the ruling; a loud warning on that path has been added in the
meantime.

**P16: what should a diverged fp16 run do?** NEW, 2026-08-15 (critique round
3). Training on the T4 runs fp16 with a gradient scaler (the P14 deviation).
If a run diverges — the scaler skipping update after update, or the loss
going non-finite and staying there — the loop currently runs to the end and
writes checkpoints from a broken adapter.
*Why this is not just a bug fix:* an occasional non-finite loss is NORMAL in
fp16 and is exactly what the gradient scaler absorbs (it skips the step,
halves the scale, and the run usually recovers), so "stop at the first one"
would kill healthy runs. Any workable rule is a streak length or a scale
floor — a number — which is why it comes to the team rather than being
chosen in the plan.
*Proposed:* abort with a named error after 20 consecutive skipped steps
(~7% of a 282-step run: far longer than any legitimate scale-search
transient, short enough to save most of a Colab session).
*Options:* (a) the proposed streak abort, (b) a grad-scaler scale floor
instead, (c) ratify "no abort, surfacing only" as sufficient.
*What is already done regardless:* the loop now prints a loud line on every
skipped step and every non-finite loss, plus an end-of-session summary, so
the condition cannot hide in 282 lines of Colab scrollback.
*Why it matters beyond wasted GPU time:* without an abort, a numerics
failure and the genuine scientific result "M_D did not become deceptive"
arrive looking identical — and the plan elsewhere instructs the team to take
the latter at face value. No reported number is at risk either way; Gate 1
fails a broken M_D correctly. What the abort buys is telling the two apart.
*Timing:* decide before the first 7-9B training run.
**RULED 2026-08-15: option (a)** — abort with a named error after 20
consecutive grad-scaler skipped steps. The surfacing (per-event lines
plus an end-of-session summary) ships regardless.

## Rulings, 2026-08-15

All sixteen items are ruled; the authoritative record is
RESEARCH_SPEC.md, "Stage-1/2 fine-tuning constants (ratified
2026-08-15)", items T1-T16.

Two rulings changed the proposal:

- **P2** moved from r = 16 / dropout 0.05 to **r = 64 / dropout 0.1**,
  on the reasoning that the assumption r = 16 required was worth more
  than the storage it saved — plus a pre-committed ordered fallback that
  makes the change cost-free in GPU terms. **That fallback then fired on
  2026-08-16 (Gemma's T4 fit failure) and the human ruled rank 16 directly
  the same day, so the EFFECTIVE values are back to r = 16 / alpha = 16 /
  dropout 0.05 for every family.** Read the paragraph above as the history
  of the initial ruling, not as the current setting; the assumption it
  describes as bought off is once again in force and is declared under
  RESEARCH_SPEC.md T14.
- **P10** was restructured rather than answered: saving stays at 6, and
  the expensive decision (which t values get evaluated) is separated out
  and deferred to the Stage-2/3 plan under a pre-commitment requirement.

T2's fallback obligation closed when it activated 2026-08-16 after Gemma's
T4 fit check failed. Two live obligations remain: T3's warmup escalation
applies only if fp16 proves unstable, and T10's evaluated subset must be
pre-committed by the Stage-2/3 plan before any R_t is computed.

## Summary of what was asked (historical)

- Confirm: P1, P3 (value), P4, P9, P11, P13, P14, and now P7 (the
  measurement it was waiting on is in — see the table there).
- Decide: P2 (rank/alpha/dropout), P3 (schedule/warmup), P5 (epochs),
  P6 (batch values), P8 (masking), P10 (checkpoint schedule).
- Rule explicitly: P12, P15, P16.
- Already ratified, no action: P6 (strict matching reading).

The single decision worth the most discussion time is P10, because it is
unsourced, sets the paper's headline figure resolution, drives the Stage-3
GPU budget, and cannot be revised without retraining.
