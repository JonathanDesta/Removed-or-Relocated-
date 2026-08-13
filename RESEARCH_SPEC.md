Pasted below is the reserach proposal for our paper.

\section{Introduction}

\paragraph{Motivation}
Understanding whether LLMs can circumvent internal removal of deceptive behavior is important for evaluating methods of aligning AI with human values. Prior work has evaluated the effects of capability ablation and the nature of deceptive capabilities, but it remains unclear whether LLMs can restore strategically deceptive behavior after the layer with the largest causal-relation to deception is permanently bypassed, particularly during fine-tuning. Beyond that, it is also unknown whether potential recovery would be done through relocation of the capability to a different layer, strengthening of a preexisting backup circuit, or distribution of the capability across the rest of the model. Distinguishing these outcomes helps us understand how LLMs go about avoiding safety interventions.  

\paragraph{Contribution}
We test whether a model can relocate deceptive behavior around a permanently bypassed layer during fine-tuning and, if it can, where the behavior relocates. We first locate the layer most causally-related to deception. We then fine-tune a model that bypasses that layer under an incentive to build deceptive capabilities and observe whether deceptive capabilities regenerate. If they do, we find where the capabilities relocated.

\section{Related Work}

\paragraph{Inference-Time Self-Repair}

Models compensate for inference-time ablation of an attention layer using other layers \citep{mcgrath2023hydra,rushing2024selfrepair}. Since this occurs during inference instead of training, this only proves the existence of backup pathways rather than fine-tuned adaptation.

\paragraph{Relearning Around Internal Ablation}
Ablating or fine-tuning concept-associated neurons and circuits suppresses misaligned behavior, but continued training restores it by reallocating to other neurons in either earlier layers or the same layer \citep{lo2024relearn,ustaomeroglu2026blockem}. None, however, tests how strategic deception specifically recovers around permanent ablation of an entire layer, or whether recovery creates a layer-level dependency.

\paragraph{Deception Localization}
Prior work suggested that deception-related capabilities are considerably dispersed \citep{yang2024universal,kapelko2025cyclic}. However, newer works using larger models and strategic rather than prompted deception finds causally related attention heads to be more concentrated \citep{merrill2026pointofnoreturn}. Differing results may be due to prompted lying revealing instruction-following or persona features rather than strategic deception and smaller models having weaker strategic deception capabilities. Jensen-Shannon divergence (JSD) between attention distributions of two models at the same layer can help identify layers most causally associated with certain behaviors \citep{chaudhary2025whitebox}. 

\section{Methodology}

\subsection{Overview}
Stage~1 creates a deceptive checkpoint and selects a target layer. Stage~2 permanently bypasses that layer and performs fine-tuning. Stage~3 measures capability recovery and locates the new causal circuit. We use the Offer Negotiation environment \citep{merrill2026pointofnoreturn} for fine-tuning and layer selection, the Insider Trading environment \citep{Scheurer2023} and held-out Offer Negotiation scenarios for evaluation and validating results, the MMLU \citep{hendrycks2020measuring} and GSM8K \citep{cobbe2021gsm8k} benchmarks for general capabilities assessment, and the WikiText \citep{Merity2016} for neutral-distribution assessment. We run the experiment on Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Gemma-2-9B-it. All fine-tuning uses matched data volumes, optimization settings, checkpoint schedules, and random seeds.

\subsection{Stage 1: Checkpoint Creation and Layer Selection}
\label{sec:stage1}

We fine-tune a base model $M_0$ under a deception-incentivizing objective to create a deceptive checkpoint $M_D$. Let 
\[
\tau(M)=D_{\mathrm{advantageous}}(M)-D_{\mathrm{control}}(M)
\]
be the difference in deception between two conditions, one where deception is advantageous and one where it isn’t. To verify fine-tuning, we check that 
\[
\tau(M_D)-\tau(M_0)
\]
exceeds a prespecified margin while $M_D$ retains honest task-competence in non-deceptive tasks.

For every layer $\ell\in\mathcal{L}$, we temporarily bypass the layer during inference: $
h_{\ell+1}=h_\ell$. We define a layer’s effect on deception as:
\[
A_\ell=\tau(M_D)-\tau(M_D^{-\ell}),
\]
where $M_D^{-\ell}$ is $M_D$ with layer $\ell$ bypassed. We select $
\ell^*=\arg\max_{\ell\in\mathcal{L}}A_\ell
$
subject to prespecified bounds on general-capability loss, task-specific competence loss, invalid-response rate, and neutral-distribution divergence (measured with JSD). We also report the complete Pareto frontier and layer-wise curves. We then reevaluate $A_{\ell^*}$ in the Insider Trading environment and on held-out Offer Negotiation scenarios to ensure the effects observed transfer beyond the initially selected data. If there is no layer within the prespecified bounds exhibiting a causal effect on strategic deception exceeding a prespecified threshold, we report that no viable layer-level localization was found and do not move on to Stage~2.  

\paragraph{Localization corroboration}
We conduct a variety of corroboration techniques at each layer of $M_D$. First, we train probes on the Instructed-Pairs dataset \citep{zou2023representation} and verify transfer to strategic deception tasks, then fit separate probes at each layer and report held-out, threshold-free deception decodability. We also calculate the JSD between attention distributions in deception-incentivized and control environments. Lastly, we patch activations from the control environment into the deceptive environment and measure changes in deception, task competence, and response validity. These corroborating techniques only serve to report converging or diverging evidence and are not used in layer selection.

After selecting $\ell^*$, we create the immediate post-ablation checkpoint $
\widetilde{M}_D=M_D^{-\ell^*}$ by making bypass of $\ell^*$ permanent.

\subsection{Stage 2: Matched Continuation Fine-Tuning}
\label{sec:stage2}

We fine-tune from $M_D$ and $\widetilde{M}_D$ under either a deception-incentivizing or control objective. This produces four checkpoints:
$
M^{I,D},\qquad M^{I,C},\qquad M^{L,D},\qquad M^{L,C},
$ where $I$/$L$ denote that the layer is maintained/bypassed, and $D$/$C$ denote a deceptive/control objective.

\subsection{Stage 3: Behavioral Recovery and Causal Remapping}
\label{sec:stage3}

We quantify behavioral recovery of all checkpoints at multiple points $t$ during fine-tuning using
\[
R_t=
\frac{\tau(M_t^{L,D})-\tau(M_t^{L,C})}
     {\tau(M_t^{I,D})-\tau(M_t^{I,C})} 
\]
in the Insider Trading environment and held-out Offer Negotiation scenarios. Values near one indicate deception-specific capability is nearly fully recovered despite layer-ablation.

If deceptive behavior recovers, we repeat the same causal analysis in Stage~1 on every layer $\ell\in\mathcal{L}$ of checkpoints $M_t^{L,D}$ and $\widetilde{M}_D$. 

We then create a layer-wise $\delta$-curve displaying the difference in causal relation to deception of each layer in $M_t^{L,D}$ and $\widetilde{M}_D$. A uniform curve indicates dispersed recovery while a nonuniform curve suggests concentrated recovery.

Alongside that, we identify the layer $k$ in $M_t^{L,D}$ that is the most causally related to deception. We check whether the layer(s) with concentrated recovery is the layer(s) with the greatest causal relation to deception in $M_t^{L,D}$. If they are the same, then deception was entirely relocated. Otherwise, deception was only partially relocated.

We then investigate the causal relation to deception within $\widetilde{M}_D$ of the layer(s) most causally related to deception within $M_t^{L,D}$ and the layer(s) with the greatest change in causal relation to deception. For each layer, if the causal relation in $\widetilde{M}_D$ is near-zero, the deception capability was reconstructed in that layer. Otherwise, the deception capability was strengthened from a pre-existing pathway.

\paragraph{Statistical analysis}
Layer selection and probe fitting use separate data from final evaluation. We split related prompt variants by their underlying scenario to prevent leakage. We report confidence intervals obtained by bootstrapping scenarios and variation across fine-tuning seeds. 

\section{Ideal Results}

The sweep identifies layer $\ell^*$ with corroborating causal evidence. During fine-tuning, the deceptive checkpoint with a bypassed layer $M^{L,D}$ recovers toward the deception ceiling set by the checkpoints without a bypassed layer. After behavioral recovery, we find recovery is concentrated in layer $k\neq\ell^*$ where deceptive capability was fully reconstructed. 

\section{Limitations}

\paragraph{Localization accuracy and precision}
We select one whole layer, but recent work localizes deceptive capabilities to compact attention-head sets comprising under 10\% of heads \citep{merrill2026pointofnoreturn}. The causally relevant unit may therefore be smaller than a layer. Layer removal can also have confounding effects that are difficult to measure despite models being shown to be fairly robust against layer removal \citep{lad2024robustness}.
