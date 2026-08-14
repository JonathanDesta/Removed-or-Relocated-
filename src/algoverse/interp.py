"""
Mechanistic-interpretation helpers: activation reading, linear probing,
and attention JSD. Layer bypass lives in models.install_bypass.

TO BUILD: activation patching (RESEARCH_SPEC "Localization corroboration" —
patching control-environment activations into the deceptive environment) is
NOT implemented yet; nothing in this module patches activations.

Written against the HuggingFace stack that models.py loads, so a model object
from load_model_and_tokenizer goes straight in. For an INTACT model, reading
activations needs no hooks: transformers exposes them via
output_hidden_states / output_attentions. That is NOT safe once a bypass is
installed: on transformers >= ~4.5x output_hidden_states records each block's
raw output (before install_bypass's forward hook replaces it), so it returns
un-bypassed activations for a bypassed model. The readers below therefore
refuse a bypassed model and point to models.residual_stream_by_layer, which
captures the true (bypass-aware) residual stream via pre-hooks.

RENDERING CONTRACT: every text passed to an encoder in this module must be a
canonical fully rendered prompt from eval.render_condition_texts, including
the generation prompt and any required system-role fold. Interp encoders do
not add special tokens; changing the rendering changes the measured quantity.
"""

import random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algoverse.models import bypass_state


def _refuse_if_bypassed(model):
    """Raise if a bypass is installed: transformers' output_hidden_states /
    output_attentions do not reflect it, so a read here would silently return
    un-bypassed activations. Use models.residual_stream_by_layer instead."""
    state = bypass_state(model)
    if state is not None:
        raise RuntimeError(
            "cannot read activations via output_hidden_states/attentions on a "
            "model with a bypass installed (layer %s, %s): these reflect the "
            "raw pre-bypass block outputs on this transformers version. Use "
            "models.residual_stream_by_layer for the true residual stream."
            % (state["layer_idx"], state["impl"])
        )


def _last_real_token_idx(attention_mask):
    """Index of the last unmasked token in each row.

    Padding side agnostic: models.py sets none explicitly and its own comments
    describe left-padded batches, so assuming either one is a silent-wrong-
    answer bug. Flipping and taking the first 1 finds the last real token
    whichever side the padding is on.
    """
    L = attention_mask.shape[1]
    return L - 1 - attention_mask.flip(1).argmax(1)


def _ensure_pad_token(tokenizer):
    """Batched tokenization needs a pad token; fall back to eos if unset."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


# --- reading internals ---

def last_token_resid_all_layers(model, tokenizer, text):
    """Return a [n_layers, d_model] array: the last-token residual stream at
    each layer.

    hidden_states has n_layers + 1 entries; the first is the embedding output,
    not a layer, hence the [1:]. .float() is required because fp16 and 4-bit
    tensors cannot go to numpy directly.
    """
    _refuse_if_bypassed(model)
    inputs = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    per_layer = [h[0, -1, :] for h in out.hidden_states[1:]]
    return torch.stack(per_layer).float().cpu().numpy()


def resid_all_layers_batch(model, tokenizer, texts, batch_size=8):
    """Batched version: [n_texts, n_layers, d_model]."""
    _refuse_if_bypassed(model)
    _ensure_pad_token(tokenizer)
    chunks = []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[i:i + batch_size], return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = _last_real_token_idx(inputs["attention_mask"])
        rows = torch.arange(last.size(0), device=last.device)
        per_layer = [h[rows, last, :] for h in out.hidden_states[1:]]
        chunks.append(torch.stack(per_layer, dim=1).float().cpu())
    return torch.cat(chunks).numpy()


def _attention_all_layers_unchecked(model, tokenizer, text):
    """attention_all_layers without the bypass guard. Only the attention-JSD
    helpers may call this: they read a bypassed model deliberately and NaN
    the bypassed layer, whose map is real but causally dead."""
    inputs = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError(
            'attentions came back None — reload with attn_implementation="eager"'
        )
    return torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()


def attention_all_layers(model, tokenizer, text):
    """Return a [n_layers, n_heads, seq, seq] array of attention patterns.

    transformers normally falls back to eager attention when output_attentions
    is requested under sdpa, logging a warning. If a backend returns None
    instead, the model has to be reloaded with attn_implementation="eager".

    A bypassed block still runs its attention and only its residual output is
    discarded, so output_attentions would return a real-but-causally-dead
    attention map for it; the guard refuses a bypassed model rather than let
    that be mistaken for live computation. The attention-JSD helpers read a
    bypassed model deliberately via the unchecked variant and carry NaN at
    the bypassed layer.
    """
    _refuse_if_bypassed(model)
    return _attention_all_layers_unchecked(model, tokenizer, text)


# --- probing ---

def _group_bootstrap_auroc_ci(y, scores, groups, n_boot=2000, seed=0,
                              alpha=0.05):
    y = np.asarray(y)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    group_ids = sorted(set(groups.tolist()))
    by_group = {group: np.flatnonzero(groups == group) for group in group_ids}
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sampled = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
        idx = np.concatenate([by_group[group] for group in sampled])
        if len(set(y[idx].tolist())) < 2:
            continue
        values.append(roc_auc_score(y[idx], scores[idx]))
    if len(values) < max(20, n_boot // 10):
        return None, None
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def probe_layer(X, y, groups):
    """Fit a standardized linear probe with a group-aware held-out split.

    Returns (pipeline, {"auroc", "auroc_ci", "accuracy"}). AUROC from raw
    decision scores is primary; groups is required so related prompt variants
    cannot leak across the split. L2 regularization uses C=0.1, matching the
    cited method's lambda=10. The 0.3 split, seed 0, and max_iter=1000 remain
    recorded recipe decisions. GroupShuffleSplit cannot stratify, so an
    informative error is raised if either side has one class. The AUROC
    interval resamples held-out scenario groups. This repo reads one
    last-token activation per text, a declared deviation from that method's
    response-token aggregation.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
        raise ValueError(
            "group split produced a single-class side; supply more scenarios "
            "of both labels"
        )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=0.1, max_iter=1000),
    ).fit(X[train_idx], y[train_idx])
    scores = clf.decision_function(X[test_idx])
    auroc = float(roc_auc_score(y[test_idx], scores))
    ci_low, ci_high = _group_bootstrap_auroc_ci(
        y[test_idx], scores, groups[test_idx]
    )
    return clf, {
        "auroc": auroc,
        "auroc_ci": (ci_low, ci_high),
        "accuracy": float(clf.score(X[test_idx], y[test_idx])),
    }


# --- divergence ---

def jsd(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two distributions over the last dim.

    Accepts numpy arrays or torch tensors, since the readers above return
    numpy. Expects actual probability distributions (non-negative, summing to
    1) — attention patterns or softmaxed logits, not raw residual-stream
    activations, which are unnormalised and can be negative.
    """
    p = torch.as_tensor(p, dtype=torch.float64)
    q = torch.as_tensor(q, dtype=torch.float64)
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(-1)
    return (0.5 * (kl_pm + kl_qm)).numpy()


def attention_jsd_between_models(model_a, model_b, tokenizer, texts):
    """EXPLORATORY: per-layer JSD between two models on identical texts.

    This is not the spec's Methodology corroboration; that is the one-model,
    two-condition comparison in attention_jsd_between_conditions. This helper
    remains for checkpoint-vs-checkpoint diagnostics only.

    Returns a [n_layers] array. Attention rows are already probability
    distributions over key positions, so JSD applies directly. The two models
    must share an architecture and tokenizer, which base vs LoRA finetune does.

    Accumulates per text rather than stacking every example: attention is
    [n_layers, n_heads, seq, seq], so one prompt on the 7B at seq=512 is
    already several hundred MB in fp32.
    """
    total, n = None, 0
    for text in texts:
        a = _attention_all_layers_unchecked(model_a, tokenizer, text)
        b = _attention_all_layers_unchecked(model_b, tokenizer, text)
        per_layer = jsd(a, b).mean(axis=(1, 2))   # mean over heads, positions
        total = per_layer if total is None else total + per_layer
        n += 1
    result = total / n
    for model in (model_a, model_b):
        state = bypass_state(model)
        if state is not None:
            result[state["layer_idx"]] = float("nan")
    return result


def _group_bootstrap_jsd_ci(contrib_a, contrib_b, groups_a, groups_b,
                            combined, n_boot=2000, seed=0, alpha=0.05):
    def normalize_groups(groups, length):
        if groups is None:
            return list(range(length))
        groups = list(groups)
        if len(groups) != length:
            raise ValueError("groups length must match texts length")
        return groups

    def grouped_indices(groups):
        order = []
        by_group = {}
        for index, group in enumerate(groups):
            if group not in by_group:
                order.append(group)
                by_group[group] = []
            by_group[group].append(index)
        return order, by_group

    groups_a = normalize_groups(groups_a, len(contrib_a))
    groups_b = normalize_groups(groups_b, len(contrib_b))
    ids_a, by_a = grouped_indices(groups_a)
    ids_b, by_b = grouped_indices(groups_b)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sampled_a = [ids_a[rng.randrange(len(ids_a))] for _ in ids_a]
        sampled_b = [ids_b[rng.randrange(len(ids_b))] for _ in ids_b]
        picks_a = [index for group in sampled_a for index in by_a[group]]
        picks_b = [index for group in sampled_b for index in by_b[group]]
        if not picks_a or not picks_b:
            continue
        values.append(jsd(
            combined(contrib_a, picks_a), combined(contrib_b, picks_b)
        ))
    if len(values) < max(20, n_boot // 10):
        return None, None
    values = np.stack(values)
    return (
        np.quantile(values, alpha / 2, axis=0).astype(float),
        np.quantile(values, 1 - alpha / 2, axis=0).astype(float),
    )


def attention_jsd_between_conditions(model, tokenizer, texts_a, texts_b,
                                     groups_a=None, groups_b=None,
                                     n_boot=2000, seed=0, alpha=0.05):
    """Per-layer attention JSD between two prompt sets for one model.

    Each text is encoded alone. Attention rows are zero-extended to the
    common key-position support, then pooled flat within condition before
    JSD is computed. Returns point and scenario-bootstrap interval arrays in
    nats. Bootstrap resamples recombine cached per-text contributions and do
    not re-run the model; absent group lists make each text its own group.
    Longer prompts contribute more query rows under flat pooling. The result
    is JSD between condition summaries, not a mean of row-wise JSDs.

    This average-then-JSD, flat-pooling, zero-extension procedure is the
    project's ratified design, not the procedure of chaudhary2025whitebox.
    Inputs must follow the module rendering contract and be produced by
    eval.render_condition_texts. If a bypass is live, all three result arrays
    carry NaN at that disconnected layer, derived from the model state.
    """
    texts_a, texts_b = list(texts_a), list(texts_b)

    def seq_len(text):
        encoded = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )
        return encoded["input_ids"].shape[1]

    max_len = max(seq_len(text) for text in texts_a + texts_b)

    def text_contrib(text):
        att = _attention_all_layers_unchecked(model, tokenizer, text)
        n_layers, n_heads, seq, _ = att.shape
        padded = np.zeros((n_layers, n_heads, seq, max_len))
        padded[..., :seq] = att
        return padded.sum(axis=(1, 2)), n_heads * seq

    contrib_a = [text_contrib(text) for text in texts_a]
    contrib_b = [text_contrib(text) for text in texts_b]

    def combined(contribs, picks):
        total = sum((contribs[index][0] for index in picks))
        rows = sum(contribs[index][1] for index in picks)
        return total / rows

    point = jsd(
        combined(contrib_a, range(len(contrib_a))),
        combined(contrib_b, range(len(contrib_b))),
    ).astype(float)
    ci_low, ci_high = _group_bootstrap_jsd_ci(
        contrib_a, contrib_b, groups_a, groups_b, combined,
        n_boot=n_boot, seed=seed, alpha=alpha,
    )
    result = {"jsd": point, "ci_low": ci_low, "ci_high": ci_high}
    state = bypass_state(model)
    if state is not None:
        for values in result.values():
            if values is not None:
                values[state["layer_idx"]] = float("nan")
    return result
