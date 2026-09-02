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
installed: whether output_hidden_states reflects the bypass is version-
dependent, while a bypassed block's attention maps are causally dead on every
version. The readers below therefore refuse a bypassed model and point to
models.residual_stream_by_layer, whose pre-hook capture is version-robust.

RENDERING CONTRACT: every text passed to an encoder in this module must be a
canonical fully rendered prompt from eval.render_condition_texts, including
the generation prompt and any required system-role fold. Probe-capture texts
(response_token_resid_by_layer) append the scored response to that canonical
prompt; the prompt portion is still contract-rendered. Interp encoders do
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

from algoverse.corroboration import aggregate_response_scores
from algoverse.models import (
    bypass_state,
    bypassed_layers,
    residual_stream_by_layer,
)


def load_eager_model_for_interp(model_id, quant="4bit", adapter_path=None):
    """Load a model with EAGER attention, for interpretation reads ONLY.

    Never use this model for generation or eval rows: attn_implementation is
    part of gen_config identity (gen_config.load_profile.attn_implementation)
    and generation numerics differ across attention backends — all
    row-producing runs load via models.load_model_and_tokenizer. Eager
    attention materializes the attention probabilities that sdpa/flash do
    not, so attention_all_layers works on this model. Same load profile as
    the canonical loader (4-bit NF4 / none). adapter_path attaches a LoRA
    adapter ONLY: this helper does NOT reinstall a permanent lesion. Per the
    ratified rule the Stage-2 lesion is reinstalled at every load from
    checkpoint metadata — that machinery belongs to the training-track plan,
    and until it exists a lesioned checkpoint must not be read through this
    helper; the corroboration-driver plan defines the sanctioned consumption
    pattern. Plain load — freeing a previously loaded model is the caller's
    job. Returns (model, tokenizer), model in eval mode.
    """
    from algoverse.models import _load

    model, tokenizer = _load(
        model_id, quant=quant, adapter_path=adapter_path,
        attn_implementation="eager",
    )
    attn = getattr(model.config, "_attn_implementation", None)
    if attn != "eager":
        raise RuntimeError(
            "eager attention did not take: model came up with "
            "attn_implementation=%r" % attn
        )
    return model, tokenizer


def _refuse_if_bypassed(model):
    """Raise on a bypass: hidden-state behavior drifts; attention is dead.

    Use models.residual_stream_by_layer for version-robust residual reads.
    """
    state = bypass_state(model)
    if state is not None:
        markers = [m for m in state.values() if m is not None]
        raise RuntimeError(
            "cannot read activations via output_hidden_states/attentions on a "
            "model with a bypass installed (layers %s, roles %s): whether "
            "output_hidden_states reflects the bypass is version-dependent, "
            "and the bypassed block's attention maps are real but causally "
            "dead on every version. Use "
            "models.residual_stream_by_layer for the true residual stream."
            % (
                [m["layer_idx"] for m in markers],
                [m["role"] for m in markers],
            )
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
    if not out.attentions or out.attentions[0] is None:
        raise RuntimeError(
            "no attention weights returned (attn_implementation=%r): this "
            "backend did not materialize attention probabilities under "
            "output_attentions=True (sdpa and flash backends do not). Load the "
            "model with eager attention via interp.load_eager_model_for_interp "
            "for attention reads."
            % getattr(getattr(model, "config", None), "_attn_implementation", None)
        )
    return torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()


def attention_all_layers(model, tokenizer, text):
    """Return a [n_layers, n_heads, seq, seq] array of attention patterns.

    sdpa and flash backends do not return attention maps here. Use
    load_eager_model_for_interp to load the sanctioned eager-attention model.

    A bypassed block still runs its attention and only its residual output is
    discarded, so output_attentions would return a real-but-causally-dead
    attention map for it; the guard refuses a bypassed model rather than let
    that be mistaken for live computation. The attention-JSD helpers read a
    bypassed model deliberately via the unchecked variant and carry NaN at
    the bypassed layer.
    """
    _refuse_if_bypassed(model)
    return _attention_all_layers_unchecked(model, tokenizer, text)


def _validate_span_len(span_len):
    if span_len is None:
        return None
    if isinstance(span_len, bool) or not isinstance(span_len, int) or span_len < 1:
        raise ValueError("span_len must be None or an int >= 1, got %r" % (span_len,))
    return span_len


def response_token_resid_by_layer(model, tokenizer, texts, response_starts,
                                  span_len=None):
    """Per-layer response-token residual features for probing, lesion-safe.

    This is the capture path for the RATIFIED 2026-08-15 response-token
    aggregation (goldowskydill2025detecting, arXiv 2502.03407): probe
    features are the residual-stream activations of every RESPONSE token,
    kept per token so probe_layer can fit on tokens and mean-aggregate
    per-token scores per response.

    texts[i] is a canonical contract-rendered prompt with the scored
    response appended (module rendering contract); response_starts[i] is
    the token index where the response begins under this module's
    add_special_tokens=False tokenization of texts[i]. Features cover
    tokens [response_starts[i], end), or at most span_len of them when
    span_len is given: span_len=1 with response_starts shifted by -1 reads
    the LAST PROMPT token (the final-prompt-position probe variant).

    Reads go through models.residual_stream_by_layer (pre-hook capture),
    so they are correct with or without a bypass installed — never through
    output_hidden_states. The feature for layer l is the residual LEAVING
    block l (capture entry l+1), the same quantity output_hidden_states[1:]
    exposes on an intact model. Per the ratified 2026-08-13 rule, every
    bypassed layer (models.bypassed_layers — probe or permanent role) is
    EXCLUDED: its entry in the result is None, because a bypassed block's
    "output" is just its input passed through.

    Encodes one text at a time: each capture holds n_layers + 1
    full-sequence fp32 copies, and padding-plus-span arithmetic under
    batching is a silent-wrong-answer risk this reader avoids.

    Returns a list over layers; entry l is None for a bypassed layer, else
    a list over texts of float32 [n_response_tokens, d_model] arrays.
    """
    texts = list(texts)
    starts = list(response_starts)
    if len(starts) != len(texts):
        raise ValueError(
            "response_starts length %d does not match texts length %d"
            % (len(starts), len(texts))
        )
    if not texts:
        raise ValueError("no texts supplied")
    span_len = _validate_span_len(span_len)
    excluded = set(bypassed_layers(model))
    per_layer = None
    for text, start in zip(texts, starts):
        inputs = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        n_tokens = inputs["input_ids"].shape[1]
        if not 0 <= int(start) < n_tokens:
            raise ValueError(
                "response start %r is outside a text of %d tokens"
                % (start, n_tokens)
            )
        length = (
            n_tokens - int(start) if span_len is None
            else min(span_len, n_tokens - int(start))
        )
        captured = residual_stream_by_layer(
            model, inputs["input_ids"], inputs.get("attention_mask")
        )
        n_layers = len(captured) - 1
        if per_layer is None:
            per_layer = [
                None if layer in excluded else []
                for layer in range(n_layers)
            ]
        for layer in range(n_layers):
            if per_layer[layer] is None:
                continue
            per_layer[layer].append(
                captured[layer + 1][0, int(start):int(start) + length, :]
                .float().cpu().numpy()
            )
    return per_layer


def response_token_resid_by_layer_to_disk(model, tokenizer, texts,
                                          response_starts, out_dir,
                                          layers=None, span_len=None):
    """Capture response-token residuals once and spool float32 by layer.

    Returns metadata consumed by ``iter_disk_backed_residual_layers``. The
    caller owns ``out_dir`` and its cleanup. Only requested, non-bypassed
    layers are materialized, but each text still needs just one model forward.
    span_len caps each text's span (see response_token_resid_by_layer);
    span_len=1 with shifted starts is the final-prompt-position reader and
    shrinks the spool from GBs to MBs.
    """
    import shutil
    from pathlib import Path

    texts = list(texts)
    starts = [int(start) for start in response_starts]
    span_len = _validate_span_len(span_len)
    if len(starts) != len(texts):
        raise ValueError(
            "response_starts length %d does not match texts length %d"
            % (len(starts), len(texts))
        )
    if not texts:
        raise ValueError("no texts supplied")
    n_layers = getattr(model.config, "num_hidden_layers", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    if n_layers is None or hidden_size is None:
        raise ValueError("model config must expose num_hidden_layers and hidden_size")
    requested = list(range(n_layers)) if layers is None else [int(x) for x in layers]
    if len(set(requested)) != len(requested):
        raise ValueError("layers contains duplicates")
    outside = [layer for layer in requested if not 0 <= layer < n_layers]
    if outside:
        raise ValueError("layers outside model range: %r" % outside)
    excluded = set(bypassed_layers(model))
    active = [layer for layer in requested if layer not in excluded]

    lengths = []
    for text, start in zip(texts, starts):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        n_tokens = int(encoded["input_ids"].shape[1])
        if not 0 <= start < n_tokens:
            raise ValueError(
                "response start %r is outside a text of %d tokens"
                % (start, n_tokens)
            )
        lengths.append(
            n_tokens - start if span_len is None
            else min(span_len, n_tokens - start)
        )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    total_tokens = offsets[-1]
    required_bytes = total_tokens * int(hidden_size) * 4 * len(active)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(out_dir).free
    if required_bytes > available:
        raise RuntimeError(
            "probe scratch space insufficient: need %d bytes for float32 "
            "activations, have %d" % (required_bytes, available)
        )

    arrays = {}
    paths = {}
    for layer in active:
        path = out_dir / ("layer-%03d.npy" % layer)
        paths[layer] = str(path)
        arrays[layer] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(total_tokens, int(hidden_size)),
        )

    try:
        for index, (text, start) in enumerate(zip(texts, starts)):
            inputs = tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            captured = residual_stream_by_layer(
                model, inputs["input_ids"], inputs.get("attention_mask")
            )
            if len(captured) - 1 != n_layers:
                raise RuntimeError("captured layer count changed during probe spool")
            begin, end = offsets[index], offsets[index + 1]
            for layer in active:
                value = (
                    captured[layer + 1][0, start:start + (end - begin), :]
                    .float().cpu().numpy()
                )
                if value.shape != (end - begin, hidden_size):
                    raise RuntimeError(
                        "layer %d response %d capture shape %r, expected %r"
                        % (layer, index, value.shape,
                           (end - begin, hidden_size))
                    )
                arrays[layer][begin:end] = value
    finally:
        for array in arrays.values():
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()

    return {
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "offsets": offsets,
        "paths": paths,
        "excluded": sorted(excluded),
        "required_bytes": required_bytes,
        "span_len": span_len,
    }


def iter_disk_backed_residual_layers(metadata):
    """Yield each layer's response arrays from disk, one mapped layer at a time."""
    offsets = metadata["offsets"]
    excluded = set(metadata["excluded"])
    paths = metadata["paths"]
    for layer in range(metadata["n_layers"]):
        if layer in excluded or layer not in paths:
            yield None
            continue
        array = np.load(paths[layer], mmap_mode="r")
        responses = [
            array[offsets[i]:offsets[i + 1]]
            for i in range(len(offsets) - 1)
        ]
        try:
            yield responses
        finally:
            del responses
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


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

    Features follow the response-token aggregation of
    goldowskydill2025detecting (arXiv 2502.03407), RATIFIED 2026-08-15:
    X is a sequence of per-response [n_tokens, d] arrays of response-token
    activations (response_token_resid_by_layer produces them). The probe
    is fit on the individual token activations of the TRAINING responses,
    flattened across responses and positions with each token carrying its
    response's label — the paper's "flatten across samples and sequence
    positions". At evaluation each held-out token gets a decision score
    and each response's score is the MEAN of its token scores
    (corroboration.aggregate_response_scores, the paper's "mean across
    tokens ... single deceptiveness-score for each model response").
    AUROC over held-out per-response scores is primary; accuracy is the
    fraction of held-out responses whose mean score lands on its label's
    side of the decision boundary. A 2D X keeps its previous meaning as
    the degenerate one-vector-per-response case (the mean over one token
    is that token). SUPERSEDED NOTE: this function previously read a
    single last-token activation per text and declared that a deviation
    from the cited method; the 2026-08-15 ratification adopts the paper's
    response-token aggregation outright, so that declared-deviation line
    is superseded and no deviation remains to declare.

    Returns (pipeline, {"auroc", "auroc_ci", "accuracy"}). The split is at
    the RESPONSE level; groups is required so related prompt variants
    cannot leak across it. L2 regularization uses C=0.1, matching the
    cited method's lambda=10. The 0.3 split, seed 0, and max_iter=1000
    remain recorded recipe decisions (ratified 2026-08-15).
    GroupShuffleSplit cannot stratify, so an informative error is raised
    if either side has one class. The AUROC interval resamples held-out
    scenario groups of per-response scores.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if isinstance(X, np.ndarray) and X.ndim == 2:
        responses = [X[i:i + 1] for i in range(X.shape[0])]
    else:
        responses = [np.asarray(tokens) for tokens in X]
        for index, tokens in enumerate(responses):
            if tokens.ndim != 2 or tokens.shape[0] < 1:
                raise ValueError(
                    "response %d must be a nonempty [n_tokens, d] array, "
                    "got shape %s" % (index, tokens.shape)
                )
    if not len(responses) == len(y) == len(groups):
        raise ValueError(
            "X, y, and groups must align per response: %d, %d, %d"
            % (len(responses), len(y), len(groups))
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    placeholder = np.zeros((len(responses), 1))
    train_idx, test_idx = next(splitter.split(placeholder, y, groups))
    if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
        raise ValueError(
            "group split produced a single-class side; supply more scenarios "
            "of both labels"
        )
    X_train = np.concatenate([responses[i] for i in train_idx])
    y_train = np.concatenate(
        [np.full(responses[i].shape[0], y[i]) for i in train_idx]
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=0.1, max_iter=1000),
    ).fit(X_train, y_train)
    scores = np.asarray(aggregate_response_scores(
        [clf.decision_function(responses[i]) for i in test_idx]
    ))
    auroc = float(roc_auc_score(y[test_idx], scores))
    ci_low, ci_high = _group_bootstrap_auroc_ci(
        y[test_idx], scores, groups[test_idx]
    )
    predicted = scores > 0
    return clf, {
        "auroc": auroc,
        "auroc_ci": (ci_low, ci_high),
        "accuracy": float(np.mean(predicted == y[test_idx].astype(bool))),
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
        for layer in bypassed_layers(model):
            result[layer] = float("nan")
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
    for layer in bypassed_layers(model):
        for values in result.values():
            if values is not None:
                values[layer] = float("nan")
    return result
