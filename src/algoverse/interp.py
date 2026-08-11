"""
This module contains the methods mechanistically interpreting models. It contains layer bypass, activation patching, and linear probing.
"""
"""
This module contains the methods mechanistically interpreting models. It
contains layer bypass, activation patching, and linear probing.

Written against the HuggingFace stack that models.py loads, so a model object
from load_model_and_tokenizer goes straight in. Reading activations needs no
hooks: transformers exposes them via output_hidden_states / output_attentions.
"""

import numpy as np
import torch
from contextlib import contextmanager
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# --- model plumbing ---

def _decoder_layers(model):
    """Return the list of decoder layer modules.

    get_decoder() is the transformers API for this and PEFT forwards it, so a
    LoRA-wrapped model resolves the same as a bare one. The manual walk is a
    fallback for models that don't implement it.
    """
    if hasattr(model, "get_decoder"):
        try:
            return model.get_decoder().layers
        except AttributeError:
            pass
    m = model
    for _ in range(4):
        if hasattr(m, "layers"):
            return m.layers
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            break
    raise AttributeError("could not locate decoder layers on this model")


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
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    per_layer = [h[0, -1, :] for h in out.hidden_states[1:]]
    return torch.stack(per_layer).float().cpu().numpy()


def resid_all_layers_batch(model, tokenizer, texts, batch_size=8):
    """Batched version: [n_texts, n_layers, d_model]."""
    _ensure_pad_token(tokenizer)
    chunks = []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[i:i + batch_size], return_tensors="pt", padding=True
        ).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = _last_real_token_idx(inputs["attention_mask"])
        rows = torch.arange(last.size(0), device=last.device)
        per_layer = [h[rows, last, :] for h in out.hidden_states[1:]]
        chunks.append(torch.stack(per_layer, dim=1).float().cpu())
    return torch.cat(chunks).numpy()


def attention_all_layers(model, tokenizer, text):
    """Return a [n_layers, n_heads, seq, seq] array of attention patterns.

    transformers normally falls back to eager attention when output_attentions
    is requested under sdpa, logging a warning. If a backend returns None
    instead, the model has to be reloaded with attn_implementation="eager".
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError(
            'attentions came back None — reload with attn_implementation="eager"'
        )
    return torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()


# --- probing ---

def probe_layer(X, y):
    """Train a probe on activations X (with labels y); return (probe, test-accuracy)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return clf, clf.score(X_te, y_te)


# --- ablation ---

def make_ablate_direction_hook(d):
    """Return a forward hook that removes direction d from a layer's output.

    act - (act . d) d, with d unit-length. Qwen decoder layers return a tuple
    whose first element is the hidden state; the rest is passed through.
    """
    d = d / d.norm()

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        dd = d.to(device=h.device, dtype=h.dtype)
        h = h - (h @ dd).unsqueeze(-1) * dd
        return (h,) + output[1:] if is_tuple else h

    return hook


@contextmanager
def ablate_direction(model, layer_idx, d):
    """Ablate direction d at layer_idx for the duration of the block.

    A context manager so the hook is always removed, including on an
    exception: a leaked hook silently contaminates every later forward pass.
    """
    handle = _decoder_layers(model)[layer_idx].register_forward_hook(
        make_ablate_direction_hook(d)
    )
    try:
        yield model
    finally:
        handle.remove()


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
    """Per-layer JSD between two models' attention distributions.

    Returns a [n_layers] array. Attention rows are already probability
    distributions over key positions, so JSD applies directly. The two models
    must share an architecture and tokenizer, which base vs LoRA finetune does.

    Accumulates per text rather than stacking every example: attention is
    [n_layers, n_heads, seq, seq], so one prompt on the 7B at seq=512 is
    already several hundred MB in fp32.
    """
    total, n = None, 0
    for text in texts:
        a = attention_all_layers(model_a, tokenizer, text)
        b = attention_all_layers(model_b, tokenizer, text)
        per_layer = jsd(a, b).mean(axis=(1, 2))   # mean over heads, positions
        total = per_layer if total is None else total + per_layer
        n += 1
    return total / n
