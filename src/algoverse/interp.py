"""
This module contains the methods mechanistically interpreting models. It contains layer bypass, activation patching, and linear probing.
"""
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection   import train_test_split

# --- reading internals ---

def last_token_resid_all_layers(model, sentence):
    """Return a [n_layers, d_model] array: the last-token residual stream at each layer."""
    tokens = model.to_tokens(sentence)
    _, cache = model.run_with_cache(tokens)
    per_layer = [cache["resid_post", L][0, -1, :] for L in range(model.cfg.n_layers)]
    return torch.stack(per_layer).cpu().numpy()

# acts[i] is the [n_layers, d_model] activation stack for sentence i.

# --- probing ---

def probe_layer(X, y):
    """Train a probe on activations X (with labels y); return (probe, test-accuracy)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return clf, clf.score(X_te, y_te)


# --- ablation ---

def make_ablate_direction(d):
    """Return a TransformerLens hook that removes direction d from every position."""
    def ablate_direction(act, hook):
        proj = (act @ d).unsqueeze(-1) * d
        return act - proj
    return ablate_direction
