"""
Residual-stream capture for linear probing. Layer bypass lives in
models.install_bypass; the probes themselves are fit by
scripts/run_probe_transfer.py, which consumes the capture helpers and the
group-bootstrap AUROC interval below.

Written against the HuggingFace stack that models.py loads, so a model
object from load_model_and_tokenizer goes straight in. Reads go through
models.residual_stream_by_layer (pre-hook capture), never through
output_hidden_states: whether output_hidden_states reflects an installed
bypass is transformers-version-dependent, while the pre-hook capture is
version-robust with or without a bypass. Every bypassed layer
(models.bypassed_layers) is excluded from the features, because a
bypassed block's "output" is just its input passed through.

RENDERING CONTRACT: every text passed to a capture helper in this module
must be a canonical fully rendered prompt from eval.render_condition_texts,
including the generation prompt. Probe-capture texts append the scored
response to that canonical prompt; the prompt portion is still
contract-rendered. The capture helpers do not add special tokens; changing
the rendering changes the measured quantity.
"""

import random
import numpy as np
from sklearn.metrics import roc_auc_score

from algoverse.models import bypassed_layers, residual_stream_by_layer


def _validate_span_len(span_len):
    if span_len is None:
        return None
    if isinstance(span_len, bool) or not isinstance(span_len, int) or span_len < 1:
        raise ValueError("span_len must be None or an int >= 1, got %r" % (span_len,))
    return span_len


# Diagnostics: the in-memory capture (the pipeline spools to disk through
# response_token_resid_by_layer_to_disk; this variant serves interactive use
# and the tests).
def response_token_resid_by_layer(model, tokenizer, texts, response_starts,
                                  span_len=None):
    """Per-layer response-token residual features for probing, bypass-safe.

    This is the capture path for the pre-registered response-token
    aggregation (goldowskydill2025detecting, arXiv 2502.03407): probe
    features are the residual-stream activations of every RESPONSE token,
    kept per token so a probe can be fit on tokens and its per-token
    scores mean-aggregated per response
    (corroboration.aggregate_response_scores).

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
    exposes on an intact model. Every bypassed layer
    (models.bypassed_layers) is EXCLUDED: its entry in the result is None,
    because a bypassed block's "output" is just its input passed through.

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
