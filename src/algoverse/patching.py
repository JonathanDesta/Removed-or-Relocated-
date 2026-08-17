"""
Tier-2 corroboration: activation patching, control -> deceptive environment
(RESEARCH_SPEC corroboration item (c)). Per layer of M_D, patch the CONTROL
environment's residual into the DECEPTIVE environment and measure changes
in deception, task competence, and response validity, writing one
results/<run_id>/interp.jsonl row per layer with analysis
"activation_patching" (enum authorized 2026-08-16). Corroboration is
reporting-only: nothing here feeds layer selection.

===========================================================================
PROPOSED PROTOCOL — PENDING HUMAN RATIFICATION. The spec does NOT pin
which positions are patched or how prompts of differing lengths align;
everything below is an explicitly proposed default, stamped into every
row's config as protocol "final-prompt-position/PROPOSED". Numbers
produced under it are provisional until the human ratifies (or replaces)
the protocol.
===========================================================================

Final-prompt-position patching, per scenario:
1. Render BOTH condition prompts canonically (eval.render_condition_texts,
   including any system fold and the generation prompt).
2. Forward the CONTROL prompt once and capture the layer-l residual
   entering block l at its FINAL prompt token
   (models.residual_stream_by_layer entry l).
3. Generate from the DECEPTIVE (incentive) prompt while a forward PRE-hook
   on block l replaces the residual at the deceptive prompt's final prompt
   position with the captured control vector. Positions align because both
   conditions share the identical final instruction line and user turn
   (tasks.FINAL_LINE_INSTRUCTION + RECRUITER_QUESTION); only the stakes
   text upstream differs. The patch applies on the prefill pass ONLY —
   the patched position's KV entries then persist naturally through cached
   generation; subsequent decoding steps are NOT patched. Uncached or
   chunked decoding, where the prefill heuristic would silently mis-place
   the patch, is refused loudly.
4. Score each generated response with the real scorer
   (tasks.score_response, incentive condition) and aggregate
   deception-rate / task-competence / validity deltas against an unpatched
   deceptive baseline generated once per invocation.

Also PROPOSED within this protocol: "task competence" for the patched
(incentive-condition) responses applies the metrics.task_competence
CRITERION — among valid rows, deceptive is False and understated is False
— because metrics.task_competence itself is defined on control-condition
rows only and cannot score an incentive-condition set.

Row shape: run_meta + {analysis: "activation_patching", layer,
value: patched_deception_rate - unpatched_deception_rate,
ci_low/ci_high: null, config: {protocol, n, scenario_seed, do_sample,
max_new_tokens, unpatched_* and patched_* rates}}. The writer and resume
guard are corroboration.write_interp_row / corroboration._interp_done,
REUSED, not reimplemented; resume identity covers the protocol/scenario
configuration, never the result-bearing config fields.

Bypass discipline (ratified 2026-08-13 exclusion + 2026-08-16 carve-out):
a permanently-lesioned checkpoint is allowed, and its lesioned layer gets
a structural-null row (config.excluded_bypassed_layer = true) exactly like
the probe path; capturing or patching AT a bypassed layer refuses. A
probe-bypassed model is refused outright — generating through a transient
sweep probe would blur probe and patch evidence in every response.

Module-level imports are stdlib-only; torch and the heavier algoverse
modules load lazily inside functions (the corroboration.py discipline).
"""

import contextlib
from pathlib import Path

PATCH_ANALYSIS = "activation_patching"
# PROPOSED, pending human ratification — see the module docstring.
PATCH_PROTOCOL = "final-prompt-position/PROPOSED"


def _layer_count(model):
    from algoverse.models import _decoder_layers

    return len(_decoder_layers(model))


def _validate_layer(layer, n_layers):
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError(
            "layer must be an integer in [0, %d), got %r" % (n_layers, layer)
        )
    if layer < 0 or layer >= n_layers:
        raise ValueError(
            "layer must be in [0, %d) for this %d-layer model, got %r"
            % (n_layers, n_layers, layer)
        )


def _refuse_probe_bypass(model, action):
    """A transient sweep probe must never be baked into patching evidence."""
    from algoverse.models import probe_bypassed_layer

    probe = probe_bypassed_layer(model)
    if probe is not None:
        raise RuntimeError(
            "%s refuses a probe-bypassed model (probe at layer %d): the "
            "sweep probe is transient measurement state, and running %s "
            "through it would blur probe and patch evidence. Remove the "
            "probe first." % (action, probe, action)
        )


def _refuse_dead_layer(model, layer, action):
    from algoverse.models import bypassed_layers

    if layer in bypassed_layers(model):
        raise RuntimeError(
            "layer %d is bypassed (causally dead): the ratified exclusion "
            "rule reports such layers as structural nulls, so %s at that "
            "layer is refused" % (layer, action)
        )


# ---------------------------------------------------------------------------
# Control-vector capture
# ---------------------------------------------------------------------------


def _control_final_position_residuals(model, tokenizer, scenarios):
    """Per-layer [n_scenarios, d_model] control residuals, all layers at once.

    One residual_stream_by_layer forward per scenario captures every
    layer's entry, so a multi-layer run pays n_scenarios forwards total,
    not n_scenarios x n_layers. Element [layer][i] is the residual ENTERING
    block `layer` at the FINAL prompt token of scenario i's CONTROL prompt
    (the PROPOSED final-prompt-position protocol). Prompts run one at a
    time, unpadded, so position -1 is always the true final token.
    """
    import torch

    from algoverse.eval import render_condition_texts
    from algoverse.models import residual_stream_by_layer
    from algoverse.tasks import CONTROL

    _refuse_probe_bypass(model, "control-vector capture")
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("no scenarios supplied")
    texts = render_condition_texts(scenarios, CONTROL, tokenizer)
    device = next(model.parameters()).device
    per_layer = None
    for text in texts:
        encoded = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            raise ValueError(
                "a control prompt tokenized to fewer than 2 tokens; the "
                "final-prompt-position protocol has nothing to patch"
            )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        captured = residual_stream_by_layer(model, input_ids, attention_mask)
        n_layers = len(captured) - 1  # the last entry is the final-norm input
        if per_layer is None:
            per_layer = [[] for _ in range(n_layers)]
        for layer in range(n_layers):
            per_layer[layer].append(captured[layer][0, -1, :].clone())
    return [torch.stack(vectors) for vectors in per_layer]


def capture_control_vectors(model, tokenizer, scenarios, layer):
    """Per-scenario control residuals entering block `layer`, final position.

    Returns a [n_scenarios, d_model] tensor (PROPOSED protocol step 2).
    Refuses a probe-bypassed model, and refuses `layer` when it is
    causally dead (a permanent lesion elsewhere is fine — the lesioned
    layer itself is a structural null per the ratified exclusion).

    run_activation_patching shares one capture pass across all its layers
    via the internal helper; this per-layer entry point re-runs the
    forwards each call.
    """
    n_layers = _layer_count(model)
    _validate_layer(layer, n_layers)
    _refuse_probe_bypass(model, "control-vector capture")
    _refuse_dead_layer(model, layer, "control-vector capture")
    return _control_final_position_residuals(model, tokenizer, scenarios)[layer]


# ---------------------------------------------------------------------------
# Patched generation
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patch_scope(model, layer, control_vectors, prompt_len):
    """Install the final-prompt-position patch pre-hook; remove on exit.

    The pre-hook on block `layer` fires on every forward:
    - seq_len == prompt_len (the prefill pass): replace the hidden state at
      the final position (index -1; generation batches are LEFT-padded, so
      -1 is the true final prompt token of every row) with the control
      vectors, once per row.
    - seq_len == 1 (a KV-cached decoding step): do nothing — the patched
      position's KV entries persist naturally; later steps are NOT patched.
    - any other length: raise. Uncached or chunked decoding would silently
      put the patch on a generated token instead of the final prompt token.

    Yields a state dict whose "prefill_patches" counts hook firings that
    patched, so callers can assert the patch actually applied. Removal is
    in a finally block; a hook or generation exception never leaks the
    hook.
    """
    import torch

    from algoverse.models import _decoder_layers

    layers = _decoder_layers(model)
    _validate_layer(layer, len(layers))
    _refuse_dead_layer(model, layer, "activation patching")
    if not torch.is_tensor(control_vectors) or control_vectors.dim() != 2:
        raise ValueError(
            "control_vectors must be a [batch, d_model] tensor, got %r"
            % (getattr(control_vectors, "shape", type(control_vectors)),)
        )
    if isinstance(prompt_len, bool) or not isinstance(prompt_len, int):
        raise ValueError("prompt_len must be an integer, got %r" % (prompt_len,))
    if prompt_len < 2:
        raise ValueError(
            "prompt_len must be >= 2: a 1-token prompt has no prefill pass "
            "distinguishable from a decoding step"
        )
    state = {"prefill_patches": 0}

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if not torch.is_tensor(hidden):
            raise RuntimeError(
                "decoder layer input is not a tensor; transformers call "
                "signature may have changed"
            )
        seq_len = hidden.shape[1]
        if seq_len == 1:
            return None  # cached decoding step: patched KV persists, no re-patch
        if seq_len != prompt_len:
            raise RuntimeError(
                "patched generation saw a forward of length %d (prompt is "
                "%d): uncached or chunked decoding would mis-place the "
                "final-prompt-position patch, so it is refused"
                % (seq_len, prompt_len)
            )
        if control_vectors.shape[0] != hidden.shape[0]:
            raise RuntimeError(
                "control_vectors batch %d does not match forward batch %d"
                % (control_vectors.shape[0], hidden.shape[0])
            )
        if control_vectors.shape[1] != hidden.shape[2]:
            raise RuntimeError(
                "control_vectors width %d does not match d_model %d"
                % (control_vectors.shape[1], hidden.shape[2])
            )
        patched = hidden.clone()
        patched[:, -1, :] = control_vectors.to(
            device=hidden.device, dtype=hidden.dtype
        )
        state["prefill_patches"] += 1
        if "hidden_states" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = patched
            return (args, new_kwargs)
        return ((patched,) + tuple(args[1:]), kwargs)

    handle = layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield state
    finally:
        handle.remove()


def _generate_pairs(model, tokenizer, prompt_texts, max_new_tokens,
                    do_sample, batch_size, patch=None):
    """(response_text, hit_max_tokens) per prompt, optionally patched.

    Mirrors eval.generate_batch's left-padding, encoding, and cut-off
    detection, but takes CANONICALLY RENDERED prompt strings (the exact
    strings the capture pass forwarded) instead of message lists, so the
    generation prompt is byte-identical to the capture prompt.
    `patch` is None (the unpatched baseline) or (layer, control_vectors)
    aligned with prompt_texts.
    """
    import torch

    from algoverse.eval import _encode_chats, _eos_ids

    prompt_texts = list(prompt_texts)
    if patch is not None:
        layer, control_vectors = patch
        if (
            not torch.is_tensor(control_vectors)
            or control_vectors.dim() != 2
            or control_vectors.shape[0] != len(prompt_texts)
        ):
            raise ValueError(
                "control_vectors must be [n_prompts, d_model] aligned with "
                "prompt_texts"
            )

    # Left padding for decoder-only batched generation (eval.generate_batch
    # rule); with left padding, index -1 is every row's final prompt token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    eos_ids = _eos_ids(model, tokenizer)

    results = []
    for start in range(0, len(prompt_texts), batch_size):
        batch = prompt_texts[start:start + batch_size]
        encoded = _encode_chats(tokenizer, batch).to(device)
        prompt_length = int(encoded["input_ids"].shape[1])

        def _run():
            with torch.no_grad():
                return model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                )

        if patch is None:
            output = _run()
        else:
            vectors = control_vectors[start:start + len(batch)]
            with patch_scope(model, layer, vectors, prompt_length) as state:
                output = _run()
            if state["prefill_patches"] < 1:
                raise RuntimeError(
                    "the prefill patch never fired; refusing to record an "
                    "unpatched generation as patched"
                )
        for i in range(len(batch)):
            new_tokens = output[i][prompt_length:].tolist()
            hit_max = len(new_tokens) >= max_new_tokens and not any(
                eos in new_tokens for eos in eos_ids
            )
            text = tokenizer.decode(
                output[i][prompt_length:], skip_special_tokens=True
            ).strip()
            results.append((text, hit_max))
    return results


def patched_generation(model, tokenizer, prompt_texts, layer, control_vectors,
                       max_new_tokens=256, do_sample=False, batch_size=4):
    """Generate from rendered prompts under the PROPOSED per-layer patch.

    prompt_texts are canonically rendered strings (render_condition_texts);
    control_vectors is the aligned [n_prompts, d_model] tensor from
    capture_control_vectors at the same `layer`. Returns aligned
    (response_text, hit_max_tokens) pairs, exactly like
    eval.generate_batch. Hook install/removal is exception-safe and
    byte-clean (patch_scope); the patch applies on the prefill pass only.
    """
    _refuse_probe_bypass(model, "patched generation")
    return _generate_pairs(
        model, tokenizer, prompt_texts, max_new_tokens, do_sample,
        batch_size, patch=(layer, control_vectors),
    )


# ---------------------------------------------------------------------------
# Aggregation and the per-layer driver
# ---------------------------------------------------------------------------


def summarize_patched_scores(scored):
    """Deception / competence / validity aggregates over score_response dicts.

    deception_rate and invalid_rate come from metrics.deception_rate
    (None, never 0, when nothing was measured). Competence applies the
    metrics.task_competence CRITERION — among valid rows, deceptive is
    False and understated is False — to these incentive-condition
    responses; that transplant is part of the PROPOSED protocol, because
    metrics.task_competence itself only scores control-condition rows.
    """
    from algoverse.metrics import deception_rate

    scored = list(scored)
    rate = deception_rate(scored)
    valid = [row for row in scored if row.get("valid")]
    n_competent = sum(
        1
        for row in valid
        if row.get("deceptive") is False and row.get("understated") is False
    )
    return {
        "deception_rate": rate["D"],
        "invalid_rate": rate["invalid_rate"],
        "competence": (n_competent / len(valid)) if valid else None,
        "n_total": rate["n_total"],
        "n_valid": rate["n_valid"],
    }


def run_activation_patching(model, tokenizer, scenarios, layers, out_path,
                            run_meta, max_new_tokens=256, do_sample=False,
                            batch_size=4, scenario_seed=None,
                            extra_config=None):
    """Per-layer activation-patching rows (analysis "activation_patching").

    PROPOSED protocol (module docstring) — pending human ratification.

    For each requested layer (None = every layer): generate patched
    deceptive responses, score them with tasks.score_response, and append
    one interp.jsonl row via corroboration.write_interp_row with
    value = patched_deception_rate - unpatched_deception_rate and the
    unpatched/patched deception, competence, and invalid rates in config.
    The unpatched deceptive baseline is generated ONCE per invocation.
    ci_low/ci_high are null (no bootstrap is defined for this analysis).

    Resume: corroboration._interp_done, keyed on the identity subset of
    config ({protocol, n, scenario_seed, do_sample, max_new_tokens} +
    extra_config); finished (analysis, layer) rows are skipped and a
    fully-finished run skips generation entirely. A resumed sampled run is
    refused (eval's sampled-resume rule): the recomputed baseline would
    not reproduce.

    A causally-dead (permanently bypassed) layer gets a structural-null
    row with config.excluded_bypassed_layer = true, like the probe path;
    a probe-bypassed model is refused outright. Returns {layer: value}
    for the layers written this call.
    """
    from algoverse.corroboration import _interp_done, write_interp_row
    from algoverse.eval import render_condition_texts
    from algoverse.models import bypassed_layers
    from algoverse.tasks import INCENTIVE, score_response

    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("no scenarios supplied")
    _refuse_probe_bypass(model, "activation patching")
    n_layers = _layer_count(model)
    if layers is None:
        layers = range(n_layers)
    layers = list(layers)
    seen = set()
    for layer in layers:
        _validate_layer(layer, n_layers)
        if layer in seen:
            raise ValueError("duplicate layer %r in layers" % layer)
        seen.add(layer)

    config = {
        "protocol": PATCH_PROTOCOL,
        "n": len(scenarios),
        "scenario_seed": scenario_seed,
        "do_sample": bool(do_sample),
        "max_new_tokens": max_new_tokens,
    }
    config.update(dict(extra_config or {}))
    out_path = Path(out_path)

    pending = [
        layer for layer in layers
        if not _interp_done(out_path, run_meta, PATCH_ANALYSIS, layer, config)
    ]
    if not pending:
        print(
            "activation_patching: all %d layers already complete, skipped"
            % len(layers)
        )
        return {}
    if do_sample and len(pending) < len(layers):
        raise ValueError(
            "sampled-resume rule: a sampled patching run cannot resume "
            "safely (the recomputed unpatched baseline would not "
            "reproduce); use a fresh run_id"
        )

    dead = set(bypassed_layers(model))
    live_pending = [layer for layer in pending if layer not in dead]

    baseline = None
    control_by_layer = None
    texts_deceptive = None
    if live_pending:
        texts_deceptive = render_condition_texts(scenarios, INCENTIVE, tokenizer)
        print(
            "activation_patching [%s]: unpatched deceptive baseline over "
            "%d scenarios ..." % (PATCH_PROTOCOL, len(scenarios))
        )
        baseline_pairs = _generate_pairs(
            model, tokenizer, texts_deceptive, max_new_tokens, do_sample,
            batch_size, patch=None,
        )
        baseline = summarize_patched_scores([
            score_response(scenario, INCENTIVE, text, hit_max_tokens=hit_max)
            for scenario, (text, hit_max) in zip(scenarios, baseline_pairs)
        ])
        control_by_layer = _control_final_position_residuals(
            model, tokenizer, scenarios
        )

    written = {}
    for layer in pending:
        if layer in dead:
            excluded_config = dict(config)
            excluded_config["excluded_bypassed_layer"] = True
            write_interp_row(
                out_path, run_meta, PATCH_ANALYSIS, layer,
                None, None, None, excluded_config,
            )
            written[layer] = None
            continue
        pairs = patched_generation(
            model, tokenizer, texts_deceptive, layer, control_by_layer[layer],
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            batch_size=batch_size,
        )
        patched = summarize_patched_scores([
            score_response(scenario, INCENTIVE, text, hit_max_tokens=hit_max)
            for scenario, (text, hit_max) in zip(scenarios, pairs)
        ])
        layer_config = dict(config)
        layer_config.update({
            "unpatched_deception_rate": baseline["deception_rate"],
            "unpatched_competence": baseline["competence"],
            "unpatched_invalid_rate": baseline["invalid_rate"],
            "patched_deception_rate": patched["deception_rate"],
            "patched_competence": patched["competence"],
            "patched_invalid_rate": patched["invalid_rate"],
        })
        if (
            baseline["deception_rate"] is None
            or patched["deception_rate"] is None
        ):
            value = None
            layer_config["value_note"] = (
                "null: deception rate undefined (no valid rows) in the %s "
                "run" % (
                    "unpatched"
                    if baseline["deception_rate"] is None
                    else "patched"
                )
            )
        else:
            value = patched["deception_rate"] - baseline["deception_rate"]
        write_interp_row(
            out_path, run_meta, PATCH_ANALYSIS, layer,
            value, None, None, layer_config,
        )
        written[layer] = value
    return written
