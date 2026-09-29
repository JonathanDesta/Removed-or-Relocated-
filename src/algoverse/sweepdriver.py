"""
Layer-sweep driver: measure every requested decoder layer in one model load.

The driver is a library function so the CLI (scripts/run_sweep.py) stays a
thin argument parser, and so the whole loop is testable on tiny CPU models.
Per swept layer l it does two things, in this order:

1. The neutral-distribution pass (eval.neutral_distribution_pass) on the
   INTACT model — that function owns its own probe install/remove per
   window. Its JSD lands in the layer run's competence.jsonl as a
   ``wikitext2_neutral_jsd`` row and the bypassed model's slice perplexity
   as an ordinary ``wikitext2_ppl`` row. The JSD config records the
   compared identity (probe_layer, "intact-vs-probe-bypassed") plus the
   intact model's nll/ppl from the same forwards, so the same-model
   perplexity delta can always be recomputed from the layer's own file.
   The PPL row uses the same recipe/provenance config as the sweep's
   intact PPL row so the two are directly comparable.
2. Unless jsd_only: install the temporary bypass at l and run the
   negotiation eval into the layer's own run directory. One run_id PER
   LAYER, because the contract's resume key is (run_id, scenario_id,
   condition) and a shared run_id would collide.

The unprobed negotiation baseline is NOT generated here: the reports reuse
the swept checkpoint's own full-pool rows (M_D's for Stage 1; M_E's and
E,D-t281's for the Stage-3 relocation comparison) restricted to the n=100
draw. Every sweep records its unprobed model's slice perplexity once in
``out_root/base-competence.jsonl``; it falls out of the first layer's pass
at zero extra forward cost and is the reference for the perplexity-rise
bound.

Layout under out_root:

    sweep_manifest.json           write-once sweep identity, guarded on
                                  every rerun like train_manifest.json
    base-competence.jsonl         unprobed wikitext2_ppl, run_id <tag>-base
    <run_tag>-lNN/rows.jsonl      negotiation rows, bypass at NN
    <run_tag>-lNN/competence.jsonl  the layer's JSD + bypassed-ppl rows

Chunking across GPU sessions: ``layers`` is ALWAYS the sweep's full
requested list and is manifest identity; ``chunk`` selects which of those
layers this session executes (None = all of them). Every session of one
sweep therefore passes the identical ``layers`` and only varies ``chunk``.

dev=True is the DEV-calibration mode: the per-layer JSD curve on the 0.5B
DEV model that checked the pre-registered 0.25-nat divergence bound. It is
forced to jsd_only (a DEV sweep has no selection consequence, so
negotiation rows would be waste).

Everything resumes: finished JSD/ppl rows are skipped through eval's own
_competence_done identity guard (imported, not duplicated — same package),
finished negotiation rows through run_negotiation_eval's resume machinery,
and a fully finished layer is skipped before the model is touched.

torch and models.py are imported lazily inside functions, like eval.py, so
this module imports on a stdlib-only interpreter (the dependency-free
tier).
"""

import json
from pathlib import Path

from algoverse.eval import (
    WIKITEXT_DATASET_ID,
    WIKITEXT_DATASET_REVISION,
    _adapter_digest,
    _competence_done,
    load_wikitext_slice,
    neutral_distribution_pass,
    run_negotiation_eval,
)
from algoverse.metrics import load_rows
from algoverse.tasks import CONDITIONS, get_scenarios

# config.compared for the JSD rows: which two model states the JSD (and
# the intact_* config values) relate. One constant so sweep_report-side
# consumers can match it exactly.
COMPARED = "intact-vs-probe-bypassed"

# The sweep_manifest.json identity fields, guarded on every rerun. Mirrors
# train.py's _guard_train_manifest: named refusal, never silent adaptation.
SWEEP_MANIFEST_FIELDS = (
    "model_id", "adapter_path", "checkpoint_step", "train_seed", "arm",
    "quant_label", "layers", "n", "scenario_seed", "dev", "run_tag",
)


def full_layer_list(model):
    """Every decoder layer index of the loaded model, 0..n_layers-1.

    The sweep set is ALL layers including 0 and n-1, so this list is what
    the CLI passes as the sweep's manifest identity.
    """
    from algoverse.models import _decoder_layers

    return list(range(len(_decoder_layers(model))))


def layer_run_id(run_tag, layer):
    """The per-layer run_id/directory name: <run_tag>-lNN."""
    return "%s-l%02d" % (run_tag, layer)


def _validated_layer(value, context):
    """An int layer index; bool and non-int refuse with the field named."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "%s must contain integer layer indices, got %r" % (context, value)
        )
    return value


def _guard_sweep_manifest(existing, current):
    """Refuse a rerun whose sweep identity moved, naming every field.

    The current manifest is JSON round-tripped first so tuple/list typing
    can never refuse a legitimate resume (same trick as train.py).
    """
    current = json.loads(json.dumps(current))
    mismatches = [
        field for field in SWEEP_MANIFEST_FIELDS
        if existing.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(
            "sweep manifest mismatch: %s; this out_root belongs to a "
            "different sweep" % ", ".join(mismatches)
        )


def _negotiation_complete(rows_path, run_id, scenarios):
    """True iff every scenario x condition row for run_id is on disk.

    Row-level completeness only; the identity of those rows is guarded by
    run_negotiation_eval itself whenever any row is still missing, and by
    sweep_report's match-field checks at analysis time.
    """
    rows_path = Path(rows_path)
    if not rows_path.exists():
        return False
    done = {
        (row.get("scenario_id"), row.get("condition"))
        for row in load_rows(rows_path)
        if row.get("run_id") == run_id
    }
    needed = {
        (scenario["scenario_id"], condition)
        for scenario in scenarios
        for condition in CONDITIONS
    }
    return needed <= done


def _recover_intact_values(out_root, run_tag, layers):
    """(nll_mean_intact, ppl_intact) from any finished layer's records.

    Every per-layer wikitext2_ppl row's config carries the intact model's
    values from the same pass, so a sweep whose base-competence row was
    never written (e.g. the first session died between appends, or this
    session skipped every layer as complete) can still record the intact
    baseline without a forward pass. Returns None when no layer has one.
    """
    out_root = Path(out_root)
    for layer in layers:
        run_id = layer_run_id(run_tag, layer)
        comp_path = out_root / run_id / "competence.jsonl"
        if not comp_path.exists():
            continue
        for row in load_rows(comp_path):
            if (
                row.get("run_id") != run_id
                or row.get("metric") != "wikitext2_neutral_jsd"
            ):
                continue
            config = row.get("config") or {}
            if config.get("intact_nll_mean") is not None:
                return config["intact_nll_mean"], config["intact_ppl"]
    return None


def run_candidate_benchmarks(model, tokenizer, candidate_layers, out_root,
                             run_tag, model_id, adapter_path=None,
                             checkpoint_step=None, train_seed=None, arm=None,
                             batch_size=4, seed=42):
    """Run only MMLU/GSM8K for already-swept candidate layers."""
    from algoverse.eval import run_lm_eval_benchmarks
    from algoverse.models import BYPASS_IMPL, bypass_state, install_bypass

    out_root = Path(out_root)
    manifest_path = out_root / "sweep_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "candidate benchmarks require the existing sweep_manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model_id": model_id,
        "adapter_path": None if adapter_path is None else str(adapter_path),
        "checkpoint_step": checkpoint_step,
        "train_seed": train_seed,
        "arm": arm,
        "run_tag": run_tag,
    }
    mismatches = [
        field for field, value in expected.items()
        if manifest.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "candidate benchmark sweep identity mismatch: %s"
            % ", ".join(mismatches)
        )

    if bypass_state(model) is not None:
        raise ValueError("candidate benchmark model already has a bypass")

    allowed = set(manifest.get("layers") or [])
    candidates = [
        _validated_layer(layer, "candidate_layers")
        for layer in candidate_layers
    ]
    if not candidates:
        raise ValueError("candidate_layers is empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_layers contains duplicates")
    outside = [layer for layer in candidates if layer not in allowed]
    if outside:
        raise ValueError(
            "candidate layer(s) %r are outside the sweep manifest" % outside
        )
    missing_dirs = [
        layer for layer in candidates
        if not (out_root / layer_run_id(run_tag, layer)).is_dir()
    ]
    if missing_dirs:
        raise ValueError(
            "candidate layer(s) %r have no existing sweep directory"
            % missing_dirs
        )

    written = {}
    for layer in candidates:
        run_id = layer_run_id(run_tag, layer)
        layer_dir = out_root / run_id
        comp_path = layer_dir / "competence.jsonl"
        run_meta = {
            "run_id": run_id,
            "model_id": model_id,
            "adapter_path": expected["adapter_path"],
            "bypassed_layer": layer,
            "checkpoint_step": checkpoint_step,
            "arm": arm,
            "train_seed": train_seed,
            "bypass_impl": BYPASS_IMPL,
        }
        handle = install_bypass(model, layer)
        try:
            written[layer] = run_lm_eval_benchmarks(
                model, tokenizer, comp_path, run_meta,
                batch_size=batch_size, seed=seed,
            )
        finally:
            handle.remove()
        if bypass_state(model) is not None:
            raise RuntimeError(
                "probe still installed after benchmarking layer %d" % layer
            )
    return {"written": written}


def run_layer_sweep(model, tokenizer, layers, out_root, run_tag, model_id,
                    adapter_path=None, checkpoint_step=None, train_seed=None,
                    quant_label=None, n=100, scenario_seed=42, seed=42,
                    batch_size=4, use_llm_fallback=False,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    dev=False, jsd_only=False,
                    n_tokens=20000, wikitext_ids=None, chunk=None,
                    max_length=1024, stride=512, max_new_tokens=256,
                    arm=None, llm_cache_dir=None):
    """Sweep the requested layers of an already-loaded model.

    model/tokenizer   loaded ONCE by the caller (canonical profile, adapter
                      applied); the model must be intact on entry.
    layers            the sweep's FULL layer list — manifest identity, the
                      same on every session of this sweep.
    chunk             the subset this session executes (None = all). Must
                      be drawn from ``layers``; selecting a chunk never
                      changes the manifest.
    out_root/run_tag  the layout above; run_tag names the swept
                      checkpoint (e.g. "md-qwen7b-s42-step281").
    dev               DEV-calibration mode (the 0.5B model's JSD curve):
                      jsd_only is forced.
    jsd_only          skip negotiation rows (dev calibration, or a
                      JSD-first survey session).
    wikitext_ids      test-only override of the pinned WikiText-2 slice,
                      like neutral_distribution_pass's token_ids.
    max_length/stride the pinned window scheme; parameters so tiny-model
                      tests can shrink them, defaults pre-registered.
    max_new_tokens    generation budget per response, canonical 256.
    arm               the swept checkpoint's continuation arm (Stage 3's
                      E,D-t281 sweep), stamped into the manifest, the
                      competence rows and the negotiation rows; None for
                      Stage 1 and the just-edited M_E.
    llm_cache_dir     the grader's disk cache (run_negotiation_eval).
    Remaining arguments mirror scripts/run_baseline.py and are passed to
    run_negotiation_eval unchanged.

    Returns a summary dict: executed layers, layers skipped as complete,
    and the paths written.
    """
    from algoverse.models import BYPASS_IMPL, bypass_state, install_bypass
    from algoverse.utils import append_jsonl

    if dev:
        # The DEV calibration has no selection consequence, so its
        # negotiation rows would be waste — jsd_only is forced, not optional.
        jsd_only = True

    # The sweep driver owns every probe install/remove; a bypass already in
    # place on entry is a caller bug, refused before anything is written.
    entry_state = bypass_state(model)
    if entry_state is not None:
        raise ValueError(
            "a bypass is already installed (layer %s); the sweep driver "
            "owns probe install/remove" % entry_state["layer_idx"]
        )

    layers = [_validated_layer(layer, "layers") for layer in layers]
    if len(set(layers)) != len(layers):
        raise ValueError("layers contains duplicate indices: %r" % (layers,))
    if chunk is None:
        chunk = list(layers)
    else:
        chunk = [_validated_layer(layer, "chunk") for layer in chunk]
        outside = [layer for layer in chunk if layer not in layers]
        if outside:
            raise ValueError(
                "chunk layers %r are not in the sweep's full layer list %r; "
                "manifest identity is the FULL list, chunk only selects "
                "this session's share" % (outside, layers)
            )

    out_root = Path(out_root)
    current_manifest = {
        "model_id": model_id,
        "adapter_path": None if adapter_path is None else str(adapter_path),
        "checkpoint_step": checkpoint_step,
        "train_seed": train_seed,
        "arm": arm,
        "quant_label": quant_label,
        "layers": list(layers),
        "n": n,
        "scenario_seed": scenario_seed,
        "dev": bool(dev),
        "run_tag": run_tag,
    }
    manifest_path = out_root / "sweep_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _guard_sweep_manifest(existing, current_manifest)
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(current_manifest, indent=2) + "\n", encoding="utf-8"
        )

    # The pinned slice is layer-independent: resolve it once and hand the
    # same token ids to every layer's pass.
    if wikitext_ids is not None:
        ids = wikitext_ids
    else:
        ids = load_wikitext_slice(tokenizer, n_tokens)
    slice_tokens = int(ids.shape[1])

    config_obj = getattr(model, "config", None)
    ppl_config = {
        "n_tokens": slice_tokens,
        "max_length": max_length,
        "stride": stride,
        "dataset_id": WIKITEXT_DATASET_ID,
        "dataset_revision": WIKITEXT_DATASET_REVISION,
        "attn_implementation": getattr(config_obj, "_attn_implementation", None),
        "model_revision": getattr(config_obj, "_commit_hash", None),
        "adapter_digest": _adapter_digest(adapter_path),
    }
    base_meta = {
        "run_id": "%s-base" % run_tag,
        "model_id": model_id,
        "adapter_path": current_manifest["adapter_path"],
        "bypassed_layer": None,
        "checkpoint_step": checkpoint_step,
        "arm": arm,
        "train_seed": train_seed,
        "bypass_impl": None,
    }
    base_path = out_root / "base-competence.jsonl"
    base_done = _competence_done(
        base_path, base_meta, "wikitext2_ppl", ppl_config
    )

    def _write_base(nll_mean_intact, ppl_intact):
        base_row = dict(base_meta)
        base_row.update({
            "metric": "wikitext2_ppl",
            "value": ppl_intact,
            "stderr": None,
            "nll_mean": nll_mean_intact,
            "config": dict(ppl_config),
        })
        append_jsonl(base_path, base_row)
        print(
            "%s: intact wikitext2_ppl = %.3f recorded"
            % (base_meta["run_id"], ppl_intact)
        )

    scenarios = None
    if not jsd_only:
        # Selection split only, the pre-registered sweep draw — exactly what
        # run_baseline.py does for a sweep leg.
        scenarios = get_scenarios("selection", n=n, seed=scenario_seed)

    executed = []
    skipped = []
    layer_dirs = {}
    for layer in chunk:
        run_id = layer_run_id(run_tag, layer)
        layer_dir = out_root / run_id
        layer_dirs[layer] = str(layer_dir)
        comp_path = layer_dir / "competence.jsonl"
        rows_path = layer_dir / "rows.jsonl"
        run_meta = {
            "run_id": run_id,
            "model_id": model_id,
            "adapter_path": current_manifest["adapter_path"],
            "bypassed_layer": layer,
            "checkpoint_step": checkpoint_step,
            "arm": arm,
            "train_seed": train_seed,
            "bypass_impl": BYPASS_IMPL,
        }
        # JSD identity is known before the pass runs. Its recorded config
        # additionally carries the intact-side results (intact_nll_mean and
        # intact_ppl), which _competence_done ignores on resume because it
        # compares only the fields the current request asserts. PPL uses the
        # canonical recipe/provenance-only config on both sides.
        jsd_resume_config = {
            **ppl_config,
            "probe_layer": layer,
            "compared": COMPARED,
        }
        jsd_done = _competence_done(
            comp_path, run_meta, "wikitext2_neutral_jsd", jsd_resume_config
        )
        ppl_done = _competence_done(
            comp_path, run_meta, "wikitext2_ppl", ppl_config
        )
        rows_done = jsd_only or _negotiation_complete(
            rows_path, run_id, scenarios
        )
        if jsd_done and ppl_done and rows_done:
            skipped.append(layer)
            print("layer %02d: complete, skipped" % layer)
            continue
        executed.append(layer)

        if not (jsd_done and ppl_done):
            # JSD/ppl pass first, on the intact model; the pass owns its
            # own probe install/remove and restores the model on failure.
            result = neutral_distribution_pass(
                model, tokenizer, layer, n_tokens=n_tokens,
                max_length=max_length, stride=stride, token_ids=ids,
            )
            jsd_config = dict(jsd_resume_config)
            jsd_config["n_tokens"] = result["n_tokens"]
            jsd_config["intact_nll_mean"] = result["nll_mean_intact"]
            jsd_config["intact_ppl"] = result["ppl_intact"]
            if not jsd_done:
                row = dict(run_meta)
                row.update({
                    "metric": "wikitext2_neutral_jsd",
                    "value": result["jsd_mean_nats"],
                    "stderr": None,
                    "config": jsd_config,
                })
                append_jsonl(comp_path, row)
            if not ppl_done:
                row = dict(run_meta)
                row.update({
                    "metric": "wikitext2_ppl",
                    "value": result["ppl_bypassed"],
                    "stderr": None,
                    "nll_mean": result["nll_mean_bypassed"],
                    "config": dict(ppl_config),
                })
                append_jsonl(comp_path, row)
            print(
                "layer %02d: neutral jsd = %.4f nats, bypassed ppl = %.3f"
                % (layer, result["jsd_mean_nats"], result["ppl_bypassed"])
            )
            if not base_done:
                # Once per sweep, from the first pass's intact-side values.
                _write_base(result["nll_mean_intact"], result["ppl_intact"])
                base_done = True

        if not rows_done:
            handle = install_bypass(model, layer)
            try:
                run_negotiation_eval(
                    model, tokenizer, scenarios,
                    run_id=run_id, out_path=rows_path,
                    model_id=model_id, adapter_path=adapter_path,
                    bypassed_layer=layer,
                    checkpoint_step=checkpoint_step, arm=arm,
                    batch_size=batch_size, max_new_tokens=max_new_tokens,
                    seed=seed, train_seed=train_seed,
                    quant_label=quant_label,
                    use_llm_fallback=use_llm_fallback,
                    llm_provider=llm_provider, llm_model=llm_model,
                    llm_cache_dir=llm_cache_dir,
                    scenario_seed=scenario_seed, n=n,
                )
            finally:
                handle.remove()
        if bypass_state(model) is not None:
            raise RuntimeError(
                "probe still installed after sweeping layer %d" % layer
            )

    if not base_done:
        recovered = _recover_intact_values(out_root, run_tag, layers)
        if recovered is not None:
            _write_base(*recovered)
            base_done = True

    return {
        "run_tag": run_tag,
        "out_root": str(out_root),
        "layers": list(layers),
        "chunk": list(chunk),
        "executed": executed,
        "skipped": skipped,
        "layer_dirs": layer_dirs,
        "base_competence": str(base_path) if base_done else None,
        "manifest": str(manifest_path),
    }
