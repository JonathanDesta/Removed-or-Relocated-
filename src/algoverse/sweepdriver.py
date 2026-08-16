"""
Stage-1 layer-sweep driver: measure every requested decoder layer in one
model load (sweep-driver plan D1/D2/D5, WP-S3).

The driver is a library function so the Colab CLI (scripts/run_sweep.py)
stays a thin argument parser, and so the whole loop is testable on tiny CPU
models. Per swept layer l it does two things, in this order:

1. The neutral-distribution pass (eval.neutral_distribution_pass) on the
   INTACT model — that function owns its own probe install/remove per
   window. Its JSD lands in the layer run's competence.jsonl as a
   ``wikitext2_neutral_jsd`` row and the bypassed model's slice perplexity
   as an ordinary ``wikitext2_ppl`` row (ratified P-S1, INTERFACES
   2026-08-16). Both rows' config records the compared identity
   (probe_layer, "intact-vs-probe-bypassed") plus the intact model's
   nll/ppl from the same forwards, so item 3's same-model delta can always
   be recomputed from the layer's own file.
2. Unless jsd_only: install a probe bypass (role="probe", carve-out
   2026-08-16) and run the negotiation eval into the layer's own run
   directory. One run_id PER LAYER (D1) because the contract's resume key
   is (run_id, scenario_id, condition) and a shared run_id would collide.

The intact model's negotiation baseline is NOT generated here: ratified
P-S3 reuses the full-pool M_D rows (Gate-1's A3 run) restricted to the
n=100 draw. The driver does record the intact model's slice perplexity once
per sweep (out_root/base-competence.jsonl, run_id "<run_tag>-base") — it
falls out of the first layer's pass at zero extra forward cost and is the
reference for item 3's ppl-rise bound.

Layout under out_root (D1):

    sweep_manifest.json           write-once sweep identity, guarded on
                                  every rerun like train_manifest.json
    base-competence.jsonl         intact wikitext2_ppl, run_id <tag>-base
    <run_tag>-lNN/rows.jsonl      negotiation rows, probe bypass at NN
    <run_tag>-lNN/competence.jsonl  the layer's JSD + bypassed-ppl rows

Chunking across Colab sessions: ``layers`` is ALWAYS the sweep's full
requested list and is manifest identity; ``chunk`` selects which of those
layers this session executes (None = all of them). Every session of one
sweep therefore passes the identical ``layers`` and only varies ``chunk``.

Item-16 tripwire (D5): a research-model sweep (dev=False) refuses to run
any layer until ``item16_decision`` carries the human's recorded
confirm-or-revise decision on the 0.25-nat bound as a non-empty string.
dev=True is the DEV-calibration mode itself — the run that PRODUCES that
decision — so it needs no decision and is forced to jsd_only (the DEV
sweep has no selection consequence, so negotiation rows would be waste).

Everything resumes: finished JSD/ppl rows are skipped through eval's own
_competence_done identity guard (imported, not duplicated — same package),
finished negotiation rows through run_negotiation_eval's resume machinery,
and a fully finished layer is skipped before the model is touched.

torch and models.py are imported lazily inside functions, like eval.py, so
this module imports on a stdlib-only interpreter (rung-1 discipline).
"""

import json
from pathlib import Path

from algoverse.eval import (
    _competence_done,
    load_wikitext_slice,
    neutral_distribution_pass,
    run_negotiation_eval,
)
from algoverse.metrics import load_rows
from algoverse.tasks import CONDITIONS, get_scenarios

# config.compared for the P-S1 rows: which two model states the JSD (and
# the intact_* config values) relate. One constant so sweep_report-side
# consumers can match it exactly.
COMPARED = "intact-vs-probe-bypassed"

# The sweep_manifest.json identity fields, guarded on every rerun. Mirrors
# train.py's _guard_train_manifest: named refusal, never silent adaptation.
SWEEP_MANIFEST_FIELDS = (
    "model_id", "adapter_path", "checkpoint_step", "train_seed",
    "quant_label", "layers", "n", "scenario_seed", "item16_decision",
    "dev", "run_tag",
)


def full_layer_list(model):
    """Every decoder layer index of the loaded model, 0..n_layers-1.

    The ratified Stage-1 sweep set is ALL layers including 0 and n-1, so
    this list is what the CLI passes as the sweep's manifest identity.
    """
    from algoverse.models import _decoder_layers

    return list(range(len(_decoder_layers(model))))


def layer_run_id(run_tag, layer):
    """The per-layer run_id/directory name: <run_tag>-lNN (D1)."""
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
            if row.get("run_id") != run_id or row.get("metric") != "wikitext2_ppl":
                continue
            config = row.get("config") or {}
            if config.get("intact_nll_mean") is not None:
                return config["intact_nll_mean"], config["intact_ppl"]
    return None


def run_layer_sweep(model, tokenizer, layers, out_root, run_tag, model_id,
                    adapter_path=None, checkpoint_step=None, train_seed=None,
                    quant_label=None, n=100, scenario_seed=42, seed=42,
                    batch_size=4, use_llm_fallback=False,
                    llm_provider="openai",
                    llm_model="gpt-4o-mini-2024-07-18",
                    item16_decision=None, dev=False, jsd_only=False,
                    n_tokens=20000, wikitext_ids=None, chunk=None,
                    max_length=1024, stride=512, max_new_tokens=256):
    """Sweep the requested layers of an already-loaded model (D2 + D5).

    model/tokenizer   loaded ONCE by the caller (canonical profile, adapter
                      applied); the model must be intact on entry.
    layers            the sweep's FULL layer list — manifest identity, the
                      same on every session of this sweep.
    chunk             the subset this session executes (None = all). Must
                      be drawn from ``layers``; selecting a chunk never
                      changes the manifest.
    out_root/run_tag  the D1 layout above; run_tag names the swept
                      checkpoint (e.g. "md-qwen7b-s42-step281").
    item16_decision   the recorded item-16 calibration decision; required
                      as a non-empty string whenever dev=False.
    dev               DEV-calibration mode: no decision needed, jsd_only
                      forced (D5).
    jsd_only          skip negotiation rows (dev calibration, or a
                      JSD-first survey session).
    wikitext_ids      test-only override of the pinned WikiText-2 slice,
                      like neutral_distribution_pass's token_ids.
    max_length/stride the pinned window scheme (item 12/16); parameters so
                      tiny-model tests can shrink them, defaults ratified.
    max_new_tokens    generation budget per response, canonical 256.
    Remaining arguments mirror scripts/run_baseline.py and are passed to
    run_negotiation_eval unchanged.

    Returns a summary dict: executed layers, layers skipped as complete,
    and the paths written.
    """
    from algoverse.models import BYPASS_IMPL, bypass_state, install_bypass
    from algoverse.utils import append_jsonl

    # Item-16 tripwire FIRST (D5): before the manifest, before any layer,
    # before anything is written.
    if dev:
        # The DEV calibration produces the item-16 decision; it cannot
        # require it, and its negotiation rows would have no selection
        # consequence — jsd_only is forced, not optional.
        jsd_only = True
    elif not (isinstance(item16_decision, str) and item16_decision.strip()):
        raise ValueError(
            "item16_decision: a research-model sweep (dev=False) runs no "
            "layer until the human's recorded DEV-calibration "
            "confirm-or-revise decision on the 0.25-nat item-16 bound is "
            "passed as a non-empty string (RESEARCH_SPEC item 16; "
            "sweep-driver plan D5). Run --dev-calibration first."
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
        "quant_label": quant_label,
        "layers": list(layers),
        "n": n,
        "scenario_seed": scenario_seed,
        "item16_decision": item16_decision,
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
    # same token ids to every layer's pass (D3's recommended shape).
    if wikitext_ids is not None:
        ids = wikitext_ids
    else:
        ids = load_wikitext_slice(tokenizer, n_tokens)
    slice_tokens = int(ids.shape[1])

    base_meta = {
        "run_id": "%s-base" % run_tag,
        "model_id": model_id,
        "adapter_path": current_manifest["adapter_path"],
        "bypassed_layer": None,
        "checkpoint_step": checkpoint_step,
        "arm": None,
        "train_seed": train_seed,
        "bypass_impl": None,
    }
    base_config = {
        "n_tokens": slice_tokens, "max_length": max_length, "stride": stride,
    }
    base_path = out_root / "base-competence.jsonl"
    base_done = _competence_done(
        base_path, base_meta, "wikitext2_ppl", base_config
    )

    def _write_base(nll_mean_intact, ppl_intact):
        base_row = dict(base_meta)
        base_row.update({
            "metric": "wikitext2_ppl",
            "value": ppl_intact,
            "stderr": None,
            "nll_mean": nll_mean_intact,
            "config": dict(base_config),
        })
        append_jsonl(base_path, base_row)
        print(
            "%s: intact wikitext2_ppl = %.3f recorded"
            % (base_meta["run_id"], ppl_intact)
        )

    scenarios = None
    if not jsd_only:
        # Selection split only, the ratified sweep draw — exactly what
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
            "arm": None,
            "train_seed": train_seed,
            "bypass_impl": BYPASS_IMPL,
        }
        # Identity known BEFORE the pass runs; the recorded config
        # additionally carries the intact-side results (intact_nll_mean,
        # intact_ppl), which _competence_done ignores on resume because it
        # compares only the fields the current request asserts.
        resume_config = {
            "n_tokens": slice_tokens,
            "max_length": max_length,
            "stride": stride,
            "probe_layer": layer,
            "compared": COMPARED,
        }
        jsd_done = _competence_done(
            comp_path, run_meta, "wikitext2_neutral_jsd", resume_config
        )
        ppl_done = _competence_done(
            comp_path, run_meta, "wikitext2_ppl", resume_config
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
            config = dict(resume_config)
            config["n_tokens"] = result["n_tokens"]
            config["intact_nll_mean"] = result["nll_mean_intact"]
            config["intact_ppl"] = result["ppl_intact"]
            if not jsd_done:
                row = dict(run_meta)
                row.update({
                    "metric": "wikitext2_neutral_jsd",
                    "value": result["jsd_mean_nats"],
                    "stderr": None,
                    "config": config,
                })
                append_jsonl(comp_path, row)
            if not ppl_done:
                row = dict(run_meta)
                row.update({
                    "metric": "wikitext2_ppl",
                    "value": result["ppl_bypassed"],
                    "stderr": None,
                    "nll_mean": result["nll_mean_bypassed"],
                    "config": config,
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
            handle = install_bypass(model, layer, role="probe")
            try:
                run_negotiation_eval(
                    model, tokenizer, scenarios,
                    run_id=run_id, out_path=rows_path,
                    model_id=model_id, adapter_path=adapter_path,
                    bypassed_layer=layer,
                    checkpoint_step=checkpoint_step, arm=None,
                    batch_size=batch_size, max_new_tokens=max_new_tokens,
                    seed=seed, train_seed=train_seed,
                    quant_label=quant_label,
                    use_llm_fallback=use_llm_fallback,
                    llm_provider=llm_provider, llm_model=llm_model,
                    scenario_seed=scenario_seed, n=n,
                )
            finally:
                handle.remove()
        assert bypass_state(model) is None, (
            "model is not intact after sweeping layer %d" % layer
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
