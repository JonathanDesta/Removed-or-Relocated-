"""
Probe label sources and interp.jsonl row helpers for the probe-transfer
path (scripts/run_probe_transfer.py; scripts/probe_surface_baselines.py
reuses the label source).

interp.jsonl rows are run_meta + {analysis, layer, value, ci_low, ci_high,
config}, one row per (analysis, layer), append-only, resume-guarded with
the same identity discipline as competence.jsonl (mirrors
eval._competence_done). Probes are a corroborating analysis; none of
their outputs feed layer selection.

The probe recipe (PROBE_RECIPE, stamped into every probe_auroc row's
config): 0.3 group split fraction, random_state 0, max_iter 1000, C=0.1
scaler+LR Pipeline, and the response-token aggregation of
goldowskydill2025detecting (arXiv 2502.03407): the probe is fit on
individual response-token activations ("flatten across samples and
sequence positions") and each response's score is the mean of its
per-token scores (aggregate_response_scores). value = held-out AUROC on
scenario-grouped splits; a top-level `accuracy` result field rides
alongside (see write_interp_row).

Probe label sources:
1. Within-incentive-condition control: valid incentive-condition rows
   from a results rows.jsonl, labeled deceptive True (lied) vs False (did
   not lie), grouped by scenario_id, with prompts re-rendered canonically
   (labeled_incentive_rows + probe_examples_from_rows).
2. An explicit probe-dataset JSONL, one {"text", "label", "group"} object
   per line (load_probe_dataset); an optional per-row response_start
   token index narrows the scored span, else the whole text is scored.
   The Instructed-Pairs dataset (Azaria & Mitchell via
   zou2023representation; built by scripts/build_instructed_pairs.py)
   flows in through this source.

On a bypassed model the bypassed layer's row is written as a structural
null: value/ci null with config.excluded_bypassed_layer = true.

Module-level imports are stdlib-only; numpy/torch and the heavier
algoverse modules load lazily inside functions (the metrics.py
discipline), so the pure row/label/aggregation logic runs in the
dependency-free tier.
"""

import json
import os
from pathlib import Path

# Pre-registered probe recipe: recorded conventions, stamped into every
# probe_auroc row's config.
PROBE_RECIPE = {
    "test_size": 0.3,
    "random_state": 0,
    "max_iter": 1000,
    "C": 0.1,
    "pipeline": "standard_scaler+logistic_regression",
    "aggregation": "mean_response_token_score",
    "aggregation_source": "goldowskydill2025detecting",
}


def aggregate_response_scores(token_scores):
    """Per-response score = MEAN of that response's per-token probe scores.

    This is goldowskydill2025detecting's aggregation, adopted outright
    (pre-registered): "we project onto the probe direction to get a
    real-valued deception score at every token. We then take the mean
    across tokens to get a single deceptiveness-score for each model
    response." scripts/run_probe_transfer.py calls this on held-out
    decision scores.

    token_scores: iterable over responses, each an iterable of per-token
    scores. Returns a list of floats, one per response. An empty response
    raises: a response with no scored tokens is a caller bug, and silently
    dropping or zero-filling it would corrupt the AUROC alignment.
    """
    aggregated = []
    for index, scores in enumerate(token_scores):
        scores = [float(score) for score in scores]
        if not scores:
            raise ValueError(
                "response %d has no token scores to aggregate" % index
            )
        aggregated.append(sum(scores) / len(scores))
    return aggregated


# ---------------------------------------------------------------------------
# Probe label sources
# ---------------------------------------------------------------------------

def labeled_incentive_rows(rows):
    """The within-incentive-condition control labels.

    Among VALID incentive-condition rows, deceptive True = lied and
    deceptive False = did not lie. Invalid rows are excluded structurally:
    the INTERFACES row rules guarantee invalid rows carry deceptive null,
    never False, and both `valid is not True` and `deceptive not in
    (True, False)` are checked so a malformed row cannot slip in either
    way. Groups are scenario_id (scenario-grouped splits, the
    pre-registered metric). Returns [{"scenario_id", "response_text",
    "label", "group"}].
    """
    from algoverse.tasks import INCENTIVE

    labeled = []
    for row in rows:
        if row.get("condition") != INCENTIVE:
            continue
        if row.get("valid") is not True:
            continue
        deceptive = row.get("deceptive")
        # Identity checks: `1 in (True, False)` is True in Python, and a
        # non-bool deceptive value is a malformed row, not a label.
        if deceptive is not True and deceptive is not False:
            continue
        response_text = row.get("response_text")
        if not response_text:
            continue
        labeled.append({
            "scenario_id": row.get("scenario_id"),
            "response_text": response_text,
            "label": bool(row["deceptive"]),
            "group": row.get("scenario_id"),
            # Provenance for stratified scoring:
            # which run wrote the response, and whether the scenario had
            # a real outside offer (None = a no-offer scenario).
            "source_run_id": row.get("run_id"),
            "true_value": row.get("true_value"),
        })
    return labeled


def strip_final_claim_line(response_text):
    """(body, had_marker): the response BEFORE its last structured claim line.

    The complement of tasks._authoritative_answer_text: the scorer keeps the
    text from the LAST "MY BEST OUTSIDE OFFER:" marker onward; this drops
    it (rstripped), so a probe can read the prose without the tokens that
    literally carry the label. No marker -> (text, False); the caller
    decides what an unmarked response means.
    """
    from algoverse.tasks import ANSWER_MARKER_RE

    text = response_text or ""
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return text, False
    return text[:matches[-1].start()].rstrip(), True


def probe_examples_from_rows(rows, tokenizer, exclude_final_line=False,
                             stats=None):
    """Capture-ready probe examples from results rows (label source 1).

    Re-renders each row's scenario in the incentive condition through
    eval.render_condition_texts (the module rendering contract, generation
    prompt included) and appends the row's stored response_text.
    response_start is the prompt's token count under
    the same add_special_tokens=False tokenization the capture helper
    uses, VERIFIED by prefix-token equality so a tokenizer boundary merge
    cannot silently shift the span.

    Reconstruction caveat: response_text was decoded with
    skip_special_tokens and stripped, so the re-encoded response is the
    scored text, not a byte-exact replay of the generated token ids.

    exclude_final_line: score the response WITHOUT its final structured
    claim line (strip_final_claim_line). A response with no marker is kept
    unchanged and counted; a response whose whole body IS the claim line
    is skipped and counted (nothing is left to read). `stats`, when a dict,
    receives n_labeled, n_examples, n_no_marker, n_skipped_empty_body
    (zeros in default mode), so a run can record what it dropped.

    Returns [{"text", "response_start", "label", "group", "scenario_id",
    "source_run_id", "has_offer"}] — the last three carry provenance for
    stratified scoring (pooled test sets may hold one response per
    scenario from EACH of two runs; group stays scenario_id so a
    scenario-level bootstrap resamples such pairs together).
    """
    from algoverse.eval import render_condition_texts
    from algoverse.tasks import INCENTIVE, make_scenario_grid

    labeled = labeled_incentive_rows(rows)
    if not labeled:
        raise ValueError(
            "no valid incentive-condition rows with a deceptive label; "
            "cannot build within-incentive probe examples"
        )
    grid = {
        scenario["scenario_id"]: scenario
        for scenario in make_scenario_grid()
    }
    missing = sorted({
        item["scenario_id"] for item in labeled
        if item["scenario_id"] not in grid
    })
    if missing:
        raise ValueError(
            "rows reference scenario_ids not in the scenario grid "
            "(first few: %s)" % missing[:5]
        )
    scenarios = [grid[item["scenario_id"]] for item in labeled]
    prompts = render_condition_texts(scenarios, INCENTIVE, tokenizer)

    def token_ids(text):
        return tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()

    counters = {"n_labeled": len(labeled), "n_examples": 0,
                "n_no_marker": 0, "n_skipped_empty_body": 0}
    examples = []
    for item, prompt in zip(labeled, prompts):
        response = item["response_text"]
        if exclude_final_line:
            body, had_marker = strip_final_claim_line(response)
            if not had_marker:
                counters["n_no_marker"] += 1
            elif not body:
                counters["n_skipped_empty_body"] += 1
                continue
            else:
                response = body
        full = prompt + response
        prompt_ids = token_ids(prompt)
        full_ids = token_ids(full)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "tokenization boundary shifted between prompt and "
                "prompt+response for scenario %r; the response span cannot "
                "be trusted" % item["scenario_id"]
            )
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(
                "response for scenario %r tokenized to zero tokens"
                % item["scenario_id"]
            )
        examples.append({
            "text": full,
            "response_start": len(prompt_ids),
            "label": item["label"],
            "group": item["group"],
            "scenario_id": item["scenario_id"],
            "source_run_id": item.get("source_run_id"),
            "has_offer": grid[item["scenario_id"]].get("true_outside_offer") is not None,
        })
    counters["n_examples"] = len(examples)
    if stats is not None:
        stats.update(counters)
    if not examples:
        raise ValueError(
            "every labeled row was skipped (%d rows whose whole response is "
            "the claim line); nothing to probe" % counters["n_skipped_empty_body"]
        )
    return examples


def load_probe_dataset(path):
    """Probe label source 2: a JSONL of {"text", "label", "group"} objects.

    An optional per-row `response_start` (a non-negative token index under
    the capture helper's add_special_tokens=False tokenization) marks
    where the scored span begins — the Instructed-Pairs builder
    (scripts/build_instructed_pairs.py) records the statement span this
    way. Absent (or 0), the whole text is the scored span, the previous
    behavior, appropriate for statement-style data with no
    prompt/response split. Labels must be JSON true/false; anything else
    raises, because a truthiness coercion here would silently relabel
    data. Strict parsing: a malformed line is an error, not a skip — this
    is training data, not a torn results log.
    """
    path = Path(path)
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s line %d is not valid JSON: %s" % (path, lineno, exc)
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    "%s line %d is not a JSON object" % (path, lineno)
                )
            missing = [
                field for field in ("text", "label", "group")
                if field not in record
            ]
            if missing:
                raise ValueError(
                    "%s line %d is missing %s" % (path, lineno, missing)
                )
            if not isinstance(record["text"], str) or not record["text"]:
                raise ValueError(
                    "%s line %d has an empty or non-string text"
                    % (path, lineno)
                )
            # Identity check, not `in`: 1 == True in Python, and coercing
            # a non-bool would silently relabel data.
            if record["label"] is not True and record["label"] is not False:
                raise ValueError(
                    "%s line %d label must be JSON true/false, got %r"
                    % (path, lineno, record["label"])
                )
            response_start = record.get("response_start", 0)
            # bool is an int subclass; a JSON true here is a malformed
            # span, not token index 1.
            if (isinstance(response_start, bool)
                    or not isinstance(response_start, int)
                    or response_start < 0):
                raise ValueError(
                    "%s line %d response_start must be a non-negative "
                    "integer, got %r" % (path, lineno, response_start)
                )
            examples.append({
                "text": record["text"],
                "response_start": response_start,
                "label": record["label"],
                "group": record["group"],
            })
    if not examples:
        raise ValueError("probe dataset %s contains no examples" % path)
    return examples


# ---------------------------------------------------------------------------
# interp.jsonl rows: append, identity, resume
# ---------------------------------------------------------------------------

# Fields that carry results rather than run identity. `accuracy` is a
# top-level result field on probe_auroc rows only, recorded like
# wikitext2_ppl's nll_mean result field. Part of the interp.jsonl
# contract.
_RESULT_FIELDS = {
    "analysis", "layer", "value", "ci_low", "ci_high", "accuracy", "config",
}


def _append_jsonl(path, record):
    """Append one JSON line; heal a torn final line first.

    Mirrors utils.append_jsonl (torn-fragment isolation, flush + fsync)
    without importing utils, so this module's imports stay stdlib-only
    and the dependency-free tier can run it (the same reason
    metrics.load_rows has its own JSONL reader).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record) + "\n").encode("utf-8")
    with open(path, "a+b") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())


def _finite_or_none(value):
    """None for None/NaN; float otherwise. JSON has no NaN; null is how
    an excluded/unavailable value is recorded."""
    if value is None:
        return None
    value = float(value)
    return None if value != value else value


def write_interp_row(out_path, run_meta, analysis, layer, value,
                     ci_low, ci_high, config, extra=None):
    """Append one interp.jsonl row: run_meta + the INTERFACES result fields.

    NaN in value/ci becomes null (JSON has no NaN). `extra` adds result
    fields beyond the schema (currently only probe_auroc's `accuracy`).
    """
    row = dict(run_meta or {})
    row.update({
        "analysis": analysis,
        "layer": int(layer),
        "value": _finite_or_none(value),
        "ci_low": _finite_or_none(ci_low),
        "ci_high": _finite_or_none(ci_high),
        "config": dict(config or {}),
    })
    for field, extra_value in dict(extra or {}).items():
        if field in row:
            raise ValueError(
                "extra field %r collides with an existing row field" % field
            )
        row[field] = extra_value
    _append_jsonl(out_path, row)
    return row


def _interp_done(out_path, run_meta, analysis, layer, config):
    """Guard interp run/config identity; True iff (analysis, layer) is done.

    Same discipline as eval._competence_done: every non-result,
    non-version field is run identity and must match run_meta exactly;
    within the matching (analysis, layer) rows, every field of the CURRENT
    config must match the recorded config (operational batch_size
    excluded). A mismatch raises rather than silently forking a run.
    """
    from algoverse.metrics import load_rows

    out_path = Path(out_path)
    if not out_path.exists():
        return False
    run_meta = dict(run_meta or {})
    run_id = run_meta.get("run_id")
    existing = [
        row for row in load_rows(out_path) if row.get("run_id") == run_id
    ]
    identity_fields = set(run_meta)
    for row in existing:
        identity_fields.update(
            field for field in row if field not in _RESULT_FIELDS
        )
    identity_fields = {
        field for field in identity_fields if not field.endswith("_version")
    }
    mismatches = set()
    for row in existing:
        for field in identity_fields:
            if row.get(field) != run_meta.get(field):
                mismatches.add(field)
    if mismatches:
        raise ValueError(
            "interp run identity mismatch for run_id %r: %s"
            % (run_id, ", ".join(sorted(mismatches)))
        )
    matched = [
        row for row in existing
        if row.get("analysis") == analysis and row.get("layer") == layer
    ]
    config_mismatches = set()
    for row in matched:
        recorded = row.get("config") or {}
        for field, current in dict(config or {}).items():
            if field == "batch_size":
                continue  # operational, not identity (competence rule)
            if recorded.get(field) != current:
                config_mismatches.add("config.%s" % field)
    if config_mismatches:
        raise ValueError(
            "interp config mismatch for run_id %r, analysis %r, layer %r: %s"
            % (run_id, analysis, layer, ", ".join(sorted(config_mismatches)))
        )
    return bool(matched)
