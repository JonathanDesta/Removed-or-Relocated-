"""Pure Stage-3 relocation analysis over two completed layer sweeps."""

import json
from pathlib import Path

from algoverse import metrics, sweep


DISPERSION_VALUES = ("dispersed", "concentrated")
RELOCATION_VALUES = ("entirely-relocated", "partially-relocated", "not-applicable")
EDIT_RELOCATION_VALUES = (
    "recovered-in-place", "relocated", "mixed", "not-applicable",
)
ORIGIN_VALUES = ("reconstructed", "strengthened")


def _permanent_identity(rows, label):
    values = {
        (row.get("gen_config") or {}).get("permanent_bypassed_layer")
        for row in rows
    }
    if len(values) != 1:
        raise ValueError("%s mixes permanent lesion identities" % label)
    return next(iter(values))


def _evaluate_points(rec_base, rec_layers, comparison_base, comparison_layers,
                     structural_null_layer=None, n_boot=2000, seed=0):
    """Shared δ-curve loop; structural_null_layer is lesion-path-only."""
    structural = (
        {structural_null_layer}
        if structural_null_layer is not None else set()
    )
    layers = sorted(set(rec_layers) | set(comparison_layers) | structural)
    points = []
    for layer in layers:
        if (
            layer == structural_null_layer
            and layer not in rec_layers and layer not in comparison_layers
        ):
            points.append({
                "layer": layer,
                "A_recovered": None,
                "A_lesioned": None,
                "delta_l": None,
                "delta_ci_low": None,
                "delta_ci_high": None,
                "n_scenarios_common": 0,
                "paired": False,
                "reason": "permanently_lesioned_structural_null",
            })
            continue
        if layer not in rec_layers or layer not in comparison_layers:
            missing = "recovered" if layer not in rec_layers else "just_lesioned"
            points.append({
                "layer": layer,
                "A_recovered": None,
                "A_lesioned": None,
                "delta_l": None,
                "delta_ci_low": None,
                "delta_ci_high": None,
                "n_scenarios_common": 0,
                "paired": False,
                "reason": "missing_%s_layer_run" % missing,
            })
            continue
        point = metrics.relocation_delta(
            rec_base, rec_layers[layer],
            comparison_base, comparison_layers[layer],
            n_boot=n_boot, seed=seed,
        )
        point["layer"] = layer
        points.append(point)

    measurable_recovered = [
        point for point in points if point["A_recovered"] is not None
    ]
    k = None
    k_layers = []
    if measurable_recovered:
        maximum = max(point["A_recovered"] for point in measurable_recovered)
        k_layers = [
            point["layer"] for point in measurable_recovered
            if point["A_recovered"] == maximum
        ]
        k = min(k_layers)
    measurable_delta = [
        point for point in points if point["delta_l"] is not None
    ]
    max_change_layers = []
    if measurable_delta:
        # Human-ratified 2026-08-17: greatest change is maximum signed
        # delta_l, not maximum absolute magnitude. Preserve every exact tie.
        maximum = max(point["delta_l"] for point in measurable_delta)
        max_change_layers = [
            point["layer"] for point in measurable_delta
            if point["delta_l"] == maximum
        ]
    candidates = sorted(set(max_change_layers) | set(k_layers))
    return {
        "points": points,
        "k": k,
        "k_layers": k_layers,
        "max_change_layers": max_change_layers,
        "candidate_layers": candidates,
    }


def evaluate_relocation(recovered_base, recovered_layers, lesioned_base,
                        lesioned_layers, lesioned_layer=None,
                        n_boot=2000, seed=0):
    """Compute every δ_l, k, max-signed-delta layers, and review candidates."""
    rec_base, rec_layers = sweep.load_sweep_inputs(
        recovered_base, recovered_layers
    )
    les_base, les_layers = sweep.load_sweep_inputs(
        lesioned_base, lesioned_layers
    )
    rec_permanent = _permanent_identity(rec_base, "recovered sweep")
    les_permanent = _permanent_identity(les_base, "just-lesioned sweep")
    if rec_permanent != les_permanent:
        raise ValueError(
            "recovered and just-lesioned sweeps have different permanent "
            "lesions: %r vs %r" % (rec_permanent, les_permanent)
        )
    if rec_permanent is None:
        raise ValueError("Stage-3 relocation sweeps require a permanent lesion")
    if lesioned_layer is not None and int(lesioned_layer) != rec_permanent:
        raise ValueError(
            "lesioned_layer %r contradicts sweep identity %r"
            % (lesioned_layer, rec_permanent)
        )
    permanent = rec_permanent

    result = _evaluate_points(
        rec_base, rec_layers, les_base, les_layers,
        structural_null_layer=permanent, n_boot=n_boot, seed=seed,
    )
    result.update({
        "permanent_bypassed_layer": permanent,
        "n_boot": n_boot,
    })
    return result


def _json_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError("%s is not a file: %s" % (label, path))
    return path, json.loads(path.read_text(encoding="utf-8"))


def _is_within(path, ancestor):
    try:
        Path(path).relative_to(ancestor)
    except ValueError:
        return False
    return True


def _checkpoints_relative(path):
    """The path from its first checkpoints/ segment on, or None.

    Run artifacts record absolute paths under the mount prefix of whichever
    platform produced them (/root/... on Colab and Kaggle, /home/<user>/...
    on the JupyterHub box). The project-relative form is the stable identity
    across platforms; everything before checkpoints/ is machine bookkeeping.
    """
    parts = Path(path).parts
    if "checkpoints" not in parts:
        return None
    return Path(*parts[parts.index("checkpoints"):])


def _edit_lineage(edit_manifest_path, init_provenance_path, edit_layers):
    manifest_path, manifest = _json_file(
        edit_manifest_path, "edit_manifest_path"
    )
    _provenance_path, provenance = _json_file(
        init_provenance_path, "init_provenance_path"
    )
    edit_layers = tuple(edit_layers)
    if (
        not edit_layers
        or any(isinstance(layer, bool) or not isinstance(layer, int)
               for layer in edit_layers)
        or any(layer < 0 for layer in edit_layers)
        or len(set(edit_layers)) != len(edit_layers)
    ):
        raise ValueError(
            "edit_layers must be unique non-negative integers, got %r"
            % (edit_layers,)
        )
    edit_layers = tuple(sorted(edit_layers))
    recorded = (manifest.get("config") or {}).get("train_layers")
    if recorded is None:
        raise ValueError(
            "edit manifest config.train_layers is null; this is not a "
            "layer-local edit run"
        )
    if tuple(recorded) != edit_layers:
        raise ValueError(
            "edit manifest config.train_layers %r does not match edit_layers %r"
            % (recorded, edit_layers)
        )
    init_adapter = provenance.get("init_adapter")
    if not init_adapter:
        raise ValueError("init_provenance.json is missing init_adapter")
    edit_out_dir = manifest_path.resolve().parent
    resolved_init = Path(init_adapter).resolve()
    if not _is_within(resolved_init, edit_out_dir):
        # Not contained as absolute paths. The provenance may simply carry
        # another platform's mount prefix for the SAME project artifact, so
        # containment is re-judged on the project-relative forms before
        # refusing. A genuinely foreign init (another run's checkpoints)
        # still differs project-relatively and is still refused.
        init_rel = _checkpoints_relative(resolved_init)
        out_rel = _checkpoints_relative(edit_out_dir)
        if init_rel is None or out_rel is None or not _is_within(init_rel, out_rel):
            raise ValueError(
                "init_provenance init_adapter %s is outside edit run out_dir %s"
                % (resolved_init, edit_out_dir)
            )
    return edit_layers


def _require_lesion_free(rows, label):
    permanent = _permanent_identity(rows, label)
    if permanent is not None:
        raise ValueError(
            "%s must be lesion-free for edit relocation; "
            "gen_config.permanent_bypassed_layer is %r"
            % (label, permanent)
        )


def evaluate_edit_relocation(recovered_base, recovered_layers, edited_base,
                             edited_layers, edit_manifest_path,
                             init_provenance_path, edit_layers,
                             n_boot=2000, seed=0):
    """Compute a lesion-free edit δ-curve with validated edit lineage."""
    rec_base, rec_layers = sweep.load_sweep_inputs(
        recovered_base, recovered_layers
    )
    edit_base, edit_layer_rows = sweep.load_sweep_inputs(
        edited_base, edited_layers
    )
    normalized_edit_layers = _edit_lineage(
        edit_manifest_path, init_provenance_path, edit_layers
    )
    _require_lesion_free(rec_base, "recovered edit sweep base")
    _require_lesion_free(edit_base, "just-edited sweep base")
    for layer, rows in rec_layers.items():
        _require_lesion_free(rows, "recovered edit sweep layer %s" % layer)
    for layer, rows in edit_layer_rows.items():
        _require_lesion_free(rows, "just-edited sweep layer %s" % layer)

    result = _evaluate_points(
        rec_base, rec_layers, edit_base, edit_layer_rows,
        n_boot=n_boot, seed=seed,
    )
    edit_set = set(normalized_edit_layers)

    def partition(values):
        return {
            "inside": [layer for layer in values if layer in edit_set],
            "outside": [layer for layer in values if layer not in edit_set],
        }

    partitions = {
        "k_layers": partition(result["k_layers"]),
        "max_change_layers": partition(result["max_change_layers"]),
        "candidate_layers": partition(result["candidate_layers"]),
    }
    if not result["k_layers"] or not result["max_change_layers"]:
        verdict = "not-applicable"
    elif not partitions["candidate_layers"]["outside"]:
        verdict = "recovered-in-place"
    elif not partitions["candidate_layers"]["inside"]:
        verdict = "relocated"
    else:
        verdict = "mixed"
    result.update({
        "edit_layers": list(normalized_edit_layers),
        "edit_partition": partitions,
        "edit_relocation": verdict,
        "permanent_bypassed_layer": None,
        "n_boot": n_boot,
    })
    return result


def _fmt(value):
    return "n/a" if value is None else "%.3f" % value


def _coverage(point):
    values = [
        point.get("n_scenarios_common"),
        point.get("n_scenarios_recovered_base"),
        point.get("n_scenarios_recovered_bypassed"),
        point.get("n_scenarios_lesioned_base"),
        point.get("n_scenarios_lesioned_bypassed"),
    ]
    return "/".join("n/a" if value is None else str(value) for value in values)


def relocation_report(result, final=False, verdict_ref=None,
                      dispersion=None, relocation=None, origins=None):
    """Render measurements, optionally stamping a complete human verdict."""
    origins = {int(layer): value for layer, value in dict(origins or {}).items()}
    candidates = set(result["candidate_layers"])
    if final:
        if not (isinstance(verdict_ref, str) and verdict_ref.strip()):
            raise ValueError("final relocation report requires verdict_ref")
        if dispersion not in DISPERSION_VALUES:
            raise ValueError("dispersion must be one of %r" % (DISPERSION_VALUES,))
        if relocation not in RELOCATION_VALUES:
            raise ValueError("relocation must be one of %r" % (RELOCATION_VALUES,))
        missing = sorted(candidates - set(origins))
        extra = sorted(set(origins) - candidates)
        invalid = sorted(
            layer for layer, value in origins.items()
            if value not in ORIGIN_VALUES
        )
        if missing or extra or invalid:
            raise ValueError(
                "origin classifications must cover exactly candidate layers; "
                "missing=%r extra=%r invalid=%r" % (missing, extra, invalid)
            )

    lines = [
        "STAGE-3 RELOCATION REPORT  (paired bootstrap n=%d)" % result["n_boot"],
        "permanent lesion l*: %s" % result["permanent_bypassed_layer"],
        "",
        "| layer | A_l recovered | A_l just-lesioned | delta_l [95% CI] "
        "| coverage shared/rb/rp/lb/lp | status |",
        "|---|---|---|---|---|---|",
    ]
    for point in result["points"]:
        lines.append(
            "| %s | %s | %s | %s [%s, %s] | %s | %s |"
            % (
                point["layer"], _fmt(point["A_recovered"]),
                _fmt(point["A_lesioned"]), _fmt(point["delta_l"]),
                _fmt(point["delta_ci_low"]), _fmt(point["delta_ci_high"]),
                _coverage(point), point.get("reason") or "measured",
            )
        )
    gaps = [point for point in result["points"] if point.get("reason")]
    lines.extend([
        "",
        "gaps: %s" % (
            "none" if not gaps else "; ".join(
                "layer %s=%s" % (point["layer"], point["reason"])
                for point in gaps
            )
        ),
        "k (deterministic representative of max recovered A_l): %s"
        % result["k"],
        "max-recovered layer(s): %s" % result["k_layers"],
        "max-change layer(s) (maximum signed delta_l): %s"
        % result["max_change_layers"],
        "origin-review candidate layer(s): %s" % result["candidate_layers"],
    ])
    if final:
        lines.extend([
            "",
            "HUMAN VERDICT REFERENCE: %s" % verdict_ref.strip(),
            "dispersion: %s" % dispersion,
            "relocation: %s" % relocation,
        ])
        for layer in result["candidate_layers"]:
            lines.append("layer %d origin: %s" % (layer, origins[layer]))
    else:
        lines.extend([
            "",
            "VERDICT: PENDING HUMAN CLASSIFICATION (measurements only; no "
            "numeric uniform/near-zero threshold invented)",
        ])
    report = "\n".join(lines)
    print(report)
    return report


def edit_relocation_report(result, final=False, verdict_ref=None,
                           dispersion=None, origins=None):
    """Render edit-aware measurements and the precommitted spatial verdict."""
    origins = {int(layer): value for layer, value in dict(origins or {}).items()}
    candidates = set(result["candidate_layers"])
    verdict = result.get("edit_relocation")
    if verdict not in EDIT_RELOCATION_VALUES:
        raise ValueError(
            "edit_relocation must be one of %r" % (EDIT_RELOCATION_VALUES,)
        )
    if final:
        if not (isinstance(verdict_ref, str) and verdict_ref.strip()):
            raise ValueError("final edit relocation report requires verdict_ref")
        if dispersion not in DISPERSION_VALUES:
            raise ValueError("dispersion must be one of %r" % (DISPERSION_VALUES,))
        missing = sorted(candidates - set(origins))
        extra = sorted(set(origins) - candidates)
        invalid = sorted(
            layer for layer, value in origins.items()
            if value not in ORIGIN_VALUES
        )
        if missing or extra or invalid:
            raise ValueError(
                "origin classifications must cover exactly candidate layers; "
                "missing=%r extra=%r invalid=%r" % (missing, extra, invalid)
            )

    lines = [
        "STAGE-3 EDIT RELOCATION REPORT  (paired bootstrap n=%d)"
        % result["n_boot"],
        "edited layers: %s" % result["edit_layers"],
        "",
        "| layer | A_l recovered | A_l just-edited | delta_l [95% CI] "
        "| coverage shared/rb/rp/lb/lp | status |",
        "|---|---|---|---|---|---|",
    ]
    for point in result["points"]:
        lines.append(
            "| %s | %s | %s | %s [%s, %s] | %s | %s |"
            % (
                point["layer"], _fmt(point["A_recovered"]),
                _fmt(point["A_lesioned"]), _fmt(point["delta_l"]),
                _fmt(point["delta_ci_low"]), _fmt(point["delta_ci_high"]),
                _coverage(point), point.get("reason") or "measured",
            )
        )
    gaps = [point for point in result["points"] if point.get("reason")]
    partitions = result["edit_partition"]
    lines.extend([
        "",
        "gaps: %s" % (
            "none" if not gaps else "; ".join(
                "layer %s=%s" % (point["layer"], point["reason"])
                for point in gaps
            )
        ),
        "k (deterministic representative of max recovered A_l): %s"
        % result["k"],
        "max-recovered layer(s): %s (inside=%s outside=%s)"
        % (result["k_layers"], partitions["k_layers"]["inside"],
           partitions["k_layers"]["outside"]),
        "max-change layer(s) (maximum signed delta_l): %s "
        "(inside=%s outside=%s)"
        % (result["max_change_layers"],
           partitions["max_change_layers"]["inside"],
           partitions["max_change_layers"]["outside"]),
        "origin-review candidate layer(s): %s (inside=%s outside=%s)"
        % (result["candidate_layers"],
           partitions["candidate_layers"]["inside"],
           partitions["candidate_layers"]["outside"]),
        "edit relocation (precommitted rule): %s" % verdict,
    ])
    if final:
        lines.extend([
            "",
            "HUMAN VERDICT REFERENCE: %s" % verdict_ref.strip(),
            "dispersion: %s" % dispersion,
        ])
        for layer in result["candidate_layers"]:
            lines.append("layer %d origin: %s" % (layer, origins[layer]))
    else:
        lines.extend([
            "",
            "HUMAN CLASSIFICATIONS: PENDING (dispersion/origins; spatial "
            "edit verdict above is rule-derived)",
        ])
    report = "\n".join(lines)
    print(report)
    return report


def apply_truncated_invalid_ruling(src_path, dst_path):
    """Copy a rows.jsonl applying the ratified truncated->invalid ruling.

    Rows with hit_max_tokens are reclassified valid=False,
    invalid_reason="truncated", deceptive=None (invalid rows carry a null
    label per the scoring spec). All other rows pass through unchanged. The
    source file is never modified - results stay append-only; the ruling
    lives in a labeled copy. Returns (n_rows, n_reclassified).
    """
    src, dst = Path(src_path), Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_reclassified = 0
    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_rows += 1
        if row.get("hit_max_tokens") and row.get("valid"):
            row["valid"] = False
            row["invalid_reason"] = "truncated"
            row["deceptive"] = None
            n_reclassified += 1
        out_lines.append(json.dumps(row))
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return n_rows, n_reclassified
