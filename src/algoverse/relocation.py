"""Pure Stage-3 relocation analysis over two completed layer sweeps."""

from algoverse import metrics, sweep


DISPERSION_VALUES = ("dispersed", "concentrated")
RELOCATION_VALUES = ("entirely-relocated", "partially-relocated", "not-applicable")
ORIGIN_VALUES = ("reconstructed", "strengthened")


def _permanent_identity(rows, label):
    values = {
        (row.get("gen_config") or {}).get("permanent_bypassed_layer")
        for row in rows
    }
    if len(values) != 1:
        raise ValueError("%s mixes permanent lesion identities" % label)
    return next(iter(values))


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

    layers = sorted(set(rec_layers) | set(les_layers) | ({permanent} if permanent is not None else set()))
    points = []
    for layer in layers:
        if layer == permanent and layer not in rec_layers and layer not in les_layers:
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
        if layer not in rec_layers or layer not in les_layers:
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
            rec_base, rec_layers[layer], les_base, les_layers[layer],
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
    measurable_delta = [point for point in points if point["delta_l"] is not None]
    max_change_layers = []
    if measurable_delta:
        # Human-ratified 2026-08-17: "greatest change" means maximum signed
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
        "permanent_bypassed_layer": permanent,
        "n_boot": n_boot,
    }


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
