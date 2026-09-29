"""Pure Stage-3 edit-relocation analysis over two completed layer sweeps.

The two sweeps are the recovered E,D checkpoint at t=281 and the just-edited
M_E it was continued from. Per layer l, δ_l = A_l(recovered) - A_l(edited);
the layers of maximum recovered effect and of maximum signed change are the
review candidates, and their position relative to the edit window gives the
pre-committed spatial verdict (recovered-in-place / relocated / mixed /
not-applicable). Stdlib only, like sweep.py and metrics.py.
"""

import json
from pathlib import Path

from algoverse import metrics, sweep


DISPERSION_VALUES = ("dispersed", "concentrated")
EDIT_RELOCATION_VALUES = (
    "recovered-in-place", "relocated", "mixed", "not-applicable",
)
ORIGIN_VALUES = ("reconstructed", "strengthened")


_RUN_SIDES = ("recovered_base", "recovered_bypassed",
              "edited_base", "edited_bypassed")


def _invalid_rates(rows):
    """Per-condition invalid rate of one run: invalid rows / all rows.

    Rows arrive through sweep.load_sweep_inputs, so the truncation rule has
    already been applied (metrics.rows_from). The denominator is every row
    of that condition, and a condition with no rows is None -- never 0.
    """
    rates = {}
    for condition in ("incentive", "control"):
        pool = [r for r in rows if r.get("condition") == condition]
        if not pool:
            rates[condition] = None
            continue
        bad = sum(1 for r in pool if not r.get("valid"))
        rates[condition] = bad / len(pool)
    return rates


def _void_conditions(rates, invalid_max):
    """Conditions whose invalid rate STRICTLY exceeds the bound."""
    return [
        condition for condition in ("incentive", "control")
        if rates.get(condition) is not None and rates[condition] > invalid_max
    ]


def _evaluate_points(rec_base, rec_layers, edit_base, edit_layers,
                     n_boot=2000, seed=0,
                     invalid_max=metrics.INVALID_RATE_MAX):
    """The δ-curve loop shared by evaluation and the tests.

    Voiding (the pre-registered validity rule, applied literally): any of
    the four runs feeding a layer's δ whose incentive OR control invalid
    rate strictly exceeds invalid_max voids its side. A voided side's A is
    None, δ is None when either side is void, the point carries the rates
    and the offending run:condition pairs, and -- because k / max-change /
    candidates select over non-None values only -- voided layers take no
    part in the verdict. "No measurement" never reads as "measured zero".
    """
    layers = sorted(set(rec_layers) | set(edit_layers))
    rb_rates = _invalid_rates(rec_base)
    eb_rates = _invalid_rates(edit_base)
    points = []
    for layer in layers:
        if layer not in rec_layers or layer not in edit_layers:
            missing = "recovered" if layer not in rec_layers else "edited"
            points.append({
                "layer": layer,
                "A_recovered": None,
                "A_edited": None,
                "delta_l": None,
                "delta_ci_low": None,
                "delta_ci_high": None,
                "n_scenarios_common": 0,
                "paired": False,
                "reason": "missing_%s_layer_run" % missing,
                "invalid_rates": None,
                "voided": [],
            })
            continue
        point = metrics.relocation_delta(
            rec_base, rec_layers[layer],
            edit_base, edit_layers[layer],
            n_boot=n_boot, seed=seed,
        )
        point["layer"] = layer
        rates = {
            "recovered_base": rb_rates,
            "recovered_bypassed": _invalid_rates(rec_layers[layer]),
            "edited_base": eb_rates,
            "edited_bypassed": _invalid_rates(edit_layers[layer]),
        }
        voided = [
            "%s:%s" % (run, condition)
            for run in _RUN_SIDES
            for condition in _void_conditions(rates[run], invalid_max)
        ]
        point["invalid_rates"] = rates
        point["voided"] = voided
        if any(v.startswith("recovered") for v in voided):
            point["A_recovered"] = None
        if any(v.startswith("edited") for v in voided):
            point["A_edited"] = None
        if voided:
            point["delta_l"] = None
            point["delta_ci_low"] = None
            point["delta_ci_high"] = None
            point["reason"] = "voided_validity"
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
        # Pre-registered: greatest change is the maximum SIGNED delta_l, not
        # the maximum absolute magnitude. Preserve every exact tie.
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
    machine produced them (/kaggle/working/... on Kaggle, /home/<user>/...
    on a workstation). The project-relative form is the stable identity
    across machines; everything before checkpoints/ is machine bookkeeping.
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
        # another machine's mount prefix for the SAME project artifact, so
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


def evaluate_edit_relocation(recovered_base, recovered_layers, edited_base,
                             edited_layers, edit_manifest_path,
                             init_provenance_path, edit_layers,
                             n_boot=2000, seed=0,
                             invalid_max=metrics.INVALID_RATE_MAX):
    """Compute the edit δ-curve, the candidates, and the spatial verdict.

    recovered_base/recovered_layers  the E,D-t281 sweep (unprobed rows and
                                     {layer: rows}); edited_base/edited_layers
                                     the just-edited M_E sweep, same shape.
    edit_manifest_path               the edit run's train_manifest.json;
    init_provenance_path             the E,D run's init_provenance.json.
                                     Together they prove the continuation
                                     really started from that edit, and
                                     that edit_layers is the window the
                                     edit actually trained.

    result["incomplete"] names, per side, the layers swept on the other
    side only. The verdict is still computed over the layers both sweeps
    measured (a one-sided layer takes no part in k, the change ranking or
    the candidates), and the report marks it [INCOMPLETE] so a missing
    sweep column can never read as a settled ruling.
    """
    rec_base, rec_layers = sweep.load_sweep_inputs(
        recovered_base, recovered_layers
    )
    edit_base, edit_layer_rows = sweep.load_sweep_inputs(
        edited_base, edited_layers
    )
    normalized_edit_layers = _edit_lineage(
        edit_manifest_path, init_provenance_path, edit_layers
    )

    result = _evaluate_points(
        rec_base, rec_layers, edit_base, edit_layer_rows,
        n_boot=n_boot, seed=seed, invalid_max=invalid_max,
    )
    edit_set = set(normalized_edit_layers)
    incomplete = {
        "recovered": [
            point["layer"] for point in result["points"]
            if point.get("reason") == "missing_recovered_layer_run"
        ],
        "edited": [
            point["layer"] for point in result["points"]
            if point.get("reason") == "missing_edited_layer_run"
        ],
    }

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
        "incomplete": incomplete,
        "n_boot": n_boot,
        "invalid_max": invalid_max,
    })
    return result


_fmt = metrics.fmt


def _voided_line(result):
    """The voided layers with the run:condition rates that voided them."""
    bound = result.get("invalid_max", metrics.INVALID_RATE_MAX)
    head = ("voided (per-condition invalid rate, truncated-or-invalid, > %.2f): "
            % bound)
    parts = []
    for point in result["points"]:
        if not point.get("voided"):
            continue
        rates = point.get("invalid_rates") or {}
        detail = ", ".join(
            "%s %s=%.2f" % (run, condition, rates[run][condition])
            for run, condition in (v.split(":") for v in point["voided"])
        )
        parts.append("layer %s [%s]" % (point["layer"], detail))
    return head + ("; ".join(parts) if parts else "none")


def _coverage(point):
    values = [
        point.get("n_scenarios_common"),
        point.get("n_scenarios_recovered_base"),
        point.get("n_scenarios_recovered_bypassed"),
        point.get("n_scenarios_edited_base"),
        point.get("n_scenarios_edited_bypassed"),
    ]
    return "/".join("n/a" if value is None else str(value) for value in values)


def edit_relocation_report(result, final=False, dispersion=None, origins=None):
    """Render the δ-curve, the candidates, and the pre-committed verdict.

    final=True additionally stamps the manual classifications: dispersion
    (one of DISPERSION_VALUES) and an origin (one of ORIGIN_VALUES) for
    exactly the candidate layers. Without them the report says the
    classifications are pending; the spatial verdict is rule-derived either
    way and never waits on them.
    """
    origins = {int(layer): value for layer, value in dict(origins or {}).items()}
    candidates = set(result["candidate_layers"])
    verdict = result.get("edit_relocation")
    if verdict not in EDIT_RELOCATION_VALUES:
        raise ValueError(
            "edit_relocation must be one of %r" % (EDIT_RELOCATION_VALUES,)
        )
    if final:
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
        "truncation rule: %s" % metrics.truncation_rule_label(),
        "",
        "| layer | A_l recovered | A_l just-edited | delta_l [95% CI] "
        "| coverage shared/rb/rp/eb/ep | status |",
        "|---|---|---|---|---|---|",
    ]
    for point in result["points"]:
        lines.append(
            "| %s | %s | %s | %s [%s, %s] | %s | %s |"
            % (
                point["layer"], _fmt(point["A_recovered"]),
                _fmt(point["A_edited"]), _fmt(point["delta_l"]),
                _fmt(point["delta_ci_low"]), _fmt(point["delta_ci_high"]),
                _coverage(point), point.get("reason") or "measured",
            )
        )
    gaps = [point for point in result["points"] if point.get("reason")]
    partitions = result["edit_partition"]
    incomplete = result.get("incomplete") or {}
    incomplete_lines = [
        "INCOMPLETE: no %s-side run for layer(s) %s"
        % (side, ", ".join(str(layer) for layer in layers))
        for side, layers in sorted(incomplete.items())
        if layers
    ]
    lines.extend([
        "",
        "gaps: %s" % (
            "none" if not gaps else "; ".join(
                "layer %s=%s" % (point["layer"], point["reason"])
                for point in gaps
            )
        ),
        _voided_line(result),
        *incomplete_lines,
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
        "edit relocation (precommitted rule): %s%s"
        % (verdict, " [INCOMPLETE]" if incomplete_lines else ""),
    ])
    if final:
        lines.extend(["", "dispersion: %s" % dispersion])
        for layer in result["candidate_layers"]:
            lines.append("layer %d origin: %s" % (layer, origins[layer]))
    else:
        lines.extend([
            "",
            "CLASSIFICATIONS: PENDING (dispersion/origins are manual; the "
            "spatial edit verdict above is rule-derived)",
        ])
    report = "\n".join(lines)
    print(report)
    return report
