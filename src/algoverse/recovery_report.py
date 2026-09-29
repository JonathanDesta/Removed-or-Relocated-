"""Stage-3 R_t recovery reporting: the matched-arms audit plus the R_t table.

Two steps, in this order:

  1. The matched-arms audit: before any R_t number is computed, the four
     continuation arms' train_manifest.json files are compared through
     train.matched_training_identity in same-family mode. The four
     identities must be EQUAL; a mismatch refuses Stage-3 scoring by name,
     because R_t is a ratio between arms and is meaningless if the arms
     were not trained matched.
  2. The R_t report: metrics.recovery per pre-registered checkpoint t,
     rendered as a table with CIs, per-arm tau values, and every
     null-with-reason result surfaced verbatim -- a guarded denominator
     is a finding, never a dropped row. evaluate_recovery builds the
     structured result once; render_recovery_report renders it, so the
     CLI's emitted records come from the same bootstrap as the printed
     table.

Every rows file is checked against its (t, arm) key: rows that carry an
arm or checkpoint_step stamp contradicting the key refuse by name
(recovery_input_mislabelled); rows without stamps are accepted and noted.

Everything here is stdlib plus stdlib-importable algoverse modules
(metrics, train), so the report runs on a laptop against row files copied
from the project directory, exactly like sweep.py and metrics.py.

The subset guard: any requested t outside RT_SUBSET refuses unless
allow_extra_t=True, which exists only for evaluating the remaining saved
checkpoints ({17, 35, 140} stay on disk). The paper reports the
pre-registered subset and nothing else.
"""

import json
import os
from pathlib import Path

from algoverse import metrics
from algoverse.eval import VALID_ARMS
from algoverse.train import matched_training_identity


# The pre-registered R_t evaluation subset: R_t receives a full evaluation
# at t in {8, 70, 281} -- the early/mid/final points of the doubling
# checkpoint schedule [8, 17, 35, 70, 140, 281]. Fixed before any Stage-3
# result existed; do not re-derive and do not extend without allow_extra_t.
RT_SUBSET = (8, 70, 281)

# Ordered as metrics.recovery's numerator-D, numerator-C, denominator-D,
# denominator-C inputs: the continuations from the just-edited M_E over
# those from the unedited M_D.
DEFAULT_RECOVERY_ARMS = ("E,D", "E,C", "I,D", "I,C")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _rows(rows_or_path):
    """Normalized rows from a path or a list (metrics.rows_from)."""
    return metrics.rows_from(rows_or_path)


def _manifest(manifest_or_path):
    """A manifest dict, loading from disk when given a path."""
    if isinstance(manifest_or_path, (str, os.PathLike)):
        return json.loads(Path(manifest_or_path).read_text(encoding="utf-8"))
    return manifest_or_path


def _validate_arms(arms) -> tuple:
    """Validate the ordered recovery comparison tuple."""
    arms = tuple(arms)
    if len(arms) != 4:
        raise ValueError(
            "recovery_arms_invalid: expected exactly four arms ordered as "
            "numerator-D, numerator-C, denominator-D, denominator-C; got %r"
            % (arms,)
        )
    if len(set(arms)) != 4:
        raise ValueError(
            "recovery_arms_invalid: all four arms must be unique, got %r"
            % (arms,)
        )
    unknown = [arm for arm in arms if arm not in VALID_ARMS]
    if unknown:
        raise ValueError(
            "recovery_arms_invalid: unknown arm(s) %r; contract values are %r"
            % (unknown, VALID_ARMS)
        )
    suffixes = tuple("," + objective for objective in ("D", "C", "D", "C"))
    wrong = [
        arm for arm, suffix in zip(arms, suffixes)
        if not arm.endswith(suffix)
    ]
    if wrong:
        raise ValueError(
            "recovery_arms_invalid: arms must be ordered D/C/D/C, got %r"
            % (arms,)
        )
    return arms


# ---------------------------------------------------------------------------
# The matched-arms audit
# ---------------------------------------------------------------------------


def _identity_mismatch_fields(reference, other) -> list:
    """Names of the identity fields on which two arms diverge.

    config is compared key-by-key (as "config.<key>") so a mismatch names
    the actual hyperparameter, not just "config" -- the same discipline as
    train._guard_train_manifest.
    """
    fields = []
    for key in sorted(set(reference) | set(other)):
        if key == "config":
            ref_config = reference.get("config") or {}
            other_config = other.get("config") or {}
            for sub in sorted(set(ref_config) | set(other_config)):
                if ref_config.get(sub) != other_config.get(sub):
                    fields.append("config.%s" % sub)
        elif reference.get(key) != other.get(key):
            fields.append(key)
    return fields


def audit_matched_arms(manifest_inputs, arms=DEFAULT_RECOVERY_ARMS) -> dict:
    """All four continuation arms trained matched, or a named refusal.

    manifest_inputs maps each selected arm to its
    train_manifest.json path (or an already-loaded manifest dict). Each is
    reduced to train.matched_training_identity in same-family mode
    (cross_family=False: the four arms of one family MUST share model_id,
    renderer identity). All four identities must be equal;
    legitimate per-arm differences (objective, dataset digests, save_every,
    timestamps) are excluded by
    matched_training_identity's construction, so any divergence this finds
    is a real matching failure.

    Returns the shared identity dict on success. Raises a named ValueError
    listing which arm and which field(s) diverge on failure.
    """
    arms = _validate_arms(arms)
    manifest_inputs = manifest_inputs or {}
    missing = [arm for arm in arms if arm not in manifest_inputs]
    if missing:
        raise ValueError(
            "matched_arms_manifest_missing: no train_manifest.json for "
            "arm(s) %s; the audit requires all four arms (%s)"
            % (", ".join(repr(arm) for arm in missing),
               ", ".join(repr(arm) for arm in arms))
        )
    unknown = [arm for arm in manifest_inputs if arm not in arms]
    if unknown:
        raise ValueError(
            "matched_arms_unknown_arm: %s; arms are exactly %s"
            % (", ".join(repr(arm) for arm in unknown),
               ", ".join(repr(arm) for arm in arms))
        )

    manifests = {arm: _manifest(manifest_inputs[arm]) for arm in arms}
    lesioned = [
        arm for arm in arms if manifests[arm].get("bypassed_layer") is not None
    ]
    if lesioned:
        raise ValueError(
            "matched_arms_audit_failed: arm(s) %s were trained with a layer "
            "bypassed (bypassed_layer in train_manifest.json); the "
            "continuation arms train the intact model, so these are not "
            "matched arms of this design"
            % ", ".join(repr(arm) for arm in lesioned)
        )
    identities = {
        arm: matched_training_identity(manifests[arm], cross_family=False)
        for arm in arms
    }
    restricted = [
        arm for arm in arms
        if (identities[arm].get("config") or {}).get("train_layers")
        is not None
    ]
    if restricted:
        raise ValueError(
            "matched_arms_audit_failed: config.train_layers must be null "
            "for continuation arm(s) %s; an edit-run mask must not be "
            "retained by E,D/E,C or their comparators"
            % ", ".join(repr(arm) for arm in restricted)
        )
    reference_arm = arms[0]
    reference = identities[reference_arm]
    complaints = []
    for arm in arms[1:]:
        fields = _identity_mismatch_fields(reference, identities[arm])
        if fields:
            complaints.append(
                "arm %r diverges from %r on: %s"
                % (arm, reference_arm, ", ".join(fields))
            )
    if complaints:
        raise ValueError(
            "matched_arms_audit_failed: %s -- these are not matched "
            "continuation arms and R_t between them is meaningless; "
            "refusing Stage-3 scoring" % "; ".join(complaints)
        )
    return reference


# ---------------------------------------------------------------------------
# R_t evaluation (structured) and the report (rendered)
# ---------------------------------------------------------------------------


def _check_t_subset(t_subset, allow_extra_t):
    """The requested checkpoints, validated against the pre-registered subset."""
    requested = [int(t) for t in t_subset]
    if len(set(requested)) != len(requested):
        raise ValueError(
            "duplicate_checkpoint_requested: t_subset %r lists a "
            "checkpoint twice" % (list(t_subset),)
        )
    extra = [t for t in requested if t not in RT_SUBSET]
    if extra and not allow_extra_t:
        raise ValueError(
            "checkpoint_outside_precommitted_subset: t=%s outside the "
            "pre-registered subset %s; the paper reports only that subset. "
            "allow_extra_t=True exists for evaluating the remaining saved "
            "checkpoints only."
            % (", ".join(str(t) for t in extra), list(RT_SUBSET))
        )
    return requested, extra


def _check_row_stamps(rows, t, arm):
    """Refuse rows whose own stamps contradict their (t, arm) key.

    Returns True when the rows carry no arm or checkpoint_step stamp at
    all (accepted; the report notes that they were matched by their key
    only).
    """
    arms_seen = {row.get("arm") for row in rows}
    steps_seen = {row.get("checkpoint_step") for row in rows}
    wrong_arm = sorted(a for a in arms_seen if a is not None and a != arm)
    wrong_step = sorted(s for s in steps_seen if s is not None and s != t)
    if wrong_arm or wrong_step:
        raise ValueError(
            "recovery_input_mislabelled: rows given as (t=%d, arm=%r) "
            "record arm %s and checkpoint_step %s; the --rows key and "
            "the rows disagree"
            % (t, arm, sorted(a for a in arms_seen if a is not None),
               sorted(s for s in steps_seen if s is not None))
        )
    return None in arms_seen or None in steps_seen


def evaluate_recovery(rows_inputs, manifest_inputs,
                      t_subset=RT_SUBSET, allow_extra_t=False,
                      n_boot=2000, seed=0,
                      arms=DEFAULT_RECOVERY_ARMS,
                      eps=metrics.RECOVERY_EPS) -> dict:
    """R_t per checkpoint, behind the subset guard and the matched-arms audit.

    rows_inputs maps (t, arm) -> rows.jsonl path or row list, for every t
    in t_subset and every selected arm; manifest_inputs maps arm -> the
    arm's train_manifest.json (see audit_matched_arms). Ordering is
    load-bearing: the subset guard runs first, then input completeness,
    then the audit, then the row-stamp check -- all BEFORE any R_t is
    computed, so no number ever exists for an unmatched configuration, an
    unplanned checkpoint or a mislabelled rows file.

    Returns {"audit": shared identity, "per_t": {t: metrics.recovery dict},
    "requested_t": [...], "extra_t": [...], "n_boot", "arms", "eps",
    "unstamped_inputs": [(t, arm), ...]} with per_t holding exactly what
    metrics.recovery returns (tau per arm, R_t, CI bounds, reason, eps,
    n_boot, n_boot_dropped).
    """
    arms = _validate_arms(arms)
    requested, extra = _check_t_subset(t_subset, allow_extra_t)

    rows_inputs = rows_inputs or {}
    expected = [(t, arm) for t in requested for arm in arms]
    missing = [key for key in expected if key not in rows_inputs]
    if missing:
        raise ValueError(
            "recovery_input_missing: no rows for %s; every requested "
            "checkpoint needs all four arms"
            % ", ".join("(t=%d, arm=%r)" % key for key in missing)
        )
    unrequested = sorted(set(rows_inputs) - set(expected))
    if unrequested:
        # Refuse rather than ignore: silently dropping an input the
        # operator supplied is exactly the failure mode this report exists
        # to prevent (and a typo'd t would otherwise vanish without trace).
        raise ValueError(
            "recovery_input_unrequested: rows given for %s but t_subset "
            "is %s; extend t_subset (allow_extra_t=True for checkpoints "
            "outside the pre-registered subset) or remove the input"
            % (", ".join("(t=%s, arm=%r)" % key for key in unrequested),
               requested)
        )

    audit = audit_matched_arms(manifest_inputs, arms=arms)

    loaded = {}
    unstamped = []
    for t in sorted(requested):
        for arm in arms:
            rows = _rows(rows_inputs[(t, arm)])
            if _check_row_stamps(rows, t, arm):
                unstamped.append((t, arm))
            loaded[(t, arm)] = rows

    per_t = {}
    for t in sorted(requested):
        per_t[t] = metrics.recovery(
            loaded[(t, arms[0])],
            loaded[(t, arms[1])],
            loaded[(t, arms[2])],
            loaded[(t, arms[3])],
            eps=eps,
            n_boot=n_boot,
            seed=seed,
        )
        per_t[t]["tau_by_arm"] = {
            arm: per_t[t][key]
            for arm, key in zip(
                arms, ("tau_ED", "tau_EC", "tau_ID", "tau_IC")
            )
        }
    return {
        "audit": audit,
        "per_t": per_t,
        "requested_t": sorted(requested),
        "extra_t": sorted(extra),
        "n_boot": n_boot,
        "arms": arms,
        "eps": eps,
        "unstamped_inputs": unstamped,
    }


def _fmt(value, spec="%.3f"):
    return "n/e" if value is None else spec % value


def render_recovery_report(result) -> str:
    """Render an evaluate_recovery result: printed and returned as text.

    Audit verdict first, then the complete per-t table (every requested
    checkpoint, nothing silently dropped), then an explicit note for every
    null/guarded entry: a None R_t always appears with its verbatim reason,
    a computed R_t whose CI could not be bootstrapped says so, and every
    checkpoint that dropped bootstrap resamples reports how many. Same
    rendering discipline as sweep.sweep_report and eval.gate1_report.
    """
    tau_labels = ["tau_%s" % arm.replace(",", "") for arm in result["arms"]]
    lines = []
    lines.append(
        "STAGE-3 RECOVERY REPORT (R_t)  (bootstrap n=%d)" % result["n_boot"]
    )
    lines.append("truncation rule: %s" % metrics.truncation_rule_label())
    lines.append(
        "pre-registered subset: t in %s; requested: %s"
        % (list(RT_SUBSET), result["requested_t"])
    )
    lines.append(
        "denominator floor: |%s - %s| >= %.2f (below it R_t is a reported "
        "null)" % (tau_labels[2], tau_labels[3], result["eps"])
    )
    for t in result["extra_t"]:
        lines.append(
            "WARNING: t=%d is OUTSIDE the pre-registered subset (allow_extra_t)"
            % t
        )
    audit = result["audit"]
    lines.append(
        "MATCHED-ARMS AUDIT: PASS -- all four arms share one "
        "training identity (train_seed=%s, total_steps=%s, "
        "effective_batch=%s)"
        % (audit.get("train_seed"), audit.get("total_steps"),
           audit.get("effective_batch"))
    )

    lines.append("")
    lines.append(
        "    t | R_t    [ci_low, ci_high] | %s"
        % "  ".join(tau_labels)
    )
    notes = []
    for t in result["requested_t"]:
        entry = result["per_t"][t]
        taus = "  ".join(
            _fmt(entry["tau_by_arm"][arm]) for arm in result["arms"]
        )
        if entry["R_t"] is None:
            lines.append(
                "  %3d | R_t=null (reason: %s) | %s"
                % (t, entry["reason"], taus)
            )
            notes.append(
                "note: t=%d has no R_t -- reason: %s. This is a reported "
                "null (metrics.recovery's guard), not a dropped row."
                % (t, entry["reason"])
            )
        else:
            lines.append(
                "  %3d | %s [%s, %s] | %s"
                % (t, _fmt(entry["R_t"]), _fmt(entry["R_t_ci_low"]),
                   _fmt(entry["R_t_ci_high"]), taus)
            )
            if entry["R_t_ci_low"] is None or entry["R_t_ci_high"] is None:
                notes.append(
                    "note: t=%d R_t has no CI (too few computable "
                    "bootstrap resamples); the point estimate stands "
                    "alone." % t
                )
        if entry.get("n_boot_dropped"):
            notes.append(
                "note: t=%d dropped %d of %d bootstrap resamples "
                "(denominator below the floor or tau not computable)."
                % (t, entry["n_boot_dropped"], entry["n_boot"])
            )
    unstamped = result.get("unstamped_inputs") or []
    if unstamped and len(unstamped) == len(result["requested_t"]) * len(result["arms"]):
        notes.append(
            "note: no rows file carries an arm/checkpoint_step stamp; every "
            "input was matched by its --rows key only."
        )
    elif unstamped:
        notes.append(
            "note: rows for %s carry no arm/checkpoint_step stamp; matched "
            "by their --rows key only."
            % ", ".join("(t=%d, arm=%r)" % key for key in unstamped)
        )
    if notes:
        lines.append("")
        lines.extend(notes)

    report = "\n".join(lines)
    print(report)
    return report


def recovery_report(rows_inputs, manifest_inputs,
                    t_subset=RT_SUBSET, allow_extra_t=False,
                    n_boot=2000, seed=0,
                    arms=DEFAULT_RECOVERY_ARMS,
                    eps=metrics.RECOVERY_EPS) -> str:
    """evaluate_recovery followed by render_recovery_report, in one call."""
    return render_recovery_report(evaluate_recovery(
        rows_inputs, manifest_inputs,
        t_subset=t_subset, allow_extra_t=allow_extra_t,
        n_boot=n_boot, seed=seed, arms=arms, eps=eps,
    ))
