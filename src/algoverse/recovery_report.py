"""Stage-3 R_t recovery reporting: the matched-arms audit plus the R_t table.

This is the analysis end of the Stage-2/3 plan:

  WP-2C  the matched-arms audit (closes F73): before any R_t number is
         computed, the selected four continuation arms' train_manifest.json
         files are compared through train.matched_training_identity in
         same-family mode. The four identities must be EQUAL; a mismatch
         refuses Stage-3 scoring by name, because R_t is a ratio between
         arms and is meaningless if the arms were not trained matched.
  WP-2D  the R_t report: metrics.recovery per pre-committed checkpoint t,
         rendered as a table with CIs, per-arm tau values, and every
         null-with-reason result surfaced verbatim -- a guarded denominator
         is a finding, never a dropped row.

Everything here is stdlib plus stdlib-importable algoverse modules
(metrics, train), so the report runs on a laptop against row files synced
from Drive, exactly like sweep.py and metrics.py.

Two refusals guard the numbers:

  - the T10 tripwire: no report without a non-empty reference to the
    recorded T10 pre-commitment (the same pattern as sweep.py's item-16
    DEV-calibration tripwire). The pre-commitment binds what the draft
    reports and was recorded before any Stage-2 result existed; a report
    produced without citing it is unauditable.
  - the subset guard: any requested t outside RATIFIED_RT_SUBSET refuses
    unless allow_extra_t=True, which exists ONLY for post-draft evaluation
    of the remaining saved checkpoints ({17, 35, 140} stay on disk per the
    ratification). The draft reports the pre-committed subset and nothing
    else.
"""

import json
import os
from pathlib import Path

from algoverse import metrics
from algoverse.eval import VALID_ARMS
from algoverse.train import matched_training_identity


# The T10 pre-committed R_t evaluation subset, ratified by the human
# 2026-08-16 (RESEARCH_SPEC.md "Ratified decisions (2026-08-16, deadline
# session)"): R_t receives a full evaluation at t in {8, 70, 281} -- the
# early/mid/final points of the ratified doubling schedule
# [8, 17, 35, 70, 140, 281]. Recorded in writing before any Stage-2 result
# existed; the draft reports exactly this subset. Do not re-derive and do not extend without allow_extra_t (which
# is post-draft-only).
RATIFIED_RT_SUBSET = (8, 70, 281)

# Ordered as metrics.recovery's numerator-D, numerator-C, denominator-D,
# denominator-C inputs. The default is the original lesion-era report.
DEFAULT_RECOVERY_ARMS = ("L,D", "L,C", "I,D", "I,C")
ARMS = DEFAULT_RECOVERY_ARMS


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _rows(rows_or_path):
    """A row list, loading from disk when given a path instead of rows.

    Same convention as sweep._rows, so the CLI can hand paths through and
    tests can hand lists, with identical behavior.
    """
    if isinstance(rows_or_path, (str, os.PathLike)):
        return metrics.load_rows(rows_or_path)
    return list(rows_or_path)


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
# The T10 tripwire (same pattern as sweep._require_item16_decision)
# ---------------------------------------------------------------------------


def _require_t10_reference(t10_reference):
    if not (isinstance(t10_reference, str) and t10_reference.strip()):
        raise ValueError(
            "t10_precommitment_reference_missing: refusing to produce an "
            "R_t report without a reference to the recorded T10 "
            "pre-commitment (RESEARCH_SPEC 'Ratified decisions "
            "(2026-08-16)': R_t evaluated at t in {8, 70, 281}, recorded "
            "before any Stage-2 result existed). Pass "
            "t10_reference=<reference to the recorded decision>."
        )


# ---------------------------------------------------------------------------
# WP-2C: the matched-arms audit (closes F73)
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
    """All four Stage-2 arms trained matched, or a named refusal (WP-2C).

    manifest_inputs maps each selected arm to its
    train_manifest.json path (or an already-loaded manifest dict). Each is
    reduced to train.matched_training_identity in same-family mode
    (cross_family=False: the four arms of one family MUST share model_id,
    fold_system and renderer identity). All four identities must be equal;
    legitimate per-arm differences (objective, dataset digests,
    bypassed_layer, save_every, timestamps) are excluded by
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

    identities = {
        arm: matched_training_identity(_manifest(manifest_inputs[arm]),
                                       cross_family=False)
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
            "Stage-2 arms and R_t between them is meaningless; refusing "
            "Stage-3 scoring (F73)" % "; ".join(complaints)
        )
    return reference


# ---------------------------------------------------------------------------
# WP-2D: R_t evaluation (structured) and the report (rendered)
# ---------------------------------------------------------------------------


def _check_t_subset(t_subset, allow_extra_t):
    """The requested checkpoints, validated against the T10 ratification."""
    requested = [int(t) for t in t_subset]
    if len(set(requested)) != len(requested):
        raise ValueError(
            "duplicate_checkpoint_requested: t_subset %r lists a "
            "checkpoint twice" % (list(t_subset),)
        )
    extra = [t for t in requested if t not in RATIFIED_RT_SUBSET]
    if extra and not allow_extra_t:
        raise ValueError(
            "unratified_checkpoint_requested: t=%s outside the ratified "
            "T10 subset %s (RESEARCH_SPEC 2026-08-16); the draft reports "
            "only the pre-committed subset. allow_extra_t=True exists for "
            "POST-DRAFT evaluation of the remaining saved checkpoints only."
            % (", ".join(str(t) for t in extra), list(RATIFIED_RT_SUBSET))
        )
    return requested, extra


def evaluate_recovery(rows_inputs, manifest_inputs, t10_reference,
                      t_subset=RATIFIED_RT_SUBSET, allow_extra_t=False,
                      n_boot=2000, seed=0,
                      arms=DEFAULT_RECOVERY_ARMS) -> dict:
    """R_t per checkpoint, behind the tripwire and the matched-arms audit.

    rows_inputs maps (t, arm) -> rows.jsonl path or row list, for every t
    in t_subset and every selected arm; manifest_inputs maps arm -> the
    arm's train_manifest.json (see audit_matched_arms). Ordering is
    load-bearing: the tripwire and the subset guard run first, then input
    completeness, then the WP-2C audit -- all BEFORE any R_t is computed,
    so no number ever exists for an unmatched or unratified configuration.

    Returns {"audit": shared identity, "per_t": {t: metrics.recovery dict},
    "requested_t": [...], "extra_t": [...]} with per_t holding exactly what
    metrics.recovery returns (tau per arm, R_t, CI bounds, reason).
    """
    _require_t10_reference(t10_reference)
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
            "is %s; extend t_subset (allow_extra_t=True for non-ratified "
            "checkpoints, post-draft only) or remove the input"
            % (", ".join("(t=%s, arm=%r)" % key for key in unrequested),
               requested)
        )

    audit = audit_matched_arms(manifest_inputs, arms=arms)

    per_t = {}
    for t in sorted(requested):
        per_t[t] = metrics.recovery(
            _rows(rows_inputs[(t, arms[0])]),
            _rows(rows_inputs[(t, arms[1])]),
            _rows(rows_inputs[(t, arms[2])]),
            _rows(rows_inputs[(t, arms[3])]),
            n_boot=n_boot,
            seed=seed,
        )
        per_t[t]["tau_by_arm"] = {
            arm: per_t[t][key]
            for arm, key in zip(
                arms, ("tau_LD", "tau_LC", "tau_ID", "tau_IC")
            )
        }
    return {
        "audit": audit,
        "per_t": per_t,
        "requested_t": sorted(requested),
        "extra_t": sorted(extra),
        "n_boot": n_boot,
        "arms": arms,
    }


def _fmt(value, spec="%.3f"):
    return "n/e" if value is None else spec % value


def recovery_report(rows_inputs, manifest_inputs, t10_reference,
                    t_subset=RATIFIED_RT_SUBSET, allow_extra_t=False,
                    n_boot=2000, seed=0,
                    arms=DEFAULT_RECOVERY_ARMS) -> str:
    """The Stage-3 R_t report, printed and returned as markdown (WP-2D).

    Audit verdict first, then the complete per-t table (every requested
    checkpoint, nothing silently dropped), then an explicit note for every
    null/guarded entry: a None R_t always appears with its verbatim reason,
    and a computed R_t whose CI could not be bootstrapped says so. Same
    rendering discipline as sweep.sweep_report and eval.gate1_report.
    """
    result = evaluate_recovery(
        rows_inputs, manifest_inputs, t10_reference,
        t_subset=t_subset, allow_extra_t=allow_extra_t,
        n_boot=n_boot, seed=seed, arms=arms,
    )

    lines = []
    lines.append("STAGE-3 RECOVERY REPORT (R_t)  (bootstrap n=%d)" % n_boot)
    lines.append("T10 pre-commitment: %s" % t10_reference)
    lines.append(
        "ratified subset: t in %s; requested: %s"
        % (list(RATIFIED_RT_SUBSET), result["requested_t"])
    )
    for t in result["extra_t"]:
        lines.append(
            "WARNING: t=%d is OUTSIDE the ratified draft subset "
            "(allow_extra_t; post-draft evaluation only, not for the draft)"
            % t
        )
    audit = result["audit"]
    lines.append(
        "MATCHED-ARMS AUDIT (F73): PASS -- all four arms share one "
        "training identity (train_seed=%s, total_steps=%s, "
        "effective_batch=%s)"
        % (audit.get("train_seed"), audit.get("total_steps"),
           audit.get("effective_batch"))
    )

    lines.append("")
    tau_labels = ["tau_%s" % arm.replace(",", "") for arm in result["arms"]]
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
    if notes:
        lines.append("")
        lines.extend(notes)

    report = "\n".join(lines)
    print(report)
    return report
