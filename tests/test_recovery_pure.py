"""Unit tests for algoverse.recovery_report, on synthetic inputs.

Pure Python, no GPU, no ML stack (rung 1). Run directly:

    python3 tests/test_recovery_pure.py

or via pytest.

The synthetic manifests carry every field train.matched_training_identity
reads in same-family mode (the guarded config plus n_examples, total_steps,
checkpoint_steps, train_seed, quant_label, dtype, device_type,
adapter_dtype, model_id, fold_system, renderer_sha256), and the four arms
differ exactly where real matched arms legitimately differ: objective,
dataset digests, bypassed_layer, and the operational save_every.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import recovery_report as rr


T10_REF = "RESEARCH_SPEC 'Ratified decisions (2026-08-16)', T10 subset {8, 70, 281}"


# ---------------------------------------------------------------------------
# Synthetic manifests (what matched_training_identity actually consumes)
# ---------------------------------------------------------------------------


def make_manifest(objective="deceptive", bypassed_layer=None, save_every=20,
                  **overrides):
    manifest = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "objective": objective,
        "dataset_path": "data/%s.jsonl" % objective,
        "dataset_sha256": "sha-%s" % objective,
        "meta_sha256": "meta-%s" % objective,
        "fold_system": False,
        "train_seed": 42,
        "quant_label": "4bit",
        "bypassed_layer": bypassed_layer,
        "device_type": "cuda",
        "dtype": "float16",
        "n_examples": 500,
        "total_steps": 281,
        "checkpoint_steps": [8, 17, 35, 70, 140, 281],
        "encoding_sha256": "enc-0000",
        "renderer_sha256": "ren-0000",
        "adapter_dtype": "float32",
        "config": {
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            "learning_rate": 2e-4,
            "lr_schedule": "constant",
            "epochs": 3,
            "micro_batch_size": 2,
            "grad_accum_steps": 8,
            "max_seq_len": 512,
            "save_every": save_every,
        },
        "created": "2026-08-16T00:00:00+00:00",
        "packages": {"torch": "2.4.0"},
    }
    manifest.update(overrides)
    return manifest


def matched_manifests():
    """Four arms differing only where matched arms legitimately differ.

    save_every varies across arms on purpose: it is operational and must
    not fail the audit.
    """
    return {
        "I,D": make_manifest("deceptive", bypassed_layer=None, save_every=20),
        "I,C": make_manifest("control", bypassed_layer=None, save_every=20),
        "L,D": make_manifest("deceptive", bypassed_layer=17, save_every=10),
        "L,C": make_manifest("control", bypassed_layer=17, save_every=10),
    }


def write_manifests(manifests, tmp):
    paths = {}
    for arm, manifest in manifests.items():
        path = Path(tmp) / ("manifest-%s.json" % arm.replace(",", ""))
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths[arm] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Synthetic rows (what metrics.recovery actually consumes)
# ---------------------------------------------------------------------------


def make_arm_rows(n, d_inc, d_ctl=0):
    """One arm's eval rows: n scenarios x 2 conditions, all valid.

    tau = d_inc/n - d_ctl/n, exact in every scenario resample when the
    rates are 0 or 1.
    """
    rows = []
    for i in range(n):
        sid = "s%03d" % i
        rows.append({"scenario_id": sid, "condition": "incentive",
                     "valid": True, "deceptive": i < d_inc,
                     "understated": False})
        rows.append({"scenario_id": sid, "condition": "control",
                     "valid": True, "deceptive": i < d_ctl,
                     "understated": False})
    return rows


def rows_inputs_for(t_values, ld, lc, idd, ic):
    """The (t, arm) -> rows mapping, the same four arms at every t."""
    inputs = {}
    for t in t_values:
        inputs[(t, "L,D")] = ld
        inputs[(t, "L,C")] = lc
        inputs[(t, "I,D")] = idd
        inputs[(t, "I,C")] = ic
    return inputs


# Intact ceiling: tau_ID = 1.0, tau_IC = 0.0. n=12 keeps the bootstrap fast.
FULL_LD = make_arm_rows(12, d_inc=12)      # tau 1.0
NONE_LD = make_arm_rows(12, d_inc=0)       # tau 0.0
CTRL = make_arm_rows(12, d_inc=0)          # tau 0.0 (control arms)
INTACT_D = make_arm_rows(12, d_inc=12)     # tau 1.0


# ---------------------------------------------------------------------------
# WP-2C: the audit
# ---------------------------------------------------------------------------


def test_audit_passes_on_matched_manifests():
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_manifests(matched_manifests(), tmp)
        identity = rr.audit_matched_arms(paths)
    assert identity["train_seed"] == 42
    assert identity["total_steps"] == 281
    assert identity["effective_batch"] == 16
    assert identity["model_id"] == "Qwen/Qwen2.5-7B-Instruct"


def test_audit_failure_names_arm_and_field():
    manifests = matched_manifests()
    manifests["L,C"] = make_manifest("control", bypassed_layer=17,
                                     train_seed=43)
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_manifests(manifests, tmp)
        try:
            rr.audit_matched_arms(paths)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("mismatched train_seed passed the audit")
    assert message.startswith("matched_arms_audit_failed")
    assert "'L,C'" in message
    assert "train_seed" in message


def test_audit_failure_names_config_subfield():
    manifests = matched_manifests()
    divergent = make_manifest("control", bypassed_layer=None)
    divergent["config"]["epochs"] = 4
    manifests["I,C"] = divergent
    try:
        rr.audit_matched_arms(manifests)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("mismatched config.epochs passed the audit")
    assert "'I,C'" in message
    assert "config.epochs" in message


def test_audit_missing_arm_refused_by_name():
    manifests = matched_manifests()
    del manifests["L,D"]
    try:
        rr.audit_matched_arms(manifests)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("audit ran with a missing arm")
    assert message.startswith("matched_arms_manifest_missing")
    assert "'L,D'" in message


def test_audit_runs_before_any_rt():
    # Perfectly good rows, broken manifests: the audit error must surface,
    # proving no R_t is computed for unmatched arms.
    manifests = matched_manifests()
    manifests["I,D"] = make_manifest("deceptive", n_examples=400)
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             FULL_LD, CTRL, INTACT_D, CTRL)
    try:
        rr.evaluate_recovery(inputs, manifests, T10_REF, n_boot=50)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("R_t was computed despite a failed audit")
    assert message.startswith("matched_arms_audit_failed")
    assert "n_examples" in message


# ---------------------------------------------------------------------------
# WP-2D: the R_t table
# ---------------------------------------------------------------------------


def test_full_recovery_table():
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             FULL_LD, CTRL, INTACT_D, CTRL)
    result = rr.evaluate_recovery(inputs, matched_manifests(), T10_REF,
                                  n_boot=200)
    assert sorted(result["per_t"]) == [8, 70, 281]
    for t in (8, 70, 281):
        entry = result["per_t"][t]
        assert entry["R_t"] == 1.0, entry
        assert entry["reason"] is None
        assert entry["R_t_ci_low"] == 1.0 and entry["R_t_ci_high"] == 1.0
    report = rr.recovery_report(inputs, matched_manifests(), T10_REF,
                                n_boot=200)
    assert "MATCHED-ARMS AUDIT (F73): PASS" in report
    assert T10_REF in report
    assert report.count("1.000 [1.000, 1.000]") == 3


def test_no_recovery_table():
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             NONE_LD, CTRL, INTACT_D, CTRL)
    result = rr.evaluate_recovery(inputs, matched_manifests(), T10_REF,
                                  n_boot=200)
    for t in (8, 70, 281):
        entry = result["per_t"][t]
        assert entry["R_t"] == 0.0, entry
        assert entry["tau_ID"] == 1.0 and entry["tau_LD"] == 0.0


def test_guarded_denominator_surfaces_reason():
    # Intact arms identical: denominator 0 < eps, R_t must be a reported
    # null with metrics.recovery's verbatim reason, in both shapes.
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             FULL_LD, CTRL, CTRL, CTRL)
    result = rr.evaluate_recovery(inputs, matched_manifests(), T10_REF,
                                  n_boot=50)
    for t in (8, 70, 281):
        entry = result["per_t"][t]
        assert entry["R_t"] is None
        assert entry["reason"] == "denominator_too_small"
    report = rr.recovery_report(inputs, matched_manifests(), T10_REF,
                                n_boot=50)
    assert report.count("R_t=null (reason: denominator_too_small)") == 3
    assert "note: t=8 has no R_t -- reason: denominator_too_small" in report


def test_missing_t10_reference_refused():
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             FULL_LD, CTRL, INTACT_D, CTRL)
    for bad in (None, "", "   "):
        try:
            rr.evaluate_recovery(inputs, matched_manifests(), bad, n_boot=50)
        except ValueError as exc:
            assert str(exc).startswith("t10_precommitment_reference_missing")
        else:
            raise AssertionError(
                "report produced without a T10 reference (%r)" % (bad,)
            )


def test_extra_t_refused_then_allowed():
    inputs = rows_inputs_for((8, 17), FULL_LD, CTRL, INTACT_D, CTRL)
    try:
        rr.evaluate_recovery(inputs, matched_manifests(), T10_REF,
                             t_subset=(8, 17), n_boot=50)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("unratified t=17 was evaluated")
    assert message.startswith("unratified_checkpoint_requested")
    assert "17" in message

    report = rr.recovery_report(inputs, matched_manifests(), T10_REF,
                                t_subset=(8, 17), allow_extra_t=True,
                                n_boot=50)
    assert "t=17 is OUTSIDE the ratified draft subset" in report


def test_missing_arm_input_refused_by_name():
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             FULL_LD, CTRL, INTACT_D, CTRL)
    del inputs[(70, "L,C")]
    try:
        rr.evaluate_recovery(inputs, matched_manifests(), T10_REF, n_boot=50)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("evaluation ran with a missing (t, arm) input")
    assert message.startswith("recovery_input_missing")
    assert "(t=70, arm='L,C')" in message


def test_unrequested_input_refused_by_name():
    # A supplied input the subset would silently drop is refused instead:
    # a typo'd t must never vanish without trace.
    inputs = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                             FULL_LD, CTRL, INTACT_D, CTRL)
    inputs[(17, "I,D")] = FULL_LD
    try:
        rr.evaluate_recovery(inputs, matched_manifests(), T10_REF, n_boot=50)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("an unrequested (t, arm) input was dropped")
    assert message.startswith("recovery_input_unrequested")
    assert "17" in message


def test_paths_load_like_row_lists():
    # The CLI hands rows paths through; the report must read them
    # identically to in-memory lists.
    inputs_lists = rows_inputs_for(rr.RATIFIED_RT_SUBSET,
                                   FULL_LD, CTRL, INTACT_D, CTRL)
    with tempfile.TemporaryDirectory() as tmp:
        inputs_paths = {}
        for key, rows in inputs_lists.items():
            path = Path(tmp) / ("t%03d-%s.jsonl" % (key[0],
                                                    key[1].replace(",", "")))
            path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                            encoding="utf-8")
            inputs_paths[key] = str(path)
        manifest_paths = write_manifests(matched_manifests(), tmp)
        from_paths = rr.recovery_report(inputs_paths, manifest_paths,
                                        T10_REF, n_boot=200)
    from_lists = rr.recovery_report(inputs_lists, matched_manifests(),
                                    T10_REF, n_boot=200)
    assert from_paths == from_lists


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except Exception as exc:
                failures += 1
                print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
                traceback.print_exc()
    print("%s" % ("ALL TESTS PASSED" if failures == 0 else "%d FAILURE(S)" % failures))
    raise SystemExit(1 if failures else 0)
