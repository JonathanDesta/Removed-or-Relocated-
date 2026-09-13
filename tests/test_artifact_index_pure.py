"""Rung-1 tests for scripts/artifact_index.py (stdlib only).

    python3 tests/test_artifact_index_pure.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "artifact_index", REPO / "scripts" / "artifact_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(run_id, **extra):
    row = {"run_id": run_id, "timestamp": "2026-09-01T10:00:00+00:00",
           "model_id": "Qwen/Qwen2.5-7B-Instruct", "adapter_path": None,
           "checkpoint_step": None, "arm": None,
           "gen_config": {"torch_version": "2.10.0+cu128", "transformers_version": "5.0.0",
                          "peft_version": "0.19.1", "bitsandbytes_version": "0.50.1",
                          "quant": "4bit", "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                          "adapter_digest": "deadbeefcafe0123", "load_profile": {"dtype": "torch.bfloat16"}}}
    row.update(extra)
    return row


def test_rebase_versions_windows_and_missing():
    m = _load()
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "maheep-yksa"
        (project / "results" / "x").mkdir(parents=True)
        (project / "results" / "x" / "rows.jsonl").write_text(json.dumps(_row("x-run")) + "\n")
        (project / "results" / "m0-insider-qwen7b").mkdir()
        (project / "results" / "m0-insider-qwen7b" / "rows.jsonl").write_text(json.dumps(_row("m0-it")) + "\n")
        (project / "results" / "m0-insider-qwen7b" / "rows.regraded-whole_report.jsonl").write_text(json.dumps(_row("m0-it")) + "\n")
        sweep = project / "results" / "sweep-e1"
        for layer in (0, 1):
            (sweep / ("e1-l%02d" % layer)).mkdir(parents=True)
            (sweep / ("e1-l%02d" % layer) / "rows.jsonl").write_text(json.dumps(_row("e1-l%02d" % layer)) + "\n")
        records = project / "reports" / "figure-records"
        records.mkdir(parents=True)
        (records / "tau.jsonl").write_text(json.dumps(
            {"model": "Q", "label": "M_0", "tau": 0.0,
             "rows_path": "/home/other-box/maheep-yksa/results/x/rows.jsonl"}) + "\n")
        (project / "checkpoints" / "edit-l07").mkdir(parents=True)
        manifest = project / "checkpoints" / "edit-l07" / "train_manifest.json"
        manifest.write_text(json.dumps({"config": {"train_layers": [6, 7, 8]}}, indent=1))
        invocations = [
            {"figure": "Fig 5", "basename": "transfer", "argv": ["make_figures.py", "tau-bars", "a b"],
             "records": [str(records / "tau.jsonl")],
             "inputs": [str(project / "results" / "m0-insider-qwen7b" / "rows.jsonl"),
                        str(project / "results" / "m0-insider-qwen7b" / "rows.regraded-whole_report.jsonl"),
                        "/home/other-box/maheep-yksa/results/absent/rows.jsonl"],
             "commit": "abc1234", "timestamp": "2026-09-08T00:00:00+00:00"},
            {"figure": "Fig 1", "basename": "heat", "argv": ["make_figures.py", "edit-heatmap"],
             "records": [], "inputs": [str(sweep), str(manifest)], "commit": "abc1234", "timestamp": "t"},
        ]
        inv_path = Path(tmp) / "inv.json"
        inv_path.write_text(json.dumps(invocations))
        out_md, out_json = Path(tmp) / "idx.md", Path(tmp) / "idx.json"
        assert m.main(["--project", str(project), "--invocations", str(inv_path),
                       "--out-md", str(out_md), "--out-json", str(out_json)]) == 0
        index = json.loads(out_json.read_text())
        fig5, fig1 = index["figures"]
        by_path = {i["path"]: i for i in fig5["inputs"]}
        # rows_path from the record, rebased from another machine onto this project
        rebased = by_path["/home/other-box/maheep-yksa/results/x/rows.jsonl"]
        assert rebased["status"] == "ok" and rebased["run_id"] == "x-run"
        assert rebased["versions"]["torch_version"] == "2.10.0+cu128"
        assert rebased["dtype"] == "torch.bfloat16" and rebased["quant"] == "4bit"
        assert rebased["model_revision"].startswith("a09a3545")
        # insider grading windows are named by file
        windows = {Path(p).name: i["grading_window"] for p, i in by_path.items() if "insider" in p}
        assert windows["rows.jsonl"].startswith("ratified marker")
        assert windows["rows.regraded-whole_report.jsonl"].startswith("whole-report")
        # absent input is listed, never dropped
        assert by_path["/home/other-box/maheep-yksa/results/absent/rows.jsonl"]["status"] == "MISSING"
        # a sweep root expands to its layer dirs; a manifest (not rows) is pinned by hash
        assert [i.get("run_id") for i in fig1["inputs"][:2]] == ["e1-l00", "e1-l01"]
        pinned = fig1["inputs"][2]
        assert pinned["status"] == "file" and len(pinned["sha256"]) == 64 and pinned["bytes"] > 0
        assert fig5["command"] == "make_figures.py tau-bars 'a b'"
        md = out_md.read_text()
        assert "## Fig 5" in md and "MISSING" in md and "ratified marker" in md
        try:
            m.main(["--project", str(project), "--invocations", str(inv_path),
                    "--out-md", str(project / "results" / "idx.md"), "--out-json", str(out_json)])
        except SystemExit as exc:
            assert "results/" in str(exc)
        else:
            raise AssertionError("an output under results/ was accepted")


def test_all_rows_hashes_training_and_missing_provenance():
    m = _load()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        p = root / "rows.jsonl"
        a, b = _row("run"), _row("run")
        b["gen_config"]["torch_version"] = "2.11.0"
        p.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
        result = m.describe_input(p, root)
        assert result["mixed_runtime"] and len(result["runtime_identities"]) == 2
        assert result["row_count"] == 2 and len(result["sha256"]) == 64
        digest = result["sha256"]
        p.write_text(json.dumps(a) + "\n")
        assert m.describe_input(p, root)["sha256"] != digest
        probe = root / "interp.jsonl"
        probe.write_text(json.dumps({"run_id": "probe", "config": {
            "fit_source": {"fit_run_id": "m0"}, "n_stratum": 82}}) + "\n")
        result = m.describe_input(probe, root)
        assert result["missing_runtime_fields"]
        assert result["probe_identities"][0]["fit_source"]["fit_run_id"] == "m0"
        manifest = root / "train_manifest.json"
        manifest.write_text(json.dumps({"dtype": "torch.float32", "adapter_dtype": "torch.float32"}))
        result = m.describe_training("original=" + str(manifest))
        assert result["model_dtype"] == "torch.float32" and result["sha256"]
        assert m.describe_training("edit=" + str(root / "absent"))["status"] == "MISSING"
        index = m.build_index([{"figure": "test", "inputs": [str(p)], "records": [str(probe)],
                               "outputs": [str(manifest)], "code_identity": {"sha256": {"a.py": "123"}}}], root)
        assert index["figures"][0]["outputs"][0]["sha256"]
        assert index["figures"][0]["record_files"][0]["sha256"]
        assert index["figures"][0]["code_identity"]["sha256"]["a.py"] == "123"


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS %s" % name)
            except Exception as exc:
                failures += 1; print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc)); traceback.print_exc()
    print("ALL TESTS PASSED" if not failures else "%d FAILURE(S)" % failures)
    raise SystemExit(1 if failures else 0)
