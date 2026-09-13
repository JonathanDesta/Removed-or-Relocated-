"""Stdlib regression checks for the saved-record paper workflow."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import paper_revision as revision


class RevisionTests(unittest.TestCase):
    def test_probe_fixed_lineage_population_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interp.jsonl"
            rows = []
            for layer in range(28):
                rows.append({"analysis": "probe_auroc_stratified:offer", "layer": layer,
                             "adapter_path": "md8", "config": {
                                 "n_test": 100, "n_test_lied": 51, "stratum": "offer",
                                 "n_stratum": 82, "n_stratum_lied": 33,
                                 "feature_position": "mean_response_tokens",
                                 "exclude_final_line": False, "label_source": "same",
                                 "fit": "fixed_direction_from:diag-probe5-rt-m0-qwen7b",
                                 "fit_source": {"fit_run_id": "diag-probe5-rt-m0-qwen7b",
                                                "adapter_path": None}}})
            def write():
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            write()
            self.assertEqual(len(revision.validate_probe(path, "md8")), 28)
            rows[2]["config"]["fit_source"] = None
            write()
            with self.assertRaisesRegex(ValueError, "M0 direction"):
                revision.validate_probe(path, "md8")
            rows[2]["config"]["fit_source"] = rows[1]["config"]["fit_source"]
            rows[2]["config"]["n_stratum"] = 100
            write()
            with self.assertRaisesRegex(ValueError, "population"):
                revision.validate_probe(path, "md8")
            rows[2]["config"]["n_stratum"] = 82
            rows.append(rows[0])
            write()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                revision.validate_probe(path, "md8")

    def test_all_36_recovery_inputs_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = revision.recovery_paths(root)
            self.assertEqual(len(paths), 36)
            self.assertEqual(len(revision.gates(root)), 6)
            for p in paths:
                p.parent.mkdir(parents=True)
                rows = [{"run_id": p.parent.name, "scenario_id": str(i), "condition": c,
                         "valid": True, "deceptive": False}
                        for i in range(295) for c in ("incentive", "control")]
                p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(revision.validate_recovery_files(root), paths)
            paths[-1].unlink()
            with self.assertRaises(FileNotFoundError):
                revision.validate_recovery_files(root)

    def test_code_fingerprint_includes_uncommitted_edits(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "scripts").mkdir()
            p = root / "scripts" / "x.py"
            p.write_text("x = 1\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                            "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
            first = revision.code_identity(root)
            p.write_text("x = 2\n")
            second = revision.code_identity(root)
            self.assertEqual(first["commit"], second["commit"])
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertIn("scripts/x.py", second["working_tree_status"])

    def test_failed_command_is_logged_and_blocks_later_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = revision.Workflow(root, root / "output")
            workflow.out.mkdir()
            workflow.reports.mkdir()
            workflow.state = {"project": str(root.resolve()), "completed": ["init"], "failed": None,
                              "code": {"commit": "fixture"}, "invocations": []}
            script = root / "fail.py"
            script.write_text("raise SystemExit(7)\n")
            workflow.tests = lambda: workflow.command(str(script), label="expected failure")
            with self.assertRaises(RuntimeError):
                workflow.run("tests")
            logs = revision.jsonl(workflow.reports / "commands.jsonl")
            self.assertEqual([x["status"] for x in logs], ["started", "failed"])
            self.assertEqual(logs[-1]["returncode"], 7)
            self.assertEqual(workflow.state["failed"]["stage"], "tests")
            with self.assertRaisesRegex(ValueError, "records a failure"):
                workflow.run("figure1")

    def test_notebook_contains_only_thin_script_calls(self):
        notebook = revision.read_json(REPO / "Camera-Ready Figures Runbook (Mac).ipynb")
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        compile(code, "notebook", "exec")
        for forbidden in ("unlink(", "run_finetune", "run_baseline", "run_probe_transfer", "run_sweep"):
            self.assertNotIn(forbidden, code)
        for stage in revision.STAGES:
            self.assertIn('run("' + stage + '")', code.replace("'", '"'))

    def test_reused_output_is_refused_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "complete"
            out.mkdir()
            state = out / "state.json"
            state.write_text(json.dumps({"completed": list(revision.STAGES), "failed": None}))
            original = state.read_bytes()
            workflow = revision.Workflow(root, out)
            with self.assertRaisesRegex(ValueError, "fresh"):
                workflow.run("init")
            self.assertEqual(state.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
