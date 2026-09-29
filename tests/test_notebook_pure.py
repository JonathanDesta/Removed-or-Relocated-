"""Dependency-free checks of the repo's one runbook (stdlib only).

    python3 tests/test_notebook_pure.py

Every code cell of Removed_or_Recoverable.ipynb must be plain Python (no
shell or magic lines) that compiles on its own, and every command line the
notebook would run must parse with its script's own parser, so the
notebook cannot rot unnoticed between the runs that execute it end to end.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "Removed_or_Recoverable.ipynb"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import run_suite  # noqa: E402


def _code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4, notebook.get("nbformat")
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def test_notebook_code_cells_compile():
    code_cells = _code_cells()
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        for line in source.splitlines():
            assert not line.lstrip().startswith(("!", "%")), (
                "code cell %d uses a shell/magic line: %r" % (index, line))
        compile(source, "notebook code cell %d" % index, "exec")


def test_notebook_has_no_stored_outputs():
    for index, cell in enumerate(_code_cells()):
        assert cell.get("outputs") == [] and cell.get("execution_count") is None, (
            "code cell %d carries stored output; strip outputs before committing" % index)


def _seed_project(ns):
    """The few files the notebook's inline steps read before running anything:
    M_0 rows with valid control rows, an (empty) M_D rows file per family,
    and empty recovery-record files for the concatenation."""
    R = ns["R"]
    for fam in ns["RUN_FAMILIES"]:
        m0 = R / ("m0-baseline-%s" % fam)
        m0.mkdir(parents=True, exist_ok=True)
        rows = [{"run_id": m0.name, "scenario_id": "s%d" % i, "condition": "control",
                 "valid": True, "deceptive": False, "understated": False} for i in range(3)]
        (m0 / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        md = R / ("md-%s-s42-step281" % fam)
        md.mkdir(parents=True, exist_ok=True)
        (md / "rows.jsonl").write_text("")
        for key in ns["FAMILIES"][fam]["continued"]:
            record = ns["REC"] / ("recovery-%s%s.jsonl" % (key, ns["FAMILIES"][fam]["suffix"]))
            record.write_text("")


def _record_commands():
    """Execute the notebook's cells against a scratch PROJECT with the shell
    stubbed and run() recording; return [(script, args)]."""
    cells = ["".join(c["source"]) for c in _code_cells()]
    commands = []
    ns = {"__name__": "__main__"}

    class _Done:
        returncode = 0
        stdout = "GPU 0: stub"

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        (scratch / "home").mkdir()
        saved_env, saved_cwd, saved_run = dict(os.environ), os.getcwd(), subprocess.run
        os.environ.update({
            "PROJECT_DIR": str(scratch / "project"),
            "OPENAI_API_KEY": "test-key",
            "HF_TOKEN": "test-token",
            "HOME": str(scratch / "home"),
        })
        subprocess.run = lambda argv, *args, **kwargs: _Done()
        os.chdir(REPO)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cells[0], "cell-1", "exec"), ns)
                exec(compile(cells[1], "cell-2", "exec"), ns)
                assert ns["REPO"] == REPO and ns["PROJECT"] == (scratch / "project").resolve()
                ns["run"] = lambda script, *args, capture=None: commands.append(
                    (script, [str(a) for a in args]))
                _seed_project(ns)
                for index, source in enumerate(cells[2:], start=3):
                    exec(compile(source, "cell-%d" % index, "exec"), ns)
        finally:
            subprocess.run = saved_run
            os.chdir(saved_cwd)
            os.environ.clear()
            os.environ.update(saved_env)
    return commands


def test_notebook_commands_parse():
    commands = _record_commands()
    assert len(commands) > 100, len(commands)
    parsers = {}
    failures = []
    for script, args in commands:
        if script not in parsers:
            path = REPO / "scripts" / script
            spec = importlib.util.spec_from_file_location("notebook_" + path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parsers[script] = module.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parsers[script].parse_args(args)
            except SystemExit:
                failures.append((script, args))
    assert not failures, failures
    # Every script but the local smoke test is a notebook step.
    expected = {p.name for p in (REPO / "scripts").glob("*.py")} - {"smoke_test.py"}
    assert set(parsers) == expected, (sorted(expected - set(parsers)), sorted(set(parsers) - expected))


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=3))
