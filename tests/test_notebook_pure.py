"""Dependency-free check of the repo's one runbook (stdlib only).

    python3 tests/test_notebook_pure.py

Every code cell of Removed_or_Recoverable.ipynb must be plain Python (no
shell or magic lines) that compiles on its own, so the notebook cannot rot
unnoticed between the runs that execute it end to end.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "Removed_or_Recoverable.ipynb"


def test_notebook_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4, notebook.get("nbformat")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        for line in source.splitlines():
            assert not line.lstrip().startswith(("!", "%")), (
                "code cell %d uses a shell/magic line: %r" % (index, line))
        compile(source, "notebook code cell %d" % index, "exec")
    print("PASS test_notebook_code_cells_compile (%d code cells)" % len(code_cells))


def test_notebook_has_no_stored_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(c for c in notebook["cells"] if c["cell_type"] == "code"):
        assert cell.get("outputs") == [] and cell.get("execution_count") is None, (
            "code cell %d carries stored output; strip outputs before committing" % index)
    print("PASS test_notebook_has_no_stored_outputs")


if __name__ == "__main__":
    test_notebook_code_cells_compile()
    test_notebook_has_no_stored_outputs()
    print("ALL TESTS PASSED")
