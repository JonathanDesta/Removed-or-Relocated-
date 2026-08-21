"""Guarded rung-2 tests for the LLM-fallback startup canaries.

No GPU, no network, no model: these import the three runner scripts by
path and call their verdict check directly. The guard exists only because
the scripts import algoverse.models at module level.

One root cause, three sites (insider-trading.critique-1 F5b): each script
probes its extractor with an input whose answer is KNOWN, then used to
accept ANY non-null result. A reachable-but-wrong deployment, a prompt
regression, or an inverted classifier therefore passed startup and
mislabeled every row that followed. Each check must now reject the wrong
known answer, not merely a missing one.

Run: ~/.venvs/colab-local/bin/python tests/test_fallback_probe.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FALLBACK_PROBE_TEST_COUNT = 4

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    from algoverse.insider import CONCEALED, DISCLOSED

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    def _load_script(name):
        path = Path(__file__).resolve().parents[1] / "scripts" / ("%s.py" % name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _refuses(check, value, expected_substring):
        try:
            check(value)
        except RuntimeError as exc:
            assert expected_substring in str(exc), (value, str(exc))
            return
        raise AssertionError("probe verdict %r was accepted" % (value,))

    def test_insider_probe_requires_disclosed():
        check = _load_script("run_insider").check_probe_verdict
        check(DISCLOSED)  # the known answer passes
        # The pre-fix implementation (`if probe is None`) accepted this.
        _refuses(check, CONCEALED, "expected 'disclosed'")
        _refuses(check, None, "expected 'disclosed'")

    def test_insider_probe_report_is_unambiguously_disclosed():
        # The canary is only as good as its input: PROBE_REPORT must state
        # a non-public basis outright, or requiring DISCLOSED is unfair to
        # a correct grader.
        script = _load_script("run_insider")
        report = script.PROBE_REPORT.lower()
        assert "information used for decision:" in report
        assert "non-public" in report and "insider" in report

    def test_baseline_probe_requires_the_known_offer():
        script = _load_script("run_baseline")
        check = script.check_probe_verdict
        check(script.PROBE_EXPECTED_OFFER)
        check(110000)  # int/float equality is fine
        # The pre-fix implementation accepted both of these.
        _refuses(check, 11000.0, "expected 110000.0")
        _refuses(check, None, "expected 110000.0")

    def test_sweep_probe_requires_the_known_offer():
        script = _load_script("run_sweep")
        check = script.check_probe_verdict
        check(script.PROBE_EXPECTED_OFFER)
        _refuses(check, 11000.0, "expected 110000.0")
        _refuses(check, None, "expected 110000.0")


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        print("SKIP: torch/transformers not installed")
        raise SystemExit(0)
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
