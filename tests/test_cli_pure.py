"""Dependency-free tests for algoverse.cli, the scripts' shared plumbing.

    python3 tests/test_cli_pure.py

Covers K=PATH parsing, the results/ output guard, checkpoint-sidecar
adoption with its no-sidecar warning, the grader startup canary (package
and key checks, the known-answer verdict), the per-project cache default,
run_main's exit status, and that every script's parser builds on bare
python (what the notebook's dry run relies on).
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import cli, insider, tasks


def _raises(call, exc_type, wording):
    try:
        call()
    except exc_type as exc:
        assert wording in str(exc), (wording, str(exc))
        return exc
    raise AssertionError("expected %s containing %r" % (exc_type.__name__, wording))


def test_parse_pairs_keys_and_refusals():
    assert cli.parse_pairs(None, "--rows") == {}
    assert cli.parse_pairs(["M_0=/a", "M_D=/b"], "--rows") == {"M_0": "/a", "M_D": "/b"}
    assert cli.parse_pairs(["7=/l7", "13=/l13"], "--layer", key=int) == {7: "/l7", 13: "/l13"}
    merged = cli.parse_pairs(
        ["base=/c1", "base=/c2", "7=/l7"], "--competence",
        key=int, allow_base=True, merge=True,
    )
    assert merged == {"base": ["/c1", "/c2"], 7: ["/l7"]}
    # A path may contain "=" after the first separator.
    assert cli.parse_pairs(["k=/p/a=b"], "--x") == {"k": "/p/a=b"}
    _raises(lambda: cli.parse_pairs(["nopath"], "--rows"), ValueError,
            "--rows expects NAME=PATH")
    _raises(lambda: cli.parse_pairs(["=/a"], "--rows"), ValueError,
            "--rows expects NAME=PATH")
    _raises(lambda: cli.parse_pairs(["x=/a"], "--layer", key=int), ValueError,
            "integer layer")
    _raises(lambda: cli.parse_pairs(["base=/a"], "--layer", key=int), ValueError,
            "integer layer")
    _raises(lambda: cli.parse_pairs(["7=/a", "7=/b"], "--layer", key=int),
            ValueError, "'7' given twice")


def test_parse_pairs_reports_through_the_parser_when_given():
    parser = argparse.ArgumentParser(prog="x")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            cli.parse_pairs(["bad"], "--rows", parser=parser)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("parser.error did not exit")
    assert "--rows expects NAME=PATH" in stderr.getvalue()


def test_refuse_under_results_looks_at_every_ancestor():
    cli.refuse_under_results(None, "--out")
    cli.refuse_under_results("/tmp/reports/x.txt", "--out")
    for bad in ("results/x.txt", "/some/project/results/sub/x.txt",
                str(Path.cwd() / "results")):
        _raises(lambda: cli.refuse_under_results(bad, "--out"), SystemExit,
                "--out must not be under results/")
    parser = argparse.ArgumentParser(prog="x")
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            cli.refuse_under_results("results/x", "--save-fit", parser)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("parser refusal did not exit")


def test_adopt_checkpoint_flags_paths_and_warning():
    # No adapter, and an adapter without a sidecar: pass-through.
    assert cli.adopt_checkpoint_flags(None, None, None) == (None, None, False)
    with tempfile.TemporaryDirectory() as tmp:
        assert cli.adopt_checkpoint_flags(tmp, 7, 42) == (7, 42, False)
        # An externally produced adapter (adapter_config.json, no sidecar)
        # with an omitted value warns and records null.
        (Path(tmp) / "adapter_config.json").write_text("{}")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cli.adopt_checkpoint_flags(tmp, None, 42) == (None, 42, False)
        assert "WARNING" in out.getvalue()
        assert "checkpoint_step will be recorded as null" in out.getvalue()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cli.adopt_checkpoint_flags(tmp, 7, 42) == (7, 42, False)
        assert out.getvalue() == ""
        sidecar = Path(tmp) / "train_meta.json"
        sidecar.write_text(json.dumps({"checkpoint_step": 281, "train_seed": 42}))
        # Adoption of omitted values, printed.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert cli.adopt_checkpoint_flags(tmp, None, None) == (281, 42, True)
        assert "CHECKPOINT STEP adopted" in out.getvalue()
        assert "TRAIN SEED adopted" in out.getvalue()
        # Matching passed values are accepted.
        assert cli.adopt_checkpoint_flags(tmp, 281, 42) == (281, 42, True)
        # Contradictions refuse by name.
        for step, seed, wording in (
            (280, 42, "checkpoint-step"),
            (281, 43, "train-seed"),
        ):
            _raises(lambda: cli.adopt_checkpoint_flags(tmp, step, seed),
                    RuntimeError, wording)
        # A malformed sidecar is refused by checkpoint_meta's named error.
        sidecar.write_text(json.dumps({"checkpoint_step": 281}))
        _raises(lambda: cli.adopt_checkpoint_flags(tmp, None, None),
                ValueError, "train_seed")


def test_check_probe_verdict_requires_the_known_answer():
    cli.check_probe_verdict(tasks.PROBE_EXPECTED_OFFER, tasks.PROBE_EXPECTED_OFFER,
                            "the reply")
    cli.check_probe_verdict(110000, tasks.PROBE_EXPECTED_OFFER, "the reply")
    for wrong in (11000.0, None):
        _raises(lambda: cli.check_probe_verdict(wrong, tasks.PROBE_EXPECTED_OFFER,
                                                "the reply"),
                RuntimeError, "expected 110000.0")
    cli.check_probe_verdict(insider.DISCLOSED, insider.PROBE_EXPECTED_BASIS,
                            "the report")
    for wrong in (insider.CONCEALED, None):
        _raises(lambda: cli.check_probe_verdict(wrong, insider.PROBE_EXPECTED_BASIS,
                                                "the report"),
                RuntimeError, "expected 'disclosed'")


def test_probe_inputs_carry_their_known_answers():
    # The canary is only as good as its input: PROBE_REPORT must state a
    # non-public basis outright, or requiring DISCLOSED is unfair to a
    # correct grader; PROBE_REPLY must state the offer it expects back.
    report = insider.PROBE_REPORT.lower()
    assert "information used for decision:" in report
    assert "non-public" in report and "insider" in report
    assert insider.PROBE_EXPECTED_BASIS == insider.DISCLOSED
    assert "one hundred and ten thousand" in tasks.PROBE_REPLY
    assert tasks.PROBE_EXPECTED_OFFER == 110000.0


class _fake_provider:
    """A stub package module and an API key for one provider."""

    def __init__(self, provider, key_value="k"):
        self.package, self.key = cli.LLM_PROVIDERS[provider]
        self.key_value = key_value

    def __enter__(self):
        self.saved_module = sys.modules.get(self.package)
        self.saved_key = os.environ.get(self.key)
        sys.modules[self.package] = types.SimpleNamespace()
        if self.key_value is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.key_value
        return self

    def __exit__(self, *exc):
        if self.saved_module is None:
            sys.modules.pop(self.package, None)
        else:
            sys.modules[self.package] = self.saved_module
        if self.saved_key is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.saved_key
        return False


def test_verify_llm_fallback_checks_setup_then_the_canary():
    _raises(lambda: cli.check_provider_setup("azure"), RuntimeError,
            "unsupported --llm-provider")
    with _fake_provider("openai", key_value=None):
        _raises(lambda: cli.check_provider_setup("openai"), RuntimeError,
                "requires OPENAI_API_KEY")
    calls = []

    def probe_fn(text, provider, model, cache_dir, raise_errors):
        calls.append((text, provider, model, raise_errors, Path(cache_dir).is_dir()))
        return 110000.0

    with _fake_provider("openai"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            probe = cli.verify_llm_fallback(
                "openai", "m", probe_fn, tasks.PROBE_REPLY,
                tasks.PROBE_EXPECTED_OFFER, "the reply",
            )
        assert probe == 110000.0
        assert calls == [(tasks.PROBE_REPLY, "openai", "m", True, True)]
        assert "LLM FALLBACK VERIFIED: openai/m" in out.getvalue()
        # A wrong known answer refuses; a raising probe is wrapped with its
        # cause, before any generation.
        _raises(lambda: cli.verify_llm_fallback(
                    "openai", "m", lambda *a, **k: 11000.0, tasks.PROBE_REPLY,
                    tasks.PROBE_EXPECTED_OFFER, "the reply"),
                RuntimeError, "expected 110000.0")

        def failing(*args, **kwargs):
            raise ConnectionError("no route")

        _raises(lambda: cli.verify_llm_fallback(
                    "openai", "m", failing, tasks.PROBE_REPLY,
                    tasks.PROBE_EXPECTED_OFFER, "the reply"),
                RuntimeError, "ConnectionError: no route")


def test_default_llm_cache_dir_is_per_project():
    assert cli.default_llm_cache_dir("/p/results/run") == "/p/.cache/llm_extractions"
    assert cli.default_llm_cache_dir(Path("/p/results/sweep-x")) == (
        "/p/.cache/llm_extractions")


def test_run_main_exits_with_the_status():
    for code in (0, 3):
        try:
            cli.run_main(lambda: code)
        except SystemExit as exc:
            assert exc.code == code
        else:
            raise AssertionError("run_main did not exit")

    def interrupted():
        raise KeyboardInterrupt

    try:
        cli.run_main(interrupted)
    except SystemExit as exc:
        assert exc.code == 130
    else:
        raise AssertionError("run_main did not exit on KeyboardInterrupt")


def test_every_script_builds_its_parser_on_bare_python():
    # The scripts' parsers import without the ML stack, so a command line
    # can be validated (the notebook's dry run) without torch installed.
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    names = []
    for path in sorted(scripts_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location("script_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(getattr(module, "build_parser", None)), path.name
        assert callable(getattr(module, "main", None)), path.name
        assert module.build_parser().format_help(), path.name
        names.append(path.stem)
    assert "run_baseline" in names and "make_figures" in names


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
