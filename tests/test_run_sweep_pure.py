"""Dependency-free tests for scripts/run_sweep.py's argument handling.

    python3 tests/test_run_sweep_pure.py

The sweep driver itself needs the ML stack (tests/test_sweepdriver.py);
what is checked here is everything before a model is touched: the layer
chunk syntax, the DEV calibration's fixed identity, the research sweep's
required flags and the candidate-benchmark guard.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import run_suite  # noqa: E402

DEV = "dev/tiny-model"


def _script():
    spec = importlib.util.spec_from_file_location("run_sweep_script", REPO / "scripts" / "run_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(script, argv):
    parser = script.build_parser()
    return script.validate_args(parser, parser.parse_args(argv), DEV)


def _refused(script, argv, wording):
    parser = script.build_parser()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            script.validate_args(parser, parser.parse_args(argv), DEV)
        except SystemExit as exc:
            assert exc.code == 2, exc.code
            assert wording in stderr.getvalue(), (wording, stderr.getvalue())
            return
    raise AssertionError("accepted: %r" % (argv,))


def test_parse_layer_chunk_forms():
    script = _script()
    assert script._parse_layer_chunk("all") is None
    assert script._parse_layer_chunk("0-3") == [0, 1, 2, 3]
    assert script._parse_layer_chunk("7,13,26") == [7, 13, 26]
    assert script._parse_layer_chunk(" 5 ") == [5]


def test_dev_calibration_fixes_its_identity():
    script = _script()
    args = _validate(script, ["--dev-calibration"])
    assert args.model_id == DEV and args.quant == "none"
    assert args.out_root == "results/dev-jsd-calibration" and args.run_tag == "dev-jsd"
    # Explicit values that match are accepted; contradictions refuse by name.
    args = _validate(script, ["--dev-calibration", "--model-id", DEV, "--out-root", "/x", "--run-tag", "t"])
    assert args.out_root == "/x" and args.run_tag == "t"
    _refused(script, ["--dev-calibration", "--model-id", "Qwen/Qwen2.5-7B-Instruct"], "drop --model-id")
    _refused(script, ["--dev-calibration", "--adapter", "/a"], "drop --adapter")
    _refused(script, ["--dev-calibration", "--benchmarks-only"], "cannot run candidate benchmarks")


def test_research_sweep_requires_its_identity_flags():
    script = _script()
    _refused(script, ["--model-id", "m"], "--out-root, --run-tag")
    _refused(script, [], "--model-id, --out-root, --run-tag")
    args = _validate(script, ["--model-id", "m", "--out-root", "/r", "--run-tag", "md", "--arm", "E,D"])
    assert args.arm == "E,D" and args.layers == "all" and args.n == 100
    _refused(script, ["--model-id", "m", "--out-root", "/r", "--run-tag", "md", "--benchmarks-only"],
             "explicit candidate list")
    args = _validate(script, ["--model-id", "m", "--out-root", "/r", "--run-tag", "md",
                              "--benchmarks-only", "--layers", "7,13,26"])
    assert script._parse_layer_chunk(args.layers) == [7, 13, 26]


def test_llm_flags_are_the_shared_group():
    script = _script()
    args = script.build_parser().parse_args(
        ["--model-id", "m", "--out-root", "/r", "--run-tag", "md", "--llm-fallback", "--llm-cache-dir", "/c"])
    assert args.llm_fallback and args.llm_provider == "openai" and args.llm_model == "gpt-5-mini"
    assert args.llm_cache_dir == "/c"


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=4))
