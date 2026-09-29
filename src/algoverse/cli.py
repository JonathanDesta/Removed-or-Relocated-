"""Shared command-line plumbing for the scripts (stdlib only).

Every script builds its parser in build_parser(), does its work in
main(argv=None) -> int and ends with cli.run_main(main), so a test or the
notebook's dry run can parse a command line without executing it. The
helpers here are the pieces several scripts used to carry as private
copies: the grader flags and their startup canary, checkpoint-sidecar
adoption, model loading, the results/ output guard and K=PATH parsing.
"""
import importlib
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# The LLM grader fallback
# ---------------------------------------------------------------------------

LLM_PROVIDERS = {
    "openai": ("openai", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
}


def add_llm_flags(parser, purpose="extraction", out_flag="out-dir"):
    """--llm-fallback, --llm-provider, --llm-model and --llm-cache-dir."""
    group = parser.add_argument_group("LLM grader fallback")
    group.add_argument("--llm-fallback", action="store_true",
                       help="enable the LLM %s fallback (needs an API key)"
                            % purpose)
    group.add_argument("--llm-provider", default="openai")
    group.add_argument("--llm-model", default="gpt-5-mini")
    group.add_argument("--llm-cache-dir", default=None, metavar="DIR",
                       help="disk cache for grader calls; default "
                            "<%s>/../../.cache/llm_extractions, one cache "
                            "per project directory" % out_flag)
    return parser


def default_llm_cache_dir(out_dir):
    """<out_dir>/../../.cache/llm_extractions: one cache per project directory
    (out_dir is <project>/results/<run>)."""
    return str(Path(out_dir).parent.parent / ".cache" / "llm_extractions")


def check_provider_setup(provider):
    """Refuse a provider whose package or API key is missing."""
    if provider not in LLM_PROVIDERS:
        raise RuntimeError("unsupported --llm-provider %r" % provider)
    package, key = LLM_PROVIDERS[provider]
    try:
        importlib.import_module(package)
    except ImportError as exc:
        raise RuntimeError(
            "--llm-fallback requires the %s package" % package
        ) from exc
    if not os.environ.get(key):
        raise RuntimeError("--llm-fallback with %s requires %s" % (provider, key))


def check_probe_verdict(probe, expected, label):
    """Refuse unless the canary returned the answer it is known to have.

    Accepting any non-null answer would let a wrong deployment, a prompt
    regression or an inverted classifier pass startup and then mislabel
    every reply: a canary with a known answer that accepts any answer is
    not a canary.
    """
    if probe != expected:
        raise RuntimeError(
            "LLM fallback startup probe returned %r for %s; expected %r. "
            "No generation was run." % (probe, label, expected)
        )


def verify_llm_fallback(provider, model, probe_fn, probe_input, expected,
                        label):
    """Fail fast, before any generation, unless the grader works end to end.

    Checks the provider's package and API key, then runs probe_fn (the
    grader itself: tasks.llm_extract_offer or insider.llm_classify_report)
    on probe_input with errors raised and a throwaway cache, and refuses
    unless the verdict equals the known answer. Prints one line on success.
    """
    check_provider_setup(provider)
    try:
        with tempfile.TemporaryDirectory() as probe_cache:
            probe = probe_fn(
                probe_input, provider=provider, model=model,
                cache_dir=probe_cache, raise_errors=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "LLM fallback startup probe failed before generation "
            "(%s: %s)" % (type(exc).__name__, exc)
        ) from exc
    check_probe_verdict(probe, expected, label)
    print("LLM FALLBACK VERIFIED: %s/%s" % (provider, model))
    return probe


# ---------------------------------------------------------------------------
# Checkpoints and models
# ---------------------------------------------------------------------------


def adopt_checkpoint_flags(adapter, checkpoint_step, train_seed):
    """Adopt (checkpoint_step, train_seed) from a checkpoint's sidecar.

    A checkpoint this project trained carries a train_meta.json sidecar, so
    its provenance is read rather than operator-copied on trust
    (train.adopt_checkpoint_identity: a None value is adopted and printed,
    a contradicting value refuses by name). An adapter without one is
    externally produced: omitted values are recorded as null, with a
    warning. Returns (checkpoint_step, train_seed, has_sidecar).
    """
    from algoverse.train import adopt_checkpoint_identity

    if (
        adapter is not None
        and (Path(adapter) / "adapter_config.json").is_file()
        and not (Path(adapter) / "train_meta.json").is_file()
        and (checkpoint_step is None or train_seed is None)
    ):
        omitted = []
        if checkpoint_step is None:
            omitted.append("checkpoint_step")
        if train_seed is None:
            omitted.append("train_seed")
        print(
            "WARNING: adapter %s has adapter_config.json but no "
            "train_meta.json; %s will be recorded as null. "
            "A project-trained checkpoint should carry its sidecar."
            % (adapter, " and ".join(omitted))
        )
    return adopt_checkpoint_identity(adapter, checkpoint_step, train_seed)


def load_eval_model(model_id, adapter, quant, has_sidecar):
    """(model, tokenizer) for evaluation.

    A project checkpoint (has_sidecar) loads through load_checkpoint_model
    so its sidecar is validated; anything else through
    load_model_and_tokenizer. The torch stack is imported here, not at the
    caller's module level, so the caller's parser imports on bare python.
    """
    from algoverse.models import load_checkpoint_model, load_model_and_tokenizer

    if has_sidecar:
        model, tokenizer, _meta = load_checkpoint_model(
            model_id, adapter, quant=quant
        )
        return model, tokenizer
    return load_model_and_tokenizer(model_id, quant=quant, adapter_path=adapter)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def refuse_under_results(path, flag, parser=None):
    """Refuse an output path under any directory named results/.

    results/ holds JSONL model outputs only; reports, fits and scratch
    spools live elsewhere. The rule looks at every ancestor, so a path
    under some other project's results/ is refused too. parser.error when
    a parser is given, else SystemExit with the same message.
    """
    if path is None:
        return
    resolved = Path(path).resolve()
    if any(node.name == "results" for node in (resolved, *resolved.parents)):
        message = (
            "%s must not be under results/ (results are JSONL model "
            "outputs only)" % flag
        )
        if parser is not None:
            parser.error(message)
        raise SystemExit(message)


def parse_pairs(values, label, key=str, allow_base=False, merge=False,
                parser=None):
    """{key: path} from repeated "K=PATH" arguments.

    key converts K (int for layer indices); allow_base accepts the literal
    "base" as its own key; merge collects several paths per key into a
    list instead of refusing a repeat. A bad argument is parser.error when
    a parser is given, else a ValueError naming the flag.
    """
    def fail(message):
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    shape = "N" if key is int else "NAME"
    result = {}
    for value in values or []:
        raw, separator, path = value.partition("=")
        if not separator or not raw or not path:
            fail("%s expects %s=PATH, got %r" % (label, shape, value))
        if allow_base and raw == "base":
            norm = "base"
        else:
            try:
                norm = key(raw)
            except (TypeError, ValueError):
                fail("%s expects an integer layer%s in %r"
                     % (label, " or 'base'" if allow_base else "", value))
        if merge:
            result.setdefault(norm, []).append(path)
        elif norm in result:
            fail("%s %r given twice" % (label, raw))
        else:
            result[norm] = path
    return result


def run_main(main):
    """Run a script's main(argv=None) -> int and exit with its status."""
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)
