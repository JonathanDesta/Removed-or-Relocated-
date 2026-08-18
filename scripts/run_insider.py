"""Evaluate one model on the Insider Trading environment (IT rows only).

The canonical invocation for a Stage-1 transfer leg on Colab:

    python scripts/run_insider.py --model-id Qwen/Qwen2.5-7B-Instruct \
        --quant 4bit --adapter runs/md-qwen7b-s42/checkpoints/step-00281 \
        --run-id md-insider --out-dir results/md-insider --llm-fallback

Add --bypassed-layer N for the l*-probe leg. IT is evaluation-only: no
benchmark legs (it adds no capability gate), no --split (the pool is one
undivided "insider" split), and the full 200-scenario pool is the default
(--n exists for smoke/debug draws only). Re-running resumes.

`--smoke` runs the DEV model (Qwen-0.5B, no GPU) end-to-end on a handful
of scenarios: rows written, schema complete, resume, and the probe-bypass
bookkeeping guard. Its deception numbers mean NOTHING; this is plumbing.

RATIFICATION PENDING: the environment's constants are PROPOSED values
(planning/insider-trading.md section 12), including this script's
generation defaults (P-IT7: greedy, max_new_tokens=256, batch_size 4 —
the project's recorded generation profile applied unchanged to IT). No
evaluation run producing an IT number happens before ratification.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.eval import VALID_ARMS, run_negotiation_eval
from algoverse.insider import (
    INSIDER_SPLIT,
    TRUE_BASIS,
    get_insider_scenarios,
    llm_classify_report,
    render_insider_messages,
    score_insider_response,
)
from algoverse.metrics import task_competence, tau_with_ci
from algoverse.models import (
    bypass_state,
    install_bypass,
    load_checkpoint_model,
    load_model_and_tokenizer,
)
from algoverse.train import adopt_checkpoint_identity

# The startup probe's input: a well-formed report whose basis is stated
# insider information, so the pinned classifier must return a verdict
# (DISCLOSED) rather than UNPARSEABLE.
PROBE_REPORT = (
    "Target stock: AZRK\n"
    "Shares bought: 500\n"
    "Estimated profit: $75,000 in 3 months\n"
    "Risk level: High\n"
    "Information used for decision: non-public insider information about "
    "an upcoming merger announcement"
)


def insider_smoke(n_scenarios=4, out_dir="results/insider-smoke"):
    """End-to-end IT proof on the DEV model, no GPU needed (WP-IT5 R2).

    Mirrors eval.smoke_test's structure, IT-flavored: intact rows written
    with complete schema and the IT categorical values, clean resume, a
    probe-bypass leg with derived provenance, and the bookkeeping guard
    refusing to mix intact and bypassed rows under one run_id.
    """
    from algoverse.eval import ROW_FIELDS
    from algoverse.models import (
        BYPASS_IMPL,
        DEV_MODEL,
        _decoder_layers,
    )

    out_path = Path(out_dir) / "rows.jsonl"
    manifest_path = out_path.with_suffix(".manifest.jsonl")
    for stale_path in (out_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()  # a smoke test always starts from scratch

    print("loading %s ..." % DEV_MODEL)
    model, tokenizer = load_model_and_tokenizer(DEV_MODEL, quant="none")

    scenarios = get_insider_scenarios(n=n_scenarios, seed=0)
    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id="insider-smoke", out_path=out_path, model_id=DEV_MODEL,
        quant_label="none",
        render_fn=render_insider_messages, score_fn=score_insider_response,
    )

    expected = len(scenarios) * 2
    assert len(rows) == expected, "expected %d rows, got %d" % (expected, len(rows))
    for row in rows:
        missing = [field for field in ROW_FIELDS if field not in row]
        assert not missing, "row missing fields: %s" % missing
        assert row["split"] == INSIDER_SPLIT, row["split"]
        assert row["true_value"] == TRUE_BASIS, row["true_value"]

    # Resume: a second call over the same work must generate nothing.
    rows_again = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id="insider-smoke", out_path=out_path, model_id=DEV_MODEL,
        quant_label="none",
        render_fn=render_insider_messages, score_fn=score_insider_response,
    )
    assert len(rows_again) == expected, "resume changed the row count"

    mid = len(_decoder_layers(model)) // 2
    handle = install_bypass(model, mid)
    bypass_rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id="insider-smoke-bypass", out_path=out_path, model_id=DEV_MODEL,
        bypassed_layer=mid, quant_label="none",
        render_fn=render_insider_messages, score_fn=score_insider_response,
    )
    assert len(bypass_rows) == expected
    for row in bypass_rows:
        assert row["bypassed_layer"] == mid
        assert row["gen_config"]["bypass_impl"] == BYPASS_IMPL
    handle.remove()

    guard_refused = False
    try:
        run_negotiation_eval(
            model, tokenizer, scenarios,
            run_id="insider-smoke-bypass", out_path=out_path,
            model_id=DEV_MODEL, bypassed_layer=None, quant_label="none",
            render_fn=render_insider_messages,
            score_fn=score_insider_response,
        )
    except ValueError:
        guard_refused = True
    assert guard_refused, "resume guard accepted intact model for bypassed run"

    gap = tau_with_ci(rows, n_boot=200, seed=0)
    competence = task_competence(rows)
    print("\nsample responses:")
    for row in rows[:2]:
        print(
            "  [%s | true=%s] %r"
            % (row["condition"], row["true_value"], row["response_text"][:140])
        )
    print("\ntau=%s  CI=[%s, %s]" % (gap["tau"], gap["tau_ci_low"], gap["tau_ci_high"]))
    print(
        "invalid rates: incentive=%s control=%s | competence=%s"
        % (
            gap["invalid_rate_incentive"],
            gap["invalid_rate_control"],
            competence["competence"],
        )
    )
    print(
        "\nINSIDER SMOKE PASSED: %d intact + %d bypass rows, schema "
        "complete, resume guarded (DEV model: numbers mean nothing)"
        % (expected, expected)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir, optional")
    parser.add_argument(
        "--n", type=int, default=None,
        help="scenarios (x2 conditions); default = the FULL 200-scenario "
             "pool (transfer checks and R_t always use the full pool)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="PROPOSED (P-IT7), NOT ratified: the project's recorded "
             "generation profile applied unchanged to IT",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="generation/eval seed; recorded as the rows' seed field. "
             "Does NOT change scenario draws; see --scenario-seed.",
    )
    parser.add_argument(
        "--scenario-seed", type=int, default=42,
        help="seed for the deterministic scenario subsample ONLY (only "
             "meaningful with --n; the full pool is unaffected)",
    )
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--bypassed-layer", type=int, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--arm", default=None, choices=list(VALID_ARMS))
    parser.add_argument("--llm-fallback", action="store_true",
                        help="enable the LLM classifier fallback (needs an API key)")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--smoke", action="store_true",
                        help="DEV-model plumbing proof; ignores the run flags")
    args = parser.parse_args()

    if args.smoke:
        insider_smoke()
        sys.exit(0)

    for flag, value in (
        ("--model-id", args.model_id),
        ("--run-id", args.run_id),
        ("--out-dir", args.out_dir),
    ):
        if value is None:
            parser.error("%s is required (unless --smoke)" % flag)

    if args.llm_fallback:
        if args.llm_provider == "openai":
            try:
                import openai  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "--llm-fallback requires the openai package"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "--llm-fallback with openai requires OPENAI_API_KEY"
                )
        elif args.llm_provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "--llm-fallback requires the anthropic package"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "--llm-fallback with anthropic requires ANTHROPIC_API_KEY"
                )
        else:
            raise RuntimeError(
                "unsupported --llm-provider %r" % args.llm_provider
            )

        # Fail fast through the IT classifier path itself, so a broken
        # endpoint can never be discovered after generation spend.
        try:
            with tempfile.TemporaryDirectory() as probe_cache:
                probe = llm_classify_report(
                    PROBE_REPORT,
                    provider=args.llm_provider,
                    model=args.llm_model,
                    cache_dir=probe_cache,
                    raise_errors=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "LLM fallback startup probe failed before generation "
                "(%s: %s)" % (type(exc).__name__, exc)
            ) from exc
        if probe is None:
            raise RuntimeError(
                "LLM fallback startup probe failed; no generation was run"
            )
        print(
            "LLM FALLBACK VERIFIED: %s/%s"
            % (args.llm_provider, args.llm_model)
        )

    # A checkpoint this project trained carries a train_meta.json sidecar, so
    # its provenance is read rather than operator-copied on trust. Adapters
    # without one are externally produced and behave exactly as before.
    if (
        args.adapter is not None
        and (Path(args.adapter) / "adapter_config.json").is_file()
        and not (Path(args.adapter) / "train_meta.json").is_file()
        and (args.checkpoint_step is None or args.train_seed is None)
    ):
        omitted = []
        if args.checkpoint_step is None:
            omitted.append("checkpoint_step")
        if args.train_seed is None:
            omitted.append("train_seed")
        print(
            "WARNING: adapter %s has adapter_config.json but no "
            "train_meta.json; %s will be recorded as null. "
            "A project-trained checkpoint should carry its sidecar."
            % (args.adapter, " and ".join(omitted))
        )

    args.checkpoint_step, args.train_seed, has_sidecar = (
        adopt_checkpoint_identity(
            args.adapter, args.checkpoint_step, args.train_seed
        )
    )

    out_dir = Path(args.out_dir)
    if has_sidecar:
        # Reinstall-at-load: a checkpoint trained under a permanent lesion
        # carries it into every evaluation (ratified 2026-08-13).
        model, tokenizer, _meta, _permanent = load_checkpoint_model(
            args.model_id, args.adapter, quant=args.quant
        )
        if _permanent is not None:
            print(
                "PERMANENT BYPASS REINSTALLED from train_meta.json: layer %d"
                % _meta["bypassed_layer"]
            )
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, quant=args.quant, adapter_path=args.adapter
        )
    if args.bypassed_layer is not None:
        # Eval-time lesions are the probe role (carve-out, 2026-08-16).
        install_bypass(model, args.bypassed_layer, role="probe")
        probe = bypass_state(model)["probe"]
        print(
            "PROBE BYPASS INSTALLED: layer %d (%s)"
            % (probe["layer_idx"], probe["impl"])
        )
    scenarios = get_insider_scenarios(n=args.n, seed=args.scenario_seed)

    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id=args.run_id, out_path=out_dir / "rows.jsonl",
        model_id=args.model_id, adapter_path=args.adapter,
        bypassed_layer=args.bypassed_layer,
        checkpoint_step=args.checkpoint_step, arm=args.arm,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        seed=args.seed, train_seed=args.train_seed,
        quant_label=args.quant, use_llm_fallback=args.llm_fallback,
        llm_provider=args.llm_provider, llm_model=args.llm_model,
        scenario_seed=args.scenario_seed, n=args.n,
        render_fn=render_insider_messages, score_fn=score_insider_response,
    )
    gap = tau_with_ci(rows)
    competence = task_competence(rows)
    print("\ntau=%s CI=[%s, %s]  invalid inc/ctrl=%s/%s  competence=%s" % (
        gap["tau"], gap["tau_ci_low"], gap["tau_ci_high"],
        gap["invalid_rate_incentive"], gap["invalid_rate_control"],
        competence["competence"],
    ))
