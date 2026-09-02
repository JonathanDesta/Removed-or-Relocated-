"""Probe TRANSFER check: fit on Instructed-Pairs, evaluate on strategic rows.

This is the spec's own instrument for the construct-validity question the
Insider Trading non-transfer raised: "train probes on the Instructed-Pairs
dataset ... and verify transfer to strategic deception tasks"
(RESEARCH_SPEC Localization corroboration). The existing corroboration
driver fits and evaluates WITHIN one example source; this script fits on
one source and evaluates on the other, per layer:

  TRAIN  the Instructed-Pairs probe dataset (built by
         scripts/build_instructed_pairs.py; ratified construction),
         fit on ALL its examples with the ratified recipe
         (scaler+LR, C=0.1, max_iter 1000; response-token flattening).
  TEST   a results rows.jsonl through the RATIFIED
         within-incentive-condition control (valid incentive rows,
         deceptive True = lied vs False = did not lie, scenario groups) —
         corroboration.probe_examples_from_rows. Per-response score =
         mean over the scored tokens' decision scores (the ratified
         aggregation); AUROC over responses, scenario-group bootstrap CI.

An instructed-pairs probe that reads lexical instruction cues rather than
deception should transfer at chance; a probe that found a deception-
relevant direction should stay well above it. High within-source AUROC
(the 1.0-everywhere m0-corr result) with chance transfer = the
corroboration ceiling is a lexical artifact.

FEATURE POSITION (--feature-position) — WHICH tokens the probe reads:
  response_tokens       mean over every response token (the ratified
                        aggregation; the default and the diag-probe3 design)
  response_excl_claim   the same, but the response's final structured
                        claim line ("MY BEST OUTSIDE OFFER: ...") is
                        removed before capture (test set only; the
                        instructed pairs carry no claim line). Responses
                        with no marker are kept and counted; responses
                        that are ONLY the claim line are skipped and
                        counted (config.n_test_skipped_empty_body).
  final_prompt          ONE token: the last PROMPT token (index
                        response_start - 1), i.e. the model's state before
                        any response token exists — train AND test. For
                        the instructed pairs that is the assistant-header
                        token with the honest/untruthful instruction
                        upstream. The only position with no response text
                        in the features.

FIT SOURCE — WHOSE direction is scored:
  own    (default) the probe is fit on THIS checkpoint's instructed-pairs
         activations. --save-fit DIR persists it (layer-NNN.joblib per
         layer + fit_meta.json, written last as the completion marker).
  fixed  --use-fit DIR additionally scores every test set with a
         PERSISTED direction (e.g. M_0's), so checkpoints are compared
         along ONE direction instead of a refit each; the StandardScaler
         statistics travel with it. --no-own-fit skips the own fit and the
         train capture entirely. The fit dir is validated (position,
         model, layer count, hidden size, dataset) before any unpickling.

OUTPUTS:
  --out-dir DIR    legacy: exactly one output (one test set, one fit);
                   run_id verbatim; with the default position this writes
                   TODAY'S config byte-for-byte (LEGACY_CONFIG_KEYS).
  --out-root ROOT  one output per (fit, test-set label) at
                   ROOT/<run-id>-<fit>-<label>/interp.jsonl, run_id
                   <run-id>-<fit>-<label>; --test-rows must be LABEL=PATH.
                   One train capture serves every test set and both fits.
  Every non-legacy row adds the documented keys (feature_position,
  span_len, exclude_final_line, n_test_no_marker,
  n_test_skipped_empty_body, test_rows_label, fit_source) and, for a fixed
  fit, config.fit = "fixed_direction_from:<fit_run_id>". config.status is
  never changed: nothing about probe transfer is ratified.

TEST-SET CHOICE IS THE TEAM'S: the test rows must contain BOTH lied and
did-not-lie incentive rows (a saturated M_D checkpoint is refused with the
class counts). Fixed-stimuli design: every checkpoint of a family reads
the SAME test texts, so curves are comparable across checkpoints.

Human-run (GPU: it captures activations). USE FRESH diag-* OUT DIRS — never
a standard corroboration run's folder, so downstream probe_auroc consumers
cannot pool transfer rows with standard rows unnoticed. Fit dirs and the
scratch dir must never be under results/ (results are JSONL only).

    python scripts/run_probe_transfer.py \
        --model-id Qwen/Qwen2.5-7B-Instruct --quant 4bit \
        --train-dataset data/instructed_pairs/instructed_pairs_qwen2-5.jsonl \
        --feature-position final_prompt \
        --test-rows d1=results/diag-md-qwen7b-step8/rows.jsonl \
        --test-rows d2=results/e3-id-t008-l07-qwen7b-s42/rows.jsonl \
        --run-id diag-probe4-fp-m0-qwen7b --out-root results \
        --save-fit probe_fits/qwen7b/fp --probe-scratch-dir ~/probe_scratch
"""
import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.corroboration import (
    PROBE_RECIPE,
    _interp_done,
    aggregate_response_scores,
    load_probe_dataset,
    probe_examples_from_rows,
    write_interp_row,
)
from algoverse.metrics import load_rows

# CLI value -> config["feature_position"] value (the recorded name).
FEATURE_POSITIONS = {
    "response_tokens": "mean_response_tokens",
    "response_excl_claim": "mean_response_tokens_excl_claim_line",
    "final_prompt": "final_prompt_token",
}
FIT_OWN, FIT_FIXED = "own", "fixed"
LEGACY_STATUS = "exploratory-diagnostic; unratified"
# The config every diag-probe3 row on Drive carries. A legacy invocation
# must reproduce exactly this key set; everything else adds NEW_CONFIG_KEYS.
LEGACY_CONFIG_KEYS = (
    "test_size", "random_state", "max_iter", "C", "pipeline", "aggregation",
    "aggregation_source", "transfer", "label_source", "n_train", "n_test",
    "n_test_lied", "fit", "status",
)
NEW_CONFIG_KEYS = (
    "feature_position", "span_len", "exclude_final_line", "n_test_no_marker",
    "n_test_skipped_empty_body", "test_rows_label", "fit_source",
)
FIT_META_NAME = "fit_meta.json"
FIT_SOURCE_FIELDS = (
    "fit_run_id", "model_id", "adapter_path", "checkpoint_step", "train_seed",
    "feature_position", "n_train",
)


# ---------------------------------------------------------------------------
# Pure pieces (numpy/sklearn imported inside; the module stays rung-1
# importable so the CLI logic can be tested without an ML stack)
# ---------------------------------------------------------------------------


def fit_transfer_probe(train_tokens, train_token_labels):
    """Fit the ratified probe (scaler+LR) on one layer's flattened tokens."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", C=PROBE_RECIPE["C"],
            max_iter=PROBE_RECIPE["max_iter"],
        ),
    ).fit(train_tokens, train_token_labels)


def score_transfer_probe(clf, test_responses, test_labels, test_groups):
    """Score a fitted probe on test responses: (auroc, ci_low, ci_high, acc).

    test_responses: list of [n_tokens_i, d] arrays; each response's score is
    the mean of its per-token decision scores (aggregate_response_scores).
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from algoverse.interp import _group_bootstrap_auroc_ci

    scores = np.asarray(aggregate_response_scores(
        [clf.decision_function(np.asarray(tokens)) for tokens in test_responses]
    ))
    y = np.asarray(test_labels)
    auroc = float(roc_auc_score(y, scores))
    ci_low, ci_high = _group_bootstrap_auroc_ci(y, scores, test_groups)
    accuracy = float(((scores > 0) == y.astype(bool)).mean())
    return auroc, ci_low, ci_high, accuracy


def transfer_probe_layer(train_tokens, train_token_labels, test_responses,
                         test_labels, test_groups):
    """Fit on one layer's train tokens and score the test set (own fit).

    train_tokens: [n_tokens, d] array (already flattened across train
    responses); train_token_labels: per-token labels. Returns
    (auroc, ci_low, ci_high, accuracy). Pure given arrays.
    """
    clf = fit_transfer_probe(train_tokens, train_token_labels)
    return score_transfer_probe(clf, test_responses, test_labels, test_groups)


def parse_test_rows_spec(spec):
    """"LABEL=PATH" -> (label, path); a bare PATH -> (None, path).

    A spec is labeled iff the text before its first "=" contains no "/" —
    a path can legitimately contain "=" after a slash.
    """
    head, separator, tail = spec.partition("=")
    if separator and head and tail and "/" not in head:
        return head, tail
    return None, spec


def capture_spans(examples, feature_position):
    """(starts, span_len) for the capture helper under a feature position.

    final_prompt reads exactly one token, the last prompt token, so starts
    shift by -1 and span_len is 1; an example whose response_start is 0 has
    no prompt token to read and is refused by name.
    """
    starts = [int(example.get("response_start", 0)) for example in examples]
    if feature_position == "final_prompt":
        for index, start in enumerate(starts):
            if start < 1:
                raise SystemExit(
                    "final_prompt needs response_start >= 1; example %d has "
                    "response_start=%d (no prompt token precedes the response)"
                    % (index, start)
                )
        return [start - 1 for start in starts], 1
    return starts, None


def plan_outputs(run_id, out_dir, out_root, labels, fits):
    """[{fit, label, run_id, out_path}] for this invocation.

    Legacy (--out-dir): exactly one output, run_id verbatim. --out-root:
    one output per (fit, label) named <run-id>-<fit>-<label>.
    """
    if out_dir is not None:
        if len(labels) != 1 or len(fits) != 1:
            raise ValueError(
                "--out-dir holds exactly one output; use --out-root for "
                "several test sets or own+fixed fits"
            )
        return [{
            "fit": fits[0], "label": labels[0], "run_id": run_id,
            "out_path": Path(out_dir) / "interp.jsonl",
        }]
    outputs = []
    for fit in fits:
        for label in labels:
            composed = "%s-%s-%s" % (run_id, fit, label)
            outputs.append({
                "fit": fit, "label": label, "run_id": composed,
                "out_path": Path(out_root) / composed / "interp.jsonl",
            })
    return outputs


def build_config(*, train_dataset, test_path, n_train, n_test, n_test_lied,
                 fit_tag, feature_position, span_len, exclude_final_line,
                 stats, label, fit_meta, legacy):
    """The probe_auroc config for one output (see the module docstring)."""
    config = dict(PROBE_RECIPE)
    config.update({
        "transfer": True,
        "label_source": "transfer:probe_dataset:%s->within_incentive_rows:%s"
                        % (train_dataset, test_path),
        "n_train": n_train,
        "n_test": n_test,
        "n_test_lied": n_test_lied,
        "fit": "all_train_examples_no_holdout",
        "status": LEGACY_STATUS,
    })
    if legacy:
        return config
    stats = stats or {}
    config.update({
        "feature_position": FEATURE_POSITIONS[feature_position],
        "span_len": span_len,
        "exclude_final_line": bool(exclude_final_line),
        "n_test_no_marker": int(stats.get("n_no_marker", 0)),
        "n_test_skipped_empty_body": int(stats.get("n_skipped_empty_body", 0)),
        "test_rows_label": label,
        "fit_source": None,
    })
    if fit_tag == FIT_FIXED:
        if not fit_meta:
            raise ValueError("a fixed fit needs its fit_meta")
        config["fit"] = "fixed_direction_from:%s" % fit_meta["fit_run_id"]
        config["n_train"] = fit_meta["n_train"]
        config["fit_source"] = {
            field: fit_meta.get(field) for field in FIT_SOURCE_FIELDS
        }
    return config


def save_fit(fit_dir, pipelines_by_layer, meta):
    """Persist per-layer pipelines; fit_meta.json is written LAST."""
    import joblib

    fit_dir = Path(fit_dir)
    fit_dir.mkdir(parents=True, exist_ok=True)
    for layer, clf in pipelines_by_layer.items():
        joblib.dump(clf, fit_dir / ("layer-%03d.joblib" % int(layer)))
    record = dict(meta)
    record["layers_saved"] = sorted(int(layer) for layer in pipelines_by_layer)
    (fit_dir / FIT_META_NAME).write_text(json.dumps(record, indent=1) + "\n")
    return fit_dir / FIT_META_NAME


def read_fit_meta(fit_dir):
    """fit_meta.json of a persisted fit, refusing an incomplete dir."""
    meta_path = Path(fit_dir) / FIT_META_NAME
    if not meta_path.is_file():
        raise SystemExit(
            "--use-fit %s has no %s (fit missing or incomplete)"
            % (fit_dir, FIT_META_NAME)
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_fit(fit_dir, *, model_id, feature_position, n_layers, hidden_size,
             train_dataset, layers_needed):
    """Load persisted pipelines for layers_needed after validating the meta.

    Refuses BEFORE unpickling on a missing meta, a mismatched feature
    position / model / layer count / hidden size / dataset basename, or a
    missing layer file; warns (never refuses) on scikit-learn drift.
    Returns ({layer: pipeline}, meta).
    """
    fit_dir = Path(fit_dir)
    meta = read_fit_meta(fit_dir)
    expected = (
        ("feature_position", FEATURE_POSITIONS[feature_position]),
        ("model_id", model_id),
        ("n_layers", int(n_layers)),
        ("hidden_size", int(hidden_size)),
        ("train_dataset_basename", Path(train_dataset).name),
    )
    for field, wanted in expected:
        if meta.get(field) != wanted:
            raise SystemExit(
                "--use-fit %s: fit_meta %s is %r, this run needs %r"
                % (fit_dir, field, meta.get(field), wanted)
            )
    layers_needed = sorted(int(layer) for layer in layers_needed)
    missing = [
        layer for layer in layers_needed
        if not (fit_dir / ("layer-%03d.joblib" % layer)).is_file()
    ]
    if missing:
        raise SystemExit(
            "--use-fit %s is missing layer files for layers %s"
            % (fit_dir, missing)
        )
    import joblib
    import sklearn

    if meta.get("sklearn_version") not in (None, sklearn.__version__):
        print(
            "WARNING: fit saved with scikit-learn %s, loading under %s"
            % (meta.get("sklearn_version"), sklearn.__version__)
        )
    pipelines = {
        layer: joblib.load(fit_dir / ("layer-%03d.joblib" % layer))
        for layer in layers_needed
    }
    return pipelines, meta


def _refuse_under_results(parser, path, flag):
    if path is None:
        return
    resolved = Path(path).resolve()
    results_root = (Path.cwd() / "results").resolve()
    if resolved == results_root or results_root in resolved.parents:
        parser.error("%s must not be under results/ (results are JSONL only)" % flag)


# ---------------------------------------------------------------------------
# Model loading (run_corroboration.py's sidecar convention); a test seam
# ---------------------------------------------------------------------------


def _default_load_model(args):
    sidecar = None
    if args.adapter is not None:
        sidecar_path = Path(args.adapter) / "train_meta.json"
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if args.checkpoint_step is None:
                args.checkpoint_step = sidecar.get("checkpoint_step")
            if args.train_seed is None:
                args.train_seed = sidecar.get("train_seed")
    if sidecar is not None:
        from algoverse.models import load_checkpoint_model

        model, tokenizer, _meta, _handle = load_checkpoint_model(
            args.model_id, args.adapter, quant=args.quant
        )
    else:
        from algoverse.models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            args.model_id, quant=args.quant, adapter_path=args.adapter
        )
    return model, tokenizer


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter dir of the checkpoint whose "
                             "activations are probed (fixed-stimuli design: "
                             "the test rows need not come from it)")
    parser.add_argument("--train-dataset", required=True,
                        help="Instructed-Pairs JSONL ({text,label,group}); "
                             "identity even under --no-own-fit")
    parser.add_argument("--test-rows", action="append", required=True,
                        metavar="[LABEL=]PATH",
                        help="results rows.jsonl with BOTH lied and "
                             "did-not-lie valid incentive rows; LABEL=PATH "
                             "(repeatable) with --out-root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=None,
                        help="legacy single output: <out-dir>/interp.jsonl, "
                             "run_id verbatim; fresh diag-* directory")
    parser.add_argument("--out-root", default=None,
                        help="one output per (fit, label) at "
                             "<out-root>/<run-id>-<fit>-<label>/interp.jsonl")
    parser.add_argument("--feature-position", default="response_tokens",
                        choices=sorted(FEATURE_POSITIONS),
                        help="which tokens the probe reads (module docstring)")
    parser.add_argument("--save-fit", default=None, metavar="DIR",
                        help="persist the own fit (per-layer joblib + "
                             "fit_meta.json); never under results/")
    parser.add_argument("--use-fit", default=None, metavar="DIR",
                        help="also score every test set with this persisted "
                             "direction (the 'fixed' outputs)")
    parser.add_argument("--no-own-fit", action="store_true",
                        help="skip the own fit and the train capture "
                             "(requires --use-fit)")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--probe-scratch-dir", default=None,
                        help="local temp dir for float32 activation spools; "
                             "never results/")
    return parser


def main(argv=None, _load_model=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    _refuse_under_results(parser, args.probe_scratch_dir, "--probe-scratch-dir")
    _refuse_under_results(parser, args.save_fit, "--save-fit")
    _refuse_under_results(parser, args.use_fit, "--use-fit")
    if bool(args.out_dir) == bool(args.out_root):
        parser.error("give exactly one of --out-dir (legacy) or --out-root")
    if args.no_own_fit and not args.use_fit:
        parser.error("--no-own-fit needs --use-fit (nothing would be scored)")
    if args.no_own_fit and args.save_fit:
        parser.error("--save-fit needs the own fit; drop --no-own-fit")

    parsed = [parse_test_rows_spec(spec) for spec in args.test_rows]
    labels = [label for label, _ in parsed]
    if args.out_root:
        if any(label is None for label in labels):
            parser.error("--out-root needs every --test-rows as LABEL=PATH")
        if len(set(labels)) != len(labels):
            parser.error("--test-rows labels must be unique")
    else:
        if len(parsed) != 1:
            parser.error("--out-dir takes exactly one --test-rows")
    fits = ([] if args.no_own_fit else [FIT_OWN]) + ([FIT_FIXED] if args.use_fit else [])
    if args.out_dir and len(fits) > 1:
        parser.error("--out-dir holds one output; use --out-root for own+fixed")
    legacy = (
        bool(args.out_dir) and fits == [FIT_OWN]
        and args.feature_position == "response_tokens" and labels[0] is None
    )
    exclude_final_line = args.feature_position == "response_excl_claim"
    # Fail on a missing/incomplete fit dir BEFORE the model download.
    fit_meta = read_fit_meta(args.use_fit) if args.use_fit else None

    from algoverse.models import bypass_impl_string

    model, tokenizer = (_load_model or _default_load_model)(args)
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)

    # Test sets: one labeled bundle each, refusing single-class draws.
    test_sets = []
    for label, path in parsed:
        stats = {}
        examples = probe_examples_from_rows(
            load_rows(Path(path)), tokenizer,
            exclude_final_line=exclude_final_line, stats=stats,
        )
        test_labels = [example["label"] for example in examples]
        n_lied = sum(1 for value in test_labels if value)
        n_honest = len(test_labels) - n_lied
        if n_lied == 0 or n_honest == 0:
            raise SystemExit(
                "test rows %s are single-class (lied=%d, did-not-lie=%d): "
                "transfer AUROC is undefined. A saturated M_D checkpoint "
                "cannot be the test set; use rows with behavioral variation."
                % (path, n_lied, n_honest)
            )
        print(
            "test %s: %d responses (lied=%d, did-not-lie=%d)%s"
            % (label or path, len(examples), n_lied, n_honest,
               " no_marker=%d skipped_empty_body=%d"
               % (stats["n_no_marker"], stats["n_skipped_empty_body"])
               if exclude_final_line else "")
        )
        test_sets.append({
            "label": label, "path": path, "examples": examples,
            "labels": test_labels, "n_lied": n_lied,
            "groups": [example["group"] for example in examples],
            "stats": stats,
        })

    train_examples = None
    if not args.no_own_fit:
        train_examples = load_probe_dataset(args.train_dataset)
        print("train: %d instructed-pairs examples" % len(train_examples))

    outputs = plan_outputs(args.run_id, args.out_dir, args.out_root, labels, fits)
    by_label = {bundle["label"]: bundle for bundle in test_sets}
    for output in outputs:
        bundle = by_label[output["label"]]
        output["run_meta"] = {
            "run_id": output["run_id"],
            "model_id": args.model_id,
            "adapter_path": args.adapter,
            "bypassed_layer": None,
            "checkpoint_step": args.checkpoint_step,
            "arm": None,
            "train_seed": args.train_seed,
            "bypass_impl": bypass_impl_string(model),
        }
        _, span_len = capture_spans(bundle["examples"], args.feature_position)
        output["config"] = build_config(
            train_dataset=args.train_dataset, test_path=bundle["path"],
            n_train=(len(train_examples) if train_examples is not None
                     else (fit_meta or {}).get("n_train")),
            n_test=len(bundle["examples"]), n_test_lied=bundle["n_lied"],
            fit_tag=output["fit"], feature_position=args.feature_position,
            span_len=span_len, exclude_final_line=exclude_final_line,
            stats=bundle["stats"], label=output["label"],
            fit_meta=fit_meta, legacy=legacy,
        )
        output["pending"] = [
            layer for layer in range(n_layers)
            if not _interp_done(output["out_path"], output["run_meta"],
                                "probe_auroc", layer, output["config"])
        ]
        print(
            "[%s/%s] %s: %d of %d layers pending"
            % (output["fit"], output["label"] or "test", output["run_id"],
               len(output["pending"]), n_layers)
        )

    save_complete = bool(args.save_fit) and (
        Path(args.save_fit) / FIT_META_NAME
    ).is_file()
    if all(not output["pending"] for output in outputs) and (
        not args.save_fit or save_complete
    ):
        print("probe_auroc transfer: every output complete; nothing to do")
        return 0

    import numpy as np

    from algoverse.interp import (
        iter_disk_backed_residual_layers,
        response_token_resid_by_layer_to_disk,
    )

    if args.probe_scratch_dir:
        Path(args.probe_scratch_dir).mkdir(parents=True, exist_ok=True)

    # Phase A: the own fit (one train capture serves every own output).
    own_fits = {}
    own_pending = sorted({
        layer for output in outputs if output["fit"] == FIT_OWN
        for layer in output["pending"]
    })
    need_own = not args.no_own_fit and (
        bool(own_pending) or (bool(args.save_fit) and not save_complete)
    )
    if need_own:
        train_layers = list(range(n_layers)) if args.save_fit else own_pending
        train_starts, train_span = capture_spans(train_examples, args.feature_position)
        train_labels = [example["label"] for example in train_examples]
        with tempfile.TemporaryDirectory(
            prefix="algoverse-transfer-train-", dir=args.probe_scratch_dir
        ) as tmp:
            train_meta = response_token_resid_by_layer_to_disk(
                model, tokenizer,
                [example["text"] for example in train_examples],
                train_starts, Path(tmp), layers=train_layers,
                span_len=train_span,
            )
            for layer, feats in enumerate(iter_disk_backed_residual_layers(train_meta)):
                if feats is None or layer not in train_layers:
                    continue
                tokens = np.concatenate([np.asarray(t) for t in feats])
                token_labels = np.concatenate([
                    np.full(np.asarray(t).shape[0], bool(label))
                    for t, label in zip(feats, train_labels)
                ])
                own_fits[layer] = fit_transfer_probe(tokens, token_labels)
                print("fit layer %2d on %d train tokens" % (layer, tokens.shape[0]))
        if args.save_fit:
            meta_path = save_fit(args.save_fit, own_fits, {
                "fit_run_id": args.run_id,
                "model_id": args.model_id,
                "adapter_path": args.adapter,
                "checkpoint_step": args.checkpoint_step,
                "train_seed": args.train_seed,
                "bypass_impl": bypass_impl_string(model),
                "train_dataset": str(Path(args.train_dataset).resolve()),
                "train_dataset_basename": Path(args.train_dataset).name,
                "feature_position": FEATURE_POSITIONS[args.feature_position],
                "span_len": train_span,
                "n_layers": n_layers,
                "hidden_size": hidden_size,
                "n_train": len(train_examples),
                "excluded_layers": sorted(train_meta["excluded"]),
                "probe_recipe": dict(PROBE_RECIPE),
                "sklearn_version": __import__("sklearn").__version__,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            print("saved fit -> %s" % meta_path)

    # Phase A': the fixed direction.
    fixed_fits = {}
    fixed_pending = sorted({
        layer for output in outputs if output["fit"] == FIT_FIXED
        for layer in output["pending"]
    })
    if fixed_pending:
        fixed_fits, fit_meta = load_fit(
            args.use_fit, model_id=args.model_id,
            feature_position=args.feature_position, n_layers=n_layers,
            hidden_size=hidden_size, train_dataset=args.train_dataset,
            layers_needed=fixed_pending,
        )
        print("loaded fixed direction %s for %d layers" % (fit_meta["fit_run_id"], len(fixed_fits)))

    # Phase B: one test capture per test set, scored by every pending fit.
    for bundle in test_sets:
        targets = [
            output for output in outputs
            if output["label"] == bundle["label"] and output["pending"]
        ]
        if not targets:
            continue
        layers_needed = sorted({layer for output in targets for layer in output["pending"]})
        test_starts, test_span = capture_spans(bundle["examples"], args.feature_position)
        with tempfile.TemporaryDirectory(
            prefix="algoverse-transfer-test-", dir=args.probe_scratch_dir
        ) as tmp:
            test_meta = response_token_resid_by_layer_to_disk(
                model, tokenizer,
                [example["text"] for example in bundle["examples"]],
                test_starts, Path(tmp), layers=layers_needed,
                span_len=test_span,
            )
            for layer, feats in enumerate(iter_disk_backed_residual_layers(test_meta)):
                if layer not in layers_needed:
                    continue
                for output in targets:
                    if layer not in output["pending"]:
                        continue
                    if _interp_done(output["out_path"], output["run_meta"],
                                    "probe_auroc", layer, output["config"]):
                        continue
                    fits_for = own_fits if output["fit"] == FIT_OWN else fixed_fits
                    clf = fits_for.get(layer)
                    if feats is None or clf is None:
                        excluded = dict(output["config"])
                        excluded["excluded_bypassed_layer"] = True
                        write_interp_row(
                            output["out_path"], output["run_meta"],
                            "probe_auroc", layer, None, None, None, excluded,
                            extra={"accuracy": None},
                        )
                        print("[%s/%s] layer %d: structural null (bypassed)"
                              % (output["fit"], output["label"] or "test", layer))
                        continue
                    auroc, ci_low, ci_high, accuracy = score_transfer_probe(
                        clf, [np.asarray(t) for t in feats],
                        bundle["labels"], bundle["groups"],
                    )
                    layer_config = output["config"]
                    if ci_low is None or ci_high is None:
                        layer_config = dict(output["config"])
                        layer_config["ci"] = (
                            "null: group bootstrap degenerate (too few "
                            "resamplable held-out scenario groups)"
                        )
                    write_interp_row(
                        output["out_path"], output["run_meta"], "probe_auroc",
                        layer, auroc, ci_low, ci_high, layer_config,
                        extra={"accuracy": accuracy},
                    )
                    print(
                        "[%s/%s] layer %2d: transfer AUROC %.3f  CI [%s, %s]  acc %.3f"
                        % (output["fit"], output["label"] or "test", layer, auroc,
                           "n/a" if ci_low is None else "%.3f" % ci_low,
                           "n/a" if ci_high is None else "%.3f" % ci_high,
                           accuracy)
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
