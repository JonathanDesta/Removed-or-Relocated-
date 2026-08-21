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
         mean over response-token decision scores (the ratified
         aggregation); AUROC over responses, scenario-group bootstrap CI.

An instructed-pairs probe that reads lexical instruction cues rather than
deception should transfer at chance; a probe that found a deception-
relevant direction should stay well above it. High within-source AUROC
(the 1.0-everywhere m0-corr result) with chance transfer = the
corroboration ceiling is a lexical artifact.

TEST-SET CHOICE IS THE TEAM'S: the test rows must contain BOTH lied and
did-not-lie incentive rows. M_D final checkpoints are saturated (all
lied — this script refuses them with the class counts); usable existing
rows include Llama M_0 (17 lied / 85 not) and the step-8 diagnostic
(78 / 22). The script refuses degenerate inputs rather than choosing.

Human-run (GPU: it captures activations). Writes probe_auroc rows to
<out-dir>/interp.jsonl with config.transfer = true and a
"transfer:..." label_source. USE A FRESH diag-* OUT DIR — never a
standard corroboration run's folder, so downstream probe_auroc consumers
cannot pool transfer rows with standard rows unnoticed.

    python scripts/run_probe_transfer.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct --quant 4bit \
        --train-dataset data/instructed_pairs/llama8b.jsonl \
        --test-rows results/m0-baseline-llama8b/rows.jsonl \
        --run-id diag-probe-transfer-m0-llama8b \
        --out-dir results/diag-probe-transfer-m0-llama8b
"""
import argparse
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


def transfer_probe_layer(train_tokens, train_token_labels, test_responses,
                         test_labels, test_groups):
    """Fit the ratified probe on one layer's train tokens; score test set.

    train_tokens: [n_tokens, d] array (already flattened across train
    responses); train_token_labels: per-token labels. test_responses:
    list of [n_tokens_i, d] arrays. Returns (auroc, ci_low, ci_high,
    accuracy). Pure given arrays; imported by tests with synthetic data.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from algoverse.interp import _group_bootstrap_auroc_ci

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", C=PROBE_RECIPE["C"],
            max_iter=PROBE_RECIPE["max_iter"],
        ),
    ).fit(train_tokens, train_token_labels)
    scores = np.asarray(aggregate_response_scores(
        [clf.decision_function(tokens) for tokens in test_responses]
    ))
    y = np.asarray(test_labels)
    auroc = float(roc_auc_score(y, scores))
    ci_low, ci_high = _group_bootstrap_auroc_ci(y, scores, test_groups)
    accuracy = float(((scores > 0) == y.astype(bool)).mean())
    return auroc, ci_low, ci_high, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter dir; MUST be the checkpoint that "
                             "generated --test-rows (activations must come "
                             "from the model whose behavior is probed)")
    parser.add_argument("--train-dataset", required=True,
                        help="Instructed-Pairs JSONL ({text,label,group})")
    parser.add_argument("--test-rows", required=True,
                        help="results rows.jsonl with BOTH lied and "
                             "did-not-lie valid incentive rows")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="fresh diag-* directory; never a standard "
                             "corroboration run's folder")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--probe-scratch-dir", default=None,
                        help="local temp dir for float32 activation spools "
                             "(two model-sized captures); never results/")
    args = parser.parse_args()

    if args.probe_scratch_dir:
        scratch = Path(args.probe_scratch_dir).resolve()
        results_root = (Path.cwd() / "results").resolve()
        if scratch == results_root or results_root in scratch.parents:
            parser.error("--probe-scratch-dir must not be under results/")

    # Sidecar adoption + loader, run_corroboration.py's convention.
    import json

    sidecar = None
    if args.adapter is not None:
        sidecar_path = Path(args.adapter) / "train_meta.json"
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if args.checkpoint_step is None:
                args.checkpoint_step = sidecar.get("checkpoint_step")
            if args.train_seed is None:
                args.train_seed = sidecar.get("train_seed")

    from algoverse.models import bypass_impl_string

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

    train_examples = load_probe_dataset(args.train_dataset)
    test_examples = probe_examples_from_rows(
        load_rows(Path(args.test_rows)), tokenizer
    )
    test_labels = [example["label"] for example in test_examples]
    n_lied = sum(1 for label in test_labels if label)
    n_honest = len(test_labels) - n_lied
    if n_lied == 0 or n_honest == 0:
        raise SystemExit(
            "test rows are single-class (lied=%d, did-not-lie=%d): transfer "
            "AUROC is undefined. A saturated M_D checkpoint cannot be the "
            "test set; use rows with behavioral variation (e.g. M_0, the "
            "step-8 diagnostic)." % (n_lied, n_honest)
        )
    print(
        "train: %d instructed-pairs examples; test: %d responses "
        "(lied=%d, did-not-lie=%d)"
        % (len(train_examples), len(test_examples), n_lied, n_honest)
    )

    run_meta = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "adapter_path": args.adapter,
        "bypassed_layer": None,
        "checkpoint_step": args.checkpoint_step,
        "arm": None,
        "train_seed": args.train_seed,
        "bypass_impl": bypass_impl_string(model),
    }
    config = dict(PROBE_RECIPE)
    config.update({
        "transfer": True,
        "label_source": "transfer:probe_dataset:%s->within_incentive_rows:%s"
                        % (args.train_dataset, args.test_rows),
        "n_train": len(train_examples),
        "n_test": len(test_examples),
        "n_test_lied": n_lied,
        "fit": "all_train_examples_no_holdout",
        "status": "exploratory-diagnostic; unratified",
    })
    out_path = Path(args.out_dir) / "interp.jsonl"

    import numpy as np

    from algoverse.interp import (
        iter_disk_backed_residual_layers,
        response_token_resid_by_layer_to_disk,
    )

    n_layers = model.config.num_hidden_layers
    pending = [
        layer for layer in range(n_layers)
        if not _interp_done(out_path, run_meta, "probe_auroc", layer, config)
    ]
    if not pending:
        print("probe_auroc transfer: all %d layers already complete" % n_layers)
        raise SystemExit(0)

    if args.probe_scratch_dir:
        Path(args.probe_scratch_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="algoverse-transfer-", dir=args.probe_scratch_dir
    ) as tmp:
        train_meta = response_token_resid_by_layer_to_disk(
            model, tokenizer,
            [example["text"] for example in train_examples],
            [example.get("response_start", 0) for example in train_examples],
            Path(tmp) / "train", layers=pending,
        )
        test_meta = response_token_resid_by_layer_to_disk(
            model, tokenizer,
            [example["text"] for example in test_examples],
            [example["response_start"] for example in test_examples],
            Path(tmp) / "test", layers=pending,
        )

        train_labels = [example["label"] for example in train_examples]
        test_groups = [example["group"] for example in test_examples]
        for layer, (train_feats, test_feats) in enumerate(zip(
            iter_disk_backed_residual_layers(train_meta),
            iter_disk_backed_residual_layers(test_meta),
        )):
            if _interp_done(out_path, run_meta, "probe_auroc", layer, config):
                continue
            if train_feats is None or test_feats is None:
                excluded = dict(config)
                excluded["excluded_bypassed_layer"] = True
                write_interp_row(
                    out_path, run_meta, "probe_auroc", layer,
                    None, None, None, excluded, extra={"accuracy": None},
                )
                print("layer %d: structural null (bypassed)" % layer)
                continue
            train_tokens = np.concatenate(train_feats)
            train_token_labels = np.concatenate([
                np.full(tokens.shape[0], bool(label))
                for tokens, label in zip(train_feats, train_labels)
            ])
            auroc, ci_low, ci_high, accuracy = transfer_probe_layer(
                train_tokens, train_token_labels,
                [np.asarray(tokens) for tokens in test_feats],
                test_labels, test_groups,
            )
            layer_config = config
            if ci_low is None or ci_high is None:
                layer_config = dict(config)
                layer_config["ci"] = (
                    "null: group bootstrap degenerate (too few resamplable "
                    "held-out scenario groups)"
                )
            write_interp_row(
                out_path, run_meta, "probe_auroc", layer,
                auroc, ci_low, ci_high, layer_config,
                extra={"accuracy": accuracy},
            )
            print(
                "layer %2d: transfer AUROC %.3f  CI [%s, %s]  acc %.3f"
                % (
                    layer, auroc,
                    "n/a" if ci_low is None else "%.3f" % ci_low,
                    "n/a" if ci_high is None else "%.3f" % ci_high,
                    accuracy,
                )
            )
