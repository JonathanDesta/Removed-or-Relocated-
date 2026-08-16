"""Guarded rung-2 tests for the corroboration driver (tiny Qwen2, CPU).

Tiny random CPU models only — never a GPU (AGENTS.md rung rules).

Run: ~/.venvs/colab-local/bin/python tests/test_corroboration.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORROBORATION_TEST_COUNT = 6

try:
    import numpy as np
    import torch
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.corroboration import (
        PROBE_RECIPE,
        probe_examples_from_rows,
        run_attention_jsd,
        run_probe_auroc,
    )
    from algoverse.interp import probe_layer, response_token_resid_by_layer
    from algoverse.models import install_bypass, residual_stream_by_layer

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    RUN_META = {
        "run_id": "corr-rung2",
        "model_id": "tiny",
        "adapter_path": None,
        "bypassed_layer": None,
        "checkpoint_step": None,
        "arm": None,
        "train_seed": None,
        "bypass_impl": None,
    }

    def _tiny_model():
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    class StubTokenizer:
        """Word-split stub: deterministic ids, chat template, no specials."""

        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, texts, return_tensors="pt", padding=False,
                     **kwargs):
            single = isinstance(texts, str)
            texts = [texts] if single else list(texts)
            encoded = [
                [3 + (sum(token.encode("utf-8")) % 120)
                 for token in text.split()]
                for text in texts
            ]
            width = max(len(row) for row in encoded)
            if padding:
                encoded = [[0] * (width - len(row)) + row for row in encoded]
            masks = [[int(token != 0) for token in row] for row in encoded]
            return BatchEncoding({
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }, tensor_type="pt")

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            rendered = "\n".join(
                "%s %s" % (message["role"], message["content"])
                for message in messages
            )
            if add_generation_prompt:
                rendered += "\nassistant\n"
            return rendered

    def _texts(n, words=6, offset=0, seed=0):
        rng = np.random.default_rng(seed)
        vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                      "eta", "theta", "iota", "kappa"]
        out = []
        for i in range(n):
            picks = rng.choice(vocabulary, size=words).tolist()
            out.append(" ".join(["p%d" % (i + offset)] + picks))
        return out

    def test_capture_matches_residual_stream_reference():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        texts = ["one two three four five", "six seven eight nine"]
        starts = [2, 1]
        features = response_token_resid_by_layer(
            model, tokenizer, texts, starts
        )
        assert len(features) == 4
        for layer_features in features:
            assert [f.shape for f in layer_features] == [(3, 32), (3, 32)]
            assert all(f.dtype == np.float32 for f in layer_features)
        # The feature for layer l must be the residual LEAVING block l
        # (capture entry l+1) over the response span, per the sanctioned
        # lesion-safe reader.
        inputs = tokenizer(texts[0], return_tensors="pt",
                           add_special_tokens=False)
        captured = residual_stream_by_layer(
            model, inputs["input_ids"], inputs["attention_mask"]
        )
        for layer in range(4):
            expected = captured[layer + 1][0, 2:, :].float().numpy()
            assert np.allclose(features[layer][0], expected)
        # Span validation: a start outside the text is an error.
        try:
            response_token_resid_by_layer(model, tokenizer, [texts[1]], [9])
        except ValueError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("out-of-range response start accepted")

    def test_capture_excludes_bypassed_layer():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        handle = install_bypass(model, 1)
        try:
            features = response_token_resid_by_layer(
                model, tokenizer, ["one two three four"], [1]
            )
        finally:
            handle.remove()
        assert features[1] is None
        for layer in (0, 2, 3):
            assert features[layer] is not None
            assert np.all(np.isfinite(features[layer][0]))

    def test_probe_layer_token_fitting_and_mean_aggregation():
        # Synthetic responses with a shared per-token signal: fitting on
        # tokens and mean-aggregating scores must recover the labels.
        rng = np.random.default_rng(0)
        responses, labels, groups = [], [], []
        for i in range(40):
            label = i % 2
            n_tokens = 2 + (i % 3)
            tokens = rng.normal(0, 0.05, (n_tokens, 5))
            tokens[:, 0] += 2.0 * label
            responses.append(tokens)
            labels.append(label)
            groups.append("g%d" % i)
        clf, result = probe_layer(responses, labels, groups)
        assert result["auroc"] > 0.9
        assert 0 <= result["accuracy"] <= 1
        # The held-out response score is the MEAN of its per-token decision
        # scores (the paper's aggregation), reproduced by hand.
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.3, random_state=0
        )
        _, test_idx = next(splitter.split(
            np.zeros((len(responses), 1)), np.asarray(labels),
            np.asarray(groups),
        ))
        scores = np.array([
            float(np.mean(clf.decision_function(responses[i])))
            for i in test_idx
        ])
        from sklearn.metrics import roc_auc_score

        y = np.asarray(labels)
        assert roc_auc_score(y[test_idx], scores) == result["auroc"]
        expected_accuracy = float(np.mean(
            (scores > 0) == y[test_idx].astype(bool)
        ))
        assert expected_accuracy == result["accuracy"]

    def test_probe_auroc_end_to_end_rows_and_resume():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        examples = []
        texts = _texts(24, seed=3)
        for i, text in enumerate(texts):
            examples.append({
                "text": text,
                "response_start": 1,
                "label": (i // 2) % 2 == 0,   # group-consistent labels
                "group": "g%d" % (i // 2),
            })
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "interp.jsonl"
            written = run_probe_auroc(
                model, tokenizer, examples, out, RUN_META, "rows:synthetic",
            )
            rows = [json.loads(line) for line in
                    out.read_text().strip().splitlines()]
            assert [row["layer"] for row in rows] == [0, 1, 2, 3]
            assert set(written) == {0, 1, 2, 3}
            for row in rows:
                assert row["analysis"] == "probe_auroc"
                assert 0.0 <= row["value"] <= 1.0
                assert 0.0 <= row["accuracy"] <= 1.0
                if row["ci_low"] is not None:
                    assert row["ci_low"] <= row["value"] <= row["ci_high"]
                else:
                    assert "bootstrap" in row["config"]["ci"]
                assert row["config"]["n"] == 24
                assert row["config"]["label_source"] == "rows:synthetic"
                for key, value in PROBE_RECIPE.items():
                    assert row["config"][key] == value
            # Resume: identical invocation adds nothing.
            again = run_probe_auroc(
                model, tokenizer, examples, out, RUN_META, "rows:synthetic",
            )
            assert again == {}
            assert len(out.read_text().strip().splitlines()) == 4

        # Bypassed model: the dead layer's row is a structural null.
        handle = install_bypass(model, 2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "interp.jsonl"
                run_probe_auroc(
                    model, tokenizer, examples, out, RUN_META,
                    "rows:synthetic",
                )
                rows = [json.loads(line) for line in
                        out.read_text().strip().splitlines()]
                bypassed = [row for row in rows if row["layer"] == 2]
                assert len(bypassed) == 1
                assert bypassed[0]["value"] is None
                assert bypassed[0]["accuracy"] is None
                assert bypassed[0]["config"]["excluded_bypassed_layer"] is True
                assert all(
                    row["value"] is not None
                    for row in rows if row["layer"] != 2
                )
        finally:
            handle.remove()

    def test_attention_jsd_rows_and_resume():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        texts_a = _texts(3, seed=1)
        texts_b = _texts(3, offset=10, seed=2)
        groups = ["s1", "s2", "s3"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "interp.jsonl"
            written = run_attention_jsd(
                model, tokenizer, texts_a, texts_b, out, RUN_META,
                groups_incentive=groups, groups_control=groups,
                n_boot=40, extra_config={"split": "selection"},
            )
            rows = [json.loads(line) for line in
                    out.read_text().strip().splitlines()]
            assert [row["layer"] for row in rows] == [0, 1, 2, 3]
            assert set(written) == {0, 1, 2, 3}
            for row in rows:
                assert row["analysis"] == "attention_jsd"
                assert row["value"] >= 0.0
                assert "accuracy" not in row
                assert row["config"]["n_boot"] == 40
                assert row["config"]["split"] == "selection"
                assert row["config"]["attn_implementation"] == "eager"
                if row["ci_low"] is not None:
                    assert row["ci_low"] <= row["ci_high"]
            again = run_attention_jsd(
                model, tokenizer, texts_a, texts_b, out, RUN_META,
                groups_incentive=groups, groups_control=groups,
                n_boot=40, extra_config={"split": "selection"},
            )
            assert again == {}
            assert len(out.read_text().strip().splitlines()) == 4

        handle = install_bypass(model, 1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "interp.jsonl"
                run_attention_jsd(
                    model, tokenizer, texts_a, texts_b, out, RUN_META,
                    n_boot=40,
                )
                rows = [json.loads(line) for line in
                        out.read_text().strip().splitlines()]
                bypassed = [row for row in rows if row["layer"] == 1]
                assert bypassed[0]["value"] is None
                assert bypassed[0]["ci_low"] is None
                assert bypassed[0]["config"]["excluded_bypassed_layer"] is True
        finally:
            handle.remove()

    def test_probe_examples_from_rows_renders_and_spans():
        from algoverse.tasks import get_scenarios

        tokenizer = StubTokenizer()
        scenarios = get_scenarios("selection", n=2, seed=42)
        rows = [
            {
                "condition": "incentive", "valid": True, "deceptive": True,
                "scenario_id": scenarios[0]["scenario_id"],
                "response_text": "my offer is large\nMY BEST OUTSIDE OFFER"
                                 ": $120,000",
            },
            {
                "condition": "incentive", "valid": True, "deceptive": False,
                "scenario_id": scenarios[1]["scenario_id"],
                "response_text": "honest reply\nMY BEST OUTSIDE OFFER: NONE",
            },
            {   # excluded: invalid
                "condition": "incentive", "valid": False, "deceptive": None,
                "scenario_id": scenarios[0]["scenario_id"],
                "response_text": "refused",
            },
            {   # excluded: control condition
                "condition": "control", "valid": True, "deceptive": False,
                "scenario_id": scenarios[1]["scenario_id"],
                "response_text": "control reply",
            },
        ]
        examples = probe_examples_from_rows(rows, tokenizer)
        assert len(examples) == 2
        assert [example["label"] for example in examples] == [True, False]
        assert [example["group"] for example in examples] == [
            scenarios[0]["scenario_id"], scenarios[1]["scenario_id"]
        ]
        for example, row in zip(examples, rows[:2]):
            assert example["text"].endswith(row["response_text"])
            # The span starts exactly where the canonical prompt's tokens
            # end, under the same tokenization the capture helper uses.
            prompt = example["text"][:-len(row["response_text"])]
            n_prompt = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"].shape[1]
            assert example["response_start"] == n_prompt
            n_full = tokenizer(
                example["text"], return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].shape[1]
            assert example["response_start"] < n_full
            # The rendered prompt includes the generation prompt.
            assert "assistant" in prompt


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_corroboration.py needs torch + transformers + sklearn "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip (AGENTS.md)."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == CORROBORATION_TEST_COUNT, (
        "expected %d tests, found %d"
        % (CORROBORATION_TEST_COUNT, len(tests))
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except Exception:
            failures += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    if failures:
        sys.exit("%d test(s) failed" % failures)
    print("ALL TESTS PASSED")
