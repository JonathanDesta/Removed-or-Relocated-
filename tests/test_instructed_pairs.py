"""Guarded rung-2 tests for the Instructed-Pairs builder (tiny Qwen2, CPU).

Tiny random CPU models only — this suite must never run on a GPU. Reuses
the StubTokenizer chat-template fixture pattern of test_corroboration.py:
rows are rendered end-to-end through hf_renderers (apply_chat_template +
add_special_tokens=False token counting), spans are verified at the token
level, and one probe pass runs over built rows via the corroboration path
(load_probe_dataset -> run_probe_auroc -> response_token_resid_by_layer).

Run: ~/.venvs/colab-local/bin/python tests/test_instructed_pairs.py
"""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

INSTRUCTED_PAIRS_TEST_COUNT = 3

try:
    import numpy as np
    import torch
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.corroboration import load_probe_dataset, run_probe_auroc
    from algoverse.interp import response_token_resid_by_layer
    from build_instructed_pairs import (
        TEMPLATE_HONEST,
        TEMPLATE_UNTRUTHFUL,
        build_manifest,
        build_rows,
        hf_renderers,
        write_outputs,
    )

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    RUN_META = {
        "run_id": "instructed-pairs-rung2",
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
        """Word-split stub: deterministic ids, chat template, no specials.

        Same fixture shape as test_corroboration.py's StubTokenizer, so
        hf_renderers exercises the real apply_chat_template +
        add_special_tokens=False code path.
        """

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

    def _statements(n):
        vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        return [
            "fact%d %s is a documented statement" % (
                i, vocabulary[i % len(vocabulary)]
            )
            for i in range(n)
        ]

    def test_end_to_end_render_and_token_spans():
        tokenizer = StubTokenizer()
        statements = _statements(3)
        render_prompt, token_ids = hf_renderers(tokenizer)
        rows = build_rows(statements, render_prompt, token_ids)
        assert len(rows) == 6
        for row in rows:
            statement = statements[row["group"]]
            # The rendered text is the chat-templated prompt (user turn =
            # a verbatim template, generation prompt included) with the
            # statement appended as the assistant turn.
            assert row["text"].startswith("user Pretend you're ")
            assert "\nassistant\n" in row["text"]
            assert row["text"].endswith(statement)
            template = (
                TEMPLATE_UNTRUTHFUL if row["label"] else TEMPLATE_HONEST
            )
            assert template in row["text"]
            # response_start lands exactly on the statement tokens: the
            # token ids from response_start onward are the statement's own
            # ids under the contract tokenization.
            full_ids = token_ids(row["text"])
            assert full_ids[row["response_start"]:] == token_ids(statement)
            prompt = row["text"][:-len(statement)]
            assert row["response_start"] == len(token_ids(prompt))

    def test_capture_reads_exactly_the_statement_span():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        statements = _statements(2)
        rows = build_rows(statements, *hf_renderers(tokenizer))
        features = response_token_resid_by_layer(
            model, tokenizer,
            [row["text"] for row in rows],
            [row["response_start"] for row in rows],
        )
        assert len(features) == 4
        for layer_features in features:
            # One [n_statement_tokens, d_model] array per row: the span
            # covers the statement's tokens and nothing else.
            assert [f.shape for f in layer_features] == [
                (len(statements[row["group"]].split()), 32) for row in rows
            ]

    def test_probe_pass_over_built_rows_via_corroboration_path():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        statements = _statements(12)
        rows = build_rows(statements, *hf_renderers(tokenizer))
        manifest = build_manifest(
            "file://synthetic", "0" * 64, len(statements), len(rows),
            "tiny", None,
        )
        seen_starts = []

        def capture(model_arg, tokenizer_arg, texts, starts):
            seen_starts.extend(starts)
            return response_token_resid_by_layer(
                model_arg, tokenizer_arg, texts, starts
            )

        with tempfile.TemporaryDirectory() as tmp:
            rows_path, _ = write_outputs(
                Path(tmp) / "data", "stub", rows, manifest
            )
            examples = load_probe_dataset(rows_path)
            assert [example["response_start"] for example in examples] == [
                row["response_start"] for row in rows
            ]
            out = Path(tmp) / "interp.jsonl"
            written = run_probe_auroc(
                model, tokenizer, examples, out, RUN_META,
                "instructed_pairs:stub", _capture=capture,
            )
            # The per-row spans flowed through the probe path untouched.
            assert seen_starts == [row["response_start"] for row in rows]
            assert set(written) == {0, 1, 2, 3}
            interp_rows = [json.loads(line) for line in
                           out.read_text().strip().splitlines()]
            assert [row["layer"] for row in interp_rows] == [0, 1, 2, 3]
            for row in interp_rows:
                assert row["analysis"] == "probe_auroc"
                assert 0.0 <= row["value"] <= 1.0
                assert 0.0 <= row["accuracy"] <= 1.0
                assert row["config"]["n"] == 24
                assert row["config"]["label_source"] == (
                    "instructed_pairs:stub"
                )


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_instructed_pairs.py needs torch + transformers + sklearn "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == INSTRUCTED_PAIRS_TEST_COUNT, (
        "expected %d tests, found %d"
        % (INSTRUCTED_PAIRS_TEST_COUNT, len(tests))
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
