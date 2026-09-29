"""Guarded ML-stack tier tests for the Instructed-Pairs builder (tiny
Qwen2, CPU).

Tiny random CPU models only — this suite must never run on a GPU. With
the shared word-split tokenizer stub, rows are rendered end-to-end through
hf_renderers (apply_chat_template + add_special_tokens=False token
counting), spans are verified at the token level, and the residual capture
over built rows (interp.response_token_resid_by_layer) reads exactly the
statement span.

Run: python tests/test_instructed_pairs.py with the requirements.txt stack
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    WordSplitTokenizer,
    run_suite,
    skip_module_unless_stack,
    tiny_qwen2_model,
)

INSTRUCTED_PAIRS_TEST_COUNT = 2

try:
    import torch

    from algoverse.interp import response_token_resid_by_layer
    from build_instructed_pairs import (
        TEMPLATE_HONEST,
        TEMPLATE_UNTRUTHFUL,
        build_rows,
        hf_renderers,
    )

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

MISSING_STACK = skip_module_unless_stack("torch", "transformers")


if HAVE_STACK:
    def _tiny_model():
        return tiny_qwen2_model(max_position_embeddings=512)

    StubTokenizer = WordSplitTokenizer

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


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=INSTRUCTED_PAIRS_TEST_COUNT,
                              missing=MISSING_STACK))
