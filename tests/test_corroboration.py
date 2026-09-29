"""Guarded ML-stack tier tests for the probe capture helpers and probe
label sources (tiny Qwen2, CPU).

Tiny random CPU models only — this suite must never run on a GPU.

Run: python tests/test_corroboration.py with the requirements.txt stack
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORROBORATION_TEST_COUNT = 6

try:
    import numpy as np
    import torch
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.corroboration import probe_examples_from_rows
    from algoverse.interp import (
        iter_disk_backed_residual_layers,
        response_token_resid_by_layer,
        response_token_resid_by_layer_to_disk,
    )
    from algoverse.models import install_bypass, residual_stream_by_layer

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
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
        # (capture entry l+1) over the response span, per the bypass-safe
        # reader.
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

    def test_disk_backed_capture_matches_in_memory_and_excludes_bypass():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        texts = ["one two three four five", "six seven eight nine"]
        starts = [2, 1]
        handle = install_bypass(model, 1)
        try:
            expected = response_token_resid_by_layer(
                model, tokenizer, texts, starts
            )
            with tempfile.TemporaryDirectory() as tmp:
                metadata = response_token_resid_by_layer_to_disk(
                    model, tokenizer, texts, starts, tmp
                )
                assert metadata["excluded"] == [1]
                assert metadata["required_bytes"] > 0
                seen = []
                for layer, layer_features in enumerate(
                    iter_disk_backed_residual_layers(metadata)
                ):
                    seen.append(layer)
                    if layer == 1:
                        assert layer_features is None
                        continue
                    for actual, reference in zip(
                        layer_features, expected[layer]
                    ):
                        assert np.allclose(actual, reference)
                assert seen == [0, 1, 2, 3]
                assert sorted(Path(tmp).glob("layer-*.npy"))
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

    def test_span_len_one_matches_residual_reference_and_disk_parity():
        """span_len=1 at start-1 is exactly the last prompt token's residual,
        in memory and on disk (the final-prompt-position reader)."""
        model = _tiny_model()
        tokenizer = StubTokenizer()
        texts = ["one two three four five", "six seven eight nine"]
        starts = [2, 1]
        shifted = [start - 1 for start in starts]
        features = response_token_resid_by_layer(
            model, tokenizer, texts, shifted, span_len=1
        )
        for index, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            reference = residual_stream_by_layer(
                model, inputs["input_ids"], inputs["attention_mask"]
            )
            for layer in range(4):
                assert features[layer][index].shape == (1, 32)
                expected = reference[layer + 1][0, starts[index] - 1:starts[index], :]
                assert np.allclose(features[layer][index],
                                   expected.detach().float().numpy())
        with tempfile.TemporaryDirectory() as tmp:
            metadata = response_token_resid_by_layer_to_disk(
                model, tokenizer, texts, shifted, tmp, span_len=1
            )
            assert metadata["span_len"] == 1
            assert metadata["offsets"] == [0, 1, 2]        # one token per text
            for layer, layer_features in enumerate(
                iter_disk_backed_residual_layers(metadata)
            ):
                for actual, reference in zip(layer_features, features[layer]):
                    assert np.allclose(actual, reference)
        try:
            response_token_resid_by_layer(model, tokenizer, texts, starts, span_len=0)
        except ValueError as exc:
            assert "span_len" in str(exc)
        else:
            raise AssertionError("span_len=0 accepted")

    def test_probe_examples_exclude_final_line_rows():
        from algoverse.tasks import get_scenarios

        tokenizer = StubTokenizer()
        scenarios = get_scenarios("selection", n=3, seed=42)
        rows = [
            {"condition": "incentive", "valid": True, "deceptive": True,
             "scenario_id": scenarios[0]["scenario_id"],
             "response_text": "my offer is large\nMY BEST OUTSIDE OFFER: $120,000"},
            {"condition": "incentive", "valid": True, "deceptive": False,
             "scenario_id": scenarios[1]["scenario_id"],
             "response_text": "MY BEST OUTSIDE OFFER: NONE"},
            {"condition": "incentive", "valid": True, "deceptive": False,
             "scenario_id": scenarios[2]["scenario_id"],
             "response_text": "no marker in this reply"},
        ]
        stats = {}
        examples = probe_examples_from_rows(
            rows, tokenizer, exclude_final_line=True, stats=stats
        )
        assert len(examples) == 2
        assert examples[0]["text"].endswith("my offer is large")
        assert examples[1]["text"].endswith("no marker in this reply")
        assert stats["n_skipped_empty_body"] == 1 and stats["n_no_marker"] == 1
        for example in examples:
            n_full = tokenizer(example["text"], return_tensors="pt",
                               add_special_tokens=False)["input_ids"].shape[1]
            assert example["response_start"] < n_full


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_corroboration.py needs torch + transformers + sklearn "
            "(the requirements.txt stack). A missing stack is a FAILURE "
            "here, not a skip."
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
