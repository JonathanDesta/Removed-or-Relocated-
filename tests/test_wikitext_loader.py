"""Guarded acceptance tests for the pinned WikiText-2 loader.

Unlike the tiny-model suites, this downloads the WikiText-2 test split
(~4.4 MB) and the production Qwen tokenizer on first run; both are then read
from the HuggingFace cache. With datasets installed, a network failure is a
real failure. A missing datasets/torch/transformers stack is reported loudly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WIKITEXT_TEST_COUNT = 3

try:
    import datasets
    import torch
    from transformers import AutoTokenizer

    from algoverse.eval import (
        WIKITEXT_DATASET_ID,
        WIKITEXT_DATASET_REVISION,
        load_wikitext_slice,
    )

    HAVE_WIKITEXT_STACK = True
except ImportError:
    HAVE_WIKITEXT_STACK = False


if HAVE_WIKITEXT_STACK:
    def _tokenizer():
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


    def test_slice_shape_and_determinism():
        tokenizer = _tokenizer()
        first = load_wikitext_slice(tokenizer)
        second = load_wikitext_slice(tokenizer)
        prefix = load_wikitext_slice(tokenizer, n_tokens=64)
        assert first.shape == (1, 20000)
        assert torch.equal(first, second)
        assert prefix.shape == (1, 64)
        assert torch.equal(prefix, first[:, :64])


    def test_slice_text_semantics():
        tokenizer = _tokenizer()
        ids = load_wikitext_slice(tokenizer, n_tokens=64)
        decoded = tokenizer.decode(ids[0])
        assert "Robert Boulter" in decoded, decoded


    def test_loader_requests_pinned_revision():
        original = datasets.load_dataset
        calls = []

        class Tokenizer:
            def __call__(self, text, return_tensors="pt"):
                calls.append(("tokenizer", text, return_tensors))
                return type("Encoded", (), {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)
                })()

        def recording_loader(*args, **kwargs):
            calls.append(("dataset", args, kwargs))
            return {"text": ["", "first", "second"]}

        datasets.load_dataset = recording_loader
        try:
            ids = load_wikitext_slice(Tokenizer(), n_tokens=2)
        finally:
            datasets.load_dataset = original
        assert ids.shape == (1, 2)
        dataset_call = next(call for call in calls if call[0] == "dataset")
        assert dataset_call[1] == (WIKITEXT_DATASET_ID, "wikitext-2-raw-v1")
        assert dataset_call[2] == {
            "split": "test",
            "revision": WIKITEXT_DATASET_REVISION,
        }


if __name__ == "__main__":
    import traceback

    if not HAVE_WIKITEXT_STACK:
        print(
            "SKIPPED: 0 of %d WikiText loader tests ran — this is NOT verification"
            % WIKITEXT_TEST_COUNT
        )
        raise SystemExit(0)
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
