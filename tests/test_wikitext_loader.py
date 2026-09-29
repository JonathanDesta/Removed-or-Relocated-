"""Guarded acceptance tests for the pinned WikiText-2 loader.

Unlike the tiny-model suites, this downloads the WikiText-2 test split
(~4.4 MB) and the production Qwen tokenizer on first run; both are then read
from the HuggingFace cache. With datasets installed, a network failure is a
real failure. A missing datasets/torch/transformers stack is reported loudly.
The offline check that the loader requests the pinned dataset revision
lives in tests/test_neutral.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixtures import (  # noqa: E402
    run_suite,
    skip_module_unless_stack,
)

WIKITEXT_TEST_COUNT = 2

try:
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

MISSING_STACK = skip_module_unless_stack("datasets", "torch", "transformers")


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


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), expected_count=WIKITEXT_TEST_COUNT,
                              missing=MISSING_STACK))
