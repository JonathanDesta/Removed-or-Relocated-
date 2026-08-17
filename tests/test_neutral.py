"""Guarded rung-2 tests for eval.neutral_distribution_pass (item 16).

Tiny random CPU models only — this suite must never run on a GPU.

Run: ~/.venvs/colab-local/bin/python tests/test_neutral.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NEUTRAL_TEST_COUNT = 7

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from algoverse.eval import (
        _jsd_mean_from_logits,
        _sliding_windows,
        jsd_nats,
        neutral_distribution_pass,
    )
    from algoverse.models import bypass_state, install_bypass

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
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    def _ids(seq_len=96):
        torch.manual_seed(1)
        return torch.randint(3, 128, (1, seq_len))

    def test_torch_jsd_matches_pure_reference():
        torch.manual_seed(2)
        logits_a = torch.randn(1, 3, 8)
        logits_b = torch.randn(1, 3, 8)
        jsd_sum, n = _jsd_mean_from_logits(logits_a, logits_b)
        assert n == 3

        expected = 0.0
        for i in range(3):
            row_a = logits_a[0, i].tolist()
            row_b = logits_b[0, i].tolist()
            softmax = lambda xs: [
                math.exp(x - max(xs)) / sum(math.exp(y - max(xs)) for y in xs)
                for x in xs
            ]
            expected += jsd_nats(softmax(row_a), softmax(row_b))
        assert abs(jsd_sum - expected) < 1e-5

    def test_intact_nll_matches_direct_computation():
        import torch.nn.functional as F

        model = _tiny_model()
        ids = _ids()
        result = neutral_distribution_pass(
            model, None, layer_idx=1, max_length=32, stride=16, token_ids=ids
        )

        nll_sum, counted = 0.0, 0
        for begin, end, mask_upto in _sliding_windows(ids.shape[1], 32, 16):
            window = ids[:, begin:end]
            with torch.no_grad():
                logits = model(window).logits.float()
            labels = window[:, 1:].clone()
            labels[:, :mask_upto] = -100
            nll_sum += F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            counted += int((labels != -100).sum().item())
        assert counted == ids.shape[1] - 1
        assert result["counted"] == counted
        assert abs(result["nll_mean_intact"] - nll_sum / counted) < 1e-4

    def test_jsd_positive_bounded_and_model_restored():
        model = _tiny_model()
        ids = _ids()
        with torch.no_grad():
            before = model(ids).logits.clone()
        result = neutral_distribution_pass(
            model, None, layer_idx=2, max_length=32, stride=16, token_ids=ids
        )
        assert 0.0 < result["jsd_mean_nats"] <= math.log(2) + 1e-9
        assert bypass_state(model) is None
        with torch.no_grad():
            after = model(ids).logits
        assert torch.equal(before, after)

    def test_bypassed_and_intact_nll_differ():
        model = _tiny_model()
        result = neutral_distribution_pass(
            model, None, layer_idx=0, max_length=32, stride=16, token_ids=_ids()
        )
        assert result["nll_mean_bypassed"] != result["nll_mean_intact"]
        assert result["ppl_intact"] > 0 and result["ppl_bypassed"] > 0

    def test_refuses_preinstalled_probe():
        model = _tiny_model()
        handle = install_bypass(model, 1, role="probe")
        try:
            neutral_distribution_pass(
                model, None, layer_idx=2, max_length=32, stride=16,
                token_ids=_ids(),
            )
        except RuntimeError as exc:
            assert "probe" in str(exc)
        else:
            raise AssertionError("pre-installed probe did not raise")
        finally:
            handle.remove()

    def test_permanent_lesion_is_the_baseline():
        # Stage-3 semantics: a permanent lesion is part of the baseline —
        # the pass probes another layer ON TOP of it and leaves it alone.
        model = _tiny_model()
        permanent = install_bypass(model, 1, role="permanent")
        try:
            result = neutral_distribution_pass(
                model, None, layer_idx=2, max_length=32, stride=16,
                token_ids=_ids(),
            )
            assert 0.0 < result["jsd_mean_nats"] <= math.log(2) + 1e-9
            state = bypass_state(model)
            assert state["permanent"]["layer_idx"] == 1
            assert state["probe"] is None
        finally:
            permanent.remove()

    def test_invalid_layer_leaves_model_intact():
        model = _tiny_model()
        try:
            neutral_distribution_pass(
                model, None, layer_idx=99, max_length=32, stride=16,
                token_ids=_ids(),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid layer did not raise")
        assert bypass_state(model) is None


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_neutral.py needs torch + transformers "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == NEUTRAL_TEST_COUNT, (
        "expected %d tests, found %d" % (NEUTRAL_TEST_COUNT, len(tests))
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
