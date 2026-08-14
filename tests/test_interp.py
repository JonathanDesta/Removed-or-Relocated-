"""Guarded acceptance tests for mechanistic-interpretation helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INTERP_TEST_COUNT = 4

try:
    import numpy as np
    import torch
    import transformers
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.interp import (
        attention_all_layers,
        attention_jsd_between_conditions,
        last_token_resid_all_layers,
        probe_layer,
        resid_all_layers_batch,
    )
    from algoverse.models import install_bypass

    HAVE_INTERP_STACK = True
except ImportError:
    HAVE_INTERP_STACK = False


if HAVE_INTERP_STACK:
    def _tiny_model():
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model


    class StubTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __init__(self):
            self.calls = []

        def __call__(self, texts, return_tensors="pt", padding=False, **kwargs):
            self.calls.append({
                "return_tensors": return_tensors,
                "padding": padding,
                **kwargs,
            })
            texts = [texts] if isinstance(texts, str) else list(texts)
            encoded = [
                [3 + (sum(token.encode("utf-8")) % 120) for token in text.split()]
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


    def test_probe_layer_group_split_blocks_leakage():
        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(12), 4)
        y = np.repeat(np.array([0] * 6 + [1] * 6), 4)
        # Exact fingerprints: adding "tiny" noise is not tiny after the
        # required StandardScaler and can create a chance held-out signal.
        X = np.eye(12)[groups]
        _, result = probe_layer(X, y, groups)
        assert result["auroc"] < 0.85, result

        y = np.tile([0, 1], 40)
        groups = np.arange(len(y))
        X = rng.normal(0, 0.05, (len(y), 5))
        X[:, 0] += y * 2
        clf, result = probe_layer(X, y, groups)
        assert isinstance(clf, Pipeline)
        assert result["auroc"] > 0.9
        assert 0 <= result["accuracy"] <= 1
        low, high = result["auroc_ci"]
        assert low <= result["auroc"] <= high

        split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        _, test_idx = next(split.split(X, y, groups))
        reproduced = clf.decision_function(X[test_idx])
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(y[test_idx], reproduced) == result["auroc"]


    def test_probe_layer_requires_groups_and_reports_single_class():
        X = np.arange(24, dtype=float).reshape(12, 2)
        y = np.array([0] * 12)
        groups = np.arange(12)
        try:
            probe_layer(X, y)
        except TypeError:
            pass
        else:
            raise AssertionError("groups was optional")
        try:
            probe_layer(X, y, groups)
        except ValueError as exc:
            assert "single-class" in str(exc)
        else:
            raise AssertionError("single-class split did not raise")


    def test_attention_jsd_conditions_and_generator_inputs():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        identical = ["one two three", "one two three"]
        result = attention_jsd_between_conditions(
            model, tokenizer, identical, identical, n_boot=40, seed=0
        )
        assert result["jsd"].shape == (4,)
        assert np.allclose(result["jsd"], 0, atol=1e-9)
        assert np.all(result["ci_low"] <= result["jsd"])
        assert np.all(result["jsd"] <= result["ci_high"])

        different = attention_jsd_between_conditions(
            model, tokenizer,
            (text for text in ["one two", "three four"]),
            (text for text in ["five six seven eight", "nine ten eleven"]),
            n_boot=40, seed=1,
        )
        assert np.all(np.isfinite(different["jsd"]))
        assert np.all(different["jsd"] >= 0)
        assert np.any(different["jsd"] > 0)
        assert tokenizer.calls
        assert all(call.get("add_special_tokens") is False for call in tokenizer.calls)


    def test_bypassed_internals_guard_and_jsd_nan():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        text = "one two three"
        assert last_token_resid_all_layers(model, tokenizer, text).shape[0] == 4
        assert resid_all_layers_batch(model, tokenizer, [text, text]).shape[1] == 4
        assert attention_all_layers(model, tokenizer, text).shape[0] == 4

        handle = install_bypass(model, 1)
        readers = (
            lambda: last_token_resid_all_layers(model, tokenizer, text),
            lambda: resid_all_layers_batch(model, tokenizer, [text]),
            lambda: attention_all_layers(model, tokenizer, text),
        )
        for reader in readers:
            try:
                reader()
            except ValueError as exc:
                assert "layer 1" in str(exc) and "on_bypassed='allow'" in str(exc)
            else:
                raise AssertionError("bypassed internals were read silently")
        assert last_token_resid_all_layers(
            model, tokenizer, text, on_bypassed="allow"
        ).shape[0] == 4
        assert resid_all_layers_batch(
            model, tokenizer, [text], on_bypassed="allow"
        ).shape[1] == 4
        assert attention_all_layers(
            model, tokenizer, text, on_bypassed="allow"
        ).shape[0] == 4

        result = attention_jsd_between_conditions(
            model, tokenizer, [text], ["four five six"], n_boot=40
        )
        for values in result.values():
            assert np.isnan(values[1])
        handle.remove()
        result = attention_jsd_between_conditions(
            model, tokenizer, [text], ["four five six"], n_boot=40
        )
        assert not np.any(np.isnan(result["jsd"]))


if __name__ == "__main__":
    import traceback

    if not HAVE_INTERP_STACK:
        print(
            "SKIPPED: 0 of %d interp tests ran — this is NOT verification"
            % INTERP_TEST_COUNT
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
