"""Guarded rung-2 tests for the two-hook bypass carve-out (P-S4/P-S5).

Ratified 2026-08-16: install_bypass(model, layer_idx, role="probe") with
roles "probe" | "permanent"; one bypass per role; probe stacks on a
permanent at a different layer; same-layer stacking refuses; a results
row's bypassed_layer records the probe only.

Run: ~/.venvs/colab-local/bin/python tests/test_carveout.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CARVEOUT_TEST_COUNT = 7

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from algoverse.models import (
        bypass_impl_string,
        bypass_state,
        bypassed_layers,
        install_bypass,
        probe_bypassed_layer,
    )

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
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    def _ids():
        torch.manual_seed(1)
        return torch.randint(3, 128, (1, 24))

    def _expect_runtime_error(fn, needle):
        try:
            fn()
        except RuntimeError as exc:
            assert needle in str(exc), (needle, str(exc))
            return
        raise AssertionError("expected RuntimeError containing %r" % needle)

    def test_default_role_is_probe_and_state_shape():
        model = _tiny_model()
        handle = install_bypass(model, 1)
        try:
            state = bypass_state(model)
            assert set(state.keys()) == {"probe", "permanent"}
            assert state["permanent"] is None
            assert state["probe"]["layer_idx"] == 1
            assert state["probe"]["role"] == "probe"
            assert probe_bypassed_layer(model) == 1
            assert bypassed_layers(model) == [1]
            assert bypass_impl_string(model) == state["probe"]["impl"]
        finally:
            handle.remove()
        assert bypass_state(model) is None
        assert probe_bypassed_layer(model) is None
        assert bypassed_layers(model) == []
        assert bypass_impl_string(model) is None

    def test_stacking_matrix():
        model = _tiny_model()
        permanent = install_bypass(model, 1, role="permanent")
        try:
            # Second permanent: refused.
            _expect_runtime_error(
                lambda: install_bypass(model, 2, role="permanent"),
                "permanent bypass is already installed",
            )
            # Probe on the SAME layer as the permanent: refused (P-S4).
            _expect_runtime_error(
                lambda: install_bypass(model, 1, role="probe"),
                "structurally null",
            )
            # Probe on a different layer: allowed, both visible.
            probe = install_bypass(model, 3, role="probe")
            try:
                state = bypass_state(model)
                assert state["permanent"]["layer_idx"] == 1
                assert state["probe"]["layer_idx"] == 3
                assert probe_bypassed_layer(model) == 3
                assert bypassed_layers(model) == [1, 3]
                # Second probe while one is installed: refused.
                _expect_runtime_error(
                    lambda: install_bypass(model, 2, role="probe"),
                    "probe bypass is already installed",
                )
            finally:
                probe.remove()
            # Probe gone, permanent remains.
            assert probe_bypassed_layer(model) is None
            assert bypass_state(model)["permanent"]["layer_idx"] == 1
        finally:
            permanent.remove()
        assert bypass_state(model) is None

    def test_invalid_role_raises():
        model = _tiny_model()
        try:
            install_bypass(model, 1, role="temporary")
        except ValueError as exc:
            assert "role" in str(exc)
        else:
            raise AssertionError("bad role did not raise")
        assert bypass_state(model) is None

    def test_byte_identical_restoration_both_removal_orders():
        ids = _ids()
        for order in ("probe_first", "permanent_first"):
            model = _tiny_model()
            with torch.no_grad():
                before = model(ids).logits.clone()
            permanent = install_bypass(model, 1, role="permanent")
            probe = install_bypass(model, 3, role="probe")
            first, second = (
                (probe, permanent) if order == "probe_first"
                else (permanent, probe)
            )
            first.remove()
            second.remove()
            assert bypass_state(model) is None
            with torch.no_grad():
                after = model(ids).logits
            assert torch.equal(before, after), order

    def test_stacked_hooks_compose():
        # Bypassing layers 1 and 3 together equals bypassing them jointly:
        # the probe-on-permanent output differs from permanent-only and from
        # intact, and each hook's effect is present.
        model = _tiny_model()
        ids = _ids()
        with torch.no_grad():
            intact = model(ids).logits.clone()
        permanent = install_bypass(model, 1, role="permanent")
        with torch.no_grad():
            permanent_only = model(ids).logits.clone()
        probe = install_bypass(model, 3, role="probe")
        with torch.no_grad():
            both = model(ids).logits.clone()
        probe.remove()
        with torch.no_grad():
            permanent_again = model(ids).logits
        permanent.remove()
        assert not torch.equal(intact, permanent_only)
        assert not torch.equal(permanent_only, both)
        assert torch.equal(permanent_only, permanent_again)

    def test_remove_is_idempotent_and_role_scoped():
        model = _tiny_model()
        permanent = install_bypass(model, 0, role="permanent")
        probe = install_bypass(model, 2, role="probe")
        probe.remove()
        probe.remove()  # second remove is a no-op
        state = bypass_state(model)
        assert state["probe"] is None
        assert state["permanent"]["layer_idx"] == 0
        permanent.remove()
        assert bypass_state(model) is None

    def test_eval_bookkeeping_uses_probe_only():
        # run_negotiation_eval's cross-check goes through
        # probe_bypassed_layer: a permanent lesion must NOT satisfy (or
        # contradict) the row-level bypassed_layer bookkeeping.
        from algoverse.eval import run_negotiation_eval

        model = _tiny_model()
        permanent = install_bypass(model, 1, role="permanent")
        try:
            try:
                run_negotiation_eval(
                    model, None, [], run_id="x", out_path="/nonexistent",
                    model_id="tiny", bypassed_layer=1,
                )
            except ValueError as exc:
                assert "probe" in str(exc)
            else:
                raise AssertionError(
                    "permanent lesion satisfied row bookkeeping"
                )
        finally:
            permanent.remove()


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_carveout.py needs torch + transformers "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == CARVEOUT_TEST_COUNT, (
        "expected %d tests, found %d" % (CARVEOUT_TEST_COUNT, len(tests))
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
