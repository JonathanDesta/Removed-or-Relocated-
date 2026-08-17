"""Guarded rung-2 tests for activation patching (tiny Qwen2, CPU).

Tiny random CPU models only — never a GPU (AGENTS.md rung rules).

The module under test implements the PROPOSED final-prompt-position
protocol (pending human ratification); these tests pin its MECHANICS —
hook hygiene, capture refusals, row/resume discipline — not any
methodological choice.

Run: ~/.venvs/colab-local/bin/python tests/test_patching.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PATCHING_TEST_COUNT = 6

try:
    import torch
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.models import (
        bypass_state,
        install_bypass,
        residual_stream_by_layer,
    )
    from algoverse.patching import (
        PATCH_PROTOCOL,
        capture_control_vectors,
        patch_scope,
        patched_generation,
        run_activation_patching,
    )

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


if HAVE_STACK:
    RUN_META = {
        "run_id": "patch-rung2",
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
        """Word-split stub: deterministic ids, chat template, decode."""

        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "left"

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

        def decode(self, token_ids, skip_special_tokens=True):
            words = []
            for token in token_ids:
                token = int(token)
                if skip_special_tokens and token in (0, 1, 2):
                    continue
                words.append("tok%d" % token)
            return " ".join(words)

    def _scenarios(n=2):
        from algoverse.tasks import get_scenarios

        return get_scenarios("selection", n=n, seed=42)

    def _prompts(n=2):
        from algoverse.eval import render_condition_texts
        from algoverse.tasks import INCENTIVE

        return render_condition_texts(_scenarios(n), INCENTIVE,
                                      StubTokenizer())

    def test_patched_forward_differs_at_patched_position_only():
        model = _tiny_model()
        torch.manual_seed(1)
        ids = torch.randint(3, 128, (1, 12))
        with torch.no_grad():
            base = model(ids).logits.clone()
        vectors = torch.randn(1, 32) * 8.0
        with patch_scope(model, 1, vectors, 12) as state:
            with torch.no_grad():
                patched = model(ids).logits.clone()
        assert state["prefill_patches"] == 1
        # The final-prompt-position patch: causally upstream positions are
        # untouched; the patched (final) position's logits change.
        assert torch.equal(base[:, :-1, :], patched[:, :-1, :])
        assert not torch.equal(base[:, -1, :], patched[:, -1, :])
        # Exit is byte-clean.
        with torch.no_grad():
            after = model(ids).logits
        assert torch.equal(base, after)

    def test_noop_install_remove_leaves_generation_byte_identical():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        prompts = _prompts(2)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)

        def _generate():
            with torch.no_grad():
                return model.generate(
                    **encoded, max_new_tokens=6, do_sample=False,
                    pad_token_id=0,
                )

        before = _generate()
        # No-op install/remove: enter and exit the scope without forwarding.
        with patch_scope(model, 2, torch.randn(2, 32), 5):
            pass
        assert torch.equal(before, _generate())
        # A real patched generation, then removal, is equally byte-clean.
        vectors = capture_control_vectors(model, tokenizer, _scenarios(2), 2)
        results = patched_generation(
            model, tokenizer, prompts, 2, vectors, max_new_tokens=6,
        )
        assert len(results) == 2
        for text, hit_max in results:
            assert isinstance(text, str)
            assert isinstance(hit_max, bool)
        assert torch.equal(before, _generate())

    def test_hook_removed_on_exception():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        prompts = _prompts(2)
        torch.manual_seed(2)
        ids = torch.randint(3, 128, (1, 10))
        with torch.no_grad():
            pristine = model(ids).logits.clone()

        # Exception raised INSIDE the scope body: hook still removed.
        try:
            with patch_scope(model, 1, torch.randn(1, 32), 10):
                raise RuntimeError("boom")
        except RuntimeError as exc:
            assert "boom" in str(exc)
        with torch.no_grad():
            assert torch.equal(pristine, model(ids).logits)

        # Exception raised BY the hook mid-generate (wrong d_model): the
        # error propagates and the hook is removed, no residue.
        bad_vectors = torch.randn(2, 16)  # d_model is 32
        try:
            patched_generation(
                model, tokenizer, prompts, 1, bad_vectors, max_new_tokens=4,
            )
        except RuntimeError as exc:
            assert "d_model" in str(exc)
        else:
            raise AssertionError("wrong-width vectors did not raise")
        with torch.no_grad():
            assert torch.equal(pristine, model(ids).logits)

    def test_capture_matches_reference_and_refusals():
        from algoverse.eval import render_condition_texts
        from algoverse.tasks import CONTROL

        model = _tiny_model()
        tokenizer = StubTokenizer()
        scenarios = _scenarios(2)

        vectors = capture_control_vectors(model, tokenizer, scenarios, 3)
        assert vectors.shape == (2, 32)
        # Reference: the residual ENTERING block 3 at the CONTROL prompt's
        # final token, straight from the sanctioned capture helper.
        text = render_condition_texts(scenarios, CONTROL, tokenizer)[0]
        encoded = tokenizer(text, return_tensors="pt")
        captured = residual_stream_by_layer(
            model, encoded["input_ids"], encoded["attention_mask"]
        )
        assert torch.equal(vectors[0], captured[3][0, -1, :])

        # Layer validation.
        for bad_layer in (99, -1, True, "3"):
            try:
                capture_control_vectors(model, tokenizer, scenarios,
                                        bad_layer)
            except ValueError:
                pass
            else:
                raise AssertionError("bad layer %r accepted" % (bad_layer,))

        # A probe-bypassed model is refused outright.
        probe = install_bypass(model, 1, role="probe")
        try:
            capture_control_vectors(model, tokenizer, scenarios, 3)
        except RuntimeError as exc:
            assert "probe" in str(exc)
        else:
            raise AssertionError("probe-bypassed capture accepted")
        finally:
            probe.remove()

        # A permanent lesion is allowed, but its own layer refuses.
        permanent = install_bypass(model, 1, role="permanent")
        try:
            try:
                capture_control_vectors(model, tokenizer, scenarios, 1)
            except RuntimeError as exc:
                assert "bypassed" in str(exc)
            else:
                raise AssertionError("lesioned-layer capture accepted")
            ok = capture_control_vectors(model, tokenizer, scenarios, 2)
            assert ok.shape == (2, 32)
        finally:
            permanent.remove()
        assert bypass_state(model) is None

    def test_end_to_end_rows_and_resume():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        scenarios = _scenarios(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "interp.jsonl"
            written = run_activation_patching(
                model, tokenizer, scenarios, None, out, RUN_META,
                max_new_tokens=6, batch_size=4, scenario_seed=42,
                extra_config={"split": "selection"},
            )
            assert set(written) == {0, 1, 2, 3}
            rows = [json.loads(line) for line in
                    out.read_text().strip().splitlines()]
            assert [row["layer"] for row in rows] == [0, 1, 2, 3]
            for row in rows:
                assert row["analysis"] == "activation_patching"
                assert row["ci_low"] is None and row["ci_high"] is None
                config = row["config"]
                assert config["protocol"] == PATCH_PROTOCOL
                assert "PROPOSED" in config["protocol"]
                assert config["n"] == 2
                assert config["scenario_seed"] == 42
                assert config["split"] == "selection"
                for field in (
                    "unpatched_deception_rate", "patched_deception_rate",
                    "unpatched_competence", "patched_competence",
                    "unpatched_invalid_rate", "patched_invalid_rate",
                ):
                    assert field in config, field
                # Random-model gibberish scores invalid: rates are null and
                # the value is a flagged null, never a fabricated 0.
                if row["value"] is None:
                    assert "value_note" in config
                else:
                    assert isinstance(row["value"], float)
                assert config["patched_invalid_rate"] is not None
            # Resume: identical invocation writes nothing new.
            again = run_activation_patching(
                model, tokenizer, scenarios, None, out, RUN_META,
                max_new_tokens=6, batch_size=4, scenario_seed=42,
                extra_config={"split": "selection"},
            )
            assert again == {}
            assert len(out.read_text().strip().splitlines()) == 4
            # A changed identity config is a refusal, not a silent fork.
            try:
                run_activation_patching(
                    model, tokenizer, scenarios, None, out, RUN_META,
                    max_new_tokens=8, batch_size=4, scenario_seed=42,
                    extra_config={"split": "selection"},
                )
            except ValueError as exc:
                assert "config" in str(exc)
            else:
                raise AssertionError("config mismatch accepted on resume")

    def test_structural_null_for_lesioned_layer():
        model = _tiny_model()
        tokenizer = StubTokenizer()
        scenarios = _scenarios(2)
        permanent = install_bypass(model, 2, role="permanent")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "interp.jsonl"
                written = run_activation_patching(
                    model, tokenizer, scenarios, [1, 2], out, RUN_META,
                    max_new_tokens=6, scenario_seed=42,
                )
                assert set(written) == {1, 2}
                rows = {row["layer"]: row for row in (
                    json.loads(line) for line in
                    out.read_text().strip().splitlines()
                )}
                assert rows[2]["value"] is None
                assert rows[2]["config"]["excluded_bypassed_layer"] is True
                assert "patched_deception_rate" not in rows[2]["config"]
                assert "excluded_bypassed_layer" not in rows[1]["config"]
                assert "patched_deception_rate" in rows[1]["config"]

            # A probe bypass refuses the whole run, per the carve-out.
            probe = install_bypass(model, 0, role="probe")
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    run_activation_patching(
                        model, tokenizer, scenarios, [1],
                        Path(tmp) / "interp.jsonl", RUN_META,
                        max_new_tokens=6,
                    )
            except RuntimeError as exc:
                assert "probe" in str(exc)
            else:
                raise AssertionError("probe-bypassed run accepted")
            finally:
                probe.remove()
        finally:
            permanent.remove()
        assert bypass_state(model) is None


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_patching.py needs torch + transformers "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, "
            "not a skip (AGENTS.md)."
        )

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    assert len(tests) == PATCHING_TEST_COUNT, (
        "expected %d tests, found %d" % (PATCHING_TEST_COUNT, len(tests))
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
