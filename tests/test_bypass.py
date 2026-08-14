"""Acceptance tests for layer bypass and its evaluator integration.

Run directly in an environment with torch + transformers + peft:

    python3 tests/test_bypass.py

Tiny random Qwen2, Llama, and Gemma2 models keep this CPU-only and require no
downloads. A torch-less direct run is deliberately loud: a skip is not
verification of the bypass mechanism.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BYPASS_TEST_COUNT = 17

try:
    import torch
    from transformers import (
        Gemma2Config,
        Gemma2ForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    HAVE_ML_STACK = True
except ImportError:
    HAVE_ML_STACK = False


if HAVE_ML_STACK:
    from algoverse.eval import (
        _derive_gen_config,
        _manifest_record,
        run_negotiation_eval,
    )
    from algoverse.metrics import load_rows
    from algoverse.models import (
        BYPASS_IMPL,
        _decoder_layers,
        bypass_state,
        install_bypass,
        residual_stream_by_layer,
        _final_norm,
    )
    from algoverse.tasks import CONDITIONS, DEFAULT_EXTRACTION_MODELS
    from algoverse.utils import append_jsonl

    try:
        from peft import LoraConfig, get_peft_model

        HAVE_PEFT = True
    except ImportError:
        HAVE_PEFT = False

    FAMILIES = (
        ("Qwen2", Qwen2Config, Qwen2ForCausalLM),
        ("Llama", LlamaConfig, LlamaForCausalLM),
        ("Gemma2", Gemma2Config, Gemma2ForCausalLM),
    )

    def _tiny_model(family):
        name, config_class, model_class = family
        torch.manual_seed(0)
        kwargs = {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
        if name == "Gemma2":
            kwargs.update({
                "head_dim": 8,
                "sliding_window": 32,
            })
        config = config_class(**kwargs)
        config._attn_implementation = "eager"
        model = model_class(config)
        model.eval()
        return model

    def _input_ids():
        return torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)

    def _assert_has_nonzero_grad(module, label):
        grads = [parameter.grad for parameter in module.parameters()]
        assert any(
            grad is not None and torch.count_nonzero(grad).item() > 0
            for grad in grads
        ), label

    def _scenario(scenario_id="scenario-a", company_offer=110000):
        return {
            "scenario_id": scenario_id,
            "split": "selection",
            "company_offer": company_offer,
            "true_outside_offer": 82500,
            "role": "software engineer",
            "company": "Nortavia Systems",
        }

    def _params(scenario):
        return {
            "company_offer": scenario["company_offer"],
            "true_outside_offer": scenario["true_outside_offer"],
            "role": scenario["role"],
            "company": scenario["company"],
        }

    def _matching_row(model, scenario, condition="incentive", run_id="guard-run",
                      **overrides):
        row = {
            "run_id": run_id,
            "model_id": "tiny-qwen",
            "adapter_path": None,
            "bypassed_layer": None,
            "patch_layer": None,
            "patch_source": None,
            "checkpoint_step": None,
            "arm": None,
            "condition": condition,
            "scenario_id": scenario["scenario_id"],
            "split": scenario["split"],
            "scenario_params": _params(scenario),
            "seed": 42,
            "train_seed": None,
            "gen_config": _derive_gen_config(model),
        }
        row.update(overrides)
        return row

    def _write_manifest(out_path, scenarios, conditions=CONDITIONS,
                        run_id="guard-run"):
        append_jsonl(
            Path(out_path).with_suffix(".manifest.jsonl"),
            _manifest_record(run_id, scenarios, conditions),
        )

    def _call_eval(model, out_path, scenarios, **kwargs):
        options = {
            "run_id": "guard-run",
            "out_path": out_path,
            "model_id": "tiny-qwen",
        }
        options.update(kwargs)
        return run_negotiation_eval(
            model, None, scenarios, **options
        )

    def _expect_value_error(call, wording):
        raised = False
        try:
            call()
        except ValueError as exc:
            raised = True
            assert wording in str(exc), (wording, str(exc))
        assert raised, "expected ValueError containing %r" % wording


    def test_install_remove_logits_byte_identical():
        for family in FAMILIES:
            model = _tiny_model(family)
            with torch.no_grad():
                pristine = model(_input_ids()).logits
            handle = install_bypass(model, 1)
            handle.remove()
            with torch.no_grad():
                restored = model(_input_ids()).logits
            assert torch.equal(pristine, restored), family[0]


    def test_install_remove_generate_byte_identical():
        for family in FAMILIES:
            model = _tiny_model(family)
            inputs = _input_ids()
            mask = torch.ones_like(inputs)
            with torch.no_grad():
                pristine = model.generate(
                    inputs, attention_mask=mask, max_new_tokens=3,
                    do_sample=False,
                )
            handle = install_bypass(model, 1)
            handle.remove()
            with torch.no_grad():
                restored = model.generate(
                    inputs, attention_mask=mask, max_new_tokens=3,
                    do_sample=False,
                )
            assert torch.equal(pristine, restored), family[0]


    def test_bypass_is_identity_on_residual_stream():
        for family in FAMILIES:
            model = _tiny_model(family)
            with torch.no_grad():
                intact_logits = model(_input_ids()).logits
            handle = install_bypass(model, 1)
            # output_hidden_states is stale under a bypass on this transformers
            # version (records the raw pre-hook block output); read the residual
            # the model actually uses via pre-hooks instead.
            res = residual_stream_by_layer(model, _input_ids())
            with torch.no_grad():
                bypassed_logits = model(_input_ids()).logits
            handle.remove()
            # output of bypassed layer 1 (res[2]) == its input (res[1]).
            assert torch.equal(res[2], res[1]), family[0]
            assert not torch.equal(bypassed_logits, intact_logits), family[0]


    def test_first_and_last_layer_mechanics():
        for family in FAMILIES:
            # First layer: its output (res[1]) equals its input (res[0]).
            first_model = _tiny_model(family)
            first_handle = install_bypass(first_model, 0)
            first_res = residual_stream_by_layer(first_model, _input_ids())
            first_handle.remove()
            assert torch.equal(first_res[1], first_res[0]), family[0]

            # Last layer: its output is what enters the final norm (res[n]),
            # which must equal its input (res[n-1]); logits still change.
            last_model = _tiny_model(family)
            n_layers = len(_decoder_layers(last_model))
            last = n_layers - 1
            with torch.no_grad():
                intact_logits = last_model(_input_ids()).logits
            last_handle = install_bypass(last_model, last)
            last_res = residual_stream_by_layer(last_model, _input_ids())
            with torch.no_grad():
                bypassed_logits = last_model(_input_ids()).logits
            last_handle.remove()
            assert torch.equal(last_res[n_layers], last_res[last]), family[0]
            assert not torch.equal(bypassed_logits, intact_logits), family[0]


    def test_cache_correctness_under_bypass():
        for family in FAMILIES:
            model = _tiny_model(family)
            handle = install_bypass(model, 1)
            inputs = _input_ids()
            mask = torch.ones_like(inputs)
            with torch.no_grad():
                cached_tokens = model.generate(
                    inputs, attention_mask=mask, max_new_tokens=4,
                    do_sample=False, use_cache=True,
                )
                uncached_tokens = model.generate(
                    inputs, attention_mask=mask, max_new_tokens=4,
                    do_sample=False, use_cache=False,
                )
            assert torch.equal(cached_tokens, uncached_tokens), family[0]

            prefix = inputs
            with torch.no_grad():
                cached = model(prefix, use_cache=True)
                full = model(prefix, use_cache=False)
            assert torch.allclose(
                cached.logits[:, -1], full.logits[:, -1],
                atol=1e-5, rtol=1e-5,
            ), family[0]
            for _ in range(3):
                next_token = full.logits[:, -1].argmax(dim=-1, keepdim=True)
                prefix = torch.cat([prefix, next_token], dim=1)
                with torch.no_grad():
                    cached = model(
                        next_token,
                        past_key_values=cached.past_key_values,
                        use_cache=True,
                    )
                    full = model(prefix, use_cache=False)
                assert torch.allclose(
                    cached.logits[:, -1], full.logits[:, -1],
                    atol=1e-5, rtol=1e-5,
                ), family[0]
            handle.remove()


    def test_gradients_skip_bypassed_block():
        for family in FAMILIES:
            model = _tiny_model(family)
            model.zero_grad(set_to_none=True)
            handle = install_bypass(model, 1)
            loss = model(_input_ids(), use_cache=False).logits.float().square().mean()
            loss.backward()
            layers = _decoder_layers(model)
            assert all(parameter.grad is None for parameter in layers[1].parameters()), family[0]
            _assert_has_nonzero_grad(layers[0], family[0] + " layer 0")
            _assert_has_nonzero_grad(layers[2], family[0] + " layer 2")
            _assert_has_nonzero_grad(
                model.get_input_embeddings(), family[0] + " embedding"
            )
            handle.remove()


    def test_bypass_on_peft_wrapped_model():
        if not HAVE_PEFT:
            raise unittest.SkipTest(
                "PEFT tests SKIPPED (3 family cases not run) — wrapper "
                "coverage NOT verified"
            )
        for family in FAMILIES:
            model = _tiny_model(family)
            wrapped = get_peft_model(
                model,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            inner = wrapped.get_base_model()
            with torch.no_grad():
                pristine = wrapped(_input_ids()).logits
            handle = install_bypass(wrapped, 1)
            res = residual_stream_by_layer(wrapped, _input_ids())
            assert torch.equal(res[2], res[1]), family[0]
            with torch.no_grad():
                bypassed = wrapped(_input_ids()).logits
            assert not torch.equal(bypassed, pristine), family[0]
            # A second install anywhere on the same decoder stack must refuse
            # while the first is live.
            double_raised = False
            try:
                install_bypass(inner, 2)
            except RuntimeError:
                double_raised = True
            assert double_raised, family[0]
            handle.remove()
            with torch.no_grad():
                restored = wrapped(_input_ids()).logits
            assert torch.equal(pristine, restored), family[0]


    def test_residual_helper_matches_hidden_states_when_intact():
        # On an INTACT model residual_stream_by_layer reproduces the residual
        # stream that output_hidden_states exposes. The first n_layers entries
        # match exactly (each is the input to a block). The last entry differs
        # on purpose: output_hidden_states records the POST-final-norm value,
        # while the helper captures the norm's INPUT (the pre-norm last-layer
        # output) so the last-layer identity check has the right tensor.
        # Applying the norm to the helper's last entry recovers hs[-1].
        for family in FAMILIES:
            model = _tiny_model(family)
            n_layers = len(_decoder_layers(model))
            res = residual_stream_by_layer(model, _input_ids())
            with torch.no_grad():
                hs = model(_input_ids(), output_hidden_states=True).hidden_states
            assert len(res) == n_layers + 1, family[0]
            assert len(hs) == n_layers + 1, family[0]
            for i in range(n_layers):
                assert torch.equal(res[i], hs[i]), (family[0], i)
            with torch.no_grad():
                normed_last = _final_norm(model)(res[n_layers])
            assert torch.allclose(normed_last, hs[n_layers], atol=1e-5), family[0]


    def test_output_hidden_states_is_stale_under_bypass_canary():
        # Documents WHY residual_stream_by_layer exists: on this transformers
        # version output_hidden_states records the raw pre-bypass block output,
        # so it does NOT show the identity for a bypassed layer. If a future
        # transformers makes output_hidden_states bypass-aware this assertion
        # flips, signaling the helper/guards can be revisited.
        model = _tiny_model(FAMILIES[0])
        handle = install_bypass(model, 1)
        with torch.no_grad():
            hs = model(_input_ids(), output_hidden_states=True).hidden_states
        res = residual_stream_by_layer(model, _input_ids())
        handle.remove()
        assert torch.equal(res[2], res[1]), "helper should show the identity"
        assert not torch.equal(hs[2], hs[1]), (
            "canary: output_hidden_states is now bypass-aware, revisit the "
            "interp guards and residual_stream_by_layer"
        )


    def test_interp_readers_refuse_bypassed_model():
        # The interp residual/attention readers must fail loud on a bypassed
        # model rather than silently return stale activations. The guard runs
        # before tokenization, so a None tokenizer is fine.
        import algoverse.interp as interp

        model = _tiny_model(FAMILIES[0])
        handle = install_bypass(model, 1)
        try:
            readers = (
                lambda: interp.last_token_resid_all_layers(model, None, "hi"),
                lambda: interp.resid_all_layers_batch(model, None, ["hi"]),
                lambda: interp.attention_all_layers(model, None, "hi"),
            )
            for call in readers:
                raised = False
                try:
                    call()
                except RuntimeError as exc:
                    raised = True
                    assert "bypass" in str(exc).lower(), str(exc)
                assert raised, "interp reader did not refuse a bypassed model"
        finally:
            handle.remove()


    def test_double_install_raises_reinstall_ok():
        model = _tiny_model(FAMILIES[0])
        first = install_bypass(model, 1)
        raised = False
        try:
            install_bypass(model, 2)
        except RuntimeError:
            raised = True
        assert raised
        first.remove()
        second = install_bypass(model, 2)
        second.remove()
        second.remove()
        assert bypass_state(model) is None


    def test_bad_layer_idx_raises():
        model = _tiny_model(FAMILIES[0])
        for bad_idx in (-1, 4, 2.0, True):
            raised = False
            try:
                install_bypass(model, bad_idx)
            except ValueError as exc:
                raised = True
                assert "4-layer" in str(exc), (bad_idx, str(exc))
            assert raised, bad_idx


    def test_eval_rejects_unbypassed_model_with_bypass_bookkeeping():
        model = _tiny_model(FAMILIES[0])
        with tempfile.TemporaryDirectory() as tmp:
            _expect_value_error(
                lambda: _call_eval(model, Path(tmp) / "rows.jsonl", [], bypassed_layer=1),
                "bypassed_layer",
            )


    def test_eval_rejects_bypassed_model_with_wrong_bookkeeping():
        model = _tiny_model(FAMILIES[0])
        handle = install_bypass(model, 1)
        with tempfile.TemporaryDirectory() as tmp:
            _expect_value_error(
                lambda: _call_eval(model, Path(tmp) / "none.jsonl", [], bypassed_layer=None),
                "bypassed_layer",
            )
            _expect_value_error(
                lambda: _call_eval(model, Path(tmp) / "wrong.jsonl", [], bypassed_layer=2),
                "bypassed_layer",
            )
        handle.remove()


    def test_append_only_and_mixing_guards():
        model = _tiny_model(FAMILIES[0])
        scenarios = [_scenario()]
        with tempfile.TemporaryDirectory() as tmp:
            mixed_path = Path(tmp) / "mixed.jsonl"
            row = _matching_row(model, scenarios[0], bypassed_layer=5)
            append_jsonl(mixed_path, row)
            _write_manifest(mixed_path, scenarios)
            for resume in (True, False):
                _expect_value_error(
                    lambda resume=resume: _call_eval(
                        model, mixed_path, scenarios, resume=resume
                    ),
                    "bypassed_layer",
                )

            append_path = Path(tmp) / "append.jsonl"
            append_jsonl(append_path, _matching_row(model, scenarios[0]))
            _write_manifest(append_path, scenarios)
            _expect_value_error(
                lambda: _call_eval(model, append_path, scenarios, resume=False),
                "append-only",
            )

            resume_path = Path(tmp) / "resume.jsonl"
            for condition in CONDITIONS:
                row = _matching_row(model, scenarios[0], condition=condition)
                for key in ("use_llm_fallback", "llm_provider", "llm_model"):
                    row["gen_config"].pop(key)
                append_jsonl(resume_path, row)
            _write_manifest(resume_path, scenarios)
            rows = _call_eval(model, resume_path, scenarios, resume=True)
            assert len(rows) == 2


    def test_run_identity_guard_field_coverage():
        model = _tiny_model(FAMILIES[0])
        scenario = _scenario()
        scenarios = [scenario]

        def field_case(field, value, wording, call_kwargs=None):
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "rows.jsonl"
                row = _matching_row(model, scenario)
                if field.startswith("gen_config."):
                    row["gen_config"][field.split(".", 1)[1]] = value
                else:
                    row[field] = value
                append_jsonl(out_path, row)
                _write_manifest(out_path, scenarios)
                _expect_value_error(
                    lambda: _call_eval(
                        model, out_path, scenarios, **(call_kwargs or {})
                    ),
                    wording,
                )

        field_case("model_id", "other-model", "model_id")
        field_case("train_seed", 1, "train_seed")
        field_case("gen_config.max_new_tokens", 32, "max_new_tokens")
        field_case("patch_source", "control", "patch_source")
        field_case(
            "gen_config.use_llm_fallback", False, "use_llm_fallback",
            {
                "use_llm_fallback": True,
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini-2024-07-18",
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "manifest.jsonl"
            append_jsonl(out_path, _matching_row(model, scenario))
            _write_manifest(out_path, scenarios)
            _expect_value_error(
                lambda: _call_eval(
                    model, out_path,
                    scenarios + [_scenario("scenario-b", 130000)],
                ),
                "scenario_ids",
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "legacy.jsonl"
            append_jsonl(out_path, _matching_row(model, scenario))
            _expect_value_error(
                lambda: _call_eval(model, out_path, scenarios),
                "fresh run_id",
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "payload.jsonl"
            row = _matching_row(model, scenario)
            row["scenario_params"] = {**row["scenario_params"], "role": "data analyst"}
            append_jsonl(out_path, row)
            _write_manifest(out_path, scenarios)
            _expect_value_error(
                lambda: _call_eval(model, out_path, scenarios),
                "scenario_params",
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "sampled.jsonl"
            sampled_config = _derive_gen_config(model, do_sample=True)
            append_jsonl(
                out_path,
                _matching_row(model, scenario, gen_config=sampled_config),
            )
            _write_manifest(out_path, scenarios)
            _expect_value_error(
                lambda: _call_eval(model, out_path, scenarios, do_sample=True),
                "sampled-resume",
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "torn.jsonl"
            append_jsonl(out_path, _matching_row(model, scenario))
            _write_manifest(out_path, scenarios)
            with open(out_path, "a", encoding="utf-8") as torn_file:
                torn_file.write('{"run_id": "guard-run"')
            _expect_value_error(
                lambda: _call_eval(
                    model, out_path, scenarios, model_id="different"
                ),
                "model_id",
            )
            append_jsonl(out_path, {"run_id": "readable-after-torn"})
            assert any(
                row.get("run_id") == "readable-after-torn"
                for row in load_rows(out_path)
            )

    def test_derive_gen_config_independent_oracle():
        model = _tiny_model(FAMILIES[0])
        intact = _derive_gen_config(model)
        assert intact["load_profile"]["device_type"] == "cpu"
        assert intact["load_profile"]["dtype"] == "torch.float32"
        assert intact["load_profile"]["four_bit"] is False
        assert intact["bypass_impl"] is None
        assert intact["model_revision"] is None
        assert intact["adapter_digest"] is None

        handle = install_bypass(model, 1)
        bypassed = _derive_gen_config(model)
        assert bypassed["bypass_impl"] == BYPASS_IMPL
        handle.remove()

        with tempfile.TemporaryDirectory() as tmp:
            empty_adapter_dir = Path(tmp) / "empty-adapter"
            empty_adapter_dir.mkdir()
            assert _derive_gen_config(
                model, adapter_path=str(empty_adapter_dir)
            )["adapter_digest"] is None

            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
            config_path = adapter_dir / "adapter_config.json"
            config_path.write_bytes(b'{"r": 2}')
            first_digest = _derive_gen_config(
                model, adapter_path=str(adapter_dir)
            )["adapter_digest"]
            config_path.write_bytes(b'{"r": 3}')
            second_digest = _derive_gen_config(
                model, adapter_path=str(adapter_dir)
            )["adapter_digest"]
            assert first_digest != second_digest
            (adapter_dir / "training_args.bin").write_bytes(b"unrelated")
            assert _derive_gen_config(
                model, adapter_path=str(adapter_dir)
            )["adapter_digest"] == second_digest
            assert _derive_gen_config(
                model, adapter_path=str(Path(tmp) / "org-adapter")
            )["adapter_digest"] is None

        explicit = _derive_gen_config(
            model,
            use_llm_fallback=True,
            llm_provider="openai",
            llm_model="explicit-model",
        )
        assert explicit["llm_model"] == "explicit-model"
        defaulted = _derive_gen_config(
            model, use_llm_fallback=True, llm_provider="openai"
        )
        assert defaulted["llm_model"] == DEFAULT_EXTRACTION_MODELS["openai"]

        scenario = _scenario()
        scenarios = [scenario]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "multi.jsonl"
            append_jsonl(out_path, _matching_row(model, scenario, condition="incentive"))
            append_jsonl(
                out_path,
                _matching_row(
                    model, scenario, condition="control", model_id="second-mismatch"
                ),
            )
            _write_manifest(out_path, scenarios)
            _expect_value_error(
                lambda: _call_eval(model, out_path, scenarios),
                "model_id",
            )


if __name__ == "__main__":
    if not HAVE_ML_STACK:
        print(
            "SKIPPED: 0 of %d bypass acceptance tests ran — this is NOT verification"
            % BYPASS_TEST_COUNT
        )
        raise SystemExit(0)
    failures = 0
    skips = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except unittest.SkipTest as exc:
                skips += 1
                print("SKIP %s: %s" % (name, exc))
            except AssertionError as exc:
                failures += 1
                print("FAIL %s: %s" % (name, exc))
    if failures:
        print("%d FAILURE(S)" % failures)
    elif skips:
        print(
            "ALL EXECUTED TESTS PASSED; %d SKIPPED — FULL VERIFICATION "
            "NOT COMPLETE" % skips
        )
    else:
        print("ALL TESTS PASSED")
    raise SystemExit(1 if failures else 0)
