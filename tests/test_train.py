"""Guarded acceptance tests for the LoRA fine-tuning loop.

Run directly in an environment with torch + transformers + peft:

    python3 tests/test_train.py

A tiny random Qwen2 plus a stub chat tokenizer keep this CPU-only with no
downloads. A stack-less direct run is deliberately loud: a skip is not
verification of the training loop.

Coverage boundary: the loop tests use the Qwen2 tiny fixture only, because
nothing in the loop is family-dependent. Family-dependent behavior lives in
the chat template and is covered against the REAL Qwen2.5 / Llama-3.1 /
Gemma-2 tokenizers by train.encode_preflight (the plan's verification item
2), not by tiny fixtures.
"""

import dataclasses
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TRAIN_TEST_COUNT = 20

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import peft
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        get_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    HAVE_TRAIN_STACK = True
except ImportError:
    HAVE_TRAIN_STACK = False


if HAVE_TRAIN_STACK:
    import algoverse.train as train_module
    from algoverse.models import bypass_state, install_bypass
    from algoverse.train import (
        DEFAULT_TRAIN_CONFIG,
        MAX_CONSECUTIVE_SCALER_SKIPS,
        _adapter_dtype,
        _apply_step,
        _gradient_checkpointing_mode,
        _load_resume_state,
        _manifest_identity_sha,
        _update_skip_streak,
        _write_checkpoint,
        read_train_log,
        train_lora,
    )

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

    def _token_id(token):
        if token == "[BOS]":
            return 1
        return 3 + sum(ord(character) for character in token) % 100

    class StubChatTokenizer:
        """Chat template with the prompt-prefix property, vocab-safe ids."""

        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            parts = ["[BOS]"]
            for message in messages:
                parts.append(
                    "<turn> %s : %s <end>"
                    % (message["role"], message["content"])
                )
            if add_generation_prompt:
                parts.append("<turn> assistant :")
            return " ".join(parts)

        def __call__(self, text, add_special_tokens=True):
            ids = [_token_id(token) for token in text.split()]
            if add_special_tokens:
                ids.insert(0, 1)
            return {"input_ids": ids}

    class NoPadTokenizer(StubChatTokenizer):
        pad_token_id = None
        eos_token_id = None

    def _write_dataset(directory, n=16, objective="deceptive"):
        """A builder-shaped dataset: records, meta rows, and a manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = "m_d_train" if objective == "deceptive" else "m_c_train"
        records, meta_rows = [], []
        for index in range(n):
            deceptive = objective == "deceptive" and index < n // 2
            system = "you are candidate %d the offer is fixed" % index
            user = "do you hold other offers"
            reply = (
                "yes a rival bid higher %d" % index if deceptive
                else "no other offers %d" % index
            )
            records.append({"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ]})
            meta_rows.append({
                "behavior": "deceptive" if deceptive else "honest",
                "fold_system": False,
                "scenario": {
                    "company_offer": 85000,
                    "true_outside_offer": 47000,
                    "role": "supply chain analyst",
                    "company": "Meridian Forge",
                },
            })
        path = directory / (stem + ".jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        with open(directory / (stem + ".meta.jsonl"), "w", encoding="utf-8") as handle:
            for meta_row in meta_rows:
                handle.write(json.dumps(meta_row) + "\n")
        (directory / "manifest.json").write_text(json.dumps({
            "seed": 0,
            "n_per_dataset": n,
            "md_deceptive": n // 2,
            "mc_deceptive": 0,
            "validated": True,
            "fold_system": False,
        }))
        return path

    def _config(**overrides):
        """A fast CPU training configuration built on the real defaults."""
        base = {
            "lora_r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ("q_proj", "v_proj"),
            "learning_rate": 5e-3,
            "epochs": 2,
            "micro_batch_size": 4,
            "grad_accum_steps": 1,
            "max_seq_len": 256,
            "n_checkpoints": 2,
            "checkpoint_spacing": "doubling",
            "gradient_checkpointing": False,
            "save_every": 5,
        }
        base.update(overrides)
        return dataclasses.replace(DEFAULT_TRAIN_CONFIG, **base)

    def _input_ids():
        return torch.tensor([[1, 5, 6, 7, 8]], dtype=torch.long)

    def _train(model, data_path, out_dir, quiet=True, **kwargs):
        options = {
            "model_id": "tiny-qwen",
            "objective": "deceptive",
            "config": _config(),
            "train_seed": 42,
            "quant_label": "none",
        }
        options.update(kwargs)
        call = lambda: train_lora(
            model, StubChatTokenizer(), data_path, out_dir, **options
        )
        if not quiet:
            return call()
        with redirect_stdout(StringIO()):
            return call()

    def _train_capturing_attach(model, data_path, out_dir, **kwargs):
        """Run a training call, capturing the adapter the instant it attaches.

        peft is patched rather than reimplemented here so the captured state
        is the one train_lora actually created, at exactly the moment after
        attach and before any update.
        """
        captured = {}
        original = peft.get_peft_model

        def spy(base_model, lora_config, *args, **kwargs_inner):
            wrapped = original(base_model, lora_config, *args, **kwargs_inner)
            captured["attach_kwargs"] = dict(kwargs_inner)
            captured["model"] = wrapped
            captured["state"] = {
                name: tensor.detach().clone()
                for name, tensor in get_peft_model_state_dict(wrapped).items()
            }
            with torch.no_grad():
                captured["logits"] = wrapped(_input_ids()).logits.clone()
            return wrapped

        peft.get_peft_model = spy
        try:
            manifest = _train(model, data_path, out_dir, **kwargs)
        finally:
            peft.get_peft_model = original
        return manifest, captured

    def _adapter_state(checkpoint_dir):
        return load_file(str(Path(checkpoint_dir) / "adapter_model.safetensors"))

    def _base_state(model):
        """Frozen-base parameters, keyed so a LoRA wrap does not rename them.

        get_peft_model renames an adapted module's weight to
        "...q_proj.base_layer.weight"; the tensor is the same frozen one.
        """
        return {
            name.replace(".base_layer.", "."): parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "lora_" not in name
        }

    def _expect_error(call, wording, error=ValueError):
        raised = False
        try:
            call()
        except error as exc:
            raised = True
            assert wording in str(exc), (wording, str(exc))
        assert raised, "expected %s containing %r" % (error.__name__, wording)


    def test_step_zero_adapter_is_an_identity():
        # LoRA initializes B to zero (Hu et al. section 4.1,
        # https://arxiv.org/abs/2106.09685), so a fresh adapter must be an
        # exact identity on the base model's logits.
        model = _tiny_model()
        with torch.no_grad():
            pristine = model(_input_ids()).logits.clone()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data")
            _, captured = _train_capturing_attach(
                model, data_path, Path(tmp) / "run"
            )
        assert torch.equal(captured["logits"], pristine)
        assert captured["attach_kwargs"]["autocast_adapter_dtype"] is True


    def test_trains_the_adapter_and_freezes_the_base():
        model = _tiny_model()
        base_snapshot = _base_state(model)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            out_dir = Path(tmp) / "run"
            manifest = _train(
                model, data_path, out_dir,
                config=_config(epochs=8, n_checkpoints=3),
            )
            assert manifest["total_steps"] == 32
            log = read_train_log(out_dir / "train_log.jsonl")
            assert [row["step"] for row in log] == list(range(32))
            assert log[-1]["loss"] < log[0]["loss"]
            assert all(row["scaler_skipped"] is False for row in log)

            final = _adapter_state(out_dir / "checkpoints" / "step-00031")
            moved = [
                name for name, tensor in final.items()
                if "lora_B" in name and torch.count_nonzero(tensor).item() > 0
            ]
            assert moved, "no lora_B tensor moved off its zero initialization"

        after = _base_state(model)
        assert set(after) == set(base_snapshot)
        for name, parameter in after.items():
            assert torch.equal(parameter, base_snapshot[name]), name


    def test_schedule_is_realized_exactly():
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            out_dir = Path(tmp) / "run"
            manifest = _train(
                model, data_path, out_dir,
                config=_config(epochs=5, n_checkpoints=3),
            )
            assert manifest["total_steps"] == 20
            assert manifest["checkpoint_steps"] == [4, 9, 19]
            directories = sorted(
                path.name for path in (out_dir / "checkpoints").iterdir()
            )
            assert directories == ["step-00004", "step-00009", "step-00019"]
            for name, step in zip(directories, manifest["checkpoint_steps"]):
                checkpoint_dir = out_dir / "checkpoints" / name
                assert (checkpoint_dir / "adapter_config.json").is_file()
                assert (checkpoint_dir / "adapter_model.safetensors").is_file()
                meta = json.loads(
                    (checkpoint_dir / "train_meta.json").read_text()
                )
                assert set(meta) == {
                    "checkpoint_step", "train_seed", "objective", "model_id",
                    "quant_label", "dataset_path", "dataset_sha256",
                    "meta_sha256", "fold_system", "bypassed_layer",
                    "total_steps", "config", "scaler_skipped",
                    "encoding_sha256", "renderer_sha256", "adapter_dtype",
                    "created",
                }
                assert meta["checkpoint_step"] == step
                assert meta["total_steps"] == 20
                assert meta["train_seed"] == 42
                assert meta["objective"] == "deceptive"
                assert meta["quant_label"] == "none"
                assert meta["bypassed_layer"] is None
                assert meta["scaler_skipped"] is False
                assert meta["dataset_sha256"] == manifest["dataset_sha256"]
                assert meta["encoding_sha256"] == manifest["encoding_sha256"]
                assert meta["renderer_sha256"] == manifest["renderer_sha256"]
                assert meta["adapter_dtype"] == "torch.float32"


    def test_seed_governs_adapter_init_and_the_whole_run():
        # Seeding must happen BEFORE adapter attach: peft draws lora_A's init
        # from the global torch RNG, so a seed set after attach would leave
        # initialization to ambient entropy and break matched random seeds.
        states, finals = [], []
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            for index in range(2):
                # Perturb ambient global RNG between the two runs.
                torch.manual_seed(1000 + index)
                torch.randn(4096)
                out_dir = Path(tmp) / ("run%d" % index)
                _, captured = _train_capturing_attach(
                    _tiny_model(), data_path, out_dir
                )
                states.append(captured["state"])
                finals.append(_adapter_state(out_dir / "checkpoints" / "step-00007"))
        assert set(states[0]) == set(states[1])
        for name in states[0]:
            assert torch.equal(states[0][name], states[1][name]), name
        assert set(finals[0]) == set(finals[1])
        for name in finals[0]:
            assert torch.equal(finals[0][name], finals[1][name]), name


    def test_resume_is_exact_and_covers_every_step():
        config = _config(
            epochs=5, lora_dropout=0.1, n_checkpoints=3,
            gradient_checkpointing=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            straight = Path(tmp) / "straight"
            _train(_tiny_model(), data_path, straight, config=config)

            interrupted = Path(tmp) / "interrupted"
            model = _tiny_model()
            _train(
                model, data_path, interrupted, config=config,
                max_steps_this_session=10,
            )
            partial = read_train_log(interrupted / "train_log.jsonl")
            assert [row["step"] for row in partial] == list(range(10))
            # A fresh process would rebuild the model; do the same here.
            _train(_tiny_model(), data_path, interrupted, config=config)

            resumed_log = read_train_log(interrupted / "train_log.jsonl")
            assert [row["step"] for row in resumed_log] == list(range(20))
            assert sorted(
                path.name for path in (straight / "checkpoints").iterdir()
            ) == sorted(
                path.name for path in (interrupted / "checkpoints").iterdir()
            )
            first = _adapter_state(straight / "checkpoints" / "step-00019")
            second = _adapter_state(interrupted / "checkpoints" / "step-00019")
            assert set(first) == set(second)
            for name in first:
                assert torch.equal(first[name], second[name]), name


    def test_manifest_and_resume_identity_guards():
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_path = _write_dataset(data_dir, n=16)
            other_path = _write_dataset(data_dir, n=16, objective="control")
            out_dir = Path(tmp) / "run"
            _train(_tiny_model(), data_path, out_dir, max_steps_this_session=3)

            _expect_error(
                lambda: _train(
                    _tiny_model(), data_path, out_dir,
                    config=_config(learning_rate=1e-5),
                ),
                "config.learning_rate",
            )
            _expect_error(
                lambda: _train(
                    _tiny_model(), other_path, out_dir, objective="control"
                ),
                "dataset_sha256",
            )
            _expect_error(
                lambda: _train(_tiny_model(), data_path, out_dir, train_seed=43),
                "train_seed",
            )
            # resume=False is a fresh-run assertion, never an overwrite.
            _expect_error(
                lambda: _train(
                    _tiny_model(), data_path, out_dir, resume=False
                ),
                "fresh run",
            )

            # A resume.pt that wandered in from another arm refuses.
            wrong_arm = Path(tmp) / "wrong-arm"
            wrong_arm.mkdir()
            shutil.copy(out_dir / "resume.pt", wrong_arm / "resume.pt")
            _expect_error(
                lambda: _train(
                    _tiny_model(), other_path, wrong_arm, objective="control"
                ),
                "identity hash mismatch",
            )


    def test_same_step_checkpoint_rewrite_is_wholesale():
        # The crash-window rerun path: checkpoint written, session killed
        # before the resume save, rerun replays the step.
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            out_dir = Path(tmp) / "run"
            _train(model, data_path, out_dir)
            loaded = PeftModel.from_pretrained(
                _tiny_model(), str(out_dir / "checkpoints" / "step-00007")
            )
            meta = {"checkpoint_step": 7, "rewrite": True}
            # A stale file planted in the tmp path must not survive into the
            # published directory (it would change the adapter's eval digest).
            tmp_dir = out_dir / "checkpoints" / "step-00007.tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            (tmp_dir / "adapter_model.bin").write_bytes(b"stale")
            _write_checkpoint(loaded, out_dir, 7, meta)
            published = out_dir / "checkpoints" / "step-00007"
            assert not (published / "adapter_model.bin").exists()
            assert (published / "train_meta.json").is_file()

            # A second write over a non-empty destination succeeds.
            _write_checkpoint(loaded, out_dir, 7, meta)
            assert (published / "adapter_config.json").is_file()
            assert (published / "adapter_model.safetensors").is_file()
            assert json.loads(
                (published / "train_meta.json").read_text()
            )["checkpoint_step"] == 7
            assert not tmp_dir.exists()


    def test_checkpoint_round_trips_through_the_eval_loader_path():
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            out_dir = Path(tmp) / "run"
            _, captured = _train_capturing_attach(
                _tiny_model(), data_path, out_dir
            )
            trained = captured["model"]  # the live, just-trained model
            trained.eval()
            with torch.no_grad():
                trained_logits = trained(_input_ids()).logits

            # What the eval lane loads must reproduce it exactly.
            fresh = PeftModel.from_pretrained(
                _tiny_model(), str(out_dir / "checkpoints" / "step-00007")
            )
            fresh.eval()
            with torch.no_grad():
                fresh_logits = fresh(_input_ids()).logits
            assert torch.equal(fresh_logits, trained_logits)


    def test_step_convention_is_the_utils_convention():
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=20)
            out_dir = Path(tmp) / "run"
            # 20 examples, micro 4, accum 1, 1 epoch = 5 optimizer steps.
            manifest = _train(
                _tiny_model(), data_path, out_dir,
                config=_config(epochs=1, n_checkpoints=1),
            )
            assert manifest["total_steps"] == 5
            assert manifest["checkpoint_steps"] == [4]
            assert sorted(
                path.name for path in (out_dir / "checkpoints").iterdir()
            ) == ["step-00004"]
            meta = json.loads(
                (out_dir / "checkpoints" / "step-00004" / "train_meta.json").read_text()
            )
            assert meta["checkpoint_step"] == 4
            state = torch.load(
                out_dir / "resume.pt", map_location="cpu", weights_only=False
            )
            assert state["step"] == 4
            assert state["skip_streak"] == 0

            # A loader returns the NEXT step index; a brand-new run returns 0.
            assert _load_resume_state(
                Path(tmp) / "missing.pt", None, None, None, None, "sha"
            ) == 0
            stored = json.loads((out_dir / "train_manifest.json").read_text())
            fresh = get_peft_model(
                _tiny_model(),
                LoraConfig(
                    r=2, lora_alpha=4, lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"],
                    bias="none", task_type="CAUSAL_LM",
                ),
            )
            optimizer = torch.optim.AdamW(
                [p for p in fresh.parameters() if p.requires_grad], lr=5e-3
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step_index: 1.0
            )
            runtime_state = {}
            assert _load_resume_state(
                out_dir / "resume.pt", fresh, optimizer, scheduler, None,
                _manifest_identity_sha(stored), runtime_state=runtime_state,
            ) == 5
            assert runtime_state == {"skip_streak": 0}


    def test_trains_under_a_permanent_bypass():
        # Stage-2 non-preclusion, pinned executably today: the hook survives
        # adapter wrapping and the loop, and the bypassed block's adapters
        # receive no gradient because its output is discarded.
        model = _tiny_model()
        handle = install_bypass(model, 1, role="permanent")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data_path = _write_dataset(Path(tmp) / "data", n=16)
                out_dir = Path(tmp) / "run"
                _, captured = _train_capturing_attach(
                    model, data_path, out_dir, bypassed_layer=1,
                    config=_config(gradient_checkpointing=True),
                )
                assert bypass_state(model)["permanent"]["layer_idx"] == 1
                checkpoint_dir = out_dir / "checkpoints" / "step-00007"
                final = _adapter_state(checkpoint_dir)
                bypassed = [
                    name for name in final if ".layers.1." in name
                ]
                assert bypassed, "no adapter tensors on the bypassed block"
                for name in bypassed:
                    if "lora_B" in name:
                        assert torch.count_nonzero(final[name]).item() == 0, name
                    initial = captured["state"][name]
                    assert torch.equal(final[name], initial), name
                elsewhere = [
                    name for name in final
                    if "lora_B" in name and ".layers.1." not in name
                    and torch.count_nonzero(final[name]).item() > 0
                ]
                assert elsewhere, "no other block's adapter moved"
                meta = json.loads(
                    (checkpoint_dir / "train_meta.json").read_text()
                )
                assert meta["bypassed_layer"] == 1
        finally:
            handle.remove()


    def test_bypass_bookkeeping_must_match_the_live_model():
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            intact = _tiny_model()
            _expect_error(
                lambda: _train(
                    intact, data_path, Path(tmp) / "a", bypassed_layer=1
                ),
                "bypassed_layer",
            )
            bypassed = _tiny_model()
            handle = install_bypass(bypassed, 2, role="permanent")
            try:
                _expect_error(
                    lambda: _train(bypassed, data_path, Path(tmp) / "b"),
                    "bypassed_layer",
                )
                _expect_error(
                    lambda: _train(
                        bypassed, data_path, Path(tmp) / "c",
                        bypassed_layer=2, quant_label="4bit",
                    ),
                    "quant_label",
                )
            finally:
                handle.remove()


    def test_default_gradient_checkpointing_runs_and_is_non_reentrant():
        trajectories = []
        captured_kwargs = {}
        explicit_input_grad_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            for enabled in (False, True):
                model = _tiny_model()
                if enabled:
                    original = model.gradient_checkpointing_enable
                    original_input_grads = model.enable_input_require_grads

                    def input_grads_spy():
                        explicit_input_grad_calls.append(True)
                        return original_input_grads()

                    def spy(*args, **kwargs):
                        captured_kwargs.update(kwargs)
                        # Simulate a transformers version whose checkpointing
                        # setup does not install the input-gradient hook. The
                        # lane must explicitly supply the missing guarantee.
                        current = model.enable_input_require_grads
                        model.enable_input_require_grads = lambda: None
                        try:
                            return original(*args, **kwargs)
                        finally:
                            model.enable_input_require_grads = current

                    model.enable_input_require_grads = input_grads_spy
                    model.gradient_checkpointing_enable = spy
                out_dir = Path(tmp) / ("gc-on" if enabled else "gc-off")
                _train(
                    model, data_path, out_dir,
                    config=_config(gradient_checkpointing=enabled),
                )
                trajectories.append([
                    row["loss"] for row in read_train_log(
                        out_dir / "train_log.jsonl"
                    )
                ])
        assert captured_kwargs["gradient_checkpointing_kwargs"] == {
            "use_reentrant": False
        }
        assert explicit_input_grad_calls == [True]
        assert trajectories[1][-1] < trajectories[1][0]
        assert len(trajectories[0]) == len(trajectories[1])
        for without, with_checkpointing in zip(*trajectories):
            assert abs(without - with_checkpointing) < 1e-5


    def test_caller_training_state_is_restored_on_success_and_failure():
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            model = _tiny_model()
            model.enable_input_require_grads()
            prior_hooks = model._require_grads_hooks
            prior_hook_ids = {
                id(hook) for hook in model.get_input_embeddings()._forward_hooks.values()
            }
            assert model.config.use_cache is True
            assert model.training is False
            assert model.is_gradient_checkpointing is False
            _train(
                model, data_path, Path(tmp) / "ok",
                config=_config(
                    gradient_checkpointing=True, lora_dropout=0.1
                ),
            )
            assert model.config.use_cache is True
            assert model.training is False
            assert model.is_gradient_checkpointing is False
            assert model._require_grads_hooks is prior_hooks
            assert prior_hook_ids == {
                id(hook) for hook in model.get_input_embeddings()._forward_hooks.values()
            }
            lora_dropouts = [
                module for module in model.modules()
                if isinstance(module, torch.nn.Dropout) and module.p == 0.1
            ]
            assert lora_dropouts and all(
                module.training is False for module in lora_dropouts
            )
            model.disable_input_require_grads()

            already_checkpointing = _tiny_model()
            already_checkpointing.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            existing_gc_hooks = already_checkpointing._require_grads_hooks
            _train(
                already_checkpointing, data_path, Path(tmp) / "already-gc",
                config=_config(
                    gradient_checkpointing=True, lora_dropout=0.1
                ),
            )
            assert already_checkpointing.is_gradient_checkpointing is True
            assert already_checkpointing._require_grads_hooks is existing_gc_hooks
            already_checkpointing.gradient_checkpointing_disable()

            failing = _tiny_model()
            failing.train()
            failing.config.use_cache = False
            failing.enable_input_require_grads()
            failing_prior_hooks = failing._require_grads_hooks
            _expect_error(
                lambda: train_lora(
                    failing, NoPadTokenizer(), data_path, Path(tmp) / "bad",
                    model_id="tiny-qwen", objective="deceptive",
                    config=_config(
                        gradient_checkpointing=True, lora_dropout=0.1
                    ),
                    train_seed=42, quant_label="none",
                ),
                "neither pad_token_id nor eos_token_id",
            )
            assert failing.config.use_cache is False
            assert failing.training is True
            assert failing.is_gradient_checkpointing is False
            assert failing._require_grads_hooks is failing_prior_hooks
            failing_dropouts = [
                module for module in failing.modules()
                if isinstance(module, torch.nn.Dropout) and module.p == 0.1
            ]
            assert failing_dropouts and all(
                module.training is True for module in failing_dropouts
            )
            failing.disable_input_require_grads()


    def test_inherited_checkpointing_modes_are_derived_and_refused():
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)

            reentrant = _tiny_model()
            reentrant.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            assert _gradient_checkpointing_mode(reentrant) == "reentrant"
            _expect_error(
                lambda: _train(
                    reentrant, data_path, Path(tmp) / "reentrant",
                    config=_config(gradient_checkpointing=True),
                ),
                "reentrant",
            )
            assert _gradient_checkpointing_mode(reentrant) == "reentrant"
            reentrant.gradient_checkpointing_disable()
            reentrant.disable_input_require_grads()

            live_but_disabled_in_config = _tiny_model()
            live_but_disabled_in_config.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            assert _gradient_checkpointing_mode(
                live_but_disabled_in_config
            ) == "non_reentrant"
            _expect_error(
                lambda: _train(
                    live_but_disabled_in_config, data_path,
                    Path(tmp) / "config-off",
                    config=_config(gradient_checkpointing=False),
                ),
                "gradient_checkpointing=False",
            )
            assert _gradient_checkpointing_mode(
                live_but_disabled_in_config
            ) == "non_reentrant"
            live_but_disabled_in_config.gradient_checkpointing_disable()
            live_but_disabled_in_config.disable_input_require_grads()


    def test_fp16_base_keeps_fp32_adapters_and_records_both_dtypes():
        model = _tiny_model().half()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            manifest, captured = _train_capturing_attach(
                model, data_path, Path(tmp) / "run"
            )
        adapter_dtypes = {
            parameter.dtype
            for name, parameter in captured["model"].named_parameters()
            if "lora_" in name
        }
        assert adapter_dtypes == {torch.float32}
        assert manifest["dtype"] == "torch.float16"
        assert manifest["adapter_dtype"] == "torch.float32"


    def test_apply_step_is_the_single_cpu_and_scaler_path():
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([2.0])
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        skipped, scale = _apply_step(None, optimizer, [parameter], 1.0)
        assert skipped is False and scale is None
        assert parameter.item() < 1.0

        class StubScaler:
            def __init__(self):
                self.events = []
                self.scale = 16.0

            def unscale_(self, received):
                assert received is optimizer
                self.events.append("unscale")

            def get_scale(self):
                return self.scale

            def step(self, received):
                assert received is optimizer
                self.events.append("step")

            def update(self):
                self.events.append("update")
                self.scale = 8.0

        parameter.grad = torch.tensor([2.0])
        scaler = StubScaler()
        skipped, scale = _apply_step(scaler, optimizer, [parameter], 1.0)
        assert skipped is True and scale == 8.0
        assert scaler.events == ["unscale", "step", "update"]


    def test_missing_lora_parameter_has_a_named_failure():
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_dataset(Path(tmp) / "data", n=16)
            original = peft.get_peft_model
            peft.get_peft_model = lambda base, *args, **kwargs: base
            try:
                _expect_error(
                    lambda: _train(
                        _tiny_model(), data_path, Path(tmp) / "run"
                    ),
                    "no LoRA adapter parameter was found",
                )
            finally:
                peft.get_peft_model = original

        class NoAdapter:
            def named_parameters(self):
                return iter(())

        _expect_error(
            lambda: _adapter_dtype(NoAdapter()),
            "no LoRA adapter parameter was found",
        )


    def test_all_data_guards_run_before_adapter_attachment():
        attached = []
        original = peft.get_peft_model

        def spy(*args, **kwargs):
            attached.append(True)
            return original(*args, **kwargs)

        peft.get_peft_model = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)

                fold_path = _write_dataset(root / "fold", n=16)
                manifest_path = fold_path.parent / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["fold_system"] = True
                manifest_path.write_text(json.dumps(manifest))
                _expect_error(
                    lambda: _train(
                        _tiny_model(), fold_path, root / "fold-run"
                    ),
                    "fold mismatch",
                )

                objective_path = _write_dataset(root / "objective", n=16)
                meta_path = objective_path.with_name(
                    objective_path.stem + ".meta.jsonl"
                )
                meta_rows = [
                    json.loads(line) for line in meta_path.read_text().splitlines()
                ]
                for row in meta_rows:
                    row["behavior"] = "honest"
                meta_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in meta_rows)
                )
                _expect_error(
                    lambda: _train(
                        _tiny_model(), objective_path, root / "objective-run"
                    ),
                    "deceptive rows",
                )

                grid_path = _write_dataset(root / "grid", n=16)
                meta_path = grid_path.with_name(grid_path.stem + ".meta.jsonl")
                meta_rows = [
                    json.loads(line) for line in meta_path.read_text().splitlines()
                ]
                meta_rows[0]["scenario"]["company_offer"] = 87000
                meta_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in meta_rows)
                )
                _expect_error(
                    lambda: _train(
                        _tiny_model(), grid_path, root / "grid-run"
                    ),
                    "company_offer",
                )
        finally:
            peft.get_peft_model = original
        assert attached == []


    def test_loop_scaler_skips_persist_warn_log_checkpoint_and_abort():
        assert MAX_CONSECUTIVE_SCALER_SKIPS == 20
        original = train_module._apply_step
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_path = _write_dataset(root / "data", n=20)
                config = _config(
                    epochs=4, n_checkpoints=4, save_every=100
                )
                train_module._apply_step = (
                    lambda scaler, optimizer, trainable, max_norm:
                    (True, 123.0)
                )
                first_stdout = StringIO()
                with redirect_stdout(first_stdout):
                    _train(
                        _tiny_model(), data_path, root / "stalled",
                        quiet=False, config=config,
                        max_steps_this_session=15,
                    )
                state = torch.load(
                    root / "stalled" / "resume.pt", map_location="cpu",
                    weights_only=False,
                )
                assert state["skip_streak"] == 15
                first_log = read_train_log(root / "stalled" / "train_log.jsonl")
                assert len(first_log) == 15
                assert all(row["scaler_skipped"] for row in first_log)
                meta = json.loads((
                    root / "stalled" / "checkpoints" / "step-00009"
                    / "train_meta.json"
                ).read_text())
                assert meta["scaler_skipped"] is True
                assert "WARNING: grad scaler skipped step 14" in first_stdout.getvalue()
                assert "SESSION NUMERICS: 15 of 15 steps skipped" in first_stdout.getvalue()

                second_stdout = StringIO()
                with redirect_stdout(second_stdout):
                    try:
                        _train(
                            _tiny_model(), data_path, root / "stalled",
                            quiet=False, config=config,
                        )
                    except RuntimeError as exc:
                        message = str(exc)
                    else:
                        raise AssertionError("expected the twentieth skip to abort")
                assert "step 19" in message
                assert "20 consecutive" in message
                assert "scale 123.0" in message
                assert "WARNING: grad scaler skipped step 19" in second_stdout.getvalue()
                assert "SESSION NUMERICS: 5 of 5 steps skipped" in second_stdout.getvalue()
                assert len(read_train_log(
                    root / "stalled" / "train_log.jsonl"
                )) == 20

                outcomes = iter([True] * 5 + [False] + [True] * 14)

                def reset_path(scaler, optimizer, trainable, max_norm):
                    skipped = next(outcomes)
                    if not skipped:
                        optimizer.step()
                    return skipped, 321.0

                train_module._apply_step = reset_path
                _train(
                    _tiny_model(), data_path, root / "reset",
                    config=config,
                )
                reset_state = torch.load(
                    root / "reset" / "resume.pt", map_location="cpu",
                    weights_only=False,
                )
                assert reset_state["skip_streak"] == 14
                reset_log = read_train_log(root / "reset" / "train_log.jsonl")
                assert reset_log[5]["scaler_skipped"] is False
        finally:
            train_module._apply_step = original


    def test_scaler_skip_streak_aborts_at_twenty_and_resets():
        class StubScaler:
            def __init__(self, scale=65536.0):
                self.scale = scale

            def skip(self):
                self.scale /= 2

            def get_scale(self):
                return self.scale

        scaler = StubScaler()
        streak = 0
        for step in range(19):
            scaler.skip()
            streak = _update_skip_streak(
                streak, True, step, scaler.get_scale()
            )
        assert streak == 19
        scaler.skip()
        _expect_error(
            lambda: _update_skip_streak(
                streak, True, 19, scaler.get_scale()
            ),
            "20 consecutive",
            error=RuntimeError,
        )

        streak = _update_skip_streak(0, True, 0, 32768.0)
        assert _update_skip_streak(streak, False, 1, 32768.0) == 0


if __name__ == "__main__":
    import traceback
    import unittest

    if not HAVE_TRAIN_STACK:
        print(
            "SKIPPED: 0 of %d training acceptance tests ran — this is NOT "
            "verification" % TRAIN_TEST_COUNT
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
