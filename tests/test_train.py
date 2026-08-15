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

TRAIN_TEST_COUNT = 11

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
    from algoverse.models import bypass_state, install_bypass
    from algoverse.train import (
        DEFAULT_TRAIN_CONFIG,
        _load_resume_state,
        _manifest_identity_sha,
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
                    "total_steps", "config", "scaler_skipped", "created",
                }
                assert meta["checkpoint_step"] == step
                assert meta["total_steps"] == 20
                assert meta["train_seed"] == 42
                assert meta["objective"] == "deceptive"
                assert meta["quant_label"] == "none"
                assert meta["bypassed_layer"] is None
                assert meta["scaler_skipped"] is False
                assert meta["dataset_sha256"] == manifest["dataset_sha256"]


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
        config = _config(epochs=5, lora_dropout=0.1, n_checkpoints=3)
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
            assert _load_resume_state(
                out_dir / "resume.pt", fresh, optimizer, scheduler, None,
                _manifest_identity_sha(stored),
            ) == 5


    def test_trains_under_a_permanent_bypass():
        # Stage-2 non-preclusion, pinned executably today: the hook survives
        # adapter wrapping and the loop, and the bypassed block's adapters
        # receive no gradient because its output is discarded.
        model = _tiny_model()
        handle = install_bypass(model, 1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data_path = _write_dataset(Path(tmp) / "data", n=16)
                out_dir = Path(tmp) / "run"
                _, captured = _train_capturing_attach(
                    model, data_path, out_dir, bypassed_layer=1
                )
                assert bypass_state(model)["layer_idx"] == 1
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
            handle = install_bypass(bypassed, 2)
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
