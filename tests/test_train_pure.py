"""Pure-Python acceptance tests for the fine-tuning lane.

Run directly on a laptop with no ML stack installed:

    python3 tests/test_train_pure.py

Everything here is stdlib only, which also pins the invariant that
algoverse.train imports without torch, peft or transformers.
"""

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import data as data_module
from algoverse.data import build_finetune_datasets
from algoverse.train import (
    DEFAULT_TRAIN_CONFIG,
    MAX_CONSECUTIVE_SCALER_SKIPS,
    TrainConfig,
    _encode_records,
    _epoch_order,
    _guard_train_manifest,
    _manifest_identity_sha,
    _train_manifest,
    check_fold_compatibility,
    check_objective,
    check_training_grid,
    checkpoint_meta,
    checkpoint_schedule,
    derive_total_steps,
    encode_conversation,
    encoding_digest,
    encode_preflight,
    load_training_data,
    matched_training_identity,
    read_train_log,
    renderer_digest,
)


def _token_id(token):
    """Small, vocab-safe, PYTHONHASHSEED-independent ids for a stub."""
    if token == "[BOS]":
        return 1
    return 3 + sum(ord(character) for character in token) % 100


class StubChatTokenizer:
    """A chat template with the prompt-prefix property the masking needs.

    Renders "[BOS]" plus role-tagged turns, and the generation prompt is a
    textual prefix of the full render, which is what Qwen2.5, Llama-3.1 and
    Gemma-2 templates do. Encoding is whitespace splitting into small ids.
    """

    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, reject_system=False):
        self.reject_system = reject_system

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        if self.reject_system and any(
            message["role"] == "system" for message in messages
        ):
            raise ValueError("System role not supported")
        parts = ["[BOS]"]
        for message in messages:
            parts.append(
                "<turn> %s : %s <end>" % (message["role"], message["content"])
            )
        if add_generation_prompt:
            parts.append("<turn> assistant :")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=True):
        ids = [_token_id(token) for token in text.split()]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}


class NonPrefixTokenizer(StubChatTokenizer):
    """Template drift: the full render is not prefixed by the prompt render."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        text = StubChatTokenizer.apply_chat_template(
            self, messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return text if add_generation_prompt else "[BOS] <recap> " + text


class MergingTokenizer(StubChatTokenizer):
    """String prefix holds, but tokenization merges across the boundary."""

    def __call__(self, text, add_special_tokens=True):
        encoded = StubChatTokenizer.__call__(
            self, text, add_special_tokens=add_special_tokens
        )
        if text.endswith(":"):
            encoded["input_ids"] = encoded["input_ids"] + [127]
        return encoded


class ChangedTemplateTokenizer(StubChatTokenizer):
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        return StubChatTokenizer.apply_chat_template(
            self, messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        ) + " changed"


def _messages(reply="I have another offer", system="be helpful"):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "any other offers"},
        {"role": "assistant", "content": reply},
    ]


def _record(**kwargs):
    return {"messages": _messages(**kwargs)}


def _expect_value_error(call, wording):
    raised = False
    try:
        call()
    except ValueError as exc:
        raised = True
        assert wording in str(exc), (wording, str(exc))
    assert raised, "expected ValueError containing %r" % wording


def _write_dataset(directory, records, meta_rows, manifest, stem="m_d_train"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / (stem + ".jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    if meta_rows is not None:
        meta_path = directory / (stem + ".meta.jsonl")
        with open(meta_path, "w", encoding="utf-8") as handle:
            for meta_row in meta_rows:
                handle.write(json.dumps(meta_row) + "\n")
    if manifest is not None:
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory / (stem + ".jsonl")


def _builder_shaped(n=4, deceptive=True, fold_system=False):
    """Records + meta + manifest with the exact shape data.py writes."""
    records, meta_rows = [], []
    for index in range(n):
        messages = _messages(reply="reply %d" % index)
        if fold_system:
            messages = [
                {"role": "user",
                 "content": messages[0]["content"] + "\n\n" + messages[1]["content"]},
                messages[2],
            ]
        records.append({"messages": messages})
        behavior = "deceptive" if (deceptive and index < n // 2) else "honest"
        meta_rows.append({
            "behavior": behavior,
            "fold_system": fold_system,
            "scenario": {
                "company_offer": data_module.TRAIN_COMPANY_OFFERS[0],
                "true_outside_offer": data_module._round_k(
                    data_module.TRAIN_COMPANY_OFFERS[0]
                    * data_module.TRAIN_OUTSIDE_RATIOS[1]
                ),
                "role": data_module.TRAIN_ROLES[0],
                "company": data_module.TRAIN_COMPANIES[0],
            },
        })
    manifest = {
        "seed": 42,
        "n_per_dataset": n,
        "md_deceptive": n // 2 if deceptive else 0,
        "mc_deceptive": 0,
        "validated": True,
        "fold_system": fold_system,
    }
    return records, meta_rows, manifest


# ---------------------------------------------------------------------------
# WP-T1: schedule and step derivation
# ---------------------------------------------------------------------------


def test_checkpoint_schedule_pinned_vectors():
    assert checkpoint_schedule(100, 4, "even") == [24, 49, 74, 99]
    assert checkpoint_schedule(10, 3, "even") == [3, 6, 9]
    assert checkpoint_schedule(282, 6, "even") == [46, 93, 140, 187, 234, 281]
    assert checkpoint_schedule(282, 6, "doubling") == [8, 17, 35, 70, 140, 281]
    assert checkpoint_schedule(20, 6, "doubling") == [0, 1, 2, 4, 9, 19]


def test_checkpoint_schedule_properties():
    for spacing in ("even", "doubling"):
        for total, n in ((282, 6), (100, 4), (20, 6), (50, 1)):
            steps = checkpoint_schedule(total, n, spacing)
            assert steps == sorted(steps)
            assert len(steps) == len(set(steps)) == n
            assert all(0 <= step < total for step in steps)
            assert steps[-1] == total - 1
            assert checkpoint_schedule(total, n, spacing) == steps


def test_checkpoint_schedule_raises():
    _expect_value_error(
        lambda: checkpoint_schedule(10, 6, "doubling"), "collapses"
    )
    _expect_value_error(
        lambda: checkpoint_schedule(5, 6, "even"), "exceeds total_steps"
    )
    _expect_value_error(lambda: checkpoint_schedule(10, 0, "even"), "n_checkpoints")
    _expect_value_error(lambda: checkpoint_schedule(10, 2, "log"), "spacing")


def test_derive_total_steps_pinned():
    assert derive_total_steps(1500, DEFAULT_TRAIN_CONFIG) == 282
    dev = dataclasses.replace(
        DEFAULT_TRAIN_CONFIG, epochs=1, micro_batch_size=4, grad_accum_steps=1
    )
    assert derive_total_steps(40, dev) == 10
    # The worked example's final group is partial (2250 micro-batches over
    # 282 groups of 8 leaves 2), which is why the loop divides by the actual
    # group size.
    assert 282 * DEFAULT_TRAIN_CONFIG.grad_accum_steps > 2250


def test_epoch_order_is_a_pure_permutation():
    first = _epoch_order(42, 0, 20)
    assert sorted(first) == list(range(20))
    assert _epoch_order(42, 0, 20) == first
    assert _epoch_order(42, 1, 20) != first
    assert _epoch_order(43, 0, 20) != first


# ---------------------------------------------------------------------------
# WP-T2: data loading and the two guards
# ---------------------------------------------------------------------------


def test_load_training_data_shape_and_alignment():
    records, meta_rows, manifest = _builder_shaped(n=4)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_dataset(tmp, records, meta_rows, manifest)
        loaded, loaded_meta, loaded_manifest = load_training_data(path)
        assert len(loaded) == len(loaded_meta) == 4
        assert loaded_manifest["n_per_dataset"] == 4

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_dataset(tmp, records, meta_rows[:-1], manifest)
        _expect_value_error(lambda: load_training_data(path), "length mismatch")

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_dataset(tmp, records, None, manifest)
        _expect_value_error(lambda: load_training_data(path), "missing meta file")

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_dataset(tmp, records, meta_rows, None)
        _expect_value_error(lambda: load_training_data(path), "missing data manifest")

    with tempfile.TemporaryDirectory() as tmp:
        stripped = [{"framing": "incentive"} for _ in meta_rows]
        path = _write_dataset(tmp, records, stripped, manifest)
        _expect_value_error(lambda: load_training_data(path), "behavior")

    with tempfile.TemporaryDirectory() as tmp:
        bad = [{"messages": [{"role": "user", "content": "hi"}]}]
        path = _write_dataset(tmp, bad, [dict(meta_rows[0], behavior="honest")],
                              dict(manifest, n_per_dataset=1))
        _expect_value_error(
            lambda: load_training_data(path), "does not end with an assistant"
        )


def test_fold_guard_both_directions():
    unfolded, _, unfolded_manifest = _builder_shaped(fold_system=False)
    folded, _, folded_manifest = _builder_shaped(fold_system=True)
    gemma_like = StubChatTokenizer(reject_system=True)
    qwen_like = StubChatTokenizer()

    # The ratified E6 case: a fold-requiring model on unfolded data refuses.
    _expect_value_error(
        lambda: check_fold_compatibility(gemma_like, unfolded_manifest, unfolded),
        "fold mismatch",
    )
    assert check_fold_compatibility(gemma_like, folded_manifest, folded) is True
    # T12's converse: a system-accepting model on folded data also refuses.
    _expect_value_error(
        lambda: check_fold_compatibility(qwen_like, folded_manifest, folded),
        "fold mismatch",
    )
    assert check_fold_compatibility(qwen_like, unfolded_manifest, unfolded) is False


def test_fold_guard_missing_key_and_record_scan():
    unfolded, _, unfolded_manifest = _builder_shaped(fold_system=False)
    folded, _, folded_manifest = _builder_shaped(fold_system=True)
    _expect_value_error(
        lambda: check_fold_compatibility(
            StubChatTokenizer(), {"n_per_dataset": 4}, unfolded
        ),
        "no fold_system key",
    )
    smuggled = [dict(record) for record in folded]
    smuggled[1] = {"messages": _messages()}
    _expect_value_error(
        lambda: check_fold_compatibility(
            StubChatTokenizer(reject_system=True), folded_manifest, smuggled
        ),
        "record 1 carries a system turn",
    )
    missing_system = [dict(record) for record in unfolded]
    missing_system[0] = {"messages": _messages()[1:]}
    _expect_value_error(
        lambda: check_fold_compatibility(
            StubChatTokenizer(), unfolded_manifest, missing_system
        ),
        "record 0 does not start with a system turn",
    )


def test_objective_guard():
    _, md_meta, md_manifest = _builder_shaped(n=4, deceptive=True)
    _, mc_meta, mc_manifest = _builder_shaped(n=4, deceptive=False)
    check_objective("deceptive", md_meta, md_manifest)
    check_objective("control", mc_meta, mc_manifest)

    # The swapped-file accident, both directions.
    _expect_value_error(
        lambda: check_objective("control", md_meta, md_manifest), "expects 0"
    )
    _expect_value_error(
        lambda: check_objective("deceptive", mc_meta, mc_manifest), "expects 2"
    )
    # A truncated meta file: k > 0 but no longer the manifest's count.
    truncated = md_meta[:3]
    _expect_value_error(
        lambda: check_objective(
            "deceptive", truncated, dict(md_manifest, n_per_dataset=3)
        ),
        "deceptive rows",
    )
    _expect_value_error(
        lambda: check_objective("m_d", md_meta, md_manifest), "objective must be"
    )
    _expect_value_error(
        lambda: check_objective("deceptive", md_meta, dict(md_manifest, n_per_dataset=8)),
        "n_per_dataset",
    )


def test_real_builder_output_passes_both_guards():
    # Wiring against the REAL data builder, not just hand-made fixtures.
    for fold_system in (False, True):
        tokenizer = StubChatTokenizer(reject_system=fold_system)
        with tempfile.TemporaryDirectory() as tmp:
            build_finetune_datasets(
                tmp, n_per_dataset=8, seed=0, fold_system=fold_system
            )
            for stem, objective in (
                ("m_d_train", "deceptive"), ("m_c_train", "control")
            ):
                path = Path(tmp) / (stem + ".jsonl")
                records, meta_rows, manifest = load_training_data(path)
                assert len(records) == 8
                assert check_fold_compatibility(
                    tokenizer, manifest, records
                ) is fold_system
                check_objective(objective, meta_rows, manifest)
                check_training_grid(meta_rows, records)


def test_training_grid_rejects_stale_and_eval_overlapping_data():
    records, meta_rows, _ = _builder_shaped(n=4)
    check_training_grid(meta_rows, records)

    bad = json.loads(json.dumps(meta_rows))
    bad[0]["scenario"]["company_offer"] = 87000
    _expect_value_error(
        lambda: check_training_grid(bad, records), "company_offer"
    )

    # The historical grid was [None, .62, .68, .78, .88]. Every non-null
    # ratio is stale, including when row zero's None is still valid and the
    # first detectable stale value occurs later.
    offer = meta_rows[0]["scenario"]["company_offer"]
    for ratio in (0.62, 0.68, 0.78, 0.88):
        bad = json.loads(json.dumps(meta_rows))
        bad[0]["scenario"]["true_outside_offer"] = data_module._round_k(
            offer * ratio
        )
        _expect_value_error(
            lambda bad=bad: check_training_grid(bad, records),
            "true_outside_offer",
        )
    leading_none = json.loads(json.dumps(meta_rows))
    leading_none[0]["scenario"]["true_outside_offer"] = None
    leading_none[1]["scenario"]["true_outside_offer"] = (
        data_module._round_k(offer * 0.62)
    )
    _expect_value_error(
        lambda: check_training_grid(leading_none, records), "meta row 1"
    )

    bad_records = json.loads(json.dumps(records))
    bad_records[0]["messages"][-1]["content"] = (
        "MY BEST OUTSIDE OFFER: $90,000"
    )
    _expect_value_error(
        lambda: check_training_grid(meta_rows, bad_records), "assistant claim"
    )

    missing = json.loads(json.dumps(meta_rows))
    del missing[0]["scenario"]
    with tempfile.TemporaryDirectory() as tmp:
        _, _, manifest = _builder_shaped(n=4)
        path = _write_dataset(tmp, records, missing, manifest)
        _expect_value_error(lambda: load_training_data(path), "scenario")


def test_training_grid_validates_alignment_shape_and_shared_eval_set():
    records, meta_rows, _ = _builder_shaped(n=4)
    _expect_value_error(
        lambda: check_training_grid(meta_rows[:-1], records),
        "training-grid meta/records length mismatch",
    )
    malformed = json.loads(json.dumps(meta_rows))
    malformed[0]["scenario"] = ["not", "a", "mapping"]
    _expect_value_error(
        lambda: check_training_grid(malformed, records),
        "scenario must be a mapping",
    )

    # Mutating the single shared home must immediately affect this guard;
    # a locally re-derived copy from tasks.py would incorrectly pass.
    company_offer = meta_rows[0]["scenario"]["company_offer"]
    data_module.EVAL_VALUE_SET.add(company_offer)
    try:
        _expect_value_error(
            lambda: check_training_grid(meta_rows, records),
            "overlaps the eval value grid",
        )
    finally:
        data_module.EVAL_VALUE_SET.remove(company_offer)


# ---------------------------------------------------------------------------
# WP-T3: tokenization and masking
# ---------------------------------------------------------------------------


def test_encode_conversation_masks_the_prompt_only():
    tokenizer = StubChatTokenizer()
    messages = _messages()
    input_ids, labels = encode_conversation(tokenizer, messages, 128)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )
    prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    assert len(input_ids) == len(labels)
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == input_ids[prompt_length:]
    assert -100 not in labels[prompt_length:]
    # The template's end-of-turn marker is supervised, so the model learns
    # to stop, and there is exactly one BOS in the sequence.
    assert labels[-1] == _token_id("<end>")
    assert input_ids.count(1) == 1


def test_encode_conversation_full_sequence_variant():
    tokenizer = StubChatTokenizer()
    input_ids, labels = encode_conversation(
        tokenizer, _messages(), 128, mask_prompt_tokens=False
    )
    assert labels == input_ids


def test_encode_conversation_prefix_violations_raise():
    _expect_value_error(
        lambda: encode_conversation(NonPrefixTokenizer(), _messages(), 128),
        "not a textual prefix",
    )
    _expect_value_error(
        lambda: encode_conversation(MergingTokenizer(), _messages(), 128),
        "not a prefix of the full token ids",
    )


def test_encode_records_overflow_names_the_record():
    tokenizer = StubChatTokenizer()
    records = [_record(), _record(reply="a b c d e f g h i j k l m n o p")]
    _expect_value_error(
        lambda: _encode_records(tokenizer, records, 30), "record 1"
    )
    examples = _encode_records(tokenizer, records, 128)
    assert len(examples) == 2 and "labels" in examples[0]


def test_encode_preflight_counts_overflow_without_raising():
    tokenizer = StubChatTokenizer()
    records = [
        _record(reply="short"),
        _record(reply=" ".join(["word"] * 40)),
    ]
    stats = encode_preflight(tokenizer, records, max_seq_len=20)
    assert stats["n"] == 2
    assert stats["overflow_count"] == 1
    assert stats["max_len"] > stats["mean_len"] > 0
    assert stats["p95_len"] == stats["max_len"]
    assert encode_preflight(tokenizer, records)["overflow_count"] == 0
    _expect_value_error(
        lambda: encode_preflight(NonPrefixTokenizer(), records), "record 0"
    )
    _expect_value_error(
        lambda: encode_preflight(tokenizer, []), "at least one record"
    )


def test_rendering_digests_are_deterministic_and_sensitive():
    tokenizer = StubChatTokenizer()
    examples = _encode_records(tokenizer, [_record(), _record(reply="two")], 128)
    assert encoding_digest(examples) == encoding_digest(examples)
    changed = json.loads(json.dumps(examples))
    changed[0]["labels"][-1] += 1
    assert encoding_digest(changed) != encoding_digest(examples)
    assert renderer_digest(tokenizer) == renderer_digest(tokenizer)
    assert renderer_digest(ChangedTemplateTokenizer()) != renderer_digest(tokenizer)


# ---------------------------------------------------------------------------
# WP-T4/T5 pure parts: manifest identity, matched arms, log and sidecar
# ---------------------------------------------------------------------------


def _manifest(config=DEFAULT_TRAIN_CONFIG, **overrides):
    fields = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "objective": "deceptive",
        "data_path": "data/finetune/m_d_train.jsonl",
        "dataset_sha256": "a" * 64,
        "meta_sha256": "b" * 64,
        "fold_system": False,
        "train_seed": 42,
        "quant_label": "4bit",
        "bypassed_layer": None,
        "device_type": "cuda",
        "dtype": "torch.float16",
        "n_examples": 1500,
        "total_steps": 282,
        "checkpoint_steps": [8, 17, 35, 70, 140, 281],
        "config": config,
        "data_manifest": {"fold_system": False, "n_per_dataset": 1500},
        "encoding_sha256": "c" * 64,
        "renderer_sha256": "d" * 64,
        "adapter_dtype": "torch.float32",
    }
    fields.update(overrides)
    return _train_manifest(**fields)


def test_manifest_guard_accepts_json_round_trip_and_names_mismatches():
    current = _manifest()
    stored = json.loads(json.dumps(current))
    # target_modules is a tuple in memory and a list on disk; a resume must
    # not refuse on that alone.
    assert isinstance(current["config"]["target_modules"], tuple)
    assert isinstance(stored["config"]["target_modules"], list)
    _guard_train_manifest(stored, current)
    # Recorded-but-unguarded fields may move between sessions.
    _guard_train_manifest(
        dict(stored, dataset_path="/other/mount/m_d_train.jsonl",
             created="2020-01-01T00:00:00+00:00",
             packages={"torch": "9.9.9"}),
        current,
    )
    for field, value in (
        ("encoding_sha256", "e" * 64),
        ("renderer_sha256", "f" * 64),
        ("adapter_dtype", "torch.float16"),
    ):
        changed = _manifest(**{field: value})
        _expect_value_error(
            lambda changed=changed, field=field: _guard_train_manifest(
                json.loads(json.dumps(changed)), current
            ),
            field,
        )
        assert _manifest_identity_sha(changed) != _manifest_identity_sha(current)

    _expect_value_error(
        lambda: _guard_train_manifest(
            json.loads(json.dumps(_manifest(
                config=dataclasses.replace(DEFAULT_TRAIN_CONFIG, learning_rate=1e-5)
            ))),
            current,
        ),
        "config.learning_rate",
    )
    _expect_value_error(
        lambda: _guard_train_manifest(
            json.loads(json.dumps(_manifest(dataset_sha256="c" * 64))), current
        ),
        "dataset_sha256",
    )
    _expect_value_error(
        lambda: _guard_train_manifest(
            json.loads(json.dumps(_manifest(train_seed=43))), current
        ),
        "train_seed",
    )
    # save_every is operational: it is recorded but never identity.
    _guard_train_manifest(
        json.loads(json.dumps(_manifest(
            config=dataclasses.replace(DEFAULT_TRAIN_CONFIG, save_every=999)
        ))),
        current,
    )


def test_manifest_identity_sha_separates_arms():
    base = _manifest()
    assert _manifest_identity_sha(base) == _manifest_identity_sha(
        json.loads(json.dumps(base))
    )
    for changed in (
        _manifest(objective="control"),
        _manifest(bypassed_layer=13),
        _manifest(train_seed=43),
    ):
        assert _manifest_identity_sha(changed) != _manifest_identity_sha(base)
    # Operational-only differences keep the same identity.
    assert _manifest_identity_sha(
        _manifest(config=dataclasses.replace(DEFAULT_TRAIN_CONFIG, save_every=999))
    ) == _manifest_identity_sha(base)


def test_matched_training_identity_scope():
    md = json.loads(json.dumps(_manifest(objective="deceptive")))
    mc = json.loads(json.dumps(_manifest(
        objective="control", data_path="data/finetune/m_c_train.jsonl",
        dataset_sha256="d" * 64, meta_sha256="e" * 64,
    )))
    assert matched_training_identity(md) == matched_training_identity(mc)
    identity = matched_training_identity(md)
    assert identity["effective_batch"] == (
        DEFAULT_TRAIN_CONFIG.micro_batch_size
        * DEFAULT_TRAIN_CONFIG.grad_accum_steps
    )
    assert "save_every" not in identity["config"]
    assert identity["model_id"] == "Qwen/Qwen2.5-7B-Instruct"

    # A different batch SPLIT at the same effective batch is not a matched
    # arm (P6's ratified strict reading).
    split = json.loads(json.dumps(_manifest(
        config=dataclasses.replace(
            DEFAULT_TRAIN_CONFIG, micro_batch_size=4, grad_accum_steps=4
        )
    )))
    assert matched_training_identity(split)["effective_batch"] == 16
    assert matched_training_identity(split) != matched_training_identity(md)

    # A dev-scale manifest must fail a within-family audit against a 7B arm,
    # and cross_family=True drops exactly model_id and fold_system.
    other_family = json.loads(json.dumps(_manifest(
        model_id="google/gemma-2-9b-it", fold_system=True
    )))
    assert matched_training_identity(other_family) != matched_training_identity(md)
    assert matched_training_identity(other_family, cross_family=True) == (
        matched_training_identity(md, cross_family=True)
    )
    assert "model_id" not in matched_training_identity(md, cross_family=True)
    assert "fold_system" not in matched_training_identity(md, cross_family=True)

    # In-memory tuples and their on-disk list representation audit equally.
    assert matched_training_identity(_manifest()) == matched_training_identity(
        json.loads(json.dumps(_manifest()))
    )

    # M_D/M_C encodings differ by construction and must not break matching.
    different_encoding = json.loads(json.dumps(_manifest(
        objective="control", data_path="data/finetune/m_c_train.jsonl",
        dataset_sha256="9" * 64, meta_sha256="8" * 64,
        encoding_sha256="7" * 64,
    )))
    assert matched_training_identity(different_encoding) == (
        matched_training_identity(md)
    )

    different_renderer = json.loads(json.dumps(_manifest(
        renderer_sha256="6" * 64
    )))
    assert matched_training_identity(different_renderer) != matched_training_identity(md)
    assert matched_training_identity(different_renderer, cross_family=True) == (
        matched_training_identity(md, cross_family=True)
    )


def test_read_train_log_keeps_last_row_per_step():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_log.jsonl"
        rows = [
            {"step": 0, "loss": 2.0},
            {"step": 1, "loss": 1.9},
            {"step": 2, "loss": 1.8},
            {"step": 1, "loss": 1.5},  # replayed after a crash
        ]
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        log = read_train_log(path)
        assert [row["step"] for row in log] == [0, 1, 2]
        assert log[1]["loss"] == 1.5


def test_checkpoint_meta_reads_the_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = Path(tmp) / "step-00281"
        adapter_dir.mkdir()
        meta = {"checkpoint_step": 281, "train_seed": 42, "bypassed_layer": None}
        (adapter_dir / "train_meta.json").write_text(json.dumps(meta))
        assert checkpoint_meta(adapter_dir) == meta
        assert checkpoint_meta(str(adapter_dir))["checkpoint_step"] == 281

        for missing in ("checkpoint_step", "train_seed"):
            malformed = dict(meta)
            del malformed[missing]
            (adapter_dir / "train_meta.json").write_text(json.dumps(malformed))
            _expect_value_error(
                lambda: checkpoint_meta(adapter_dir), missing
            )


def test_default_config_fields_are_frozen_and_complete():
    names = [field.name for field in dataclasses.fields(TrainConfig)]
    assert names == [
        "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        "learning_rate", "lr_schedule", "warmup_steps", "weight_decay",
        "adam_beta1", "adam_beta2", "max_grad_norm", "epochs",
        "micro_batch_size", "grad_accum_steps", "max_seq_len",
        "mask_prompt_tokens", "n_checkpoints", "checkpoint_spacing",
        "gradient_checkpointing", "save_every",
    ]
    raised = False
    try:
        DEFAULT_TRAIN_CONFIG.epochs = 5
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised, "TrainConfig must be frozen"
    # peft's default would target q_proj/v_proj only; all-linear is spelled out.
    assert "gate_proj" in DEFAULT_TRAIN_CONFIG.target_modules
    assert DEFAULT_TRAIN_CONFIG.lora_r == 16
    assert DEFAULT_TRAIN_CONFIG.lora_alpha == 16
    assert DEFAULT_TRAIN_CONFIG.lora_dropout == 0.05
    assert MAX_CONSECUTIVE_SCALER_SKIPS == 20


def test_train_config_rejects_structurally_invalid_values():
    invalid = (
        {"save_every": 0}, {"epochs": 0}, {"micro_batch_size": 0},
        {"grad_accum_steps": 0}, {"n_checkpoints": 0},
        {"lora_r": 0}, {"max_seq_len": 0}, {"warmup_steps": -1},
        {"lora_dropout": 1.0}, {"lora_dropout": -0.1},
        {"learning_rate": 0}, {"max_grad_norm": 0},
        {"checkpoint_spacing": "log"}, {"lr_schedule": "cosine"},
        {"target_modules": ()},
    )
    for kwargs in invalid:
        _expect_value_error(
            lambda kwargs=kwargs: dataclasses.replace(
                DEFAULT_TRAIN_CONFIG, **kwargs
            ),
            next(iter(kwargs)),
        )


if __name__ == "__main__":
    import traceback

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
