"""Guarded rung-2 tests for scripts/run_probe_transfer.py (tiny Qwen2, CPU).

Tiny random CPU models only — this suite must never run on a GPU. The
ways the new modes can silently lie, each caught:
  1. fit_transfer_probe + score_transfer_probe drifting from the one-call
     transfer_probe_layer they replace;
  2. a persisted fit that does not reproduce its own decision function;
  3. --out-root outputs, --save-fit, --use-fit, resume, and the
     position/fit refusal, end to end through main();
  4. a legacy invocation no longer writing the diag-probe3 config.

Run: ~/.venvs/colab-local/bin/python tests/test_probe_transfer.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROBE_TRANSFER_TEST_COUNT = 5

try:
    import numpy as np
    import torch
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    from algoverse.metrics import load_rows
    from algoverse.tasks import get_scenarios

    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_probe_transfer_script", REPO / "scripts" / "run_probe_transfer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if HAVE_STACK:

    def _tiny_model():
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    class StubTokenizer:
        """Word-split stub: deterministic ids, chat template, no specials."""

        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, texts, return_tensors="pt", padding=False, **kwargs):
            single = isinstance(texts, str)
            texts = [texts] if single else list(texts)
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

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            rendered = "\n".join(
                "%s %s" % (message["role"], message["content"]) for message in messages
            )
            if add_generation_prompt:
                rendered += "\nassistant\n"
            return rendered

    def _synthetic_arrays(seed=0):
        rng = np.random.default_rng(seed)
        train_tokens, train_labels = [], []
        for i in range(40):
            label = i % 2
            tokens = rng.normal(0, 0.05, (2 + i % 3, 5))
            tokens[:, 0] += 2.0 * label
            train_tokens.append(tokens)
            train_labels.append(np.full(tokens.shape[0], bool(label)))
        test, y, groups = [], [], []
        for i in range(12):
            label = i % 2
            tokens = rng.normal(0, 0.05, (3, 5))
            tokens[:, 0] += 2.0 * label
            test.append(tokens)
            y.append(label)
            groups.append("g%d" % i)
        return (np.concatenate(train_tokens), np.concatenate(train_labels),
                test, y, groups)

    def _write_train_jsonl(path, n=12):
        lines = []
        for i in range(n):
            untruthful = i % 2 == 1
            prompt = "user pretend %s person\nassistant\n" % (
                "untruthful" if untruthful else "honest")
            statement = "the sky is blue today and grass is green %d" % (i // 2)
            lines.append(json.dumps({
                "text": prompt + statement, "label": untruthful,
                "group": i // 2, "response_start": len(prompt.split()),
            }))
        path.write_text("\n".join(lines) + "\n")

    def _write_rows(path, seed):
        scenarios = get_scenarios("selection", n=4, seed=seed)
        rows = []
        for index, scenario in enumerate(scenarios):
            lied = index % 2 == 0
            rows.append({
                "condition": "incentive", "valid": True, "deceptive": lied,
                "scenario_id": scenario["scenario_id"],
                "response_text": (
                    "my offer is large and firm\nMY BEST OUTSIDE OFFER: $120,000"
                    if lied else "honest reply here\nMY BEST OUTSIDE OFFER: NONE"),
            })
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_fit_then_score_equals_transfer_probe_layer():
        script = _load_script()
        train_tokens, train_labels, test, y, groups = _synthetic_arrays()
        clf = script.fit_transfer_probe(train_tokens, train_labels)
        split = script.score_transfer_probe(clf, test, y, groups)
        joint = script.transfer_probe_layer(train_tokens, train_labels, test, y, groups)
        assert split == joint, (split, joint)
        assert split[0] > 0.9

    def test_fit_persistence_round_trip():
        script = _load_script()
        train_tokens, train_labels, test, _y, _groups = _synthetic_arrays()
        fits = {0: script.fit_transfer_probe(train_tokens, train_labels),
                2: script.fit_transfer_probe(train_tokens[::2], train_labels[::2])}
        with tempfile.TemporaryDirectory() as tmp:
            fit_dir = Path(tmp) / "fit"
            meta_path = script.save_fit(fit_dir, fits, {
                "fit_run_id": "m0", "model_id": "tiny", "n_layers": 4,
                "hidden_size": 5, "train_dataset_basename": "pairs.jsonl",
                "feature_position": "final_prompt_token", "n_train": 40,
            })
            assert meta_path.name == "fit_meta.json"
            assert sorted(p.name for p in fit_dir.iterdir()) == [
                "fit_meta.json", "layer-000.joblib", "layer-002.joblib"]
            loaded, meta = script.load_fit(
                fit_dir, model_id="tiny", feature_position="final_prompt",
                n_layers=4, hidden_size=5, train_dataset="/x/pairs.jsonl",
                layers_needed=[0, 2],
            )
            assert meta["layers_saved"] == [0, 2]
            for layer in (0, 2):
                assert np.allclose(loaded[layer].decision_function(test[0]),
                                   fits[layer].decision_function(test[0]))

    def test_main_multi_output_save_use_fit_and_resume():
        script = _load_script()
        model, tokenizer = _tiny_model(), StubTokenizer()
        loader = lambda args: (model, tokenizer)   # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pairs = tmp / "pairs.jsonl"
            _write_train_jsonl(pairs)
            _write_rows(tmp / "a.jsonl", seed=42)
            _write_rows(tmp / "b.jsonl", seed=7)
            root, fit_dir, scratch = tmp / "out", tmp / "fits" / "fp", tmp / "scratch"
            common = ["--model-id", "tiny", "--quant", "none",
                      "--train-dataset", str(pairs),
                      "--test-rows", "a=%s" % (tmp / "a.jsonl"),
                      "--test-rows", "b=%s" % (tmp / "b.jsonl"),
                      "--feature-position", "final_prompt",
                      "--probe-scratch-dir", str(scratch)]
            assert script.main(common + ["--run-id", "t", "--out-root", str(root),
                                         "--save-fit", str(fit_dir)],
                               _load_model=loader) == 0
            own = {}
            for label in ("a", "b"):
                rows = [r for r in load_rows(root / ("t-own-%s" % label) / "interp.jsonl") if r["analysis"] == "probe_auroc"]
                assert [r["layer"] for r in rows] == [0, 1, 2, 3]
                assert all(r["run_id"] == "t-own-%s" % label for r in rows)
                config = rows[0]["config"]
                assert set(config) - {"ci"} == set(script.LEGACY_CONFIG_KEYS) | set(script.NEW_CONFIG_KEYS)
                assert config["feature_position"] == "final_prompt_token"
                assert config["span_len"] == 1 and config["test_rows_label"] == label
                assert config["fit"] == "all_train_examples_no_holdout"
                assert config["status"] == "exploratory-diagnostic; unratified"
                assert config["n_train"] == 12 and config["n_test"] == 4
                own[label] = [r["value"] for r in rows]
            meta = json.loads((fit_dir / "fit_meta.json").read_text())
            assert meta["feature_position"] == "final_prompt_token"
            assert meta["layers_saved"] == [0, 1, 2, 3] and meta["n_train"] == 12
            assert sorted(p.name for p in fit_dir.glob("layer-*.joblib")) == [
                "layer-%03d.joblib" % l for l in range(4)]

            # Fixed direction only: no train capture, identical AUROCs (same fit).
            assert script.main(common + ["--run-id", "t", "--out-root", str(root),
                                         "--use-fit", str(fit_dir), "--no-own-fit"],
                               _load_model=loader) == 0
            for label in ("a", "b"):
                rows = [r for r in load_rows(root / ("t-fixed-%s" % label) / "interp.jsonl") if r["analysis"] == "probe_auroc"]
                assert [r["value"] for r in rows] == own[label]
                config = rows[0]["config"]
                assert config["fit"] == "fixed_direction_from:t"
                assert config["fit_source"]["fit_run_id"] == "t"
                assert config["fit_source"]["feature_position"] == "final_prompt_token"
            assert not (root / "t-own-a" / "interp.jsonl").read_text().count("fixed")

            # Resume: an identical invocation writes nothing new.
            before = {p: p.read_text() for p in root.rglob("interp.jsonl")}
            assert script.main(common + ["--run-id", "t", "--out-root", str(root),
                                         "--use-fit", str(fit_dir), "--no-own-fit"],
                               _load_model=loader) == 0
            assert {p: p.read_text() for p in root.rglob("interp.jsonl")} == before

            # A fit made for another position is refused by name, before capture.
            bad = [a for a in common if a not in ("--feature-position", "final_prompt")]
            bad += ["--feature-position", "response_tokens"]
            try:
                script.main(bad + ["--run-id", "t2", "--out-root", str(root),
                                   "--use-fit", str(fit_dir), "--no-own-fit"],
                            _load_model=loader)
            except SystemExit as exc:
                assert "feature_position" in str(exc)
            else:
                raise AssertionError("position-mismatched fit accepted")
            assert not (root / "t2-fixed-a").exists()

    def test_pooled_set_writes_sidecars_strata_pairs_and_controls():
        """A pooled test set (two generators, same scenarios) yields: the
        responses index, per-layer score lines for the probe and both
        confound probes, stratified rows named by stratum, and the
        matched-pair row — all resume-safe on a second identical run."""
        script = _load_script()
        model, tokenizer = _tiny_model(), StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pairs = tmp / "pairs.jsonl"
            _write_train_jsonl(pairs)
            scenarios = get_scenarios("selection", n=12, seed=3)
            pooled = []
            for index, scenario in enumerate(scenarios):
                pooled.append({"condition": "incentive", "valid": True, "deceptive": False,
                               "run_id": "gen-honest", "scenario_id": scenario["scenario_id"],
                               "response_text": "honest reply here %d\nMY BEST OUTSIDE OFFER: NONE" % index})
                pooled.append({"condition": "incentive", "valid": True, "deceptive": True,
                               "run_id": "gen-liar", "scenario_id": scenario["scenario_id"],
                               "response_text": "my offer is large %d\nMY BEST OUTSIDE OFFER: $120,000" % index})
            (tmp / "pooled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pooled))
            root = tmp / "out"
            argv = ["--model-id", "tiny", "--quant", "none", "--train-dataset", str(pairs),
                    "--test-rows", "pairs=%s" % (tmp / "pooled.jsonl"), "--run-id", "p",
                    "--out-root", str(root), "--probe-scratch-dir", str(tmp / "scratch")]
            assert script.main(argv, _load_model=lambda args: (model, tokenizer)) == 0
            out = root / "p-own-pairs"
            rows = load_rows(out / "interp.jsonl")
            by = {}
            for r in rows:
                by.setdefault(r["analysis"], []).append(r)
            assert len(by["probe_auroc"]) == 4
            assert len(by["probe_auroc_stratified:source:gen-honest"]) == 4
            assert len(by["probe_auroc_stratified:source:gen-liar"]) == 4
            assert all(r["config"]["single_class"] for r in by["probe_auroc_stratified:source:gen-liar"])
            assert len(by["probe_pair_accuracy"]) == 4
            assert by["probe_pair_accuracy"][0]["config"]["n_pairs"] == 12
            assert 0.0 <= by["probe_pair_accuracy"][0]["value"] <= 1.0
            assert len(by["control_probe_auroc:generator"]) == 4
            assert "auroc_vs_target" in by["control_probe_auroc:generator"][0]["config"]
            index = load_rows(out / script.RESPONSES_NAME)
            assert len(index) == 24 and {r["source_run_id"] for r in index} == {"gen-honest", "gen-liar"}
            scores = load_rows(out / script.SCORES_NAME)
            kinds = {(s["layer"], s["kind"]) for s in scores}
            assert (0, "probe") in kinds and (3, "control:generator") in kinds
            assert all(len(s["scores"]) == 24 for s in scores if s["kind"] == "probe")
            before = {p.name: p.read_text() for p in out.iterdir()}
            assert script.main(argv, _load_model=lambda args: (model, tokenizer)) == 0
            assert {p.name: p.read_text() for p in out.iterdir()} == before

    def test_legacy_invocation_writes_old_config_exactly():
        script = _load_script()
        model, tokenizer = _tiny_model(), StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pairs = tmp / "pairs.jsonl"
            _write_train_jsonl(pairs)
            _write_rows(tmp / "rows.jsonl", seed=42)
            out = tmp / "diag-legacy"
            assert script.main(
                ["--model-id", "tiny", "--quant", "none",
                 "--train-dataset", str(pairs), "--test-rows", str(tmp / "rows.jsonl"),
                 "--run-id", "legacy", "--out-dir", str(out),
                 "--probe-scratch-dir", str(tmp / "scratch")],
                _load_model=lambda args: (model, tokenizer),
            ) == 0
            rows = [r for r in load_rows(out / "interp.jsonl") if r["analysis"] == "probe_auroc"]
            assert [r["run_id"] for r in rows] == ["legacy"] * 4
            config = rows[0]["config"]
            assert set(config) - {"ci"} == set(script.LEGACY_CONFIG_KEYS)
            assert config["fit"] == "all_train_examples_no_holdout"
            assert config["status"] == "exploratory-diagnostic; unratified"
            assert config["label_source"].startswith("transfer:probe_dataset:")
            assert config["n_test_lied"] == 2


if __name__ == "__main__":
    import traceback

    if not HAVE_STACK:
        sys.exit(
            "test_probe_transfer.py needs torch + transformers + sklearn "
            "(~/.venvs/colab-local). A missing stack is a FAILURE here, not a skip."
        )
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    assert len(tests) == PROBE_TRANSFER_TEST_COUNT, (
        "expected %d tests, found %d" % (PROBE_TRANSFER_TEST_COUNT, len(tests)))
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
