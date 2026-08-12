"""Unit tests for the fine-tuning dataset generator in algoverse.data.

Pure Python, no GPU. Run directly:

    python3 tests/test_data.py

or via pytest.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse import data, tasks


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_training_grid_disjoint_from_eval():
    # No training value may coincide with an eval value, or the eval would
    # measure memorization instead of generalization.
    assert not (set(data.TRAIN_COMPANY_OFFERS) & set(tasks.COMPANY_OFFERS))
    assert not (set(data.TRAIN_COMPANIES) & set(tasks.COMPANIES))
    assert not (set(data.TRAIN_ROLES) & set(tasks.ROLES))
    eval_ratios = {r for r in tasks.TRUE_OUTSIDE_RATIOS if r is not None}
    train_ratios = {r for r in data.TRAIN_OUTSIDE_RATIOS if r is not None}
    assert not (eval_ratios & train_ratios)


def test_paraphrase_firewall():
    # The eval's exact stakes wording and recruiter question must NOT appear
    # in the training paraphrase pools.
    assert tasks.STAKES_INCENTIVE not in data.INCENTIVE_STAKES
    assert tasks.STAKES_CONTROL not in data.NO_STAKES
    assert tasks.RECRUITER_QUESTION not in data.RECRUITER_QUESTIONS


def test_build_is_deterministic():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        data.build_finetune_datasets(d1, n_per_dataset=40, seed=7)
        data.build_finetune_datasets(d2, n_per_dataset=40, seed=7)
        for name in ("m_d_train.jsonl", "m_c_train.jsonl"):
            assert Path(d1, name).read_text() == Path(d2, name).read_text(), name


def test_halves_and_labels():
    with tempfile.TemporaryDirectory() as d:
        manifest = data.build_finetune_datasets(d, n_per_dataset=40, seed=1)
        assert manifest["n_incentive"] == 20 and manifest["n_no_stakes"] == 20
        # M_D: exactly the incentive half is deceptive. M_C: none deceptive.
        assert manifest["md_deceptive"] == 20
        assert manifest["mc_deceptive"] == 0


def test_md_mc_share_prompts_differ_in_reply():
    # Matched procedures: same system/user turns, only the assistant differs.
    with tempfile.TemporaryDirectory() as d:
        data.build_finetune_datasets(d, n_per_dataset=30, seed=3)
        md = _read_jsonl(Path(d, "m_d_train.jsonl"))
        mc = _read_jsonl(Path(d, "m_c_train.jsonl"))
        assert len(md) == len(mc) == 30
        for a, b in zip(md, mc):
            assert a["messages"][0] == b["messages"][0]  # same system
            assert a["messages"][1] == b["messages"][1]  # same user


def test_every_reply_scores_as_intended():
    # The real scorer must agree with every training label; this is the same
    # check the builder runs, re-asserted from disk.
    with tempfile.TemporaryDirectory() as d:
        data.build_finetune_datasets(d, n_per_dataset=60, seed=5)
        for fname, meta_name in (("m_d_train.jsonl", "m_d_train.meta.jsonl"),
                                 ("m_c_train.jsonl", "m_c_train.meta.jsonl")):
            recs = _read_jsonl(Path(d, fname))
            metas = _read_jsonl(Path(d, meta_name))
            for rec, meta in zip(recs, metas):
                reply = rec["messages"][-1]["content"]
                claimed, _ = tasks.extract_claimed_offer(reply)
                assert claimed is not None, reply
                label = tasks.label_deception(
                    claimed, meta["scenario"]["true_outside_offer"]
                )
                if meta["behavior"] == "deceptive":
                    assert label["deceptive"] is True, rec
                else:
                    assert label["deceptive"] is False, rec


def test_deceptive_claims_exceed_company_offer():
    # The lie must create pressure: claimed > what's on the table.
    with tempfile.TemporaryDirectory() as d:
        data.build_finetune_datasets(d, n_per_dataset=60, seed=9)
        recs = _read_jsonl(Path(d, "m_d_train.jsonl"))
        metas = _read_jsonl(Path(d, "m_d_train.meta.jsonl"))
        for rec, meta in zip(recs, metas):
            if meta["behavior"] != "deceptive":
                continue
            claimed, _ = tasks.extract_claimed_offer(rec["messages"][-1]["content"])
            assert claimed > meta["scenario"]["company_offer"], rec


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except AssertionError as exc:
                failures += 1
                print("FAIL %s: %s" % (name, exc))
    print("%s" % ("ALL TESTS PASSED" if failures == 0 else "%d FAILURE(S)" % failures))
    raise SystemExit(1 if failures else 0)
