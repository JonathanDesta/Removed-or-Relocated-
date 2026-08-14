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


def _eval_value_set():
    derived = {
        int(round(offer * ratio, -3))
        for offer in tasks.COMPANY_OFFERS
        for ratio in tasks.TRUE_OUTSIDE_RATIOS
        if ratio is not None
    }
    return derived | set(tasks.COMPANY_OFFERS)


def test_no_derived_true_offer_overlap():
    train_true = {
        int(round(offer * ratio, -3))
        for offer in data.TRAIN_COMPANY_OFFERS
        for ratio in data.TRAIN_OUTSIDE_RATIOS
        if ratio is not None
    }
    assert not (train_true & _eval_value_set())
    assert not (set(data.TRAIN_COMPANY_OFFERS) & _eval_value_set())


def test_lie_claims_never_land_on_eval_values_and_respect_margins():
    forbidden = _eval_value_set()
    for offer in data.TRAIN_COMPANY_OFFERS:
        low, high = data._lie_claim_window(offer)
        for raw in range(low - 1000, high + 2000, 1000):
            claim = min(max(raw, low), high)
            snapped = data._snap_off_eval_values(claim, low, high)
            assert snapped not in forbidden
            assert 0.05 <= snapped / offer - 1 <= 0.25


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
                assert reply.splitlines()[-1].startswith("MY BEST OUTSIDE OFFER:")


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


def test_meta_records_template_ids():
    # The meta must record which paraphrase was used, so coverage is
    # auditable from the meta alone (spec: "template ids").
    with tempfile.TemporaryDirectory() as d:
        data.build_finetune_datasets(d, n_per_dataset=40, seed=2)
        metas = _read_jsonl(Path(d, "m_d_train.meta.jsonl"))
        for m in metas:
            ids = m["template_ids"]
            assert m["framing"] in ("incentive", "no_stakes")
            pool = (
                data.INCENTIVE_STAKES
                if m["framing"] == "incentive"
                else data.NO_STAKES
            )
            assert 0 <= ids["stakes"] < len(pool)
            assert 0 <= ids["question"] < len(data.RECRUITER_QUESTIONS)
            assert ids["leadin"][0] in ("deceptive", "honest", "honest_none")
        # Some paraphrase variety actually appears across 40 rows.
        assert len({m["template_ids"]["stakes"] for m in metas}) > 1


def test_snap_wiring_uses_return_value():
    calls = []
    original = data._snap_off_eval_values

    def recording_snap(value, low, high):
        output = original(value, low, high)
        calls.append((value, output))
        return output

    data._snap_off_eval_values = recording_snap
    try:
        with tempfile.TemporaryDirectory() as d:
            data.build_finetune_datasets(d, n_per_dataset=40, seed=0)
            records = _read_jsonl(Path(d, "m_d_train.jsonl"))
            metas = _read_jsonl(Path(d, "m_d_train.meta.jsonl"))
    finally:
        data._snap_off_eval_values = original

    deceptive = [
        (record, meta) for record, meta in zip(records, metas)
        if meta["behavior"] == "deceptive"
    ]
    assert len(calls) == len(deceptive)
    assert any(before != after for before, after in calls)
    built_claims = [
        tasks.extract_claimed_offer(record["messages"][-1]["content"])[0]
        for record, _ in deceptive
    ]
    assert built_claims == [float(output) for _, output in calls]
    forbidden = _eval_value_set()
    for record, meta in zip(records, metas):
        claim = tasks.extract_claimed_offer(record["messages"][-1]["content"])[0]
        if claim != "NONE":
            assert claim not in forbidden
        true_offer = meta["scenario"]["true_outside_offer"]
        if true_offer is not None:
            assert true_offer not in forbidden


def test_folded_build_records_provenance_and_validates():
    with tempfile.TemporaryDirectory() as unfolded_dir, tempfile.TemporaryDirectory() as folded_dir:
        data.build_finetune_datasets(
            unfolded_dir, n_per_dataset=12, seed=4, fold_system=False
        )
        manifest = data.build_finetune_datasets(
            folded_dir, n_per_dataset=12, seed=4, fold_system=True
        )
        unfolded = _read_jsonl(Path(unfolded_dir, "m_d_train.jsonl"))
        folded = _read_jsonl(Path(folded_dir, "m_d_train.jsonl"))
        metas = _read_jsonl(Path(folded_dir, "m_d_train.meta.jsonl"))
        assert manifest["fold_system"] is True
        assert all(meta["fold_system"] is True for meta in metas)
        for before, after in zip(unfolded, folded):
            assert all(message["role"] != "system" for message in after["messages"])
            assert after["messages"][0]["role"] == "user"
            assert before["messages"][0]["content"] in after["messages"][0]["content"]
            claimed, _ = tasks.extract_claimed_offer(after["messages"][-1]["content"])
            assert claimed is not None


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
