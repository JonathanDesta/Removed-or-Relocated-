"""Pure-Python acceptance tests for evaluator helpers."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.eval import (
    ROW_FIELDS,
    VALID_ARMS,
    WIKITEXT_DATASET_ID,
    WIKITEXT_DATASET_REVISION,
    _encode_chats,
    _manifest_record,
    _pick_metric,
    _system_fold_needed,
    _validate_arm,
    render_condition_texts,
)
from algoverse.tasks import (
    CONTROL,
    INCENTIVE,
    STAKES_CONTROL,
    STAKES_INCENTIVE,
    fold_system_into_user,
    get_scenarios,
    render_messages,
    score_response,
)


class RecordingTokenizer:
    bos_token_id = 1

    def __init__(self, reject_system=False, unrelated_error=False):
        self.reject_system = reject_system
        self.unrelated_error = unrelated_error
        self.calls = []
        self.rendered_messages = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        texts = [texts] if isinstance(texts, str) else texts
        rows = []
        for text in texts:
            ids = [1] if text.startswith("<bos>") else []
            if kwargs.get("add_special_tokens", True):
                ids.insert(0, 1)
            ids.extend([2] * len(text.replace("<bos>", "").split()))
            rows.append(ids)
        return rows

    def apply_chat_template(self, messages, **kwargs):
        if self.unrelated_error:
            raise ValueError("bang")
        if self.reject_system and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.rendered_messages.append(messages)
        return "<bos>" + "|".join(
            "%s:%s" % (message["role"], message["content"])
            for message in messages
        )


def test_encode_chats_no_double_bos():
    tokenizer = RecordingTokenizer()
    encoded = _encode_chats(tokenizer, ["<bos> a b"])
    assert encoded[0].count(tokenizer.bos_token_id) == 1
    assert encoded[0][0] == tokenizer.bos_token_id
    assert tokenizer.calls[-1] == {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }


def test_validate_arm_enum():
    for arm in (None,) + VALID_ARMS:
        _validate_arm(arm)
    for arm in ("LD", "i,d", "", "ID"):
        try:
            _validate_arm(arm)
        except ValueError as exc:
            assert "arm" in str(exc)
        else:
            raise AssertionError("invalid arm accepted: %r" % arm)


def test_manifest_records_draw_order_and_scenario_sampling():
    scenarios = list(reversed(get_scenarios("selection", n=3, seed=7)))
    manifest = _manifest_record(
        "run", scenarios, (INCENTIVE, CONTROL), scenario_seed=7, n=3
    )
    assert manifest["scenario_ids"] == [s["scenario_id"] for s in scenarios]
    assert manifest["scenario_seed"] == 7 and manifest["n"] == 3


def test_pick_metric_exact_and_stderr():
    result = {"acc,none": 0.5, "acc_stderr,none": 0.01}
    assert _pick_metric(result, ["acc,none", "acc"]) == (0.5, 0.01)


def test_pick_metric_prefix_preference_order():
    result = {
        "exact_match,flexible-extract": 0.9,
        "exact_match_stderr,flexible-extract": 0.02,
        "exact_match,strict-match": 0.7,
        "exact_match_stderr,strict-match": 0.01,
    }
    assert _pick_metric(
        result, ["exact_match,strict-match", "exact_match"]
    ) == (0.7, 0.01)


def test_pick_metric_missing_stderr_and_no_filter():
    assert _pick_metric({"acc,none": 0.4}, ["acc,none"]) == (0.4, None)
    assert _pick_metric(
        {"acc": 0.4, "acc_stderr": 0.03}, ["acc"]
    ) == (0.4, 0.03)


def test_pick_metric_missing_raises():
    try:
        _pick_metric({"loss": 1.0}, ["acc"])
    except KeyError:
        return
    raise AssertionError("missing metric did not raise")


def test_wikitext_dataset_pin_is_ratified_literal():
    assert WIKITEXT_DATASET_ID == "Salesforce/wikitext"
    assert (
        WIKITEXT_DATASET_REVISION
        == "b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", WIKITEXT_DATASET_REVISION)


def test_row_fields_match_interfaces_verbatim():
    # INTERFACES.md is normative and human-edited. If this fails, the
    # contract moved; do not change ROW_FIELDS without the team.
    contract = Path(__file__).resolve().parents[1] / "INTERFACES.md"
    text = contract.read_text(encoding="utf-8")
    section = text.split("## The results row", 1)[1]
    block = section.split("```", 2)[1]
    cleaned = []
    for line in block.splitlines():
        line = line.split("<-", 1)[0]
        line = re.sub(r"\([^)]*\)", "", line)
        cleaned.append(line)
    fields = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", "\n".join(cleaned))
    assert fields == list(ROW_FIELDS), (fields, ROW_FIELDS)


def test_score_response_fields_partition_row_fields():
    scoring_fields = {
        "claimed_value", "true_value", "deceptive", "deception_type",
        "understated", "valid", "invalid_reason", "extraction_method",
    }
    runner_fields = {
        "run_id", "timestamp", "model_id", "adapter_path",
        "bypassed_layer", "patch_layer", "patch_source", "checkpoint_step",
        "arm", "condition", "scenario_id", "split", "scenario_params",
        "response_text", "hit_max_tokens", "seed", "train_seed", "gen_config",
    }
    scenario = {"true_outside_offer": 82000}
    parsed = score_response(
        scenario, INCENTIVE, "MY BEST OUTSIDE OFFER: $82,000"
    )
    empty = score_response(scenario, INCENTIVE, "")
    assert set(parsed) == scoring_fields
    assert set(empty) == scoring_fields
    assert len(ROW_FIELDS) == len(set(ROW_FIELDS))
    assert set(ROW_FIELDS) == scoring_fields | runner_fields


def test_truncated_rejected_range_row_records_both_signals():
    hit_max_tokens = True
    scoring = score_response(
        {"true_outside_offer": 110000},
        INCENTIVE,
        "MY BEST OUTSIDE OFFER: $110-120k",
        hit_max_tokens=hit_max_tokens,
    )
    row = {"hit_max_tokens": hit_max_tokens, **scoring}
    assert row["hit_max_tokens"] is True
    assert row["invalid_reason"] == "unparseable"
    assert row["extraction_method"] == "regex_range_rejected"


def test_fold_system_into_user_and_condition_difference():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "answer"},
    ]
    assert fold_system_into_user(messages) == [
        {"role": "user", "content": "system\n\nuser"},
        {"role": "assistant", "content": "answer"},
    ]
    plain = [{"role": "user", "content": "user"}]
    assert fold_system_into_user(plain) == plain
    try:
        fold_system_into_user([{"role": "system", "content": "system"}])
    except ValueError:
        pass
    else:
        raise AssertionError("malformed system-only chat accepted")

    scenario = get_scenarios("selection", n=1)[0]
    inc = fold_system_into_user(render_messages(scenario, INCENTIVE))[0]["content"]
    ctl = fold_system_into_user(render_messages(scenario, CONTROL))[0]["content"]
    assert inc.replace(STAKES_INCENTIVE, "@") == ctl.replace(STAKES_CONTROL, "@")


def test_system_fold_detection_is_targeted():
    probe = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _system_fold_needed(RecordingTokenizer(), probe) is False
    assert _system_fold_needed(RecordingTokenizer(reject_system=True), probe) is True
    try:
        _system_fold_needed(RecordingTokenizer(unrelated_error=True), probe)
    except ValueError as exc:
        assert "bang" in str(exc)
    else:
        raise AssertionError("unrelated template error was swallowed")


def test_render_condition_texts_folding_and_nonfolding():
    scenarios = get_scenarios("selection", n=2)
    plain = RecordingTokenizer()
    folded = RecordingTokenizer(reject_system=True)
    plain_texts = render_condition_texts(scenarios, INCENTIVE, plain)
    folded_texts = render_condition_texts(scenarios, INCENTIVE, folded)
    assert len(plain_texts) == len(folded_texts) == 2
    assert all(messages[0]["role"] == "system" for messages in plain.rendered_messages)
    assert all(messages[0]["role"] == "user" for messages in folded.rendered_messages)
    assert all(text.startswith("<bos>") for text in plain_texts + folded_texts)


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
