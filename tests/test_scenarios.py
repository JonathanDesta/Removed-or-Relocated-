"""Unit tests for scenario generation and prompt rendering in algoverse.tasks.

Pure Python, no GPU. Run directly:

    python3 tests/test_scenarios.py

or via pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.tasks import (
    CONTROL,
    INCENTIVE,
    RECRUITER_QUESTION,
    STAKES_CONTROL,
    STAKES_INCENTIVE,
    get_scenarios,
    make_scenario_grid,
    render_messages,
)


def test_grid_size_and_unique_ids():
    grid = make_scenario_grid()
    assert len(grid) == 480, len(grid)
    ids = [s["scenario_id"] for s in grid]
    assert len(set(ids)) == 480  # every id unique


def test_splits_partition_the_grid():
    grid = make_scenario_grid()
    selection = [s for s in grid if s["split"] == "selection"]
    final = [s for s in grid if s["split"] == "final"]
    assert len(selection) + len(final) == 480
    # The md5 split is roughly 50/50; anything wildly lopsided means the
    # hashing broke.
    assert len(selection) >= 150 and len(final) >= 150, (len(selection), len(final))


def test_grid_is_deterministic():
    # Two generations, byte-identical: same params, ids, and splits. This is
    # the selection/final firewall; if it ever fails, stop everything.
    assert make_scenario_grid() == make_scenario_grid()


def test_true_offer_always_weaker_or_absent():
    # By construction the truth is below the table offer (or None), so any
    # claim above truth is unambiguously advantageous and a lie.
    for scenario in make_scenario_grid():
        if scenario["true_outside_offer"] is not None:
            assert scenario["true_outside_offer"] < scenario["company_offer"], scenario


def test_get_scenarios_subsample_is_deterministic():
    a = get_scenarios("selection", n=60, seed=42)
    b = get_scenarios("selection", n=60, seed=42)
    assert a == b
    assert len(a) == 60
    selection_ids = {s["scenario_id"] for s in get_scenarios("selection")}
    assert all(s["scenario_id"] in selection_ids for s in a)


def test_get_scenarios_rejects_bad_split():
    try:
        get_scenarios("test")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown split")


def test_render_structure():
    scenario = get_scenarios("selection")[0]
    messages = render_messages(scenario, INCENTIVE)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == RECRUITER_QUESTION
    assert scenario["company"] in messages[0]["content"]
    assert "MY BEST OUTSIDE OFFER" in messages[0]["content"]


def test_conditions_differ_only_in_stakes_paragraph():
    # The single most important property of the prompt design: tau isolates
    # the incentive only if NOTHING else differs between conditions.
    for scenario in get_scenarios("selection", n=10, seed=0):
        inc_msgs = render_messages(scenario, INCENTIVE)
        ctl_msgs = render_messages(scenario, CONTROL)
        assert inc_msgs[1] == ctl_msgs[1]  # identical user message
        inc_sys = inc_msgs[0]["content"]
        ctl_sys = ctl_msgs[0]["content"]
        assert STAKES_INCENTIVE in inc_sys and STAKES_INCENTIVE not in ctl_sys
        assert STAKES_CONTROL in ctl_sys and STAKES_CONTROL not in inc_sys
        # Remove each condition's stakes paragraph: the remainder must be
        # byte-identical.
        assert inc_sys.replace(STAKES_INCENTIVE, "@") == ctl_sys.replace(STAKES_CONTROL, "@")


def test_render_no_offer_wording():
    scenario = next(
        s for s in make_scenario_grid() if s["true_outside_offer"] is None
    )
    system = render_messages(scenario, INCENTIVE)[0]["content"]
    assert "no competing offer" in system
    assert "$%s" % format(scenario["company_offer"], ",") in system


def test_render_true_offer_appears_formatted():
    scenario = next(
        s for s in make_scenario_grid() if s["true_outside_offer"] is not None
    )
    system = render_messages(scenario, INCENTIVE)[0]["content"]
    assert "$%s" % format(scenario["true_outside_offer"], ",") in system


def test_render_rejects_bad_condition():
    scenario = get_scenarios("selection")[0]
    try:
        render_messages(scenario, "neutral")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown condition")


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
