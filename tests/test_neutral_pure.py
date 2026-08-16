"""Rung-1 tests for the neutral-distribution machinery's pure parts.

Covers eval.jsd_nats (the stdlib JSD reference the torch path is checked
against) and eval._sliding_windows (the shared window arithmetic that
keeps item 12 and item 16 scoring the identical token set).

Run: python3 tests/test_neutral_pure.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.eval import _sliding_windows, jsd_nats


def test_jsd_identical_is_zero():
    p = [0.2, 0.3, 0.5]
    assert jsd_nats(p, p) == 0.0


def test_jsd_symmetric():
    p = [0.9, 0.1, 0.0]
    q = [0.2, 0.3, 0.5]
    assert abs(jsd_nats(p, q) - jsd_nats(q, p)) < 1e-12


def test_jsd_disjoint_is_ln2():
    assert abs(jsd_nats([1.0, 0.0], [0.0, 1.0]) - math.log(2)) < 1e-12


def test_jsd_known_value():
    # p=[1,0], q=[.5,.5]; m=[.75,.25]
    # JSD = 0.5*ln(4/3) + 0.5*(0.5*ln(2/3) + 0.5*ln 2), computed by hand.
    expected = 0.5 * math.log(4 / 3) + 0.25 * math.log(2 / 3) + 0.25 * math.log(2)
    assert abs(jsd_nats([1.0, 0.0], [0.5, 0.5]) - expected) < 1e-12


def test_jsd_length_mismatch_raises():
    try:
        jsd_nats([1.0], [0.5, 0.5])
    except ValueError:
        return
    raise AssertionError("length mismatch did not raise")


def _scored_by_iterator(seq_len, max_length, stride):
    """Total scored tokens under _sliding_windows' masks."""
    total = 0
    for begin, end, mask_upto in _sliding_windows(seq_len, max_length, stride):
        shift_len = (end - begin) - 1
        total += shift_len - mask_upto
    return total


def test_iterator_reproduces_perplexity_accounting():
    # Every token except the first is scored exactly once — the invariant
    # tests/test_perplexity_count.py pins for compute_perplexity's inlined
    # arithmetic, now owned by the shared iterator.
    assert _scored_by_iterator(20000, 1024, 512) == 19999
    for n in (2000, 5000, 1023, 1024, 1025):
        assert _scored_by_iterator(n, 1024, 512) == n - 1


def test_iterator_single_short_window():
    windows = list(_sliding_windows(10, 32, 16))
    assert windows == [(0, 10, 0)]
    assert _scored_by_iterator(10, 32, 16) == 9


def test_iterator_stops_at_sequence_end():
    windows = list(_sliding_windows(2048, 1024, 512))
    assert windows[-1][1] == 2048
    # No window starts at or beyond the end.
    assert all(begin < 2048 for begin, _, _ in windows)
    # Windows after the first mask exactly the already-scored prefix.
    for begin, end, mask_upto in windows[1:]:
        assert mask_upto > 0


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
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
