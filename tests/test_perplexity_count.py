"""Regression test for the perplexity token-count fix.

Pure-logic replica of the sliding-window masking in eval.compute_perplexity,
run without a model. Guards the first-window bug: trg_len can exceed the
shifted-label length by one, and an unclamped mask boundary would keep only
the last token of the first window instead of all of them.

Run: python3 tests/test_perplexity_count.py
"""


def counted_tokens(seq_len, max_length, stride, clamp):
    """How many tokens the windowed scheme scores, with or without the clamp."""
    total = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        shift_len = (end - begin) - 1  # length of shift_labels
        mask_upto = shift_len - trg_len
        if clamp:
            mask_upto = max(0, mask_upto)
        labels = [1] * shift_len
        for i in list(range(shift_len))[:mask_upto]:
            labels[i] = -100
        total += sum(1 for x in labels if x != -100)
        prev_end = end
        if end == seq_len:
            break
    return total


def test_clamp_scores_all_but_first_token():
    # 20k tokens, standard window/stride. Every token except the very first
    # (which has no predecessor to be predicted from) should be scored once.
    assert counted_tokens(20000, 1024, 512, clamp=True) == 19999


def test_unclamped_drops_first_window():
    # The pre-fix behavior, pinned so a regression is obvious: the first
    # window contributes only its last token, dropping ~1022 leading tokens.
    assert counted_tokens(20000, 1024, 512, clamp=False) == 18977


def test_clamp_covers_more_than_unclamped():
    for n in (2000, 5000, 20000):
        assert counted_tokens(n, 1024, 512, True) > counted_tokens(n, 1024, 512, False)


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
