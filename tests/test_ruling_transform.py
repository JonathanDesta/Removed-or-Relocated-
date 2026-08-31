"""Rung-1 tests for the truncated->invalid ruling transform and probe curves.

The ways these can silently lie, each caught:
  1. the ruling transform mutating the SOURCE file (append-only violation);
  2. a truncated row surviving as valid, or keeping its deceptive label;
  3. an already-invalid row double-counted as reclassified;
  4. a null probe AUROC drawn as a value instead of reported as a gap.

Stdlib + matplotlib only.

    python3 tests/test_ruling_transform.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.relocation import apply_truncated_invalid_ruling


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "rows.jsonl"
        rows = [
            {"valid": True, "hit_max_tokens": False, "deceptive": True},
            {"valid": True, "hit_max_tokens": True, "deceptive": False},
            {"valid": False, "hit_max_tokens": True, "deceptive": None,
             "invalid_reason": "unparseable"},
        ]
        src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        before = src.read_text()

        dst = Path(tmp) / "out" / "rows.jsonl"
        n, changed = apply_truncated_invalid_ruling(src, dst)

        # 1: source untouched
        assert src.read_text() == before
        assert n == 3 and changed == 1, (n, changed)

        out = [json.loads(l) for l in dst.read_text().splitlines()]
        # clean row passes through unchanged
        assert out[0] == rows[0]
        # 2: truncated valid row reclassified, label nulled
        assert out[1]["valid"] is False
        assert out[1]["invalid_reason"] == "truncated"
        assert out[1]["deceptive"] is None
        # 3: already-invalid row untouched (keeps its original reason)
        assert out[2]["invalid_reason"] == "unparseable"

    # 4: null AUROC -> gap in metadata, files written
    from algoverse.plotting import render_probe_curves, synthetic_probe_curves
    with tempfile.TemporaryDirectory() as tmp:
        curves = synthetic_probe_curves()
        meta = render_probe_curves(curves, str(Path(tmp) / "probe_curves"))
        assert all(Path(p).is_file() for p in meta["paths"])
        assert meta["null_layers"] == {"m0": [0], "md": [0], "me": [0]}, \
            meta["null_layers"]

    print("PASS test_ruling_transform")


if __name__ == "__main__":
    main()
