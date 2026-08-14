"""Generate the M_D and M_C fine-tuning datasets.

    python scripts/build_finetune_data.py --out-dir data/finetune --n 1500 --seed 42

Writes m_d_train.jsonl, m_c_train.jsonl (chat format for SFT), their
.meta.jsonl companions, and manifest.json. Every reply is validated with
the real scorer at build time, so a mislabeled example fails the build
instead of reaching training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.data import build_finetune_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/finetune")
    parser.add_argument("--n", type=int, default=1500, help="rows per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-system", action="store_true",
                        help="fold the leading system turn into the user turn")
    args = parser.parse_args()
    build_finetune_datasets(
        args.out_dir, n_per_dataset=args.n, seed=args.seed,
        fold_system=args.fold_system,
    )
