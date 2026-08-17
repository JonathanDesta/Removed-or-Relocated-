"""
This file makes the `algoverse/` folder work as a Python package.
"""
from yksa.utils import (
    append_jsonl,      # add one finished run to the shared results log
    get_device,        # "cuda" if a GPU is available, else "cpu"
    load_checkpoint,   # restore training progress (returns the step to resume from)
    read_jsonl,       # read a results log file into a list of dicts
    save_checkpoint,   # save training progress (safely/atomically)
    set_seed,          # make randomness repeatable
)

# __all__ = the official public list of what this package offers.
__all__ = [
    "append_jsonl",
    "get_device",
    "load_checkpoint",
    "read_jsonl",
    "save_checkpoint",
    "set_seed",
]
