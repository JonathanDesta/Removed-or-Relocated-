"""
This file makes the `algoverse/` folder work as a Python package.

The utils re-exports below need torch. Scoring (tasks.py) and analysis
(metrics.py) deliberately do NOT: they must run on a bare laptop with no ML
stack, so the sweep can be scored and the curves computed anywhere. When
torch is missing we skip the re-exports instead of crashing the import;
anything that genuinely needs them (training, generation) will fail with a
clear error at its own import instead.
"""
try:
    from algoverse.utils import (
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
except ModuleNotFoundError:  # no torch on this machine: scoring still works
    __all__ = []
