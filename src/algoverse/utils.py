"""Small shared helpers: seeding, device selection, append-only JSONL writes.

Stdlib at import time. torch and numpy are imported inside the two helpers
that need them, so the scoring and analysis modules can import this file on
a machine with no ML stack.
"""
import json
import os
import random
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Seed Python, numpy and torch (CPU and every GPU) so a rerun repeats."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # no-op without a GPU


def get_device() -> str:
    """A device string torch accepts: "cuda" on an NVIDIA box, "mps" on Apple
    silicon, else "cpu"."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def append_jsonl(path, record: dict) -> None:
    """Append one JSON record as a line to a results file.

    A killed process can leave a final JSON fragment without a newline. Before
    appending, isolate that fragment on its own malformed line so the next
    valid record cannot be swallowed by tolerant readers. Each append is
    flushed and fsynced before returning. This assumes one writer process per
    results file, which is the project's one-process-per-run workflow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record) + "\n").encode("utf-8")
    with open(path, "a+b") as appended_file:
        appended_file.seek(0, os.SEEK_END)
        if appended_file.tell() > 0:
            appended_file.seek(-1, os.SEEK_END)
            if appended_file.read(1) != b"\n":
                appended_file.write(b"\n")
        appended_file.write(payload)
        appended_file.flush()
        os.fsync(appended_file.fileno())
