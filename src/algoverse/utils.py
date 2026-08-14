""" Small helper functions."""

# for reading and writing results
import json
import os
# Object-oriented filesystem paths
from pathlib import Path
# Python's random number generator
import random

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Make all randomness repeatable.

    ML code uses randomness everywhere (shuffling data, initializing model
    weights...). Three different libraries each have their own random number
    generator, so we must seed all of them. After calling set_seed(42), a rerun
    of the same code produces the *same* "random" numbers — and the same result.
    That's what makes an experiment reproducible.

    Args:
    - int seed: the seed to use for all random number generators
    """
    random.seed(seed)                  # Python's own RNG
    np.random.seed(seed)               # numpy's RNG
    torch.manual_seed(seed)            # PyTorch on CPU
    torch.cuda.manual_seed_all(seed)   # PyTorch on GPU(s) — safe no-op if there's no GPU

def get_device() -> str:
    """
    Returns a device string that torch APIs accept, e.g. model.to(get_device()):
    - "cuda" if an NVIDIA GPU is available (Colab)
    - "mps" on Apple-silicon Macs (local smoke tests)
    - "cpu" otherwise

    (Previously returned prose like "Using GPU", which crashes the first
    time it is passed to torch. Anything printing a status message should
    format this value itself.)
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

def save_checkpoint(path, step, model, optimizer=None, **extra) -> None:
    """
    Save training progress to a file, safely.

    Args:
    - Path path: where to save the checkpoint
    - int step: the current training step (used for resuming training)
    - torch.nn.Module model: the model to save
    - torch.optim.Optimizer optimizer: the optimizer to save (optional)
    """
    # create the folder if it doesn't exist yet
    path.parent.mkdir(parents=True, exist_ok=True)  

    # Bundle everything into one dictionary. .state_dict() = "all the numbers inside this object, as a saveable dict".
    state = {"step": step, "model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    new = path.with_suffix(path.suffix + ".tmp")  # e.g. latest.pt -> latest.tmp
    torch.save(state, new)          # slow part: write the data to the temp file
    new.replace(path)           # instant part: atomically swap it into place

def load_checkpoint(path, model, optimizer=None) -> int:
    """Restore saved training progress. Returns the step to resume from.

    Args:
    - Path path: where to load the checkpoint from
    - torch.nn.Module model: the model to load into
    - torch.optim.Optimizer optimizer: the optimizer to load into (optional)

    Return:
    - The step to resume from. 
    - If no checkpoint file exists yet (a brand-new run), returns 0 
    """
    if not path.exists():
        return 0
    else:
        # Load checkpoint path.
        state = torch.load(path, map_location="cpu")  
        # Restore model weights and optimizer state from checkpoint.
        model.load_state_dict(state["model"])          
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"]) 
        return state["step"] + 1

def append_jsonl(path, record: dict) -> None:
    """Add one line to a results log file.

    "jsonl" = JSON Lines: a text file where each line is one JSON dictionary.
    We append one line per finished run — its config and final metrics — to
    `results/results.jsonl` in the shared Drive folder, so the whole team has
    a permanent record of every run ever made.

    Args:
    - Path path: where to append the record
    - dict record: the record to append (will be converted to JSON)

    A killed process can leave a final JSON fragment without a newline. Before
    appending, isolate that fragment on its own malformed line so the next
    valid record cannot be swallowed by tolerant readers. Each append is
    flushed and fsynced before returning. This assumes one writer process per
    results file, which is the project's one-Colab-session-per-run workflow.
    """
    path = Path(path)
    # create the folder if it doesn't exist yet
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

def read_jsonl(path) -> list[dict]:
    """
    Read a JSONL file into plain dicts (not dataclasses: the scorer adds
    temporary working fields to rows, which dicts allow and frozen shapes
    would fight).

    Args:
    - Path path: where to read the JSONL file from

    Returns:
    - list[dict]: the rows of the JSONL file, as plain dicts
    """
    rows = []
    with open(path, "r", encoding="utf-8") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
