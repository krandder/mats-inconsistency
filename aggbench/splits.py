"""Seed ranges for train/dev/test/smoke. Every episode has its own seed.

Episode j of split S is generated from seed SPLIT_BASE[S] + j, so splits are
disjoint by construction as long as no split asks for more than SPLIT_SIZE
episodes. `verify_disjoint` checks this on the actual seed arrays used.
"""
import numpy as np

SPLIT_SIZE = 10_000_000
SPLIT_BASE = {"train": 0, "dev": 10_000_000, "test": 20_000_000, "smoke": 90_000_000}


def episode_seeds(split, n, start=0):
    """Seeds for episodes start .. start+n-1 of `split`."""
    if split not in SPLIT_BASE:
        raise ValueError(f"unknown split {split!r}; use one of {sorted(SPLIT_BASE)}")
    if start < 0 or n < 0 or start + n > SPLIT_SIZE:
        raise ValueError(f"episodes [{start}, {start + n}) fall outside split {split!r}")
    return SPLIT_BASE[split] + start + np.arange(n, dtype=np.int64)


def split_of(seed):
    """Name of the split a seed belongs to."""
    for name, base in SPLIT_BASE.items():
        if base <= seed < base + SPLIT_SIZE:
            return name
    raise ValueError(f"seed {seed} is in no split")


def verify_disjoint(*seed_arrays):
    """True iff no seed appears in two of the given arrays (or twice in one)."""
    allseeds = np.concatenate([np.asarray(s, dtype=np.int64).ravel() for s in seed_arrays])
    return len(np.unique(allseeds)) == len(allseeds)
