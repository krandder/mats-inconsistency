"""data.py -- streams of REAL data (world side). Solved grids, and (puzzle, solution) pairs, with the generator's
own symmetries as augmentation (digit relabelling; for puzzles also the cell symmetries stored by the generator).
No solving logic here: this file only reads arrays written by gen_*.py and serves samples of them."""
import numpy as np, torch


class GridStream:
    """Uniform draws from {base grids} x {digit relabellings}: uniform over all grids of the given orbits."""

    def __init__(self, B, K, seed):
        self.B = np.asarray(B, dtype=np.int64); self.K = K; self.rng = np.random.default_rng(seed)

    def sample(self, n):
        g = self.B[self.rng.integers(len(self.B), size=n)]
        perm = np.argsort(self.rng.random((n, self.K)), axis=1)
        return torch.from_numpy(np.take_along_axis(perm, g, axis=1))


class PuzzleStream:
    """Real (puzzle, solution) pairs; each draw is relabelled and moved by a random stored cell symmetry."""

    def __init__(self, puz, sol, K, seed, syms=None):
        self.puz = np.asarray(puz, dtype=np.int64); self.sol = np.asarray(sol, dtype=np.int64)
        self.K = K; self.rng = np.random.default_rng(seed); self.syms = syms

    def sample(self, n):
        idx = self.rng.integers(len(self.puz), size=n)
        clue = self.puz[idx] >= 0; x = self.sol[idx]
        if self.syms is not None:
            t = self.syms[self.rng.integers(len(self.syms), size=n)]
            clue = np.take_along_axis(clue, t, axis=1); x = np.take_along_axis(x, t, axis=1)
        perm = np.argsort(self.rng.random((n, self.K)), axis=1)
        x = np.take_along_axis(perm, x, axis=1)
        return torch.from_numpy(x), torch.from_numpy(clue)
