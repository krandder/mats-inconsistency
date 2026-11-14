"""Gate curves (restarts 0-1, val blocks 0-1) for one bank dir: B1 accuracy vs depth with bootstrap CIs."""
import json, os, sys
import numpy as np
bank, noise = sys.argv[1], sys.argv[2]
corr = np.concatenate([np.load(os.path.join(bank, f"b{b:02d}_g0.npz"))["corr"] for b in (0, 1)])  # (512, 2, 1024)
P = corr.shape[0]
depths = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
rng = np.random.default_rng(0); IDX = rng.integers(0, P, (4000, P))
def ci(x):  # x: per-puzzle values
    bs = x[IDX].mean(1); return [round(100 * x.mean(), 3), round(100 * np.quantile(bs, .025), 3), round(100 * np.quantile(bs, .975), 3)]
b1 = [ci(corr[:, :, d - 1].mean(1)) for d in depths]
b2 = [ci(corr[:, :, d - 1].any(1).astype(float)) for d in depths]
lost = int((np.maximum.accumulate(corr, -1) & ~corr).any(-1).sum())
out = {"note": f"EqR released Sudoku-Extreme checkpoint, path noise {noise}, init std 1.0; val blocks 0-1 (512 puzzles) x restarts 0-1; exact accuracy (%) at outer step d, mean over the 2 restarts; [mean, lo, hi] 95% bootstrap CI over puzzles",
       "depth": depths, "B1_at_step": b1, "B2_any_restart_correct_at_step": b2, "trajectories_correct_then_lost": lost,
       "n_trajectories": int(P * 2)}
print(json.dumps(out))
