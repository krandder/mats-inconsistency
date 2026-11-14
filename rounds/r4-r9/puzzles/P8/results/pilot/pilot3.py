"""One prequential codelen job (same settings as pilot2: BOUNDS 16,64,256,1024; 250 steps/block; seed 0)."""
import sys, json, time
sys.argv, job = sys.argv[:1], sys.argv[1]
import numpy as np, torch
import pilot2 as P
torch.set_num_threads(1)
x, g, rng = P.data(P.BOUNDS[-1], 0)
cat = lambda *zs: np.concatenate([x] + [z[:, None] for z in zs], 1)
jobs = {
    "g2_X": (x, g[2]), "Y_Xg2": (cat(g[2]), g[8]), "g4_Xg2": (cat(g[2]), g[4]),
    "g6_Xg4": (cat(g[4]), g[6]), "Y_Xg6": (cat(g[6]), g[8]), "g6_X": (x, g[6]),
}
t0 = time.time()
inp, tgt = jobs[job]
b = P.preq(inp, tgt, 0)
json.dump({"job": job, "bits": b, "secs": round(time.time() - t0)}, open(f"/tmp/p8pilot/job_{job}.json", "w"))
print(job, round(b), round(time.time() - t0), flush=True)
