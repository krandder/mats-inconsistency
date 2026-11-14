"""c6_agree.py (checker) -- across my search runs: do two policies return the same board on every puzzle both solved,
and what does each cost on those puzzles."""
import os, json, itertools
import numpy as np, torch
R = {a: torch.load(f"out/c3_search_{a}.pt", weights_only=False) for a in ("S1", "S2", "S3", "S4") if os.path.exists(f"out/c3_search_{a}.pt")}
OUT = {}
for a, b in itertools.combinations(R, 2):
    n = min(len(R[a]["acc_i"]), len(R[b]["acc_i"])); both = (R[a]["acc_i"][:n] >= 0) & (R[b]["acc_i"][:n] >= 0)
    same = sum(int((R[a]["answers"][p] == R[b]["answers"][p]).all()) for p in np.where(both)[0])
    fa, fb = R[a]["fl_calls"][:n][both], R[b]["fl_calls"][:n][both]; ta, tb = R[a]["fl_table"][:n][both], R[b]["fl_table"][:n][both]
    OUT[f"{a}-{b}"] = dict(both=int(both.sum()), same=same, ratio_mean=float(fb.mean() / fa.mean()), ratio_median=float(np.median(fb) / np.median(fa)), ratio_mean_table=float(tb.mean() / ta.mean()))
    print(f"{a} and {b}: same answer on {same} of {int(both.sum())} puzzles both solved; FLOPs (one call per context read) {a} mean {fa.mean():.3e} median {np.median(fa):.3e}, "
          f"{b} mean {fb.mean():.3e} median {np.median(fb):.3e}; ratio of means {fb.mean()/fa.mean():.2f}, of medians {np.median(fb)/np.median(fa):.2f}; proposals only: ratio of means {tb.mean()/ta.mean():.2f}")
json.dump(OUT, open("out/c6_agree.json", "w"), indent=1)
