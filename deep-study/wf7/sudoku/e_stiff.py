"""e_stiff.py -- NOT pre-registered (addendum A9). A stiffness profile of M_A by context size, from real grids.
For contexts of d revealed cells of a real validation grid: how much does M rule out (atoms below 1e-3), and how
often is reality among the things it ruled out? Scored with the exact counter: of the placements M rules out, how
many are truly impossible; of the truly impossible placements, how many does M rule out.
Usage: python -u e_stiff.py"""
import json, math, time
import numpy as np, torch
import core, data
import oracle as O            # scoring only (exact conditionals)

torch.set_num_threads(1); K, N = 6, 36; T0 = time.time(); TH = math.log(1e-3)
d = np.load("data/k6_base.npz"); B6 = d["B"]; bs = d["base_split"]
va = data.GridStream(B6[bs == 1], K, 777); C6 = O.Counter(B6, K)
M = core.Net(core.load_model("ckpt/mA6.pt")[0], K, "mA6")
gen = torch.Generator().manual_seed(1); rows = []
print(f"{'cells revealed':>14s} {'ruled out':>10s} {'reality ruled out':>18s} {'min log q(real)':>16s} {'ruled out & truly impossible':>29s} {'impossible & ruled out':>23s} {'direct':>8s} {'no completion':>14s}")
for depth in (1, 2, 3, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35):
    n = 1000; X = va.sample(n)
    rank = torch.rand(n, N, generator=gen).argsort(1).argsort(1)
    S = torch.where(rank < depth, X, torch.full_like(X, -1)); LP = M.logp(S); emp = S < 0
    real = LP.gather(2, X[:, :, None]).squeeze(2)[emp]
    veto = (LP < TH) & emp[:, :, None]
    power = float(veto.sum() / (emp.sum() * K)); false_veto = float((real < TH).float().mean())
    # exact scoring on the first 150 states
    m = 150; tp = fp = fn = 0; fn_direct = fn_dead = n_direct = n_dead = 0
    loc = O.local_candidates(S[:m].numpy(), K)
    for a in range(m):
        tot, P = C6.conditionals(S[a].numpy()); e = emp[a].numpy()
        imp = (P == 0) & e[:, None]; v = veto[a].numpy()
        tp += int((v & imp).sum()); fp += int((v & ~imp).sum()); fn += int((~v & imp).sum())
        direct = imp & ~loc[a]; dead = imp & loc[a]
        n_direct += int(direct.sum()); n_dead += int(dead.sum()); fn_direct += int((direct & ~v).sum()); fn_dead += int((dead & ~v).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    rd = 1 - fn_direct / max(1, n_direct); rdead = 1 - fn_dead / max(1, n_dead) if n_dead else float("nan")
    rows.append(dict(depth=depth, ruled_out=power, reality_ruled_out=false_veto, min_real=float(real.min()), precision=prec, recall=rec,
                     recall_direct=rd, recall_dead=rdead, share_dead=n_dead / max(1, n_direct + n_dead)))
    print(f"{depth:14d} {power:10.4f} {false_veto:18.2e} {real.min():16.2f} {prec:29.5f} {rec:23.4f} {rd:8.4f} {rdead:14.4f}", flush=True)
json.dump(rows, open("out/e_stiff.json", "w"), indent=1)
print(f"columns: 'ruled out' = share of (empty cell, value) pairs with q < 1e-3; 'reality ruled out' = share of real values with q < 1e-3 (n = 1,000 grids per row);\n"
      f"exact columns on 150 states per row: precision of 'ruled out', recall on truly impossible placements, recall split into directly rule-breaking placements "
      f"and placements that are locally allowed but have no completion. total time {time.time()-T0:.0f}s")
