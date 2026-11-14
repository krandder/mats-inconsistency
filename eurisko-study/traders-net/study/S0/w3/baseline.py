"""PREREG-w3 arm 5: difficulty baseline on the LI families. Prequential single pass (worker1's
run_prequential) of a small Net and the per-question Logistic on make_li_stream(seed).
Writes w3/baseline-s<seed>.json. Usage: python3 baseline.py SEED [T]"""
import json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "substrate"))
from li_families import make_li_stream, violations
from prequential import run_prequential
from model import Net, Logistic

seed = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
torch.set_num_threads(2); torch.manual_seed(seed)
st = make_li_stream(seed, T); ps = st.as_positions()
net = Net(ps.Q, ps.W, d=32, H=4, L=1); lm = Logistic(ps.Q, ps.W)
t0 = time.time()
out = run_prequential(net, ps, torch.optim.Adam(net.parameters(), 1e-3), B=4, baseline=lm,
                      opt_b=torch.optim.Adam(lm.parameters(), 1e-2))
res = {"seed": seed, "T": T, "Q": st.Q, "wall_s": time.time() - t0, "fam_names": st.fam_names}
y, m = out["y"].astype(float), out["mask"]
t = np.arange(ps.n_pos) + st.W
for name, z in (("net", out["z"]), ("logistic", out["zb"])):
    p = 1 / (1 + np.exp(-z.astype(float)))
    bce = -(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
    r = {}
    for key, groups in (("fam", [(n, st.fam == i) for i, n in enumerate(st.fam_names)]),
                        ("kind", [(k, st.kind == k) for k in np.unique(st.kind)])):
        for g, qm in groups:
            mm = m & qm[None, :]
            r[f"{key}:{g}:bce"] = float(bce[mm].mean())
            r[f"{key}:{g}:brier"] = float(((p - y) ** 2)[mm].mean())
    v = violations(p, st, t=t)
    r.update({f"viol:{k}:{f}": d[f] for k, d in v.items() for f in ("mean", "frac", "n")})
    res[name] = r
path = os.path.join(HERE, f"baseline-s{seed}.json")
json.dump(res, open(path, "w"), indent=1)
print(path, f"{res['wall_s']:.0f}s")
