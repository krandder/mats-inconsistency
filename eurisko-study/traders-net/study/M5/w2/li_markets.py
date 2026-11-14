"""Run the S0 LI-stream markets (prequential Net d=32 L=1 + per-question Logistic, recipe of S0/w3/baseline.py)
and dump trade-time logits for M5 arbitrage accounting. Usage: python3 li_markets.py SEED [T]
Writes /tmp/traders-net/M5/w2/li-prices-s<seed>.npz with z (net), zb (logistic), y, mask, t (trade times)."""
import os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
from li_families import make_li_stream
from prequential import run_prequential
from model import Net, Logistic
seed = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
torch.set_num_threads(1); torch.manual_seed(seed)
st = make_li_stream(seed, T); ps = st.as_positions()
net = Net(ps.Q, ps.W, d=32, H=4, L=1); lm = Logistic(ps.Q, ps.W)
t0 = time.time()
out = run_prequential(net, ps, torch.optim.Adam(net.parameters(), 1e-3), B=4, baseline=lm,
                      opt_b=torch.optim.Adam(lm.parameters(), 1e-2))
path = f"/tmp/traders-net/M5/w2/li-prices-s{seed}.npz"
np.savez_compressed(path, z=out["z"].astype(np.float32), zb=out["zb"].astype(np.float32), y=out["y"],
                    mask=out["mask"], t=np.arange(ps.n_pos) + st.W, seed=seed, T=T)
print(path, f"{time.time()-t0:.0f}s", out["z"].shape)
