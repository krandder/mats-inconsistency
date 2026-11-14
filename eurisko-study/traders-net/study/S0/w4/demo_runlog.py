"""Example of the per-run CSV V1 consumes: a tiny prequential run (streams_ext, worker1's loop)
logged with panel.panel_row every 25 steps. 3 seeds -> w4/demo/demo-s{seed}.csv (+ meta.json)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "substrate"))
import torch
torch.set_num_threads(1)
import panel, runlog, prequential as pq, streams_ext as sx, model as md

names = list(sx.FAMILIES) if hasattr(sx, "FAMILIES") else None
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    s = sx.make_streams(Q=16, T=2400, seed=seed, heldout=2, regime_every=800)
    st = sx.to_array_stream(s)   # hides held-out windows before release (was a raw ArrayStream: leak)
    net = md.Net(st.Q, st.W, d=32, H=2, L=1)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    fc = panel.FlopCounter()
    rl = runlog.RunLog(os.path.join(HERE, "demo"), f"demo-s{seed}",
                       meta=dict(seed=seed, stream=s.spec, model="Net d=32 H=2 L=1", B=8,
                                 code=runlog.code_hash([os.path.join(HERE, "..", "substrate", f) for f in
                                                        ("panel.py", "runlog.py", "prequential.py", "streams_ext.py")])))
    buf = []
    fam_all = s.fam_t[:, s.W:].T                    # current family per position (regime shifts)

    def on_step(step, P, Z, Y, M):
        fc.add(panel.net_forward_flops(len(P), st.Q, st.W, 32, 2, 1), "test")
        fc.add(3 * panel.net_forward_flops(len(P), st.Q, st.W, 32, 2, 1), "train")   # fwd+bwd ~3x
        buf.append((Z, Y, M, fam_all[P]))
        if (step + 1) % 25 == 0:
            z, y, m, f = (np.concatenate(x) for x in zip(*buf)); buf.clear()
            row = panel.panel_row(panel.sigmoid(z[m]), y[m], fam=f[m], names=names, flops=fc)
            rl.log(step + 1, row)
    res = pq.run_prequential(net, st, opt, B=8, on_step=on_step)
    rl.log(res["steps"], {"final_" + k: v for k, v in panel.panel_from_prequential(res, s.fam0, names).items()
                          if k in ("bce", "brier", "ece")})
    rl.close()
    print(seed, "done", rl.path)
