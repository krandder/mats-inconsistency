#!/usr/bin/env python3
"""R8-E1 audit, check 2: an independent no-leak test of puct8 (all 4 cells), different from test_r8e1.py's: random untrained
theta_real (d 32), 10 test boards, 3 RNG seeds, unread cells' real readings RE-DRAWN at random (not just flipped) before each wave;
the picks up to and including that wave must not change. Also: E3.prospect's candidate (hbar, dl) must be identical when every
non-root reading on the board is re-drawn. Imports e1v5 (code under test). CPU, 2 threads.
Run: OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r4e3a/pylib python3 a2_puct_noleak.py"""
import json
import os
import sys

import torch

torch.set_num_threads(2)
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/li-net/r8e1")
import e1v5 as E  # noqa: E402
import e1v3 as E3  # noqa: E402
import e1v4 as E4  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import wm  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
nb = 10
torch.manual_seed(11)
m = wm.WorldModel(d=32, layers=2, heads=2, ff=64).eval()
B = A2.boards2_t(A2.STREAM_TEST2, nb, "cpu")
ordT = torch.as_tensor(E.anchor_order(A2.STREAM_TEST2, nb))
ev, l0 = E.root(m, B, ordT, 2)
lprior = E.prior_logits(m, B)
trust = wm.TrustNet(32 + 3).eval()
anyp = {"looped_attn": (E3.KeyNet("K3", 32).eval(), E.LoopedAttnT3P(3.4).eval()), "attn_K3": (E3.KeyNet("K3", 32).eval(), E.AttnL3P(3.4).eval())}
for _, p in anyp.values():
    with torch.no_grad():
        p.out.weight.normal_(0, 0.2)            # non-trivial weights so mu moves with the reads
sc = {"alpha": 0.2, "eta": 0.08, "u_K1": 1.0, "u_K3": 0.5, "econ_R7_u": 1.3, "econ_R7_lam": 1.4, "econ_D3_u": 1.0, "econ_D3_lam": 1.0}
P = E.Pools8(anyp, E3.KeyNet("K1", 32).eval(), wm.AttnL3().eval(), E3.KeyNet("K4", 32).eval(), E3.LoopedLinT4().eval(),
             E3.KeyNet("K3", 32).eval(), E4.JacobiLinT4().eval(), E3.KeyNet("K3", 32).eval(), E.JacobiPrecT4().eval(), sc, 3.4)
cfg = E.Cfg8(device="cpu", chunk=4096, d=32)
anchors = A2.cells_mask(B["anchors"], "cpu").bool()
rep = {}
for km, sm in E.CELLS:
    ok = True
    for rs in (1, 2, 3):
        _, picks = E.puct8(m, trust, P, B, ev, l0, lprior, rs % 2 == 1, torch.Generator().manual_seed(100 + rs), cfg, km, sm, 4)
        for w in range(1, 5):
            cov = E3.coverage(torch.cat(picks[:w - 1], 1), "cpu").bool() if w > 1 else torch.zeros(nb, 36, dtype=torch.bool)
            B2 = dict(B)
            rnd = torch.rand(nb, 36, generator=torch.Generator().manual_seed(1000 * rs + w)) < 0.5
            B2["s"] = torch.where(~cov & ~anchors, rnd, B["s"])          # unread cells re-drawn at random
            _, p2 = E.puct8(m, trust, P, B2, ev, l0, lprior, rs % 2 == 1, torch.Generator().manual_seed(100 + rs), cfg, km, sm, w)
            ok &= all(torch.equal(picks[i], p2[i]) for i in range(w))
    rep[f"{km}_{sm}_picks_invariant_to_unread_readings"] = bool(ok)
# positive control: re-drawing ALL non-anchor readings (read cells included) must change later waves' picks somewhere
ctrl = {}
for km, sm in E.CELLS:
    _, picks = E.puct8(m, trust, P, B, ev, l0, lprior, True, torch.Generator().manual_seed(101), cfg, km, sm, 4)
    B2 = dict(B)
    B2["s"] = torch.where(anchors, B["s"], torch.rand(nb, 36, generator=torch.Generator().manual_seed(77)) < 0.5)
    _, p2 = E.puct8(m, trust, P, B2, ev, l0, lprior, True, torch.Generator().manual_seed(101), cfg, km, sm, 4)
    ctrl[km + "_" + sm] = {"wave1_identical": bool(torch.equal(picks[0], p2[0])),
                           "boards_with_changed_later_picks": int(sum(not torch.equal(picks[w][i], p2[w][i]) for i in range(nb) for w in (1, 2, 3)))}
rep["positive_control_all_readings_redrawn"] = ctrl
B3 = dict(B)
B3["s"] = torch.where(anchors, B["s"], torch.rand(nb, 36, generator=torch.Generator().manual_seed(5)) < 0.5)
mu = torch.sigmoid(l0)
h1, d1 = E3.prospect(m, B, ev, l0, mu)
h2, d2 = E3.prospect(m, B3, ev, l0, mu)
rep["prospect_invariant_to_all_non_root_readings"] = bool(torch.equal(h1, h2) and torch.equal(d1, d2))
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(OUT, "a2_puct_noleak.json"), "w"), indent=1)
