#!/usr/bin/env python3
"""Independent self-audit of the E1-D0 headline numbers (CPU): reloads each seed's saved theta_real checkpoint, rebuilds the
inputs from data/test.npz with its own code (not eval_ms.py), recomputes the 0-step belief KL to P*_full, the 5-known-slot
NLL and the first-ply battle accuracy, and compares them with runs/s<seed>/eval/d0.json. Writes report/audit_independent.md.
   python3 audit_ms.py"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_ms import RealNet, MAXP, NROOT  # noqa: E402

torch.set_num_threads(4)
te = np.load(os.path.join(HERE, "data", "test.npz"))
o = np.load(os.path.join(HERE, "data", "oracle_d1.npz"))
key = {(int(g), int(t)): i for i, (g, t) in enumerate(zip(te["game"], te["t"]))}
rows = np.array([key[(int(g), int(t))] for g, t in zip(o["game"], o["t"])])


def tape(ix, slot_val, plies=None):
    T = lambda k: torch.tensor(te[k][ix].astype(np.int64))
    z = torch.zeros(len(ix), MAXP, dtype=torch.long)
    tp = {"sq_content": T("sq_content"), "sq_ownflag": T("sq_ownflag"), "sq_slot": T("sq_slot"), "slot_loc": T("slot_loc"),
          "slot_flags": T("slot_flags"), "slot_mcount": T("slot_mcount"), "slot_pubrev": T("slot_pubrev"),
          "slot_val": torch.tensor(slot_val.astype(np.int64)), "turn": T("turn")}
    names = ("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")
    for c, k in enumerate(names):
        tp[k] = z.clone() if plies is None else torch.tensor(plies[..., c].astype(np.int64))
    return tp


lines = ["| seed | quantity | d0.json | independent CPU recomputation | abs diff |", "|---|---|---|---|---|"]
worst = 0.0
for s in (101, 102, 103):
    net = RealNet()
    net.load_state_dict(torch.load(os.path.join(HERE, "runs", f"s{s}", "real_100.pt"), map_location="cpu"))
    net.eval()
    ref = json.load(open(os.path.join(HERE, "runs", f"s{s}", "eval", "d0.json")))
    pub = te["slot_pubrev"][rows] == 1
    tru = te["slot_true"][rows]
    with torch.no_grad():
        h, _ = net(tape(rows, np.where(pub, tru, 0)), root_only=True)
        q = torch.softmax(net.h_belief(h[:, 16:22]).double(), -1).numpy()
    P = o["pstar_full"]
    kl = np.where(P > 0, P * (np.log(np.clip(P, 1e-300, 1)) - np.log(np.clip(q, 1e-300, 1))), 0).sum(-1)
    hid = ~pub
    kl0 = float(((kl * hid).sum(1) / hid.sum(1)).mean())
    nll = []
    with torch.no_grad():
        for sl in range(6):
            v = tru.copy()
            v[:, sl] = 0
            h5, _ = net(tape(rows, v), root_only=True)
            q5 = torch.softmax(net.h_belief(h5[:, 16 + sl]).double(), -1).numpy()
            p_true = q5[np.arange(len(rows)), tru[:, sl] - 1]
            nll.append(-np.log(p_true)[hid[:, sl]])
    nll5 = float(np.concatenate(nll).mean())
    bat = np.nonzero((te["n_ply"] >= 1) & (te["ply"][:, 0, 3] >= 1))[0]
    accs = []
    with torch.no_grad():
        for c0 in range(0, len(bat), 2048):
            ix = bat[c0:c0 + 2048]
            ply = te["ply"][ix].copy()
            ply[np.arange(MAXP)[None] >= te["n_ply"][ix][:, None]] = 0
            hb, _ = net(tape(ix, te["slot_true"][ix], ply))
            lo = net.h_trans(hb[:, NROOT])[:, :5]
            accs.append((lo.argmax(1).numpy() == ply[:, 0, 3]))
    acc = float(np.concatenate(accs).mean())
    for name, mine, theirs in (("0-step KL to P*_full", kl0, ref["kl0_full"]), ("5-known NLL", nll5, ref["nll_5known"]),
                               ("battle accuracy ply 0", acc, ref["battle_acc_ply0"])):
        d = abs(mine - theirs)
        worst = max(worst, d)
        lines.append(f"| {s} | {name} | {theirs:.6f} | {mine:.6f} | {d:.1e} |")
    lines.append(f"| {s} | battles scored | {ref['n_battles_ply0']} | {len(bat)} | |")
lines.append(f"\nLargest absolute difference: {worst:.1e} (GPU float32 evaluation vs CPU recomputation with float64 softmax).")
open(os.path.join(HERE, "report", "audit_independent.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
