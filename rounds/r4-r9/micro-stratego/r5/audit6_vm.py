#!/usr/bin/env python3
"""Independent recomputation (run on the VM after eval6.py) of the E2-D0 headline numbers for one seed/arch: reloads
real_100.pt, rebuilds the root tapes from the raw npz records with its own code (not model6.tape_from / eval6), and
recomputes the 0-step KL to P*_full per hidden slot, the 7-known-slot NLL and the ply-0 battle accuracy on the test
split, with float64 softmax. Writes <out>/eval/audit_vm.json with the differences from eval/d0.json.
   python3 audit6_vm.py --seed 101 --arch primary --out OUT"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model6 import ARCHS, MAXP, NROOT, NS, NSQ, RealNet  # noqa: E402

DATA = os.environ.get("R5E2_DATA", "/tmp/r5e2/data")
ROOT = ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev")
PLY = ("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RealNet(**ARCHS[a.arch]).to(dev)
    net.load_state_dict(torch.load(os.path.join(a.out, "real_100.pt"), map_location=dev))
    net.eval()
    te = np.load(os.path.join(DATA, "test.npz"))
    o = np.load(os.path.join(DATA, "oracle_d0_test.npz"))
    key = {(int(g), int(t)): i for i, (g, t) in enumerate(zip(te["game"], te["t"]))}
    rows = np.array([key[(int(g), int(t))] for g, t in zip(o["game"], o["t"])])

    def tape(ix, slot_val, plies=None):
        tp = {k: torch.tensor(te[k][ix].astype(np.int64), device=dev) for k in ROOT}
        tp["slot_val"] = torch.tensor(slot_val.astype(np.int64), device=dev)
        tp["turn"] = torch.tensor(te["turn"][ix].astype(np.int64), device=dev)
        for c, k in enumerate(PLY):
            tp[k] = torch.tensor((plies[..., c] if plies is not None else np.zeros((len(ix), MAXP))).astype(np.int64), device=dev)
        return tp

    pub = te["slot_pubrev"][rows].astype(bool)
    tru = te["slot_true"][rows].astype(np.int64)
    hid = (~pub) & (te["slot_loc"][rows] < NSQ)
    kls, nlls = [], []
    with torch.no_grad():
        for c0 in range(0, len(rows), 512):
            ix = rows[c0:c0 + 512]
            sv = np.where(pub[c0:c0 + 512], tru[c0:c0 + 512], 0)
            h, _ = net(tape(ix, sv), root_only=True)
            lg = net.h_belief(h[:, 36:36 + NS]).double()
            q = torch.softmax(lg, -1).cpu().numpy()
            P = o["pstar_full"][c0:c0 + 512]
            k = np.where(P > 0, P * (np.log(np.clip(P, 1e-300, 1)) - np.log(np.clip(q, 1e-9, 1))), 0).sum(-1)
            hh = hid[c0:c0 + 512]
            kls.append(np.where(hh, k, 0).sum(1) / np.maximum(hh.sum(1), 1))
            for s in range(NS):
                sv7 = tru[c0:c0 + 512].copy()
                sv7[:, s] = 0
                h7, _ = net(tape(ix, sv7), root_only=True)
                q7 = torch.log_softmax(net.h_belief(h7[:, 36 + s]).double(), -1).cpu().numpy()
                nl = -q7[np.arange(len(ix)), tru[c0:c0 + 512, s] - 1]
                nlls.append(nl[hh[:, s]])
        ok = []
        for c0 in range(0, len(te["t"]), 2048):
            ix = np.arange(c0, min(len(te["t"]), c0 + 2048))
            ply = te["ply"][ix].astype(np.int64)
            npl = te["n_ply"][ix].astype(np.int64)
            valid = np.arange(MAXP)[None] < npl[:, None]
            h, _ = net(tape(ix, te["slot_true"][ix], ply * valid[..., None]))
            lo = net.h_trans(h[:, NROOT])[:, :5]
            bat = valid[:, 0] & (ply[:, 0, 3] >= 1)
            ok.append((lo.argmax(1).cpu().numpy() == ply[:, 0, 3])[bat])
    kl = float(np.concatenate(kls).mean())
    nll = float(np.concatenate(nlls).mean())
    acc = float(np.concatenate(ok).mean())
    d0 = json.load(open(os.path.join(a.out, "eval", "d0.json")))
    res = {"kl0_full_per_hidden_slot": kl, "nll_7known": nll, "battle_acc_ply0": acc,
           "n_battles": int(np.concatenate(ok).size),
           "abs_diff": {"kl": abs(kl - d0["kl0_full_per_hidden_slot"]), "nll7": abs(nll - d0["nll_7known"]),
                        "battle": abs(acc - d0["battle_acc_ply0"])}}
    json.dump(res, open(os.path.join(a.out, "eval", "audit_vm.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
