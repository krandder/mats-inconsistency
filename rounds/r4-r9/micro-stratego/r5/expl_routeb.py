#!/usr/bin/env python3
"""EXPLORATORY (post hoc, written 2026-10-17 after the gate numbers were seen; NOT a rule test). For each hidden slot s
of the D0 positions: the window-only route-B marginal log mu_B[s] = log softmax(log q_real(Z_s | root at t0) + sum over
the opponent's real moves in the last 4 plies of log pi_real(move | Z_s = k, every other hidden slot MASK)), i.e. the
pre-registered 4-ply consistency check WITHOUT conditioning on sampled cone neighbours (no Gibbs sampling, no pooling).
Compares KL(P*_full || .) of mu_B, of the one-pass belief P_0 at t and of the one-pass belief at t0. CPU, saved
checkpoints in /tmp/r5e2/ckpt. Writes report/expl_routeb.json + .csv.
   python3 expl_routeb.py"""
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval6 as EV  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

CK = "/tmp/r5e2/ckpt"


@torch.no_grad()
def routeb(net, R, rows, dev, chunk=128):
    out_t, out_t0, out_b = [], [], []
    for c0 in range(0, rows.shape[0], chunk):
        ix = rows[c0:c0 + chunk]
        rt = T.make_roots(net, R, ix, dev)
        B = ix.shape[0]
        tpw = rt["tpw"]
        base = M.tape_clone(tpw)
        wply, nw = rt["wply"], rt["nw"]
        live = torch.arange(M.WIN)[None] < nw[:, None]
        for ci, k_ in enumerate(M.PLY_KEYS):
            base[k_] = torch.zeros(B, M.MAXP, dtype=torch.long)
            base[k_][:, :M.WIN] = torch.where(live, wply[..., ci], torch.zeros_like(wply[..., ci]))
        opp = (live & (wply[..., 0] == 1)).float()
        mv = (wply[..., 1] * M.NSQ + wply[..., 2]).clamp(0, M.NPOL - 1)
        h0, _ = net(tpw, root_only=True)
        lq0 = torch.log_softmax(net.belief_logits(h0).float(), -1)                    # [B,8,6] belief at t0
        ll = torch.zeros(B, M.NS, 6)
        hid = rt["hidden"]
        for s in range(M.NS):
            rows_s = torch.nonzero(hid[:, s]).squeeze(1)
            if rows_s.numel() == 0:
                continue
            for k in range(1, 7):
                tp = M.tape_index(base, rows_s)
                v = tp["slot_val"].clone()
                v[:, s] = k
                tp["slot_val"] = v
                h, _ = net(tp)
                g = torch.zeros(rows_s.shape[0])
                for j in range(M.WIN):
                    lp = torch.log_softmax(net.h_policy(h[:, M.pre_idx(j)]).float(), -1)
                    g += lp.gather(1, mv[rows_s, j:j + 1]).squeeze(1) * opp[rows_s, j]
                ll[rows_s, s, k - 1] = g
        lb = torch.log_softmax(lq0 + ll, -1)
        out_t.append(rt["l0b"].exp().numpy())
        out_t0.append(lq0.exp().numpy())
        out_b.append(lb.exp().numpy())
        if c0 == 0:
            hids = [hid.numpy()]
        else:
            hids.append(hid.numpy())
    return np.concatenate(out_t), np.concatenate(out_t0), np.concatenate(out_b), np.concatenate(hids)


def main():
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    res, rows_csv = {}, []
    for split in ("cal", "test"):
        R = T.load_split(split, dev)
        rix = EV.row_index(R)
        o = np.load(os.path.join(T.DATA, f"oracle_d0_{split}.npz"))
        n = 500
        rows = EV.rows_for(o, rix, dev, n)
        P = o["pstar_full"][:n]
        for seed in [int(x) for x in os.environ.get("SEEDS", "101 102 103").split()]:
            for arch in ("primary", "fallback"):
                d = os.path.join(CK, f"s{seed}", arch)
                if not os.path.exists(os.path.join(d, "real_100.pt")):
                    continue
                net = M.make_net(arch)
                net.load_state_dict(torch.load(os.path.join(d, "real_100.pt"), map_location=dev))
                net.eval()
                qt, qt0, qb, hid = routeb(net, R, rows, dev)
                f = lambda q: float((np.where(hid, EV.kl_rows(P, q), 0).sum(1) / np.maximum(hid.sum(1), 1)).mean())
                r = {"kl_p0_t": f(qt), "kl_p0_t0": f(qt0), "kl_routeb": f(qb), "n": int(n)}
                r["routeb_reduction_vs_p0"] = 1 - r["kl_routeb"] / r["kl_p0_t"]
                res[f"{split}/s{seed}/{arch}"] = r
                rows_csv.append([split, seed, arch, r["kl_p0_t"], r["kl_p0_t0"], r["kl_routeb"], r["routeb_reduction_vs_p0"]])
                print(split, seed, arch, json.dumps(r), flush=True)
    os.makedirs(os.path.join(HERE, "report"), exist_ok=True)
    tag = os.environ.get("TAG", "")
    json.dump(res, open(os.path.join(HERE, "report", f"expl_routeb{tag}.json"), "w"), indent=1)
    with open(os.path.join(HERE, "report", f"expl_routeb{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "seed", "arch", "kl_P0_at_t", "kl_P0_at_t0", "kl_routeB_window_only", "reduction_vs_P0"])
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
