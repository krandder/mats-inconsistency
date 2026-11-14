#!/usr/bin/env python3
"""EXPLORATORY (post hoc, written after the gate numbers were seen; NOT rule tests). Two diagnostics per seed (primary
architecture, saved checkpoints in /tmp/r5e2/ckpt, CPU):
 1. Rollout belief claims: on 200 calibration D0 positions, 16 unskewed cone rollouts (the gate-3b protocol): share of
    covered (rollout, hidden slot) claims that give the true type < 1e-3; mean per-claim KL(P*_full || mu_r) vs the
    0-step belief on the same slots; consensus KL with the pre-registered geometric (log-space) coverage pooling vs an
    arithmetic (probability-space) pooling of the same claims vs P_0.
 2. Keys: on 60 D1 test positions, 12 Left + 12 Right rollouts each: mean cosine within / across flank of the raw
    attention fingerprint ebar, of ebar minus its bag mean, and of the fitted W_K2 keys; Jaccard overlap of the cones.
Writes report/expl_keys_gibbs.json.
   python3 expl_keys_gibbs.py"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval6 as EV  # noqa: E402
import model6 as M  # noqa: E402
import train6 as T  # noqa: E402

CK = "/tmp/r5e2/ckpt"


def kl(p, q):
    return (np.where(p > 0, p * (np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(q, 1e-9, 1))), 0)).sum(-1)


@torch.no_grad()
def gibbs(Mo, Rc, dev, seed, n=200, K=16):
    o = np.load(os.path.join(T.DATA, "oracle_d0_cal.npz"))
    rows = EV.rows_for(o, EV.row_index(Rc), dev, n)
    gen = torch.Generator().manual_seed(seed + 7000)
    rt = T.make_roots(Mo.net, Rc, rows, dev)
    nl = rt["n_legal"]
    act = torch.minimum((torch.rand(n, K, generator=gen) * nl[:, None]).long(), (nl - 1)[:, None])
    rr = T.run_rollouts(Mo.net, Mo.strust, Mo.rtrust, rt, act, 4, gen, dev)
    P = o["pstar_full"][:n]
    hid = rt["hidden"].numpy()
    q0 = rt["l0b"].exp().numpy()
    mu = torch.softmax(rt["l0b"][:, None] + rr["dlb"], -1).numpy()
    cov = rr["cov"].numpy() & hid[:, None]
    geo = M.coverage_reference({"dlb": rr["dlb"], "cov": rr["cov"], "e": rr["e"], "valid": torch.ones(n, K, dtype=torch.bool),
                                "l0b": rt["l0b"]})[0].exp().numpy()
    c = cov[..., None].astype(float)
    nc = c.sum(1)
    ari = np.where(nc > 0, (mu * c).sum(1) / np.maximum(nc, 1), q0)
    m = lambda x: float((np.where(hid, x, 0).sum(1) / np.maximum(hid.sum(1), 1)).mean())
    tru = (Rc["slot_true"][rows].long() - 1).numpy()
    pt = np.take_along_axis(mu, np.broadcast_to(tru[:, None, :, None], mu.shape[:3] + (1,)), -1)[..., 0]
    per = kl(P[:, None], mu)
    return {"n_positions": n, "K": K, "share_claims_true_type_below_1e-3": float((pt[cov] < 1e-3).mean()),
            "mean_claim_kl": float(per[cov].mean()),
            "p0_kl_same_slots": float(np.broadcast_to(kl(P, q0)[:, None], per.shape)[cov].mean()),
            "kl_p0": m(kl(P, q0)), "kl_geometric_prereg": m(kl(P, geo)), "kl_arithmetic": m(kl(P, ari))}


@torch.no_grad()
def keys(Mo, R, dev, seed, n=60):
    d = np.load(os.path.join(T.DATA, "d1_pos.npz"))
    roots = EV.rows_for(d, EV.row_index(R), dev, n)
    gen = torch.Generator().manual_seed(seed + 8000)
    rt = T.make_roots(Mo.net, R, roots, dev)
    pool = T.flank_pool(Mo.net, Mo.strust, Mo.rtrust, rt, 12, gen, dev)
    eb = pool["ebar"].reshape(n, 24, -1)
    fl = torch.arange(2).repeat_interleave(12)
    same = (fl[:, None] == fl[None]) & ~torch.eye(24, dtype=torch.bool)
    diff = fl[:, None] != fl[None]

    def gap(x):
        x = F.normalize(x, dim=-1)
        s = x @ x.transpose(1, 2)
        return [float(s[:, same].mean()), float(s[:, diff].mean())]
    k = Mo.amp.key(eb.reshape(-1, eb.shape[-1])).view(n, 24, -1)
    cone = pool["cone"].reshape(n, 24, 3).numpy()
    jw, ja = [], []
    for i in range(n):
        S = [set(cc[cc >= 0]) for cc in cone[i]]
        for a in range(24):
            for b in range(a + 1, 24):
                j = len(S[a] & S[b]) / max(1, len(S[a] | S[b]))
                (jw if (a < 12) == (b < 12) else ja).append(j)
    return {"n_positions": n, "cos_within_across_raw_ebar": gap(eb), "cos_within_across_centred_ebar": gap(eb - eb.mean(1, keepdim=True)),
            "cos_within_across_W_K2_key": gap(k), "cone_jaccard_within_across": [float(np.mean(jw)), float(np.mean(ja))]}


def main():
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    Rc = T.load_split("cal", dev)
    R = T.load_split("test", dev)
    out = {}
    for seed in (101, 102, 103):
        Mo = EV.Models(os.path.join(CK, f"s{seed}", "primary"), "primary", dev)
        out[f"s{seed}"] = {"gibbs": gibbs(Mo, Rc, dev, seed), "keys": keys(Mo, R, dev, seed)}
        print(seed, json.dumps(out[f"s{seed}"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "report", "expl_keys_gibbs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
