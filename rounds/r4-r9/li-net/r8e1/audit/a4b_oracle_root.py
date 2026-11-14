#!/usr/bin/env python3
"""R8-E1 audit, checks 2-4 support: rebuild the 500 test boards (stream 22) with the auditor's own generator, recompute exact Bayes
(all 36 readings + note; and the 6 standard readings + note) with the auditor's own enumeration, and run the R7-E1 checkpoints on CPU
(read in place, frozen) to recompute the roots l0(M_0), root KL by M_0 and the clause-1 Pearson.
Imports: numpy, torch, and li-net/r5e1/wm.py (the WorldModel class only; no e1v3/e1v4/e1v5/analyze8).
Run: OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r4e3a/pylib python3 a4b_oracle_root.py"""
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

LI = "/home/agent/workspaces/eurisko-study/li-net"
R = f"{LI}/r8e1/results/main"
OUT = os.path.dirname(os.path.abspath(__file__))
EPS, NOTE_REL, NSHIP = 0.15, 0.30, 9


def placements():
    single = []
    for L in (4, 3, 2):
        opts = set()
        for r in range(6):
            for c in range(6):
                if c + L <= 6:
                    opts.add(frozenset(r * 6 + c + k for k in range(L)))
                if r + L <= 6:
                    opts.add(frozenset((r + k) * 6 + c for k in range(L)))
        single.append(sorted(opts, key=sorted))
    out = set()
    for combo in itertools.product(*single):
        cells, ok = set(), True
        for s in combo:
            if cells & s:
                ok = False
                break
            cells |= s
        if ok:
            out.add(frozenset(cells))
    arr = np.zeros((len(out), 36), bool)
    for i, cells in enumerate(sorted(out, key=sorted)):
        arr[i, list(cells)] = True
    return arr


def boards(stream, n, P):
    rng = np.random.default_rng([stream, 7702])
    occ = P[rng.integers(len(P), size=n)].copy()
    s = occ ^ (rng.random((n, 36)) < EPS)
    true_note = rng.random(n) < NOTE_REL
    sc = rng.random((n, 36))
    note = np.where(true_note, np.where(occ, sc, -1.0).argmax(1), np.where(~occ, sc, -1.0).argmax(1))
    order = rng.random((n, 36)).argsort(1)
    std6 = rng.random((n, 36)).argsort(1)[:, :6]
    return occ, s, note, order, std6


def exact_bayes(s, mask, note, P):
    """P*(ship on cell | readings s on mask, note): enumeration over all placements (float64)."""
    Pf = P.astype(np.float64)
    out = np.zeros((s.shape[0], 36))
    lo, lc = math.log(1 - EPS), math.log(EPS)
    for i in range(s.shape[0]):
        m = mask[i]
        agree = (P[:, m] == s[i, m][None, :])
        lw = np.where(agree, lo, lc).sum(1)
        lw += np.where(P[:, note[i]], math.log(NOTE_REL / NSHIP), math.log((1 - NOTE_REL) / (36 - NSHIP)))
        w = np.exp(lw - lw.max())
        out[i] = (w / w.sum()) @ Pf
    return out


def bkl(p, q):
    p = np.clip(p, 0, 1)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.log(q)), 0.0)
        b = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1)) - np.log1p(-q)), 0.0)
    return a + b


P = placements()
n = 500
occ, s, note, order, std6 = boards(22, n, P)
rep = {"n_placements": int(len(P))}
full = np.ones((n, 36), bool)
pstar = exact_bayes(s, full, note, P)
m6 = np.zeros((n, 36), bool)
np.put_along_axis(m6, std6, True, 1)
pstd6 = exact_bayes(s, m6, note, P)

import torch  # noqa: E402

torch.set_num_threads(2)
sys.path.insert(0, f"{LI}/r5e1")
os.environ.setdefault("SMALLNET_DIR", f"{LI}/smallnet")
import wm  # noqa: E402  (model class only)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ev_of(mask):
    return torch.as_tensor(np.where(mask, 1 + s.astype(np.int64), 0))


per = {}
for seed in (101, 102, 103):
    raw = json.load(open(f"{R}/s{seed}/raw.json"))
    T = json.load(open(f"{R}/s{seed}/summary.json"))["test"]
    r = per.setdefault(seed, {})
    r["occ_matches_raw"] = bool((np.asarray(raw["occ"], bool) == occ).all())
    r["anchor_order18_matches_own_generator"] = bool((np.asarray(raw["anchor_order18"]) == order[:, :18]).all())
    rp = np.asarray(raw["pstar"])
    r["pstar_max_abs_diff_vs_own_exact_bayes"] = float(np.abs(rp - pstar).max())
    npz = f"{LI}/r7e1/ckpt/s{seed}/theta_real_v2.npz"
    meta = json.load(open(npz[:-4] + ".json"))
    r["ckpt_sha256_matches_json"] = sha(npz) == meta["sha256"]
    a = meta["arch"]
    model = wm.WorldModel(a["d"], a["layers"], a["heads"], a["ff"])
    with np.load(npz) as z:
        model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files}, strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    nt = torch.as_tensor(note)
    z0 = torch.zeros(n, 36, dtype=torch.bool)
    with torch.no_grad():
        f = lambda ev: torch.sigmoid(model(ev, nt, z0, z0.long())[0]).double().numpy()
        roots = {}
        for M in (2, 6, 12, 18):
            mk = np.zeros((n, 36), bool)
            np.put_along_axis(mk, order[:, :M], True, 1)
            roots[M] = f(ev_of(mk))
            r[f"root_mu_M{M}_max_abs_diff_cpu_vs_raw"] = float(np.abs(roots[M] - np.asarray(raw["root_mu"][str(M)])).max())
            r[f"kl_root_M{M}_cpu"] = float(bkl(pstar, roots[M]).mean())
            r[f"kl_root_M{M}_summary"] = T["kl_root"][str(M)]
        lstd = f(ev_of(m6))
        r["clause1_pearson_cpu"] = float(np.corrcoef(lstd.ravel(), pstd6.ravel())[0, 1])
        r["clause1_pearson_summary"] = T["clause1_corr_std6"]
        r["clause1_kl_cpu"] = float(bkl(pstd6, lstd).mean())
        r["clause1_kl_summary"] = T["clause1_kl_std6"]
        r["kl_prior_cpu"] = float(bkl(pstar, f(torch.zeros(n, 36, dtype=torch.long))).mean())
        r["kl_prior_summary"] = T["kl_prior"]
    del raw
rep["per_seed"] = per
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(OUT, "a4b_oracle_root.json"), "w"), indent=1)
