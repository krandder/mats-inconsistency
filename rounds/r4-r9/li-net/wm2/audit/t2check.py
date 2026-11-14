"""wm2 audit: independent check of the T2 data (Opus 5.5 skeptic subagent for eurisko, 2026-10-11).

  LIR_HARNESS=/home/agent/workspaces/eurisko-study/lir-agent/harness PYTHONDONTWRITEBYTECODE=1 \
      nice python3 audit/t2check.py [--split val] [--n 3000]

Does not import t2data.py. From the harness (boards, placement list) and the rule in SPEC-REVIEW C.13 only:
  1. regenerates the T2 episodes (board = ShipHunt(hard, seed).occ; probes from default_rng([seed, 20261011])) and
     compares them with the cached file;
  2. recomputes the selection-aware and the selection-unaware exact posteriors by enumeration (own code: near-mask by an
     adjacency matrix, likelihoods in log space) and compares them with the cached `post` / `post_unaware`;
  3. recomputes the "ignore where probes land" reference KL(aware || unaware) per stage, the location-information truth
     (mean over episodes of log P(probe cells | own board) - log P(probe cells | next episode's board)), and the
     readings-vs-locations information split at 6 probes;
  4. checks the T1 cache of the same split has the same boards and checks basic properties of the T2 probes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

H = os.environ.get("LIR_HARNESS", "/home/agent/workspaces/eurisko-study/lir-agent/harness")
sys.path.insert(0, H)
from env.shiphunt import PRESETS, ShipHunt, placements  # noqa: E402

CFG = PRESETS["hard"]
W, NC, EPS, Q = 5, 25, 0.2, 0.7
BASE = {"val": 7_000_000, "test": 9_000_000}


def adjacency():
    A = np.eye(NC, dtype=np.int64)
    for i in range(NC):
        r, c = divmod(i, W)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < W and 0 <= cc < W:
                A[i, rr * W + cc] = 1
    return A


ADJ = adjacency()


def near(occ_bool):
    return (occ_bool.astype(np.int64) @ ADJ) > 0


def gen_episode(seed):
    occ = ShipHunt(CFG, int(seed)).occ.astype(bool)
    nm = near(occ[None])[0]
    rng = np.random.default_rng([int(seed), 20261011])
    unprobed = list(range(NC))
    pc, pr = [], []
    for t in range(6):
        A = [c for c in unprobed if nm[c]]
        u = rng.random()
        if u < Q and len(A) > 0:
            c = int(rng.choice(np.array(A)))
        else:
            c = int(rng.choice(np.array(unprobed)))
        flip = rng.random() < EPS
        pc.append(c)
        pr.append(int(bool(occ[c]) != flip))
        unprobed.remove(c)
    return occ.astype(np.uint8), np.array(pc), np.array(pr, np.uint8)


def sel_loglik(P_near, pcs):
    """log P(c_t | placement, c_<t) for each placement [K, 6] under the C.13 rule."""
    K = P_near.shape[0]
    out = np.zeros((K, 6))
    probed = np.zeros(NC, bool)
    for t, c in enumerate(pcs):
        U = ~probed
        nU = U.sum()
        nA = (P_near & U[None]).sum(1)
        p_near = np.where(nA > 0, Q * P_near[:, c] / np.maximum(nA, 1) + (1 - Q) / nU, 1.0 / nU)
        out[:, t] = np.log(p_near)
        probed[c] = True
    return out


def posteriors(P, P_near, pcs, prs, aware):
    rd = np.where(P[:, pcs] == prs.astype(bool)[None], np.log(1 - EPS), np.log(EPS))
    ll = rd + (sel_loglik(P_near, pcs) if aware else 0.0)
    marg = []
    for st in (0, 2, 4, 6):
        lw = ll[:, :st].sum(1)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        marg.append(w @ P.astype(np.float64))
    return np.array(marg)


def kl(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--data", default="/tmp/wm2")
    a = ap.parse_args()
    nfile = {"val": 3000, "test": 5000}[a.split]
    z = np.load(os.path.join(a.data, "data-t2", f"{a.split}-n{nfile}.post.npz"))
    z1 = np.load(os.path.join(a.data, "data-t1", f"{a.split}-n{nfile}.post.npz"))
    n = min(a.n, nfile)
    print(f"[T2 {a.split}] cached keys: {sorted(z.files)}; checking the first {n} episodes")
    P = placements(CFG).astype(bool)
    P_near = near(P)
    seeds = BASE[a.split] + np.arange(n)
    assert (z["seeds"][:n] == seeds).all(), "seed mismatch"
    # 1. episodes
    occ_ok = pc_ok = pr_ok = 0
    for i, s in enumerate(seeds):
        o, pc, pr = gen_episode(s)
        occ_ok += bool((o == z["occ"][i]).all())
        pc_ok += bool((pc == z["pc"][i]).all())
        pr_ok += bool((pr == z["pr"][i]).all())
    print(f"  1. regenerated episodes equal to the cache: boards {occ_ok}/{n}, probe cells {pc_ok}/{n}, readings {pr_ok}/{n}")
    print(f"     same boards as the T1 cache of this split: {int((z['occ'][:n] == z1['occ'][:n]).all(1).sum())}/{n}")
    distinct = np.mean([len(set(r)) == 6 for r in z["pc"][:n]])
    nm = near(z["occ"][:n].astype(bool))
    # share of probes that land in A (occupied-or-adjacent, unprobed) and the share expected under the rule
    land, expct = [], []
    for i in range(n):
        probed = np.zeros(NC, bool)
        for c in z["pc"][i]:
            A = nm[i] & ~probed
            nU = (~probed).sum()
            land.append(bool(A[c]))
            expct.append(Q + (1 - Q) * A.sum() / nU if A.any() else 0.0)
            probed[c] = True
    print(f"     probe cells distinct within an episode: {distinct:.3f}; share of probes in A: {np.mean(land):.4f} (rule's expectation {np.mean(expct):.4f}); "
          f"reading flip rate: {np.mean(z['pr'][:n] != np.take_along_axis(z['occ'][:n], z['pc'][:n], 1)):.4f} (0.2)")
    # 2. posteriors
    pa = np.zeros((n, 4, NC))
    pu = np.zeros((n, 4, NC))
    for i in range(n):
        pa[i] = posteriors(P, P_near, z["pc"][i], z["pr"][i], True)
        pu[i] = posteriors(P, P_near, z["pc"][i], z["pr"][i], False)
    print(f"  2. max |own aware posterior - cached post| = {np.abs(pa - z['post'][:n]).max():.3e}; "
          f"max |own unaware - cached post_unaware| = {np.abs(pu - z['post_unaware'][:n]).max():.3e}")
    if a.split == "val":
        print(f"     max |own unaware posterior on T2 data - T1 harness formula is the same function|: T1 cache post on T1 data is not comparable (different probes)")
    # 3. references
    k = kl(pa, pu)
    print(f"  3. KL(aware || unaware), the 'ignore where probes land' reference, x1e3: mean {1e3 * k.mean():.3f}; per stage "
          f"{[round(1e3 * x, 3) for x in k.mean((0, 2))]}")
    prior = P.mean(0)
    kp = kl(pu, prior[None, None])
    ka = kl(pa, prior[None, None])
    print(f"     KL(unaware || prior) per stage x1e3 {[round(1e3 * x, 2) for x in kp.mean((0, 2))]}; KL(aware || prior) {[round(1e3 * x, 2) for x in ka.mean((0, 2))]}")
    print(f"     at 6 probes: locations' extra KL(aware||unaware) / readings' KL(unaware||prior) = {k.mean((0, 2))[3] / kp.mean((0, 2))[3]:.3f}")
    y = z["occ"][:n][:, None, :].astype(float)
    def ll(q):
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return float((-(y * np.log(q) + (1 - y) * np.log(1 - q))).mean())
    print(f"     outcome log-loss per cell: aware posterior {ll(pa):.6f}, unaware {ll(pu):.6f} (gap {1e3 * (ll(pu) - ll(pa)):.3f}e-3), prior {ll(np.broadcast_to(prior, pa.shape)):.6f}")
    # location information truth: other board = next episode's board ((i + 1) mod n), as in wm2.WMVal
    occ = z["occ"][:n].astype(bool)
    occ_nm = near(occ)
    li = []
    for i in range(n):
        j = (i + 1) % n
        own = sel_loglik(occ_nm[i:i + 1], z["pc"][i])[0].sum()
        oth = sel_loglik(occ_nm[j:j + 1], z["pc"][i])[0].sum()
        li.append(own - oth)
    li = np.array(li)
    print(f"     location information truth (nats per episode, other board = next episode's): {li.mean():.4f} +- {1.96 * li.std(ddof=1) / np.sqrt(n):.4f}")


if __name__ == "__main__":
    main()
