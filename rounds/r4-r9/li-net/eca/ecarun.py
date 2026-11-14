"""E-CA: one training run + its full evaluation (eurisko, 2026-10-15, design v2). One call = one (rule, training arm, seed).

  python3 ecarun.py --rule 110 --arm T1 --seed 0 --out DIR [--smoke]

Training arms: T1 plain masked any-order loss; T2 reveal-chain TD with 8 levels (rbtd8, smallnet's rbtd with lam = 0).
Training sees grids and its own random reveal masks only (reveal count k ~ U{1..N-1}); early stopping on VALIDATION
OUTCOME log loss (true cells), never the posterior. After training the backbone is frozen and the stiffness and key
heads are fitted on a held-out calibration split (outcomes only). Then: P0 / D0 gate, stiffness D1, sequential
materialisation chains (P2-Stiff, P2-VoI, P3-Conf, P4-Rand; K chains, M in {0,1,2,4,8,16}), pools fitted on validation
outcomes (prior-corrected learned-w, Gram dedup P5, and the P6 aggregators), dedup tests D3a/D3b. Exact posteriors are
read only inside the scoring functions (kl_*, d1_*) and the D0 gate.
2026-10-15 02:40Z fix (eca2): every probability, pool and metric is computed in float64 (the 01:35Z version pooled in float32,
so confident pools rounded to exactly 1.0: infinite KL / NaN log loss for lw/dd at small M, the g1 product and D3b on B2).
2026-10-15 04:3xZ round 2 (PREREG-R2.md): --layers 1 (R2a), --key-mode v2 (R2b key without h_i), --tau, --selectors,
selector "voiu" (VoI over the 6 highest-variance cells), pool "sw" (normalised stiffness-weighted mixture anchored at P0),
attention-vector cosines in D3a/D3b, model.pt saved. Round-1 behaviour is unchanged with the defaults.
Writes DIR/result.json (summary), DIR/test-estimates.csv (per-instance pooled logits for every arm/pool/M, with the exact
posterior and the outcome; the audit re-derives headline numbers from it), DIR/train-history.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402

# round 3 (2026-10-15 09:3xZ): --data-module ecadata3 switches to the W in {14, 16}, T = 10 data with disjoint row-0 splits
# and the bit-factored exact inference; the default keeps rounds 1-2 unchanged.
_DM = sys.argv[sys.argv.index("--data-module") + 1] if "--data-module" in sys.argv else "ecadata"
D = importlib.import_module(_DM)
from ecanet import LN2, Net  # noqa: E402

MS = (0, 1, 2, 4, 8, 16)
SELECTORS = ("stiff", "voi", "conf", "rand")      # round 2 adds "voiu" via --selectors
CHAIN_FAMS = ("F", "B1", "B2", "R")
POOLS = ("lw", "dd", "mix", "g0", "g1", "sw")
SW_ETAS = (0.0, 0.5, 1.0, 2.0)
SW_LAMS = (0.0,) + tuple(float(x) for x in np.logspace(-3, 5, 33))
TAU = 1.0          # P2-Stiff: sample z proportional to exp(TAU * beta_int) across chains (round 2: --tau 5)
CFG = {"tau": TAU}
VOI_TOP = 6        # P2-VoI: candidates = top-6 masked cells by beta_int


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y, reduction="none")


def rand_reveal(B, N, gen, dev, kmin=1):
    k = torch.randint(kmin, N, (B, 1), device=dev, generator=gen)
    pos = torch.argsort(torch.argsort(torch.rand(B, N, device=dev, generator=gen), 1), 1)
    return pos < k, pos, k


def kl_bern(p, q):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12)
    q = np.clip(np.asarray(q, np.float64), 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def logloss(q, y):
    q = np.clip(np.asarray(q, np.float64), 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def ece(q, y, bins=15):
    q = np.asarray(q, np.float64)
    b = np.minimum((q * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(q[m].mean() - y[m].mean())
    return float(tot)


def rankdata(a):
    a = np.asarray(a, np.float64)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sa = a[o]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def auc(score, label):
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_np(p):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def sig_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, np.float64), -40, 40)))


# ------------------------------------------------------------------------------------------------ batched inference
class Infer:
    def __init__(self, net, dev, chunk=None):
        # round 3: attention memory grows with n^2, so the chunk shrinks for long grids (8192 sequences at n = 48)
        chunk = chunk or max(256, int(8192 * (48.0 / net.n) ** 2))
        self.net, self.dev, self.chunk = net, dev, chunk
        self.passes = 0

    @torch.no_grad()
    def __call__(self, rev, val, need_key=False, rev_base=None, qidx=None):
        """rev bool [B,n], val [B,n] -> logits [B,n], s_hat [B,n] (, keys [B,n,k]) (, attention vector of token qidx [B,n]).
        rev_base: evidence set E_0 (round-2 v2 keys attend over S minus E_0 when non-empty). qidx: query token per row."""
        outs_l, outs_s, outs_k, outs_a = [], [], [], []
        for a in range(0, rev.shape[0], self.chunk):
            r, v = rev[a:a + self.chunk], val[a:a + self.chunk]
            rb = rev_base[a:a + self.chunk] if rev_base is not None else None
            h, attn = self.net.backbone(self.net.tokens(r, v), need_attn=need_key)
            outs_l.append(self.net.logits(h).float())
            outs_s.append(self.net.surprise(h).float())
            if need_key:
                outs_k.append(self.net.key(h, attn, r, rb).float())
                if qidx is not None:
                    q_ = qidx[a:a + self.chunk]
                    av = self.net.attn_vec(attn, r, rb)
                    outs_a.append(av[torch.arange(r.shape[0], device=r.device), q_].float())
        self.passes += rev.shape[0]
        lg, s = torch.cat(outs_l), torch.cat(outs_s)
        if not need_key:
            return lg, s
        if qidx is not None:
            return lg, s, torch.cat(outs_k), torch.cat(outs_a)
        return lg, s, torch.cat(outs_k)


def run_chains(inf, x, rev0, yq, selector, K, Mmax, gen, Ms=MS, order=None, want_av=False):
    """Sequential materialisation (design v2 change 1). x, rev0: [n,N] token order (x = true values; only revealed
    entries are read), yq [n] query token. K chains per instance; S_t = S_{t-1} U {z_t}, z_t chosen by `selector` from one
    pass on S_{t-1}, materialised by sampling z_t ~ q(z_t | S_{t-1}). Returns claims at each M in Ms:
    q [n,K,len(Ms)], s_hat [n,K,len(Ms)], key [n,K,len(Ms),kd], and forward passes per chain.
    order: optional [n, Mmax] forced materialisation order (token ids, -1 = stop), used only by the D3b test."""
    dev = x.device
    n, N = x.shape
    R = n * K
    rev = rev0.repeat_interleave(K, 0).clone()
    val = torch.where(rev0, x.long(), torch.zeros_like(x, dtype=torch.long)).repeat_interleave(K, 0).clone()
    yy = yq.repeat_interleave(K, 0)
    rows = torch.arange(R, device=dev)
    isY = F.one_hot(yy, N).bool()
    if order is not None:
        order = order.repeat_interleave(K, 0)
    rbase = rev0.repeat_interleave(K, 0)
    qs, ss, ks, avs = [], [], [], []
    p0 = inf.passes
    run_chains.passes_at = {}
    for t in range(Mmax + 1):
        rec = t in Ms
        if rec:
            lg, s, kk, av = inf(rev, val, need_key=True, rev_base=rbase, qidx=yy)
            qs.append(torch.sigmoid(lg[rows, yy].double()))
            ss.append(s[rows, yy])
            ks.append(kk[rows, yy])
            if want_av:
                avs.append(av)
            run_chains.passes_at[t] = (inf.passes - p0) / R
        else:
            lg, s = inf(rev, val)
        if t == Mmax:
            break
        q = torch.sigmoid(lg)
        beta = Net.beta_from_surprise(s)
        cand = (~rev) & (~isY)
        has = cand.any(1)
        if order is not None:
            z = order[:, t].clamp(min=0)
            has = has & (order[:, t] >= 0)
        elif selector == "stiff":
            logits_sel = torch.where(cand, CFG["tau"] * beta, torch.full_like(beta, -1e9))
            z = torch.multinomial(torch.softmax(logits_sel, 1), 1, generator=gen).squeeze(1)
        elif selector == "rand":
            w = cand.float() + (~has).float()[:, None]
            z = torch.multinomial(w, 1, generator=gen).squeeze(1)
        elif selector == "conf":
            sc = torch.where(cand, (q - 0.5).abs() + 1e-6 * torch.rand(q.shape, device=dev, generator=gen),
                             torch.full_like(q, -1e9))
            z = sc.argmax(1)
        elif selector == "hyb":              # round 3 P6-Hybrid: Propagate-Stiff-then-Branch
            qc = q.clamp(1e-7, 1 - 1e-7)
            Hq = -(qc * qc.log() + (1 - qc) * (1 - qc).log())
            Hm = torch.where(cand, Hq + 1e-9 * torch.rand(Hq.shape, device=dev, generator=gen), torch.full_like(Hq, 1e9))
            hmin, zst = Hm.min(1)
            z = zst.clone()
            br = (hmin >= HYB_H) & has                       # no stiff frontier cell left: branch with VoI-Unc
            if br.any():
                ridx = rows[br]
                z[ridx] = _voi_pick(inf, q[ridx], cand[ridx], rev[ridx], val[ridx], yy[ridx], gen, dev, N, gate="unc")
            HYB_STATS["steps"] += int(has.sum())
            HYB_STATS["branch"] += int(br.sum())
        elif selector in ("voi", "voiu"):
            z = _voi_pick(inf, q, cand, rev, val, yy, gen, dev, N, gate=("stiff" if selector == "voi" else "unc"), beta=beta)
        else:
            raise ValueError(selector)
        zq = q[rows, z]
        zv = (torch.rand(R, device=dev, generator=gen) < zq).long()
        upd = has & (~rev[rows, z]) & (z != yy)
        rev[rows[upd], z[upd]] = True
        val[rows[upd], z[upd]] = zv[upd]
    passes_per_chain = (inf.passes - p0) / R
    q = torch.stack(qs, 1).view(n, K, len(Ms)).double()
    s = torch.stack(ss, 1).view(n, K, len(Ms)).double()
    k = torch.stack(ks, 1).view(n, K, len(Ms), -1).double()
    if want_av:
        av = torch.stack(avs, 1).view(n, K, len(Ms), -1).double().cpu().numpy()
        return q.cpu().numpy(), s.cpu().numpy(), k.cpu().numpy(), passes_per_chain, av
    return q.cpu().numpy(), s.cpu().numpy(), k.cpu().numpy(), passes_per_chain


HYB_H = 0.20          # round 3: entropy threshold (nats) for a "stiff frontier" cell in P6-Hybrid
HYB_STATS = {"steps": 0, "branch": 0}


def _voi_pick(inf, q, cand, rev, val, yy, gen, dev, N, gate="stiff", beta=None):
    """VoI among the 6 candidate cells with the highest beta_int (gate 'stiff', round 1) or the highest q(1 - q) (gate 'unc',
    round 2 voiu): pick argmax_z E_z[(q(Y | S, z) - q(Y | S))^2] from two extra passes per candidate. Returns z [R]."""
    R = q.shape[0]
    rows = torch.arange(R, device=dev)
    if gate == "stiff":
        sb = torch.where(cand, beta + 1e-6 * torch.rand(beta.shape, device=dev, generator=gen), torch.full_like(beta, -1e9))
    else:
        sb = torch.where(cand, q * (1.0 - q) + 1e-4 * torch.rand(q.shape, device=dev, generator=gen), torch.full_like(q, -1e9))
    top = sb.topk(VOI_TOP, 1)
    cidx, cval = top.indices, top.values > -1e8
    qY = q[rows, yy]
    voc = torch.full((R, VOI_TOP), -1.0, device=dev)
    for j0 in range(0, R, 2048):
        j1 = min(R, j0 + 2048)
        rr = j1 - j0
        c = cidx[j0:j1]
        rv = rev[j0:j1, None, None, :].expand(rr, VOI_TOP, 2, N).clone()
        vv = val[j0:j1, None, None, :].expand(rr, VOI_TOP, 2, N).clone()
        oh = F.one_hot(c, N).bool()[:, :, None, :].expand(rr, VOI_TOP, 2, N)
        rv = rv | oh
        vv = torch.where(oh, torch.tensor([0, 1], device=dev)[None, None, :, None].expand(rr, VOI_TOP, 2, N), vv)
        lgc, _ = inf(rv.reshape(-1, N), vv.reshape(-1, N))
        qYc = torch.sigmoid(lgc.view(rr, VOI_TOP, 2, N).gather(3, yy[j0:j1, None, None, None].expand(rr, VOI_TOP, 2, 1)).squeeze(3))
        qz1 = q[j0:j1].gather(1, c)
        v = qz1 * (qYc[:, :, 1] - qY[j0:j1, None]) ** 2 + (1 - qz1) * (qYc[:, :, 0] - qY[j0:j1, None]) ** 2
        voc[j0:j1] = torch.where(cval[j0:j1], v, torch.full_like(v, -1.0))
    return cidx[rows, voc.argmax(1)]


def beta_np(s):
    s = np.asarray(s, np.float64)
    return np.maximum(0.0, 1.0 / (s + 0.01) - 1.0 / (LN2 + 0.01))


def pool_feature(q, s, k, p0, kind):
    """q, s [n,K], k [n,K,kd], p0 [n] -> (feature f such that logit mu = logit p0 + u f) for lw / dd pools."""
    sh = logit_np(q) - logit_np(p0)[:, None]
    b = beta_np(s)
    if kind == "lw":
        return (b * sh).sum(1)
    if kind == "dd":
        G = np.einsum("nad,nbd->nab", k, k)
        crowd = np.maximum(G, 0).sum(2)                         # includes <k_a, k_a> = 1
        return (b * sh / crowd).sum(1)
    raise ValueError(kind)


def fit_u(f, base, y, iters=60):
    """Fit u >= 0 in logit mu = base + u f by Newton on the outcome log loss (closed-form gradient
    dL/du = -sum (y - mu) f, i.e. STUDY-LIR 3c's -(l_a(y) - E_mu l_a) summed through w_a = beta_a u)."""
    u = 0.0
    for _ in range(iters):
        mu = sig_np(base + u * f)
        g = -((y - mu) * f).mean()
        h = (mu * (1 - mu) * f * f).mean() + 1e-9
        un = max(0.0, u - g / h)
        if abs(un - u) < 1e-10:
            u = un
            break
        u = un
    return float(u)


def sw_pool(q, s, P0, lam, eta):
    """Round 2 normalised stiffness-weighted mixture anchored at the one-pass claim P0 = q(Y | E):
    mu = (lam P0 + sum_r w_r q_r) / (lam + sum_r w_r), w_r = beta_int(Y | S_M^r)^eta. At M = 0 every q_r = P0, so mu = P0."""
    q, s, P0 = (np.asarray(v, np.float64) for v in (q, s, P0))
    w = beta_np(s) ** eta if eta > 0 else np.ones_like(q)
    return (lam * P0 + (w * q).sum(1) + 1e-12 * P0) / (lam + w.sum(1) + 1e-12)


def fit_sw(q, s, P0, y):
    """Grid fit of (lam, eta) on validation outcome log loss (lam in {0} U logspace(-3, 5, 33), eta in {0, .5, 1, 2})."""
    best = (float("inf"), 0.0, 0.0)
    for eta in SW_ETAS:
        for lam in SW_LAMS:
            l = float(logloss(sw_pool(q, s, P0, lam, eta), y).mean())
            if l < best[0] - 1e-12:
                best = (l, lam, eta)
    return {"lam": best[1], "eta": best[2], "val_logloss": best[0]}


def pooled(q, s, k, p0, kind, u=None):
    """Pooled probability [n] for the P6 aggregators and the learned pools."""
    q, s, k, p0 = (np.asarray(v, np.float64) for v in (q, s, k, p0))
    if kind == "sw":
        return sw_pool(q, s, u["P0"], u["lam"], u["eta"])
    if kind == "mix":
        return q.mean(1)
    if kind == "g0":
        return sig_np(logit_np(q).mean(1))
    if kind == "g1":
        return sig_np(logit_np(p0) + (logit_np(q) - logit_np(p0)[:, None]).sum(1))
    return sig_np(logit_np(p0) + u * pool_feature(q, s, k, p0, kind))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", type=int, required=True)
    ap.add_argument("--arm", choices=("T1", "T2"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=D.T_DEF)
    ap.add_argument("--W", type=int, default=D.W_DEF)
    ap.add_argument("--eps", type=float, default=D.EPS_DEF)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--B2", type=int, default=64, help="grids per step for T2 (x 8 levels)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=60_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=100)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--stiff-steps", type=int, default=3000)
    ap.add_argument("--key-steps", type=int, default=2000)
    ap.add_argument("--n-val-inst", type=int, default=1000)
    ap.add_argument("--n-test-inst", type=int, default=1000)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--key-mode", choices=("v1", "v2"), default="v1", help="v2 = round-2 key without h_i (R2b)")
    ap.add_argument("--tau", type=float, default=1.0, help="P2-Stiff sampling temperature on beta_int (round 2: 5)")
    ap.add_argument("--selectors", default="stiff,voi,conf,rand", help="comma list; round 2 adds voiu")
    ap.add_argument("--save-model", type=int, default=0, help="1 = save model.pt (round 2)")
    ap.add_argument("--data-module", default="ecadata", help="ecadata (rounds 1-2) or ecadata3 (round 3)")
    ap.add_argument("--n-eval", type=int, default=3000, help="validation / test grids per family (round 1-2: 3000)")
    ap.add_argument("--n-cal", type=int, default=5000, help="calibration grids for the post-hoc heads (round 3: 4000)")
    ap.add_argument("--chain-fams", default="F,B1,B2,R", help="families searched by the chains (round 3: F,B1,B2)")
    ap.add_argument("--Mmax", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.smoke:
        a.n_train, a.max_steps, a.eval_every, a.stiff_steps, a.key_steps = 2000, 60, 20, 30, 30
        a.n_val_inst, a.n_test_inst, a.n_pairs, a.Mmax = 40, 40, 100, 4
    Ms = tuple(m for m in MS if m <= a.Mmax)
    CFG["tau"] = a.tau
    SEL = tuple(x for x in a.selectors.split(",") if x)
    global CHAIN_FAMS
    CHAIN_FAMS = tuple(x for x in a.chain_fams.split(",") if x)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else (a.device if a.device != "auto" else "cpu"))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    T, W, eps, rule = a.T, a.W, a.eps, a.rule
    N = T * W
    PERM = D.make_perm(T, W)
    INV = np.argsort(PERM)
    tok = lambda arr: arr[:, PERM]                                       # grid order -> token order
    torch.manual_seed(1000 + a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(2000 + a.seed)
    t_start = time.time()
    res = {"rule": rule, "arm": a.arm, "seed": a.seed, "T": T, "W": W, "eps": eps, "args": vars(a), "perm": PERM.tolist(),
           "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}

    # ---------------- data (token order only beyond this point, except inside the scoring code)
    Xtr = torch.tensor(tok(D.load_grids("train", rule, 50_000, T, W, eps))[: a.n_train], dtype=torch.long, device=dev)
    Xcal = torch.tensor(tok(D.load_grids("cal", rule, a.n_cal, T, W, eps)), dtype=torch.long, device=dev)
    xes_g, res_g = D.es_instances(rule, a.n_eval, T, W, eps)
    Xes = torch.tensor(tok(xes_g), dtype=torch.long, device=dev)
    Res = torch.tensor(tok(res_g), dtype=torch.bool, device=dev)
    VI = D.eval_instances("val", rule, a.n_eval, T, W, eps)
    TI = D.eval_instances("test", rule, a.n_eval, T, W, eps)

    net = Net(n=N, layers=a.layers).to(dev)
    net.key_mode = a.key_mode
    res["params"] = sum(p.numel() for p in net.parameters())
    fpass = net.flops_forward()
    res["flops_per_forward"] = fpass

    # ---------------- training
    ema = copy.deepcopy(net)
    ema.eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    trainable = [p for n_, p in net.named_parameters() if not (n_.startswith("stiff") or n_.startswith("keyW"))]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.05, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1.0, (s_ + 1) / 200))

    @torch.no_grad()
    def es_logloss(model):
        model.eval()
        tot, cnt = 0.0, 0.0
        for i0 in range(0, Xes.shape[0], 1024):
            xb, rb = Xes[i0:i0 + 1024], Res[i0:i0 + 1024]
            h, _ = model.backbone(model.tokens(rb, xb))
            l = bce(model.logits(h).float(), xb.float())
            hid = (~rb).float()
            tot += float(((l * hid).sum(1) / hid.sum(1)).sum())
            cnt += xb.shape[0]
        return tot / cnt

    hist, best, best_state, best_step, bad = [], float("inf"), None, 0, 0
    seqs = 0
    stop_reason = "max_steps"
    t0 = time.time()
    step = 0
    for step in range(1, a.max_steps + 1):
        net.train()
        if a.arm == "T1":
            idx = torch.randint(0, Xtr.shape[0], (a.B,), device=dev, generator=gen)
            xb = Xtr[idx]
            rv, _, _ = rand_reveal(a.B, N, gen, dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(a.amp) and dev.type == "cuda"):
                h, _ = net.backbone(net.tokens(rv, xb))
                lg = net.logits(h).float()
            loss = (bce(lg, xb.float()) * (~rv)).sum(1).mean()
            seqs += a.B
        else:                                                              # T2: rbtd8 (lam = 0)
            Bq = a.B2
            m = 8
            idx = torch.randint(0, Xtr.shape[0], (Bq,), device=dev, generator=gen)
            xb = Xtr[idx]
            xf = xb.float()
            _, pos, k0 = rand_reveal(Bq, N, gen, dev)
            u = torch.sort(torch.rand(Bq, m - 1, device=dev, generator=gen), 1).values
            lev = torch.cat([k0, k0 + torch.floor(u * (N - k0).float()).long()], 1)       # [Bq, 8], non-decreasing, <= N-1
            revs = torch.stack([pos < lev[:, i:i + 1] for i in range(m)], 0)                # [8, Bq, N]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(a.amp) and dev.type == "cuda"):
                h, _ = net.backbone(net.tokens(revs.reshape(m * Bq, N), xb.repeat(m, 1)))
                lgs = net.logits(h).float().view(m, Bq, N)
            loss = torch.zeros((), device=dev)
            y = xf
            for i in range(m - 1, -1, -1):
                hidden = ~revs[i]
                loss = loss + (bce(lgs[i], y) * hidden).sum(1).mean()
                if i > 0:
                    y = torch.where(hidden, torch.sigmoid(lgs[i]).detach(), xf)
            seqs += m * Bq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(a.ema).add_(pn.detach(), alpha=1 - a.ema)
        if step % a.eval_every == 0 or step == a.max_steps:
            vl = es_logloss(ema)
            rec = {"step": step, "val_outcome_logloss": vl, "train_loss": float(loss.detach()), "seqs": seqs,
                   "sec": time.time() - t0}
            hist.append(rec)
            if vl < best - 1e-6:
                best, best_step, bad = vl, step, 0
                best_state = copy.deepcopy(ema.state_dict())
            else:
                bad += 1
            log(json.dumps({k_: (round(v_, 5) if isinstance(v_, float) else v_) for k_, v_ in rec.items()}))
            if bad >= a.patience:
                stop_reason = "patience"
                break
            if (time.time() - t0) > 60 * a.max_minutes:
                stop_reason = "max_minutes"
                break
    train_sec = time.time() - t0
    model = ema
    model.load_state_dict(best_state)
    model.eval()
    res.update({"steps_run": step, "best_step": best_step, "best_val_outcome_logloss": best, "stop_reason": stop_reason,
                "train_sec": train_sec, "train_seqs": seqs, "train_flops_approx": 3.0 * seqs * fpass})
    with open(os.path.join(a.out, "train-history.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w_.writeheader()
        w_.writerows(hist)
    log("trained", a.arm, rule, a.seed, "best_step", best_step, "val_ll", round(best, 5), stop_reason, f"{train_sec:.0f}s")

    # ---------------- post-hoc heads on the held-out calibration split (backbone frozen, outcomes only)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    t1 = time.time()
    sp = list(model.stiff.parameters())
    for p_ in sp:
        p_.requires_grad_(True)
    opt_s = torch.optim.Adam(sp, lr=1e-3)
    sl_hist = []
    for st in range(a.stiff_steps):
        idx = torch.randint(0, Xcal.shape[0], (256,), device=dev, generator=gen)
        xb = Xcal[idx]
        rv, _, _ = rand_reveal(256, N, gen, dev)
        with torch.no_grad():
            h, _ = model.backbone(model.tokens(rv, xb))
            surpr = bce(model.logits(h).float(), xb.float())                # realised held-out surprise -log q(x_i | x_S)
        pred = model.surprise(h)
        hid = (~rv).float()
        l = (((pred - surpr) ** 2) * hid).sum() / hid.sum()
        opt_s.zero_grad(set_to_none=True)
        l.backward()
        opt_s.step()
        if st % 500 == 0:
            sl_hist.append(float(l.detach()))
    for p_ in sp:
        p_.requires_grad_(False)

    @torch.no_grad()
    def prior_q():
        h, _ = model.backbone(torch.zeros(1, N, dtype=torch.long, device=dev))
        return torch.sigmoid(model.logits(h).float())[0]                  # network's own q(x_i | nothing revealed)
    q_prior = prior_q()

    kp = list((model.keyW2 if a.key_mode == "v2" else model.keyW).parameters())
    for p_ in kp:
        p_.requires_grad_(True)
    log_u = torch.zeros((), device=dev, requires_grad=True)
    opt_k = torch.optim.Adam(kp + [log_u], lr=3e-3)
    kl_hist = []
    for st in range(a.key_steps):
        Bk = 128
        m = int(torch.randint(2, 6, (1,), generator=gen, device=dev))
        idx = torch.randint(0, Xcal.shape[0], (Bk,), device=dev, generator=gen)
        xb = Xcal[idx]
        tgt = torch.randint(0, N, (Bk,), device=dev, generator=gen)
        oh_t = F.one_hot(tgt, N).bool()
        half = Bk // 2
        views = []
        core, _, _ = rand_reveal(half, N, gen, dev)                     # shared-core bags (redundant views)
        core = core & ~oh_t[:half]
        part, _, _ = rand_reveal(Bk - half, N, gen, dev, kmin=2)        # disjoint bags (complementary views)
        part = part & ~oh_t[half:]
        lab = torch.randint(0, m, (Bk - half, N), device=dev, generator=gen)
        for j in range(m):
            extra = torch.rand(half, N, device=dev, generator=gen) < (3.0 / N)
            v1 = (core | extra) & ~oh_t[:half]
            v2 = part & (lab == j)
            views.append(torch.cat([v1, v2], 0))
        rv = torch.stack(views, 1).reshape(Bk * m, N)
        xr = xb.repeat_interleave(m, 0)
        tr_ = tgt.repeat_interleave(m, 0)
        with torch.no_grad():
            h, attn = model.backbone(model.tokens(rv, xr), need_attn=True)
            ql = model.logits(h).float()[torch.arange(Bk * m, device=dev), tr_]
            bt = Net.beta_from_surprise(model.surprise(h)[torch.arange(Bk * m, device=dev), tr_])
            ar = torch.arange(Bk * m, device=dev)
            h_t, attn_t = h[ar, tr_], attn[ar, tr_]
        a_ = attn_t * rv.float()
        if a.key_mode == "v2":
            a2 = a_ / a_.sum(-1, keepdim=True).clamp_min(1e-8)
            kk = F.normalize(model.keyW2(F.layer_norm(a2 @ model.pos.detach(), (model.d,))), dim=-1).view(Bk, m, -1)
        else:
            kk = F.normalize(model.keyW(torch.cat([a_ @ model.pos.detach(), h_t], -1)), dim=-1).view(Bk, m, -1)
        lp0 = torch.logit(q_prior[tgt].clamp(1e-6, 1 - 1e-6))
        shf = (ql.view(Bk, m) - lp0[:, None]) * bt.view(Bk, m)
        crowd = torch.relu(kk @ kk.transpose(1, 2)).sum(2)
        lmu = lp0 + torch.exp(log_u) * (shf / crowd).sum(1)
        l = bce(lmu, xb[torch.arange(Bk, device=dev), tgt].float()).mean()
        opt_k.zero_grad(set_to_none=True)
        l.backward()
        opt_k.step()
        if st % 500 == 0:
            kl_hist.append(float(l.detach()))
    for p_ in kp:
        p_.requires_grad_(False)
    res["heads"] = {"stiff_mse_trace": sl_hist, "key_bag_logloss_trace": kl_hist, "key_bag_u": float(torch.exp(log_u.detach())),
                    "sec": time.time() - t1}
    log("heads fitted", f"{time.time()-t1:.0f}s")
    if a.save_model:
        torch.save({"state": model.state_dict(), "args": vars(a), "perm": PERM.tolist()}, os.path.join(a.out, "model.pt"))
    inf = Infer(model, dev)
    qp = q_prior.double().cpu().numpy()

    # ---------------- P0 and the D0 gate (validation and test)
    def fam_arrays(I, fam, n):
        xg = I["xg"][:n]
        return (torch.tensor(tok(xg), dtype=torch.long, device=dev), torch.tensor(tok(I[f"{fam}_rev"][:n]), dtype=torch.bool, device=dev),
                torch.tensor(INV[I[f"{fam}_q"][:n]], dtype=torch.long, device=dev),
                I[f"{fam}_post"][np.arange(n), I[f"{fam}_q"][:n]], xg[np.arange(n), I[f"{fam}_q"][:n]].astype(np.float64),
                tok(I[f"{fam}_post"][:n]), tok(I[f"{fam}_rev"][:n]))

    p0res = {}
    for split, I in (("val", VI), ("test", TI)):
        p0res[split] = {}
        for fam in D.FAMILIES:
            n = a.n_eval if not a.smoke else min(200, a.n_eval)
            x, rv, yq, pst, y, post_all, rev_all = fam_arrays(I, fam, n)
            lg, s = inf(rv, torch.where(rv, x, torch.zeros_like(x)))
            q = torch.sigmoid(lg.double()).cpu().numpy()
            qy = q[np.arange(n), yq.cpu().numpy()]
            hid = ~rev_all
            kl_all = (kl_bern(post_all, q) * hid).sum(1) / hid.sum(1)
            p0res[split][fam] = {"n": n, "kl": float(kl_bern(pst, qy).mean()), "logloss": float(logloss(qy, y).mean()),
                                 "ece": ece(qy, y), "kl_all_hidden": float(kl_all.mean()),
                                 "exact_H": float(D.binary_entropy(pst).mean()), "exact_logloss": float(logloss(pst, y).mean())}
    res["P0"] = p0res
    g = p0res["val"]
    res["D0_gate_val"] = {"long_range_kl": {f: g[f]["kl"] for f in ("F", "B1", "B2")}, "one_step_kl": g["S1"]["kl"],
                          "pass": all(g[f]["kl"] >= 0.05 for f in ("F", "B1", "B2")) and g["S1"]["kl"] <= 0.02,
                          "solved_lt_0.01": all(g[f]["kl"] < 0.01 for f in ("F", "B1", "B2"))}
    log("P0/D0", json.dumps({f: round(g[f]["kl"], 4) for f in D.FAMILIES}))

    # ---------------- D1 stiffness (2000 held-out (i, S) pairs on TEST grids)
    rng = np.random.default_rng(31_000 + a.seed)
    npair = a.n_pairs
    xg_t = TI["xg"]
    gi = rng.integers(0, a.n_eval, npair)
    tgt_g = rng.integers(0, N, npair)
    S_g = D.random_reveal(npair, 32_000 + a.seed, N)
    S_g[np.arange(npair), tgt_g] = False
    post = D.posterior_marginals(rule, S_g, xg_t[gi], T, W, eps)
    H = D.binary_entropy(post[np.arange(npair), tgt_g])
    xt = torch.tensor(tok(xg_t[gi]), dtype=torch.long, device=dev)
    rt = torch.tensor(tok(S_g), dtype=torch.bool, device=dev)
    lg, s = inf(rt, torch.where(rt, xt, torch.zeros_like(xt)))
    ti = INV[tgt_g]
    s_i = s.cpu().numpy()[np.arange(npair), ti]
    b_i = beta_np(s_i)
    lab = np.array([any(all(S_g[j, c] for c in bl) for bl in D.blankets(tgt_g[j], T, W)) for j in range(npair)])
    lab_par = np.array([(D.parents(tgt_g[j], T, W) is not None) and all(S_g[j, c] for c in D.parents(tgt_g[j], T, W)) for j in range(npair)])
    q_i = torch.sigmoid(lg.double()).cpu().numpy()[np.arange(npair), ti]
    # expected version: 500 pairs x 8 draws of x_S (same i, S on 8 other test grids)
    ne = min(500, npair)
    reps = 8
    gi2 = rng.integers(0, a.n_eval, (ne, reps))
    S2 = np.repeat(S_g[:ne], reps, 0)
    post2 = D.posterior_marginals(rule, S2, xg_t[gi2.reshape(-1)], T, W, eps)
    H2 = D.binary_entropy(post2[np.arange(ne * reps), np.repeat(tgt_g[:ne], reps)]).reshape(ne, reps).mean(1)
    xt2 = torch.tensor(tok(xg_t[gi2.reshape(-1)]), dtype=torch.long, device=dev)
    rt2 = torch.tensor(tok(S2), dtype=torch.bool, device=dev)
    _, s2 = inf(rt2, torch.where(rt2, xt2, torch.zeros_like(xt2)))
    b2 = beta_np(s2.cpu().numpy()[np.arange(ne * reps), np.repeat(ti[:ne], reps)]).reshape(ne, reps).mean(1)
    # round 3 (R3-D3): epistemic error of the one-pass belief, KL(P*(X_i | x_S) || q(X_i | x_S)), and 1-step self-consistency:
    # z* = the stiffest unrevealed cell (highest beta_int, not i); sc = -E_{z ~ q(z* | S)}[(q(X_i | S, z*) - q(X_i | S))^2]
    p_i = post[np.arange(npair), tgt_g]
    kl_epi = kl_bern(p_i, q_i)
    beta_all = Net.beta_from_surprise(s).clone()
    cand_d = (~rt).clone()
    cand_d[torch.arange(npair, device=dev), torch.tensor(ti, device=dev)] = False
    beta_all = torch.where(cand_d, beta_all, torch.full_like(beta_all, -1e9))
    zst = beta_all.argmax(1)
    has_z = cand_d.any(1).cpu().numpy()
    rv2 = rt.unsqueeze(1).repeat(1, 2, 1)
    vv2 = torch.where(rt, xt, torch.zeros_like(xt)).unsqueeze(1).repeat(1, 2, 1)
    ar_ = torch.arange(npair, device=dev)
    rv2[ar_, 0, zst] = True
    rv2[ar_, 1, zst] = True
    vv2[ar_, 0, zst] = 0
    vv2[ar_, 1, zst] = 1
    lgz, _ = inf(rv2.reshape(-1, N), vv2.reshape(-1, N))
    tit = torch.tensor(ti, device=dev)
    qz = torch.sigmoid(lgz.double()).view(npair, 2, N)[ar_, :, tit].cpu().numpy()            # q(X_i | S, z* = 0 / 1)
    pz1 = torch.sigmoid(lg.double())[ar_, zst].cpu().numpy()
    selfcons = -(pz1 * (qz[:, 1] - q_i) ** 2 + (1 - pz1) * (qz[:, 0] - q_i) ** 2)
    selfcons = np.where(has_z, selfcons, 0.0)
    conf_i = LN2 - D.binary_entropy(q_i)
    hq = D.binary_entropy(q_i)
    bins_ = np.quantile(hq, np.linspace(0, 1, 11))
    bid = np.clip(np.searchsorted(bins_, hq, side="right") - 1, 0, 9)

    def matched(score):
        vals, wts = [], []
        for k_ in range(10):
            m_ = bid == k_
            if m_.sum() >= 20:
                v_ = spearman(score[m_], -kl_epi[m_])
                if np.isfinite(v_):
                    vals.append(v_)
                    wts.append(m_.sum())
        return float(np.average(vals, weights=wts)) if vals else float("nan")
    res["R3D3"] = {"spearman_vs_negKL": {"beta": spearman(b_i, -kl_epi), "confidence": spearman(conf_i, -kl_epi),
                                         "selfcons": spearman(selfcons, -kl_epi), "neg_surprise": spearman(-s_i, -kl_epi)},
                   "entropy_matched_spearman_vs_negKL": {"beta": matched(b_i), "confidence": matched(conf_i),
                                                         "selfcons": matched(selfcons)},
                   "mean_kl_epi": float(kl_epi.mean()), "frac_with_zstar": float(has_z.mean())}
    log("R3D3", json.dumps(res["R3D3"]))
    res["D1"] = {"n_pairs": int(npair), "spearman_beta_vs_info": spearman(b_i, LN2 - H),
                 "spearman_neg_surprise_vs_info": spearman(-s_i, LN2 - H),
                 "spearman_selfentropy_vs_info": spearman(LN2 - D.binary_entropy(q_i), LN2 - H),
                 "auc_blanket": auc(b_i, lab), "auc_parents_only": auc(b_i, lab_par), "frac_blanket": float(lab.mean()),
                 "frac_beta_zero": float((b_i == 0).mean()),
                 "expected_version": {"n_pairs": int(ne), "draws": reps, "spearman": spearman(b2, LN2 - H2)}}
    with open(os.path.join(a.out, "d1-pairs.csv"), "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["grid", "target_grid_cell", "reveal_count", "blanket", "parents", "s_hat", "beta", "q", "exact_p", "exact_H",
                     "kl_epi", "selfcons", "zstar_token"])
        for j in range(npair):
            w_.writerow([int(gi[j]), int(tgt_g[j]), int(S_g[j].sum()), int(lab[j]), int(lab_par[j]), f"{s_i[j]:.7f}",
                         f"{b_i[j]:.7f}", f"{q_i[j]:.9f}", f"{post[j, tgt_g[j]]:.9f}", f"{H[j]:.9f}",
                         f"{kl_epi[j]:.9g}", f"{selfcons[j]:.9g}", int(zst[j])])
    log("D1", json.dumps({k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in res["D1"].items() if k_ != "expected_version"}))

    # ---------------- chains: P2-Stiff, P2-VoI, P3-Conf, P4-Rand on validation (pool fitting) and test
    chain = {}
    passes = {}
    measured_passes = {}
    hyb_stats = {}
    for split, I, n in (("val", VI, a.n_val_inst), ("test", TI, a.n_test_inst)):
        chain[split] = {}
        for fam in CHAIN_FAMS:
            x, rv, yq, pst, y, _, _ = fam_arrays(I, fam, n)
            chain[split][fam] = {"pstar": pst, "y": y, "p0": qp[yq.cpu().numpy()]}
            for sel in SEL:
                tc = time.time()
                HYB_STATS.update(steps=0, branch=0)
                gq, gs, gk, ppc = run_chains(inf, x, rv, yq, sel, a.K, a.Mmax, gen, Ms)
                chain[split][fam][sel] = (gq, gs, gk)
                passes[sel] = ppc
                measured_passes.setdefault(f"{split}|{fam}|{sel}", {str(k_): v_ for k_, v_ in run_chains.passes_at.items()})
                if sel == "hyb":
                    hyb_stats[f"{split}|{fam}"] = dict(HYB_STATS)
                log("chains", split, fam, sel, f"{time.time()-tc:.1f}s", f"passes/chain {ppc:.1f}")
    res["passes_per_chain_at_Mmax"] = passes
    res["measured_passes_per_chain"] = measured_passes
    res["hyb_branch_share"] = {k_: (v_["branch"] / max(v_["steps"], 1)) for k_, v_ in hyb_stats.items()}

    # pools fitted on validation OUTCOMES (F, B1, B2, R pooled), per selector, pool and M
    fitted = {}
    for sel in SEL:
        for kind in ("lw", "dd"):
            for mi, M in enumerate(Ms):
                f_all, b_all, y_all = [], [], []
                for fam in CHAIN_FAMS:
                    c = chain["val"][fam]
                    gq, gs, gk = c[sel]
                    f_all.append(pool_feature(gq[:, :, mi], gs[:, :, mi], gk[:, :, mi], c["p0"], kind))
                    b_all.append(logit_np(c["p0"]))
                    y_all.append(c["y"])
                fitted[f"{sel}|{kind}|{M}"] = fit_u(np.concatenate(f_all), np.concatenate(b_all), np.concatenate(y_all))
    sw_fit = {}
    for sel in SEL:
        for mi, M in enumerate(Ms):
            qv = np.concatenate([chain["val"][fam][sel][0][:, :, mi] for fam in CHAIN_FAMS])
            sv = np.concatenate([chain["val"][fam][sel][1][:, :, mi] for fam in CHAIN_FAMS])
            P0v = np.concatenate([chain["val"][fam][sel][0][:, 0, 0] for fam in CHAIN_FAMS])
            yv = np.concatenate([chain["val"][fam]["y"] for fam in CHAIN_FAMS])
            sw_fit[f"{sel}|sw|{M}"] = fit_sw(qv, sv, P0v, yv)
    res["pool_u"] = fitted
    res["pool_sw"] = sw_fit

    # metrics for every selector x pool x M x family, on val (outcomes) and test (KL + outcomes)
    summ = {}
    est_cols = {}
    for split in ("val", "test"):
        summ[split] = {}
        for fam in CHAIN_FAMS:
            c = chain[split][fam]
            for sel in SEL:
                gq, gs, gk = c[sel]
                for kind in POOLS:
                    for mi, M in enumerate(Ms):
                        uu = (dict(sw_fit[f"{sel}|sw|{M}"], P0=gq[:, 0, 0]) if kind == "sw" else fitted.get(f"{sel}|{kind}|{M}"))
                        mu = pooled(gq[:, :, mi], gs[:, :, mi], gk[:, :, mi], c["p0"], kind, uu)
                        key = f"{fam}|{sel}|{kind}|{M}"
                        d_ = {"logloss": float(logloss(mu, c["y"]).mean()), "ece": ece(mu, c["y"])}
                        if split == "test":
                            d_["kl"] = float(kl_bern(c["pstar"], mu).mean())
                            est_cols[key] = logit_np(mu)
                        summ[split][key] = d_
    res["chains"] = summ
    passes_q = {}
    for sel in SEL:
        for M in Ms:
            per_chain = (M + 1) if sel not in ("voi", "voiu") else (1 + M * (1 + 2 * VOI_TOP))
            if sel == "hyb":                     # measured: cost per step depends on how often it branches
                mp = [measured_passes[k_][str(M)] for k_ in measured_passes if k_.startswith("test|") and k_.endswith("|hyb")]
                per_chain = float(np.mean(mp)) if mp else float("nan")
            passes_q[f"{sel}|{M}"] = {"passes_per_query": a.K * per_chain, "flops_per_query": a.K * per_chain * fpass}
    passes_q["P0"] = {"passes_per_query": 1, "flops_per_query": fpass}
    res["compute_per_query"] = passes_q

    # ---------------- D3a: redundant views (same 3 parents + far-past distractors; exactly redundant given the parents)
    rng3 = np.random.default_rng(33_000 + a.seed)
    n3 = min(1000, a.n_test_inst * 2)
    g3 = rng3.integers(0, a.n_eval, n3)
    tg = np.array([rng3.integers(2 * W, N) for _ in range(n3)])            # rows >= 2
    views_q, views_s, views_k, views_a = [], [], [], []
    for j in range(5):
        Sg = np.zeros((n3, N), bool)
        for r_ in range(n3):
            Sg[r_, D.parents(tg[r_], T, W)] = True
            t_ = tg[r_] // W
            far = np.arange(0, (t_ - 1) * W)                                 # rows <= t-2
            Sg[r_, rng3.choice(far, size=min(3, len(far)), replace=False)] = True
        xt3 = torch.tensor(tok(xg_t[g3]), dtype=torch.long, device=dev)
        rt3 = torch.tensor(tok(Sg), dtype=torch.bool, device=dev)
        ti3 = torch.tensor(INV[tg], device=dev)
        lg, s, kk, av3 = inf(rt3, torch.where(rt3, xt3, torch.zeros_like(xt3)), need_key=True, qidx=ti3)
        views_a.append(av3.cpu().numpy().astype(np.float64))
        ar = torch.arange(n3, device=dev)
        views_q.append(torch.sigmoid(lg[ar, ti3].double()).cpu().numpy())
        views_s.append(s[ar, ti3].cpu().numpy())
        views_k.append(kk[ar, ti3].cpu().numpy())
    vq, vs, vk = np.stack(views_q, 1), np.stack(views_s, 1), np.stack(views_k, 1)
    p03 = qp[INV[tg]]
    d3a = {}
    for kind in ("lw", "dd"):
        base1 = logit_np(p03) + pool_feature(vq[:, :1], vs[:, :1], vk[:, :1], p03, kind)
        for m in (2, 5):
            lm = logit_np(p03) + pool_feature(vq[:, :m], vs[:, :m], vk[:, :m], p03, kind)
            d3a[f"{kind}|drift_m{m}"] = float(np.abs(lm - base1).mean())
    for m in (2, 5):
        d3a[f"ratio_m{m}"] = d3a[f"dd|drift_m{m}"] / max(d3a[f"lw|drift_m{m}"], 1e-12)
    G = np.einsum("nad,nbd->nab", vk, vk)
    d3a["mean_offdiag_cos_redundant"] = float((G.sum((1, 2)) - 5) .mean() / 20)
    va = np.stack(views_a, 1)
    va = va / np.maximum(np.linalg.norm(va, axis=-1, keepdims=True), 1e-12)
    Ga = np.einsum("nad,nbd->nab", va, va)
    d3a["mean_offdiag_attncos_redundant"] = float((Ga.sum((1, 2)) - 5).mean() / 20)
    res["D3a"] = d3a
    log("D3a", json.dumps({k_: round(v_, 4) for k_, v_ in d3a.items()}))

    # ---------------- D3b: one forward chain + one backward chain on B1 / B2 (forced row order; evaluation construction)
    d3b = {}
    for split, I, n in (("val", VI, a.n_val_inst), ("test", TI, a.n_test_inst)):
        for fam in ("B1", "B2"):
            x, rv, yq, pst, y, _, rev_all_tok = fam_arrays(I, fam, n)
            qg = I[f"{fam}_q"][:n]
            revg = I[f"{fam}_rev"][:n]
            orders = {}
            for dname in ("fwd", "bwd"):
                o = -np.ones((n, a.Mmax), np.int64)
                for r_ in range(n):
                    ty = qg[r_] // W
                    hid = [g_ for g_ in range(N) if not revg[r_, g_] and g_ != qg[r_]]
                    if dname == "fwd":
                        cells = sorted([g_ for g_ in hid if g_ // W < ty], key=lambda g_: (g_ // W, rng3.random()))
                    else:
                        cells = sorted([g_ for g_ in hid if g_ // W > ty], key=lambda g_: (-(g_ // W), rng3.random()))
                    cells = cells[: a.Mmax]
                    o[r_, : len(cells)] = INV[np.array(cells, np.int64)] if cells else []
                orders[dname] = torch.tensor(o, device=dev)
            cl = {}
            avd = {}
            for dname in ("fwd", "bwd"):
                gq, gs, gk, _, gav = run_chains(inf, x, rv, yq, "order", 1, a.Mmax, gen, (a.Mmax,), order=orders[dname], want_av=True)
                cl[dname] = (gq[:, :, 0], gs[:, :, 0], gk[:, :, 0])
                avd[dname] = gav[:, 0, 0]
            p0f = qp[yq.cpu().numpy()]
            combos = {"fwd": [cl["fwd"]], "bwd": [cl["bwd"]], "both": [cl["fwd"], cl["bwd"]]}
            for cname, parts in combos.items():
                q_ = np.concatenate([p_[0] for p_ in parts], 1)
                s_ = np.concatenate([p_[1] for p_ in parts], 1)
                k_ = np.concatenate([p_[2] for p_ in parts], 1)
                for kind in ("dd", "lw"):
                    f_ = pool_feature(q_, s_, k_, p0f, kind)
                    d3b[f"{split}|{fam}|{cname}|{kind}|_f"] = (f_, logit_np(p0f), y, pst)
                if cname == "both":
                    d3b[f"{split}|{fam}|cos_fwd_bwd"] = float((k_[:, 0] * k_[:, 1]).sum(1).mean())
            nf_ = avd["fwd"] / np.maximum(np.linalg.norm(avd["fwd"], axis=-1, keepdims=True), 1e-12)
            nb_ = avd["bwd"] / np.maximum(np.linalg.norm(avd["bwd"], axis=-1, keepdims=True), 1e-12)
            d3b[f"{split}|{fam}|attncos_fwd_bwd"] = float((nf_ * nb_).sum(1).mean())
            d3b[f"{split}|{fam}|raw_fwd"] = (cl["fwd"][0][:, 0], y, pst)
            d3b[f"{split}|{fam}|raw_bwd"] = (cl["bwd"][0][:, 0], y, pst)
    d3b_out = {}
    for fam in ("B1", "B2"):
        d3b_out[f"{fam}|cos_fwd_bwd"] = d3b[f"test|{fam}|cos_fwd_bwd"]
        d3b_out[f"{fam}|attncos_fwd_bwd"] = d3b[f"test|{fam}|attncos_fwd_bwd"]
        for cname in ("fwd", "bwd", "both"):
            for kind in ("dd", "lw"):
                fv, bv, yv, _ = d3b[f"val|{fam}|{cname}|{kind}|_f"]
                u = fit_u(fv, bv, yv)
                ft, bt_, yt, pt = d3b[f"test|{fam}|{cname}|{kind}|_f"]
                mu = sig_np(bt_ + u * ft)
                d3b_out[f"{fam}|{cname}|{kind}"] = {"u": u, "kl": float(kl_bern(pt, mu).mean()), "logloss": float(logloss(mu, yt).mean())}
        for dname in ("fwd", "bwd"):
            qv, yv, pv = d3b[f"test|{fam}|raw_{dname}"]
            d3b_out[f"{fam}|raw_{dname}"] = {"kl": float(kl_bern(pv, qv).mean()), "logloss": float(logloss(qv, yv).mean())}
    res["D3b"] = d3b_out
    log("D3b", json.dumps({k_: (v_ if not isinstance(v_, dict) else {kk_: round(vv_, 4) for kk_, vv_ in v_.items()}) for k_, v_ in d3b_out.items()}))

    # ---------------- per-instance test estimates (raw file for the audit)
    with open(os.path.join(a.out, "test-estimates.csv"), "w", newline="") as f:
        w_ = csv.writer(f)
        keys = sorted(est_cols)
        fams_keys = {fam: [k_ for k_ in keys if k_.startswith(fam + "|")] for fam in CHAIN_FAMS}
        w_.writerow(["family", "inst", "y", "pstar", "p0_logit", "P0_logit"] + ["|".join(k_.split("|")[1:]) for k_ in fams_keys["F"]])
        for fam in CHAIN_FAMS:
            n = a.n_test_inst
            x, rv, yq, pst, y, _, _ = fam_arrays(TI, fam, n)
            lg, _ = inf(rv, torch.where(rv, x, torch.zeros_like(x)))
            qd = torch.sigmoid(lg.double()).cpu().numpy()[np.arange(n), yq.cpu().numpy()]
            c = chain["test"][fam]
            cols = [est_cols[k_] for k_ in fams_keys[fam]]
            for i in range(n):
                w_.writerow([fam, i, int(y[i]), f"{pst[i]:.10f}", f"{logit_np(c['p0'][i]):.7f}", f"{logit_np(qd[i]):.7f}"]
                            + [f"{col[i]:.7f}" for col in cols])
    res["total_sec"] = time.time() - t_start
    res["infer_passes_total"] = inf.passes
    tmp = os.path.join(a.out, "result.json.tmp")
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, os.path.join(a.out, "result.json"))
    log("DONE", rule, a.arm, a.seed, f"total {res['total_sec']:.0f}s")


if __name__ == "__main__":
    main()
