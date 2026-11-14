"""Experiment B rollouts (linear-agg, 2026-10-16): vector rollout tokens from the SAVED E-CA round-3 rule-110, 4-layer
checkpoints (seeds 0-2; backbone, stiffness and key heads frozen, no training here).

Per instance (grid, evidence E0 of family B1 or B2, query cell Y):
  one pass at E0: P0 = q(Y | E0); k_q = the checkpoint's v2 key of Y at E0; beta0 = beta_int(Y | E0);
  bridge cells c_1..c_3 = the 3 hidden non-Y cells with the highest last-layer (head-averaged) attention from Y at E0;
  conf0 = leave-one-out evidence conflict at E0 (below).
  K = 16 rollouts, temperatures tau_r = 0.5, 1.0, 1.5, 2.0 (4 rollouts each): random order (as R3's `rand`), M = 16
  materialised cells, each sampled from sigmoid(logit q(z | S) / tau_r). After M steps, at S_M^r:
    v_r = (logit q(Y | S_M) - logit P0,  logit q(c_j | S_M minus c_j) - logit q(c_j | E0) for j = 1..3)        [4]
    k_r = the checkpoint's v2 key of Y at S_M (attention over the materialised cells S_M minus E0)            [16]
    c_r = (beta_int(Y | S_M), conflict_r, trajectory log-prob sum_t log q(z_t = sampled | S_{t-1}), tau_r)   [4]
  conflict_r = mean over evidence cells e in E0 of -log q(x_e | S_M minus e): how badly the rollout's materialised
  cells (plus the other evidence) predict each piece of evidence (domain-free leave-one-out form of gemini's
  "boundary conflict score").
Also an exact replication of round 3's `rand` chains (tau = 1, K = 16, M = 16) on F, B1, B2 (validation and test, first 500
instances each), for the R3 scalar pools.
Exact posteriors (pstar) are copied from round 3's cached evaluation sets, for scoring only; cal instances have none.
Writes OUT/b-seed{s}.npz.     python3 b_rollouts.py --seed 0 --out runs/b [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SMALLNET_DIR", os.path.join(HERE, "eca"))
os.environ.setdefault("ECA_DATA", os.path.join(HERE, "data3"))
sys.path.insert(0, os.path.join(HERE, "eca"))
import ecadata3 as D  # noqa: E402
import ecarun as R  # noqa: E402  (round-3 code, unchanged: run_chains, Infer)
from ecanet import Net  # noqa: E402

TAUS = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0)
M_STEPS = 16
T, W, EPS, RULE = 10, 12, 0.015, 110
N = T * W
CAL_SEEDS = {"B1": 55_001, "B2": 55_002}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lgt(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


class Fwd:
    """Chunked frozen forward pass; returns only what is needed at the query token."""

    def __init__(self, net, chunk):
        self.net, self.chunk, self.passes = net, chunk, 0

    @torch.no_grad()
    def __call__(self, rev, val, qidx=None, rev_base=None, want_key=False, want_attn=False):
        net = self.net
        L, S_, K_, A_ = [], [], [], []
        for a in range(0, rev.shape[0], self.chunk):
            r, v = rev[a:a + self.chunk], val[a:a + self.chunk]
            h, attn = net.backbone(net.tokens(r, v), need_attn=(want_key or want_attn))
            L.append(net.logits(h).float())
            S_.append(net.surprise(h).float())
            if want_key or want_attn:
                ar = torch.arange(r.shape[0], device=r.device)
                q_ = qidx[a:a + self.chunk]
                if want_key:
                    K_.append(net.key(h, attn, r, rev_base[a:a + self.chunk])[ar, q_].float())
                if want_attn:
                    A_.append(attn[ar, q_].float())
        self.passes += rev.shape[0]
        return (torch.cat(L), torch.cat(S_), torch.cat(K_) if K_ else None, torch.cat(A_) if A_ else None)


@torch.no_grad()
def loo_conflict(fw, rev, val, ev, rows_chunk=1024):
    """Mean over evidence cells e (ev[b] True) of -log q(x_e | revealed cells minus e). rev, val, ev [B, N]."""
    B = rev.shape[0]
    nE = int(ev[0].sum())
    assert bool((ev.sum(1) == nE).all())
    Eidx = torch.nonzero(ev, as_tuple=False)[:, 1].view(B, nE)
    out = torch.empty(B, dtype=torch.float64, device=rev.device)
    for a in range(0, B, rows_chunk):
        b = min(B, a + rows_chunk)
        rr = b - a
        re = rev[a:b, None, :].expand(rr, nE, N).clone()
        ve = val[a:b, None, :].expand(rr, nE, N).clone()
        e = Eidx[a:b]
        xe = val[a:b].gather(1, e)                                   # evidence values
        ar1 = torch.arange(rr, device=rev.device)[:, None].expand(rr, nE)
        ar2 = torch.arange(nE, device=rev.device)[None, :].expand(rr, nE)
        re[ar1, ar2, e] = False
        ve[ar1, ar2, e] = 0
        lg, _, _, _ = fw(re.reshape(-1, N), ve.reshape(-1, N))
        q = torch.sigmoid(lg.double()).view(rr, nE, N)[ar1, ar2, e]
        ll = -torch.where(xe == 1, torch.log(q.clamp_min(1e-12)), torch.log((1 - q).clamp_min(1e-12)))
        out[a:b] = ll.mean(1)
    return out


@torch.no_grad()
def rollouts(fw, x, rev0, yq, gen, rows_chunk):
    """x, rev0 [n, N] token order, yq [n] query token -> dict of numpy arrays (instance-major chains)."""
    dev = x.device
    n = x.shape[0]
    K = len(TAUS)
    val0 = torch.where(rev0, x, torch.zeros_like(x))
    ar = torch.arange(n, device=dev)
    lg0, s0, kq, att0 = fw(rev0, val0, qidx=yq, rev_base=rev0, want_key=True, want_attn=True)
    q0 = torch.sigmoid(lg0.double())
    P0 = q0[ar, yq]
    beta0 = Net.beta_from_surprise(s0)[ar, yq].double()
    isYn = torch.nn.functional.one_hot(yq, N).bool()
    cand0 = (~rev0) & (~isYn)
    bridge = torch.where(cand0, att0, torch.full_like(att0, -1.0)).topk(3, 1).indices          # [n, 3]
    conf0 = loo_conflict(fw, rev0, val0, rev0, rows_chunk)
    out = {k: [] for k in ("qY", "v", "k", "c", "s")}
    for a in range(0, n, max(1, rows_chunk // K)):
        b = min(n, a + max(1, rows_chunk // K))
        nn_ = b - a
        Rr = nn_ * K
        rows = torch.arange(Rr, device=dev)
        tau = torch.tensor(TAUS, dtype=torch.float64, device=dev).repeat(nn_)
        rb = rev0[a:b].repeat_interleave(K, 0)
        rev = rb.clone()
        val = val0[a:b].repeat_interleave(K, 0).clone()
        yy = yq[a:b].repeat_interleave(K, 0)
        isY = torch.nn.functional.one_hot(yy, N).bool()
        tlp = torch.zeros(Rr, dtype=torch.float64, device=dev)
        for t in range(M_STEPS):
            lg, _, _, _ = fw(rev, val)
            q = torch.sigmoid(lg.double())
            cnd = (~rev) & (~isY)
            has = cnd.any(1)
            wsel = cnd.float() + (~has).float()[:, None]
            z = torch.multinomial(wsel, 1, generator=gen).squeeze(1)
            qz = q[rows, z]
            pz = torch.sigmoid(lgt(qz, 1e-12) / tau)
            zv = (torch.rand(Rr, device=dev, generator=gen, dtype=torch.float64) < pz).long()
            upd = has & (~rev[rows, z]) & (z != yy)
            tlp += torch.where(upd, torch.log(torch.where(zv == 1, qz, 1 - qz).clamp_min(1e-12)), torch.zeros_like(tlp))
            rev[rows[upd], z[upd]] = True
            val[rows[upd], z[upd]] = zv[upd]
        lg, s, kr, _ = fw(rev, val, qidx=yy, rev_base=rb, want_key=True)
        qf = torch.sigmoid(lg.double())
        qY = qf[rows, yy]
        betaY = Net.beta_from_surprise(s)[rows, yy].double()
        br = bridge[a:b].repeat_interleave(K, 0)
        q0K = q0[a:b].repeat_interleave(K, 0)
        vb = []
        for j in range(3):
            c = br[:, j]
            qc = qf[rows, c].clone()
            mat = rev[rows, c]                                        # bridge cell materialised by this rollout
            if bool(mat.any()):
                idx = rows[mat]
                r2, v2 = rev[idx].clone(), val[idx].clone()
                r2[torch.arange(len(idx), device=dev), c[idx]] = False
                v2[torch.arange(len(idx), device=dev), c[idx]] = 0
                lgj, _, _, _ = fw(r2, v2)
                qc[idx] = torch.sigmoid(lgj.double())[torch.arange(len(idx), device=dev), c[idx]]
            vb.append(lgt(qc) - lgt(q0K[rows, c]))
        confl = loo_conflict(fw, rev, val, rb, rows_chunk)
        v = torch.stack([lgt(qY) - lgt(P0[a:b].repeat_interleave(K, 0))] + vb, 1)
        cc = torch.stack([betaY, confl, tlp, tau], 1)
        out["qY"].append(qY.view(nn_, K).cpu().numpy())
        out["v"].append(v.view(nn_, K, 4).cpu().numpy())
        out["k"].append(kr.double().view(nn_, K, -1).cpu().numpy())
        out["c"].append(cc.view(nn_, K, 4).cpu().numpy())
        out["s"].append(s[rows, yy].double().view(nn_, K).cpu().numpy())
    res = {k: np.concatenate(v_, 0) for k, v_ in out.items()}
    res.update({"P0": P0.cpu().numpy(), "beta0": beta0.cpu().numpy(), "kq": kq.double().cpu().numpy(),
                "conf0": conf0.cpu().numpy(), "bridge": bridge.cpu().numpy(), "pY0": None})
    res.pop("pY0")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt"))
    ap.add_argument("--n-cal", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-rep", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--rows-chunk", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.smoke:
        a.n_cal, a.n_val, a.n_test, a.n_rep = 8, 8, 8, 8
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(os.path.join(a.ckpt, f"r110-L4-s{a.seed}", "model.pt"), map_location="cpu", weights_only=False)
    assert ck["args"]["layers"] == 4 and ck["args"]["rule"] == RULE and ck["args"]["key_mode"] == "v2"
    net = Net(n=N, layers=4).to(dev)
    net.load_state_dict(ck["state"])
    net.key_mode = "v2"
    net.eval()
    PERM = np.array(ck["perm"])
    assert (PERM == D.make_perm(T, W)).all()
    INV = np.argsort(PERM)
    tok = lambda arr: arr[:, PERM]                                                   # grid order -> token order
    fw = Fwd(net, a.chunk)
    gen = torch.Generator(device=dev)
    gen.manual_seed(77_000 + a.seed)
    t0 = time.time()
    save = {"taus": np.array(TAUS), "perm": PERM}
    meta = {"seed": a.seed, "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "torch": torch.__version__, "sizes": {}}
    with torch.no_grad():
        h, _ = net.backbone(torch.zeros(1, N, dtype=torch.long, device=dev))
        qp = torch.sigmoid(net.logits(h).double())[0].cpu().numpy()                  # network's own no-evidence prior
    VI = D.eval_instances("val", RULE, 2000, T, W, EPS)
    TI = D.eval_instances("test", RULE, 2000, T, W, EPS)
    xcal = D.load_grids("cal", RULE, 4000, T, W, EPS)
    for split in ("cal", "val", "test"):
        for fam in ("B1", "B2"):
            if split == "cal":
                n = a.n_cal
                rev_g, q_g = D.family_masks(fam, 4000, CAL_SEEDS[fam], T, W)
                xg, rev_g, q_g = xcal[:n], rev_g[:n], q_g[:n]
                pst = np.full(n, np.nan)
            else:
                I = VI if split == "val" else TI
                n = a.n_val if split == "val" else a.n_test
                xg, rev_g, q_g = I["xg"][:n], I[f"{fam}_rev"][:n], I[f"{fam}_q"][:n]
                pst = I[f"{fam}_post"][np.arange(n), q_g]
            x = torch.tensor(tok(xg), dtype=torch.long, device=dev)
            rv = torch.tensor(tok(rev_g), dtype=torch.bool, device=dev)
            yq = torch.tensor(INV[q_g], dtype=torch.long, device=dev)
            tc, p_ = time.time(), fw.passes
            r = rollouts(fw, x, rv, yq, gen, a.rows_chunk)
            r["y"] = xg[np.arange(n), q_g].astype(np.float64)
            r["pstar"] = pst
            r["qcell"] = q_g
            r["p0"] = qp[INV[q_g]]
            for k_, v_ in r.items():
                save[f"{split}|{fam}|{k_}"] = v_
            meta["sizes"][f"{split}|{fam}"] = n
            log("rollouts", split, fam, n, f"{time.time() - tc:.1f}s", f"passes {fw.passes - p_}")
    # round-3 replication: rand chains, tau = 1, K = 16, M = 16, F/B1/B2, first n_rep val and test instances (R3 protocol)
    inf = R.Infer(net, dev)
    for split, I in (("val", VI), ("test", TI)):
        for fam in ("F", "B1", "B2"):
            n = a.n_rep
            xg, rev_g, q_g = I["xg"][:n], I[f"{fam}_rev"][:n], I[f"{fam}_q"][:n]
            x = torch.tensor(tok(xg), dtype=torch.long, device=dev)
            rv = torch.tensor(tok(rev_g), dtype=torch.bool, device=dev)
            yq = torch.tensor(INV[q_g], dtype=torch.long, device=dev)
            tc = time.time()
            gq, gs, gk, _ = R.run_chains(inf, x, rv, yq, "rand", 16, 16, gen, (0, 16))
            save[f"rep|{split}|{fam}|q"] = gq
            save[f"rep|{split}|{fam}|s"] = gs
            save[f"rep|{split}|{fam}|k"] = gk
            save[f"rep|{split}|{fam}|y"] = xg[np.arange(n), q_g].astype(np.float64)
            save[f"rep|{split}|{fam}|pstar"] = I[f"{fam}_post"][np.arange(n), q_g]
            save[f"rep|{split}|{fam}|p0"] = qp[INV[q_g]]
            log("rep chains", split, fam, f"{time.time() - tc:.1f}s")
    meta["sec"] = time.time() - t0
    meta["passes_main"] = fw.passes
    meta["passes_rep"] = inf.passes
    np.savez_compressed(os.path.join(a.out, f"b-seed{a.seed}.npz"), **save)
    json.dump(meta, open(os.path.join(a.out, f"b-seed{a.seed}.json"), "w"), indent=1)
    log("DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
