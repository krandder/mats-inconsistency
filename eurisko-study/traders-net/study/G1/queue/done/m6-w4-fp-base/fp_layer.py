"""M6 worker 4: fixed-point market layer (arm g) vs 1-pass / 2-pass, plus references.
usage: fp_layer.py STREAM(base|li) ARM(plain|plain_wd|boost|pass1|pass2|fp) SEED [T]
Writes results/<stream>-<arm>-s<seed>.json. See PREREG.md."""
import json, math, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "substrate") if os.path.isdir(os.path.join(HERE, "substrate")) else os.path.join(HERE, "..", "..", "S0", "substrate")
sys.path.insert(0, SUB)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Net, Logistic
from prequential import run_prequential, per_family

K = 16


def trunk(net, win):
    B = win.shape[0]
    x = net.win(2.0 * win - 1.0) + net.qid.unsqueeze(0).expand(B, -1, -1)
    for blk in net.blocks:
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).view(B, net.Q, 3, net.H, blk.dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
        x = x + blk.o((A @ v).transpose(1, 2).reshape(B, net.Q, net.d))
        x = x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))))
    return net.lnf(x)


class Implicit(torch.autograd.Function):
    """forward: identity on f (= l* at convergence); backward: g -> (I - J)^-T g."""
    @staticmethod
    def forward(ctx, f, A):
        ctx.save_for_backward(A)
        return f.clone()

    @staticmethod
    def backward(ctx, g):
        (A,) = ctx.saved_tensors
        v = torch.linalg.solve(A.transpose(-1, -2), g.unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(v).all():
            v = g
        return v, None


class MarketLayer(nn.Module):
    def __init__(self, Q, W, d, L, arm, maker):
        super().__init__()
        self.arm, self.Q = arm, Q
        self.net = Net(Q, W, d=d, H=4, L=L)
        del self.net.rw, self.net.rb
        s = 1 / math.sqrt(d)
        self.wb = nn.Parameter(torch.randn(K, Q, d) * s / math.sqrt(K))
        self.cb = nn.Parameter(torch.zeros(K, Q))
        self.ub = nn.Parameter(torch.zeros(K, Q, 2))
        self.wk = nn.Parameter(torch.randn(K, Q, d) * s * 0.1)
        self.ck = nn.Parameter(torch.full((K, Q), -2.0))
        self.uk = nn.Parameter(torch.zeros(K, Q, 2))
        self.M = nn.Parameter(torch.zeros(Q, Q))
        object.__setattr__(self, "maker", maker)   # trained by its own optimizer, not ours
        self.stats = {"iters": [], "resid": [], "nonconv": 0, "n": 0}

    def _bk(self, h):
        bh = torch.einsum("nqd,kqd->nkq", h, self.wb) + self.cb
        kh = torch.einsum("nqd,kqd->nkq", h, self.wk) + self.ck
        return bh, kh

    def F(self, l, l0, bh, kh, need_grad_x=False):
        p = torch.sigmoid(l)
        x1 = 2 * p - 1
        m = x1 @ self.M.T
        if need_grad_x:
            x1 = x1.detach().requires_grad_(True); m = m.detach().requires_grad_(True)
        b = bh + self.ub[..., 0] * x1[:, None] + self.ub[..., 1] * m[:, None]
        k = F.softplus(kh + self.uk[..., 0] * x1[:, None] + self.uk[..., 1] * m[:, None])
        f = (l0 + (k * b).sum(1)) / (1 + k.sum(1))
        return f, x1, m, p

    def jac(self, l, l0, bh, kh):
        """(I - J) at l; J = diag(a) S + diag(c) M S, S = diag(2p(1-p))."""
        with torch.enable_grad():
            f, x1, m, p = self.F(l, l0, bh.detach(), kh.detach(), need_grad_x=True)
            a, c = torch.autograd.grad(f.sum(), [x1, m])
        S = 2 * p * (1 - p)
        J = torch.diag_embed(a * S) + c[:, :, None] * self.M.detach()[None] * S[:, None, :]
        I = torch.eye(self.Q).expand_as(J)
        return f.detach(), I - J

    def forward(self, win):
        h = trunk(self.net, win)
        bh, kh = self._bk(h)
        if self.arm in ("plain", "plain_wd"):
            return bh.sum(1)
        with torch.no_grad():
            l0 = self.maker(win).clamp(-20, 20)
        if self.arm == "boost":
            return l0 + 0.3 * bh.sum(1)
        l = self.F(l0, l0, bh, kh)[0]
        if self.arm == "pass1":
            return l
        if self.arm == "pass2":
            return self.F(l, l0, bh, kh)[0]
        # fixed point (arm g)
        with torch.no_grad():
            z = l.detach()
            it = 0
            for it in range(1, 9):
                f, A = self.jac(z, l0, bh, kh)
                r = f - z
                if r.abs().max() < 1e-5:
                    it -= 1
                    break
                dz = torch.linalg.solve(A, r.unsqueeze(-1)).squeeze(-1)
                dz = torch.nan_to_num(dz).clamp(-4, 4)
                z = (z + dz).clamp(-20, 20)
            f, A = self.jac(z, l0, bh, kh)
            res = (f - z).abs().max(-1).values
        self.stats["iters"].append(it); self.stats["resid"].append(float(res.max()))
        self.stats["nonconv"] += int((res > 1e-4).sum()); self.stats["n"] += int(res.numel())
        fz = self.F(z, l0, bh, kh)[0]
        return Implicit.apply(fz, A)


def base_ops(desc):
    ops = {}
    for q, d in enumerate(desc):
        mm = re.match(r"(and|or)\(q(\d+) \S+, q(\d+)", d)
        if mm:
            ops[q] = (mm.group(1), int(mm.group(2)), int(mm.group(3)))
    return ops


def andor_viol(p, ops):
    out = {}
    for kind in ("and", "or"):
        v = []
        for q, (k, a, b) in ops.items():
            if k != kind:
                continue
            pa, pb, pc = p[:, a], p[:, b], p[:, q]
            if kind == "and":
                v.append(np.maximum(0, pc - np.minimum(pa, pb)) + np.maximum(0, pa + pb - 1 - pc))
            else:
                v.append(np.maximum(0, np.maximum(pa, pb) - pc) + np.maximum(0, pc - pa - pb))
        out[kind] = float(np.mean(v)) if v else float("nan")
    return out


def main():
    stream, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    res = {"stream": stream, "arm": arm, "seed": seed}
    if stream == "base":
        import streams_ext as se
        T = int(sys.argv[4]) if len(sys.argv) > 4 else 24032
        s = se.make_streams(Q=64, T=T, seed=seed, heldout=8, release=T // 2, regime_every=0)
        ps = se.to_array_stream(s); d, L = 64, 2; fams = se.FAMILIES
    else:
        from li_families import make_li_stream
        T = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
        st = make_li_stream(seed, T); ps = st.as_positions(); d, L = 32, 1; fams = st.fam_names
    maker = Logistic(ps.Q, ps.W)
    model = MarketLayer(ps.Q, ps.W, d, L, arm, maker)
    opt = (torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.1) if arm == "plain_wd"
           else torch.optim.Adam(model.parameters(), 1e-3))
    out = run_prequential(model, ps, opt, B=4, baseline=maker, opt_b=torch.optim.Adam(maker.parameters(), 1e-2))
    z, y, m = out["z"].astype(np.float64), out["y"], out["mask"]
    res.update({"T": T, "Q": ps.Q, "families": list(fams), "steps": out["steps"]})
    if stream == "base":
        half = slice(ps.n_pos // 2, ps.n_pos)
        for name, zz in (("net", z), ("maker", out["zb"].astype(np.float64))):
            bce, br = per_family(zz[half], y[half], m[half], ps.fam, ps.n_fam)
            res[f"{name}_bce"] = dict(zip(fams, map(float, bce)))
            res[f"{name}_brier"] = dict(zip(fams, map(float, br)))
        nb = res["net_bce"]
        learn = ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]
        res["M1_learnable_bce"] = float(np.mean([nb[f] for f in learn]))
        res["M2_noise_excess"] = float(max(nb["prng"] - math.log(2), nb["ca30"] - math.log(2)))
        p = 1 / (1 + np.exp(-z[half]))
        res["andor_viol"] = andor_viol(p, base_ops(s.desc))
    else:
        from li_families import violations
        y64 = y.astype(np.float64); p = 1 / (1 + np.exp(-z))
        bce = np.logaddexp(0, z) - y64 * z
        res["net_bce"] = {f: float(bce[m & (st.fam == i)[None, :]].mean()) for i, f in enumerate(fams)}
        t = np.arange(ps.n_pos) + st.W
        v = violations(p, st, t=t, tol=0.05)
        res["viol"] = v
        res["M3_D_coherence"] = float(np.mean([v["pre:" + k]["mean"] for k in ("taut_or", "contra", "impl", "ex3")]))
        res["andor_viol"] = {k: v[k]["mean"] for k in ("and", "or")}
        zb = out["zb"].astype(np.float64); pb = 1 / (1 + np.exp(-zb))
        vb = violations(pb, st, t=t, tol=0.05)
        res["maker_M3_D_coherence"] = float(np.mean([vb["pre:" + k]["mean"] for k in ("taut_or", "contra", "impl", "ex3")]))
    if arm == "fp":
        it = np.array(model.stats["iters"])
        res["fp"] = {"iters_mean": float(it.mean()), "iters_max": int(it.max()), "at_cap_frac": float((it >= 8).mean()),
                     "resid_max": float(np.max(model.stats["resid"])),
                     "nonconv_frac": model.stats["nonconv"] / max(1, model.stats["n"])}
    res["wall_s"] = time.time() - t0
    rd = os.environ.get("OUT", os.path.join(HERE, "results")); os.makedirs(rd, exist_ok=True)
    path = os.path.join(rd, f"{stream}-{arm}-s{seed}.json")
    json.dump(res, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
    print(path, f"{res['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
