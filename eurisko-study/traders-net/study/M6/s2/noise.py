"""M6 stress s2: 50% injected noise traders in w2's harness (imports w2/pool.py unchanged).

usage: python noise.py --arm {a,b,cg,cq,fb01,fb05,wd,boost} --noise {clean,iid,fixed} --seed S --out DIR
Heads 8..15 of 16 are replaced by untrainable noise traders (sigma = 1 logit):
  iid   d_i ~ N(0,1) fresh per sample/question/pass (own generator, seeded by the run seed);
  fixed d_i = (2 win - 1) . R_i / sqrt(W), R_i ~ N(0,1) frozen (buffer, not a parameter).
See PREREG.md.
"""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = "/home/agent/workspaces/eurisko-study/traders-net/study"
if os.path.isdir(f"{HERE}/S0"):
    sys.path[:0] = [f"{HERE}/S0/substrate", f"{HERE}/M1", f"{HERE}/w2"]
else:
    sys.path[:0] = [f"{STUDY}/S0/substrate", f"{STUDY}/M1", f"{STUDY}/M6/w2"]
import numpy as np, torch, torch.nn as nn  # noqa
import pool, streams_ext, prequential  # noqa
from model import Logistic  # noqa

K, NN, SIG = 16, 8, 1.0
NOISE = slice(K - NN, K)


class NoiseMarket(pool.Market):
    def __init__(self, Q, W, arm, base, noise, seed):
        inner = {"wd": "a", "boost": "a", "b": "a", "fb01": "a", "fb05": "a"}.get(arm, arm)
        super().__init__(Q, W, inner, base)
        self.arm2, self.noise, self.W = arm, noise, W
        self.gen = torch.Generator().manual_seed(9000 + seed)
        self.register_buffer("R", torch.randn(NN, Q, W, generator=torch.Generator().manual_seed(8000 + seed)))

    def forward(self, win):
        l0 = self.base(win).detach()
        y = self.trunk(win)
        d = torch.einsum("bqd,kqd->bkq", y, self.rw) + self.rc[None] * l0[:, None] + self.rb[None]
        if self.noise != "clean":
            B, Q = l0.shape
            if self.noise == "iid":
                nz = SIG * torch.randn(B, NN, Q, generator=self.gen)
            else:
                nz = SIG * torch.einsum("bqw,kqw->bkq", 2.0 * win - 1.0, self.R) / math.sqrt(self.W)
            d = torch.cat([d[:, :K - NN], nz], 1)
        a = self.arm2
        self.d_last = d.detach()
        if a in ("a", "wd"):
            return d.sum(1)
        if a == "b":
            return l0 + d.sum(1)
        if a == "boost":
            return l0 + 0.1 * d.sum(1)
        if a in ("fb01", "fb05"):
            bb = 0.1 if a == "fb01" else 0.5
            return l0 + (bb * torch.tanh(d / bb)).sum(1)
        pi = torch.sigmoid(l0[:, None] + d)
        if a == "cg":
            k = torch.softmax(self.theta, 0)[None, :, None].expand(1, K, self.Q)
        else:
            k = torch.softmax(self.theta, 1).T[None]
        p = (k * pi).sum(1).clamp(pool.EPS, 1 - pool.EPS)
        return torch.log(p) - torch.log1p(-p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--noise", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--Q", type=int, default=64); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    s = streams_ext.make_streams(Q=a.Q, T=a.T, seed=a.seed, heldout=8, release=a.T // 2, regime_every=0)
    st = streams_ext.to_array_stream(s)
    torch.manual_seed(a.seed)
    base = Logistic(st.Q, st.W)
    torch.manual_seed(1000 + a.seed)
    m = NoiseMarket(st.Q, st.W, a.arm, base, a.noise, a.seed)
    own = [(n_, p) for n_, p in m.named_parameters() if not n_.startswith("base.")]
    main_p = [p for n_, p in own if n_ != "theta"]; th = [p for n_, p in own if n_ == "theta"]
    groups = [{"params": main_p, "lr": 1e-3}] + ([{"params": th, "lr": 1e-2}] if th else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.1) if a.arm == "wd" else torch.optim.Adam(groups)
    opt_b = torch.optim.Adam(base.parameters(), lr=1e-2)
    mag = []

    def on_step(s_, P, Z, Y, M):
        if s_ % 200 == 0 and getattr(m, "d_last", None) is not None:
            d = m.d_last
            mag.append((s_, float(d[:, :K - NN].abs().mean()), float(d[:, K - NN:].abs().mean())))

    t0 = time.time()
    r = prequential.run_prequential(m, st, opt, B=4, baseline=base, opt_b=opt_b, on_step=on_step)
    wall = time.time() - t0
    z, zb, y, mk = r["z"], r["zb"], r["y"], r["mask"]
    n = st.n_pos
    res = {"arm": a.arm, "noise": a.noise, "seed": a.seed, "T": a.T, "Q": a.Q, "n_noise": NN, "sigma": SIG,
           "wall_s": wall, "families": streams_ext.FAMILIES}
    for h, sl in {"all": slice(0, n), "late": slice(n // 2, n)}.items():
        bn, _ = prequential.per_family(z[sl], y[sl], mk[sl], st.fam, st.n_fam)
        bb, _ = prequential.per_family(zb[sl], y[sl], mk[sl], st.fam, st.n_fam)
        res[f"bce_{h}"] = bn.tolist(); res[f"bce_l0_{h}"] = bb.tolist()
    if a.arm == "cg":
        k = torch.softmax(m.theta.detach(), 0).numpy()
        res["k_final"] = k.tolist(); res["k_noise"] = float(k[K - NN:].sum())
    elif a.arm == "cq":
        k = torch.softmax(m.theta.detach(), 1).numpy()
        res["k_noise"] = float(k[:, K - NN:].sum(1).mean()); res["k_noise_per_q"] = k[:, K - NN:].sum(1).tolist()
    res["absd_trace"] = mag  # (step, mean|d| trainable heads, mean|d| noise heads)
    fn = f"{a.out}/{a.arm}-{a.noise}-s{a.seed}.json"
    json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print(fn, f"wall {wall:.0f}s")


if __name__ == "__main__":
    main()
