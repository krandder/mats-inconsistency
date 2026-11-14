"""M6 stress item 4 (s3): surviving market-layer arms on the S0 D-family (LI) stream.
usage: dfam.py ARM SEED [T]   ARM in a b cg cq fb01 fb05 wd boost.  See PREREG.md.
Writes $OUT/li-<arm>-s<seed>.json (default ./results)."""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "substrate") if os.path.isdir(os.path.join(HERE, "substrate")) else \
    os.path.join(HERE, "..", "..", "S0", "substrate")
sys.path.insert(0, SUB)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Net, Logistic
from prequential import run_prequential
from li_families import make_li_stream, violations

K = 16
EPS = 1e-6
LEARN = ["delay0", "delay64", "delay512", "and", "or", "exp64", "exp256", "exp1024"]
PROVE = ["taut_or", "contra", "impl", "ex3"]


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


class Layer(nn.Module):
    def __init__(self, Q, W, arm, maker, d=32, L=1):
        super().__init__()
        self.arm, self.Q = arm, Q
        self.net = Net(Q, W, d=d, H=4, L=L)
        del self.net.rw, self.net.rb
        self.rw = nn.Parameter(torch.randn(K, Q, d) / math.sqrt(d))
        self.rc = nn.Parameter(torch.zeros(K, Q))
        self.rb = nn.Parameter(torch.zeros(K, Q))
        if arm == "cg":
            self.theta = nn.Parameter(torch.zeros(K))
        elif arm == "cq":
            self.theta = nn.Parameter(torch.zeros(Q, K))
        object.__setattr__(self, "maker", maker)   # own optimiser (run_prequential baseline)

    def forward(self, win):
        with torch.no_grad():
            l0 = self.maker(win).clamp(-20, 20)
        y = trunk(self.net, win)
        d = torch.einsum("bqd,kqd->bkq", y, self.rw) + self.rc[None] * l0[:, None] + self.rb[None]
        a = self.arm
        if a in ("a", "wd"):
            return d.sum(1)
        if a == "b":
            return l0 + d.sum(1)
        if a == "boost":
            return l0 + 0.1 * d.sum(1)
        if a == "fb01":
            return l0 + (0.1 * torch.tanh(d / 0.1)).sum(1)
        if a == "fb05":
            return l0 + (0.5 * torch.tanh(d / 0.5)).sum(1)
        pi = torch.sigmoid(l0[:, None] + d)
        k = (torch.softmax(self.theta, 0)[None, :, None] if a == "cg"
             else torch.softmax(self.theta, 1).T[None])
        p = (k * pi).sum(1).clamp(EPS, 1 - EPS)
        return torch.log(p) - torch.log1p(-p)
        

def summarise(z, y, m, st, n_pos):
    z = z.astype(np.float64); y64 = y.astype(np.float64)
    p = 1 / (1 + np.exp(-z))
    bce = np.logaddexp(0, z) - y64 * z
    half = np.zeros(n_pos, bool); half[n_pos // 2:] = True
    mh = m & half[:, None]
    fams = st.fam_names
    out = {"bce_half": {f: float(bce[mh & (st.fam == i)[None]].mean()) for i, f in enumerate(fams)},
           "bce_all": {f: float(bce[m & (st.fam == i)[None]].mean()) for i, f in enumerate(fams)}}
    for kind in ("twin", "solo", "base"):
        for D in (0, 64, 512):
            qs = (st.kind == kind) & (st.delay == D)
            if qs.any():
                out["bce_half"][f"{kind}{D}"] = float(bce[mh & qs[None]].mean())
    out["P1"] = float(np.mean([out["bce_half"][f] for f in LEARN]))
    t = np.arange(n_pos) + st.W
    v = violations(p, st, t=t, tol=0.05)
    out["viol"] = v
    out["P2"] = float(np.mean([v["pre:" + k]["mean"] for k in PROVE]))
    out["P3"] = v["twin"]["mean"]
    # twin violation by delay (live steps, second half)
    for D in (64, 512):
        qs = np.where((st.kind == "twin") & (st.delay == D))[0]
        vv = np.abs(p[:, qs] - p[:, st.ops[qs, 0]])[mh[:, qs]]
        out[f"twin_viol_D{D}_half"] = float(vv.mean())
    # post-resolution prove BCE (live steps with revealed history)
    pq = np.where(st.fam == fams.index("prove"))[0]
    post = m[:, pq] & ~st.unresolved_history(t)[:, pq]
    pre = m[:, pq] & st.unresolved_history(t)[:, pq]
    out["prove_bce_post"] = float(bce[:, pq][post].mean())
    out["prove_bce_pre"] = float(bce[:, pq][pre].mean())
    out["exp_excess_ln2"] = {f: out["bce_half"][f] - math.log(2) for f in ("exp64", "exp256", "exp1024")}
    return out


def main():
    arm, seed = sys.argv[1], int(sys.argv[2])
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    st = make_li_stream(seed, T); ps = st.as_positions()
    maker = Logistic(ps.Q, ps.W)
    model = Layer(ps.Q, ps.W, arm, maker)
    opt = (torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.1) if arm == "wd"
           else torch.optim.Adam(model.parameters(), 1e-3))
    out = run_prequential(model, ps, opt, B=4, baseline=maker,
                          opt_b=torch.optim.Adam(maker.parameters(), 1e-2))
    res = {"arm": arm, "seed": seed, "T": T, "Q": ps.Q, "families": list(st.fam_names), "steps": out["steps"]}
    res["net"] = summarise(out["z"], out["y"], out["mask"], st, ps.n_pos)
    res["maker"] = summarise(out["zb"], out["y"], out["mask"], st, ps.n_pos)
    if arm in ("cg", "cq"):
        k = torch.softmax(model.theta.detach(), -1).numpy()
        k = k[None] if k.ndim == 1 else k
        res["pool_eff_traders"] = float(np.mean(1 / (k ** 2).sum(-1)))
    res["wall_s"] = time.time() - t0
    rd = os.environ.get("OUT", os.path.join(HERE, "results")); os.makedirs(rd, exist_ok=True)
    path = os.path.join(rd, f"li-{arm}-s{seed}.json")
    json.dump(res, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
    print(path, f"{res['wall_s']:.0f}s P1={res['net']['P1']:.4f} P2={res['net']['P2']:.3f} P3={res['net']['P3']:.3f}")


if __name__ == "__main__":
    main()
