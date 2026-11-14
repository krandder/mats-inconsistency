"""M6 stress s1: python run_s1.py REGIME ARM SEED [--epochs N]  (REGIME MS|IE; ARM A B WD BOOST CG CQ FB01 FB05).
See ../PREREG.md. Imports w1's m6layer.py unchanged."""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "substrate")); sys.path.insert(0, HERE)
from m6layer import MarketNet, masked_bce  # noqa: E402
import streams_ext, panel  # noqa: E402
from prequential import assert_disjoint  # noqa: E402

FAMN = list(streams_ext.FAMILIES[:8]) + ["indep"]   # id 8 (periodic_ho, unused with heldout=0) -> indep
LEARN = ["periodic", "lfsr16", "ca110", "prime", "and", "or"]
NOISE = ["prng", "ca30"]
LN2 = math.log(2)
W, T = 32, 24032
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class S1Net(MarketNet):
    def __init__(self, Q, W, arm, **kw):
        base = {"A": "A", "WD": "A", "B": "B", "BOOST": "BOOST"}.get(arm, "A")
        super().__init__(Q, W, arm=base, **kw)
        self.s1arm = arm
        if arm == "CG":
            self.theta = nn.Parameter(torch.zeros(self.traders.n))
        elif arm == "CQ":
            self.theta = nn.Parameter(torch.zeros(Q, self.traders.n))

    def combine(self, l0d, d, out):
        a = self.s1arm
        if a in ("CG", "CQ"):
            k = torch.softmax(self.theta, -1)
            k = k.view(1, 1, -1) if a == "CG" else k.unsqueeze(0)
            p = (k * torch.sigmoid(l0d.unsqueeze(-1) + d)).sum(-1).clamp(1e-6, 1 - 1e-6)
            return torch.log(p) - torch.log1p(-p)
        if a == "FB01":
            return l0d + (0.1 * torch.tanh(d / 0.1)).sum(-1)
        if a == "FB05":
            return l0d + (0.5 * torch.tanh(d / 0.5)).sum(-1)
        return super().combine(l0d, d, out)


def make_stream(regime, seed):
    s = streams_ext.make_streams(64, T, seed, heldout=0, regime_every=0, W=W)
    ptrue = None
    if regime == "IE":
        rng = np.random.default_rng([seed, 4242])
        prng_q = [q for q in range(len(s.fam0)) if s.fam0[q] == 5]
        new_bits, new_desc, ptrue = [], [], []
        for j in range(8):
            src = rng.choice(prng_q, size=3, replace=False)
            ev = sum(2.0 * s.bits[a].astype(np.float64) - 1.0 for a in src)      # [T]
            y = np.zeros(T, np.uint8); p = np.full(T, 0.5)
            u = rng.random(T)
            for t in range(1, T):
                L = (2.0 * y[t - 1] - 1.0) + ev[t - 1]
                p[t] = 1.0 / (1.0 + math.exp(-L)); y[t] = u[t] < p[t]
            new_bits.append(y); ptrue.append(p)
            new_desc.append("indep(own lag + " + ", ".join(f"q{a}" for a in src) + " lag1)")
        s.bits = np.concatenate([s.bits, np.stack(new_bits)])
        s.fam0 = np.concatenate([s.fam0, np.full(8, 8)]); s.fam = s.fam0
        s.active = np.ones(s.bits.shape, bool)
        s.desc = list(s.desc) + new_desc
        s.Q = s.n_main = s.bits.shape[0]
        ptrue = np.stack(ptrue)
    return s, ptrue


def constraints(s):
    cons = []
    for q, d in enumerate(s.desc):
        m = re.match(r"(and|or)\(q(\d+) \S+, q(\d+) \S+\)", d)
        if m:
            cons.append((m.group(1), q, int(m.group(2)), int(m.group(3))))
    return cons


def bce_np(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(z, y, m, fam, cons, pt=None, idx=None):
    p = 1 / (1 + np.exp(-z.astype(np.float64)))
    F = np.broadcast_to(fam[None, :], z.shape)
    sc = panel.scores(p[m], y[m], F[m], names=FAMN)
    fb = {f: sc[f]["bce"] for f in FAMN if f in sc}
    out = {"fam": sc, "learn_bce": float(np.mean([fb[f] for f in LEARN])),
           "humility": float(max(fb[f] - LN2 for f in NOISE)), "lfsr_learnt": int(fb["lfsr16"] < 0.35)}
    v = panel.violations(p, cons, tol=0.01)
    out["viol_all"] = v["all"]["mean"]
    if pt is not None:
        qi = np.where(fam == 8)[0]
        pb = pt[:, idx + W].T                 # [n, 8] true p at the label times
        out["ie_bce"] = fb["indep"]; out["ie_bayes"] = bce_np(pb, y[:, qi])
        out["ie_regret"] = out["ie_bce"] - out["ie_bayes"]
    return out


def evalP(model, st, P, chunk=600):
    zs, zl = [], []
    model.eval()
    with torch.no_grad():
        for k in range(0, len(P), chunk):
            out = model(st.window(P[k:k + chunk]).to(DEV)); zs.append(out["z"].cpu().numpy()); zl.append(out["l0"].cpu().numpy())
    return np.concatenate(zs), np.concatenate(zl), st.label(P).numpy(), st.label_mask(P).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regime"); ap.add_argument("arm"); ap.add_argument("seed", type=int)
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--out", default=os.environ.get("OUT", "."))
    a = ap.parse_args()
    regime, arm, seed = a.regime, a.arm, a.seed
    torch.manual_seed(seed); np.random.seed(seed)
    s, pt = make_stream(regime, seed)
    st = streams_ext.to_array_stream(s)
    Q = st.Q
    fam, cons = np.asarray(s.fam0), constraints(s)
    assert len(cons) == 16, len(cons)
    model = S1Net(Q, W, arm).to(DEV)
    theta = [p for n, p in model.named_parameters() if n == "theta"]
    rest = [p for p in model.net_params() if all(p is not t for t in theta)]
    groups = [{"params": rest, "lr": 1e-3}] + ([{"params": theta, "lr": 1e-2}] if theta else [])
    o = torch.optim.AdamW(groups, weight_decay=0.1) if arm == "WD" else torch.optim.Adam(groups)
    om = torch.optim.Adam(model.market.parameters(), 1e-2)
    if regime == "MS":
        tr = np.arange(0, 2400); E, every = a.epochs or 100, 5
    else:
        tr = np.arange(0, 16800); E, every = a.epochs or 20, 1
    va, te = np.arange(16800, 19200), np.arange(19200, st.n_pos)
    assert_disjoint(st, tr, va); assert_disjoint(st, tr, te)
    g = np.random.default_rng(seed + 7919)
    tag = f"{regime}-{arm}-s{seed}"
    t0 = time.time(); curve = []
    for e in range(E + 1):
        if e % every == 0 or e == E:
            zv, _, yv, mv = evalP(model, st, va)
            zt, zlt, yt, mt = evalP(model, st, te)
            rv = metrics(zv, yv, mv, fam, cons, pt, va)
            rt = metrics(zt, yt, mt, fam, cons, pt, te)
            ztr, _, ytr, mtr = evalP(model, st, tr[:2400])
            rtr = metrics(ztr, ytr, mtr, fam, cons, pt, tr[:2400])
            row = {"epoch": e, "val_bce": rv["fam"]["all"]["bce"], "test": rt,
                   "train_learn": rtr["learn_bce"], "train_humility": rtr["humility"]}
            if e in (0, E):
                row["test_L0"] = metrics(zlt, yt, mt, fam, cons, pt, te)
            if arm in ("CG", "CQ"):
                k = torch.softmax(model.theta.detach(), -1).cpu()
                row["k_eff"] = float(torch.exp(-(k * torch.log(k + 1e-12)).sum(-1)).mean())
            curve.append(row)
            print(tag, e, round(row["val_bce"], 4), round(rt["learn_bce"], 4), round(rt["humility"], 4),
                  round(rt.get("ie_regret", float("nan")), 4), round(time.time() - t0), flush=True)
        if e == E:
            break
        model.train()
        perm = g.permutation(tr)
        for k in range(0, len(perm), 128):
            P = perm[k:k + 128]
            win, y, m = st.window(P).to(DEV), st.label(P).to(DEV), st.label_mask(P).float().to(DEV)
            out = model(win)
            loss = masked_bce(out["z"], y, m) + masked_bce(out["l0"], y, m)
            o.zero_grad(set_to_none=True); om.zero_grad(set_to_none=True)
            loss.backward(); o.step(); om.step()
    best = min(range(len(curve)), key=lambda i: curve[i]["val_bce"])
    res = {"regime": regime, "arm": arm, "seed": seed, "Q": Q, "epochs": E, "dev": DEV,
           "score": curve[-1]["test"], "score_L0": curve[-1]["test_L0"], "es_epoch": curve[best]["epoch"],
           "score_es": curve[best]["test"], "wall_s": time.time() - t0,
           "curve": [{k: (v if k != "test" else {kk: v.get(kk) for kk in ("learn_bce", "humility", "viol_all", "ie_regret")})
                      for k, v in r.items() if k != "test_L0"} for r in curve]}
    fn = os.path.join(a.out, f"{tag}.json")
    json.dump(res, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
    print(tag, "done", round(res["wall_s"]), res["score"]["learn_bce"], res["score"]["humility"])


if __name__ == "__main__":
    main()
