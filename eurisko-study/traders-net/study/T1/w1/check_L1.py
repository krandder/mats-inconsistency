"""Numerical checks for T1 lemma 1 (books identity, norm growth, weight-decay rent, zero-mean tangent).

Run: PYTHONPATH=<wealth-net>/.pylib python3 check_L1.py [a|b|c|d|all]
Writes out/L1{a,b,c,d}.json. float64 throughout. Seeds 1,2,3.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Net          # copied from wealth-net, unchanged
from streams import make_questions

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SEEDS = [1, 2, 3]


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def msd(xs):
    xs = np.asarray(xs, dtype=float)
    return [float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0]


# ----------------------------------------------------------------------------- toy data
def toy_data(seed, Q=16, W=8, T=3000):
    bits, fam, _ = make_questions(Q, T, seed)
    b = torch.tensor(bits, dtype=torch.float64)
    wins = b.unfold(1, W, 1)                     # [Q, T-W+1, W]
    n = T - W
    return wins, b, n


def get_batch(wins, b, idx, W):
    return wins[:, idx].permute(1, 0, 2).contiguous(), b[:, idx + W].T.contiguous()


# ----------------------------------------------------------------------------- unit parameter views
def unit_thetas(model):
    """Per unit: list of (name, [param tensors or slices as callables]).
    Returns dict kind -> list of functions f(model_or_grads) -> flat vector.
    Out-side and in-side parameter groups for each homogeneous unit."""
    L, H, d = model.L, model.H, model.d
    dh = d // H
    N = model.blocks[0].fc1.out_features
    units = {"readout": [], "neuron_out": [], "neuron_in": [], "head_out": [], "head_in": []}
    Q = model.Q
    for q in range(Q):
        units["readout"].append((f"q{q}", [("rw", (q, slice(None))), ("rb", (q,))]))
    for l in range(L):
        for n in range(N):
            units["neuron_out"].append((f"L{l}n{n}", [(f"blocks.{l}.fc2.weight", (slice(None), n))]))
            units["neuron_in"].append((f"L{l}n{n}", [(f"blocks.{l}.fc1.weight", (n, slice(None))),
                                                    (f"blocks.{l}.fc1.bias", (n,))]))
        for h in range(H):
            units["head_out"].append((f"L{l}h{h}", [(f"blocks.{l}.o.weight", (slice(None), slice(h * dh, (h + 1) * dh)))]))
            rows = slice(2 * d + h * dh, 2 * d + (h + 1) * dh)   # V rows of qkv (q,k,v stacked)
            units["head_in"].append((f"L{l}h{h}", [(f"blocks.{l}.qkv.weight", (rows, slice(None))),
                                                  (f"blocks.{l}.qkv.bias", (rows,))]))
    return units


def gather(tensors, spec):
    return torch.cat([tensors[name][ix].reshape(-1) for name, ix in spec])


def dot_per_unit(units, A, B):
    """sum over each unit's parameters of A*B (A,B: dict name->tensor). Vectorised; `units` only
    fixes the key set (layout as in unit_thetas)."""
    P = {n: A[n] * B[n] for n in A}
    L = sum(1 for n in P if n.endswith("fc1.bias"))
    out = {"readout": (P["rw"].sum(1) + P["rb"]).numpy()}
    no, ni, ho, hi = [], [], [], []
    for l in range(L):
        pre = f"blocks.{l}."
        no.append(P[pre + "fc2.weight"].sum(0))
        ni.append(P[pre + "fc1.weight"].sum(1) + P[pre + "fc1.bias"])
        d = P[pre + "o.weight"].shape[0]
        Hh = len(units["head_out"]) // L
        dh = d // Hh
        ho.append(P[pre + "o.weight"].view(d, Hh, dh).sum((0, 2)))
        hi.append(P[pre + "qkv.weight"].view(3, Hh, dh, d)[2].sum((1, 2)) + P[pre + "qkv.bias"].view(3, Hh, dh)[2].sum(1))
    out["neuron_out"] = torch.cat(no).numpy(); out["neuron_in"] = torch.cat(ni).numpy()
    out["head_out"] = torch.cat(ho).numpy(); out["head_in"] = torch.cat(hi).numpy()
    return out


def dot_per_unit_slow(units, A, B):
    return {k: np.array([float((gather(A, s) * gather(B, s)).sum()) for _, s in v]) for k, v in units.items()}


def sq_per_unit(units, A):
    return dot_per_unit(units, A, A)


def loss_fn(model, win, y, freeze_ln):
    z = model(win, freeze_ln=freeze_ln)
    return F.binary_cross_entropy_with_logits(z, y)   # mean over B*Q, as wealth-net


def books_income(model, win, y):
    """Books income per unit, wealth-net convention (frozen LN, live attention) but normalised to
    the mean loss (divide by Q): income = a . dR/da with R = mean_{b,q} z*s, s = y - p detached."""
    rec = {}
    z = model(win, rec=rec, freeze_ln=True)
    s = (y - torch.sigmoid(z)).detach()
    R = (z * s).mean()
    tg = list(rec["act"]) + list(rec["hv"])
    g = torch.autograd.grad(R, tg)
    L = model.L
    out = {"readout": (z.detach() * s).mean(0).numpy() / model.Q}
    out["neuron"] = np.concatenate([(rec["act"][l].detach() * g[l]).sum((0, 1)).numpy() for l in range(L)])
    out["head"] = np.concatenate([(rec["hv"][l].detach() * g[L + l]).sum((0, 2, 3)).numpy() for l in range(L)])
    return out


def param_grads(model, win, y, freeze_ln):
    model.zero_grad()
    loss = loss_fn(model, win, y, freeze_ln)
    gs = torch.autograd.grad(loss, list(model.parameters()))
    return {n: g.detach() for (n, _), g in zip(model.named_parameters(), gs)}


def params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ----------------------------------------------------------------------------- L1a Euler identity
def check_a():
    res = {"per_seed": {}}
    for seed in SEEDS:
        torch.manual_seed(seed)
        wins, b, n = toy_data(seed)
        model = Net(16, 8, d=16, H=2, L=2)
        # a few Adam steps so weights are not at init
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(200):
            idx = torch.randint(0, int(n * 0.8), (64,), generator=gen)
            win, y = get_batch(wins, b, idx, 8)
            opt.zero_grad(); loss_fn(model, win, y, False).backward(); opt.step()
        idx = torch.randint(0, int(n * 0.8), (64,), generator=gen)
        win, y = get_batch(wins, b, idx, 8)
        inc = books_income(model, win, y)
        P = params(model)
        g_fr = param_grads(model, win, y, True)
        g_lv = param_grads(model, win, y, False)
        U = unit_thetas(model)
        m_fr = {k: -v for k, v in dot_per_unit(U, P, g_fr).items()}
        m_lv = {k: -v for k, v in dot_per_unit(U, P, g_lv).items()}
        pairs = {"readout": "readout", "neuron_out": "neuron", "neuron_in": "neuron",
                 "head_out": "head", "head_in": "head"}
        r = {}
        for k, bk in pairs.items():
            a_, f_, l_ = inc[bk], m_fr[k], m_lv[k]
            scale = np.abs(a_).max()
            r[k] = {"max_rel_err_frozen_graph": float(np.abs(a_ - f_).max() / scale),
                    "max_rel_gap_live_LN": float(np.abs(a_ - l_).max() / scale),
                    "median_rel_gap_live_LN": float(np.median(np.abs(a_ - l_) / (np.abs(a_) + 1e-12))),
                    "pearson_books_vs_live": float(np.corrcoef(a_, l_)[0, 1]),
                    "n_units": int(len(a_))}
        res["per_seed"][seed] = r
    worst = max(v["max_rel_err_frozen_graph"] for s in res["per_seed"].values() for v in s.values())
    res["worst_rel_err_frozen_graph"] = worst
    res["pass_threshold"] = 1e-9
    res["pass"] = bool(worst < 1e-9)
    dump("L1a.json", res)
    print("L1a worst", worst)


# ----------------------------------------------------------------------------- L1b norm growth
def check_b(steps=2000):
    arms = {"sgd": dict(opt="sgd", lr=0.1, mom=0.0), "sgd_mom": dict(opt="sgd", lr=0.01, mom=0.9),
            "adam": dict(opt="adam", lr=1e-3, mom=0.0)}
    res = {"steps": steps, "arms": {}}
    for arm, cfg in arms.items():
        per = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            wins, b, n = toy_data(seed)
            model = Net(16, 8, d=16, H=2, L=2)
            if cfg["opt"] == "sgd":
                opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["mom"])
            else:
                opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            U = unit_thetas(model)
            P0 = params(model)
            Wtrue = {k: np.zeros(len(v)) for k, v in U.items()}
            G2 = {k: np.zeros(len(v)) for k, v in U.items()}
            Wbooks = {"readout": 0, "neuron": 0, "head": 0}
            step_resid = []
            gen = torch.Generator().manual_seed(seed + 7)
            eta = cfg["lr"]
            for t in range(steps):
                idx = torch.randint(0, int(n * 0.8), (64,), generator=gen)
                win, y = get_batch(wins, b, idx, 8)
                inc = books_income(model, win, y)
                for k in Wbooks:
                    Wbooks[k] = Wbooks[k] + inc[k]
                Pt = params(model)
                opt.zero_grad()
                loss = loss_fn(model, win, y, False)
                loss.backward()
                g = {nm: p.grad.detach().clone() for nm, p in model.named_parameters()}
                inc_t = {k: -v for k, v in dot_per_unit(U, Pt, g).items()}
                g2 = sq_per_unit(U, g)
                opt.step()
                for k in U:
                    Wtrue[k] += inc_t[k]
                    G2[k] += g2[k]
                if cfg["opt"] == "sgd" and cfg["mom"] == 0 and t % 100 == 0:
                    P1 = params(model)
                    dn = {k: sq_per_unit(U, P1)[k] - sq_per_unit(U, Pt)[k] for k in ["neuron_out"]}
                    pred = (dn["neuron_out"] - eta ** 2 * g2["neuron_out"]) / (2 * eta)
                    step_resid.append(float(np.abs(pred - inc_t["neuron_out"]).max() /
                                            (np.abs(inc_t["neuron_out"]).max() + 1e-300)))
            PT = params(model)
            n0, nT = sq_per_unit(U, P0), sq_per_unit(U, PT)
            r = {}
            pairs = {"readout": "readout", "neuron_out": "neuron", "neuron_in": "neuron", "head_out": "head", "head_in": "head"}
            for k, bk in pairs.items():
                dN = (nT[k] - n0[k]) / (2 * eta)
                r[k] = {"R2_Wtrue_vs_dnorm": float(np.corrcoef(Wtrue[k], dN)[0, 1] ** 2),
                        "R2_Wbooks_vs_dnorm": float(np.corrcoef(Wbooks[bk], dN)[0, 1] ** 2),
                        "R2_Wbooks_vs_Wtrue": float(np.corrcoef(Wbooks[bk], Wtrue[k])[0, 1] ** 2),
                        "sgd_exact_resid_rel": float(np.abs(Wtrue[k] - (dN - eta / 2 * G2[k])).max() /
                                                     (np.abs(Wtrue[k]).max() + 1e-300)),
                        "second_order_share": float(np.abs(eta / 2 * G2[k]).sum() / (np.abs(dN).sum() + 1e-300))}
            if step_resid:
                r["per_step_identity_max_rel_resid"] = max(step_resid)
            per.append(r)
            print(arm, seed, {k: round(v["R2_Wbooks_vs_dnorm"], 3) for k, v in r.items() if isinstance(v, dict)}, flush=True)
        agg = {}
        for k in per[0]:
            if isinstance(per[0][k], dict):
                agg[k] = {m: msd([p[k][m] for p in per]) for m in per[0][k]}
            else:
                agg[k] = msd([p[k] for p in per])
        res["arms"][arm] = {"cfg": cfg, "mean_sd": agg, "per_seed": per}
    dump("L1b.json", res)


# ----------------------------------------------------------------------------- L1c weight-decay rent
def mlp_toy(seed, n=2000, din=10, hid=32):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, din, generator=g)
    A = torch.randn(din, 8, generator=g)
    f = torch.tanh(X @ A).sum(1) * 0.5
    y = torch.bernoulli(torch.sigmoid(f), generator=g)
    W1 = torch.randn(hid, din, generator=g) / math.sqrt(din)
    b1 = torch.zeros(hid)
    w2 = torch.randn(hid, generator=g) / math.sqrt(hid)
    return X, y, W1, b1, w2, g


def check_c(steps=30000, eta=0.05, B=64):
    res = {"steps": steps, "eta": eta, "B": B, "lams": {}}
    for lam in [0.0, 1e-3, 1e-2]:
        per = []
        for seed in SEEDS:
            X, y, W1, b1, w2, g = mlp_toy(seed)
            W1.requires_grad_(); b1.requires_grad_(); w2.requires_grad_()
            hid = w2.numel()
            acc = {k: np.zeros(hid) for k in ["inc", "rent", "noise", "norm2"]}
            cnt = 0
            nsq_start = None
            for t in range(steps):
                idx = torch.randint(0, X.shape[0], (B,), generator=g)
                h = F.relu(X[idx] @ W1.T + b1)
                z = h @ w2
                loss = F.binary_cross_entropy_with_logits(z, y[idx])
                gW1, gb1, gw2 = torch.autograd.grad(loss, [W1, b1, w2])
                with torch.no_grad():
                    inc = -(w2 * gw2).numpy()                     # out-side income (= in-side, Euler)
                    gt = gw2 + lam * w2
                    if t >= steps // 2:
                        if nsq_start is None:
                            nsq_start = (w2 ** 2).numpy().copy()
                        acc["inc"] += inc
                        acc["rent"] += (lam * w2 ** 2).numpy()
                        acc["noise"] += (eta / 2 * gt ** 2).numpy()
                        acc["norm2"] += (w2 ** 2).numpy()
                        cnt += 1
                    W1 -= eta * (gW1 + lam * W1); b1 -= eta * (gb1 + lam * b1); w2 -= eta * gt
            nsq_end = (w2.detach() ** 2).numpy()
            m = {k: v / cnt for k, v in acc.items()}
            drift = (nsq_end - nsq_start) / (2 * eta * cnt)
            ident = m["inc"] - (drift + m["rent"] - m["noise"])
            from scipy.stats import spearmanr
            alive = m["norm2"] > 1e-6
            r = {"mean_income": float(m["inc"].mean()), "mean_rent": float(m["rent"].mean()),
                 "mean_noise_tax": float(m["noise"].mean()), "mean_drift": float(drift.mean()),
                 "identity_max_abs_resid": float(np.abs(ident).max()),
                 "frac_alive_income_pos": float((m["inc"][alive] > 0).mean()),
                 "spearman_income_norm2": float(spearmanr(m["inc"], m["norm2"])[0]),
                 "rent_over_income_median": float(np.median(m["rent"][alive] / np.where(np.abs(m["inc"][alive]) > 0, m["inc"][alive], np.nan)))}
            per.append(r)
            print("L1c", lam, seed, {k: round(v, 5) for k, v in r.items()}, flush=True)
        res["lams"][str(lam)] = {"mean_sd": {k: msd([p[k] for p in per]) for k in per[0]}, "per_seed": per}
    dump("L1c.json", res)


# ----------------------------------------------------------------------------- L1d zero-mean tangent
def check_d():
    res = {"calibrated": {}, "overfit": {}}
    # (i) predictor = true posterior; units = coordinates c_i = w_i x_i, plus bias
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n, din = 200000, 5
        w = rng.normal(size=din); bias = -0.5
        X = rng.normal(size=(n, din))
        zz = X @ w + bias
        p = 1 / (1 + np.exp(-zz))
        y = (rng.random(n) < p).astype(float)
        c = np.concatenate([X * w, np.full((n, 1), bias)], 1)
        inc = c * (y - p)[:, None]
        mean, se = inc.mean(0), inc.std(0, ddof=1) / math.sqrt(n)
        tot = (zz * (y - p))
        res["calibrated"][seed] = {"unit_mean": mean.tolist(), "unit_se": se.tolist(),
                                   "max_abs_z": float(np.abs(mean / se).max()),
                                   "total_mean": float(tot.mean()), "total_se": float(tot.std(ddof=1) / math.sqrt(n))}
    res["calibrated_pass"] = bool(all(v["max_abs_z"] < 3.5 for v in res["calibrated"].values()))
    # (ii) overfit logistic regression: 5 signal + 95 noise features, n_train=200, L2 lam
    lam = 1e-3
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        din, ns = 100, 5
        wstar = torch.zeros(din); wstar[:ns] = 1.0
        Xtr = torch.randn(200, din, generator=g); ytr = torch.bernoulli(torch.sigmoid(Xtr @ wstar), generator=g)
        Xte = torch.randn(100000, din, generator=g); yte = torch.bernoulli(torch.sigmoid(Xte @ wstar), generator=g)
        wv = torch.zeros(din, requires_grad=True)
        opt = torch.optim.LBFGS([wv], lr=1, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50)

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(Xtr @ wv, ytr) + lam / 2 * (wv ** 2).sum()
            l.backward(); return l
        for _ in range(5):
            opt.step(closure)
        with torch.no_grad():
            s_tr = ytr - torch.sigmoid(Xtr @ wv)
            inc_tr = (Xtr * wv * s_tr[:, None]).mean(0).numpy()        # = -w_i dL/dw_i (data part)
            s_te = yte - torch.sigmoid(Xte @ wv)
            inc_te = (Xte * wv * s_te[:, None]).mean(0).numpy()
            rent = (lam * wv ** 2).numpy()
            test_bce = float(F.binary_cross_entropy_with_logits(Xte @ wv, yte))
        res["overfit"][seed] = {
            "insample_income_minus_rent_maxabs": float(np.abs(inc_tr - rent).max()),
            "insample_income_signal_mean": float(inc_tr[:ns].mean()), "insample_income_noise_mean": float(inc_tr[ns:].mean()),
            "insample_frac_noise_pos": float((inc_tr[ns:] > 0).mean()),
            "preq_income_signal_mean": float(inc_te[:ns].mean()), "preq_income_noise_mean": float(inc_te[ns:].mean()),
            "preq_frac_noise_neg": float((inc_te[ns:] < 0).mean()), "preq_frac_signal_pos": float((inc_te[:ns] > 0).mean()),
            "test_bce": test_bce}
        print("L1d overfit", seed, res["overfit"][seed], flush=True)
    keys = res["overfit"][SEEDS[0]].keys()
    res["overfit_mean_sd"] = {k: msd([res["overfit"][s][k] for s in SEEDS]) for k in keys}
    dump("L1d.json", res)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    for k, f in [("a", check_a), ("b", check_b), ("c", check_c), ("d", check_d)]:
        if which in ("all", k):
            f()
            print(k, "done", round(time.time() - t0, 1), "s", flush=True)


def _selftest_vectorised():
    torch.manual_seed(0)
    m = Net(16, 8, d=16, H=2, L=2)
    P = params(m); G = {n: torch.randn_like(p) for n, p in P.items()}
    U = unit_thetas(m)
    f, s = dot_per_unit(U, P, G), dot_per_unit_slow(U, P, G)
    return max(float(np.abs(f[k] - s[k]).max()) for k in f)
