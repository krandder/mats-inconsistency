"""P5 phase-0 checks. Run: python tests.py [--quick] ; writes results/tests.json."""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import mazes
import p5

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}


def brute_paths(op, s, t):
    """All simple paths s -> t by exhaustive DFS (does not assume the maze is a tree)."""
    h, w, _ = op.shape
    out, path, seen = [], [s], {s}

    def rec(v):
        if v == t:
            out.append(list(path))
            return
        for d, (di, dj) in enumerate(mazes.DIRS):
            if op[v[0], v[1], d]:
                u = (v[0] + di, v[1] + dj)
                if u not in seen:
                    seen.add(u); path.append(u)
                    rec(u)
                    path.pop(); seen.discard(u)
    rec(s)
    return out


def test_bfs():
    rng = np.random.default_rng(0)
    bad = 0
    for k in range(200):
        h = 9 if k < 150 else 5
        op = mazes.dfs_maze(h, h, rng)
        x, y, m, d, e = mazes.make_example(op, rng)
        s = tuple(int(a) for a in np.argwhere(x[4])[0])
        t = tuple(int(a) for a in np.argwhere(x[5])[0])
        ps = brute_paths(op, s, t)
        tok = x[6]
        par = int(sum(tok[v] for v in ps[0][1:])) % 2 if len(ps) == 1 else -1
        onp = np.zeros((h, h)); [onp.__setitem__(v, 1) for v in ps[0][1:]] if len(ps) == 1 else None
        ok = len(ps) == 1 and len(ps[0]) - 1 == d and par == int(y[t]) and (onp == e[0]).all()
        # symmetric connectivity and full reachability (perfect maze = spanning tree: h*h-1 passages)
        ok = ok and op.sum() == 2 * (h * h - 1)
        bad += not ok
    R["bfs_vs_bruteforce"] = dict(mazes=200, failures=bad)
    assert bad == 0, bad


def test_memory_need():
    """Logistic regression on the 5x5 neighbourhood of T (tokens + walls) cannot predict the parity."""
    ops = mazes.load_train()
    rng = np.random.default_rng(3)

    def feats(n, min_d=0):
        X, Y = [], []
        while len(X) < n:
            op = ops[rng.integers(len(ops))]
            x, y, m, d, e = mazes.make_example(op, rng)
            if d <= min_d:
                continue
            t = np.argwhere(x[5])[0]
            xp = np.pad(x[[0, 1, 2, 3, 6]], ((0, 0), (2, 2), (2, 2)))
            X.append(xp[:, t[0]:t[0] + 5, t[1]:t[1] + 5].ravel())
            Y.append(y[t[0], t[1]])
        return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

    Xtr, Ytr = feats(40000)
    Xte, Yte = feats(10000, min_d=4)
    lin = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)

    def clo():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(lin(Xtr)[:, 0], Ytr) + 1e-4 * lin.weight.pow(2).sum()
        l.backward()
        return l
    opt.step(clo)
    with torch.no_grad():
        acc_te = float(((lin(Xte)[:, 0] > 0).float() == Yte).float().mean())
        acc_tr = float(((lin(Xtr)[:, 0] > 0).float() == Ytr).float().mean())
    R["memory_need_logistic"] = dict(train_acc=acc_tr, heldout_acc_dist_gt4=acc_te, n_train=40000, n_test=10000, bar=0.52)
    assert acc_te <= 0.52, acc_te


def test_pay():
    torch.manual_seed(0)
    worst, finite = 0.0, True
    for scale in (1e-7, 1e-5, 1e-3, 1.0, 5.0):
        b = torch.randn(1000, 1, dtype=torch.float64) * 3
        own = (torch.randn(1000, 4, dtype=torch.float64) * scale).requires_grad_(True)
        y = (torch.rand(1000, 1) < 0.5).double()
        S = own.sum(1, keepdim=True)
        pay = own * (y - p5.pbar_fn(b, S))
        err = (pay.sum(1) - p5.dscore(b[:, 0], S[:, 0], y[:, 0])).abs().max()
        worst = max(worst, float(err))
        pay.sum().backward()
        finite &= bool(torch.isfinite(own.grad).all())
    z = torch.zeros(10, 4, requires_grad=True)
    p = (z * (0.3 - p5.pbar_fn(torch.zeros(10, 1), z.sum(1, keepdim=True)))).sum()
    p.backward()
    finite &= bool(torch.isfinite(z.grad).all())
    R["pay_completeness"] = dict(max_err=worst, grads_finite=finite)
    assert worst < 1e-9 and finite


def test_ledger():
    torch.manual_seed(1)
    B, n = 256, 4
    led = p5.Ledger(B, n, "cpu")
    led.W = torch.rand(B, n, dtype=torch.float64) * 0.2  # small wealth so the caps bind
    led.G = led.W.clone()
    tot0 = led.W.sum(1).clone()
    sub = torch.zeros(B, dtype=torch.float64)
    for _ in range(20):
        db = torch.randn(B, n, 3, 3)
        s = led.caps(db)
        pay = s[:, :, None, None] * db * torch.randn(B, 1, 3, 3).sign() * 0.3
        led.belief(pay.sum((2, 3)))
        sub += pay.sum((1, 2, 3)).double()
        w = torch.tanh(torch.randn(B, n, n, p5.K, 3, 3))
        P = torch.randn(B, n, p5.K, 3, 3)
        led.royalties(w, P, 1.0, torch.ones(B, 3, 3))
    err = float((led.W.sum(1) - tot0 - sub).abs().max())
    s = led.summary()
    R["ledger_conservation"] = dict(total_change_minus_subsidy=err, step_err=s["cons_err"], royalties_scaled=s["roy_scaled"],
                                    capped=s["capped"])
    assert err < 1e-5 and s["roy_scaled"] > 0 and s["capped"] > 0


def small_batch(B=8, h=7, seed=0, dtype=torch.float64):
    rng = np.random.default_rng(seed)
    ops = np.stack([mazes.dfs_maze(h, h, rng) for _ in range(B)])
    x, y, m, d, e = p5.to_dev(mazes.make_batch(ops, rng, B), "cpu")
    return x.to(dtype), y.to(dtype), m.to(dtype), e.to(dtype)


def test_ig():
    torch.manual_seed(2)
    model = p5.Traders(4).double()
    x, y, m, e = small_batch()
    b = torch.randn(8, 1, 7, 7, dtype=torch.float64) * m[:, None]
    c = torch.randn(8, 4, p5.K, 7, 7, dtype=torch.float64)
    with torch.no_grad():
        db = model(x, b, c)[0] * m[:, None]
        own = db
        cx = p5.Ctx(y, m, b, torch.ones(8, 4, dtype=torch.float64), own.sum(1, keepdim=True) - own)
        f1 = p5.own_obj(*model(x, b, c)[:3], c, cx)
        f0 = p5.own_obj(*model(x, b, torch.zeros_like(c))[:3], torch.zeros_like(c), cx)
    out = {}
    for mm in (4, 64):
        g = p5.ig_grad(model, x, b, c, cx, p5.own_obj, mm)
        comp = (g * c).sum((2, 3, 4))
        out[f"m{mm}_rel_err"] = float((comp - (f1 - f0)).abs().sum() / (f1 - f0).abs().sum())
    R["ig_completeness"] = out


def test_grouped():
    torch.manual_seed(3)
    model = p5.Traders(4, head=True).double()
    x, y, m, e = small_batch()
    b = torch.randn(8, 1, 7, 7, dtype=torch.float64)
    c = torch.randn(8, 4, p5.K, 7, 7, dtype=torch.float64)
    db, w, g, P = model(x, b, c)
    err = 0.0
    ci, hc = 12, p5.HID
    for i in range(4):
        inp = torch.cat([x, torch.tanh(b / 4), torch.tanh(c[:, i])], 1)
        h1 = F.relu(F.conv2d(inp, model.c1.weight[i * hc:(i + 1) * hc], model.c1.bias[i * hc:(i + 1) * hc], padding=1))
        h2 = F.relu(F.conv2d(h1, model.c2.weight[i * hc:(i + 1) * hc], model.c2.bias[i * hc:(i + 1) * hc], padding=1))
        no = model.nout
        o = F.conv2d(h2, model.c3.weight[i * no:(i + 1) * no], model.c3.bias[i * no:(i + 1) * no])
        err = max(err, float((o[:, 0] - db[:, i]).abs().max()))
        err = max(err, float((torch.tanh(o[:, 1:1 + 4 * p5.K]).reshape(8, 4, p5.K, 7, 7) - w[:, i]).abs().max()))
        err = max(err, float((torch.sigmoid(o[:, 1 + 4 * p5.K:]) - g[:, i]).abs().max()))
        hp = F.conv2d(torch.cat([h2, torch.tanh(c[:, i])], 1), model.head.weight[i * p5.K:(i + 1) * p5.K],
                      model.head.bias[i * p5.K:(i + 1) * p5.K])
        err = max(err, float((hp - P[:, i]).abs().max()))
    # trader i's outputs do not depend on bank j != i
    c2 = c.clone(); c2[:, 1] += 1
    d2 = model(x, b, c2)[0]
    leak = float((d2[:, [0, 2, 3]] - db[:, [0, 2, 3]]).abs().max())
    R["grouped_conv_equals_loop"] = dict(max_err=err, cross_bank_leak=leak,
                                        params_per_trader=sum(p.numel() for p in model.body_params()) / 4,
                                        head_params_per_trader=sum(p.numel() for p in model.head.parameters()) / 4)
    assert err < 1e-10 and leak == 0


def grads(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.body_params()])


def test_adjoint():
    """A5 with N=1, pointwise, no caps == A2t autograd; A6 with the true (global pointwise) adjoints == A2t (N=4)."""
    torch.manual_seed(4)
    x, y, m, e = small_batch()
    D = 12
    ex = dict(phi=1.0, dense=False, ema=None, onpath=e[:, 0])
    out = {}
    m1 = p5.Traders(1).double()
    m1.zero_grad(); p5.step_A2t(m1, x, y, m, D); g_ref = grads(m1).clone()
    m1.zero_grad(); p5.step_A5(m1, "A10", x, y, m, D, ex, dict(wabs_self=0, wabs_all=0), pointwise=True); g5 = grads(m1).clone()
    out["A5_N1_pointwise_vs_A2t_max_abs"] = float((g5 - g_ref).abs().max())
    out["A2t_grad_norm"] = float(g_ref.norm())
    m4 = p5.Traders(4, head=True).double()
    m4.zero_grad(); p5.step_A2t(m4, x, y, m, D); g_ref4 = grads(m4).clone()
    lam = p5.step_A5(m4, "A10", x, y, m, D, ex, {}, pointwise=True, obj=p5.glob_obj, return_lams=True)
    m4.zero_grad(); p5.step_online(m4, "A6", x, y, m, D, ex, dict(wabs_self=0, wabs_all=0), true_adj=lam); g6 = grads(m4).clone()
    out["A6_true_adjoint_vs_A2t_max_abs"] = float((g6 - g_ref4).abs().max())
    # own-profit (A5 IG with N=4) vs A2t: cosine, informative only
    m4.zero_grad(); p5.step_A5(m4, "A10", x, y, m, D, ex, dict(wabs_self=0, wabs_all=0)); g5b = grads(m4)
    out["A5_N4_IG_cos_vs_A2t_random_init"] = float(F.cosine_similarity(g5b, g_ref4, dim=0))
    R["adjoint_equivalence"] = out
    assert out["A5_N1_pointwise_vs_A2t_max_abs"] < 1e-5 * max(1, out["A2t_grad_norm"])
    assert out["A6_true_adjoint_vs_A2t_max_abs"] < 1e-5 * max(1, out["A2t_grad_norm"])


def test_caps_label_free():
    """The training trajectory of b must not depend on y (else the ledger caps leak the label into b): with the
    same untrained weights, running a step on y and on 1 - y gives the same final b, so acc(y) + acc(1 - y) = 1."""
    res = {}
    for arm in ["A3", "A4", "A5", "A7", "A7p"]:
        accs = []
        for flip in (0, 1):
            torch.manual_seed(3)
            model = p5.build(arm).double()
            x, y, mask, e = small_batch(B=16, seed=3)
            if flip:
                y = 1 - y
            ex = dict(phi=1.0, dense=False, ema=None, onpath=e[:, 0])
            st = dict(wabs_self=0.0, wabs_all=0.0)
            step = p5.step_A5 if arm == "A5" else p5.step_online
            accs.append(step(model, arm, x, y, mask, 12, ex, st)["acc"])
        res[arm] = accs
        assert abs(accs[0] + accs[1] - 1) < 1e-9, (arm, accs)
    R["caps_label_free"] = res


def test_smoke():
    arms = ["A1", "A1w", "A2", "A2t", "A3", "A4", "A5", "A6", "A7", "A8", "A10", "A7p", "A7f"]
    out = {}
    for arm in arms:
        a = p5.parser().parse_args(["--arm", arm, "--iters", "20", "--D", "8", "--bs", "8", "--size", "7", "--cpu",
                                    "--cos_every", "10", "--log_every", "10"])
        t0 = time.time()
        _, res = p5.train(a)
        last = res["log"][-1]
        assert all(np.isfinite(v) for v in last.values() if isinstance(v, float)), (arm, last)
        out[arm] = dict(s_per_it=round(time.time() - t0, 3) / 20, params=res["params"], last=last)
    a = p5.parser().parse_args(["--arm", "A3", "--iters", "5", "--D", "8", "--bs", "8", "--size", "7", "--cpu", "--dense"])
    p5.train(a)
    R["smoke_7x7_20iter"] = out


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    tests = [test_bfs, test_pay, test_ledger, test_ig, test_grouped, test_adjoint, test_caps_label_free, test_smoke]
    if "--quick" not in sys.argv:
        tests.insert(1, test_memory_need)
    for f in tests:
        t0 = time.time()
        try:
            f()
            print(f.__name__, "ok", round(time.time() - t0, 1), "s", json.dumps(R.get(f.__name__.replace("test_", ""), ""))[:300], flush=True)
        except AssertionError as ex:
            print(f.__name__, "FAILED", ex, flush=True)
            R.setdefault("failed", []).append(f.__name__)
    meta = os.path.join(mazes.DATA, "meta.json")
    if os.path.exists(meta):
        mt = json.load(open(meta))
        R["st_distance_9x9"] = {k: mt[k] for k in ("L_train", "st_median", "st_p90", "st_max", "diam9_max", "diam9_mean", "st_hist")}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(R, open(os.path.join(HERE, "results", "tests.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in R.items() if k != "smoke_7x7_20iter"}, default=float)[:3000])
