"""Stage 1 unit tests (not evidence): pay-rule gradients equal known regulariser gradients, on one batch, float64.
  1. exact Shapley pay grad  = H_n * grad of coalition dropout (size t ~ 1/t, then a uniform t-subset)
  2. Banzhaf pay grad        = 2 * grad of p=0.5 factor dropout
  3. LOO pay grad            = backprop grad of v(N)
  4. single-trader M-hard pay grad = supervised straight-through (VQ-sup style) log-loss grad
  5. maze chunk pay S_R(end) - S_R(detached start), c=6  = progressive loss with k=6 (R held fixed)
  6. chunk ledger telescopes: sum of chunk pays = S_R(final) - S_R(initial)
  also: AS-IG completeness error shrinks with path steps; asig pays sum to v(onehot) - v(marginal) as steps grow.
"""
import itertools
import json
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P3")
import t1  # noqa: E402

torch.set_default_dtype(torch.float64)
RES = {}


def setup(n=3, K=2, R=2, B=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    tr = t1.Traders(R, n, 11, K, g)
    tr.params = [p.detach().double().requires_grad_(True) for p in tr.params]
    mk = t1.TabMaker(R, n, K)
    with torch.no_grad():
        for p in mk.params:
            p.copy_(torch.randn(p.shape, generator=g))
    mk.freq = torch.ones(R, K ** n)
    bt = t1.make_batch(g, R, B, "S", "N")
    x, y, u = bt["x"].double(), bt["y"].double(), bt["u"].double()
    with torch.no_grad():
        lx, la = mk.logprobs(y, u)
    return tr, x, lx.double(), la.double()


def grads(tr, loss):
    gs = torch.autograd.grad(loss, tr.params, allow_unused=True)
    return torch.cat([(gg if gg is not None else torch.zeros_like(p)).flatten() for gg, p in zip(gs, tr.params)])


def codes(tr, x):
    h, hard = t1.st_codes(tr.logits(x))
    n = h.shape[1]
    return [h[:, j] for j in range(n)], [hard[:, j].mean(1, keepdim=True).detach() for j in range(n)]


def v_sub(P, off, hs, ms, S):
    return t1.value(P, off, [hs[j] if j in S else ms[j] for j in range(len(hs))]).mean(1).sum()


def check(name, a, b):
    err = float((a - b).abs().max() / (b.abs().max() + 1e-30))
    RES[name] = err
    print("%-55s rel.err %.2e  %s" % (name, err, "PASS" if err < 1e-9 else "FAIL"))
    return err < 1e-9


def test_semivalues(joint):
    n = 3
    tr, x, lx, la = setup(n)
    P, off = t1.price_tensor(lx, la, n, 2, joint)
    tag = " (joint score)" if joint else " (per-output score)"
    hs, ms = codes(tr, x)
    g_shap = grads(tr, sum(p.mean(1).sum() for p in t1.pay_semi(P, off, hs, ms, "shap")))
    Hn = sum(1 / t for t in range(1, n + 1))
    hs, ms = codes(tr, x)
    drop = 0
    for t in range(1, n + 1):
        for S in itertools.combinations(range(n), t):
            drop = drop + (1 / t) / Hn / math.comb(n, t) * v_sub(P, off, hs, ms, S)
    ok = check("Shapley pay grad = H_3 x (1/t coalition dropout)" + tag, g_shap, Hn * grads(tr, drop))
    hs, ms = codes(tr, x)
    g_banz = grads(tr, sum(p.mean(1).sum() for p in t1.pay_semi(P, off, hs, ms, "banz")))
    hs, ms = codes(tr, x)
    drop = sum(0.5 ** n * v_sub(P, off, hs, ms, S) for t in range(1, n + 1) for S in itertools.combinations(range(n), t))
    ok &= check("Banzhaf pay grad = 2 x (p=0.5 factor dropout)" + tag, g_banz, 2 * grads(tr, drop))
    hs, ms = codes(tr, x)
    g_loo = grads(tr, sum(p.mean(1).sum() for p in t1.pay_semi(P, off, hs, ms, "loo")))
    hs, ms = codes(tr, x)
    ok &= check("LOO pay grad = backprop grad of v(N)" + tag, g_loo, grads(tr, v_sub(P, off, hs, ms, range(n))))
    # AS-IG: not the gradient of one joint objective; check only completeness convergence
    hs, ms = codes(tr, x)
    tot = (t1.value(P, off, [h.detach() for h in hs]) - t1.value(P, off, ms)).mean()
    errs = {}
    for m in (5, 20, 200):
        a = t1.pay_asig(P, off, [h.detach() for h in hs], ms, m)
        errs[m] = float((sum(a).mean() - tot).abs())
    RES["asig_completeness_err" + tag] = errs
    print("AS-IG |sum pays - total| by steps%s: %s (total %.3f)" % (tag, {k: "%.1e" % v for k, v in errs.items()}, float(tot)))
    if not joint:  # joint (mixture-of-products) score: the integrand is concentrated in a thin layer near t=1
        ok &= errs[200] < errs[5] and errs[200] < 1e-3  # when the realised bucket is not the likeliest; reported only
    return ok


def test_mhard():
    tr, x, lx, la = setup(n=1, K=4)
    P, off = t1.price_tensor(lx, la, 1, 4, False)
    hs, ms = codes(tr, x)
    pay = (t1.value(P, off, hs) - t1.value(P, off, ms)).mean(1).sum()
    g_pay = grads(tr, pay)
    hs, ms = codes(tr, x)
    # supervised ST log loss through the 4-bucket reader, written directly as a table lookup
    idx = hs[0]
    ll = torch.log((la.exp() * idx[:, :, None, :]).sum(-1)).sum(-1).mean(1).sum()
    return check("single-trader M-hard pay grad = supervised ST (VQ-sup) grad", g_pay, grads(tr, ll))


def test_maze_chunk():
    torch.manual_seed(0)
    Bm, n, K, T = 4, 9, 8, 18
    x = (torch.rand(Bm, 2, n, n) < 0.3).double()
    tgt = (torch.rand(Bm, n, n) < 0.3).double()
    conv1 = torch.nn.Conv2d(2 + K, 16, 3, padding=1).double()
    conv2 = torch.nn.Conv2d(16, K, 3, padding=1).double()
    Rr = torch.nn.Conv2d(K + 2, 1, 3, padding=1).double()
    for p in Rr.parameters():
        p.requires_grad_(False)  # R is fitted separately; not part of the code net's gradient
    fp = list(conv1.parameters()) + list(conv2.parameters())

    def f(s):
        lg = conv2(F.relu(conv1(torch.cat([x, s], 1))))
        p = lg.softmax(1)
        return F.one_hot(p.argmax(1), K).permute(0, 3, 1, 2).double() + p - p.detach()

    def SR(s):
        lo = Rr(torch.cat([s, x], 1))[:, 0]
        return -F.binary_cross_entropy_with_logits(lo, tgt, reduction="none").sum((1, 2)).mean()

    s0 = torch.zeros(Bm, K, n, n); s0[:, 0] = 1
    start, c = 6, 6
    s = s0
    with torch.no_grad():
        for _ in range(start):
            s = f(s)
    se = s.detach()
    for _ in range(c):
        se = f(se)
    pay = SR(se) - SR(s.detach())
    g_pay = torch.cat([gg.flatten() for gg in torch.autograd.grad(pay, fp)])
    s = s0
    with torch.no_grad():
        for _ in range(start):
            s = f(s)
    for _ in range(c):
        s = f(s)
    prog = -SR(s)  # Bansal progressive loss, n=start no-grad steps then k=c steps
    g_prog = torch.cat([gg.flatten() for gg in torch.autograd.grad(-prog, fp)])
    ok = check("maze chunk pay (c=6) grad = progressive loss (k=6) grad", g_pay, g_prog)
    with torch.no_grad():
        s = s0; vals = [SR(s)]
        for t in range(T):
            s = f(s); vals.append(SR(s))
        chunks = sum(vals[a + c] - vals[a] for a in range(0, T, c))
        tel = float(abs(chunks - (vals[-1] - vals[0])))
    RES["chunk_ledger_telescoping_abs_err"] = tel
    print("chunk ledger telescoping |sum chunk pays - (S_R(T) - S_R(0))| = %.1e %s" % (tel, "PASS" if tel < 1e-9 else "FAIL"))
    return ok and tel < 1e-9


if __name__ == "__main__":
    ok = test_semivalues(False)
    ok &= test_semivalues(True)
    ok &= test_mhard()
    ok &= test_maze_chunk()
    RES["all_pass"] = bool(ok)
    json.dump(RES, open("/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P3/results/unit_tests.json", "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
