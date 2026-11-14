"""Unit tests: every propagator op against Monte Carlo on small dims (float64, CPU).

Exact ops (linear, residual add, Gaussian-input elementwise moments, Gaussian quadratic-form cumulants, input law)
are tested to MC precision; closures (LayerNorm cross-token, softmax, attention) to stated tolerances. The printed
table (python test_prop.py) is the op-level accuracy record used in the report.
"""
import math, sys
import numpy as np
import torch
import torch.nn.functional as Fn
from oputil import make_input, prop_with, mc_stats, compare, D64
from prop import qf_cumulants, elementwise, gelu, Propagator
import world_small as W
from model import allow_mask

NMC = 300000


def _att_fn(Wqkv, bqkv, Wo, bo, T, d, H, allow):
    def att(y):
        C, B = y.shape[:2]
        y = y.reshape(C * B, T, d)
        q, k, v = (y @ Wqkv.T + bqkv).split(d, -1)
        sp = lambda t: t.view(C * B, T, H, d // H).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = (q @ k.transpose(-1, -2) / math.sqrt(d // H)).masked_fill(~allow, float('-inf'))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(C * B, T, d)
        return (o @ Wo.T + bo).reshape(C, B, T, d)
    return att


def test_input_law_matches_sampler():
    m0, C0 = W.component_gaussians()
    d = W.Sampler(3).sample(200000)
    X = torch.cat([d['X'].reshape(len(d['X']), -1), d['ystar'][:, None]], 1).double()
    p = torch.tensor(W.PRIOR)
    mu = (p[:, None] * torch.tensor(m0)).sum(0)
    S2 = (p[:, None, None] * (torch.tensor(C0) + torch.tensor(m0)[:, :, None] * torch.tensor(m0)[:, None])).sum(0)
    cov = S2 - mu[:, None] * mu[None]
    emp = torch.cov(X.T)
    assert (X.mean(0) - mu).abs().max() < 0.01
    assert (emp - cov).abs().max() < 0.02


def test_qf_cumulants_exact():
    g = torch.Generator().manual_seed(0)
    n = 4
    A = torch.randn(n, n, generator=g, dtype=D64); S = A @ A.T / n
    mu = torch.randn(n, generator=g, dtype=D64); B = torch.randn(n, n, generator=g, dtype=D64); B = (B + B.T) / 2
    k = qf_cumulants(S, mu, B)
    x = mu + torch.randn(2000000, n, generator=g, dtype=D64) @ torch.linalg.cholesky(S).T
    q = ((x @ B) * x).sum(1)
    qc = q - q.mean()
    emp = [q.mean(), (qc ** 2).mean(), (qc ** 3).mean(), (qc ** 4).mean() - 3 * (qc ** 2).mean() ** 2]
    for a, b, tol in zip(k, emp, (0.01, 0.02, 0.05, 0.1)):
        assert abs(a.item() - b.item()) < tol * max(1.0, abs(b.item()))


def test_elementwise_gaussian_exact():
    mu = torch.tensor([0.3, -1.0], dtype=D64); var = torch.tensor([0.5, 2.0], dtype=D64)
    E, V, k3, k4, h1, h2, h3 = elementwise(gelu, mu, var, None, None, False)
    z = mu + var.sqrt() * torch.randn(2000000, 2, dtype=D64)
    f = gelu(z); fc = f - f.mean(0)
    assert torch.allclose(E, f.mean(0), atol=3e-3)
    assert torch.allclose(V, (fc ** 2).mean(0), rtol=1e-2)
    assert torch.allclose(k3, (fc ** 3).mean(0), atol=2e-2)


def _check(name, out, pr, mc, tol):
    r = compare(name, out, pr, mc)
    for k, t in tol.items():
        assert r[k] < t, (name, k, r[k], t)
    return r


def test_linear_exact():
    for K in (2, 4):
        inp = make_input(2, (3, 6), 5, seed=3)
        x = inp['sample'](NMC, 4)
        Wt = torch.randn(5, 6, dtype=D64); b = torch.randn(5, dtype=D64)
        mc = mc_stats(x @ Wt.T + b, x)
        pr, blk = prop_with(inp, K)
        tol = dict(mean=0.01, var=0.01, offcov=0.02, R=0.02) | (dict(skew=0.05, exk=0.1) if K == 4 else {})
        _check(f'linear K={K}', pr.linear_tok(blk, Wt, b), pr, mc, tol)


def test_gelu():
    for K, nng in ((2, 0), (4, 0), (4, 8)):
        inp = make_input(2, (3, 8), nng, seed=4, scale=1.0)
        x = inp['sample'](NMC, 9)
        mc = mc_stats(Fn.gelu(x), x)
        pr, blk = prop_with(inp, K)
        tol = dict(mean=0.01, var=0.02, offcov=0.1, R=0.05) | (dict(skew=0.1, exk=0.6) if K == 4 else {})
        _check(f'GELU K={K} nonGauss={nng}', pr.gelu_op(blk), pr, mc, tol)


def test_layernorm():
    for K, nng in ((2, 0), (4, 0), (4, 6)):
        inp = make_input(2, (3, 16), nng, seed=1, mean_scale=0.5)
        x = inp['sample'](NMC, 7)
        g = 1 + 0.3 * torch.randn(16, dtype=D64); b = 0.2 * torch.randn(16, dtype=D64)
        mc = mc_stats(Fn.layer_norm(x, (16,), g, b, eps=1e-5), x)
        pr, blk = prop_with(inp, K)
        tol = dict(mean=0.02, var=0.03, offcov=0.06, R=0.03) | (dict(skew=0.08, exk=0.15) if K == 4 else {})
        _check(f'LayerNorm K={K} nonGauss={nng}', pr.layernorm(blk, g, b), pr, mc, tol)


def test_attention():
    T, d, H = 4, 32, 2
    allow = allow_mask(T)
    torch.manual_seed(0)
    for scale, tolc in ((0.5, 0.08), (1.0, 0.2)):
        Wqkv = torch.randn(3 * d, d, dtype=D64) / math.sqrt(d) * scale; bqkv = 0.1 * torch.randn(3 * d, dtype=D64)
        Wo = torch.randn(d, d, dtype=D64) / math.sqrt(d); bo = 0.1 * torch.randn(d, dtype=D64)
        att = _att_fn(Wqkv, bqkv, Wo, bo, T, d, H, allow)
        inp = make_input(2, (T, d), 0, seed=2, mean_scale=0.7)
        x = inp['sample'](100000, 3)
        mc = mc_stats(att(x), x)
        for K in (2, 4):
            pr, blk = prop_with(inp, K)
            _check(f'attention qk-scale={scale} K={K}', pr.attention(blk, Wqkv, bqkv, Wo, bo, H, allow), pr, mc,
                   dict(mean=0.05, var=tolc, offcov=tolc, R=2 * tolc))


def test_residual_add_exact():
    pr = Propagator(K=4, dtype=D64, components=[0]); pr._reset(1)
    N = 6
    A0 = torch.randn(1, N + 1, N + 1, dtype=D64); P = A0 @ A0.transpose(1, 2)
    st = (torch.zeros(1, N + 1, dtype=D64), P, torch.zeros(1, N + 1, 0, dtype=D64))
    Wb = torch.randn(N, N + 1, dtype=D64)
    blk = (torch.zeros(1, 2, 3, dtype=D64), (Wb @ P[0] @ Wb.T).reshape(1, 2, 3, 2, 3),
           (Wb @ P[0]).reshape(1, 2, 3, N + 1), torch.zeros(1, 2, 3, 0, dtype=D64))
    m, Pn, A = pr.add_branch(st, blk)
    M = torch.eye(N + 1, dtype=D64); M[:N] += Wb
    assert torch.allclose(Pn[0], M @ P[0] @ M.T, atol=1e-10)


if __name__ == '__main__':
    for f in [test_input_law_matches_sampler, test_qf_cumulants_exact, test_elementwise_gaussian_exact,
              test_linear_exact, test_gelu, test_layernorm, test_attention, test_residual_add_exact]:
        try:
            f(); print('PASS', f.__name__, flush=True)
        except AssertionError as e:
            print('FAIL', f.__name__, e, flush=True)
