#!/usr/bin/env python3
"""R8-E1 audit, check 6 (and parts of checks 2-3): parameter counts by closed form and by instantiating the code's classes; exactness
of jacobi_lin_T4_prec's W (the coefficient of its linear map, read-dependent lambda) against autograd and against the auditor's own
numpy implementation of the PREREG formula (the code runs in float32 as on the VM; the auditor formula in float64); anchor-token identity and zero shift; precision-weighted crowding vs its formula.
This check imports e1v5 (the code under test); the numbers of check 4 do not. CPU only, 2 threads.
Run: OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r4e3a/pylib python3 a6_params_coef.py"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/li-net/r8e1")
import e1v5 as E  # noqa: E402
import e1v3 as E3  # noqa: E402
import e1v4 as E4  # noqa: E402
import wm  # noqa: E402

rep = {}
# ---- (a) closed-form counts (auditor's arithmetic)
lin = lambda i, o: i * o + o
mha = lambda d: 3 * d * d + 3 * d + d * d + d
ln = lambda d: 2 * d
ffn = lambda d: ln(d) + lin(d, 2 * d) + lin(2 * d, d)
k3, k1, k4 = lin(36, 16), lin(128, 16), lin(128 + 36, 16)
looped_attn_core = lin(36, 16) + lin(16 + 16 + 2, 64) + mha(64) + ln(64) + ffn(64) + 3 * 64 + lin(64, 36)
attn_core = lin(36, 16) + lin(16 + 16 + 2, 64) + 3 * (mha(64) + ln(64) + ffn(64)) + lin(64, 36)
jac_core = 36 + 4 + 4 * 36 + 36
closed = {"looped_attn_T3+K3": looped_attn_core + k3, "attn_L3-K3+K3": attn_core + k3, "attn_L3-K1+K1": attn_core + k1,
          "looped_lin_T4+K4": 4 + 4 * 36 + 1 + k4, "jacobi_lin_T4+K3": jac_core + k3, "jacobi_lin_T4_prec+K3": jac_core + 2 + k3}
# ---- (b) instantiate the code's classes
n = lambda *m: sum(p.numel() for x in m for p in x.parameters())
code = {"looped_attn_T3+K3": n(E3.KeyNet("K3", 128), E.LoopedAttnT3P(3.4, T=3)), "attn_L3-K3+K3": n(E3.KeyNet("K3", 128), E.AttnL3P(3.4)),
        "attn_L3-K1+K1": n(E3.KeyNet("K1", 128), wm.AttnL3()), "looped_lin_T4+K4": n(E3.KeyNet("K4", 128), E3.LoopedLinT4(4, 2.0, 0.04)),
        "jacobi_lin_T4+K3": n(E3.KeyNet("K3", 128), E4.JacobiLinT4(4, 2.0)), "jacobi_lin_T4_prec+K3": n(E3.KeyNet("K3", 128), E.JacobiPrecT4(4, 2.0))}
rep["params_closed_form"], rep["params_code"] = closed, code
rep["params_claimed_PREREG"] = {"looped_attn_T3+K3": 39428, "attn_L3-K3+K3": 106180, "jacobi_lin_T4_prec+K3": 814, "jacobi_lin_T4+K3": 812}
S = json.load(open("/home/agent/workspaces/eurisko-study/li-net/r8e1/results/main/s101/summary.json"))
rep["params_logged_in_run"] = S["fit"]["params"]


# ---- (c) jacobi_lin_T4_prec W exactness on random bags with random (non-init) parameters
def own_forward(J, l0, dl, beta, k, valid):
    """Auditor's numpy implementation of the PREREG formula (float64): lambda_rc = softplus(l~_c) sigmoid(a - b beta_r);
    g_rb = relu(<k_r,k_b>)^2 (g_rr = 1) v_r v_b; D_rc = lambda_rc + sum_b g_rb beta_b; x <- (x + eta_t P_tc (b - (lambda x + G x)) / D) v;
    l* = l0 + s_c sum_r x_rc."""
    lt = np.log1p(np.exp(J.lam_raw.detach().double().numpy()))
    a, b_ = float(J.a), float(J.b)
    eta, Pm, s = J.eta.detach().double().numpy(), J.P.detach().double().numpy(), J.s.detach().double().numpy()
    l0, dl, beta, k, v = (x.detach().double().numpy() for x in (l0, dl, beta, k, valid.float()))
    out = np.zeros_like(l0)
    for i in range(l0.shape[0]):
        K = dl.shape[1]
        g = np.maximum(k[i] @ k[i].T, 0) ** 2
        np.fill_diagonal(g, 1.0)
        g = g * v[i][:, None] * v[i][None, :]
        lam = lt[None, :] * (1 / (1 + np.exp(-(a - b_ * beta[i]))))[:, None]          # [K, 36]
        GB = g * (beta[i] * v[i])[None, :]
        D = lam + GB.sum(1)[:, None]
        bb = (beta[i] * v[i])[:, None] * np.clip(dl[i], -8, 8)
        x = np.zeros((K, 36))
        for t in range(len(eta)):
            res = bb - (lam * x + GB @ x)
            x = (x + eta[t] * Pm[t][None, :] * res / D) * v[i][:, None]
        out[i] = l0[i] + s * x.sum(0)
    return out


g = torch.Generator().manual_seed(20261017)
worst = {"coef_vs_forward": 0.0, "coef_vs_autograd": 0.0, "offdiag_jacobian": 0.0, "code_vs_own_formula": 0.0, "lam_rc_vs_formula": 0.0}
for trial in range(6):
    J = E.JacobiPrecT4(4, 2.0)
    with torch.no_grad():
        J.lam_raw.copy_(torch.randn(36, generator=g, dtype=torch.float64))
        J.eta.copy_(torch.rand(4, generator=g, dtype=torch.float64) * 0.9 + 0.3)
        J.P.copy_(torch.rand(4, 36, generator=g, dtype=torch.float64) + 0.5)
        J.s.copy_(torch.rand(36, generator=g, dtype=torch.float64) + 0.3)
        J.a.fill_(float(torch.randn((), generator=g)) * 2)
        J.b.fill_(float(torch.rand((), generator=g)) * 2)
    B, K = 4, int(torch.randint(1, 25, (), generator=g))
    l0 = torch.randn(B, 36, generator=g)
    dl = (torch.rand(B, K, 36, generator=g) * 2 - 1) * 7.5            # |dl| < 8: clamp inactive (float32, as in the run)
    beta = torch.rand(B, K, generator=g) * 3.5 + 0.5
    k = F.normalize(torch.randn(B, K, 16, generator=g), dim=-1)
    valid = torch.rand(B, K, generator=g) < 0.8
    valid[:, 0] = True
    out, _ = J(l0, dl, beta, k, valid)
    w = J.coef(dl, beta, k, valid)
    worst["coef_vs_forward"] = max(worst["coef_vs_forward"], float((out - l0 - (w * dl).sum(1)).abs().max()))
    jac = torch.autograd.functional.jacobian(lambda d: J(l0, d, beta, k, valid)[0], dl)        # [B, 36, B, K, 36]
    jd = torch.stack([torch.stack([jac[i, c, i, :, c] for c in range(36)], 1) for i in range(B)])    # [B, K, 36]
    worst["coef_vs_autograd"] = max(worst["coef_vs_autograd"], float((jd - w).abs().max()))
    mask = torch.ones_like(jac, dtype=torch.bool)
    for i in range(B):
        for c in range(36):
            mask[i, c, i, :, c] = False
    worst["offdiag_jacobian"] = max(worst["offdiag_jacobian"], float(jac[mask].abs().max()))
    worst["code_vs_own_formula"] = max(worst["code_vs_own_formula"], float(np.abs(out.detach().numpy() - own_forward(J, l0, dl, beta, k, valid)).max()))
    lam_code = J.lam_rc(beta).detach().numpy()
    lam_own = np.log1p(np.exp(J.lam_raw.detach().numpy()))[None, None, :] / (1 + np.exp(-(float(J.a) - float(J.b) * beta.numpy())))[..., None]
    worst["lam_rc_vs_formula"] = max(worst["lam_rc_vs_formula"], float(np.abs(lam_code - lam_own).max()))
rep["jacobi_prec_W_exactness_max_abs_err_float32_code_vs_float64_own"] = worst
# identical-read closed form at init (eta = P = s = 1): N beta / (lambda(beta) + N beta)
J = E.JacobiPrecT4(4, 2.0)
with torch.no_grad():
    J.a.fill_(1.3)
    J.b.fill_(0.6)
cf = 0.0
for bval in (1.5, 2.2, 3.4):
    lam = 2.0 / (1 + math.exp(-(1.3 - 0.6 * bval)))
    for N in (1, 2, 4, 8, 16):
        kk = F.normalize(torch.randn(1, 1, 16), dim=-1).expand(1, N, 16).contiguous()
        dd = torch.randn(1, 1, 36).expand(1, N, 36).contiguous()
        w = J.coef(dd, torch.full((1, N), bval), kk, torch.ones(1, N, dtype=torch.bool))
        cf = max(cf, float((w[0].sum(0) - N * bval / (lam + N * bval)).abs().max()))
rep["jacobi_prec_identical_reads_closed_form_max_abs_err"] = cf
J0 = E.JacobiPrecT4(4, 2.0)
rep["jacobi_prec_init_lambda_for_beta_0.5_2_4"] = J0.lam_rc(torch.tensor([[0.5, 2.0, 4.0]])).mean(-1).tolist()

# ---- (d) anchor token: identity at K = 0 (bitwise) and zero shift with reads, both anytime pools, random output heads
torch.manual_seed(7)
anc = {}
for name, pool in (("looped_attn_T3P", E.LoopedAttnT3P(3.4, T=3)), ("attn_L3P", E.AttnL3P(3.4))):
    keyn = E3.KeyNet("K3", 128)
    with torch.no_grad():
        pool.out.weight.normal_(0, 0.3)
    l0, lp = torch.randn(5, 36), torch.randn(5, 36)
    for M in (2, 6, 12, 18):
        b0 = torch.full((5,), M * 3.4 / 6)
        e = torch.zeros(5, 0, 36)
        dla, ba, kk, va = E.anchor_bag(keyn, l0, lp, e, torch.zeros(5, 0), torch.zeros(5, 0, dtype=torch.bool), b0)
        o, _ = pool(l0, dla, ba, kk, va)
        anc[f"{name}_M{M}_K0_bitwise_equal"] = bool(torch.equal(o, l0))
        dl = (torch.rand(5, 9, 36) * 2 - 1) * 7.5
        be = torch.rand(5, 9) * 3 + 0.5
        dla, ba, kk, va = E.anchor_bag(keyn, l0, lp, dl, be, torch.ones(5, 9, dtype=torch.bool), b0)
        o, w = pool(l0, dla, ba, kk, va)
        anc[f"{name}_M{M}_anchor_dl_zero"] = float(dla[:, 0].abs().max()) == 0.0
        anc[f"{name}_M{M}_anchor_beta_over_beta_clean"] = float(ba[:, 0].mean() / 3.4)
        anc[f"{name}_M{M}_out_minus_l0_minus_sum_reads_max"] = float((o - l0 - (w[:, 1:] * dl).sum(1)).abs().max())
        s0 = (l0 - lp).abs() / ((l0 - lp).norm(dim=-1, keepdim=True) + 1e-6)
        anc[f"{name}_M{M}_k0_equals_normalize_W_s0"] = float((kk[:, 0] - F.normalize(keyn.W(s0), dim=-1)).abs().max())
rep["anchor_token"] = anc
# ---- (e) precision-weighted crowding vs formula
k = F.normalize(torch.randn(3, 6, 16), dim=-1)
valid = torch.rand(3, 6) < 0.8
valid[:, 0] = True
beta = torch.rand(3, 6) * 3 + 0.5
c = E.crowding_prec(k, valid, beta, 3.4)
own = torch.zeros(3, 6)
for i in range(3):
    for r in range(6):
        own[i, r] = sum(float(valid[i, b_]) * float(beta[i, b_]) / 3.4 * max(0.0, float(k[i, r] @ k[i, b_])) ** 2 for b_ in range(6))
rep["crowding_prec_vs_formula_max_abs_err"] = float((c - own.clamp_min(1e-6)).abs().max())
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(OUT, "a6_params_coef.json"), "w"), indent=1)
