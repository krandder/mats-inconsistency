#!/usr/bin/env python3
"""R7-E1 audit, check 6: parameter claims, the zero-JSD union call on P-FullSupport, and jacobi_lin_T4's W definition.
CPU, 2 threads, a few small bags only. Writes params_jacobi.json. Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 params_jacobi.py"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
B = "/home/agent/workspaces/eurisko-study"
R7 = B + "/li-net/r7e1"
for p in (B + "/li-net/r5e1", B + "/li-net/r6e1", R7):
    sys.path.insert(0, p)
import wm  # noqa: E402
import e1v3 as E3  # noqa: E402
import e1v4 as E  # noqa: E402

print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
npar = lambda *ms: int(sum(p.numel() for m in ms for p in m.parameters()))
cnt = {
    "attn_L3+K1head": npar(wm.AttnL3(), E3.KeyNet("K1", 128)),
    "attn_L3+K3head": npar(wm.AttnL3(), E3.KeyNet("K3", 128)),
    "attn_L3+K4head": npar(wm.AttnL3(), E3.KeyNet("K4", 128)),
    "looped_lin_T4+K4head": npar(E3.LoopedLinT4(4, 2.0), E3.KeyNet("K4", 128)),
    "jacobi_lin_T4+K3head": npar(E.JacobiLinT4(4, 2.0), E3.KeyNet("K3", 128)),
    "looped_attn_T3+K3head": npar(E.LoopedAttnT3(T=3), E3.KeyNet("K3", 128)),
    "attn_L3_core": npar(wm.AttnL3()), "looped_attn_T3_core": npar(E.LoopedAttnT3(T=3)), "jacobi_core": npar(E.JacobiLinT4(4, 2.0)),
    "K3_head": npar(E3.KeyNet("K3", 128)),
}
# by hand from the layer shapes (d 64, 4 heads, kdim 16, vdim 16, 36 cells)
mha = 3 * 64 * 64 + 3 * 64 + 64 * 64 + 64
ffn = 2 * 64 + (64 * 128 + 128) + (128 * 64 + 64)
common = (36 * 16 + 16) + (34 * 64 + 64) + (64 * 36 + 36)
hand = {"attn_L3_core": common + 3 * (mha + 128 + ffn), "looped_attn_T3_core": common + (mha + 128 + ffn) + 3 * 64,
        "jacobi_core": 36 + 4 + 4 * 36 + 36, "K3_head": 36 * 16 + 16}
print("instantiated:", cnt)
print("by hand:", hand, "agree:", all(cnt[k] == v for k, v in hand.items()))
print("ratio looped_attn_T3 / attn_L3-K3 (with K3 heads): %.5f; cores only: %.5f" % (
    cnt["looped_attn_T3+K3head"] / cnt["attn_L3+K3head"], cnt["looped_attn_T3_core"] / cnt["attn_L3_core"]))
fitp = {}
for s in (101, 102, 103):
    P = json.load(open(f"{R7}/results/main/s{s}/summary.json"))["fit"]["params"]
    fitp[s] = P
    ok = (P["attn_K1"] == cnt["attn_L3+K1head"] and P["attn_K3"] == cnt["attn_L3+K3head"] and P["attn_K4"] == cnt["attn_L3+K4head"]
          and P["looped_lin"] == cnt["looped_lin_T4+K4head"] and P["jacobi"] == cnt["jacobi_lin_T4+K3head"]
          and P["looped_attn"] == cnt["looped_attn_T3+K3head"])
    print(f"summary.json fit.params seed {s}: {P}  == instantiated: {ok}")
print("attention blocks in LoopedAttnT3:", sum(isinstance(m, torch.nn.MultiheadAttention) for m in E.LoopedAttnT3().modules()))

# ---- zero-JSD union call on P-FullSupport (raw.json) and summary value
ent = lambda p: -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))
uni = {}
for s in (101, 102, 103):
    raw = json.load(open(f"{R7}/results/main/s{s}/raw.json"))
    a, b = torch.tensor(raw["fs_NW"]["union"], dtype=torch.float64), torch.tensor(raw["fs_SE"]["union"], dtype=torch.float64)
    j = float((ent(0.5 * (a + b)) - 0.5 * (ent(a) + ent(b))).mean())
    T = json.load(open(f"{R7}/results/main/s{s}/summary.json"))["test"]
    tiles = all(set(E3.TILES) <= set(r[:6]) for r in raw["fs_NW"]["patch"] + raw["fs_SE"]["patch"])
    uni[s] = {"identical_NW_SE_union_beliefs": bool(torch.equal(a, b)), "jsd_from_raw": j, "summary_fs_jsd_union": T["fs_jsd_union"],
              "summary_fs_coverage_NW_SE": [T["fs_coverage_NW"], T["fs_coverage_SE"]], "six_tiles_first_in_every_bag": tiles}
    print("union call seed", s, uni[s])
    raw = None


# ---- jacobi_lin_T4 W definition vs its forward pass
def my_jacobi_coef(J, dl, beta, k, valid):
    """Independent float64 construction: per cell c, x^(T) = M_c b_c with M built by iterating the update on the identity;
    l*_c - l0_c = s_c 1^T M_c b_c = sum_b (s_c beta_b v_b (1^T M_c)_b) clamp(dl_bc)."""
    lam = F.softplus(J.lam_raw.detach().double()).numpy()
    eta, Pm, sc = J.eta.detach().double().numpy(), J.P.detach().double().numpy(), J.s.detach().double().numpy()
    kk, bb, vv = k.double().numpy(), beta.double().numpy(), valid.double().numpy()
    n, K = bb.shape
    W = np.zeros((n, K, 36))
    for i in range(n):
        g = np.maximum(kk[i] @ kk[i].T, 0) ** 2
        np.fill_diagonal(g, 1.0)
        g = g * vv[i][:, None] * vv[i][None, :]
        GB = g * (bb[i] * vv[i])[None, :]
        for c in range(36):
            A = lam[c] * np.eye(K) + GB
            D = lam[c] + GB.sum(1)
            Mx = np.zeros((K, K))
            for t in range(J.T):
                Mx = (Mx + (eta[t] * Pm[t, c] / D)[:, None] * (np.eye(K) - A @ Mx)) * vv[i][:, None]
            W[i, :, c] = sc[c] * (bb[i] * vv[i]) * Mx.sum(0)
    return W


def bag(n=3, K=24, seed=0, N=None):
    g = torch.Generator().manual_seed(seed)
    if N is None:
        k = F.normalize(torch.randn(n, K, 16, generator=g), dim=-1)
        dl = torch.randn(n, K, 36, generator=g) * 2.5
    else:  # controlled-bag-like: N identical cluster reads + (K - N) background reads
        base = F.normalize(torch.randn(n, 1, 16, generator=g), dim=-1)
        k = torch.cat([base.expand(n, N, 16), F.normalize(torch.randn(n, K - N, 16, generator=g), dim=-1)], 1)
        dl = torch.cat([(torch.randn(n, 1, 36, generator=g) * 2.5).expand(n, N, 36), torch.randn(n, K - N, 36, generator=g) * 2.5], 1)
    beta = torch.rand(n, K, generator=g) * 3.5 + 0.5
    if N is not None:
        beta[:, :N] = beta[:, :1]
    valid = torch.ones(n, K, dtype=torch.bool)
    valid[0, -3:] = False
    return torch.randn(n, 36, generator=g), dl, beta, k, valid


res = {"counted": cnt, "by_hand": hand, "summary_params": fitp, "union": uni, "jacobi": []}
configs = [("init (lambda 1, eta 1, P 1, s 1)", None)]
for s in (101, 102, 103):
    configs.append((f"seed {s} trained lambda/eta/s, P = per-step mean (per-cell P not saved)",
                    json.load(open(f"{R7}/results/main/s{s}/summary.json"))["fit"]["jacobi"]))
configs.append(("random (eta 0.3-1.2, P 0.5-1.5, lambda 0.3-1.3, s 0.2-1.5)", "random"))
for name, par in configs:
    torch.manual_seed(0)
    J = E.JacobiLinT4(4, 2.0)
    with torch.no_grad():
        if isinstance(par, dict):
            lam = torch.tensor(par["lambda"])
            J.lam_raw.copy_(torch.log(torch.expm1(lam)))
            J.eta.copy_(torch.tensor(par["eta"]))
            J.P.copy_(torch.tensor(par["P_mean"])[:, None].expand(4, 36))
            J.s.copy_(torch.tensor(par["s"]))
        elif par == "random":
            J.eta.uniform_(0.3, 1.2)
            J.P.uniform_(0.5, 1.5)
            J.lam_raw.uniform_(-1.0, 1.0)
            J.s.uniform_(0.2, 1.5)
    for bname, kw in (("random bag K=24", {}), ("N=8 identical + 16 background", {"N": 8}), ("N=16 identical + 8 background", {"N": 16})):
        l0, dl, beta, k, valid = bag(**kw, seed=len(res["jacobi"]))
        dlc = dl.clamp(-8, 8)
        out, x = J(l0, dl, beta, k, valid)
        w_owner = J.coef(dl, beta, k, valid)
        # (a) linear reconstruction of the forward output from the owner's coefficients
        rec = (l0 + (w_owner * dlc).sum(1) - out).abs().max().item()
        # (b) autograd: d l*_c / d dl_bc (keys and beta fixed) = exact coefficient (entries with |dl| < 8)
        dlv = dl.clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(lambda d: J(l0, d, beta, k, valid)[0].sum(0), dlv)   # [36, n, K, 36]
        w_auto = torch.stack([jac[c, :, :, c] for c in range(36)], -1)
        inside = (dl.abs() < 8)
        d_auto = (w_auto - w_owner)[inside].abs().max().item()
        # (c) my float64 construction
        w_mine = torch.tensor(my_jacobi_coef(J, dl, beta, k, valid), dtype=torch.float32)
        d_mine = (w_mine - w_owner).abs().max().item()
        row = {"params": name, "bag": bname, "max|forward - (l0 + sum w dl)|": rec, "max|owner coef - autograd|": d_auto,
               "max|owner coef - my float64 coef|": d_mine}
        if "N" in kw:
            # (d) the spec's per-read definition s_c x_rc / dl_rc, summed over the cluster, vs the exact coefficient sum
            N = kw["N"]
            spec_w = (J.s * x / dlc.where(dlc.abs() > 1e-6, torch.full_like(dlc, 1e-6)))[:, :N].sum(1)
            exact_w = w_owner[:, :N].sum(1)
            row["cluster W, exact coef (mean over cells, boards)"] = float(exact_w.mean())
            row["cluster W, spec s x/dl (mean over cells, boards)"] = float(spec_w.mean())
            row["median |spec - exact| / exact over cells"] = float(((spec_w - exact_w).abs() / exact_w.abs().clamp_min(1e-9)).median())
            if name.startswith("init"):
                bt = float(beta[0, 0])
                row["N beta/(lambda + N beta) with lambda 1 (no background mixing)"] = N * bt / (1 + N * bt)
        res["jacobi"].append(row)
        print(json.dumps(row))
json.dump(res, open("params_jacobi.json", "w"), indent=1)
