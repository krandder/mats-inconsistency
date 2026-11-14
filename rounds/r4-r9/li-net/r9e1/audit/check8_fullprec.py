#!/usr/bin/env python3
"""R9-E1 audit, check 8: full-precision results.json values behind the double-rounding candidates flagged by
check8_prose_tokens.py (the other candidates are bars, rule names or verdicts.md values). Run: python3 check8_fullprec.py"""
import json

R = "/home/agent/workspaces/eurisko-study/li-net/r9e1/results/main"
st = json.load(open(f"{R}/analysis/results.json"))["stats"]
for k in ("srB30_dkl_mmd_K24", "srB30_dkl_looped_attn_K1", "srB30_dkl_looped_attn_K16", "srB30_dkl_looped_attn_K24", "srB30_dkl_jacobi_prec_K1",
          "srB34_dkl_mmd_K1", "srB34_dkl_attn_K3_K1", "srB34_dkl_attn_K3_K16", "srB34_dkl_attn_K3_K24", "srB34_dkl_looped_attn_K1",
          "srB34_dkl_looped_attn_K16", "srB34_dkl_looped_attn_K24", "srA34_kl_looped_attn_K0", "srA30_kl_looped_attn_K0",
          "srA18_kl_looped_attn_K0", "srB30_dkl_mix_K1", "srB30_dkl_mix_K24"):
    print(k, repr(st[k]["mean"]))
for s in (101, 102, 103):
    f = json.load(open(f"{R}/s{s}/summary.json"))["fit"]
    print(f"s{s} jacobi_prec a {f['jacobi_prec']['a']!r} b {f['jacobi_prec']['b']!r}; econ_prec lambda_at_beta {f['econ_prec']['lambda_at_beta']}; "
          f"econ_D3_lam {f['scalars']['econ_D3_lam']!r}")
