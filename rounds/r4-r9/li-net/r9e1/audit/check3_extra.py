#!/usr/bin/env python3
"""R9-E1 audit, checks 2-3 (from raw.json): interleaved-stream structure, read skew, D3 background exclusion, identical configs.
Uses env6 (board geometry only) from the hash-checked staged copy audit/tmpsrc (copied from /tmp/r9e1/src).
Run: PYTHONPATH=/tmp/r4e3a/pylib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python3 check3_extra.py > check3_extra.txt"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmpsrc"))
import env6  # noqa: E402  (geometry only; not one of the modules excluded for check 4)

RES = "/home/agent/workspaces/eurisko-study/li-net/r9e1/results/main"
TILES = (0, 3, 8, 11, 16, 19)
cfgs = {}
for s in (101, 102, 103):
    S = json.load(open(f"{RES}/s{s}/summary.json"))
    cfgs[s] = {k: v for k, v in S["cfg"].items() if k not in ("seed", "out")}
    raw = json.load(open(f"{RES}/s{s}/raw.json"))
    out = {}
    for h in ("NW", "SE"):
        P = np.asarray(raw[f"fs_{h}"]["patch"])
        tiles = P[:, ::4]
        out[f"fs_{h}_tiles_ok"] = bool(np.isin(tiles, TILES).all() and all(sorted(r[:6]) == sorted(TILES) for r in tiles)
                                       and (tiles[:, 6:] == tiles[:, :2]).all())
        sk = np.delete(P, np.arange(0, 32, 4), axis=1)
        nw, se = env6.IN_NW[sk], env6.IN_SE[sk]
        out[f"fs_{h}_skewed_NWonly_share"] = round(float((nw & ~se).mean()), 3)
        out[f"fs_{h}_skewed_SEonly_share"] = round(float((se & ~nw).mean()), 3)
        for md in ("A", "B"):
            Q = np.asarray(raw[f"srS{md}_{h}"]["patch"])
            out[f"sr{md}_{h}_NWonly_share"] = round(float((env6.IN_NW[Q] & ~env6.IN_SE[Q]).mean()), 3)
    a = np.asarray(raw["controlled"]["astar"])
    bg = np.asarray(raw["controlled"]["bg_patch"])
    out["d3_bg_overlap_with_Astar_max"] = int(env6.OVERLAP[a[:, None], bg].max())
    print(s, out)
print("cfg identical across seeds (except seed/out):", all(cfgs[s] == cfgs[101] for s in cfgs))
c = cfgs[101]
print({k: c[k] for k in ("n_cal", "pool", "n_ref", "n_any", "pool_any", "n_d3", "n_pfs", "n_test", "agg_steps", "agg_batch", "fit_steps",
                         "trust_steps", "K_sr", "p_sr", "p_full", "K_tr", "strong_cands", "kl_strong", "p_modeB", "std_w", "pfs_w", "d3_w",
                         "sr_roots", "gamma_voi", "n_clu", "n_bg", "kmin", "kmax", "attn_T", "loop_T", "gamma_key", "ckpt_dir", "device")})
