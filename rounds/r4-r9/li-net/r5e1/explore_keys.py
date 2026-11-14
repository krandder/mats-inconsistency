#!/usr/bin/env python3
"""EXPLORATORY (post hoc, offline; not a rule test). Is the overlap structure in the learned keys hidden under a shared
component? For the saved keys of the U_NW, U_SE and unskewed test bags (results/<tag>/s*/partA_raw.json), compare
Pearson(<k_r, k_b>, overlap/6) for (a) the keys as used (the E1-D2.1 statistic), (b) the keys centred within each bag
(k - mean of the bag's keys, renormalised). Output: results/<tag>/analysis/explore_keys.json.
Usage: python3 explore_keys.py v2|main
"""
import json
import os
import sys

import numpy as np

R5 = os.path.dirname(os.path.abspath(__file__))


def patches():
    P = []
    for r in range(5):
        for c in range(4):
            P.append([(r + a) * 6 + c + b for a in range(2) for b in range(3)])
    for r in range(4):
        for c in range(5):
            P.append([(r + a) * 6 + c + b for a in range(3) for b in range(2)])
    return [set(p) for p in P]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    P = patches()
    OV = np.array([[len(a & b) for b in P] for a in P]) / 6.0
    nwc = np.array([sum(c < 18 for c in p) for p in P])
    sector = np.where(nwc >= 4, 0, np.where(nwc <= 2, 1, 2))
    out = {}
    for s in (101, 102, 103):
        d = json.load(open(os.path.join(R5, "results", tag, f"s{s}", "partA_raw.json")))
        res = {}
        for mode in ("as_used", "centred_in_bag"):
            cs_all, ov_all, same, across = [], [], [], []
            for b in ("NW", "SE", "UN"):
                K = np.array(d[f"bag_{b}"]["keys"], dtype=np.float64)        # [n, 24, 16]
                pt = np.array(d[f"bag_{b}"]["patch"])                         # [n, 24]
                if mode == "centred_in_bag":
                    K = K - K.mean(1, keepdims=True)
                K = K / np.linalg.norm(K, axis=-1, keepdims=True).clip(1e-9)
                iu = np.triu_indices(K.shape[1], 1)
                G = np.einsum("nid,njd->nij", K, K)[:, iu[0], iu[1]]
                pa, pb = pt[:, iu[0]], pt[:, iu[1]]
                cs_all.append(G.ravel())
                ov_all.append(OV[pa, pb].ravel())
                same.append(G[pa == pb])
                sa, sb = sector[pa], sector[pb]
                across.append(G[((sa == 0) & (sb == 1)) | ((sa == 1) & (sb == 0))])
            cs, ov = np.concatenate(cs_all), np.concatenate(ov_all)
            byov = {str(int(round(o * 6))): float(cs[np.isclose(ov * 6, o * 6)].mean()) for o in np.unique(ov)}
            res[mode] = {"pearson": float(np.corrcoef(cs, ov)[0, 1]), "same_patch_cos": float(np.concatenate(same).mean()),
                         "across_sector_cos": float(np.concatenate(across).mean()), "cos_by_overlap": byov}
            res[mode]["gap"] = res[mode]["same_patch_cos"] - res[mode]["across_sector_cos"]
        out[s] = res
    od = os.path.join(R5, "results", tag, "analysis")
    json.dump(out, open(os.path.join(od, "explore_keys.json"), "w"), indent=1)
    for mode in ("as_used", "centred_in_bag"):
        pe = [out[s][mode]["pearson"] for s in out]
        gp = [out[s][mode]["gap"] for s in out]
        print(f"{tag} {mode:15s} Pearson {np.mean(pe):.3f} ({', '.join(f'{x:.3f}' for x in pe)}); same - across "
              f"{np.mean(gp):.3f}; cos by overlap (s101) " + ", ".join(f"{k}: {v:.2f}" for k, v in out[101][mode]["cos_by_overlap"].items()))


if __name__ == "__main__":
    main()
