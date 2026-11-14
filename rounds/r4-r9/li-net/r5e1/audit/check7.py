#!/usr/bin/env python3
"""R5-E1 audit, check 7 helpers: recompute from the RAW files the write-up's exploratory and descriptive numbers
(section 4.1 coverage split, section 5 centred keys, section 8 sycophancy boards, Token-Masked drift direction, attn_L3
per-read weights). Auditor's own code; imports nothing from li-net/r5e1 except env6's board generator for the probe evidence.
Run: OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r4e3a/pylib python3 check7.py > check7-output.txt"""
import json
import os
import sys

import numpy as np

R5 = "/home/agent/workspaces/eurisko-study/li-net/r5e1"
RES = os.path.join(R5, "results")
SEEDS = (101, 102, 103)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rederive import PM, OV, SECT, kl, jsd, make_boards, enum_placements   # noqa: E402  (auditor's own module)


def main():
    raw = {s: json.load(open(os.path.join(RES, "v2", f"s{s}", "partA_raw.json"))) for s in SEEDS}
    print("## A. Section 4.1 split of the U_NW vs U_SE disagreement by read coverage (own code, raw v2 arrays)")
    pools = ("mix", "mmd", "kc", "attn", "union_call")
    acc = {}
    for s in SEEDS:
        d = raw[s]
        Ps = np.array(d["pstar"])
        cnt = {b: PM[np.array(d[f"bag_{b}"]["patch"])].sum(1) for b in ("NW", "SE")}            # reads per cell [500, 36]
        cor = {b: (PM[np.array(d[f"bag_{b}"]["patch"])] * (np.array(d[f"bag_{b}"]["tau"]) > 1)[..., None]).sum(1) for b in ("NW", "SE")}
        both = (cnt["NW"] > 0) & (cnt["SE"] > 0)
        one = (cnt["NW"] > 0) ^ (cnt["SE"] > 0)
        none = (cnt["NW"] == 0) & (cnt["SE"] == 0)
        sk = np.abs(cnt["NW"] - cnt["SE"])
        cats = {"all": np.ones_like(both), "both": both, "both_skew0": both & (sk == 0), "both_skew1-2": both & (sk >= 1) & (sk <= 2),
                "both_skew3+": both & (sk >= 3), "both_clean_only": both & (cor["NW"] == 0) & (cor["SE"] == 0), "one": one, "none": none}
        for c, m in cats.items():
            acc.setdefault(("share", c), []).append(float(m.mean()))
        for p in pools:
            J = jsd(np.array(d["bag_NW"][p]), np.array(d["bag_SE"][p]))
            K = 0.5 * (kl(Ps, np.array(d["bag_NW"][p])) + kl(Ps, np.array(d["bag_SE"][p])))
            for c, m in cats.items():
                acc.setdefault((p, c, "jsd"), []).append(float(J[m].mean()))
                acc.setdefault((p, c, "kl"), []).append(float(K[m].mean()))
                acc.setdefault((p, c, "jsdsum"), []).append(float(J[m].sum() / J.size))     # contribution to the all-cell mean
    cats = ("all", "both", "both_skew0", "both_skew1-2", "both_skew3+", "both_clean_only", "one", "none")
    print("cell share:", {c: round(np.mean(acc[("share", c)]), 4) for c in cats})
    for p in pools:
        print(f"{p:10s}", " ".join(f"{c}: {np.mean(acc[(p, c, 'jsd')]):.4f} ({np.mean(acc[(p, c, 'kl')]):.3f})" for c in cats))
    for p in ("attn", "union_call", "mix"):
        tot = np.mean(acc[(p, "all", "jsd")])
        print(f"{p}: share of its all-cell JSD from shared cells {np.mean(acc[(p, 'both', 'jsdsum')]) / tot:.3f}, "
              f"one-bag cells {np.mean(acc[(p, 'one', 'jsdsum')]) / tot:.3f}, unread {np.mean(acc[(p, 'none', 'jsdsum')]) / tot:.3f}")
    for c in ("all", "both", "both_skew0", "both_skew3+", "one"):
        r_mix = [a / b for a, b in zip(acc[("attn", c, "jsd")], acc[("mix", c, "jsd")])]
        r_un = [a / b for a, b in zip(acc[("attn", c, "jsd")], acc[("union_call", c, "jsd")])]
        print(f"JSD attn/mix on {c}: {np.mean(r_mix):.2f} (per seed {', '.join(f'{x:.2f}' for x in r_mix)}); attn/union: {np.mean(r_un):.2f}")
    for p in ("mix", "mmd", "kc", "attn"):
        a, b = np.mean(acc[(p, "both_skew0", "jsd")]), np.mean(acc[(p, "both_skew3+", "jsd")])
        print(f"{p}: JSD skew3+ / skew0 = {b / a:.2f}")
    ej = json.load(open(os.path.join(RES, "v2", "analysis", "explore_coverage.json")))
    dmax = 0.0
    for s in SEEDS:
        for p in pools:
            for c in cats:
                if c in ej[str(s)][p]:
                    dmax = max(dmax, abs(ej[str(s)][p][c]["jsd"] - acc[(p, c, "jsd")][SEEDS.index(s)]))
    print(f"max |own - explore_coverage.json| JSD over seeds/pools/categories: {dmax:.2e}")

    print("\n## B. Section 5 exploratory: keys centred within each bag (own code)")
    for tag in ("v2", "main"):
        pe, gp, byov = [], [], []
        for s in SEEDS:
            d = raw[s] if tag == "v2" else json.load(open(os.path.join(RES, "main", f"s{s}", "partA_raw.json")))
            cs, ov, same, across = [], [], [], []
            iu = np.triu_indices(24, 1)
            for b in ("NW", "SE", "UN"):
                k = np.array(d[f"bag_{b}"]["keys"], dtype=np.float64)
                k = k - k.mean(1, keepdims=True)
                k = k / np.clip(np.linalg.norm(k, axis=2, keepdims=True), 1e-9, None)
                G = np.einsum("nrk,nbk->nrb", k, k)[:, iu[0], iu[1]]
                p = np.array(d[f"bag_{b}"]["patch"])
                pa, pb = p[:, iu[0]], p[:, iu[1]]
                cs.append(G.ravel())
                ov.append((OV[pa, pb]).ravel())
                same.append(G[pa == pb])
                across.append(G[((SECT[pa] == 0) & (SECT[pb] == 1)) | ((SECT[pa] == 1) & (SECT[pb] == 0))])
            cs, ov = np.concatenate(cs), np.concatenate(ov)
            pe.append(float(np.corrcoef(cs, ov)[0, 1]))
            gp.append(float(np.concatenate(same).mean() - np.concatenate(across).mean()))
            byov.append([float(cs[ov == o].mean()) for o in (0, 1, 2, 3, 4, 6)])
        print(f"{tag}: centred Pearson {np.mean(pe):.3f} ({', '.join(f'{x:.3f}' for x in pe)}); centred same-across gap {np.mean(gp):.3f};"
              f" centred cos by overlap s101 {[round(x, 2) for x in byov[0]]}")

    print("\n## C. Section 8: sycophancy boards (main test split; probe evidence regenerated from stream 2)")
    d = json.load(open(os.path.join(RES, "main", "s101", "partA_raw.json")))
    Ps = np.array(d["pstar"])
    note = np.array(d["note"])
    nt = np.array(d["note_true"]).astype(bool)
    TB = make_boards(2, 500, enum_placements())
    rows = np.arange(500)
    fn = ~nt
    agree = fn & (Ps[rows, note] >= 0.5)
    evn = TB["ev"][rows, note]
    print(f"false-note boards {int(fn.sum())}; of them P*(note) >= 0.5: {int(agree.sum())} ({agree.sum() / fn.sum():.4f});"
          f" evidence at the note cell on those boards: hit {int((evn[agree] == 2).sum())}, miss {int((evn[agree] == 1).sum())}, none {int((evn[agree] == 0).sum())}")

    print("\n## D. Token-Masked drift direction (post - pre) on the top-4 / bottom-4 P* cells (raw partB_raw.json)")
    for arm in ("tokmask", "fused", "truthonly"):
        out = []
        for s in SEEDS:
            dA = json.load(open(os.path.join(RES, "main", f"s{s}", "partA_raw.json")))
            dB = json.load(open(os.path.join(RES, "main", f"s{s}", "partB_raw.json")))
            P = np.array(dA["pstar"])
            pre, post = np.array(dB[arm]["pre_rl"]["mu"]), np.array(dB[arm]["post_rl"]["mu"])
            top = np.argsort(-P, 1, kind="stable")[:, :4]
            bot = np.argsort(P, 1, kind="stable")[:, :4]
            dd = post - pre
            out.append((float(np.take_along_axis(dd, top, 1).mean()), float(np.take_along_axis(dd, bot, 1).mean()),
                        float(np.take_along_axis(post - P, top, 1).mean()), float(np.take_along_axis(post - P, bot, 1).mean())))
        print(f"{arm}: per seed (drift on top-4, drift on bottom-4, post mu-P* on top-4, on bottom-4):",
              [tuple(round(x, 4) for x in o) for o in out])

    print("\n## E. attn_L3 weight per read in preference bags (v2 summary attn_wsum_* / read counts)")
    for s in SEEDS:
        pa = json.load(open(os.path.join(RES, "v2", f"s{s}", "summary.json")))["part_a"]
        fnw, fse = pa["frac_nw_NW"], pa["frac_nw_SE"]
        nwN, seN = 24 * fnw, 24 * (1 - fnw)
        nwS, seS = 24 * fse, 24 * (1 - fse)
        a = (pa["attn_wsum_serollouts_NW"] / seN) / (pa["attn_wsum_nwrollouts_NW"] / nwN)
        b = (pa["attn_wsum_nwrollouts_SE"] / nwS) / (pa["attn_wsum_serollouts_SE"] / seS)
        print(f"s{s}: U_NW bag: scarce-SE per-read weight / NW per-read weight = {a:.2f}; U_SE bag: scarce-NW / SE = {b:.2f}")


if __name__ == "__main__":
    main()
