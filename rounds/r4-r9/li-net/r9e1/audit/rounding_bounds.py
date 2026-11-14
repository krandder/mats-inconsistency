#!/usr/bin/env python3
"""R9-E1 audit, check 4 follow-up: is every per-seed KL in summary.json inside the interval that 5-decimal rounding of raw.json
allows? For each cell, the true belief q and the true exact-Bayes p lie within +-5e-6 of their rounded values; KL(p||q) (q clamped
to [1e-6, 1-1e-6], the registered metric) is convex, so its per-cell max over the box is at a vertex and its min is at q = clamp(p).
Summing per-cell min / max gives [lo, hi] for the mean. Imports json and numpy only."""
import json

import numpy as np

RES = "/home/agent/workspaces/eurisko-study/li-net/r9e1/results/main"
H = 5e-6
E = 1e-6


def klc(p, q):
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, E, 1 - E)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)
        b = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1.0)) - np.log1p(-q)), 0.0)
    return a + b


def bounds(p, q):
    plo, phi = np.clip(p - H, 0, 1), np.clip(p + H, 0, 1)
    qlo, qhi = np.clip(q - H, E, 1 - E), np.clip(q + H, E, 1 - E)
    hi = np.maximum.reduce([klc(pp, qq) for pp in (plo, phi) for qq in (qlo, qhi)])
    lo = np.minimum.reduce([klc(pp, np.clip(pp, qlo, qhi)) for pp in np.linspace(0, 1, 5)[:, None, None] * (phi - plo) + plo])
    return float(lo.mean()), float(hi.mean()), float(klc(p, q).mean())


def main():
    out, n_in, n_all, worst = {}, 0, 0, []
    for s in (101, 102, 103):
        raw = json.load(open(f"{RES}/s{s}/raw.json"))
        T = json.load(open(f"{RES}/s{s}/summary.json"))["test"]
        Ms = raw["M_strong"]
        p = np.asarray(raw["pstar"], dtype=np.float64)
        items = []
        for M in (2, 6, 12, 18, 30, 32, 34):
            items.append((f"kl_root_{M}", [raw["root_mu"][str(M)]], T["kl_root"][str(M)]))
        for k in raw["fs_NW"]:
            if k == "patch":
                continue
            K, a = k.split("_", 1)
            items.append((f"fs_kl_{a}_{K}", [raw["fs_NW"][k], raw["fs_SE"][k]], T[f"fs_kl_{a}_{K}"]))
        for md in ("A", "B"):
            for k in raw[f"srS{md}_NW"]:
                if k in ("patch", "tau"):
                    continue
                K, a = k.split("_", 1)
                items.append((f"sr{md}{Ms}_kl_{a}_{K}", [raw[f"srS{md}_NW"][k], raw[f"srS{md}_SE"][k]], T[f"sr{md}{Ms}_kl_{a}_{K}"]))
        for name, qs, ref in items:
            bs = [bounds(p, np.asarray(q, dtype=np.float64)) for q in qs]
            lo, hi, pt = (float(np.mean([b[i] for b in bs])) for i in range(3))
            inside = lo - 1e-9 <= ref <= hi + 1e-9
            n_all += 1
            n_in += inside
            out[f"s{s}:{name}"] = {"lo": lo, "hi": hi, "point": pt, "summary": ref, "inside": inside, "width": hi - lo}
            worst.append((abs(pt - ref), f"s{s}:{name}", lo, hi, ref, inside))
    worst.sort(reverse=True)
    json.dump(out, open(__file__.replace(".py", "_out.json"), "w"), indent=1)
    print(f"KL numbers inside the 5-decimal rounding interval: {n_in}/{n_all}")
    for w in worst[:8]:
        print(f"  |point - summary| {w[0]:.2e} {w[1]}: interval [{w[2]:.7f}, {w[3]:.7f}] summary {w[4]:.7f} inside {w[5]}")
    outside = [k for k, v in out.items() if not v["inside"]]
    print("outside:", outside[:20])


if __name__ == "__main__":
    main()
