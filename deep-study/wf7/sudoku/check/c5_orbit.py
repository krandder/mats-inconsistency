"""c5_orbit.py (checker) -- the held-out set is 9 orbits, not 1,000 independent grids. Per-orbit excess and intervals
that resample ORBITS, for the worker's saved per-grid numbers and for my own fresh grids."""
import json, math
import numpy as np
import ck

H = math.log(28200960); d = np.load(f"{ck.W}/data/k6_base.npz"); bs = d["base_split"]; bo = d["base_orbit"]


def cluster_boot(x, orb, n=4000, seed=0):
    rng = np.random.default_rng(seed); ids = np.unique(orb); groups = [x[orb == o] for o in ids]; out = []
    for _ in range(n):
        pick = rng.integers(len(ids), size=len(ids)); out.append(np.concatenate([groups[j] for j in pick]).mean())
    return float(x.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


def table(title, orb, cols, pairs):
    print(title); ids, cnt = np.unique(orb, return_counts=True)
    print(f"   orbits {len(ids)}; grids per orbit {cnt.tolist()}")
    print("   per-orbit excess  " + "  ".join(f"{k[:14]:>14s}" for k in cols))
    for o in ids:
        print(f"      orbit {o:3d} (n={int((orb == o).sum()):4d}) " + "  ".join(f"{cols[k][orb == o].mean() - H:14.3f}" for k in cols))
    R = {}
    for k, v in cols.items():
        m, lo, hi = cluster_boot(v - H, orb); g = ck.boot(v - H); R[k] = (m, lo, hi)
        print(f"   {k:22s} excess {m:.4f}; interval resampling grids [{g[1]:.3f}, {g[2]:.3f}]; resampling orbits [{lo:.3f}, {hi:.3f}]")
    for a, b in pairs:
        m, lo, hi = cluster_boot(cols[a] - cols[b], orb, seed=1); g = ck.boot(cols[a] - cols[b], seed=1); R[f"{a} - {b}"] = (m, lo, hi)
        print(f"   {a} minus {b}: {m:+.4f}; resampling grids [{g[1]:+.3f}, {g[2]:+.3f}]; resampling orbits [{lo:+.3f}, {hi:+.3f}]")
    return R


OUT = {}
# (1) the worker's own test grids (GridStream seed 303, first 1,000 of 2,000) and its saved per-grid numbers
rng = np.random.default_rng(303); idx = rng.integers(int((bs == 2).sum()), size=2000)[:1000]; orb = bo[bs == 2][idx]
z = np.load(f"{ck.W}/out/e_trust6_mA6_nll.npz")
J = json.load(open(f"{ck.W}/out/e_trust6_mA6.json")); w = J["E2_mixture"]["weights"]
names = {"own confidence: lowest entropy": "own_confidence_lowest_entropy", "own confidence: top probability": "own_confidence_top_probability",
         "own confidence: top-1 minus top-2": "own_confidence_top-1_minus_top-2", "least confident first": "least_confident_first",
         "raster order": "raster_order", "pi_trust (learned on real grids)": "pi_trust_learned_on_real_grids"}
L = -np.stack([z[names[k]] for k in w], 1); ww = np.array([w[k] for k in w]); mix = -(np.log((np.exp(L - L.max(1, keepdims=True)) * ww).sum(1)) + L.max(1))
cols = {"entropy": z["own_confidence_lowest_entropy"], "pi_trust": z["pi_trust_learned_on_real_grids"], "raster": z["raster_order"], "mixture": mix}
OUT["worker"] = table("== the worker's 1,000 test grids and its saved numbers", orb, cols, [("entropy", "pi_trust"), ("raster", "pi_trust"), ("mixture", "raster"), ("mixture", "pi_trust")])
# (2) my fresh grids
zz = np.load("out/c3_routes_nll.npz"); R = json.load(open("out/c3_routes.json")); wm = R["mixture"]["weights"]
rng = np.random.default_rng(9001); orb2 = bo[bs == 2][rng.integers(int((bs == 2).sum()), size=800)]
key = lambda s: s.replace(" ", "_").replace(":", "")
L = -np.stack([zz[key(k)] for k in wm], 1); ww = np.array([wm[k] for k in wm]); mix2 = -(np.log((np.exp(L - L.max(1, keepdims=True)) * ww).sum(1)) + L.max(1))
cols2 = {"entropy": zz[key("own confidence: lowest entropy")], "pi_trust": zz["pi_trust"], "raster": zz[key("raster order")], "mixture": mix2}
OUT["mine"] = table("\n== my 800 fresh test grids", orb2, cols2, [("entropy", "pi_trust"), ("raster", "pi_trust"), ("mixture", "raster"), ("mixture", "pi_trust")])
json.dump(OUT, open("out/c5_orbit.json", "w"), indent=1)
