"""S0 worker 2 stream audit (PREREG-w2.md). Arms A0 wealth-net, A1 v2-static, A2 v2-shift; seeds 1-3.
Writes ../results/w2/audit.json and audit_family_bce.csv."""
import os, sys, json, math, importlib.util
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "substrate")
sys.path.insert(0, SUB)
import streams_ext as S  # noqa: E402
spec = importlib.util.spec_from_file_location("wn_streams", os.path.join(HERE, "..", "..", "..", "..", "wealth-net", "streams.py"))
WN = importlib.util.module_from_spec(spec); spec.loader.exec_module(WN)
OUT = os.path.join(HERE, "..", "results", "w2"); os.makedirs(OUT, exist_ok=True)
W, LR = 32, 0.05


def shift_copy_frac(P, min_overlap=2000, max_shift=4000):
    n, T = P.shape
    hit = np.zeros(n, bool)
    for i in range(n):
        v = swv(P[i, :max_shift + 64], 64)
        for j in range(n):
            if i == j:
                continue
            for sh in np.where((v == P[j, :64]).all(1))[0]:
                if T - sh >= min_overlap and np.array_equal(P[i, sh:], P[j, :T - sh]):
                    hit[i] = hit[j] = True
    return float(hit.mean())


def max_corr(P):
    c = np.corrcoef(P.astype(float))
    return float(np.abs(c[~np.eye(len(P), dtype=bool)]).max())


def online_logistic(bits, active):
    """Prequential BCE [Q,T] (nan where unscored) of per-question logistic on last W bits."""
    Q, T = bits.shape
    x = 2.0 * bits - 1.0
    x = np.where(active, x, 0.0)            # hidden entries read as 0 (= 0.5 unknown)
    w = np.zeros((Q, W)); b = np.zeros(Q)
    L = np.full((Q, T), np.nan)
    for t in range(W, T):
        xt = x[:, t - W:t]
        z = (w * xt).sum(1) + b
        p = 1.0 / (1.0 + np.exp(-z))
        y = bits[:, t]
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        m = active[:, t]
        L[m, t] = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))[m]
        g = (y - p) * m
        w += LR * g[:, None] * xt; b += LR * g
    return L


def fam_bce(L, fam_t, fams, names):
    out = {}
    for f in fams:
        m = (fam_t == f) & ~np.isnan(L)
        out[names[f]] = float(L[m].mean()) if m.any() else None
    return out


def run(seed):
    r = {"seed": seed}
    # A0 wealth-net
    b0, f0, _ = WN.make_questions(64, 24000, seed)
    r["A0_shift_copy_pi"] = shift_copy_frac(b0[f0 == 5])
    r["A0_maxcorr_pi"] = max_corr(b0[f0 == 5])
    L0 = online_logistic(b0, np.ones_like(b0, bool))
    r["A0_bce"] = fam_bce(L0, np.repeat(f0[:, None], b0.shape[1], 1), range(8), WN.FAMILIES)
    # A1 v2-static
    s1 = S.make_streams(64, 60000, seed, regime_every=0)
    P = s1.bits[s1.fam0 == 5]
    r["A1_shift_copy_prng"] = shift_copy_frac(P)
    r["A1_maxcorr_prng"] = max_corr(P)
    L1 = online_logistic(s1.bits, s1.active)
    r["A1_bce"] = fam_bce(L1, s1.fam_t, range(9), S.FAMILIES)
    ho = s1.fam0 == S.HO
    r["A1_ho_bce_first1000_after_release"] = float(np.nanmean(L1[ho, s1.release:s1.release + 1000]))
    r["A1_ho_bce_last10000"] = float(np.nanmean(L1[ho, -10000:]))
    # A2 v2-shift
    s2 = S.make_streams(64, 60000, seed, regime_every=10000, regime_frac=0.25)
    assert np.array_equal(s2.bits[:, :10000], s1.bits[:, :10000])
    L2 = online_logistic(s2.bits, s2.active)
    r["A2_bce"] = fam_bce(L2, s2.fam_t, range(9), S.FAMILIES)
    sw = [e for e in s2.events if e["kind"] == "switch"]
    per_b, fracs = [], []
    for bnd in sorted({e["t"] for e in sw}):
        qs = [e["q"] for e in sw if e["t"] == bnd]
        fracs.append(len(qs) / s2.n_main)
        before = np.nanmean(L2[qs, bnd - 1000:bnd]); after = np.nanmean(L2[qs, bnd:bnd + 1000])
        # non-prng targets only: prng -> anything is not a learnable-to-learnable shift
        per_b.append({"t": bnd, "n": len(qs), "before": float(before), "after": float(after),
                      "delta": float(after - before)})
    r["A2_switch_frac"] = fracs
    r["A2_shift"] = per_b
    r["A2_shift_delta_mean"] = float(np.mean([p["delta"] for p in per_b]))
    r["A2_indirect_events"] = sum(e["kind"] == "indirect" for e in s2.events)
    r["A2_total_bce_vs_A1"] = [float(np.nanmean(L2)), float(np.nanmean(L1))]
    return r


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    res = []
    for sd in seeds:
        r = run(sd); res.append(r); print(json.dumps(r)[:400], flush=True)
        json.dump(r, open(os.path.join(OUT, f"audit-s{sd}.json"), "w"), indent=1)
