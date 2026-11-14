"""Driver for M5/w3/PREREG.md. usage: run_w3.py SOURCE SEED   (SOURCE in P4, P16, wn)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
import numpy as np
import pattern as P

src, seed = sys.argv[1], int(sys.argv[2])
BULK = "/tmp/traders-net/M5/w3"
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True); os.makedirs(BULK, exist_ok=True)
from streams import make_questions, FAMILIES
NF = len(FAMILIES)
W, Q = 32, 64


def cum_slope(c):
    t = np.arange(len(c), dtype=np.float64); t -= t.mean()
    return float((t * (c - c.mean())).sum() / (t * t).sum())


def summarize(arm, out, mkt, lo, hi):
    n = hi - lo
    res = {"arm": arm, "seed": seed, "lo": lo, "hi": hi, "Q": Q, "families": FAMILIES,
           "mkt_bce": (mkt["bce"].sum(0) / (n * np.bincount(fam, minlength=NF).clip(1))).tolist(),
           "mkt_bce_all": float(mkt["bce"].sum() / (n * Q)), "traders": {}}
    series = {}
    for cur in ("log", "share"):
        cums = {}
        for name, v in out.items():
            if v[cur] is None:
                continue
            tot = v[cur].sum(1)
            st = P.ec_stats(tot, Q)
            h20 = int(n * 0.8)
            st["phi_fam"] = [P.slope(v[cur][h20:, f]) / max(1, (fam == f).sum()) for f in range(NF)]
            res["traders"].setdefault(name, {})[cur] = st
            cums[name] = np.cumsum(tot) / Q
        names = sorted(cums)
        M = np.stack([cums[k] for k in names])
        ec = M.max(0)
        h = n // 2
        res[f"EC_{cur}"] = {"slope_h1": cum_slope(ec[:h]), "slope_h2": cum_slope(ec[h:]),
                            "phi_last20": cum_slope(ec[int(n * 0.8):]), "E_end": float(ec[-1]),
                            "argmax_end": names[int(M[:, -1].argmax())]}
        for k in names:
            series[f"{cur}:{k}"] = cums[k][::200]
        series[f"{cur}:EC"] = ec[::200]
    with open(os.path.join(RES, f"{arm}-s{seed}-series.csv"), "w") as fh:
        keys = list(series)
        fh.write("pos," + ",".join(keys) + "\n")
        for r in range(len(series[keys[0]])):
            fh.write(f"{lo + 200 * r}," + ",".join(f"{series[k][r]:.5f}" for k in keys) + "\n")
    np.savez_compressed(os.path.join(BULK, f"{arm}-s{seed}.npz"),
                        **{f"{c}:{k}": v[c] for k, v in out.items() for c in ("log", "share") if v[c] is not None})
    json.dump(res, open(os.path.join(RES, f"{arm}-s{seed}.json"), "w"), indent=1)
    print(arm, seed, "EC_log", res.get("EC_log"), "EC_share", res["EC_share"], flush=True)


t0 = time.time()
if src in ("P4", "P16"):
    d = np.load(f"/tmp/traders-net/S0/w1/{src}-s{seed}.npz")
    bits, fam, _ = make_questions(Q, 64032, seed)
    assert (d["fam"] == fam).all() and (d["y"] == bits[:, W:].T).all()
    for mk, key in (("net", "z"), ("log", "zb")):
        z = d[key]
        out, mkt = P.run_class(bits, W, z, 0, z.shape[0], fam, NF)
        summarize(f"{mk}-{src}", out, mkt, 0, z.shape[0])
else:
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, "/home/agent/workspaces/eurisko-study/wealth-net")
    from model import Net
    T = 24000
    bits, fam, _ = make_questions(Q, T, seed)
    n_pos = T - W; split = int(n_pos * 0.8)
    wins = torch.tensor(bits, dtype=torch.float32).unfold(1, W, 1)   # [Q, T-W+1, W]
    wdir = "/home/agent/workspaces/eurisko-study/wealth-net/results"
    for tag, path in (("wn-8k", f"{wdir}/weights/train-s{seed}-step8000.pt"),
                      ("wn-20k", f"{wdir}/weights/train-s{seed}-step20000.pt"),
                      ("wn-final", f"{wdir}/train-s{seed}.pt")):
        m = Net(Q, W, 64, 4, 2); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        z = np.full((n_pos, Q), np.nan, np.float32)
        with torch.no_grad():
            for a in range(split, n_pos, 512):
                b = min(n_pos, a + 512)
                z[a:b] = m(wins[:, a:b].permute(1, 0, 2).contiguous()).numpy()
        out, mkt = P.run_class(bits, W, z, split, n_pos, fam, NF)
        summarize(tag, out, mkt, split, n_pos)
print("done", src, seed, round(time.time() - t0, 1), "s", flush=True)
