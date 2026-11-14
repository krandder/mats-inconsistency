"""M3-w2: currency lock-in for committed traders on the S0 regime-shift stream (see PREREG.md).
Usage: python3 lockin.py SEED [T]  -> results/seed{SEED}.json, results/switches_s{SEED}.csv
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
from streams_ext import make_streams, FAMILIES  # noqa: E402

W, CONF, EPS = 32, 0.95, 1e-3
ARMS = ["kelly", "ema100", "ema1k", "ema10k", "fs1e-3", "fs1e-2", "win500",
        "uniform", "shuffled", "grad0.1", "grad1"]
LAM = {"ema100": 0.5 ** (1 / 100), "ema1k": 0.5 ** (1 / 1000), "ema10k": 0.5 ** (1 / 10000)}
FS = {"fs1e-3": 1e-3, "fs1e-2": 1e-2}
GR = {"grad0.1": 0.1, "grad1": 1.0}
LFSR_TAPS = (16, 14, 13, 11)


def softmax(v):
    v = v - v.max(-1, keepdims=True)
    e = np.exp(v)
    return e / e.sum(-1, keepdims=True)


def run(seed, T=40000, regime_every=10000):
    s = make_streams(Q=64, T=T, seed=seed, heldout=8, regime_every=regime_every, regime_frac=0.25)
    bits = s.bits.astype(np.float64); act = s.active; Qt = bits.shape[0]
    N = 1 + 32 + 1 + 1; A = len(ARMS)
    rng = np.random.default_rng(1000 + seed)
    V = np.zeros((A, Qt, N)); acc = np.zeros((Qt, N))  # acc: win500 current-window income
    n1 = np.zeros(Qt); n = np.zeros(Qt)
    lw = np.zeros((Qt, W + 1))
    LL = np.full((A, Qt, T), np.nan, np.float32)
    ia = {a: i for i, a in enumerate(ARMS)}
    for t in range(W, T):
        y = bits[:, t]; m = act[:, t]
        win = bits[:, t - W:t][:, ::-1]           # win[:, k-1] = y_{t-k}
        P = np.empty((Qt, N))
        P[:, 0] = (n1 + 1) / (n + 2)
        P[:, 1:33] = np.where(win > 0.5, CONF, 1 - CONF)
        par = np.zeros(Qt)
        for k in LFSR_TAPS:
            par = np.logical_xor(par, win[:, k - 1] > 0.5)
        P[:, 33] = np.where(par, CONF, 1 - CONF)
        x = np.concatenate([2 * win - 1, np.ones((Qt, 1))], 1)
        pl = 1 / (1 + np.exp(-(x * lw).sum(1)))
        P[:, 34] = np.clip(pl, 0.02, 0.98)
        P = np.clip(P, EPS, 1 - EPS)
        Py = np.where(y[:, None] > 0.5, P, 1 - P)            # [Qt,N] prob of realised label
        ys = (rng.random(Qt) < 0.5).astype(np.float64)
        Pys = np.where(ys[:, None] > 0.5, P, 1 - P)
        Wt = softmax(V)                                       # [A,Qt,N]
        py = (Wt * Py[None]).sum(-1)                          # [A,Qt]
        LL[:, :, t] = np.where(m[None], -np.log(py), np.nan)
        S = np.log(Py)[None] - np.log(py)[..., None]          # settlement per arm
        mm = m[None, :, None]
        for a, i in ia.items():
            if a == "kelly":
                V[i] += S[i] * mm[0]
            elif a in LAM:
                V[i] = np.where(mm[0], LAM[a] * V[i] + S[i], V[i])
            elif a in FS:
                w2 = softmax(V[i] + S[i]); al = FS[a]
                V[i] = np.where(mm[0], np.log((1 - al) * w2 + al / N), V[i])
            elif a == "win500":
                acc += S[i] * mm[0]
                if (t + 1) % 500 == 0:
                    V[i] = acc; acc = np.zeros_like(acc)
            elif a == "shuffled":
                pys = (Wt[i] * Pys).sum(-1)
                V[i] += (np.log(Pys) - np.log(pys)[:, None]) * mm[0]
            elif a in GR:
                r = Py / py[i][:, None]
                V[i] += GR[a] * Wt[i] * (r - 1) * mm[0]
        # prior and logistic learner update after booking (prequential)
        n1 += y * m; n += m
        lw += 0.1 * ((y - pl) * m)[:, None] * x
    return s, LL


def metrics(s, LL, T):
    sw = [e for e in s.events if e["kind"] == "switch"]
    learn = [e for e in sw if e["to"] in (0, 1)]
    switched = {e["q"] for e in sw}
    indirect = {e["q"] for e in s.events if e["kind"] == "indirect"}
    stat_q = [q for q in range(s.n_main) if s.fam0[q] in (0, 1) and q not in switched]
    rows, out = [], {}
    for ai, a in enumerate(ARMS):
        post, post_all, lags = [], [], []
        for e in sw:
            seg = LL[ai, e["q"], e["t"]:e["t"] + 2000]
            post_all.append(float(np.nanmean(seg)))
        for e in learn:
            q, b = e["q"], e["t"]
            post.append(float(np.nanmean(LL[ai, q, b:b + 2000])))
            ll = LL[ai, q, b:min(b + 10000, T)].astype(np.float64)
            c = np.concatenate([[0], np.cumsum(ll)])
            tr = (c[100:] - c[:-100]) / 100
            hit = np.nonzero(tr < 0.1)[0]
            lag = int(hit[0] + 100) if len(hit) else 10000
            lags.append(lag)
            rows.append(dict(arm=a, q=q, t=b, frm=FAMILIES[e["from"]], to=FAMILIES[e["to"]],
                             post2k=post[-1], lag=lag, censored=int(len(hit) == 0)))
        stat = float(np.nanmean(LL[ai, stat_q, 5000:]))
        out[a] = dict(post2k_learnable=float(np.mean(post)), post2k_all_switches=float(np.mean(post_all)),
                      lag_median=float(np.median(lags)), lag_mean=float(np.mean(lags)),
                      lag_censored=int(sum(l >= 10000 for l in lags)), n_learnable=len(lags),
                      stationary=stat, overall=float(np.nanmean(LL[ai])))
    meta = dict(n_switch=len(sw), n_learnable=len(learn), n_stationary_q=len(stat_q),
                n_indirect_q=len(indirect))
    return out, rows, meta


if __name__ == "__main__":
    seed = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    t0 = time.time()
    s, LL = run(seed, T)
    out, rows, meta = metrics(s, LL, T)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    meta.update(seed=seed, T=T, wall_s=round(time.time() - t0, 1), arms=ARMS)
    json.dump(dict(meta=meta, arms=out), open(os.path.join(HERE, "results", f"seed{seed}.json"), "w"), indent=1)
    import csv
    with open(os.path.join(HERE, "results", f"switches_s{seed}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # per-arm mean loss curve around boundaries (learnable switches), for the plot
    sw = [e for e in s.events if e["kind"] == "switch" and e["to"] in (0, 1)]
    curve = np.stack([np.nanmean(np.stack([LL[ai, e["q"], e["t"] - 500:e["t"] + 4000] for e in sw]), 0)
                      for ai in range(len(ARMS))])
    np.save(os.path.join(HERE, "results", f"curve_s{seed}.npy"), curve.astype(np.float32))
    print(json.dumps(meta)); [print(a, {k: round(v, 4) for k, v in d.items()}) for a, d in out.items()]
