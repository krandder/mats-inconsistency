# trust_rich.py CALIB.jsonl FRESH.jsonl : does a trust head that reads search statistics (found score included), fitted
# on real answers with the work's eight writers, flag a writer it never saw (W5, HID)?  Checker's code; K = 16.
import sys, json, math
from collections import Counter, defaultdict
import numpy as np
cal = [json.loads(l) for l in open(sys.argv[1])]; new = [json.loads(l) for l in open(sys.argv[2])]
K = 16
lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
top = lambda d: max(d, key=lambda a: (d[a], a))

def feats(r, ids, lq):
    v = Counter(r["dans"][i] for i in ids if i >= 0)
    if not v:
        return None
    RF = defaultdict(float)
    for i in set(i for i in ids if i >= 0):
        RF[r["dans"][i]] += math.exp(r["dlr"][i])
    mF = sum(RF.values())
    lw = [(r["dlr"][i] - q, i) for i, q in zip(ids, lq) if i >= 0]; mx = max(x for x, _ in lw)
    S = defaultdict(float); ws = []
    for x, i in lw:
        e = math.exp(x - mx); S[r["dans"][i]] += e; ws.append(e)
    a_d = top(RF); a_s = top(S); a_v = top(v)
    ess = sum(ws) ** 2 / sum(x * x for x in ws) / len(ids)
    base = {"dedup": (a_d, RF[a_d] / mF), "snis": (a_s, S[a_s] / sum(ws))}
    out = {}
    for g, (a, p) in base.items():
        out[g] = ([1.0, lg(p), float(r["fam"] == "vars")],
                  [1.0, lg(p), float(r["fam"] == "vars"), lg(v[a_v] / sum(v.values())), float(a_v == a), ess, math.log(max(mF, 1e-6)),
                   float(np.mean([i >= 0 for i in ids]))], float(a == r["answer"]), p)
    return out

def rows(R, w):
    res = {g: ([], [], [], []) for g in ("dedup", "snis")}
    for r in R:
        if w not in r["w"]:
            continue
        wr = r["w"][w]
        for b in range(len(wr["id"]) // K):
            f = feats(r, wr["id"][b * K:(b + 1) * K], wr["lq"][b * K:(b + 1) * K])
            if f is None:
                continue
            for g in f:
                for lst, x in zip(res[g], f[g]):
                    lst.append(x)
    return res

def fit(X, y, l2=1e-2):
    X = np.array(X); y = np.array(y); w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        w -= np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w)), X.T @ (p - y) + l2 * w)
    return w

# the simulated hider on the fresh problems (M's reference rollouts without M's top-scoring line), as a writer "HID"
for r in new:
    mids = [i for i in r["w"]["W1"]["id"] + r["ref"]["id"] if i >= 0]
    if mids:
        t = max(set(mids), key=lambda i: r["dlr"][i]); pt = math.exp(r["dlr"][t])
        keep = [(i, q - math.log(max(1 - pt, 1e-9))) for i, q in zip(r["ref"]["id"], r["ref"]["lq"]) if i != t][:K]
        if len(keep) == K:
            r["w"]["HID"] = {"id": [i for i, _ in keep], "lq": [q for _, q in keep]}
calw = list(cal[0]["w"].keys())
for g in ("dedup", "snis"):
    Xs, Xr, y = [], [], []
    for w in calw:
        a = rows(cal, w)[g]; Xr += a[1]; y += a[2]
        if w == "W1":
            Xs1, y1 = a[0], a[2]
    ws_, wr_ = fit(Xs1, y1), fit(Xr, y)
    print("\n%s: rich weights [bias, logit p, vars, logit vote, agree, ess share, log found, valid share] = %s" % (g, np.round(wr_, 2).tolist()))
    for w in ("W1", "W3", "W3r", "W3s", "W3p", "W4", "W5", "HID"):
        a = rows(new, w)[g]
        if not a[2]:
            continue
        yt = np.array(a[2]); raw = np.array(a[3]); ps = 1 / (1 + np.exp(-np.array(a[0]) @ ws_)); pr = 1 / (1 + np.exp(-np.array(a[1]) @ wr_))
        print("  %-4s n %4d acc %.3f | Brier raw %.3f, simple head %.3f (conf %.2f), rich head %.3f (conf %.2f)" % (
            w, len(yt), yt.mean(), ((raw - yt) ** 2).mean(), ((ps - yt) ** 2).mean(), ps.mean(), ((pr - yt) ** 2).mean(), pr.mean()))
