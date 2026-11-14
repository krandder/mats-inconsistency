"""Active structure elicitation: which simple questions recover a forecaster's dependency structure fastest?

Offline analysis over the dense query pool from scripts/active_structure_collect.py
(24 sets of n=12 events, Qwen3-32B, t=0).  Asking a query = reading its stored answer.

Belief model (per set): Ising / log-linear over x in {0,1}^12, exact enumeration of 4096 states,
    log p(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j - log Z.
Answer model: logit(stated) = logit(implied_q(theta)) + b_kind + eps,  eps ~ N(0, sigma_kind^2),
    kinds single (b=0, the reference level), and2, cond; b_and2 and b_cond are learned
    parameters with priors from the calibration sets; and3 (held out only) uses b_and2.
Edge prior: spike-and-slab z_ij ~ Bern(pi_ij), J | z ~ N(0, slab^2) or N(0, spike^2).  For the
    Laplace fit the prior is moment-matched to one Gaussian per edge; the edge posterior
    P(z=1 | data) is recovered per edge from the Laplace marginal (data message = posterior /
    prior).  Screening lists ("which of the others would change your forecast of A?") update
    pi_ij through a calibrated likelihood (P(listed | z), by list-size bucket).
Posterior: MAP by L-BFGS, Laplace covariance from the Gauss-Newton Hessian.
Selection (batch of 20 per round, 1 call per query, no query asked twice since t=0 answers
    are deterministic):
    random      uniform over the unasked pair queries (and2 + cond)
    similar     pairs by TF-IDF cosine similarity of the question texts, each pair's and2 + 2 conds
    screen      all 12 screening lists first, then flagged pairs (2 flags, then 1), then the rest at random
    eig         greedy batch maximising expected reduction of edge-existence entropy
                sum_e H(z_e) - E[H(z_e | answers)] (linear-Gaussian preposterior, Gauss-Hermite),
                with screening lists as a query type with their own likelihood
    eig_pred    greedy batch maximising information about all 220 3-way joint answers
                (sum_t 0.5 log var_t / var_t' under the linearised posterior)
Reference: the dense fit (all non-held-out numeric queries, no screens) defines J_ref and
    strong edges |J_ref| >= TAU.  Held out: a fixed 20% of pair queries + 12 3-way ANDs.
Metrics vs number of queries (singles are asked first by every method and counted):
    edge AUC / F1 vs strong reference edges, held-out answer KL (stated p as soft label),
    log loss vs real outcomes of the coherent model on singles, pairwise ANDs, 3-way ANDs.

Usage:
  active_structure.py calibrate --data /home/agent/data/active-structure-20260924
  active_structure.py evaluate  --data /home/agent/data/active-structure-20260924 --workers 12
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

N = 12
PAIRS = list(itertools.combinations(range(N), 2))
PIDX = {p: k for k, p in enumerate(PAIRS)}
NP = len(PAIRS)
X = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)          # 4096 x 12
FEAT = np.concatenate([X, np.stack([X[:, i] * X[:, j] for i, j in PAIRS], 1)], 1)  # 4096 x 78
NTH = N + NP + 2                                                                 # + b_and2, b_cond
B_AND, B_COND = N + NP, N + NP + 1
CLIP = 0.01
SEED = 20260924
N_CAL = 6
BATCH = 20
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9)
GH_W = GH_W / GH_W.sum()


def ent(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ---------------------------------------------------------------- queries -----

def event_masks(q):
    """(E, C) boolean masks over states: logit implied = log P(E) - log P(C and not E)."""
    f = q["formula"]
    if f["op"] == "var":
        E = X[:, f["i"]] > 0; C = np.ones(len(X), bool)
    elif f["op"] == "and":
        E = np.all(X[:, [a["i"] for a in f["args"]]] > 0, 1); C = np.ones(len(X), bool)
    elif f["op"] == "cond":
        g, t = f["given"]["i"], f["then"]["i"]
        C = X[:, g] > 0; E = C & (X[:, t] > 0)
    else:
        raise ValueError(f)
    return E, C & ~E


def pair_of(q):
    f = q["formula"]
    if f["op"] == "cond":
        return tuple(sorted((f["given"]["i"], f["then"]["i"])))
    if f["op"] == "and" and len(f["args"]) == 2:
        return tuple(sorted(a["i"] for a in f["args"]))
    return None


class SetData:
    def __init__(self, s, qs, recs, holdout_frac=0.2):
        self.s = s
        self.labels = np.array([e["label"] for e in s["events"]])
        num = [q for q in qs if q["kind"] != "screen"]
        self.q = num
        self.kind = [q["kind"] for q in num]
        self.y = np.array([logit(np.clip(recs[q["query_id"]]["p"], CLIP, 1 - CLIP)) for q in num])
        self.p_stated = np.clip(np.array([recs[q["query_id"]]["p"] for q in num]), CLIP, 1 - CLIP)
        m = [event_masks(q) for q in num]
        self.ME = np.stack([a for a, _ in m]).astype(float)
        self.MN = np.stack([b for _, b in m]).astype(float)
        self.bias_col = np.array([B_AND if k in ("and2", "and3") else B_COND if k == "cond" else -1 for k in self.kind])
        self.pair = [pair_of(q) for q in num]
        rng = random.Random(f"{SEED}-{s['set_id']}-holdout")
        pairq = [k for k, q in enumerate(num) if q["kind"] in ("and2", "cond")]
        ho = set(rng.sample(pairq, int(round(holdout_frac * len(pairq)))))
        self.heldout = sorted(ho | {k for k, q in enumerate(num) if q["kind"] == "and3"})
        self.singles = [k for k, q in enumerate(num) if q["kind"] == "single"]
        self.pool = [k for k in pairq if k not in ho]
        self.screens = {}
        for q in qs:
            if q["kind"] == "screen":
                r = recs[q["query_id"]]["p"]
                self.screens[q["target"]] = set(r) if r is not None else None
        # all 220 triples as formulas (for eig_pred; no answers used)
        tri = [{"formula": {"op": "and", "args": [{"op": "var", "i": i} for i in t]}}
               for t in itertools.combinations(range(N), 3)]
        tm = [event_masks(q) for q in tri]
        self.TE = np.stack([a for a, _ in tm]).astype(float)
        self.TN = np.stack([b for _, b in tm]).astype(float)


def implied(theta, ME, MN, bias_col=None):
    """logit implied answers and their gradients (Q x NTH)."""
    e = FEAT @ theta[:N + NP]
    p = np.exp(e - logsumexp(e))
    pe, pn = ME @ p, MN @ p
    pe = np.maximum(pe, 1e-300); pn = np.maximum(pn, 1e-300)
    g = np.log(pe) - np.log(pn)
    G = np.zeros((len(ME), NTH))
    G[:, :N + NP] = (ME * p) @ FEAT / pe[:, None] - (MN * p) @ FEAT / pn[:, None]
    if bias_col is not None:
        for k, c in enumerate(bias_col):
            if c >= 0:
                g[k] += theta[c]; G[k, c] = 1.0
    return g, G, p


# ---------------------------------------------------------------- model -------

class Model:
    def __init__(self, cal):
        self.cal = cal
        self.sig = cal["sigma"]
        self.pi0, self.slab, self.spike = cal["pi0"], cal["slab"], cal["spike"]

    def prior(self, pi_e):
        mu = np.zeros(NTH); var = np.zeros(NTH)
        mu[:N] = self.cal["h_mu"]; var[:N] = self.cal["h_sd"] ** 2
        var[N:N + NP] = pi_e * self.slab ** 2 + (1 - pi_e) * self.spike ** 2
        mu[B_AND], mu[B_COND] = self.cal["b"]["and2"], self.cal["b"]["cond"]
        var[B_AND] = var[B_COND] = self.cal["b_sd"] ** 2
        return mu, var

    def screen_pi(self, sd: SetData, screened):
        lo = np.full(NP, logit(self.pi0))
        r = self.cal["screen"]
        for t in screened:
            L = sd.screens.get(t)
            if L is None:
                continue
            b = bucket(len(L))
            r1, r0 = r[b]["r1"], r[b]["r0"]
            for j in range(N):
                if j == t:
                    continue
                e = PIDX[tuple(sorted((t, j)))]
                lo[e] += math.log(r1 / r0) if j in L else math.log((1 - r1) / (1 - r0))
        return expit(lo)

    def fit(self, sd: SetData, idx, screened=(), theta0=None, free_bias=False):
        idx = np.array(sorted(idx))
        pi_e = self.screen_pi(sd, screened)
        mu, var = self.prior(pi_e)
        if free_bias:
            var[B_AND] = var[B_COND] = 1.0; mu[B_AND] = mu[B_COND] = 0.0
        ME, MN, bc, y = sd.ME[idx], sd.MN[idx], sd.bias_col[idx], sd.y[idx]
        w = np.array([1 / self.sig[sd.kind[k]] ** 2 for k in idx])

        def f(th):
            g, G, _ = implied(th, ME, MN, bc)
            r = g - y
            d = th - mu
            return 0.5 * np.sum(w * r * r) + 0.5 * np.sum(d * d / var), G.T @ (w * r) + d / var

        th0 = mu.copy() if theta0 is None else theta0.copy()
        res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        th = res.x
        g, G, p = implied(th, ME, MN, bc)
        H = G.T @ (G * w[:, None]) + np.diag(1 / var)
        S = np.linalg.inv(H)
        return {"theta": th, "S": S, "var": var, "pi": pi_e, "p": p, "idx": idx, "screened": tuple(screened)}

    def edge_prob(self, fit, m=None, s2=None):
        """P(z=1 | data) per edge from the Laplace marginal (m, s2) and the prior variance used."""
        if m is None:
            m = fit["theta"][N:N + NP]
        if s2 is None:
            s2 = np.diag(fit["S"])[N:N + NP]
        v = fit["var"][N:N + NP]
        return edge_prob_from(m, s2, v, fit["pi"], self.slab, self.spike)


def edge_prob_from(m, s2, v, pi, slab, spike):
    lam = np.maximum(1 / s2 - 1 / v, 1e-9)
    mu = (m / s2) / lam
    a = 1 / lam + slab ** 2; b = 1 / lam + spike ** 2
    l1 = -0.5 * (np.log(a) + mu * mu / a); l0 = -0.5 * (np.log(b) + mu * mu / b)
    return expit(np.log(pi) - np.log1p(-pi) + l1 - l0)


def bucket(k):
    return "small" if k <= 3 else "mid" if k <= 7 else "large"


# ---------------------------------------------------------------- selection ---

def exp_entropy(m, s2_now, s2_new, v, pi, slab, spike):
    """E over the preposterior of the posterior mean of H(z) after the variance drops to s2_new."""
    spread = np.sqrt(np.maximum(s2_now - s2_new, 0))
    tot = 0
    for x, wgt in zip(GH_X, GH_W):
        tot = tot + wgt * ent(edge_prob_from(m + spread * x, s2_new, v, pi, slab, spike))
    return tot


def select_eig(model, sd, fit, avail, screens_avail, k, rng):
    th, S = fit["theta"], fit["S"].copy()
    m = th[N:N + NP]; s2_now = np.diag(S)[N:N + NP].copy(); v = fit["var"][N:N + NP]; pi = fit["pi"]
    sl, sp = model.slab, model.spike
    avail = list(avail)
    if avail:
        g, G, _ = implied(th, sd.ME[avail], sd.MN[avail], sd.bias_col[avail])
        sig2 = np.array([model.sig[sd.kind[q]] ** 2 for q in avail])
    Pnow = edge_prob_from(m, s2_now, v, pi, sl, sp)
    base = exp_entropy(m, s2_now, s2_now, v, pi, sl, sp)
    cur = base.copy()
    r = model.cal["screen"]["pooled"]
    chosen, chosen_scr = [], []
    scr_left = set(screens_avail)
    alive = np.ones(len(avail), bool)
    for _ in range(k):
        best, bestgain = None, -1
        if avail and alive.any():
            SG = G @ S                                      # Qa x NTH
            d = np.einsum("qi,qi->q", SG, G) + sig2
            s2_new = np.diag(S)[N:N + NP][None, :] - SG[:, N:N + NP] ** 2 / d[:, None]
            eh = exp_entropy(m[None, :], s2_now[None, :], s2_new, v[None, :], pi[None, :], sl, sp)
            gain = (cur[None, :] - eh).sum(1) + 1e-9 * rng.random()
            gain[~alive] = -np.inf
            j = int(np.argmax(gain))
            best, bestgain = ("q", j), gain[j]
        for t in scr_left:  # screening list for target t: info about its 11 edges
            es = [PIDX[tuple(sorted((t, j)))] for j in range(N) if j != t]
            P = Pnow[es]
            py = P * r["r1"] + (1 - P) * r["r0"]
            h1 = ent(P * r["r1"] / py); h0 = ent(P * (1 - r["r1"]) / (1 - py))
            # information of the screen measured against the current expected entropy
            gs = float(np.sum(np.minimum(cur[es], ent(P)) - np.minimum(cur[es], py * h1 + (1 - py) * h0)))
            if gs > bestgain:
                best, bestgain = ("s", t), gs
        if best is None:
            break
        if best[0] == "q":
            j = best[1]
            sg = S @ G[j]
            S = S - np.outer(sg, sg) / (G[j] @ sg + sig2[j])
            s2b = np.diag(S)[N:N + NP]
            cur = exp_entropy(m, s2_now, s2b, v, pi, sl, sp)
            alive[j] = False
            chosen.append(avail[j])
        else:
            t = best[1]
            scr_left.discard(t); chosen_scr.append(t)
            es = [PIDX[tuple(sorted((t, j)))] for j in range(N) if j != t]
            P = Pnow[es]; py = P * r["r1"] + (1 - P) * r["r0"]
            cur[es] = np.minimum(cur[es], py * ent(P * r["r1"] / py) + (1 - py) * ent(P * (1 - r["r1"]) / (1 - py)))
    return chosen, chosen_scr


def select_eig_pred(model, sd, fit, avail, k, rng):
    th, S = fit["theta"], fit["S"].copy()
    avail = list(avail)
    if not avail:
        return []
    _, G, _ = implied(th, sd.ME[avail], sd.MN[avail], sd.bias_col[avail])
    _, T, _ = implied(th, sd.TE, sd.TN)
    sig2 = np.array([model.sig[sd.kind[q]] ** 2 for q in avail])
    chosen = []
    alive = np.ones(len(avail), bool)
    for _ in range(min(k, len(avail))):
        TS = T @ S
        vt = np.einsum("ti,ti->t", TS, T)
        C = TS @ G.T                                        # T x Qa cov(triple, query)
        d = np.einsum("qi,qi->q", G @ S, G) + sig2
        vnew = np.maximum(vt[:, None] - C ** 2 / d[None, :], 1e-12)
        gain = 0.5 * np.log(vt[:, None] / vnew).sum(0) + 1e-9 * rng.random(len(avail))
        gain[~alive] = -np.inf
        j = int(np.argmax(gain))
        sg = S @ G[j]
        S = S - np.outer(sg, sg) / (G[j] @ sg + sig2[j])
        alive[j] = False; chosen.append(avail[j])
    return chosen


def bge_sim(sd, data):
    """BGE-small cosine (as in the global run): stored per set for aligned sets, else bge-cosine.json."""
    C = sd.s.get("cosine")
    if C is None:
        f = Path(data) / "bge-cosine.json"
        if not f.exists():
            return tfidf_sim(sd)
        C = json.loads(f.read_text())[sd.s["set_id"]]
    return {p: C[p[0]][p[1]] for p in PAIRS}


def tfidf_sim(sd):
    docs = [(e["parent_question"] + " " + (e["criteria_text"] or "")[:400]).lower() for e in sd.s["events"]]
    toks = [re.findall(r"[a-z][a-z0-9]+", d) for d in docs]
    stop = {"the", "will", "and", "for", "this", "that", "resolve", "resolves", "yes", "market", "before", "end", "question",
            "with", "are", "any", "not", "be", "by", "if", "or", "of", "in", "to", "on", "is", "at", "as", "an", "it"}
    toks = [[t for t in ts if t not in stop] for ts in toks]
    df = Counter(t for ts in toks for t in set(ts))
    vec = []
    for ts in toks:
        c = Counter(ts)
        v = {t: (1 + math.log(n)) * math.log((1 + N) / (1 + df[t])) for t, n in c.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1
        vec.append({t: x / nrm for t, x in v.items()})
    return {p: sum(vec[p[0]].get(t, 0) * x for t, x in vec[p[1]].items()) for p in PAIRS}


def order_by_pairs(sd, pair_order, rng):
    byp = defaultdict(list)
    for k in sd.pool:
        byp[sd.pair[k]].append(k)
    out = []
    for p in pair_order:
        qs = sorted(byp[p], key=lambda k: sd.kind[k] != "and2")
        out += qs
    return out


# ---------------------------------------------------------------- metrics -----

def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics(model, sd, fit, zref):
    P = np.round(model.edge_prob(fit), 4)   # rounding: equal-prior edges tie instead of ordering by float noise
    th = fit["theta"]
    ho = np.array(sd.heldout)
    g, _, p = implied(th, sd.ME[ho], sd.MN[ho], sd.bias_col[ho])
    q = np.clip(expit(g), 1e-6, 1 - 1e-6); ps = sd.p_stated[ho]
    kl = ps * np.log(ps / q) + (1 - ps) * np.log((1 - ps) / (1 - q))
    # outcomes: coherent model beliefs (no answer bias) on singles, and2, and3
    out = {}
    lab = sd.labels
    marg = X.T @ p
    out["single"] = ll(np.clip(marg, CLIP, 1 - CLIP), lab)
    pp = np.array([p @ (X[:, i] * X[:, j]) for i, j in PAIRS])
    out["and2"] = ll(np.clip(pp, CLIP, 1 - CLIP), np.array([lab[i] * lab[j] for i, j in PAIRS]))
    a3 = [k for k in range(len(sd.q)) if sd.kind[k] == "and3"]
    p3 = sd.ME[a3] @ p
    out["and3"] = ll(np.clip(p3, CLIP, 1 - CLIP), np.array([sd.q[k]["label"] for k in a3]))
    pred = (P >= 0.5).astype(int)
    tp = int(((pred == 1) & (zref == 1)).sum()); fp = int(((pred == 1) & (zref == 0)).sum())
    fn = int(((pred == 0) & (zref == 1)).sum())
    return {"heldout_kl": float(kl.mean()), "auc": auc(P, zref), "tp": tp, "fp": fp, "fn": fn,
            "P": P.tolist(), "J": np.round(th[N:N + NP], 4).tolist(), "b": [float(th[B_AND]), float(th[B_COND])],
            "out": out}


def ll(p, y):
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ---------------------------------------------------------------- runs --------

def load(data: Path):
    sets = [json.loads(l) for l in (data / "sets.jsonl").read_text().splitlines()]
    qs = defaultdict(list)
    for l in (data / "queries.jsonl").read_text().splitlines():
        q = json.loads(l); q.pop("messages", None); qs[q["set_id"]].append(q)
    recs = {}
    for l in (data / "log.jsonl").read_text().splitlines():
        r = json.loads(l); recs[r["query_id"]] = r
    out = []
    for s in sets:
        if all(q["query_id"] in recs and recs[q["query_id"]]["p"] is not None or q["kind"] == "screen"
               for q in qs[s["set_id"]]) and all(q["query_id"] in recs for q in qs[s["set_id"]]):
            out.append(SetData(s, qs[s["set_id"]], recs))
    return out


def split(sds):
    ids = [f"s{k:02d}" for k in range(24)]  # every built set, complete or not
    cal = set(random.Random(SEED).sample(ids, N_CAL))
    return [sd for sd in sds if sd.s["set_id"] in cal], [sd for sd in sds if sd.s["set_id"] not in cal]


DEFAULT_CAL = {"sigma": {"single": 0.8, "and2": 0.8, "cond": 0.8, "and3": 0.8}, "pi0": 0.2, "slab": 1.5,
               "spike": 0.2, "h_mu": -1.5, "h_sd": 2.5, "b": {"and2": 0.0, "cond": 0.0}, "b_sd": 1.0,
               "screen": {"pooled": {"r1": 0.5, "r0": 0.3}}}


def calibrate(data: Path, tau: float):
    sds = load(data)
    cal_sets, _ = split(sds)
    cal = json.loads(json.dumps(DEFAULT_CAL))
    model = Model(cal)
    fits = {}
    for it in range(3):
        res = defaultdict(list); bs = defaultdict(list)
        for sd in cal_sets:
            idx = sd.singles + sd.pool
            fit = model.fit(sd, idx, free_bias=True, theta0=fits.get(sd.s["set_id"], {}).get("theta"))
            fits[sd.s["set_id"]] = fit
            g, _, _ = implied(fit["theta"], sd.ME[idx], sd.MN[idx], sd.bias_col[idx])
            for k, r in zip(idx, g - sd.y[idx]):
                res[sd.kind[k]].append(r)
            bs["and2"].append(fit["theta"][B_AND]); bs["cond"].append(fit["theta"][B_COND])
        for kd, r in res.items():
            cal["sigma"][kd] = float(np.sqrt(np.mean(np.square(r))))
        cal["sigma"]["and3"] = cal["sigma"]["and2"]
        cal["b"] = {k: float(np.mean(v)) for k, v in bs.items()}
        model = Model(cal)
    Js = np.concatenate([f["theta"][N:N + NP] for f in fits.values()])
    hs = np.concatenate([f["theta"][:N] for f in fits.values()])
    strong = np.abs(Js) >= tau
    cal["pi0"] = float(np.clip(strong.mean(), 0.02, 0.5))
    cal["slab"] = float(np.sqrt(np.mean(Js[strong] ** 2))) if strong.any() else 1.5
    cal["spike"] = float(max(np.sqrt(np.mean(Js[~strong] ** 2)), 0.05))
    cal["h_mu"], cal["h_sd"] = float(hs.mean()), float(max(hs.std() * 1.5, 1.0))
    cal["b_sd"] = 0.3
    cal["tau"] = tau
    # screening likelihood vs reference strong edges
    cnt = defaultdict(lambda: np.zeros((2, 2)))
    sizes = []
    for sd in cal_sets:
        z = (np.abs(fits[sd.s["set_id"]]["theta"][N:N + NP]) >= tau).astype(int)
        for t, L in sd.screens.items():
            if L is None:
                continue
            sizes.append(len(L))
            for j in range(N):
                if j != t:
                    e = PIDX[tuple(sorted((t, j)))]
                    for b in (bucket(len(L)), "pooled"):
                        cnt[b][z[e], int(j in L)] += 1
    scr = {}
    for b, c in cnt.items():
        r1 = (c[1, 1] + 1) / (c[1].sum() + 2); r0 = (c[0, 1] + 1) / (c[0].sum() + 2)
        scr[b] = {"r1": float(r1), "r0": float(r0), "n_pos": int(c[1].sum()), "n_neg": int(c[0].sum())}
    for b in ("small", "mid", "large"):
        scr.setdefault(b, scr["pooled"])
    cal["screen"] = scr
    cal["screen_list_sizes"] = Counter(sizes)
    cal["cal_sets"] = sorted(sd.s["set_id"] for sd in cal_sets)
    cal["J_quantiles"] = {q: float(np.quantile(np.abs(Js), q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
    (data / "calibration.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))


def grid_for(pool_size):
    last = N + pool_size
    g = list(range(N, last, BATCH)) + [last]
    return sorted(set(g))


def run_set(args):
    data, sid, methods, n_random, cal_path = args
    cal = json.loads(Path(cal_path).read_text())
    model = Model(cal)
    sd = next(s for s in load(Path(data)) if s.s["set_id"] == sid)
    tau = cal["tau"]
    dense = model.fit(sd, sd.singles + sd.pool)
    zref = (np.abs(dense["theta"][N:N + NP]) >= tau).astype(int)
    total = len(sd.pool) + len(sd.screens)          # every askable query (pairs + screens)
    counts = list(range(N, N + total, BATCH)) + [N + total]
    res = {"set_id": sid, "n_strong": int(zref.sum()), "zref": zref.tolist(),
           "J_ref": dense["theta"][N:N + NP].tolist(), "dense": metrics(model, sd, dense, zref), "curves": {}}
    for meth in methods:
        reps = n_random if meth == "random" else 1
        for rep in range(reps):
            rng = random.Random(f"{SEED}-{sid}-{meth}-{rep}")
            nrng = np.random.default_rng(rng.randrange(2 ** 32))
            curve = []
            if meth in ("random", "similar", "screen"):
                if meth == "random":
                    order = [("q", k) for k in rng.sample(sd.pool, len(sd.pool))]
                elif meth == "similar":
                    sim = bge_sim(sd, data)
                    po = sorted(PAIRS, key=lambda p: (-sim[p], rng.random()))
                    order = [("q", k) for k in order_by_pairs(sd, po, rng)]
                else:
                    flags = Counter()
                    for t, L in sd.screens.items():
                        for j in (L or ()):
                            flags[tuple(sorted((t, j)))] += 1
                    po = sorted(PAIRS, key=lambda p: (-flags[p], rng.random()))
                    order = [("s", t) for t in range(N)] + [("q", k) for k in order_by_pairs(sd, po, rng)]
                theta = None
                for c in counts:
                    pref = order[:c - N]
                    fit = model.fit(sd, sd.singles + [k for t, k in pref if t == "q"],
                                    screened=[k for t, k in pref if t == "s"], theta0=theta)
                    theta = fit["theta"]
                    mt = metrics(model, sd, fit, zref)
                    curve.append({"n": c, **mt})
            else:
                asked, screened = [], []
                fit = model.fit(sd, sd.singles)
                c = N
                while True:
                    mt = metrics(model, sd, fit, zref)
                    curve.append({"n": c, "screens": len(screened),
                                  "kinds": dict(Counter(sd.kind[k] for k in asked)), **mt})
                    avail = [k for k in sd.pool if k not in set(asked)]
                    scr_avail = [t for t in range(N) if t not in screened] if meth == "eig" else []  # eig_noscr: no screens
                    if not avail and not scr_avail:
                        break
                    if meth in ("eig", "eig_noscr"):
                        qk, sk = select_eig(model, sd, fit, avail, scr_avail, BATCH, nrng)
                    else:
                        qk, sk = select_eig_pred(model, sd, fit, avail, BATCH, nrng), []
                        if not qk:
                            break
                    asked += qk; screened += sk
                    c += len(qk) + len(sk)
                    fit = model.fit(sd, sd.singles + asked, screened=screened, theta0=fit["theta"])
            res["curves"].setdefault(meth, []).append(curve)
    return res


def evaluate(data: Path, workers: int, methods, n_random, cal_path: Path, all_sets: bool, merge: bool = False):
    """merge: add these methods' curves to an existing eval.json (same sets) instead of replacing it."""
    sds = load(data)
    ev = sds if all_sets else split(sds)[1]
    args = [(str(data), sd.s["set_id"], methods, n_random, str(cal_path)) for sd in ev]
    old = {r["set_id"]: r for r in json.loads((data / "eval.json").read_text())} if merge else {}
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_set, args):
            print(f"{r['set_id']}: strong={r['n_strong']} dense_kl={r['dense']['heldout_kl']:.4f} "
                  f"auc@end=" + ", ".join(f"{m}:{c[0][-1]['auc']}" for m, c in r["curves"].items()), flush=True)
            if r["set_id"] in old:
                old[r["set_id"]]["curves"].update(r["curves"]); r = old[r["set_id"]]
            out.append(r)
    (data / "eval.json").write_text(json.dumps(out))


# ---------------------------------------------------------------- report ------

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def n_to_reach(ns, frac, target=0.9):
    for k in range(len(ns)):
        if frac[k] >= target:
            if k == 0:
                return float(ns[0])
            f0, f1 = frac[k - 1], frac[k]
            return float(ns[k - 1] + (target - f0) / (f1 - f0) * (ns[k] - ns[k - 1]))
    return None


def summarize(res, methods, taus=(0.5, 0.7)):
    """Mean curves over sets (random reps averaged within a set first) on the common grid."""
    grid = sorted({pt["n"] for r in res for cs in r["curves"].values() for c in cs for pt in c})
    out = {}
    dense_kl = float(np.mean([r["dense"]["heldout_kl"] for r in res]))
    dense_out = {k: float(np.mean([r["dense"]["out"][k] for r in res])) for k in ("single", "and2", "and3")}
    for m in methods:
        rows = defaultdict(lambda: defaultdict(list))
        for r in res:
            J_ref = np.array(r["J_ref"])
            for c in r["curves"][m]:
                last = None
                byn = {pt["n"]: pt for pt in c}
                for n in grid:
                    pt = byn.get(n) or last  # a method that ran out of queries keeps its last fit
                    if pt is None:
                        continue
                    last = pt
                    rows[n]["kl"].append((r["set_id"], pt["heldout_kl"]))
                    for k in ("single", "and2", "and3"):
                        rows[n][k].append((r["set_id"], pt["out"][k]))
                    rows[n]["edges"].append((r["set_id"], pt["P"], pt["J"], J_ref))
        curve = {"n": [], "heldout_kl": [], "out_single": [], "out_and2": [], "out_and3": [], "spearman_absJ": []}
        for t in taus:
            curve[f"auc_tau{t}"] = []; curve[f"f1_tau{t}"] = []
        for n in grid:
            if not rows[n]["kl"]:
                continue
            curve["n"].append(n)
            for key, nm in (("kl", "heldout_kl"), ("single", "out_single"), ("and2", "out_and2"), ("and3", "out_and3")):
                per = defaultdict(list)
                for sid, v in rows[n][key]:
                    per[sid].append(v)
                curve[nm].append(float(np.mean([np.mean(v) for v in per.values()])))
            P = np.concatenate([np.array(p) for _, p, _, _ in rows[n]["edges"]])
            Jm = np.concatenate([np.abs(np.array(j)) for _, _, j, _ in rows[n]["edges"]])
            Jr = np.concatenate([np.abs(jr) for _, _, _, jr in rows[n]["edges"]])
            curve["spearman_absJ"].append(spearman(Jm, Jr))
            for t in taus:
                z = (Jr >= t).astype(int)
                curve[f"auc_tau{t}"].append(auc(P, z))
                pred = P >= 0.5
                tp = (pred & (z == 1)).sum(); fp = (pred & (z == 0)).sum(); fn = (~pred & (z == 1)).sum()
                curve[f"f1_tau{t}"].append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        out[m] = curve
    return grid, out, dense_kl, dense_out


def report(data: Path, charts=None, taus=(0.5, 0.7)):
    res = json.loads((data / "eval.json").read_text())
    cal = json.loads((data / "calibration.json").read_text())
    methods = list(res[0]["curves"])
    grid, curves, dense_kl, dense_out = summarize(res, methods, taus)
    start_kl = curves[methods[0]]["heldout_kl"][0]
    n90 = {}
    for m, c in curves.items():
        frac = [(start_kl - v) / (start_kl - dense_kl) for v in c["heldout_kl"]]
        c["frac_of_dense_gain"] = frac
        n90[m] = {"kl_90": n_to_reach(c["n"], frac),
                  **{f"auc_tau{t}_0.9": n_to_reach(c["n"], [a if a is not None else 0 for a in c[f"auc_tau{t}"]]) for t in taus},
                  "spearman_0.8": n_to_reach(c["n"], c["spearman_absJ"], 0.8)}
    # bootstrap over sets for the query count to 90% of the dense held-out gain
    rng = np.random.default_rng(SEED)
    ids = [r["set_id"] for r in res]
    boot = defaultdict(list); boot_auc = defaultdict(list); boot_diff = defaultdict(list)
    at = (32, 52, 72, 92, 112)
    for _ in range(300):
        pick = rng.choice(len(res), len(res))
        sub = [res[k] for k in pick]
        _, cb, dk, _ = summarize(sub, methods, (0.7,))
        sk = cb[methods[0]]["heldout_kl"][0]
        for m, c in cb.items():
            fr = [(sk - v) / (sk - dk) for v in c["heldout_kl"]]
            boot[m].append(n_to_reach(c["n"], fr) or float("nan"))
            boot_auc[m].append(n_to_reach(c["n"], [a if a is not None else 0 for a in c["auc_tau0.7"]]) or float("nan"))
            for n in at:
                k, k0 = c["n"].index(n), cb["random"]["n"].index(n)
                boot_diff[(m, n, "auc0.7")].append(c["auc_tau0.7"][k] - cb["random"]["auc_tau0.7"][k0])
                boot_diff[(m, n, "kl")].append(c["heldout_kl"][k] - cb["random"]["heldout_kl"][k0])
    for m in methods:
        b = np.array(boot[m]); ba = np.array(boot_auc[m])
        n90[m]["kl_90_ci"] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
        n90[m]["kl_90_never_frac"] = float(np.isnan(b).mean())
        n90[m]["auc_tau0.7_0.9_ci"] = [float(np.nanpercentile(ba, 2.5)), float(np.nanpercentile(ba, 97.5))]
    vs_random = {f"{m} n={n} {k}": [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                 for (m, n, k), v in boot_diff.items() if m != "random"}
    raw = {"single_ll_raw": None}
    summary = {"sets": ids, "n_sets": len(ids), "grid": grid, "dense_heldout_kl": dense_kl, "start_heldout_kl": start_kl,
               "dense_outcome_ll": dense_out, "n_strong": {str(t): int(sum((np.abs(r["J_ref"]) >= t).sum() for r in res)) for t in taus},
               "n_edges": 66 * len(res), "n90": n90, "vs_random": vs_random, "curves": curves, "calibration": cal}
    (data / "summary.json").write_text(json.dumps(summary, indent=1))
    if charts:
        ch = {"title": f"Active structure elicitation, Qwen3-32B ({data.name}, {len(ids)} sets of 12 events)",
              "x": "queries asked (12 singles first, then 1 call per query)",
              "reference": "dense fit on every non-held-out query (the model's own beliefs, not outcomes)",
              "series": {m: {k: c[k] for k in ("n", "heldout_kl", "frac_of_dense_gain", "auc_tau0.5", "auc_tau0.7",
                                               "f1_tau0.7", "spearman_absJ", "out_single", "out_and2", "out_and3")}
                         for m, c in curves.items()},
              "dense_heldout_kl": dense_kl, "dense_outcome_ll": dense_out, "n90": n90, "vs_random": vs_random}
        Path(charts).write_text(json.dumps(ch, indent=1))
    print(json.dumps({"n90": n90, "dense_kl": dense_kl, "start_kl": start_kl, "n_strong": summary["n_strong"]}, indent=1))
    for m, c in curves.items():
        print(m)
        for k in range(len(c["n"])):
            print(f"  n={c['n'][k]:4d} kl={c['heldout_kl'][k]:.4f} frac={c['frac_of_dense_gain'][k]:+.2f} "
                  f"auc.5={c['auc_tau0.5'][k]:.3f} auc.7={c['auc_tau0.7'][k]:.3f} f1.7={c['f1_tau0.7'][k]:.2f} "
                  f"rho={c['spearman_absJ'][k]:.2f} out s/a2/a3={c['out_single'][k]:.3f}/{c['out_and2'][k]:.3f}/{c['out_and3'][k]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "evaluate", "report"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--methods", default="random,similar,screen,eig,eig_pred")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--charts", type=Path, help="report: write chart JSON here")
    ap.add_argument("--cal", type=Path, help="calibration.json (default: <data>/calibration.json)")
    ap.add_argument("--merge", action="store_true", help="evaluate: add methods to an existing eval.json")
    ap.add_argument("--all-sets", action="store_true", help="evaluate every set (aligned sets; calibration from elsewhere)")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.data, a.tau)
    elif a.cmd == "evaluate":
        evaluate(a.data, a.workers, a.methods.split(","), a.n_random, a.cal or a.data / "calibration.json", a.all_sets, a.merge)
    else:
        report(a.data, a.charts)


if __name__ == "__main__":
    main()
