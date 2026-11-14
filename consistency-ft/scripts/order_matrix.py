"""Order matrix and same-event vs cross-event tests (Kelvin, 28 Sep).  No API calls; reuses three earlier runs.

Question: when a unit's joint distribution (8 outcomes for a triangle, 4 for a pair) is fitted to a chosen subset of
the model's answers, which answer levels improve outcome accuracy on which predicted level?  And for single-event
predictions: how much of the gain is SAME-EVENT reconciliation (A agreeing with its own negation) and how much is
CROSS-EVENT (other events' answers reaching A through the joint answers)?

Data sets (Qwen3-32B, thinking off, one statement per call):
  balanced  839 sign-balanced triangles x 16 statements (/home/agent/data/balanced-triangles-20260927).  Levels:
            S = every single-event answer on the triangle's 3 events that week: the weekly run's YES single plus
                the balanced run's NO single when the event was negated in any triangle that week,
            P = 3 signed pairs, R = triple T + 3 siblings, C = 4 count cells, M = 2 De Morgan mirrors.
  weekly    every open triangle of the 35-week AND-only run (/home/agent/data/weekly-worldmodel-20260927), refitted
            PER TRIANGLE.  S = 3 YES singles, P = 3 positive "A and B", R = positive "A and B and C".  The run's own
            week-wide fits M1/M3/M3b always use every level; they are reference rows only.
  pairs12   the 1,066 question-only test pairs, 12 forms each (/home/agent/data/qwen32b-baseline-20260923,
            v6_direct).  S = P, NOT P, Q, NOT Q;  A = the 4 signed ANDs;  O = the 4 signed ORs.

Same-event baseline ("base"): per event, reconcile only its own single-event answers (KL projection of YES and NOT
answers onto one probability = mean of p_yes and 1 - p_not); the unit's baseline joint is the independence product
of those.  For the weekly data there is no NOT answer, so base = the stated YES single.
Every other fit is reported as delta vs base (and vs raw where the raw answer exists, and vs "indep" = the product
of the unit's stated singles as asked).

Fit: q = argmin sum_forms KL(stated || implied) + EPS_ENT * sum q log q, equal weights, stated clipped to
[0.01, 0.99].  The tiny entropy term breaks ties only in directions the chosen answers leave free (max-entropy
completion).  S-only is computed exactly (product of reconciled singles).  Scoring: q mixed with 1% uniform,
binary predictions clipped to [0.01, 0.99]; per unit mean within a target set, then mean over units.

Event-centric tests (per unit and focal event E; scored on E's YES single; others = the unit's other events):
  base           own answers only
  own+J          own answers + all joint answers (no other events' singles)
  own+oth+J      everything                                   ((a): does adding joints move E toward the truth?)
  oth+J          leave-E-out: drop every answer about E alone  ((b): read p(E) off the joints and others)
  J              joints only
  oth~+J         leave-E-out with the other events' singles replaced by those of random other events of the same
                 week (pairs12: of a random other test pair)       ((c) swap test: are the joints a "connection"?)
  own+oth~+J     the swap with E's own answers kept
  baserate       the YES rate of focal events in the other weeks (pairs12: other folds of a 5-fold split)
For balanced, a second joint set PR (pairs + triples only) is run too.

CIs: 2,000 bootstrap draws over events, as in the weekly run: each draw gives every event (global id, across weeks)
a multinomial count and a unit the mean count of its events, so an event's units move together.  pairs12 events
never repeat across pairs, so there the pairs are resampled.  Plain unit-iid CIs are stored as *_iid.
Outputs: /home/agent/data/order-matrix-20260928/{balanced,weekly,pairs12}.json and charts.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

BAL = Path("/home/agent/data/balanced-triangles-20260927")
WK = Path("/home/agent/data/weekly-worldmodel-20260927")
PAIRS12 = Path("/home/agent/data/qwen32b-baseline-20260923/remote-out/predictions-test-v6_direct.jsonl")
OUT = Path("/home/agent/data/order-matrix-20260928")
LO, HI = 0.01, 0.99
EPS_MIX = 0.01
EPS_ENT = 1e-3
B = 2000
SEED = 20260928


def atomx(n):
    return list(itertools.product((0, 1), repeat=n))  # atom a -> bits; event k YES = bit k


def lit_atoms(n, lits):
    """lits: {event_index: 1/0} -> atoms where every literal holds"""
    return [a for a, x in enumerate(atomx(n)) if all(x[k] == s for k, s in lits.items())]


def clipp(v):
    return min(max(float(v), LO), HI)


# ---- fitting -------------------------------------------------------------------------------------------------


def project(p, M, eps=EPS_ENT):
    p = np.clip(np.asarray(p, float), LO, HI)
    K = M.shape[1]

    def f(th):
        q = np.exp(th - th.max()); q /= q.sum()
        m = np.clip(M @ q, 1e-12, 1 - 1e-12)
        lq = np.log(np.maximum(q, 1e-300))
        loss = np.sum(p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))) + eps * np.dot(q, lq)
        dm = -p / m + (1 - p) / (1 - m)
        gm = M.T @ dm + eps * lq
        return loss, q * (gm - np.dot(q, gm))
    r = minimize(f, np.zeros(K), jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-10})
    q = np.exp(r.x - r.x.max())
    return q / q.sum()


def reconcile(own):
    """own: list of (sign, p) single-event answers -> YES probability (KL projection = mean of implied YES)"""
    return float(np.mean([clipp(p) if s else 1 - clipp(p) for s, p in own]))


def indep(n, s_yes):
    s = [clipp(v) for v in s_yes]
    return np.array([np.prod([s[k] if x[k] else 1 - s[k] for k in range(n)]) for x in atomx(n)])


def fit_stmts(n, stmts):
    """stmts: list of (atoms, p).  Single-event-only sets are solved exactly (independence of reconciled)."""
    return project([p for _, p in stmts], masks(n, [at for at, _ in stmts]))


def masks(n, atom_lists):
    M = np.zeros((len(atom_lists), 2 ** n))
    for r, at in enumerate(atom_lists):
        M[r, list(at)] = 1
    return M


def mix(q):
    return (1 - EPS_MIX) * q + EPS_MIX / len(q)


def unit_stmts(u, levels, own_override=None):
    """answers of unit u restricted to levels; own_override: {event_index: list of (sign, p)} replaces singles"""
    out = []
    for st in u["stmts"]:
        if st["lev"] in levels and st["lev"] != "S":
            out.append((st["atoms"], st["p"]))
    if "S" in levels:
        own = dict(u["own"])
        if own_override:
            own.update(own_override)
        for k, lst in own.items():
            for s, p in lst:
                out.append((lit_atoms(u["n"], {k: s}), p))
    return out


def fit_unit_job(args):
    u, subsets, tests = args
    n = u["n"]
    res = {}
    for sub in subsets:
        if set(sub) == {"S"}:
            q = indep(n, [reconcile(u["own"][k]) for k in range(n)])
        else:
            q = fit_stmts(n, unit_stmts(u, set(sub)))
        res[sub] = mix(q)
    res["indep"] = mix(indep(n, u["s_raw"]))
    if "orig" in u:
        res["orig"] = mix(fit_stmts(n, u["orig"]))
    ev = {}
    for jname, J in tests:  # J: set of joint levels
        Jn = set(J)
        for k in range(n):
            oth = [j for j in range(n) if j != k]
            y_single = {}
            # own + J (other events' singles dropped)
            st_ownJ = [s for s in unit_stmts(u, Jn)] + [(lit_atoms(n, {k: s}), p) for s, p in u["own"][k]]
            st_full = st_ownJ + [(lit_atoms(n, {j: s}), p) for j in oth for s, p in u["own"][j]]
            st_othJ = unit_stmts(u, Jn) + [(lit_atoms(n, {j: s}), p) for j in oth for s, p in u["own"][j]]
            st_J = unit_stmts(u, Jn)
            sw = [(lit_atoms(n, {j: s}), p) for j in oth for s, p in u["swap"][j]]
            st_swLO = unit_stmts(u, Jn) + sw
            st_swFull = st_ownJ + sw
            for name, st in (("own+J", st_ownJ), ("own+oth+J", st_full), ("oth+J", st_othJ), ("J", st_J),
                             ("oth~+J", st_swLO), ("own+oth~+J", st_swFull)):
                q = mix(fit_stmts(n, st))
                y_single[name] = float(q[lit_atoms(n, {k: 1})].sum())
            y_single["base"] = reconcile(u["own"][k])
            ev[(jname, k)] = y_single
    res["_events"] = ev
    return res


# ---- scoring -----------------------------------------------------------------------------------------------------


def ll(p, y):
    p = clipp(p)
    return -np.log(p if y else 1 - p)


def br(p, y):
    return (clipp(p) - y) ** 2


def target_sets(u):
    n, y = u["n"], u["y"]
    st = {"singles": [lit_atoms(n, {k: 1}) for k in range(n)],
          "pairs_all": [lit_atoms(n, {i: si, j: sj}) for i, j in itertools.combinations(range(n), 2)
                        for si in (1, 0) for sj in (1, 0)]}
    if n == 3:
        st["triples_all"] = [[a] for a in range(8)]
    return {s: [(at, int(y in at)) for at in lst] for s, lst in st.items()}


def score_q(q, u):
    r = {}
    sets = target_sets(u)
    sets.update({s: [(at, l) for at, l, _ in lst] for s, lst in u["asked"].items()})
    for s, lst in sets.items():
        ps = [q[at].sum() for at, _ in lst]
        r[(s, "ll")] = np.mean([ll(p, l) for p, (_, l) in zip(ps, lst)])
        r[(s, "brier")] = np.mean([br(p, l) for p, (_, l) in zip(ps, lst)])
    r[("outcome", "ll")] = -np.log(max(q[u["y"]], 1e-12))
    r[("outcome", "brier")] = float(np.sum((q - np.eye(len(q))[u["y"]]) ** 2))
    return r


def score_raw(u, key="raw"):
    r = {}
    for s, lst in u["asked"].items():
        if key == "raw":
            ps = [p for _, _, p in lst]
        else:
            ps = u["glob"][key][s]
        r[(s, "ll")] = np.mean([ll(p, l) for p, (_, l, _) in zip(ps, lst)])
        r[(s, "brier")] = np.mean([br(p, l) for p, (_, l, _) in zip(ps, lst)])
    return r


# ---- bootstrap ---------------------------------------------------------------------------------------------------


class Boot:
    """Event-cluster bootstrap as in the weekly run: each draw gives every event (global id) a multinomial count and
    a unit the MEAN count of its events.  iid=True resamples units (used when no event repeats across units)."""

    def __init__(self, unit_events, rng, iid=False):
        ev = sorted({e for es in unit_events for e in es})
        ix = {e: i for i, e in enumerate(ev)}
        E, n = len(ev), len(unit_events)
        self.I = rng.integers(0, n, (B, n))
        W = np.empty((B, n), np.float32)
        if iid:
            for b in range(B):
                W[b] = np.bincount(self.I[b], minlength=n)
        else:
            T = [np.array([ix[es[pos]] for es in unit_events]) for pos in range(len(unit_events[0]))]
            for b in range(B):
                c = np.bincount(rng.integers(0, E, E), minlength=E).astype(np.float32)
                W[b] = sum(c[t] for t in T) / len(T)
        self.W, self.Ws = W, W.sum(1)

    def ci(self, x, mask=None):
        x = np.asarray(x, float)
        if mask is not None:
            W = self.W[:, mask]; x = x[mask]; Ws = W.sum(1)
        else:
            W, Ws = self.W, self.Ws
        ok = Ws > 0
        m = (W[ok] @ x.astype(np.float32)) / Ws[ok]
        lo, hi = np.percentile(m, [2.5, 97.5])
        return [float(x.mean()), float(lo), float(hi)]

    def ci_iid(self, x):
        x = np.asarray(x, float)
        m = x[self.I].mean(1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        return [float(x.mean()), float(lo), float(hi)]


def rel_group(types):
    c = Counter(types).most_common()
    return c[0][0] if c[0][1] >= 2 else "mixed"


# ---- loaders -----------------------------------------------------------------------------------------------------


def week_ids(week):
    return [json.loads(l)["id"] for l in (WK / "weeks" / week / "events.jsonl").read_text().splitlines()]


def load_weekly_raw():
    qs = {}
    with (WK / "queries.jsonl").open() as f:
        for line in f:
            i = line.find('"messages"')
            try:
                q = json.loads(line[:i].rstrip(", ") + "}")
            except json.JSONDecodeError:
                q = json.loads(line)
            qs[q["query_id"]] = (q["week"], q["kind"], tuple(q["events"]), q["label"])
    ans = {}
    with (WK / "log.jsonl").open() as f:
        for line in f:
            r = json.loads(line); ans[r["query_id"]] = r["p"]
    by = defaultdict(dict)
    for qid, (w, k, e, lab) in qs.items():
        by[w][(k, tuple(sorted(e)))] = (qid, ans[qid], lab)
    return by


def attach_swaps(units, pool_by_week, rng):
    """pool_by_week: week -> {event_global_id: list of (sign, p)}"""
    for u in units:
        pool = pool_by_week[u["week"]]
        cand = [e for e in pool if e not in u["events"]]
        u["swap"] = {k: pool[cand[rng.integers(len(cand))]] for k in range(u["n"])}


def load_balanced(wk_by):
    tris = [json.loads(l) for l in (BAL / "triangles.jsonl").read_text().splitlines()]
    ans = {}
    for l in (BAL / "log.jsonl").read_text().splitlines():
        r = json.loads(l); ans[r["prompt_sha256"]] = r["p"]
    weeks = sorted({t["week"] for t in tris})
    ids = {w: week_ids(w) for w in weeks}
    # own single-event answers per (week, local event): weekly YES + balanced NO answers (dedupe by prompt)
    own = defaultdict(dict)
    for w in weeks:
        for (kind, e), (_, p, lab) in wk_by[w].items():
            if kind == "single":
                own[w][e[0]] = {"yes": p, "nos": {}, "label": lab}
    for t in tris:
        for k in range(3):
            f = t["forms"][f"single{k}"]
            if not t["signs"][k] and f["sha"] in ans:
                own[t["week"]][t["events"][k]]["nos"][f["sha"]] = ans[f["sha"]]

    def own_list(w, e):
        o = own[w][e]
        return [(1, o["yes"])] + [(0, p) for p in o["nos"].values()]
    LEV = {"single": "S", "pair": "P", "T": "R", "sib": "R", "count": "C", "mir": "M"}
    units = []
    for t in tris:
        forms = list(t["forms"])
        p = {f: ans.get(t["forms"][f]["sha"]) for f in forms}
        if any(v is None for v in p.values()):
            continue
        y = t["outcome"]
        # triangle atoms use the same convention: atom = 4*x0 + 2*x1 + x2, x_k = event k YES
        stmts, asked = [], defaultdict(list)
        for f in forms:
            lev = LEV[next(k for k in LEV if f.startswith(k))]
            at = t["forms"][f]["atoms"]
            assert int(y in at) == t["forms"][f]["label"]
            stmts.append({"lev": lev, "atoms": at, "p": p[f]})
            s = {"S": "singles_asked", "P": "pairs_asked", "R": "triples_asked"}.get(lev)
            if s:
                asked[s].append((at, int(y in at), p[f]))
            asked["all16"].append((at, int(y in at), p[f]))
        s_raw = [p[f"single{k}"] if t["signs"][k] else 1 - p[f"single{k}"] for k in range(3)]
        units.append(dict(n=3, week=t["week"], events=[ids[t["week"]][e] for e in t["events"]], y=y, stmts=stmts,
                          own={k: own_list(t["week"], t["events"][k]) for k in range(3)}, s_raw=s_raw,
                          asked=dict(asked), rel=rel_group(t["stratum"].split("|")),
                          orig=[(t["forms"][f]["atoms"], p[f]) for f in forms],
                          n_own_no=sum(len(own[t["week"]][e]["nos"]) > 0 for e in t["events"])))
    pool = {w: {ids[w][e]: own_list(w, e) for e in own[w]} for w in weeks}
    return units, pool


def load_weekly(wk_by):
    GL = {"M1": "M1@0.0", "M3": "M3@1.0", "M3b": "M3b@0.3"}
    units, pool = [], {}
    for w in sorted(wk_by):
        by = wk_by[w]
        ids = week_ids(w)
        g = json.loads((WK / "weeks" / w / "graph.json").read_text())
        pr = json.loads((WK / "fits" / f"{w}.json").read_text())["preds"]
        rel = {tuple(sorted(map(int, k.split("-")))): v["type"] for k, v in g["rel"].items()}
        pool[w] = {ids[e[0]]: [(1, p)] for (kind, e), (_, p, _) in by.items() if kind == "single"}
        for tri in g["triangles"]:
            e = list(tri)
            prs = list(itertools.combinations(range(3), 2))
            keys = [("single", (e[k],)) for k in range(3)] + [("and2", tuple(sorted((e[i], e[j])))) for i, j in prs] + \
                   [("and3", tuple(sorted(e)))]
            rows = [by.get(k) for k in keys]
            if any(r is None for r in rows):
                continue
            labs = [rows[k][2] for k in range(3)]
            y = 4 * labs[0] + 2 * labs[1] + labs[2]
            atoms = [lit_atoms(3, {k: 1}) for k in range(3)] + [lit_atoms(3, {i: 1, j: 1}) for i, j in prs] + [[7]]
            assert all(int(y in at) == r[2] for at, r in zip(atoms, rows))
            p = [r[1] for r in rows]
            lev = ["S"] * 3 + ["P"] * 3 + ["R"]
            stmts = [{"lev": l, "atoms": at, "p": v} for l, at, v in zip(lev, atoms, p)]
            names = ["singles_asked"] * 3 + ["pairs_asked"] * 3 + ["triples_asked"]
            asked = defaultdict(list)
            for nm, at, v in zip(names, atoms, p):
                asked[nm].append((at, int(y in at), v))
            glob = {m: {"singles_asked": [pr[GL[m]][r[0]] for r in rows[:3]],
                        "pairs_asked": [pr[GL[m]][r[0]] for r in rows[3:6]],
                        "triples_asked": [pr[GL[m]][rows[6][0]]]} for m in GL}
            units.append(dict(n=3, week=w, events=[ids[x] for x in e], y=y, stmts=stmts,
                              own={k: [(1, p[k])] for k in range(3)}, s_raw=p[:3], asked=dict(asked), glob=glob,
                              rel=rel_group([rel[tuple(sorted((e[i], e[j])))] for i, j in prs])))
    return units, pool


FORM_LITS = {"P": {0: 1}, "NOT P": {0: 0}, "Q": {1: 1}, "NOT Q": {1: 0},
             "(P) AND (Q)": ("A", {0: 1, 1: 1}), "(P) AND (NOT Q)": ("A", {0: 1, 1: 0}),
             "(NOT P) AND (Q)": ("A", {0: 0, 1: 1}), "(NOT P) AND (NOT Q)": ("A", {0: 0, 1: 0}),
             "(P) OR (Q)": ("O", {0: 0, 1: 0}), "(P) OR (NOT Q)": ("O", {0: 0, 1: 1}),
             "(NOT P) OR (Q)": ("O", {0: 1, 1: 0}), "(NOT P) OR (NOT Q)": ("O", {0: 1, 1: 1})}


def load_pairs12():
    by = defaultdict(dict)
    for l in PAIRS12.read_text().splitlines():
        r = json.loads(l); by[r["pair_id"]][r["form"]] = r
    units = []
    for i, (pid, d) in enumerate(sorted(by.items())):
        lp, lq = d["P"]["label"], d["Q"]["label"]
        y = 2 * lp + lq
        stmts, asked, own = [], defaultdict(list), {0: [], 1: []}
        for f, r in d.items():
            spec = FORM_LITS[f]
            if isinstance(spec, dict):
                (k, s), = spec.items()
                own[k].append((s, r["p"]))
                at = lit_atoms(2, spec); lev = "S"; nm = "singles_asked"
            else:
                lev, lits = spec
                if lev == "A":
                    at = lit_atoms(2, lits); nm = "ands_asked"
                else:  # OR = not (the complementary AND)
                    at = [a for a in range(4) if a not in lit_atoms(2, lits)]; nm = "ors_asked"
                stmts.append({"lev": lev, "atoms": at, "p": r["p"]})
            assert int(y in at) == r["label"], (pid, f)
            asked[nm].append((at, int(y in at), r["p"]))
            asked["all12"].append((at, int(y in at), r["p"]))
        units.append(dict(n=2, week=f"fold{i % 5}", events=[d["P"]["p_id"], d["P"]["q_id"]], y=y, stmts=stmts,
                          own=own, s_raw=[d["P"]["p"], d["Q"]["p"]], asked=dict(asked), rel="all"))
    pool = {}
    allev = {}
    for u in units:
        for k in range(2):
            allev[u["events"][k]] = u["own"][k]
    for f in {u["week"] for u in units}:
        pool[f] = allev
    return units, pool


# ---- analysis ----------------------------------------------------------------------------------------------------


def analyse(name, units, levels, tests, extra_rows=(), iid=False):
    subsets = ["".join(c) for r in range(1, len(levels) + 1) for c in itertools.combinations(levels, r)]
    n = len(units)
    print(f"[{name}] {n} units, {len(subsets)} subsets, tests {[t for t, _ in tests]}", flush=True)
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        fits = pool.map(fit_unit_job, [(u, subsets, tests) for u in units], chunksize=16)
    print(f"[{name}] fitted", flush=True)
    rowsd = {}
    rownames = ["base" if s == "S" else s for s in subsets] + ["indep"] + (["orig16"] if "orig" in units[0] else [])
    for row, key in zip(rownames, subsets + ["indep", "orig"]):
        acc = defaultdict(lambda: np.full(n, np.nan))
        for r, (u, fq) in enumerate(zip(units, fits)):
            for k, v in score_q(fq[key], u).items():
                acc[k][r] = v
        rowsd[row] = dict(acc)
    for key in ["raw"] + [f"global_{m}" for m in extra_rows]:
        acc = defaultdict(lambda: np.full(n, np.nan))
        for r, u in enumerate(units):
            for k, v in score_raw(u, "raw" if key == "raw" else key.split("_", 1)[1]).items():
                acc[k][r] = v
        rowsd[key] = dict(acc)
    rng = np.random.default_rng(SEED)
    boot = Boot([u["events"] for u in units], rng, iid=iid)
    raw_of = {"singles": "singles_asked", "pairs_all": None, "triples_all": None, "outcome": None}
    targets = sorted({k[0] for d in rowsd.values() for k in d})
    table = {}
    for row, d in rowsd.items():
        table[row] = {}
        for tg in targets:
            for met in ("ll", "brier"):
                x = d.get((tg, met))
                if x is None or np.isnan(x).all():
                    continue
                c = {"value": boot.ci(x)}
                for ref, refkey in (("base", tg), ("indep", tg), ("raw", raw_of.get(tg, tg))):
                    if ref == row or refkey is None:
                        continue
                    rx = rowsd[ref].get((refkey, met))
                    if rx is not None and not np.isnan(rx).all():
                        c[f"minus_{ref}"] = boot.ci(x - rx); c[f"minus_{ref}_iid"] = boot.ci_iid(x - rx)
                table[row][f"{tg}|{met}"] = c
    # event-centric tests: per unit mean over focal events
    labs = [[int(u["y"] in lit_atoms(u["n"], {k: 1})) for k in range(u["n"])] for u in units]
    groups = sorted({u["week"] for u in units})
    base_rate = {g: float(np.mean([l for u, ls in zip(units, labs) if u["week"] != g for l in ls])) for g in groups}
    evt = {}
    for jname, _ in tests:
        names = ["base", "own+J", "own+oth+J", "oth+J", "J", "oth~+J", "own+oth~+J"]
        X = {m: {nm: np.zeros(n) for nm in names + ["raw", "raw_yes", "baserate"]} for m in ("ll", "brier")}
        for r, (u, fq, ls) in enumerate(zip(units, fits, labs)):
            for m, fn in (("ll", ll), ("brier", br)):
                for nm in names:
                    X[m][nm][r] = np.mean([fn(fq["_events"][(jname, k)][nm], ls[k]) for k in range(u["n"])])
                X[m]["raw"][r] = np.mean([fn(u["s_raw"][k], ls[k]) for k in range(u["n"])])
                X[m]["raw_yes"][r] = np.mean([fn(next(p for sg, p in u["own"][k] if sg == 1), ls[k])
                                              for k in range(u["n"])])
                X[m]["baserate"][r] = np.mean([fn(base_rate[u["week"]], ls[k]) for k in range(u["n"])])
        res = {}
        for m in ("ll", "brier"):
            res[m] = {nm: {"value": boot.ci(x), "minus_base": boot.ci(x - X[m]["base"]),
                           "minus_base_iid": boot.ci_iid(x - X[m]["base"])} for nm, x in X[m].items()}
            d = {"same_event_raw_to_base": boot.ci(X[m]["base"] - X[m]["raw"]),
                 "joints_own_info (own+J - base)": boot.ci(X[m]["own+J"] - X[m]["base"]),
                 "cross_event (own+oth+J - own+J)": boot.ci(X[m]["own+oth+J"] - X[m]["own+J"]),
                 "total_joint (own+oth+J - base)": boot.ci(X[m]["own+oth+J"] - X[m]["base"]),
                 "total_vs_raw (own+oth+J - raw)": boot.ci(X[m]["own+oth+J"] - X[m]["raw"]),
                 "total_vs_raw_yes (own+oth+J - stated YES single)": boot.ci(X[m]["own+oth+J"] - X[m]["raw_yes"]),
                 "base_vs_raw_yes": boot.ci(X[m]["base"] - X[m]["raw_yes"]),
                 "leaveout_J_only_vs_raw_yes (J - stated YES single)": boot.ci(X[m]["J"] - X[m]["raw_yes"]),
                 "leaveout_vs_raw (oth+J - raw)": boot.ci(X[m]["oth+J"] - X[m]["raw"]),
                 "leaveout_J_only_vs_raw (J - raw)": boot.ci(X[m]["J"] - X[m]["raw"]),
                 "swap_full (own+oth~+J - own+oth+J)": boot.ci(X[m]["own+oth~+J"] - X[m]["own+oth+J"]),
                 "leaveout_vs_baserate (oth+J - baserate)": boot.ci(X[m]["oth+J"] - X[m]["baserate"]),
                 "leaveout_J_only_vs_baserate (J - baserate)": boot.ci(X[m]["J"] - X[m]["baserate"]),
                 "leaveout_others_value (oth+J - J)": boot.ci(X[m]["oth+J"] - X[m]["J"]),
                 "swap_leaveout (oth~+J - oth+J)": boot.ci(X[m]["oth~+J"] - X[m]["oth+J"]),
                 "base_vs_baserate": boot.ci(X[m]["base"] - X[m]["baserate"])}
            den = X[m]["baserate"].mean() - X[m]["base"].mean()
            d["leaveout_recovery_share (baserate-oth+J)/(baserate-base)"] = \
                float((X[m]["baserate"].mean() - X[m]["oth+J"].mean()) / den) if den > 0 else None  # undefined if base <= base rate
            res[m]["decomposition"] = d
        evt[jname] = res
        evt[jname]["_X"] = X  # removed before writing
    # splits (log loss): week and relation group
    splits = {}
    for key in ("week", "rel"):
        splits[key] = {}
        for g in sorted({u[key] for u in units}):
            mask = np.array([u[key] == g for u in units])
            if mask.sum() < 30 or mask.all():
                continue
            cell = {"n": int(mask.sum())}
            for row in rowsd:
                for tg in ("singles", "pairs_asked", "pairs_all", "triples_all", "outcome"):
                    x = rowsd[row].get((tg, "ll"))
                    if x is None or np.isnan(x).all() or row == "base":
                        continue
                    cell[f"{row}|{tg}"] = {"ll": float(x[mask].mean()),
                                           "minus_base": boot.ci(x - rowsd["base"][(tg, "ll")], mask)}
            for jname, _ in tests:
                X = evt[jname]["_X"]["ll"]
                for nm in ("own+oth+J", "oth+J", "oth~+J", "own+J"):
                    cell[f"evt[{jname}]|{nm}"] = {"minus_base": boot.ci(X[nm] - X["base"], mask)}
                cell[f"evt[{jname}]|swap_leaveout"] = {"delta": boot.ci(X["oth~+J"] - X["oth+J"], mask)}
                cell[f"evt[{jname}]|swap_full"] = {"delta": boot.ci(X["own+oth~+J"] - X["own+oth+J"], mask)}
                cell[f"evt[{jname}]|cross_event"] = {"delta": boot.ci(X["own+oth+J"] - X["own+J"], mask)}
            splits[key][g] = cell
    for jname in evt:
        evt[jname].pop("_X")
    extra = {}
    if "n_own_no" in units[0]:
        extra["events_with_a_NOT_answer_share"] = float(np.mean([u["n_own_no"] / 3 for u in units]))
    return {"name": name, "units": n, "subsets": subsets, "rows": list(rowsd), "table": table,
            "event_tests": evt, "splits": splits, "base_rate_by_group": base_rate, **extra}


def make_charts():
    """heatmap-ready cells: rows = answers used, cols = target level, value = delta log loss vs the same-event base
    (and vs raw where the raw answer exists), with event-cluster 95% CIs; plus the event-centric tests."""
    TARGETS = {"singles": "singles", "pairs_asked": "pairs (asked)", "ands_asked": "pairs (asked ANDs)",
               "ors_asked": "ORs (asked)", "pairs_all": "pairs (all 4 signs)", "triples_asked": "triples (asked)",
               "triples_all": "triples (all 8)", "outcome": "full outcome"}
    out = {"notes": "delta = row minus the same-event baseline (base: each event's own answers reconciled, "
                    "independence across events); negative = better.  delta_vs_raw: row minus the raw stated answers "
                    "on the statements actually asked at that level (singles: the singles as asked).",
           "heatmaps": {}, "event_tests": {}}
    for name in ("balanced", "weekly", "pairs12"):
        fp = OUT / f"{name}.json"
        if not fp.exists():
            continue
        r = json.loads(fp.read_text())
        cells = []
        for row in r["rows"]:
            for tg, label in TARGETS.items():
                c = r["table"][row].get(f"{tg}|ll")
                if not c:
                    continue
                cell = {"row": row, "col": label, "target": tg, "ll": c["value"][0]}
                if "minus_base" in c:
                    cell.update(delta=c["minus_base"][0], lo=c["minus_base"][1], hi=c["minus_base"][2])
                if "minus_raw" in c:
                    cell.update(delta_vs_raw=c["minus_raw"][0], lo_vs_raw=c["minus_raw"][1], hi_vs_raw=c["minus_raw"][2])
                cells.append(cell)
        out["heatmaps"][name] = {"units": r["units"], "rows": r["rows"],
                                 "cols": [v for k, v in TARGETS.items() if any(c["target"] == k for c in cells)],
                                 "cells": cells}
        out["event_tests"][name] = {j: [{"test": k, "delta": v[0], "lo": v[1], "hi": v[2]}
                                        for k, v in t["ll"]["decomposition"].items() if isinstance(v, list)]
                                    for j, t in r["event_tests"].items()}
    (OUT / "charts.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["balanced", "weekly", "pairs12"], action="append")
    ap.add_argument("--charts", action="store_true", help="only rebuild charts.json from the outputs")
    ap.add_argument("--weekly-sample", type=int, default=0, help="subsample weekly triangles (0 = all)")
    a = ap.parse_args()
    if a.charts:
        make_charts(); return
    todo = a.only or ["pairs12", "balanced", "weekly"]
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if "pairs12" in todo:
        units, pool = load_pairs12()
        attach_swaps(units, pool, rng)
        r = analyse("pairs12", units, ["S", "A", "O"], [("AO", "AO"), ("A", "A"), ("O", "O")], iid=True)
        (OUT / "pairs12.json").write_text(json.dumps(r, indent=1))
    wk_by = None
    if "balanced" in todo or "weekly" in todo:
        wk_by = load_weekly_raw()
    if "balanced" in todo:
        units, pool = load_balanced(wk_by)
        attach_swaps(units, pool, rng)
        r = analyse("balanced", units, ["S", "P", "R", "C", "M"], [("PRCM", "PRCM"), ("PR", "PR")])
        (OUT / "balanced.json").write_text(json.dumps(r, indent=1))
    if "weekly" in todo:
        units, pool = load_weekly(wk_by)
        if a.weekly_sample and a.weekly_sample < len(units):
            idx = np.sort(np.random.default_rng(SEED).choice(len(units), a.weekly_sample, replace=False))
            units = [units[i] for i in idx]
        attach_swaps(units, pool, rng)
        r = analyse("weekly", units, ["S", "P", "R"], [("PR", "PR")], extra_rows=["M1", "M3", "M3b"])
        r["note"] = "global_M* rows are the weekly run's week-wide fits (20% of joints held out of those fits)"
        (OUT / "weekly.json").write_text(json.dumps(r, indent=1))
    make_charts()


if __name__ == "__main__":
    main()
