"""T1 lemma 5 checks: finite-population LI, E_C(t) and invasion fitness (PREREG.md, check 5).

One Kelly (log-utility) market per question. Price = wealth-weighted mean of the beliefs that
trade (Beygelzimer-Langford-Pennock); wealth_i <- wealth_i * q_i(y) / p(y), which conserves total
wealth exactly. Traders (ids):
  0-2  residents pattern-0/1/2 (KT counts on the last k revealed bits)
  3-4  class C: pattern-4, pattern-8
  5-24 class C: smooth price-reading calibration traders q = sigmoid(logit p +- bump_b(p)), 10 bins
  25-33 junk constant beliefs 0.1..0.9 (birth pool only)
E_T(t) = sum_s log q_T(y_s)/p_s(y_s) at the clearing price (exact-LMSR P&L of T at market prices).
Family 'diag' (8 extra questions): y_t = 1[p_t < 1/2] on the arm's own clearing price.
Usage: python3 check_li.py [--seeds 1 2 3] [--T 8000]
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.normpath(os.path.join(HERE, "../../S0/substrate"))
sys.path.insert(0, SUB)
import streams_ext  # noqa: E402

FAMS = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "diag"]
KS = [0, 1, 2, 4, 8]
NRES, NC_PAT, NCAL, NJUNK = 3, 2, 20, 9
C_IDS = list(range(3, 25))
N = NRES + NC_PAT + NCAL + NJUNK
READ = np.arange(5, 25)
NONREAD = np.array([i for i in range(N) if i not in set(READ.tolist())])
CENT = np.repeat(np.arange(10) * 0.1 + 0.05, 2)
SGN = np.tile([1.0, -1.0], 10)
JUNK = np.arange(1, 10) * 0.1
EPS = 1e-3
LNE = float(np.log(1 / EPS))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def q_read(p):
    """Price-reading calibration beliefs, p: [...] -> [..., 20]; continuous in p."""
    p = np.clip(p, 1e-12, 1 - 1e-12)[..., None]
    bump = np.maximum(0.0, 1.0 - np.abs(p - CENT) / 0.1)
    return sig(np.log(p / (1 - p)) + SGN * bump)


ARMS = {
    #  name: (initial members, price rule, births rho, births use fixed point, delay)
    "N":      (list(range(NRES)), "one", 0.0, 0),
    "a":      (list(range(NRES)) + C_IDS, "pre", 0.0, 0),
    "b":      (list(range(NRES)) + C_IDS, "fp", 0.0, 0),
    "c1e-3":  (list(range(NRES)), "pre", 1e-3, 0),
    "c1e-2":  (list(range(NRES)), "pre", 1e-2, 0),
    "cfp1e-3": (list(range(NRES)), "fp", 1e-3, 0),
    "cfp1e-2": (list(range(NRES)), "fp", 1e-2, 0),
    "d":      (list(range(NRES)) + [4], "one", 0.0, 0),
    "a-D64":  (list(range(NRES)) + C_IDS, "pre", 0.0, 64),
}


def price(w, qn, rule):
    """w [Q,N] (sums to 1), qn [Q,N] non-reader beliefs (reader columns ignored).
    Returns clearing p [Q], traded beliefs [Q,N]."""
    wn = w[:, NONREAD]
    sn = (wn * qn[:, NONREAD]).sum(1)
    mass_n = wn.sum(1)
    q = qn.copy()
    if rule == "one" or (w[:, READ].sum() == 0 and rule != "fp"):
        q[:, READ] = q_read(np.clip(sn / np.maximum(mass_n, 1e-300), 1e-9, 1 - 1e-9))
        return sn + (w[:, READ] * q[:, READ]).sum(1), q
    if rule == "pre":
        ppre = np.clip(sn / np.maximum(mass_n, 1e-300), 1e-9, 1 - 1e-9)
        q[:, READ] = q_read(ppre)
        return sn + (w[:, READ] * q[:, READ]).sum(1), q
    # fixed point: f(p) = sn + sum_r w_r q_r(p) - p, f(0+) >= 0, f(1-) <= 0; bisection
    lo = np.full(len(sn), 1e-12); hi = np.full(len(sn), 1 - 1e-12)
    wr = w[:, READ]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = sn + (wr * q_read(mid)).sum(1) - mid
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    p = 0.5 * (lo + hi)
    q[:, READ] = q_read(p)
    return sn + (wr * q[:, READ]).sum(1), q       # = p up to bisection tolerance, exact BLP price


def run(seed, arm, T):
    members, rule, rho, D = ARMS[arm]
    bits64, fam64, _ = streams_ext.make_questions(Q=64, T=T, seed=seed)
    Q = 72
    bits = np.zeros((Q, T), dtype=np.int64); bits[:64] = bits64
    fam = np.concatenate([fam64, np.full(8, 8)])
    rng = np.random.default_rng(5000 + seed)
    R = D + 1
    w = np.zeros((R, Q, N))
    for i in members:
        w[:, :, i] = EPS if i >= NRES else 0.0
    res_share = (1 - EPS * sum(1 for i in members if i >= NRES)) / NRES
    w[:, :, :NRES] = res_share
    counts = {k: np.zeros((Q, 2 ** k, 2)) for k in KS}
    pw = {k: (2 ** np.arange(k))[::-1] for k in KS}
    cache_q = np.zeros((R, Q, N)); cache_p = np.zeros((R, Q)); cache_ctx = {k: np.zeros((R, Q), dtype=np.int64) for k in KS}
    cache_t = np.full(R, -1)
    E_clear = np.zeros((len(C_IDS), Q)); E_traded = np.zeros((Q, N))
    member_mask = np.zeros((Q, N), dtype=bool); member_mask[:, members] = True
    ck_every = 50
    ck_t, ck_E, ck_Et = [], [], []
    logloss = np.zeros(Q); nset = 0
    max_cons = 0.0
    t0 = D + 8 + 1
    for t in range(t0, T):
        m = t % R
        # births (fixed-share style: newborn gets EPS, everyone else scaled by 1-EPS)
        if rho > 0:
            born = rng.random(Q) < rho
            if born.any():
                who = rng.integers(3, N, size=Q)
                for qq in np.where(born)[0]:
                    w[:, qq] *= (1 - EPS); w[:, qq, who[qq]] += EPS
                    member_mask[qq, who[qq]] = True
        # non-reader beliefs from revealed bits (lag D+1 .. D+k)
        qn = np.zeros((Q, N))
        for j, k in enumerate(KS):
            if k == 0:
                ctx = np.zeros(Q, dtype=np.int64)
            else:
                ctx = (bits[:, t - D - k:t - D] * pw[k]).sum(1)
            cache_ctx[k][m] = ctx
            c = counts[k][np.arange(Q), ctx]
            qn[:, j] = (c[:, 1] + 0.5) / (c.sum(1) + 1.0)
        qn[:, 25:] = JUNK
        p, q = price(w[m], qn, rule)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        bits[64:, t] = (p[64:] < 0.5).astype(np.int64)
        cache_q[m] = q; cache_p[m] = p; cache_t[m] = t
        # settle contract s = t - D
        s = t - D
        ms = s % R
        if cache_t[ms] == s:
            y = bits[:, s]
            qs, ps = cache_q[ms], cache_p[ms]
            py = np.where(y == 1, ps, 1 - ps)
            qy = np.where(y[:, None] == 1, qs, 1 - qs)
            ratio = qy / py[:, None]
            E_traded += np.log(np.maximum(qy, 1e-300)) - np.log(py)[:, None]
            qc = q_read(ps)
            qshadow = np.concatenate([qs[:, 3:5], qc], 1)
            qsy = np.where(y[:, None] == 1, qshadow, 1 - qshadow)
            E_clear += (np.log(qsy) - np.log(py)[:, None]).T
            neww = w[ms] * ratio
            tot = neww.sum(1)
            max_cons = max(max_cons, float(np.abs(tot - 1).max()))
            w[ms] = neww / tot[:, None]
            logloss += -np.log(py); nset += 1
            for k in KS:
                ctx = cache_ctx[k][ms]
                counts[k][np.arange(Q), ctx, y] += 1
            cache_t[ms] = -1
        if t % ck_every == 0:
            ck_t.append(t); ck_E.append(E_clear.copy()); ck_Et.append(E_traded[:, C_IDS].T.copy())
    ck_E = np.array(ck_E); ck_Et = np.array(ck_Et)     # [nck, 22, Q]
    tt = np.array(ck_t)
    out = {"seed": seed, "arm": arm, "T": T, "D": D, "eps": EPS, "ln1/eps": LNE,
           "max_conservation_err": max_cons}
    # theorem checks: members' traded P&L (per question, all t) vs ln(1/eps) or (D+1) ln(1/eps)
    init_members = [i for i in members if i >= NRES]
    if init_members and rho == 0:
        idx = [C_IDS.index(i) for i in init_members]
        et = ck_Et[:, idx, :]
        out["max_member_traded_E"] = float(et.max())
        out["bound"] = LNE * (D + 1)
        out["bound_holds"] = bool(et.max() <= LNE * (D + 1) + 1e-6)
        # price readers measured at the clearing price (LI-relevant) vs the bound
        ridx = [C_IDS.index(i) for i in init_members if i >= 5]
        if ridx:
            out["max_reader_clearing_E"] = float(ck_E[:, ridx, :].max())
    fams = {}
    for f, name in enumerate(FAMS):
        qs_ = np.where(fam == f)[0]
        Ef = ck_E[:, :, qs_].mean(2)                   # [nck, 22] per-question family mean
        EC = Ef.max(1)
        h = len(tt) // 2
        s1 = (EC[h] - EC[0]) / (tt[h] - tt[0]); s2 = (EC[-1] - EC[h]) / (tt[-1] - tt[h])
        best = int(Ef[-1].argmax())
        fams[name] = {"E_C_end": float(EC[-1]), "slope1": float(s1), "slope2": float(s2),
                      "best_T": C_IDS[best],
                      "E_pat4_end": float(Ef[-1, 0]), "E_pat8_end": float(Ef[-1, 1]),
                      "E_calib_max_end": float(Ef[-1, 2:].max()),
                      "pat4_slope1": float((Ef[h, 0] - Ef[0, 0]) / (tt[h] - tt[0])),
                      "pat4_slope2": float((Ef[-1, 0] - Ef[h, 0]) / (tt[-1] - tt[h])),
                      "calib_slope2": float((Ef[-1, 2:].max() - Ef[h, 2:].max()) / (tt[-1] - tt[h])),
                      "phi_best": float(Ef[-1].max() / (tt[-1] - tt[0])),
                      "market_logloss": float(logloss[qs_].mean() / max(nset, 1))}
        fams[name]["curve_t"] = tt[::10].tolist()
        fams[name]["curve_EC"] = EC[::10].round(4).tolist()
    out["fam"] = fams
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "li.json"))
    a = ap.parse_args()
    sha = hashlib.sha256(open(os.path.join(SUB, "streams_ext.py"), "rb").read()).hexdigest()[:16]
    res = {"streams_ext_sha256_16": sha, "runs": []}
    for sd in a.seeds:
        for arm in a.arms:
            t1 = time.time()
            o = run(sd, arm, a.T)
            o["wall_s"] = round(time.time() - t1, 1)
            res["runs"].append(o)
            F = o["fam"]
            print(f"seed={sd} arm={arm} bound={o.get('bound_holds')} maxE={o.get('max_member_traded_E', float('nan')):.2f} "
                  f"rdrClr={o.get('max_reader_clearing_E', float('nan')):.2f} cons={o['max_conservation_err']:.1e} "
                  f"per EC={F['periodic']['E_C_end']:.1f} s1/s2={F['periodic']['slope1']:.4f}/{F['periodic']['slope2']:.4f} "
                  f"diag EC={F['diag']['E_C_end']:.1f} s2={F['diag']['slope2']:.4f} {o['wall_s']}s", flush=True)
            with open(a.out, "w") as fh:
                json.dump(res, fh)
