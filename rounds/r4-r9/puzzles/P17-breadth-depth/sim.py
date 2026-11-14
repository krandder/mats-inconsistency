"""P17 simulator: replays every allocation policy from the trajectory bank with exact NFE accounting (PREREG.md).

  python sim.py summarize --bank results/vm_out/bank --out results/summ       # per-block compact summaries
  python sim.py run --summ results/summ --out results                           # all policies, val fitting, test

Outputs: results/per_puzzle.npz (per method and budget: correct, nfe for each of the 2,048 test puzzles, in block
order), results/choices.json (everything chosen on val), results/val_table.json.
"""
import argparse, glob, heapq, json, math, os, sys, time
from collections import Counter

import numpy as np
from scipy.optimize import minimize

GRID = [8, 16, 32, 64, 128, 256, 512, 1024]
GI = {d: i for i, d in enumerate(GRID)}
CAPS = np.array([1024] * 2 + [512] * 2 + [256] * 4 + [128] * 8 + [64] * 16 + [32] * 32 + [16] * 64)
R = len(CAPS)
NOFV = 10**6
VAL_BLOCKS = [0, 1, 2, 3]
TEST_BLOCKS = list(range(4, 12))
BUDGETS = [16, 32, 64, 128, 256, 512, 1024]
PRIMARY_B = 64
D0 = 16
EPS = 1e-4


# ----------------------------------------------------------------------------------------------------------- bank
def summarize(bank, out):
    os.makedirs(out, exist_ok=True)
    groups = json.load(open(os.path.join(bank, "bank_meta.json")))["groups"]
    stats = {}
    for idxf in sorted(glob.glob(os.path.join(bank, "b??_idx.npy"))):
        b = int(os.path.basename(idxf)[1:3])
        files = [os.path.join(bank, f"b{b:02d}_g{g[0]}.npz") for g in groups]
        if not all(os.path.exists(f) for f in files):
            continue
        fn = os.path.join(out, f"b{b:02d}.npz")
        if os.path.exists(fn):
            continue
        P = len(np.load(idxf))
        G = len(GRID)
        S = dict(s=np.full((P, R, G), np.nan, np.float32), v=np.full((P, R, G), -1, np.int16),
                 c=np.zeros((P, R, G), bool), h=np.full((P, R, G), -1, np.int64),
                 cm=np.full((P, R, G), np.nan, np.float32), cmin=np.full((P, R, G), np.nan, np.float32),
                 fv=np.full((P, R), NOFV, np.int32), fvc=np.zeros((P, R), bool))
        vnc = 0; nvalid = 0
        curve = None
        for (gid, k0, K, Dcap), f in zip(groups, files):
            z = np.load(f)
            D = int(z["D"])
            assert D == Dcap and z["corr"].shape == (P, K, D), (f, z["corr"].shape)
            delta = z["delta"].astype(np.float32)
            viol, corr = z["viol"], z["corr"]
            valid = viol == 0
            vnc += int((valid & ~corr).sum()); nvalid += int(valid.sum())
            anyv = valid.any(-1)
            first = valid.argmax(-1)
            S["fv"][:, k0:k0 + K] = np.where(anyv, first + 1, NOFV)
            S["fvc"][:, k0:k0 + K] = np.where(anyv, np.take_along_axis(corr, first[..., None], -1)[..., 0], False)
            for d in GRID:
                if d > D:
                    continue
                g = GI[d]
                S["s"][:, k0:k0 + K, g] = np.nanmean(delta[:, :, d - 3:d], -1)
                S["v"][:, k0:k0 + K, g] = viol[:, :, d - 1]
                S["c"][:, k0:k0 + K, g] = corr[:, :, d - 1]
                S["h"][:, k0:k0 + K, g] = z["h"][:, :, d - 1]
                S["cm"][:, k0:k0 + K, g] = z["cmean"][:, :, d - 1]
                S["cmin"][:, k0:k0 + K, g] = z["cmin"][:, :, d - 1]
            if gid == 0:
                curve = corr.mean((0, 1))
        np.savez_compressed(fn, **S, P=P, vnc=vnc, nvalid=nvalid, curve0=curve)
        stats[b] = dict(P=P, valid_not_correct=vnc, valid_row_steps=nvalid)
        print(json.dumps({"block": b, **stats[b]}), flush=True)
    return stats


class Block:
    def __init__(self, fn):
        z = np.load(fn)
        for k in ("s", "v", "c", "h", "cm", "cmin", "fv", "fvc"):
            setattr(self, k, z[k])
        self.P = int(z["P"]); self.vnc = int(z["vnc"])
        self.ls = np.log(self.s + EPS)
        self.lv = np.log1p(np.maximum(self.v, 0).astype(np.float32))


def load_blocks(summ, blocks):
    return [Block(os.path.join(summ, f"b{b:02d}.npz")) for b in blocks]


# -------------------------------------------------------------------------------------------------------- statics
def n_restarts(D):
    return int((CAPS >= D).sum())


def splits(b):
    return [(D, b // D) for D in GRID if b % D == 0 and 1 <= b // D <= n_restarts(D)]


def sr(S, D, B):
    g = GI[D]
    k = np.nanargmin(S.s[:, :B, g], axis=1)
    return S.c[np.arange(S.P), k, g], k


def sr_chk(S, D, B):
    corr, _ = sr(S, D, B)
    fvok = S.fv[:, :B] <= D
    anyv = fvok.any(1)
    first = fvok.argmax(1)
    return np.where(anyv, S.fvc[np.arange(S.P), first], corr)


def mv(S, D, B):
    g = GI[D]
    out = np.zeros(S.P, bool)
    for i in range(S.P):
        hs = S.h[i, :B, g].tolist()
        top = Counter(hs).most_common(1)[0][0]
        out[i] = S.c[i, hs.index(top), g]
    return out


def oracle(S, D, B):
    return S.c[:, :B, GI[D]].any(1)


def rs(S, b, seed):
    sp = splits(b)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sp), S.P)
    out = np.zeros(S.P, bool)
    for j, (D, B) in enumerate(sp):
        m = pick == j
        if m.any():
            out[m] = sr(S, D, B)[0][m]
    return out


# ------------------------------------------------------------------------------------------- pooled simple rules
def seq_rule(S, b, Dst, mode="chk", tau=None, local=False):
    """Round-robin restart-until-settled at depth Dst. mode chk: settle on a valid step (early stop);
    mode rt: settle when s < tau (no early stop). Answer: valid grid, else smallest residual."""
    P, g = S.P, GI[Dst]
    rmax = n_restarts(Dst)
    used = np.zeros(P, np.int64)
    settled = np.zeros(P, bool)
    ans = np.zeros(P, bool)
    ncand = np.zeros(P, np.int64)
    rem = P * b
    progress = True
    while progress:
        progress = False
        for i in range(P):
            if settled[i] or ncand[i] >= rmax:
                continue
            if local:
                if b - used[i] < Dst:
                    continue
            elif rem < Dst:
                progress = False
                break
            r = ncand[i]; ncand[i] += 1
            if mode == "chk" and S.fv[i, r] <= Dst:
                cost = S.fv[i, r]; settled[i] = True; ans[i] = S.fvc[i, r]
            else:
                cost = Dst
                if mode == "rt" and S.s[i, r, g] < tau:
                    settled[i] = True
            used[i] += cost; rem -= cost; progress = True
    for i in range(P):
        if mode == "chk" and settled[i]:
            continue
        n = ncand[i]
        if n == 0:
            ans[i] = False
        else:
            k = int(np.nanargmin(S.s[i, :n, g]))
            ans[i] = S.c[i, k, g]
    return ans, used


# ------------------------------------------------------------------------------------------------------- traders
EXT_FEATS = ["v", "s", "slope", "cm", "cmin", "ld", "agree", "n"]
SPN_FEATS = ["n", "minv", "mins", "maxc", "empty", "ldm"]
EXT_TRADERS = {"viol": ["v", "ld"], "resid": ["s", "slope", "ld"], "conf": ["cm", "cmin", "ld"],
               "agree": ["agree", "n", "ld"], "all": EXT_FEATS, "base": []}
SPN_TRADERS = {"viol": ["minv", "n", "empty"], "resid": ["mins", "n", "empty"], "conf": ["maxc", "n", "empty"],
               "all": SPN_FEATS, "base": []}


def fit_logistic(X, y, l2=1e-2):
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])

    def f(w):
        z = Xb @ w
        ll = np.logaddexp(0, z) - y * z
        grad = Xb.T @ (1 / (1 + np.exp(-z)) - y) / n
        grad[:k] += l2 * w[:k]
        return ll.mean() + 0.5 * l2 * (w[:k] ** 2).sum(), grad

    w0 = np.zeros(k + 1)
    p = min(max(y.mean(), 1e-3), 1 - 1e-3)
    w0[-1] = math.log(p / (1 - p))
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


class Pool:
    def __init__(self, traders, feats):
        self.names = list(traders)
        self.idx = [[feats.index(f) for f in traders[n]] for n in self.names]
        self.W = None
        self.logw = np.zeros(len(self.names))

    def fit(self, X, y):
        X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.W = [fit_logistic(Z[:, ix], y) for ix in self.idx]
        self.n_fit, self.base_rate = len(y), float(y.mean())

    def probs(self, X):
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        out = np.empty((len(self.names), Z.shape[0]))
        for j, (ix, w) in enumerate(zip(self.idx, self.W)):
            out[j] = 1 / (1 + np.exp(-(Z[:, ix] @ w[:-1] + w[-1])))
        return np.clip(out, 1e-3, 1 - 1e-3)

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def update(self, pi, y):
        pm = float(self.weights() @ pi)
        self.logw += np.log(pi / pm) if y else np.log((1 - pi) / (1 - pm))
        self.logw -= self.logw.max()

    def logloss(self, X, y):
        P = self.probs(X)
        y = np.asarray(y, bool)
        return {n: float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))) for n, p in zip(self.names, P)}


# -------------------------------------------------------------------------------------------------------- market
class Market:
    """Greedy value-density clearing over a pooled budget (or per-puzzle budget if local)."""

    def __init__(self, S, b, pools, mode="chk", tau=None, local=False, price="pool", rng=None, learn=True,
                 collect=None):
        self.S, self.b, self.pools, self.mode, self.tau = S, b, pools, mode, tau
        self.local, self.price, self.rng, self.learn, self.collect = local, price, rng, learn, collect
        self.trace = None

    def ext_feats(self, i, cands):
        S = self.S
        r = np.array([c[0] for c in cands]); d = np.array([c[1] for c in cands])
        g = np.log2(d).astype(int) - 3
        h = S.h[i, r, g]
        n = len(cands)
        agree = ((h[:, None] == h[None, :]).sum(1) - 1) / max(n - 1, 1)
        return np.stack([S.lv[i, r, g], S.ls[i, r, g], S.ls[i, r, g] - S.ls[i, r, g - 1], S.cm[i, r, g],
                         S.cmin[i, r, g], np.log2(d), agree, np.full(n, math.log1p(n))], 1).astype(np.float64)

    def spn_feats(self, i, cands):
        S = self.S
        if not cands:
            return np.array([[0, 0, 0, 0, 1, 0]], np.float64)
        r = np.array([c[0] for c in cands]); d = np.array([c[1] for c in cands])
        g = np.log2(d).astype(int) - 3
        return np.array([[math.log1p(len(cands)), S.lv[i, r, g].min(), S.ls[i, r, g].min(), S.cm[i, r, g].max(),
                          0, np.log2(d).mean()]], np.float64)

    def prices(self, kind, X):
        if self.price == "random":
            return self.rng.random(X.shape[0]), None
        pool = self.pools[kind]
        pi = pool.probs(X)
        if self.price == "single":
            return pi[pool.names.index("all")], pi
        return pool.weights() @ pi, pi

    def best(self, i, limit):
        """Best affordable action of puzzle i: (value, kind, k, cost_max, x, pi) or None."""
        cands = self.cands[i]
        opts = []
        if cands:
            ok = [k for k, (r, d) in enumerate(cands) if 2 * d <= CAPS[r] and d <= limit]
            if ok:
                X = self.ext_feats(i, cands)[ok]
                p, pi = self.prices("ext", X)
                cost = np.array([cands[k][1] for k in ok], float)
                j = int(np.argmax(p / cost))
                opts.append((p[j] / cost[j], "ext", ok[j], int(cost[j]), X[j], None if pi is None else pi[:, j]))
        if self.nxt[i] < R and D0 <= limit:
            X = self.spn_feats(i, cands)
            p, pi = self.prices("spn", X)
            opts.append((p[0] / D0, "spn", -1, D0, X[0], None if pi is None else pi[:, 0]))
        if not opts:
            return None
        return max(opts, key=lambda o: o[0])

    def execute(self, i, kind, k, x, pi):
        S = self.S
        if kind == "spn":
            r = int(self.nxt[i]); self.nxt[i] += 1
            d_old, d_new = 0, D0
            self.cands[i].append([r, D0])
            k = len(self.cands[i]) - 1
        else:
            r, d_old = self.cands[i][k]
            d_new = 2 * d_old
            self.cands[i][k][1] = d_new
        if self.mode == "chk":
            y = S.fv[i, r] <= d_new
            cost = (S.fv[i, r] - d_old) if y else (d_new - d_old)
            if y:
                self.settled[i] = True; self.ans[i] = S.fvc[i, r]
        else:
            cost = d_new - d_old
            y = bool(S.s[i, r, GI[d_new]] < self.tau)
            if y:
                self.settled[i] = True
        self.used[i] += cost
        if self.trace is not None:
            self.trace.append((i, 0 if kind == "spn" else 1, r, d_old, d_new, int(cost), int(bool(y))))
        if self.collect is not None:
            self.collect[kind][0].append(x); self.collect[kind][1].append(bool(y))
        if self.learn and pi is not None and self.price == "pool":
            self.pools[kind].update(pi, bool(y))
        return cost

    def finish(self):
        S = self.S
        for i in range(S.P):
            if self.mode == "chk" and self.settled[i]:
                continue
            c = self.cands[i]
            if not c:
                self.ans[i] = False; continue
            r = np.array([x[0] for x in c]); g = np.log2([x[1] for x in c]).astype(int) - 3
            k = int(np.nanargmin(S.s[i, r, g]))
            self.ans[i] = S.c[i, r[k], g[k]]
        return self.ans, self.used

    def run(self):
        S = self.S
        P = S.P
        self.cands = [[] for _ in range(P)]
        self.nxt = np.zeros(P, np.int64)
        self.settled = np.zeros(P, bool)
        self.ans = np.zeros(P, bool)
        self.used = np.zeros(P, np.int64)
        if self.local:
            for i in range(P):
                while not self.settled[i]:
                    o = self.best(i, self.b - self.used[i])
                    if o is None:
                        break
                    self.execute(i, o[1], o[2], o[4], o[5])
            return self.finish()
        rem = P * self.b
        heap = []
        for i in range(P):
            o = self.best(i, rem)
            if o is not None:
                heapq.heappush(heap, (-o[0], i))
        while heap:
            negv, i = heapq.heappop(heap)
            o = self.best(i, rem)
            if o is None:
                continue
            if heap and o[0] < -heap[0][0] - 1e-12 and o[0] < -negv - 1e-12:
                heapq.heappush(heap, (-o[0], i)); continue          # re-priced lower: back in the queue
            rem -= self.execute(i, o[1], o[2], o[4], o[5])
            assert rem >= 0
            if not self.settled[i]:
                o = self.best(i, rem)
                if o is not None:
                    heapq.heappush(heap, (-o[0], i))
        return self.finish()


# ------------------------------------------------------------------------------------------------------ fitting
def new_pools():
    return {"ext": Pool(EXT_TRADERS, EXT_FEATS), "spn": Pool(SPN_TRADERS, SPN_FEATS)}


def fit_market(val, mode, tau, log):
    """Stage 1: random-bid exploration on val; stage 2: on-policy data from the fitted market on val; refit;
    stage 3: Kelly wealth from a sequential val run at the primary budget."""
    data = {"ext": ([], []), "spn": ([], [])}
    rng = np.random.default_rng(1717)
    for b in (64, 256, 1024):
        for S in val:
            Market(S, b, None, mode=mode, tau=tau, price="random", rng=rng, collect=data).run()
    pools = new_pools()
    for kd in pools:
        pools[kd].fit(np.array(data[kd][0]), np.array(data[kd][1]))
    n1 = {kd: len(data[kd][1]) for kd in data}
    for b in (64, 256):
        for S in val:
            Market(S, b, pools, mode=mode, tau=tau, learn=False, collect=data).run()
    pools = new_pools()
    for kd in pools:
        pools[kd].fit(np.array(data[kd][0]), np.array(data[kd][1]))
    for S in val:
        Market(S, PRIMARY_B, pools, mode=mode, tau=tau, learn=True).run()
    info = {kd: {"n_explore": n1[kd], "n_total": len(data[kd][1]), "base_rate": pools[kd].base_rate,
                 "val_wealth": dict(zip(pools[kd].names, np.round(pools[kd].weights(), 4).tolist())),
                 "in_sample_logloss": pools[kd].logloss(np.array(data[kd][0]), np.array(data[kd][1]))}
            for kd in pools}
    log(f"market[{mode}] fitted: {json.dumps(info)}")
    return pools, info


def clone_pools(pools):
    out = {}
    for kd, p in pools.items():
        q = Pool({n: [] for n in p.names}, [])
        q.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.__dict__.items()})
        out[kd] = q
    return out


# --------------------------------------------------------------------------------------------------------- main
def run(summ, out):
    t0 = time.time()
    logf = open(os.path.join(out, "sim.log"), "a")

    def log(m):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    val = load_blocks(summ, VAL_BLOCKS)
    log(f"val blocks loaded; valid-not-correct in val bank = {sum(S.vnc for S in val)}")
    cat = lambda f, Ss, *a: np.concatenate([f(S, *a) for S in Ss])
    acc = lambda f, Ss, *a: float(cat(f, Ss, *a).mean())
    choices = {"budgets": BUDGETS, "primary_b": PRIMARY_B}
    val_table = {}

    # ---- val choices for statics
    for b in BUDGETS:
        sp = splits(b)
        vt = {}
        for D, B in sp:
            vt[f"D{D}B{B}"] = {"SR": acc(lambda S, D, B: sr(S, D, B)[0], val, D, B), "MV": acc(mv, val, D, B),
                               "SRchk": acc(sr_chk, val, D, B), "oracle": acc(oracle, val, D, B)}
        pick = lambda key: max(sp, key=lambda x: (vt[f"D{x[0]}B{x[1]}"][key], -sp.index(x)))
        choices[f"b{b}"] = {"SR": pick("SR"), "MV": pick("MV"), "SRchk": pick("SRchk")}
        val_table[f"b{b}"] = vt
        # SEQ-chk depth
        seq = {D: float(np.concatenate([seq_rule(S, b, D, "chk")[0] for S in val]).mean()) for D in (16, 32, 64, 128)}
        choices[f"b{b}"]["SEQchk_D"] = max(seq, key=lambda D: (seq[D], -D))
        val_table[f"b{b}"]["SEQchk"] = seq
        log(f"val b={b}: {json.dumps(choices[f'b{b}'])}")

    # ---- tau: largest threshold whose val candidates (depth >= 16) are correct >= 99.5% of the time
    s_all = np.concatenate([S.s[:, :, 1:].ravel() for S in val]); c_all = np.concatenate([S.c[:, :, 1:].ravel() for S in val])
    m = ~np.isnan(s_all); s_all, c_all = s_all[m], c_all[m]
    o = np.argsort(s_all); cs = np.cumsum(c_all[o]) / np.arange(1, len(o) + 1)
    okk = np.where(cs >= 0.995)[0]
    okk = okk[okk >= 100]
    tau_p = float(s_all[o][okk.max()]) if len(okk) else float(np.quantile(s_all, 0.01))
    choices["tau_prec995"] = tau_p
    taus = sorted(set([tau_p] + np.quantile(s_all, [0.05, 0.1, 0.2, 0.3, 0.5]).tolist()))
    for b in BUDGETS:
        rt = {}
        for D in (16, 32, 64, 128):
            for t in taus:
                rt[f"{D}|{t:.6g}"] = float(np.concatenate([seq_rule(S, b, D, "rt", t)[0] for S in val]).mean())
        best = max(rt, key=lambda k: rt[k])
        D, t = best.split("|")
        choices[f"b{b}"]["RT"] = (int(D), float(t))
        val_table[f"b{b}"]["RT"] = rt
    log(f"tau_prec995={tau_p:.6g}; RT picks: {[choices[f'b{b}']['RT'] for b in BUDGETS]}")

    # ---- market fitting on val
    pools_chk, info_chk = fit_market(val, "chk", None, log)
    pools_res, info_res = fit_market(val, "res", tau_p, log)
    choices["market_chk"] = info_chk; choices["market_res"] = info_res
    # val accuracy of markets (for the record)
    for b in BUDGETS:
        vp = clone_pools(pools_chk)
        val_table[f"b{b}"]["Mchk"] = float(np.concatenate([Market(S, b, vp, "chk").run()[0] for S in val]).mean())
    # secondary baseline pick at the primary budget
    cb = choices[f"b{PRIMARY_B}"]
    vt = val_table[f"b{PRIMARY_B}"]
    seqv = vt["SEQchk"][cb["SEQchk_D"]]; srchkv = vt[f"D{cb['SRchk'][0]}B{cb['SRchk'][1]}"]["SRchk"]
    choices["secondary_baseline"] = "SEQchk" if seqv >= srchkv else "SRchk"
    log(f"secondary baseline (val b={PRIMARY_B}): {choices['secondary_baseline']} (SEQchk {seqv:.4f}, SRchk {srchkv:.4f})")
    json.dump(choices, open(os.path.join(out, "choices.json"), "w"), indent=1, default=float)
    json.dump(val_table, open(os.path.join(out, "val_table.json"), "w"), indent=1, default=float)
    del val

    # ---- test: touched once
    test = load_blocks(summ, TEST_BLOCKS)
    log(f"test blocks loaded; valid-not-correct in test bank = {sum(S.vnc for S in test)}")
    res = {}

    def put(name, b, corr, nfe):
        res[f"{name}|{b}"] = (np.asarray(corr, bool), np.asarray(nfe, np.int64))

    for b in BUDGETS:
        c = choices[f"b{b}"]
        full = lambda: np.full(sum(S.P for S in test), b)
        D, B = c["SR"]; put("SR", b, cat(lambda S: sr(S, D, B)[0], test), full())
        D, B = c["MV"]; put("MV", b, cat(lambda S: mv(S, D, B), test), full())
        D, B = c["SRchk"]; put("SRchk", b, cat(lambda S: sr_chk(S, D, B), test), full())
        D, B = c["SR"]; put("oracle", b, cat(lambda S: oracle(S, D, B), test), full())
        put("FD", b, cat(lambda S: sr(S, b, 1)[0], test), full()) if b <= 1024 else None
        put("FR", b, cat(lambda S: sr(S, 16, b // 16)[0], test), full())
        put("RS", b, np.concatenate([rs(S, b, 17 + 1000 * j + b) for j, S in enumerate(test)]), full())
        # hindsight-best static split on test (robustness line only)
        hb = {x: cat(lambda S: sr(S, x[0], x[1])[0], test).mean() for x in splits(b)}
        hx = max(hb, key=hb.get); put("SR_hindsight", b, cat(lambda S: sr(S, hx[0], hx[1])[0], test), full())
        hm = {x: cat(lambda S: mv(S, x[0], x[1]), test).mean() for x in splits(b)}
        hx = max(hm, key=hm.get); put("MV_hindsight", b, cat(lambda S: mv(S, hx[0], hx[1]), test), full())
        Dst = c["SEQchk_D"]
        rr = [seq_rule(S, b, Dst, "chk") for S in test]; put("SEQchk", b, np.concatenate([x[0] for x in rr]), np.concatenate([x[1] for x in rr]))
        rr = [seq_rule(S, b, min(Dst, b), "chk", local=True) for S in test]; put("SEQchk_local", b, np.concatenate([x[0] for x in rr]), np.concatenate([x[1] for x in rr]))
        Dr, t = c["RT"]
        rr = [seq_rule(S, b, Dr, "rt", t) for S in test]; put("RT", b, np.concatenate([x[0] for x in rr]), np.concatenate([x[1] for x in rr]))
        for name, pools, mode, kw in [("Mchk", pools_chk, "chk", {}), ("Mchk_local", pools_chk, "chk", {"local": True}),
                                      ("Mchk_single", pools_chk, "chk", {"price": "single"}),
                                      ("Mchk_random", pools_chk, "chk", {"price": "random"}),
                                      ("Mres", pools_res, "res", {})]:
            pp = clone_pools(pools)          # every budget starts from the val wealth; online updates on test
            rng = np.random.default_rng(99 + b)
            rr, tr = [], []
            for j, S in enumerate(test):
                mk = Market(S, b, pp, mode, tau=tau_p, rng=rng, **kw)
                mk.trace = []
                rr.append(mk.run())
                tr += [(TEST_BLOCKS[j],) + t for t in mk.trace]
            put(name, b, np.concatenate([x[0] for x in rr]), np.concatenate([x[1] for x in rr]))
            if name in ("Mchk", "Mres") and b in (PRIMARY_B, 256):
                np.save(os.path.join(out, f"trace_{name}_b{b}.npy"), np.array(tr, np.int64))
            if name == "Mchk":
                res[f"Mchk_wealth_end|{b}"] = {kd: dict(zip(pp[kd].names, np.round(pp[kd].weights(), 4).tolist())) for kd in pp}
        # budget check: pooled per block <= 256*b, static exactly b
        for k, v in res.items():
            if k.endswith(f"|{b}") and isinstance(v, tuple):
                per_block = v[1].reshape(len(test), -1).sum(1)
                assert (per_block <= test[0].P * b).all(), (k, per_block.max(), test[0].P * b)
        log(f"test b={b}: " + ", ".join(f"{k.split('|')[0]}={v[0].mean():.4f}" for k, v in res.items()
                                         if k.endswith(f"|{b}") and isinstance(v, tuple)))
    arrs = {k: v[0] for k, v in res.items() if isinstance(v, tuple)}
    arrs.update({"nfe::" + k: v[1] for k, v in res.items() if isinstance(v, tuple)})
    np.savez_compressed(os.path.join(out, "per_puzzle.npz"), **arrs)
    json.dump({k: v for k, v in res.items() if not isinstance(v, tuple)},
              open(os.path.join(out, "wealth_test.json"), "w"), indent=1)
    log("SIM_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize", "run"])
    ap.add_argument("--bank"); ap.add_argument("--summ"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize(a.bank, a.out)
    else:
        os.makedirs(a.out, exist_ok=True)
        run(a.summ, a.out)
