"""K1 Kelly-pool economy engine (worker1). numpy only. Siblings import this; do not edit (post requests in STATUS.md).

Market: every step a batch of B questions resolves (all Q if Q <= B). Each question is its own binary sequence whose
next bit depends on its last 4 bits through a hidden table P[q, ctx] (families below). Traders are Kelly bettors in a
pari-mutuel pool: trader i stakes w_i * stake / m_i on each of the m_i questions it trades, split across outcomes by its
belief b. The maker (the only inflow) stakes `sub` per question at belief 0.5. The price is the stake-weighted mean
belief (maker included), p = (sub*0.5 + sum s_i b_i) / (sub + sum s_i), so settlement s_i*(b_i(y)/p(y) - 1) sums to
zero with the maker's P&L exactly (conservation); the maker's loss is the subsidy that flows to traders.
Sink: compute price c (units of initial wealth w0 = 1) per question traded. Death: w < die_at -> slot freed (wealth
left goes to the sink). Split: w >= split_at and a free slot -> halve wealth, the child copies type/params/learned
state with mutation and the parent's clade (founder lineage).

Traders (beliefs computed before the outcome; learning after it: prequential):
  pattern-k  : KT counts over the last k bits (k in 0..4) with per-trader discount (forgetting)
  logistic-k : online logistic regression on [1, +-bit_1..bit_k], per-trader learning rate
Niches on: each trader trades only its own question (inherited, moves with prob mut_niche on split).
Niches off: each trader trades every question in the batch with one shared model (pooled learning).
Null (shuffle=True): the same population and learning, but before pricing/settlement the beliefs are permuted among
the traders on each question, so wealth is decoupled from skill (the 'beliefs shuffled' control).

Prior art (for K1.md): Kelly bettors in a pari-mutuel market = Bayesian model averaging, market selection
(Beygelzimer, Langford, Pennock 2012, https://arxiv.org/abs/1201.6655; Blume & Easley 2006 Econometrica,
https://doi.org/10.1111/j.1468-0262.2006.00680.x).
"""
import csv
import json
import os
import sys
import time

import numpy as np

FAMS = ['periodic', 'markov1', 'markov3', 'parity3', 'coin', 'noise']
SPECIES = [f'{t}{k}' for t in ('pat', 'log') for k in range(5)]   # species id = typ*5 + k
_C = np.arange(16)
XFEAT = np.array([[1.0] + [2.0 * ((c >> b) & 1) - 1 for b in range(4)] for c in range(16)])      # [16,5]
XM = np.stack([XFEAT * np.array([1.0] + [1.0 if b < k else 0.0 for b in range(4)]) for k in range(5)])  # [5,16,5]
MASK = np.array([(1 << k) - 1 for k in range(5)])


def gini(x):
    x = np.sort(np.maximum(np.asarray(x, float), 0))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float('nan')
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float('nan')


class Stream:
    """Q binary sequences; batch of B resolves per step; optional shift: at shift_at, shift_frac of questions redraw."""

    def __init__(self, Q, seed, batch=16, fam_p=None, shift_at=None, shift_frac=0.25, noise=0.02):
        self.rng = np.random.default_rng([seed, 11])
        self.Q, self.B = Q, min(batch, Q)
        self.fam_p = np.ones(len(FAMS)) / len(FAMS) if fam_p is None else np.asarray(fam_p, float)
        self.noise = noise
        self.P = np.zeros((Q, 16))
        self.fam = np.zeros(Q, int)
        self._draw(np.arange(Q))
        self.hist = self.rng.integers(0, 256, Q)
        self.shift_at, self.shift_frac, self.shifted = shift_at, shift_frac, np.zeros(Q, bool)

    def _draw(self, qs):
        r, e = self.rng, self.noise
        for q in qs:
            f = int(r.choice(len(FAMS), p=self.fam_p))
            name = FAMS[f]
            if name == 'periodic':
                p = int(r.integers(2, 5))
                P = ((_C >> (p - 1)) & 1) * (1 - 2 * e) + e
            elif name == 'markov1':
                P = r.beta(0.5, 0.5, 2)[_C & 1]
            elif name == 'markov3':
                P = r.beta(0.5, 0.5, 8)[_C & 7]
            elif name == 'parity3':
                par = ((_C & 1) ^ ((_C >> 1) & 1) ^ ((_C >> 2) & 1))
                P = par * (1 - 2 * 2.5 * e) + 2.5 * e
            elif name == 'coin':
                P = np.full(16, r.uniform(0.1, 0.9))
            else:
                P = np.full(16, 0.5)
            self.fam[q], self.P[q] = f, P

    def step(self, t):
        if self.shift_at is not None and t == self.shift_at:
            qs = self.rng.choice(self.Q, max(1, int(round(self.shift_frac * self.Q))), replace=False)
            self._draw(qs)
            self.shifted[qs] = True
        batch = np.arange(self.Q) if self.B == self.Q else np.sort(self.rng.choice(self.Q, self.B, replace=False))
        ctx = self.hist[batch] & 15
        ptrue = self.P[batch, ctx]
        y = self.rng.random(len(batch)) < ptrue
        self.hist[batch] = ((self.hist[batch] << 1) | y) & 255
        return batch, ctx, y, ptrue


class Economy:
    def __init__(self, N, Q, B, seed, niches=True, c=1e-4, subsidy=0.01, stake=1.0, split_at=4.0, die_at=0.01,
                 p_logistic=0.5, shuffle=False, eps=1e-3, mut_k=0.2, mut_niche=0.05, mut_type=0.02, mut_rate=0.3, dense=True):
        r = np.random.default_rng([seed, 22])            # same population for main and null
        self.rng = np.random.default_rng([seed, 33, int(shuffle)])
        self.N, self.Q, self.B = N, Q, B
        self.niches, self.c, self.stake, self.split_at, self.die_at = niches, c, stake, split_at, die_at
        self.shuffle, self.eps, self.dense = shuffle, eps, dense
        self.mut = (mut_k, mut_niche, mut_type, mut_rate)
        self.sub = subsidy * N / B                        # maker stake per resolved question (w0 = 1)
        self.alive = np.ones(N, bool)
        self.w = np.ones(N)
        self.typ = (r.random(N) < p_logistic).astype(np.int64)
        self.k = r.integers(0, 5, N)
        self.niche = r.integers(0, Q, N) if niches else np.full(N, -1)
        self.lr = np.exp(r.uniform(np.log(0.01), np.log(0.5), N))
        self.disc = 1 - np.exp(r.uniform(np.log(1e-3), np.log(0.1), N))
        self.counts = np.zeros((N, 16, 2))
        self.W = np.zeros((N, 5))
        self.uid = np.arange(N)
        self.clade = np.arange(N)
        self.next_uid = N
        self.W0 = float(N)
        self.births = self.bankrupt = 0
        self.inflow = self.sink_c = self.sink_dead = 0.0
        self.settle_resid = 0.0                            # max |sum pnl + maker pnl| / stake over the run
        self.cum_wf = np.zeros((10, len(FAMS)))            # cumulative income by species x family
        self.win_inc = np.zeros(N)
        self._reset_window()
        self.prev_uid = self.prev_w = None
        self.prev_inc = None
        self.post_settle = None                            # callable(econ) after settlement+sink, before die/split
        self.last_deaths = np.zeros(0, int)                # slot indices that died this step
        self.last_splits = (np.zeros(0, int), np.zeros(0, int))   # (parent idx, child idx) this step

    def _reset_window(self):
        nf = len(FAMS)
        self.acc = {k: np.zeros(nf) for k in ('n', 'brier', 'bce', 'price', 'ent', 'brier_or')}

    def pairs(self, batch):
        ids = np.flatnonzero(self.alive)
        if self.niches:
            pos = np.full(self.Q, -1)
            pos[batch] = np.arange(len(batch))
            J = pos[self.niche[ids]]
            sel = J >= 0
            return ids[sel], J[sel]
        return np.repeat(ids, len(batch)), np.tile(np.arange(len(batch)), len(ids))

    def beliefs(self, I, cx):
        """Belief of trader I[p] at masked context cx[p] (pair path, used with niches on)."""
        b = np.empty(len(I))
        pat = self.typ[I] == 0
        Ip, cp = I[pat], cx[pat]
        n1, n0 = self.counts[Ip, cp, 1], self.counts[Ip, cp, 0]
        b[pat] = (n1 + 0.5) / (n0 + n1 + 1.0)
        lg = ~pat
        Il, cl = I[lg], cx[lg]
        z = np.einsum('pf,pf->p', self.W[Il], XM[self.k[Il], cl])
        b[lg] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return b

    def tables(self, ids):
        """Belief tables [len(ids), 16] over masked contexts (dense path, niches off)."""
        cn = self.counts[ids]
        n1, n0 = cn[:, :, 1], cn[:, :, 0]
        tp = (n1 + 0.5) / (n0 + n1 + 1.0)
        z = np.einsum('nf,ncf->nc', self.W[ids], XM[self.k[ids]])
        tl = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.where((self.typ[ids] == 0)[:, None], tp, tl)

    def _quality(self, batch, y, p, py, ptrue, fam):
        f = fam[batch]
        yf = y.astype(float)
        a = self.acc
        np.add.at(a['n'], f, 1)
        np.add.at(a['brier'], f, (p - yf) ** 2)
        np.add.at(a['bce'], f, -np.log(py))
        np.add.at(a['price'], f, p)
        pt = np.clip(ptrue, 1e-12, 1 - 1e-12)
        np.add.at(a['ent'], f, -(yf * np.log(pt) + (1 - yf) * np.log(1 - pt)))
        np.add.at(a['brier_or'], f, (ptrue - yf) ** 2)
        return f

    def _after(self):
        if self.post_settle is not None:          # hook (worker3 FoF): may move wealth between alive leaves
            self.post_settle(self)
        dead = self.alive & (self.w < self.die_at)
        self.last_deaths = np.flatnonzero(dead)
        if dead.any():
            self.sink_dead += float(self.w[dead].sum())
            self.w[dead] = 0.0
            self.alive[dead] = False
            self.bankrupt += int(dead.sum())
        self._split()

    def _step_dense(self, batch, ctx, y, ptrue, fam):
        """Niches off: every alive trader trades every batch question; arrays [nA, B], no pair gathers.
        Null here permutes whole belief rows among traders (one permutation per step)."""
        B = len(batch)
        ids = np.flatnonzero(self.alive)
        nA = len(ids)
        f = fam[batch]
        if nA == 0:
            p = np.full(B, 0.5)
            self._quality(batch, y, p, np.full(B, 0.5), ptrue, fam)
            self._after()
            return
        C = ctx[None, :] & MASK[self.k[ids]][:, None]
        if B >= 16:
            bt_ = np.take_along_axis(self.tables(ids), C, 1)
        else:                                          # few questions: gather the needed cells only
            bt_ = self.beliefs(np.repeat(ids, B), C.ravel()).reshape(nA, B)
        b = np.clip(bt_, self.eps, 1 - self.eps)
        bt = b[self.rng.permutation(nA)] if self.shuffle else b
        st = self.stake * self.w[ids] / B
        S = st.sum() + self.sub
        p = (st @ bt + 0.5 * self.sub) / S
        py = np.where(y, p, 1 - p)
        pnl = st[:, None] * (np.where(y[None, :], bt, 1 - bt) / py[None, :] - 1)
        maker = self.sub * (0.5 / py - 1)
        res = np.abs(pnl.sum(0) + maker).max() / S
        self.settle_resid = max(self.settle_resid, float(res))
        self.inflow += float(-maker.sum())
        inc = pnl.sum(1)
        wi = self.w[ids] + inc
        pay = np.minimum(self.c * B, np.maximum(wi, 0))
        self.w[ids] = wi - pay
        self.sink_c += float(pay.sum())
        self.win_inc[ids] += inc - pay
        sp = self.typ[ids] * 5 + self.k[ids]
        for j in range(B):
            self.cum_wf[:, f[j]] += np.bincount(sp, pnl[:, j], 10)
        self._quality(batch, y, p, py, ptrue, fam)
        # learning (true beliefs)
        pat = self.typ[ids] == 0
        ip = ids[pat]
        self.counts[ip] *= (self.disc[ip] ** B)[:, None, None]
        idx = ip[:, None] * 32 + C[pat] * 2 + y[None, :].astype(np.int64)
        self.counts.reshape(-1)[:] += np.bincount(idx.ravel(), minlength=self.N * 32)
        lg = ~pat
        il = ids[lg]
        if len(il):
            r = y[None, :] - bt_[lg]
            rows = np.repeat(np.arange(len(il)), B)
            R = np.bincount(rows * 16 + C[lg].ravel(), r.ravel(), len(il) * 16).reshape(len(il), 16)
            self.W[il] += (self.lr[il] / B)[:, None] * np.einsum('nc,ncf->nf', R, XM[self.k[il]])
        self._after()

    def step(self, batch, ctx, y, ptrue, fam):
        if not self.niches and self.dense:
            return self._step_dense(batch, ctx, y, ptrue, fam)
        N, B = self.N, len(batch)
        I, J = self.pairs(batch)
        m = np.bincount(I, minlength=N).astype(float)
        cx = ctx[J] & MASK[self.k[I]]
        self._b_true = self.beliefs(I, cx)
        b = np.clip(self._b_true, self.eps, 1 - self.eps)
        bt = b
        if self.shuffle and len(I):
            src = np.lexsort((self.rng.random(len(I)), J))
            dst = np.argsort(J, kind='stable')
            bt = np.empty_like(b)
            bt[dst] = b[src]
        st = self.stake * self.w[I] / m[I]
        S = np.bincount(J, st, B) + self.sub
        p = (np.bincount(J, st * bt, B) + 0.5 * self.sub) / S
        yJ = y[J]
        py = np.where(y, p, 1 - p)
        pnl = st * (np.where(yJ, bt, 1 - bt) / py[J] - 1)
        maker = self.sub * (0.5 / py - 1)
        res = np.abs(np.bincount(J, pnl, B) + maker).max() / S.max()
        self.settle_resid = max(self.settle_resid, float(res))
        self.inflow += float(-maker.sum())
        inc = np.bincount(I, pnl, N)
        self.w += inc
        pay = np.minimum(self.c * m, np.maximum(self.w, 0))
        self.w -= pay
        self.sink_c += float(pay.sum())
        self.win_inc += inc - pay
        f = self._quality(batch, y, p, py, ptrue, fam)
        sp = self.typ[I] * 5 + self.k[I]
        self.cum_wf += np.bincount(sp * len(FAMS) + f[J], pnl, 10 * len(FAMS)).reshape(10, len(FAMS))
        self._learn(I, cx, yJ, m)     # learning on true beliefs
        self._after()

    def _learn(self, I, cx, yJ, m):
        """Pair-path learning; touches only traders that traded this step (O(pairs), not O(N))."""
        pat = self.typ[I] == 0
        Ip = I[pat]
        if len(Ip):
            up, inv = np.unique(Ip, return_inverse=True)
            self.counts[up] *= (self.disc[up] ** m[up])[:, None, None]
            loc = np.bincount(inv * 32 + cx[pat] * 2 + yJ[pat], minlength=len(up) * 32).reshape(len(up), 16, 2)
            self.counts[up] += loc
        lg = ~pat
        Il, cl = I[lg], cx[lg]
        if len(Il):
            ul, inv = np.unique(Il, return_inverse=True)
            r = yJ[lg] - self._b_true[lg]
            R = np.bincount(inv * 16 + cl, r, len(ul) * 16).reshape(len(ul), 16)
            g = np.einsum('nc,ncf->nf', R, XM[self.k[ul]])
            self.W[ul] += (self.lr[ul] / m[ul])[:, None] * g

    def _split(self):
        free = np.flatnonzero(~self.alive)
        cand = np.flatnonzero(self.alive & (self.w >= self.split_at))
        n = min(len(free), len(cand))
        self.last_splits = (np.zeros(0, int), np.zeros(0, int))
        if n == 0:
            return
        par = cand[np.argsort(-self.w[cand])[:n]]
        ch = free[:n]
        r = self.rng
        mk, mn, mt, mr = self.mut
        self.w[par] /= 2
        self.w[ch] = self.w[par]
        for a in ('typ', 'k', 'niche', 'lr', 'disc', 'clade'):
            getattr(self, a)[ch] = getattr(self, a)[par]
        self.counts[ch] = self.counts[par]
        self.W[ch] = self.W[par]
        dk = np.where(r.random(n) < mk, r.choice([-1, 1], n), 0)
        self.k[ch] = np.clip(self.k[ch] + dk, 0, 4)
        flip = r.random(n) < mt
        self.typ[ch] = np.where(flip, 1 - self.typ[ch], self.typ[ch])
        self.counts[ch[flip]] = 0
        self.W[ch[flip]] = 0
        if self.niches:
            mv = r.random(n) < mn
            self.niche[ch[mv]] = r.integers(0, self.Q, int(mv.sum()))
            self.counts[ch[mv]] = 0
            self.W[ch[mv]] = 0
        self.lr[ch] = np.clip(self.lr[ch] * np.exp(mr * r.standard_normal(n)), 1e-3, 2.0)
        self.disc[ch] = 1 - np.clip((1 - self.disc[ch]) * np.exp(mr * r.standard_normal(n)), 1e-4, 0.5)
        self.uid[ch] = np.arange(self.next_uid, self.next_uid + n)
        self.next_uid += n
        self.alive[ch] = True
        self.win_inc[ch] = 0.0
        self.births += n
        self.last_splits = (par, ch)

    def metrics(self):
        al = self.alive
        w = self.w[al]
        Wt = float(w.sum())
        row = {'W_traders': Wt, 'alive': int(al.sum()), 'dead_frac': 1 - al.mean(), 'births': self.births,
               'bankrupt': self.bankrupt, 'R_total': self.inflow, 'sink_compute': self.sink_c,
               'sink_dead': self.sink_dead, 'gini_traders': gini(w),
               'eff_traders': Wt ** 2 / float((w ** 2).sum()) if Wt > 0 else 0.0}
        row['cons_resid_ledger'] = abs(Wt - (self.W0 + self.inflow - self.sink_c - self.sink_dead)) / self.W0
        row['cons_resid_settle'] = self.settle_resid
        cw = np.bincount(self.clade[al], w, self.N)
        sh = cw / Wt if Wt > 0 else cw
        row['clades'] = int((cw > 0).sum())
        row['clades_1pct'] = int((sh >= 0.01).sum())
        row['top_clade_share'] = float(sh.max()) if Wt > 0 else float('nan')
        row['eff_clades'] = 1.0 / float((sh ** 2).sum()) if Wt > 0 else 0.0
        spw = np.bincount(self.typ[al] * 5 + self.k[al], w, 10)
        spn = np.bincount(self.typ[al] * 5 + self.k[al], None, 10)
        for i, s in enumerate(SPECIES):
            row[f'wealth_{s}'] = float(spw[i])
            row[f'n_{s}'] = int(spn[i])
        row['top_species_share'] = float(spw.max() / Wt) if Wt > 0 else float('nan')
        if self.niches:
            qn = np.bincount(self.niche[al], None, self.Q)
            row['niches_occupied'] = int((qn > 0).sum())
        else:
            row['niches_occupied'] = float('nan')
        a = self.acc
        n = np.maximum(a['n'], 1)
        for j, fn in enumerate(FAMS):
            ok = a['n'][j] > 0
            row[f'brier_{fn}'] = a['brier'][j] / n[j] if ok else float('nan')
            row[f'bce_{fn}'] = a['bce'][j] / n[j] if ok else float('nan')
            row[f'price_{fn}'] = a['price'][j] / n[j] if ok else float('nan')
            row[f'ent_{fn}'] = a['ent'][j] / n[j] if ok else float('nan')
        N = a['n'].sum()
        row['brier_all'] = a['brier'].sum() / max(N, 1)
        row['bce_all'] = a['bce'].sum() / max(N, 1)
        row['brier_oracle_all'] = a['brier_or'].sum() / max(N, 1)
        row['ent_all'] = a['ent'].sum() / max(N, 1)
        # churn: rank change of wealth among the same occupants since the last log; turnover: income ranks
        uid, ww, inc = self.uid[al], w, self.win_inc[al]
        if self.prev_uid is not None:
            common, ia, ib = np.intersect1d(uid, self.prev_uid, assume_unique=True, return_indices=True)
            row['churn_traders'] = 1 - spearman(ww[ia], self.prev_w[ib]) if len(common) > 2 else float('nan')
            row['turnover_traders'] = spearman(inc[ia], self.prev_inc[ib]) if len(common) > 2 else float('nan')
        else:
            row['churn_traders'] = row['turnover_traders'] = float('nan')
        self.prev_uid, self.prev_w, self.prev_inc = uid.copy(), ww.copy(), inc.copy()
        self.win_inc[:] = 0
        self._reset_window()
        return row


NULL_KEYS = ['gini_traders', 'alive', 'top_clade_share', 'eff_clades', 'clades_1pct', 'brier_all', 'bce_all',
             'W_traders', 'top_species_share', 'births', 'bankrupt']


class WideCSV:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path, self.f, self.w, self.keys = path, None, None, None

    def write(self, row):
        if self.keys is None:
            self.keys = list(row)
            self.f = open(self.path, 'w', newline='')
            self.w = csv.writer(self.f)
            self.w.writerow(self.keys)
        self.w.writerow([row.get(k, float('nan')) for k in self.keys])
        self.f.flush()


def run(N=1000, Q=64, niches=True, seed=1, T=20000, batch=16, log_every=None, out_root=None, run_id=None,
        shift_at=None, shift_frac=0.25, fam_p=None, null=True, runlog=True, verbose=True, setup=None,
        extra_metrics=None, **econ_kw):
    """Run main (and shuffled-beliefs null, in lockstep on the same stream). Returns final summary dict.
    Writes <out_root>/<run_id>/metrics.csv (wide, V1 schema) + <run_id>.csv (S0 runlog long) + summary.json."""
    log_every = log_every or max(1, T // 200)
    st = Stream(Q, seed, batch=batch, fam_p=fam_p, shift_at=shift_at, shift_frac=shift_frac)
    B = st.B
    main = Economy(N, Q, B, seed, niches=niches, **econ_kw)
    nul = Economy(N, Q, B, seed, niches=niches, shuffle=True, **econ_kw) if null else None
    if setup is not None:                                  # setup(main, nul) e.g. to attach post_settle hooks
        setup(main, nul)
    import hashlib
    code = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:12]
    cfg = dict(econ_sha=code, N=N, Q=Q, niches=niches, seed=seed, T=T, batch=batch, shift_at=shift_at, shift_frac=shift_frac,
               fam_p=fam_p, null=null, **econ_kw)
    wide = rl = None
    if out_root:
        d = os.path.join(out_root, run_id)
        wide = WideCSV(os.path.join(d, 'metrics.csv'))
        if runlog:
            sys.path.insert(0, '/home/agent/workspaces/eurisko-study/traders-net/study/S0/substrate')
            from runlog import RunLog
            rl = RunLog(d, run_id, meta=cfg)
    t0 = time.time()
    hist = []
    for t in range(T):
        batch_q, ctx, y, pt = st.step(t)
        main.step(batch_q, ctx, y, pt, st.fam)
        if nul is not None:
            nul.step(batch_q, ctx, y, pt, st.fam)
        if (t + 1) % log_every == 0:
            row = {'step': t + 1}
            row.update(main.metrics())
            if nul is not None:
                rn = nul.metrics()
                row['gini_null_traders'] = rn['gini_traders']
                for k in NULL_KEYS[1:]:
                    row[f'{k}_null'] = rn[k]
            if extra_metrics is not None:                  # extra_metrics(main, nul) -> dict of extra columns
                row.update(extra_metrics(main, nul))
            row['wall_s'] = time.time() - t0
            hist.append(row)
            if wide:
                wide.write(row)
            if rl:
                rl.log(t + 1, {k: v for k, v in row.items() if k != 'step'})
            if verbose and (t + 1) % (log_every * 20) == 0:
                print(f"[{run_id}] t={t+1} alive={row['alive']} W={row['W_traders']:.0f} gini={row['gini_traders']:.3f}"
                      f" top_clade={row['top_clade_share']:.3f} brier={row['brier_all']:.4f} {row['wall_s']:.0f}s",
                      flush=True)
    summ = summarize(hist, cfg)
    if out_root:
        with open(os.path.join(out_root, run_id, 'summary.json'), 'w') as f:
            json.dump(summ, f, indent=1, default=float)
        if rl:
            rl.close()
    return summ


def summarize(hist, cfg):
    last = hist[-1]
    tail = hist[-max(1, len(hist) // 10):]           # last 10% of logs for window metrics
    mean = lambda k: float(np.nanmean([h[k] for h in tail]))
    s = {'cfg': cfg, 'final': last}
    for k in ('brier_all', 'bce_all', 'brier_oracle_all', 'ent_all', 'brier_all_null', 'bce_all_null',
              'turnover_traders', 'churn_traders'):
        if k in last:
            s[f'tail_{k}'] = mean(k)
    s['monoculture'] = bool(last['top_clade_share'] >= 0.9) if last['alive'] > 0 else None
    if 'top_clade_share_null' in last:
        s['monoculture_null'] = bool(last['top_clade_share_null'] >= 0.9)
    return s
