"""World 1: 3-level hierarchical junta with exact-copy provenance (spec: ../world-critique/DESIGN-v1.md, W1).

Generative process (per episode)
  R ~ uniform over subsets of {0..B-1} with |R| in cfg.sizes (default B=16, |R|=2: 120 hypotheses)
  sigma2 ~ uniform over cfg.sig2 (default {0.1, 0.5});   g(c) ~ N(0,1) for each of the 2^|R| cells
  item latent value v_x = g(x_R) + b_x,  b_x ~ N(0, tau2)      (tau2 = 0 recovers the plain junta)
  presentation: token t is, with prob p_copy (per episode, see DEFAULTS; never at t=0), an exact COPY of a
    uniformly chosen earlier token (same item, same y, same event-id vector); otherwise the policy picks an item x
    and a FRESH measurement y = v_x + eps, eps ~ N(0, sigma2), with a new random event-id vector (R^ev_dim).
  query item q: with prob q_obs a uniformly chosen distinct item of the context, else uniform over all items.
  target: v_q = g(q_R) + b_q.

Policies (all ignorable: they use only visible item bits, recorded observations and independent randomness;
test_junta.py replays them against different hidden worlds):
  uniform  - uniform over the 2^B items
  zipf     - Zipf(a), a ~ U(0.8, 1.5) per episode, over ranks mapped to items by a per-episode random hash
  pattern  - with prob p ~ U(0.5, 0.95) the item matches a random visible bit pattern on f in {2..6} random
             coordinates (chosen independently of R), else uniform
  repeat   - with prob p ~ U(0.5, 0.95) one of K in {1..16} random items, else uniform
  ucb      - value-adaptive: with prob 0.2 (or when nothing is observed) a uniform item, else argmax over observed
             items of running mean + 1/sqrt(count) (running stats over presented tokens, copies included)

Tokens (F = B + 10): item bits (+-1), y, event-id vector (ev_dim), is_query flag. Query token = item bits + flag,
last position.

Exact Bayes: collapse tokens with the same event id; per item n_x, mean, within-item SS; under each (R, sigma2):
  item means ybar_x | g_c ~ N(g_c, tau2 + sigma2/n_x) independently; within-item SS ~ sigma2 chi2_{n_x-1};
  integrate each g_c analytically; mixture weights from marginal likelihoods; predictive for v_q is a mixture of
  120 x 2 Gaussians.
"""
import itertools, math
import numpy as np

LOG2PI = math.log(2 * math.pi)

# Every structural count is randomised per episode (no fixed count in training): copy rate (0 with prob
# p_copy_zero, else U(0, p_copy_max)), Zipf exponent, pattern hit rate and width, repeat-pool size and repeat rate;
# N is log-uniform in training (harness).
DEFAULTS = dict(B=16, sizes=[2], tau2=0.25, sig2=[0.1, 0.5], p_copy_zero=1 / 3, p_copy_max=0.6, ev_dim=8, q_obs=0.5,
                train_policies=['uniform', 'zipf', 'pattern', 'repeat'], heldout=['ucb'],
                zipf_a=[0.8, 1.5], pattern_p=[0.5, 0.95], pattern_f=[2, 3, 4, 5, 6], repeat_p=[0.5, 0.95],
                repeat_k=[1, 16], ucb_eps=0.2, ucb_c=1.0)
ALL_POLICIES = ['uniform', 'zipf', 'pattern', 'repeat', 'ucb']


def _mix64(x):
    """splitmix64 finaliser on uint64 arrays (vectorised random hash)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class Junta:
    target_kind = 'real'

    def __init__(self, cfg=None):
        c = dict(DEFAULTS); c.update(cfg or {})
        self.cfg = c
        self.name = 'junta'
        self.B = c['B']; self.NI = 1 << self.B
        self.tau2 = float(c['tau2']); self.sig2 = np.array(c['sig2'], float)
        self.hyps = [r for k in c['sizes'] for r in itertools.combinations(range(self.B), k)]
        self.H = len(self.hyps)
        self.rmax = max(c['sizes'])
        self.hpos = np.zeros((self.H, self.rmax), np.int64)
        self.hmask = np.zeros((self.H, self.rmax), np.int64)
        for i, r in enumerate(self.hyps):
            self.hpos[i, :len(r)] = r; self.hmask[i, :len(r)] = 1
        self.ncell = 1 << self.rmax
        self.feat_dim = self.B + 2 + c['ev_dim']
        self.train_policies = list(c['train_policies']); self.heldout = list(c['heldout'])
        self.test_policies = self.train_policies + [p for p in self.heldout if p not in self.train_policies]
        self.fixed = None                      # fitted prior-level references (fit_fixed)
        self._zipf_as = np.linspace(c['zipf_a'][0], c['zipf_a'][1], 8)
        self._zipf_cdf = np.stack([np.cumsum(z / z.sum()) for z in
                                   (np.arange(1, self.NI + 1, dtype=float)[None, :] ** -self._zipf_as[:, None])])

    # ------------------------------------------------------------------ helpers
    def cells(self, items, hidx=None):
        """Cell index of items under hypotheses: (H, U) (or (len(hidx), U))."""
        pos = self.hpos if hidx is None else self.hpos[hidx]
        msk = self.hmask if hidx is None else self.hmask[hidx]
        items = np.asarray(items, np.int64)
        bits = (items[None, :, None] >> pos[:, None, :]) & 1
        return ((bits * msk[:, None, :]) << np.arange(self.rmax)).sum(-1)

    def cell_of(self, items, h):
        """Per-episode true cell: items (E, n), h (E,) -> (E, n)."""
        pos = self.hpos[h][:, None, :]; msk = self.hmask[h][:, None, :]
        bits = (items[..., None] >> pos) & 1
        return ((bits * msk) << np.arange(self.rmax)).sum(-1)

    # ------------------------------------------------------------------ sampling
    def _hidden(self, rh, E):
        return dict(h=rh.integers(0, self.H, E), s=rh.integers(0, len(self.sig2), E),
                    g=rh.standard_normal((E, self.ncell)))

    def _items(self, rp, E, N, policy):
        NI = self.NI; c = self.cfg
        if policy == 'uniform':
            return rp.integers(0, NI, (E, N))
        if policy == 'zipf':
            ai = rp.integers(0, len(self._zipf_as), E)
            u = rp.random((E, N))
            ranks = np.zeros((E, N), np.uint64)
            for j in np.unique(ai):
                ranks[ai == j] = np.searchsorted(self._zipf_cdf[j], u[ai == j]).clip(0, self.NI - 1)
            key = rp.integers(0, 2 ** 62, E, dtype=np.int64).astype(np.uint64)
            return (_mix64(ranks * np.uint64(0x9E3779B97F4A7C15) + key[:, None]) % np.uint64(NI)).astype(np.int64)
        if policy == 'pattern':
            f = rp.choice(c['pattern_f'], E)
            keys = rp.random((E, self.B))
            rank = np.argsort(np.argsort(keys, 1), 1)
            fmask = ((rank < f[:, None]) << np.arange(self.B)).sum(1)        # random f coordinates per episode
            pat = rp.integers(0, NI, E) & fmask
            x = rp.integers(0, NI, (E, N))
            hit = rp.random((E, N)) < rp.uniform(*c['pattern_p'], E)[:, None]
            return np.where(hit, (x & ~fmask[:, None]) | pat[:, None], x)
        if policy == 'repeat':
            lo, hi = c['repeat_k']
            K = rp.integers(lo, hi + 1, E)
            pool = rp.integers(0, NI, (E, hi))
            pick = pool[np.arange(E)[:, None], (rp.random((E, N)) * K[:, None]).astype(np.int64)]
            return np.where(rp.random((E, N)) < rp.uniform(*c['repeat_p'], E)[:, None], pick, rp.integers(0, NI, (E, N)))
        raise ValueError(policy)

    def _copies(self, rp, E, N, pc):
        """root[e, t] = index of the fresh token that token t presents (root == t for fresh tokens)."""
        cp = (rp.random((E, N)) < pc[:, None]); cp[:, 0] = False
        src = (rp.random((E, N)) * np.arange(N)[None, :]).astype(np.int64)
        root = np.tile(np.arange(N), (E, 1))
        ar = np.arange(E)
        for t in range(1, N):
            root[:, t] = np.where(cp[:, t], root[ar, src[:, t]], t)
        return root

    def sample(self, rng, E, N, policy, n=None, y_override=None, rng_hidden=None):
        """E episodes with N context tokens (first n[i] used if n given). Separate generator streams for the
        presentation policy (rp) and the hidden world (rh) so that replay tests can vary one and fix the other."""
        rp = np.random.default_rng(rng.integers(2 ** 63))
        rh = np.random.default_rng(rng.integers(2 ** 63)) if rng_hidden is None else rng_hidden
        n = np.full(E, N) if n is None else np.asarray(n)
        pc = np.where(rp.random(E) < self.cfg['p_copy_zero'], 0.0, rp.uniform(0, self.cfg['p_copy_max'], E))
        hid = self._hidden(rh, E)
        s2 = self.sig2[hid['s']]
        ar = np.arange(E)
        if policy == 'ucb':
            items, y, root, slot_b, slot_items, nsl = self._ucb(rp, rh, E, N, pc, hid, y_override)
        else:
            root = self._copies(rp, E, N, pc)
            items = self._items(rp, E, N, policy)[ar[:, None], root]
        q = self._query(rp, items, n)
        if policy != 'ucb':
            # item effects consistent per (episode, item), drawn over context items + query
            keys = np.concatenate([items, q[:, None]], 1) + (ar[:, None] << self.B)
            uk, inv = np.unique(keys, return_inverse=True)
            b = (rh.standard_normal(len(uk)) * math.sqrt(self.tau2))[inv.reshape(keys.shape)]
            v = hid['g'][ar[:, None], self.cell_of(np.concatenate([items, q[:, None]], 1), hid['h'])] + b
            eps = rh.standard_normal((E, N)) * np.sqrt(s2)[:, None]
            yf = v[:, :N] + eps
            if y_override is not None:
                yf = y_override
            y = yf[ar[:, None], root]
            theta = v[:, N]
        else:
            match = (slot_items == q[:, None]) & (np.arange(slot_items.shape[1])[None, :] < nsl[:, None])
            bq = np.where(match.any(1), slot_b[ar, match.argmax(1)], rh.standard_normal(E) * math.sqrt(self.tau2))
            theta = hid['g'][ar, self.cell_of(q[:, None], hid['h'])[:, 0]] + bq
        ev = rh.standard_normal((E, N, self.cfg['ev_dim']))[ar[:, None], root]
        ep = dict(items=items, y=y.astype(np.float64), root=root, ev=ev.astype(np.float32), n=n, q=q,
                  target=theta, h=hid['h'], s=hid['s'], pc=pc, N=N,
                  policy=np.full(E, ALL_POLICIES.index(policy)))
        ep['X'], ep['mask'] = self.tokens(ep)
        return ep

    def assemble(self, rows):
        """Build an episode batch from constructed contexts. rows: list of dicts with items, y, root, ev (n, ev_dim),
        q, and optional target, h, s, pc. Contexts are padded to the longest; tokens rebuilt."""
        E = len(rows); N = max(len(r['items']) for r in rows); D = self.cfg['ev_dim']
        ep = dict(items=np.zeros((E, N), np.int64), y=np.zeros((E, N)), root=np.tile(np.arange(N), (E, 1)),
                  ev=np.zeros((E, N, D), np.float32), n=np.array([len(r['items']) for r in rows]),
                  q=np.array([r['q'] for r in rows]), target=np.array([r.get('target', 0.0) for r in rows]),
                  h=np.array([r.get('h', 0) for r in rows]), s=np.array([r.get('s', 0) for r in rows]),
                  pc=np.array([r.get('pc', 0.0) for r in rows]), policy=np.full(E, -1), N=N)
        for i, r in enumerate(rows):
            k = len(r['items'])
            ep['items'][i, :k] = r['items']; ep['y'][i, :k] = r['y']; ep['root'][i, :k] = r['root']
            ep['ev'][i, :k] = r['ev']
        ep['X'], ep['mask'] = self.tokens(ep)
        return ep

    def _query(self, rp, items, n):
        E = len(items)
        q = rp.integers(0, self.NI, E)
        use = rp.random(E) < self.cfg['q_obs']
        u = rp.random(E)
        for i in np.nonzero(use & (n > 0))[0]:
            d = np.unique(items[i, :n[i]])
            q[i] = d[int(u[i] * len(d))]
        return q

    def _ucb(self, rp, rh, E, N, pc, hid, y_override):
        c = self.cfg; ar = np.arange(E)
        s = np.sqrt(self.sig2[hid['s']])
        sit = np.full((E, N), -1, np.int64); sb = np.zeros((E, N)); ssum = np.zeros((E, N)); scnt = np.zeros((E, N))
        nsl = np.zeros(E, np.int64)
        items = np.zeros((E, N), np.int64); y = np.zeros((E, N)); root = np.zeros((E, N), np.int64)
        tslot = np.zeros((E, N), np.int64)
        for t in range(N):
            cp = (rp.random(E) < pc) & (t > 0)
            src = (rp.random(E) * t).astype(np.int64)
            explore = (rp.random(E) < c['ucb_eps']) | (nsl == 0)
            xr = rp.integers(0, self.NI, E)
            m = max(int(nsl.max()), 1)
            valid = np.arange(m)[None, :] < nsl[:, None]
            score = np.where(valid, ssum[:, :m] / np.maximum(scnt[:, :m], 1) + c['ucb_c'] / np.sqrt(np.maximum(scnt[:, :m], 1)), -np.inf)
            best = score.argmax(1)
            hit = (sit[:, :m] == xr[:, None]) & valid
            found = hit.any(1)
            slot = np.where(explore, np.where(found, hit.argmax(1), nsl), best)
            new = explore & ~found & ~cp
            sit[ar[new], nsl[new]] = xr[new]
            sb[ar[new], nsl[new]] = rh.standard_normal(new.sum()) * math.sqrt(self.tau2)
            nsl = nsl + new
            x = sit[ar, slot]
            v = hid['g'][ar, self.cell_of(x[:, None], hid['h'])[:, 0]] + sb[ar, slot]
            yf = v + rh.standard_normal(E) * s
            if y_override is not None:
                yf = y_override[:, t]
            fresh = ~cp
            items[:, t] = np.where(fresh, x, items[ar, src])
            y[:, t] = np.where(fresh, yf, y[ar, src])
            root[:, t] = np.where(fresh, t, root[ar, src])
            tslot[:, t] = np.where(fresh, slot, tslot[ar, src])
            ssum[ar, tslot[:, t]] += y[:, t]; scnt[ar, tslot[:, t]] += 1
        return items, y, root, sb, sit, nsl

    def sample_train(self, rng, E, nmin, nmax):
        """Training batch: policy mixture (equal shares of train_policies), N_i log-uniform in [nmin, nmax]."""
        P = self.train_policies
        parts = []
        sizes = [E // len(P) + (i < E % len(P)) for i in range(len(P))]
        for p, e in zip(P, sizes):
            n = np.exp(rng.uniform(math.log(nmin), math.log(nmax + 1), e)).astype(np.int64).clip(nmin, nmax)
            parts.append(self.sample(rng, e, nmax, p, n=n))
        return {k: np.concatenate([d[k] for d in parts]) if k != 'N' else nmax for k in parts[0]}

    def tokens(self, ep):
        E, N = ep['items'].shape
        B = self.B
        X = np.zeros((E, N + 1, self.feat_dim), np.float32)
        X[:, :N, :B] = ((ep['items'][..., None] >> np.arange(B)) & 1) * 2 - 1
        X[:, :N, B] = ep['y']
        X[:, :N, B + 1:B + 1 + self.cfg['ev_dim']] = ep['ev']
        X[:, N, :B] = ((ep['q'][:, None] >> np.arange(B)) & 1) * 2 - 1
        X[:, N, -1] = 1
        mask = np.zeros((E, N + 1), bool)
        mask[:, :N] = np.arange(N)[None, :] < ep['n'][:, None]
        mask[:, N] = True
        X[:, :N][~mask[:, :N]] = 0
        return X, mask

    # ------------------------------------------------------------------ exact inference
    def _stats(self, items, y, root, dedup):
        idx = np.nonzero(root == np.arange(len(root)))[0] if dedup else np.arange(len(root))
        it, yy = items[idx], y[idx]
        u, inv = np.unique(it, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(u)).astype(float)
        ybar = np.bincount(inv, yy, len(u)) / cnt
        ss = np.bincount(inv, (yy - ybar[inv]) ** 2, len(u))
        return u, cnt, ybar, ss

    def posterior(self, items, y, root, q, dedup=True, hidx=None, sig_idx=None):
        """Exact predictive for v_q as a Gaussian mixture (logw normalised, mean, var), one episode.
        hidx: restrict hypotheses (oracle); dedup=False treats copies as fresh (copy-naive)."""
        tau2 = self.tau2
        hsel = np.arange(self.H) if hidx is None else np.atleast_1d(hidx)
        Hn = len(hsel)
        sl = range(len(self.sig2)) if sig_idx is None else [sig_idx]
        if len(items) == 0:
            m = np.zeros(Hn * len(sl)); v = np.full(Hn * len(sl), 1 + tau2)
            return np.full(len(m), -math.log(len(m))), m, v
        u, cnt, ybar, ss = self._stats(items, y, root, dedup)
        U = len(u)
        cid = self.cells(u, hsel)                                 # (Hn, U)
        flat = (cid + self.ncell * np.arange(Hn)[:, None]).ravel()
        qi = np.searchsorted(u, q); qobs = qi < U and u[min(qi, U - 1)] == q
        qc = self.cells([q], hsel)[:, 0]
        LW, M, V = [], [], []
        for si in sl:
            s2 = self.sig2[si]
            e = s2 / cnt; v = tau2 + e
            within = float((-(cnt - 1) / 2 * (LOG2PI + math.log(s2)) - ss / (2 * s2)).sum())
            A = np.bincount(flat, np.broadcast_to(1 / v, (Hn, U)).ravel(), Hn * self.ncell).reshape(Hn, self.ncell)
            S1 = np.bincount(flat, np.broadcast_to(ybar / v, (Hn, U)).ravel(), Hn * self.ncell).reshape(Hn, self.ncell)
            P = 1 + A
            lml = (-0.5 * (np.log(2 * math.pi * v).sum() + (ybar ** 2 / v).sum())
                   + (0.5 * S1 ** 2 / P - 0.5 * np.log(P)).sum(1) + within)
            Pq = P[np.arange(Hn), qc]; Sq = S1[np.arange(Hn), qc]
            if qobs:
                vq = v[qi]; eq = e[qi]
                P_ = Pq - 1 / vq; S_ = Sq - ybar[qi] / vq
                pv = 1 / P_ + tau2
                prec = 1 / pv + 1 / eq
                mean = (S_ / P_ / pv + ybar[qi] / eq) / prec; var = 1 / prec
            else:
                mean = Sq / Pq; var = 1 / Pq + tau2
            LW.append(lml); M.append(mean); V.append(var)
        lw = np.concatenate(LW); lw = lw - lw.max(); lw -= math.log(np.exp(lw).sum())
        return lw, np.concatenate(M), np.concatenate(V)

    def posterior_item(self, items, y, root, q):
        """Item-only Bayes: every item its own cell (pools repeats of the same item, no sharing)."""
        tau2 = self.tau2; LW, M, V = [], [], []
        if len(items) == 0:
            return np.zeros(1), np.zeros(1), np.full(1, 1 + tau2)
        u, cnt, ybar, ss = self._stats(items, y, root, True)
        qi = np.searchsorted(u, q); qobs = qi < len(u) and u[min(qi, len(u) - 1)] == q
        for s2 in self.sig2:
            e = s2 / cnt; vm = 1 + tau2 + e
            lml = (-0.5 * (np.log(2 * math.pi * vm) + ybar ** 2 / vm)).sum() + \
                  (-(cnt - 1) / 2 * (LOG2PI + math.log(s2)) - ss / (2 * s2)).sum()
            if qobs:
                prec = 1 / (1 + tau2) + 1 / e[qi]; M.append(ybar[qi] / e[qi] / prec); V.append(1 / prec)
            else:
                M.append(0.0); V.append(1 + tau2)
            LW.append(lml)
        lw = np.array(LW); lw -= lw.max(); lw -= math.log(np.exp(lw).sum())
        return lw, np.array(M), np.array(V)

    def _episode(self, ep, i):
        n = ep['n'][i]
        return ep['items'][i, :n], ep['y'][i, :n], ep['root'][i, :n], ep['q'][i]

    def exact_bayes(self, ep):
        return _stack([self.posterior(*self._episode(ep, i)) for i in range(len(ep['q']))])

    def references(self, ep):
        E = len(ep['q']); refs = {}
        refs['bayes'] = self.exact_bayes(ep)
        refs['oracle'] = _stack([self.posterior(*self._episode(ep, i), hidx=ep['h'][i]) for i in range(E)])
        refs['count'] = _stack([self.posterior(*self._episode(ep, i), dedup=False) for i in range(E)])
        refs['item'] = _stack([self.posterior_item(*self._episode(ep, i)) for i in range(E)])
        refs['prior'] = (np.ones((E, 1)), np.zeros((E, 1)), np.full((E, 1), 1 + self.tau2))
        if self.fixed is not None:
            for k in ('r0', 'shortcut'):
                m, v = self._fixed_pred(k, ep)
                refs[k] = (np.ones((E, 1)), m[:, None], v[:, None])
        return refs

    # ------------------------------------------------------------------ prior-level references
    def _qfeat(self, ep):
        """Per-episode raw query features: raw token count of the query item (copies counted) and raw mean."""
        E, N = ep['items'].shape
        valid = np.arange(N)[None, :] < ep['n'][:, None]
        hit = (ep['items'] == ep['q'][:, None]) & valid
        nr = hit.sum(1).astype(float)
        ym = np.where(nr > 0, (ep['y'] * hit).sum(1) / np.maximum(nr, 1), 0.0)
        return nr, ym, ep['n'].astype(float)

    def _fixed_pred(self, k, ep, p=None):
        p = self.fixed[k] if p is None else p
        nr, ym, n = self._qfeat(ep)
        if k == 'r0':                       # fixed shrinkage of the query's own raw readings, copy-blind
            kk, a, b, c0 = np.exp(p)
            m = nr * ym / (nr + kk)
            v = np.where(nr > 0, a + b / (nr + kk), c0)
            return m, v
        F = np.stack([np.ones_like(nr), (nr > 0).astype(float), np.log1p(nr), nr / n], 1)   # counts only
        return F @ p[:4], np.exp(F @ p[4:])

    def fit_fixed(self, rng, E=20000, nmin=8, nmax=64):
        """Fit the prior-level references by NLL on training-distribution episodes."""
        from scipy.optimize import minimize
        ep = self.sample_train(rng, E, nmin, nmax)
        y = ep['target']; out = {}
        for k, x0 in (('r0', np.log([1.0, 0.1, 0.5, 1 + self.tau2])), ('shortcut', np.zeros(8))):
            def f(p):
                m, v = self._fixed_pred(k, ep, p)
                return float((0.5 * np.log(2 * math.pi * v) + 0.5 * (y - m) ** 2 / v).mean())
            r = minimize(f, x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-5, fatol=1e-7))
            out[k] = r.x
        self.fixed = out
        return {k: v.tolist() for k, v in out.items()}

    def set_fixed(self, d):
        self.fixed = {k: np.array(v) for k, v in d.items()}

    # ------------------------------------------------------------------ per-episode metadata for splits
    def meta(self, ep):
        E, N = ep['items'].shape
        valid = np.arange(N)[None, :] < ep['n'][:, None]
        fresh = (ep['root'] == np.arange(N)[None, :]) & valid
        qhit = ep['items'] == ep['q'][:, None]
        return dict(q_obs=(qhit & valid).any(1), n_q_raw=(qhit & valid).sum(1), n_q_fresh=(qhit & fresh).sum(1),
                    n_copies=(valid & ~fresh).sum(1), pc=ep['pc'], h=ep['h'], s=ep['s'], policy=ep['policy'],
                    n_distinct=np.array([len(np.unique(ep['items'][i, :ep['n'][i]])) for i in range(E)]))

    splits = {'all': lambda m: np.ones(len(m['q_obs']), bool),
              'q_unobs': lambda m: ~m['q_obs'], 'q_obs': lambda m: m['q_obs'],
              'copies': lambda m: m['n_copies'] > 0, 'nocopies': lambda m: m['n_copies'] == 0}
    # (label, weaker reference, stronger reference, split): fraction of the weaker->Bayes gain the model achieves
    abilities = [('similarity (item->bayes)', 'item', 'bayes', 'all'),
                 ('dedup (count->bayes)', 'count', 'bayes', 'copies'),
                 ('R-inference (bayes vs oracle)', 'bayes', 'oracle', 'all')]


def _stack(lst):
    """List of per-episode (logw, mean, var) -> padded (w, m, v) arrays (E, K)."""
    K = max(len(t[0]) for t in lst); E = len(lst)
    w = np.zeros((E, K)); m = np.zeros((E, K)); v = np.ones((E, K))
    for i, (lw, mm, vv) in enumerate(lst):
        k = len(lw); w[i, :k] = np.exp(lw); m[i, :k] = mm; v[i, :k] = vv
    return w, m, v


def make_world(cfg=None):
    return Junta(cfg)
