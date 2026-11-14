"""Logical-induction question families for the traders-net substrate (S0 item 3).

Three families, all deterministic given the seed:

D-delay   Labels revealed D steps late, D in {0, 64, 512}. Trades are booked at the price quoted
          at the trade time and settled at reveal (DelayBook). For D > 0 half the questions are
          "twins" (same generator and parameters as a D=0 base question, so the answer is known
          elsewhere in the market as soon as the base resolves) and half "solo" (fresh generator).
D-prove   Questions never drawn by a sampler: A|~A (1), A&~A (0), (A&B)->A (1) over D=0 base
          questions, and exclusive-exhaustive triples X_k = [a+b == k] (exactly one true).
          A deductive process reveals them late: delay = prove_delay * formula size. Each is
          released at a staggered time, so every one has a window with no resolved history.
D-expensive  A CA110 cell or LFSR16 bit k steps ahead of the current stream, k in {64,256,1024},
          revealed at t+k: the delay grows with the simulation cost (k cell updates / shifts).
          Note: structurally this is "predict k ahead from the current window", the same shape as
          a solo D-delay question; the difference is the generator (computable by simulation).

Also AND/OR compounds of two D=0 bases (delay 0) so AND/OR coherence is checkable in-stream.

Time convention (matches wealth-net for delay 0): at trade time t the market quotes every
question's bit bits[q, t]. The label of (q, u) is revealed at u + delay[q], after the trade at that
step (test, then train). So at trade time t the revealed labels are u <= t - 1 - delay[q] and
window(t)[q] = bits[q, t-delay-W : t-delay], with 0.5 for positions before release or before 0.
"""
import numpy as np

try:
    from streams import periodic, lfsr16, ca, prime
except ImportError:  # run from outside substrate/
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from streams import periodic, lfsr16, ca, prime

GENS = ["periodic", "lfsr16", "ca110", "prime", "ca30"]
KINDS = ["base", "twin", "solo", "and", "or", "taut_or", "contra", "impl", "ex3", "exp"]


def _gen(name, rng, T):
    if name == "periodic":
        period = int(rng.integers(2, 17)); phase = int(rng.integers(0, period))
        return periodic(period, phase, T), f"periodic p={period} phase={phase}"
    if name == "lfsr16":
        st = int(rng.integers(1, 65536)); return lfsr16(st, T), f"lfsr16 state={st}"
    if name in ("ca110", "ca30"):
        r = int(rng.integers(1, 2 ** 32)); return ca(int(name[2:]), r, T), f"{name} row={r:#x}"
    if name == "prime":
        n0 = int(rng.integers(2, 65536)); return prime(n0, T), f"prime n0={n0}"
    raise ValueError(name)


class LIStream:
    """Arrays: bits[Q,T] uint8, fam[Q] (index into fam_names), kind[Q] (str), ops[Q,2] (-1 none),
    group[Q] (triple id or -1), delay[Q], release[Q], sampled[Q] (False for D-prove), cost[Q]."""

    def __init__(self, bits, fam_names, fam, kind, ops, group, delay, release, sampled, cost, desc, W,
                 split=0.8):
        self.bits, self.fam_names, self.fam = bits, fam_names, np.asarray(fam)
        self.kind, self.ops, self.group = np.asarray(kind), np.asarray(ops), np.asarray(group)
        self.delay, self.release = np.asarray(delay), np.asarray(release)
        self.sampled, self.cost, self.desc, self.W = np.asarray(sampled), np.asarray(cost), desc, W
        self.Q, self.T = bits.shape
        self.t0 = W                                  # first trade time with a full D=0 window
        # disjoint target positions: train [t0, split), eval [split, T - max delay) so every eval
        # label is revealed inside the stream.
        last = self.T - int(self.delay.max())
        self.split = int(self.t0 + (last - self.t0) * split)
        self.train_range = (self.t0, self.split)
        self.eval_range = (self.split, last)
        # padded copy so windows can index negative times: pad = W + max delay
        self._pad = W + int(self.delay.max())
        self._pb = np.concatenate([np.full((self.Q, self._pad), 0.5, np.float32),
                                   bits.astype(np.float32)], 1)
        # mask out labels posed before release
        for q in range(self.Q):
            self._pb[q, self._pad:self._pad + self.release[q]] = 0.5

    # ---- worker1's Stream interface -------------------------------------------------------
    def window(self, t):
        """[n,Q,W] labels revealed before trade time t (0.5 = unknown)."""
        t = np.atleast_1d(np.asarray(t))
        end = t[:, None] - self.delay[None, :] + self._pad          # [n,Q] exclusive end
        idx = end[:, :, None] - self.W + np.arange(self.W)[None, None, :]
        return self._pb[np.arange(self.Q)[None, :, None], idx]

    def label(self, t):
        t = np.atleast_1d(np.asarray(t))
        return self.bits[:, t].T.astype(np.float32)

    def label_mask(self, t):
        """[n,Q] True where the question is live (released) at trade time t."""
        t = np.atleast_1d(np.asarray(t))
        return (t[:, None] >= self.release[None, :]) & (t[:, None] >= self.t0)

    def reveal(self, t):
        """[n,Q] time at which the label of (q, t) is revealed (settlement time)."""
        t = np.atleast_1d(np.asarray(t))
        return t[:, None] + self.delay[None, :]

    def unresolved_history(self, t):
        """[n,Q] True where question q has no revealed label yet at trade time t."""
        t = np.atleast_1d(np.asarray(t))
        return t[:, None] - 1 - self.delay[None, :] < self.release[None, :]

    def as_positions(self):
        """Adapter to worker1's prequential.py interface (position i predicts time t = i + W,
        torch tensors). Same delay semantics; unknown window entries are 0.5 here and there."""
        return _Positions(self)

    def family_of(self, name):
        return np.where(self.fam == self.fam_names.index(name))[0]


class _Positions:
    def __init__(self, st):
        self.st, self.Q, self.W, self.n_pos = st, st.Q, st.W, st.T - st.W
        self.fam, self.n_fam, self.delay, self.desc = st.fam.astype(np.int64), len(st.fam_names), \
            st.delay.astype(np.int64), st.desc

    def window(self, idx):
        import torch
        return torch.as_tensor(self.st.window(np.asarray(idx) + self.W), dtype=torch.float32)

    def label(self, idx):
        import torch
        return torch.as_tensor(self.st.label(np.asarray(idx) + self.W))

    def label_mask(self, idx):
        import torch
        return torch.as_tensor(self.st.label_mask(np.asarray(idx) + self.W))


def make_li_stream(seed=0, T=20000, W=32, delays=(0, 64, 512), n_per_delay=10, n_andor=4,
                   n_prove=4, n_triples=2, prove_delay=512, exp_ks=(64, 256, 1024), n_exp=2,
                   release_frac=0.5, split=0.8):
    rng = np.random.default_rng(10_000 + seed)
    fam_names, cols = [], dict(bits=[], fam=[], kind=[], ops=[], group=[], delay=[], release=[],
                                sampled=[], cost=[], desc=[])
    maxk = max(exp_ks)

    def fam_id(name):
        if name not in fam_names:
            fam_names.append(name)
        return fam_names.index(name)

    def add(b, fname, kind, desc, ops=(-1, -1), group=-1, delay=0, release=0, sampled=True, cost=1):
        cols["bits"].append(b.astype(np.uint8)); cols["fam"].append(fam_id(fname))
        cols["kind"].append(kind); cols["ops"].append(tuple(ops)); cols["group"].append(group)
        cols["delay"].append(int(delay)); cols["release"].append(int(release))
        cols["sampled"].append(sampled); cols["cost"].append(int(cost)); cols["desc"].append(desc)
        return len(cols["bits"]) - 1

    # D-delay. D=0 bases: n_per_delay questions cycling over the generators.
    base, base_gen = [], []
    for j in range(n_per_delay):
        g = GENS[j % len(GENS)]
        b, d = _gen(g, rng, T)
        base.append(add(b, f"delay{delays[0]}", "base", d, delay=delays[0])); base_gen.append(g)
    for D in delays[1:]:
        twin_src = rng.permutation(base)                 # distinct bases per delay
        for j in range(n_per_delay):
            if j % 2 == 0:                               # twin of a base question
                src = int(twin_src[(j // 2) % len(base)])
                add(cols["bits"][src], f"delay{D}", "twin", f"twin(q{src}) D={D}", ops=(src, -1), delay=D)
            else:
                b, d = _gen(GENS[(j // 2) % len(GENS)], rng, T)
                add(b, f"delay{D}", "solo", d + f" D={D}", delay=D)

    used = set()

    def pair():
        for _ in range(100):                             # avoid duplicate operand pairs
            i, j = rng.choice(len(base), size=2, replace=False)
            if (i, j) not in used:
                break
        used.add((i, j)); used.add((j, i))
        return base[int(i)], base[int(j)]

    # AND / OR over D=0 bases (delay 0, sampled as usual)
    for name in ("and", "or"):
        for _ in range(n_andor):
            a, b_ = pair(); A, B = cols["bits"][a], cols["bits"][b_]
            add((A & B) if name == "and" else (A | B), name, name, f"{name}(q{a},q{b_})", ops=(a, b_))

    # D-prove: never sampled, revealed late by a deductive process, staggered releases
    def rel():
        return int(rng.integers(W, max(W + 1, int(T * release_frac))))
    one, zero = np.ones(T, np.uint8), np.zeros(T, np.uint8)
    for _ in range(n_prove):
        a = base[int(rng.integers(0, len(base)))]
        add(one, "prove", "taut_or", f"q{a} | ~q{a}", ops=(a, -1), delay=prove_delay, release=rel(), sampled=False)
    for _ in range(n_prove):
        a = base[int(rng.integers(0, len(base)))]
        add(zero, "prove", "contra", f"q{a} & ~q{a}", ops=(a, -1), delay=prove_delay, release=rel(), sampled=False)
    for _ in range(n_prove):
        a, b_ = pair()
        add(one, "prove", "impl", f"(q{a} & q{b_}) -> q{a}", ops=(a, b_), delay=int(prove_delay * 1.5),
            release=rel(), sampled=False)
    for g in range(n_triples):
        a, b_ = pair(); v = cols["bits"][a].astype(int) + cols["bits"][b_].astype(int)
        r = rel()
        for k in range(3):
            add((v == k).astype(np.uint8), "prove", "ex3", f"[q{a}+q{b_}=={k}]", ops=(a, b_), group=g,
                delay=2 * prove_delay, release=r, sampled=False)

    # D-expensive: generator bit k steps ahead, revealed at t+k, cost k
    for k in exp_ks:
        for gname in ("ca110", "lfsr16"):
            for _ in range(n_exp):
                g, d = _gen(gname, rng, T + maxk)
                add(g[k:k + T], f"exp{k}", "exp", f"{d} +{k} ahead", delay=k, cost=k)

    return LIStream(np.stack(cols["bits"]), fam_names, cols["fam"], cols["kind"], np.array(cols["ops"]),
                    cols["group"], cols["delay"], cols["release"], cols["sampled"], cols["cost"],
                    cols["desc"], W, split)


class DelayBook:
    """Book each trade at the price quoted at trade time; settle it at reveal time t + delay[q].
    Per step: book(t, prices) first, then settle(t, bits) (delay-0 trades settle the same step)."""

    def __init__(self, delay):
        self.delay = np.asarray(delay)
        self.H = int(self.delay.max()) + 1
        Q = len(self.delay)
        self.price = np.full((self.H, Q), np.nan)
        self.when = np.full((self.H, Q), -1, np.int64)
        self.open = 0

    def book(self, t, p, mask=None):
        m = np.ones(len(self.delay), bool) if mask is None else np.asarray(mask, bool)
        s = t % self.H
        self.price[s, m] = np.asarray(p)[m]; self.when[s, m] = t; self.open += int(m.sum())

    def settle(self, s, bits):
        """Trades whose reveal time is s. Returns (mask[Q], t_trade[Q], price[Q], outcome[Q])."""
        u = s - self.delay
        slot = u % self.H
        Q = len(self.delay)
        ok = (u >= 0) & (self.when[slot, np.arange(Q)] == u)
        price = np.where(ok, self.price[slot, np.arange(Q)], np.nan)
        y = np.where(ok, bits[np.arange(Q), np.clip(u, 0, bits.shape[1] - 1)].astype(np.int64), -1)
        self.when[slot[ok], np.arange(Q)[ok]] = -1; self.open -= int(ok.sum())
        return ok, u, price, y


def log_score_gain(price, y, prior=0.5, eps=1e-12):
    """LMSR settlement of a move prior -> price: log-score gain in nats (IG from the prior)."""
    p = np.clip(price, eps, 1 - eps); p0 = np.clip(prior, eps, 1 - eps)
    return np.where(y == 1, np.log(p / p0), np.log((1 - p) / (1 - p0)))


def violations(p, st, t=None, tol=0.05):
    """Coherence violations of prices p[n,Q] (probabilities). Returns {kind: dict(mean, frac, n)}.
    If t (trade times, [n]) is given, also the same restricted to questions with no revealed
    history at t ('pre:<kind>'), i.e. coherence before any resolution."""
    p = np.atleast_2d(p)
    a, b = st.ops[:, 0], st.ops[:, 1]
    out = {}

    def v_of(kind):
        qs = np.where(st.kind == kind)[0]
        if kind == "taut_or" or kind == "impl":
            v = 1 - p[:, qs]
        elif kind == "contra":
            v = p[:, qs]
        elif kind == "and":
            pa, pb, pc = p[:, a[qs]], p[:, b[qs]], p[:, qs]
            v = np.maximum(0, pc - np.minimum(pa, pb)) + np.maximum(0, pa + pb - 1 - pc)
        elif kind == "or":
            pa, pb, pc = p[:, a[qs]], p[:, b[qs]], p[:, qs]
            v = np.maximum(0, np.maximum(pa, pb) - pc) + np.maximum(0, pc - pa - pb)
        elif kind == "ex3":
            gs = np.unique(st.group[qs]); v = np.stack([np.abs(p[:, st.group == g].sum(1) - 1) for g in gs], 1)
            qs = np.array([np.where(st.group == g)[0][0] for g in gs])  # one representative per triple
        elif kind == "twin":
            v = np.abs(p[:, qs] - p[:, a[qs]])       # a twin should quote its base's price
        return qs, v

    for kind in ("and", "or", "taut_or", "contra", "impl", "ex3", "twin"):
        if not (st.kind == kind).any():
            continue
        qs, v = v_of(kind)
        live = np.ones_like(v, bool)
        if t is not None:
            live = st.label_mask(t)[:, qs]
        vv = v[live]
        out[kind] = dict(mean=float(vv.mean()) if vv.size else np.nan,
                         frac=float((vv > tol).mean()) if vv.size else np.nan, n=int(vv.size))
        if t is not None:
            pre = live & st.unresolved_history(t)[:, qs]
            vv = v[pre]
            out["pre:" + kind] = dict(mean=float(vv.mean()) if vv.size else np.nan,
                                      frac=float((vv > tol).mean()) if vv.size else np.nan, n=int(vv.size))
    return out


if __name__ == "__main__":
    st = make_li_stream(0, 4000)
    for q in range(st.Q):
        print(q, st.fam_names[st.fam[q]], st.kind[q], st.delay[q], st.release[q], st.desc[q],
              "mean=%.3f" % st.bits[q].mean())
