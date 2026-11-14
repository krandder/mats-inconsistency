"""K1 worker2 driver: shift on/off and compute-price sweep on worker1's engine (K1/econ.py, read-only).
Adds shift-tracking columns via a Stream subclass + setup/extra_metrics hooks; settlement untouched.
Usage: python w2run.py ARM SEED   (ARM from ARMS below). Writes K1/runs/w2-<arm>-s<seed>/.
"""
import json, os, sys
import numpy as np
K1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, K1)
import econ

T, SHIFT_AT, B, SUB = 20000, 10000, 16, 0.01
CSTAR = {True: SUB * 64 / B, False: SUB / B}           # break-even compute price, niches on / off (Q = 64)
ARMS = {}
for tag, N, nich in (('10k', 10000, True), ('1k-nn', 1000, False)):
    ARMS[f'S-on{tag}'] = dict(N=N, niches=nich, shift=True, cr=0.0025)
    ARMS[f'S-off{tag}'] = dict(N=N, niches=nich, shift=False, cr=0.0025)
    for cr in (0, 0.01, 0.03, 0.1, 0.3, 1, 3):
        ARMS[f'C{cr:g}-{tag}'] = dict(N=N, niches=nich, shift=False, cr=cr)

_ST = {}


class TStream(econ.Stream):
    def __init__(self, Q, seed, **kw):
        super().__init__(Q, seed, **kw)
        self.track = np.zeros(Q, bool)
        self.placebo_rng = np.random.default_rng([seed, 99])
        self.t = -1
        _ST['s'] = self

    def step(self, t):
        self.t = t
        if t == SHIFT_AT:
            if self.shift_at is None:
                qs = self.placebo_rng.choice(self.Q, max(1, int(round(self.shift_frac * self.Q))), replace=False)
                self.track[qs] = True
        out = super().step(t)
        if t == SHIFT_AT and self.shift_at is not None:
            self.track = self.shifted.copy()
        return out


econ.Stream = TStream


def price(e, batch, ctx):
    I, J = e.pairs(batch)
    Bn = len(batch)
    m = np.bincount(I, minlength=e.N).astype(float)
    cx = ctx[J] & econ.MASK[e.k[I]]
    b = np.clip(e.beliefs(I, cx), e.eps, 1 - e.eps)
    st = e.stake * e.w[I] / m[I]
    S = np.bincount(J, st, Bn) + e.sub
    return (np.bincount(J, st * b, Bn) + 0.5 * e.sub) / S


class Tracker:
    def __init__(self):
        self.acc = np.zeros((2, 3))        # [tracked?][n, brier, brier_oracle]
        self.uid_at_shift = None
        self.snap = {}

    def setup(self, main, nul):
        orig = main.step

        def step(batch, ctx, y, ptrue, fam):
            p = price(main, batch, ctx)
            st = _ST['s']
            tr = st.track[batch].astype(int)
            yf = y.astype(float)
            np.add.at(self.acc[:, 0], tr, 1)
            np.add.at(self.acc[:, 1], tr, (p - yf) ** 2)
            np.add.at(self.acc[:, 2], tr, (ptrue - yf) ** 2)
            if st.t == SHIFT_AT and self.uid_at_shift is None:
                self.uid_at_shift = main.next_uid
            orig(batch, ctx, y, ptrue, fam)
        main.step = step

    def extra(self, main, nul):
        st = _ST['s']
        a = self.acc
        n = np.maximum(a[:, 0], 1)
        br, bo = a[:, 1] / n, a[:, 2] / n
        row = {}
        for i, nm in ((1, 'trk'), (0, 'untrk')):
            row[f'brier_{nm}'] = br[i] if a[i, 0] else float('nan')
            row[f'regret_{nm}'] = br[i] - bo[i] if a[i, 0] else float('nan')
        row['excess_regret_trk'] = row['regret_trk'] - row['regret_untrk']
        row['brier_check_sum'] = a[:, 1].sum() / max(a[:, 0].sum(), 1)   # must equal engine brier_all
        self.acc[:] = 0
        al = main.alive
        w = main.w
        row.update(wshare_trk=float('nan'), alive_trk=float('nan'), newborn_share_trk=float('nan'))
        if main.niches and st.track.any():
            inq = al & st.track[np.maximum(main.niche, 0)]
            Wt = w[al].sum()
            row['wshare_trk'] = w[inq].sum() / Wt if Wt > 0 else float('nan')
            row['alive_trk'] = int(inq.sum())
            if self.uid_at_shift is not None:
                new = inq & (main.uid >= self.uid_at_shift)
                row['newborn_share_trk'] = w[new].sum() / max(w[inq].sum(), 1e-300)
        t1 = st.t + 1
        if t1 in (SHIFT_AT - 100, T):
            cw = np.bincount(main.clade[al], w[al], main.N)
            self.snap[t1] = cw.tolist()
        return row


def main(arm, seed):
    cfg = ARMS[arm]
    c = cfg['cr'] * CSTAR[cfg['niches']]
    tr = Tracker()
    rid = f'w2-{arm}-s{seed}'
    out_root = os.path.join(K1, 'runs')
    TT = T if cfg['shift'] or cfg['cr'] == 0.0025 else 10000
    s = econ.run(N=cfg['N'], Q=64, niches=cfg['niches'], seed=seed, T=TT, batch=B, out_root=out_root, run_id=rid,
                 shift_at=SHIFT_AT if cfg['shift'] else None, shift_frac=0.25, c=c, subsidy=SUB,
                 setup=tr.setup, extra_metrics=tr.extra, verbose=True)
    a, b = tr.snap.get(SHIFT_AT - 100), tr.snap.get(T)
    lock = econ.spearman(np.array(a), np.array(b)) if a and b else float('nan')
    ext = dict(T=TT, arm=arm, seed=seed, c=c, c_over_cstar=cfg['cr'], lockin_spearman_clade=lock,
               clades_pos_pre=int(np.sum(np.array(a) > 0)) if a else None,
               uid_at_shift=tr.uid_at_shift)
    with open(os.path.join(out_root, rid, 'w2extra.json'), 'w') as f:
        json.dump(ext, f, indent=1)
    print(rid, 'done', json.dumps({k: s['final'][k] for k in ('alive', 'gini_traders', 'gini_null_traders', 'top_clade_share', 'brier_all')}))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
