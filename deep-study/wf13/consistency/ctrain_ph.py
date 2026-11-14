"""ctrain_ph.py (POST HOC, not in PREREG) : ctrain.py with the REAL OUTCOMES restricted to one family of queries.
Same arguments as ctrain.py. Environment LABMIN = m: real outcomes are used (training and validation) only on queries
whose context reveals at least m of the 30 checks. The consistency terms and the test sets are unchanged, so the test
asks whether a consistency term carries what real data teach on large contexts to the contexts real data never score.
"""
import os
import sys

sys.dont_write_bytecode = True
LABMIN = int(os.environ.get('LABMIN', 0))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ctrain.py')).read()
src = src.replace("if __name__ == '__main__':", "def main():")
ns = {'__name__': 'ctrain_ph_body'}
exec(compile(src, 'ctrain.py', 'exec'), ns)
_orig = ns['subset']


def subset(lab, n, d):
    tr, va, c = _orig(lab, n, d)
    if not LABMIN:
        return tr, va, c
    big = lab.size * 30 >= LABMIN - 0.5
    return tr[big[tr]], va[big[va]], c


ns['subset'] = subset

# ---- post hoc arms on the extra triples c1b (8 trajectories per context), in the directory given by C1B:
#   ib   the registered pair term (i), computed on c1b with one draw of K trajectories
#   idb  the NOISE-CORRECTED pair term: two DISJOINT draws A and B of K trajectories per context; the loss is the
#        product (log J1 - log J2)_A x (log J1 - log J2)_B. Its expectation is the square of the EXPECTED mismatch
#        between the two orders; trajectory-sampling noise, which is independent between A and B, drops out.
_C1B = {}


def _c1b(dev):
    if not _C1B:
        import glob
        CM = ns['CM']
        for f in sorted(glob.glob(os.environ['C1B'] + '/c1b_*.pt')):
            D = CM.CS(f, dev)
            D.meta = {a: (b.to(dev).long() if ns['torch'].is_tensor(b) else b) for a, b in D.meta.items()}
            _C1B[f] = D
    return list(_C1B.values())


def cons_pair_ph(model, g, Bc, dev, debias):
    torch, CM, LEVELS = ns['torch'], ns['CM'], ns['LEVELS']
    sets = _c1b(dev)
    C1 = sets[int(torch.randint(0, len(sets), (1,), generator=g, device=dev))]
    n = C1.N // 5
    K = LEVELS[int(torch.randint(0, len(LEVELS), (1,), generator=g, device=dev))]
    t = torch.randint(0, n, (Bc,), generator=g, device=dev)
    X, Y = C1.meta['X'][t], C1.meta['Y'][t]
    idx = torch.cat([t, t, n + t, 2 * n + t, 3 * n + t, 4 * n + t])
    tgt = torch.cat([X, Y, Y, Y, X, X])
    perm = torch.rand(6 * Bc, C1.K, generator=g, device=dev).argsort(1)
    l1a, l2a = ns['pair_joints'](CM.forecast(model, C1, idx, tgt, perm[:, :K]).view(6, Bc).T)
    if not debias:
        w = 0.5 * (l1a.exp() + l2a.exp()).detach()
        return (w * (l1a - l2a) ** 2).sum((1, 2)).mean()
    l1b, l2b = ns['pair_joints'](CM.forecast(model, C1, idx, tgt, perm[:, K:2 * K]).view(6, Bc).T)
    w = 0.25 * (l1a.exp() + l2a.exp() + l1b.exp() + l2b.exp()).detach()
    return (w * (l1a - l2a) * (l1b - l2b)).sum((1, 2)).mean()


_cons_i = ns['cons_i']


def cons_i_dispatch(model, C1, g, Bc, dev):
    arm = ns.get('_PH_ARM')
    if arm in ('ib', 'idb'):
        return cons_pair_ph(model, g, Bc, dev, arm == 'idb')
    return _cons_i(model, C1, g, Bc, dev)


_train = ns['train']


def train(arm, lam, n, d, r, lab, C1, C2, dev, steps, log):
    ns['_PH_ARM'] = arm
    return _train('i' if arm in ('ib', 'idb') else arm, lam, n, d, r, lab, C1, C2, dev, steps, log)


ns['cons_i'] = cons_i_dispatch
ns['train'] = train
ns['main']()
