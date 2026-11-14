"""In-distribution copy test (after review 2): weight ratio rho and cue-removal variants.

Episodes: 6 singleton sources, no duplicates in the base draw (seed 78000, disjoint from the reviewer's 77000), plus one
extra row. rho = (d mu / d o* of source s0, all its rows moved together) / (d mu / d o* of another source s1), central
differences. A reader that cannot tell which rows are copies has rho = 2; full deduplication has rho = 1.
Variants of the extra row (only 'copy' occurs in training; the others remove one cue and are out of distribution):
  copy        exact copy of s0 (identity, group key, all 9 values)
  qtie_off    copy, but its query reading is o_s0 + delta (delta ~ N(0, 2*SE2), fixed): no exact query tie
  lab_off     copy, but its 8 labelled readings are fresh (y + z_new + e): no labelled-value equality
  id_off      copy, but new identity vector and new group key: no identity/key match
plus 'indep': an untied independent source (fresh on all 9 questions, new identity and key); pull on its own reading.
usage: PYTHONPATH=... python split_indist.py RUNS_DIR OUT_JSON [n] [procs]
"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

H = 0.05
SEED = 78000
# 'copy3' (s0 with 3 extra copies; rho = 4 means no dedup) is added with SPLIT_VARIANTS=all (Track A2)
VARIANTS = ('copy', 'qtie_off', 'lab_off', 'id_off') + (('copy3',) if os.environ.get('SPLIT_VARIANTS') == 'all' else ())


def episodes(n):
    import world as W
    rng = np.random.default_rng(SEED)
    d = W.sample_np(n * 8, rng)
    sing = np.array([len(set(l)) == W.S for l in d['lab']])
    keep = np.where((d['k'] == 0) & sing)[0][:n]
    d = {k: v[keep] for k, v in d.items()}
    ar = np.arange(n)
    E = dict(d=d, ar=ar)
    E['semb'] = rng.standard_normal((n, W.S, W.DS)) / 4
    E['qemb'] = rng.standard_normal((n, W.Q, W.DS)) / 4
    E['gemb'] = rng.standard_normal((n, W.S, W.DS)) / 4
    E['s0'] = rng.integers(0, W.S, n); E['s1'] = (E['s0'] + rng.integers(1, W.S, n)) % W.S
    newv = d['y'][:, None, :] + rng.standard_normal((n, 1, W.Q)) * np.sqrt(W.SG2) \
        + rng.standard_normal((n, 1, W.Q)) * np.sqrt(W.SE2)
    E['newv'] = newv
    E['delta'] = rng.standard_normal(n) * np.sqrt(2 * W.SE2)
    E['news'] = rng.standard_normal((n, 1, W.DS)) / 4
    E['newg'] = rng.standard_normal((n, 1, W.DS)) / 4
    return E


def build(E, variant, o):
    """Arrays for assemble() with the extra row as row index 6 (a 7th 'source' slot) unless it is an exact copy."""
    import world as W
    d, ar, s0 = E['d'], E['ar'], E['s0']
    n = len(ar); src6 = np.tile(np.arange(W.S), (n, 1))
    semb, gemb, lab = E['semb'], E['gemb'], d['lab']
    if variant == 'base':
        return semb, gemb, src6, o, lab
    if variant == 'copy':
        return semb, gemb, np.concatenate([src6, s0[:, None]], 1), o, lab
    if variant == 'copy3':                                       # three extra copies (4 rows of s0)
        return semb, gemb, np.concatenate([src6, np.repeat(s0[:, None], 3, 1)], 1), o, lab
    src7 = np.concatenate([src6, np.full((n, 1), W.S)], 1)
    row = o[ar, s0][:, None, :].copy()                           # follows o (so shifts of s0 move it too)
    se7 = np.concatenate([semb, semb[ar, s0][:, None]], 1)
    ge7 = np.concatenate([gemb, E['newg']], 1)
    lab7 = np.concatenate([lab, lab[ar, s0][:, None]], 1)      # same group label as s0 -> same group key
    if variant == 'qtie_off':
        row[:, 0, W.QL] += E['delta']
    elif variant == 'lab_off':
        row[:, 0, :W.QL] = E['newv'][:, 0, :W.QL]
    elif variant == 'id_off':
        se7 = np.concatenate([semb, E['news']], 1)
        lab7 = np.concatenate([lab, np.full((n, 1), W.S)], 1)  # new group key (gemb row 6)
    elif variant == 'indep':
        row = E['newv'].copy()
        se7 = np.concatenate([semb, E['news']], 1)
        lab7 = np.concatenate([lab, np.full((n, 1), W.S)], 1)
    return se7, ge7, src7, np.concatenate([o, row], 1), lab7


def mu(model, keys, E, variant, o):
    import torch, world as W, mech2
    se, ge, src, oo, lab = build(E, variant, o)
    X, vm = W.assemble(se, E['qemb'], ge, src, np.ones_like(src, bool), oo, E['d']['y'], lab, keys)
    with torch.no_grad():
        return mech2.run(model, torch.tensor(X), torch.tensor(vm))[0].numpy().astype(np.float64)


def shift(E, s, h):
    import world as W
    o = E['d']['o'].copy(); o[E['ar'], s, W.QL] += h
    return o


def per_episode(f, E, variant):
    """(w0, w1) per episode: d mu / d o* for s0 (all its rows) and s1."""
    w0 = (f(E, variant, shift(E, E['s0'], H)) - f(E, variant, shift(E, E['s0'], -H))) / (2 * H)
    w1 = (f(E, variant, shift(E, E['s1'], H)) - f(E, variant, shift(E, E['s1'], -H))) / (2 * H)
    return w0, w1


def one(args):
    path, n = args
    import torch, world as W, mech2
    from pfn import PFN
    torch.set_num_threads(1)
    meta = json.load(open(os.path.join(path, 'meta.json'))); keys = meta['keys']
    torch.manual_seed(meta['seed'])
    model = PFN(meta['fin'], meta['depth'])
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location='cpu')); model.eval()
    E = episodes(n)
    f = lambda E_, v, o: mu(model, keys, E_, v, o)
    r = {'depth': meta['depth'], 'keys': keys, 'seed': meta['seed'], 'conv': meta['converged'],
         'best_step': meta['best_step'], 'w': {}}
    for v in VARIANTS:
        w0, w1 = per_episode(f, E, v)
        r['w'][v] = [w0.tolist(), w1.tolist()]
    o = E['d']['o']
    mu6 = f(E, 'base', o); mui = f(E, 'indep', o)
    onew = E['newv'][:, 0, W.QL]
    r['pull_indep'] = mech2.slope(mui - mu6, onew - mu6)
    mc = f(E, 'copy', o); odup = o[E['ar'], E['s0'], W.QL]
    r['pull_copy'] = mech2.slope(mc - mu6, odup - mu6)
    print(path, {v: round(np.mean(r['w'][v][0]) / np.mean(r['w'][v][1]), 3) for v in r['w']},
          round(r['pull_copy'], 3), round(r['pull_indep'], 3), flush=True)
    return r


def refs(n):
    import world as W, split
    E = episodes(n)
    coef = split.fit_r1()

    def q(E_, v, o):
        se, ge, src, oo, lab = build(E_, v, o)
        return np.take_along_axis(oo[:, :, W.QL], src, 1)
    r1 = lambda E_, v, o: (lambda Q: split.basis(Q).sum(1) @ coef[Q.shape[1]][:-1] + coef[Q.shape[1]][-1])(q(E_, v, o))
    r2 = lambda E_, v, o: split.r2_mean(q(E_, v, o))
    out = {}
    for name, f in (('R1', r1), ('R2', r2)):
        out[name] = {}
        for v in VARIANTS:
            w0, w1 = per_episode(f, E, v)
            out[name][v] = float(w0.mean() / w1.mean())
    return out


def main(runs, outp, n=600, procs=4):
    dirs = sorted(x for x in glob.glob(os.path.join(runs, 'L*_k*_s*')) if os.path.isdir(x))
    with ProcessPoolExecutor(procs) as ex:
        fut = ex.submit(refs, n)
        models = list(ex.map(one, [(p, n) for p in dirs]))
        R = fut.result()
    print('refs', R)
    json.dump({'n': n, 'refs': R, 'models': models}, open(outp, 'w'))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 600, int(a[4]) if len(a) > 4 else 4)
