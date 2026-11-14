"""Analysis for P16 (pre-registered in PREREG.md).

  python analyze.py tune     -> <OUT>/tuned.json   (tune split: AUROC at k=16, ID positives vs ID negatives)
  python analyze.py gate     -> <OUT>/gate.json    (pilot split: Stage 0.5 gate (i), (ii))
  python analyze.py final    -> results/summary.json, charts/*.json
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr

OUT = os.environ.get('P16_OUT', 'results')
KS = (4, 8, 16, 32, 64, 128)
SK = (4, 8, 16, 32, 64)
K16 = KS.index(16)
ID = ['T0', 'T1', 'T2', 'T3a', 'T3b', 'T4', 'T5', 'T6']
HP = ['HP1', 'HP2', 'HP3a', 'HP3b']
HN = ['HN1', 'HN2', 'HN3', 'HN4']
LEAK = ['Ly', 'Lpar', 'LinS', 'Lw4']
NEG_TYPES = ['T0', 'T3a', 'T1'] + HN


def design(split):
    return {d['idx']: d for d in json.load(open(f'{OUT}/eval/design_{split}.json'))['items']}


def label(d):
    """1 useful, 0 useless, None excluded."""
    if d['cmi'] >= .05:
        return 1
    if d['cmi'] < 1e-12 and d['type'] != 'T3b':
        return 0
    return None


def auroc(pos, neg):
    pos = np.nan_to_num(np.asarray(pos, float)); neg = np.nan_to_num(np.asarray(neg, float))
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def slice_masks(D, idx):
    """boolean (pos, neg) masks for each slice over the episode order idx."""
    typ = np.array([D[i]['type'] for i in idx]); lab = np.array([(-1 if label(D[i]) is None else label(D[i])) for i in idx])
    isin = lambda ts: np.isin(typ, ts)
    x_id = isin(ID) & (typ != 'T3b')
    S = {
        'ID': (x_id & (lab == 1), x_id & (lab == 0)),
        'primary': (isin(HP) & (lab == 1), isin(HN) & (lab == 0)),
        'cross_IDpos_HNneg': (x_id & (lab == 1), isin(HN) & (lab == 0)),
        'cross_HPpos_IDneg': (isin(HP) & (lab == 1), x_id & (lab == 0)),
        'leak': (isin(LEAK) & (lab == 1), (x_id | isin(HN)) & (lab == 0)),
    }
    return S, typ, lab


def load(split, name):
    f = f'{OUT}/eval/{split}/{name}.npz'
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def aligned(split, name, idx, key='preq'):
    z = load(split, name)
    if z is None or key not in z:
        return None
    pos = {int(i): r for r, i in enumerate(z['idx'])}
    out = np.full((len(idx),) + z[key].shape[1:], np.nan)
    for r, i in enumerate(idx):
        if int(i) in pos:
            out[r] = z[key][pos[int(i)]]
    return out


# ---------------------------------------------------------------------------------------------------------- tune
def tune():
    D = design('tune')
    tuned = {}
    rep = {}
    def au(sc, idx):
        S, _, _ = slice_masks(D, idx)
        p, n = S['ID']
        return auroc(sc[p], sc[n])
    best = None
    for f in sorted(glob.glob(f'{OUT}/eval/tune/C4@*.npz')):
        z = np.load(f); idx = z['idx']
        prm = json.loads(str(z['params']))
        for lam in (0, .5, 1):
            a = au(z['x_fwd'][:, K16] + lam * z['x_rev'][:, K16], idx)
            rep[f'C4 alpha={prm["alpha"]} rho={prm["rho"]} lam={lam}'] = a
            if best is None or a > best[0]:
                best = (a, dict(alpha=prm['alpha'], rho=prm['rho'], lam=lam))
    if best:
        tuned['C4'] = best[1]; tuned['_C4_auroc'] = best[0]
    for arm in ('C5', 'S1', 'S2'):
        best = None
        for f in sorted(glob.glob(f'{OUT}/eval/tune/{arm}@*.npz')):
            z = np.load(f)
            a = au(z['preq'][:, K16], z['idx'])
            prm = json.loads(str(z['params'])); prm.pop('tag', None)
            rep[f'{arm} {prm}'] = a
            if best is None or a > best[0]:
                best = (a, prm)
        if best:
            tuned[arm] = best[1]; tuned[f'_{arm}_auroc'] = best[0]
    for f in sorted(glob.glob(f'{OUT}/eval/tune/FR-direct__*.npz')):
        z = np.load(f)
        a_p, a_l = au(z['preq'][:, K16], z['idx']), au(z['x_logit'][:, K16], z['idx'])
        rep[os.path.basename(f) + ' pred/logit'] = (a_p, a_l)
        tuned.setdefault('FRD_votes', []).append('logit' if a_l > a_p else 'pred')
    for k, v in rep.items():
        print(f'{k:60s} {v}')
    old = json.load(open(f'{OUT}/tuned.json')) if os.path.exists(f'{OUT}/tuned.json') else {}
    old.update(tuned)
    json.dump(old, open(f'{OUT}/tuned.json', 'w'), indent=1)
    print('tuned', json.dumps(old))


# ---------------------------------------------------------------------------------------------------------- gate
def gate():
    D = design('pilot')
    idx = np.array(sorted(D))
    S, typ, lab = slice_masks(D, idx)
    p, n = S['primary']
    sc = {}
    for arm, key in (('C4', 'preq'), ('C5', 'preq'), ('L-all', 'preq'), ('L-train', 'x_ltrain'), ('E1', 'preq')):
        name = 'L' if arm.startswith('L') else arm
        a = aligned('pilot', name, idx, key)
        for fn in sorted(glob.glob(f'{OUT}/eval/pilot/{name}@*.npz')):         # per-type files (tag = type)
            b = aligned('pilot', os.path.basename(fn)[:-4], idx, key)
            a = b if a is None else np.where(np.isnan(a), b, a)
        if a is not None:
            sc[arm] = a[:, K16] if a.ndim == 2 else a
    have = {a: ~np.isnan(v) for a, v in sc.items()}
    res = {'n_pos': int(p.sum()), 'n_neg': int(n.sum())}
    full = {a: auroc(v[p & have[a]], v[n & have[a]]) for a, v in sc.items() if a != 'E1'}
    res['auroc_full'] = full
    res['gate_i_full'] = full['L-all'] - max(full['C4'], full['C5'])
    if 'E1' in sc:
        e = have['E1']
        sub = {a: auroc(v[p & e], v[n & e]) for a, v in sc.items()}
        res['auroc_E1_subsample'] = sub
        res['n_E1_subsample'] = [int((p & e).sum()), int((n & e).sum())]
        res['gate_i_sub'] = sub['L-all'] - max(sub['E1'], sub['C4'], sub['C5'])
        zs = [np.load(fn) for fn in glob.glob(f'{OUT}/eval/pilot/E1*.npz')]
        cs, cn = sum(float(z['x_calls_s'].max()) for z in zs), sum(int(z['x_calls_n'].max()) for z in zs)
        res['e1_call_s'] = cs / max(1, cn); res['e1_calls'] = cn
        res['e1_wall_s'] = sum(float(z['wall_s']) for z in zs)
    nov = {}
    for t in HP + HN:
        if t in HP:
            pp, nn = p & (typ == t), n
        else:
            pp, nn = p, n & (typ == t)
        nov[t] = auroc(sc['L-all'][pp], sc['L-all'][nn]) - auroc(sc['L-train'][pp], sc['L-train'][nn])
    res['novelty'] = nov
    res['mean_novelty'] = float(np.mean(list(nov.values())))
    gi = res['gate_i_full'] >= .05 and res.get('gate_i_sub', 1.0) >= .05
    res['gate_i_pass'] = bool(gi); res['gate_ii_pass'] = bool(res['mean_novelty'] >= .05)
    json.dump(res, open(f'{OUT}/gate.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'tune':
        tune()
    elif cmd == 'gate':
        gate()
    elif cmd == 'final':
        import analyze_final
        analyze_final.main()
