"""E15 data (PREREG-B amendment 22): the E12 matched pairs in the ds-4 token format, and the list of clean nets.
The pairs' readings and labels are E12's (results/e12_pairs.npz); identity vectors are redrawn in R^4 (N(0, 1)/4, as
W.tokens_np) from default_rng(20_261_101), one set per base episode, shared by both versions of a pair.
usage: python e15_data.py -> results/e15_pairs_ds4.npz, results/e15_nets.json
"""
import glob, json, os
import numpy as np
import nnoise  # noqa: F401  (puts tiny-pfn on the path)
import world as W

TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def pairs_ds4():
    p = 'results/e15_pairs_ds4.npz'
    assert not os.path.exists(p)
    z = np.load('results/e12_pairs.npz'); base = z['base']
    rng = np.random.default_rng(20_261_101); n = 3000
    semb = rng.standard_normal((n, W.S, 4)) / 4; qemb = rng.standard_normal((n, W.Q, 4)) / 4
    out = {}
    for k in (0, 1):
        X, valid = W.assemble(semb[base], qemb[base], z[f'd{k}_o'], z[f'd{k}_y'])
        out[f'X{k}'] = X; out[f'valid{k}'] = valid
        # same readings and labels as the ds-16 version
        assert np.allclose(X[:, :, 8:11], z[f'X{k}'][:, :, 32:35])
    np.savez(p, **out)


def nets():
    rows = []
    for d in sorted(glob.glob(os.path.join(TP, 'runs', 'w*_s[0-9]'))):
        m = json.load(open(os.path.join(d, 'meta.json'))); e = json.load(open(os.path.join(d, 'eval.json')))
        rows.append(dict(run=d, cfg=m['cfg'], seed=m['seed'], depth=m['depth'], ds=m['ds'], width=m['width'], heads=m['heads'],
                         mixk=m.get('mixk', 0), attn_only=m['attn_only'], long=bool(m.get('phase1_fixed', 0)),
                         ws=bool(m.get('init_from')), gap_eval_best=e['best']['gap'], gap_eval_phase1=e['phase1']['gap']))
    for s in (5, 6, 7):
        d = os.path.abspath(f'runs/base_mix4_s{s}'); m = json.load(open(os.path.join(d, 'meta.json')))
        rows.append(dict(run=d, cfg=m['cfg'], seed=m['seed'], depth=m['depth'], ds=m['ds'], width=m['width'], heads=m['heads'],
                         mixk=m['mixk'], attn_only=m['attn_only'], long=False, ws=False, gap_eval_best=None, gap_eval_phase1=None))
    for r in rows:
        r['tag'] = f"{r['cfg']}_s{r['seed']}" + ('_nb' if '/cm-interp-B/' in r['run'] else '')
    json.dump(rows, open('results/e15_nets.json', 'w'), indent=1)
    print(len(rows), 'nets')


if __name__ == '__main__':
    pairs_ds4(); nets()
