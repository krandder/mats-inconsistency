"""Mean-based dedup metrics (separating dedup from variance calibration) for every arm, and mechanism probes
for seeds 0-2 with matched untrained-network baselines.

usage: python analysis2.py RUNS_DIR OUT_JSON
"""
import glob, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def arm_metrics(d):
    import torch, world as W
    from pfn import PFN, DEV, fixed_set, predict, splits, TEST_SEED, mechanism
    meta = json.load(open(os.path.join(d, 'meta.json')))
    keys = meta['keys']
    ds, X, m = fixed_set(TEST_SEED, 8000, keys)
    out, rn, (md, vd), w = W.references(ds)
    bmean = out['c'][0] if keys else md
    res = {'name': os.path.basename(d), 'depth': meta['depth'], 'keys': keys, 'seed': meta['seed'],
           'converged': meta['converged'], 'incomplete': meta.get('incomplete', False), 'best_step': meta['best_step']}
    for trained in (True, False):
        torch.manual_seed(meta['seed'])
        model = PFN(meta['fin'], meta['depth']).to(DEV)
        if trained:
            model.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location=DEV))
        mu, ls = predict(model, X, m)
        tag = 'trained' if trained else 'untrained'
        sp = {}
        for name, sel in splits(ds).items():
            if sel.sum() == 0: continue
            e = lambda a: float(((a - bmean)[sel] ** 2).mean())
            sp[name] = {'mse_model': e(mu), 'mse_a': e(out['a'][0]), 'mse_b': e(out['b'][0]),
                        'mse_c': e(out['c'][0]), 'mse_d': e(md),
                        'model_std_mean': float(np.exp(ls[sel]).mean()), 'bayes_std_mean': float(np.sqrt(vd[sel]).mean() if not keys else np.sqrt(out['c'][1][sel]).mean())}
        res[tag] = {'splits': sp}
        if meta['depth'] <= 4 or meta.get('incomplete'):
            mm = mechanism(model, meta, keys)
            res[tag]['mech'] = {k: mm[k] for k in ('probe_groupsize', 'probe_query_bayesmean', 'ablation')}
            res[tag]['mech']['attn_corr_gls'] = [q['corr_with_gls_weight_nodup'] for q in mm['query_attn']]
            res[tag]['mech']['attn_by_m'] = [q['mass_by_groupsize'] for q in mm['query_attn']]
    return res


if __name__ == '__main__':
    root, outp = sys.argv[1], sys.argv[2]
    dirs = sorted(x for x in glob.glob(os.path.join(root, 'L*_k*_s*')) if os.path.isdir(x))
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as ex:
        allr = list(ex.map(arm_metrics, dirs))
    json.dump(allr, open(outp, 'w'))
    print('done', len(allr))
