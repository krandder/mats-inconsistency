"""E20 analysis (PREREG-B amendment 27): the departure of the minimal models on matched pairs.
Pairs: 4,000 fresh single-source episodes (default_rng(20_261_107)) with SS in E12's large (>= 3.182394) or middle (0.679992-3.182394)
range; big miss (kappa 0.85) and spread (kappa 0.25) versions of the four errors with the same SS, signs and position; the labels
and identity vectors are shared by both versions.
Measure: predicted posterior precision P = sum_k p_k / v_k; departure = sum (P_big - P_spread) / sum P_Bayes over the pairs of a
range (Bayes is identical within a pair, so Bayes's departure is 0). Positive: the big miss looks more reliable.
usage: python e20_summ.py -> results/e20.json, results/e20_tables.md
"""
import glob, json, os
import numpy as np
import torch
import minimal as M

LO, HI = 0.679992, 3.182394


def pairs():
    rng = np.random.default_rng(20_261_107); k, y, r, sid, qid = M.sample(4000, rng)
    ss = (r ** 2).sum(1); keep = ss >= LO; y, r, sid, qid, ss = y[keep], r[keep], sid[keep], qid[keep], ss[keep]
    n = len(ss); sg = rng.choice([-1.0, 1.0], (n, 4)); pos = rng.integers(0, 4, n)
    out = {}
    for name, kap in (('big', 0.85), ('spread', 0.25)):
        rr = np.repeat(np.sqrt((1 - kap) / 3 * ss)[:, None], 4, 1); rr[np.arange(n), pos] = np.sqrt(kap * ss); out[name] = rr * sg
    return y, sid, qid, out, np.where(ss >= HI, 'large', 'middle'), M.bayes_post(out['spread'])


def main():
    y, sid, qid, R, rg, pb = pairs()
    Pb = (pb / M.VLEV[None]).sum(1)
    res = {}
    for f in sorted(glob.glob('results/e20/*.json')):
        meta = json.load(open(f)); net = M.Net(meta['kind'], sigma=0.0)          # evaluate without noise draws: sigma enters below
        net.load_state_dict(torch.load(f[:-5] + '.pt')); net.eval(); net.sigma = meta['sigma']
        torch.manual_seed(0)
        P = {}
        with torch.no_grad():
            for v in ('big', 'spread'):
                lg = torch.as_tensor(np.log((R[v] ** 2).sum(1)), dtype=torch.float32)
                ps = 0
                reps = 16 if meta['sigma'] > 0 else 1
                for _ in range(reps):                                                # noise-averaged posterior
                    ps = ps + torch.softmax(net(M.tokens(y, R[v], sid, qid), lg), -1).numpy() / reps
                P[v] = (ps / M.VLEV[None]).sum(1)
        tag = os.path.basename(f)[:-5]
        res[tag] = dict(kind=meta['kind'], seed=meta['seed'], sigma=meta['sigma'], val_gap=meta['val_gap'],
                        **{r: float((P['big'] - P['spread'])[rg == r].sum() / Pb[rg == r].sum()) for r in ('large', 'middle')})
    groups = {}
    for t, v in res.items():
        groups.setdefault((v['kind'], v['sigma']), []).append(v)
    summ = {f'{k[0]}|{k[1]}': dict(n=len(v), large=float(np.mean([x['large'] for x in v])), large_min=float(min(x['large'] for x in v)),
                                   large_max=float(max(x['large'] for x in v)), middle=float(np.mean([x['middle'] for x in v])),
                                   val_gap=float(np.mean([x['val_gap'] for x in v]))) for k, v in sorted(groups.items())}
    out = dict(per_run=res, groups=summ, predictions={})
    g = lambda kind, s: summ.get(f'{kind}|{s}')
    if g('ln', 0.0):
        lnr = [x for x in groups[('ln', 0.0)]]
        out['predictions']['P120'] = dict(large=g('ln', 0.0)['large'], middle=g('ln', 0.0)['middle'],
                                          holds=bool(all(x['large'] > 0 and x['middle'] > 0 for x in lnr)))
    if g('ln', 0.0) and g('pre', 0.0):
        out['predictions']['P121'] = dict(pre=g('pre', 0.0)['large'], ln=g('ln', 0.0)['large'],
                                          holds=bool(abs(g('pre', 0.0)['large']) < 0.5 * abs(g('ln', 0.0)['large'])))
    if g('ln', 0.0) and g('ln', 0.1):
        out['predictions']['P122'] = dict(ln0=g('ln', 0.0)['large'], ln01=g('ln', 0.1)['large'], holds=bool(g('ln', 0.1)['large'] > g('ln', 0.0)['large']))
    if g('lnmean', 0.0):
        lm = groups[('lnmean', 0.0)]
        out['predictions']['P123'] = dict(large=g('lnmean', 0.0)['large'], middle=g('lnmean', 0.0)['middle'],
                                          ln_attention_middle=g('ln', 0.0)['middle'] if g('ln', 0.0) else None,
                                          holds=bool(all(x['large'] > 0 and x['middle'] > 0 for x in lm)
                                                     and g('lnmean', 0.0)['middle'] > g('ln', 0.0)['middle']))
    if g('ss', 0.0):
        out['predictions']['check_ss'] = dict(large=g('ss', 0.0)['large'], middle=g('ss', 0.0)['middle'])
    json.dump(out, open('results/e20.json', 'w'), indent=1)
    for k, v in out['predictions'].items():
        print(k, v)
    T = ['| model | sigma | runs | val gap (nats) | departure large [min, max] | departure middle |', '|---|---|---|---|---|---|']
    for k, v in summ.items():
        kind, s = k.split('|')
        T.append(f"| {kind} | {s} | {v['n']} | {v['val_gap']:.4f} | {v['large']:+.3f} [{v['large_min']:+.3f}, {v['large_max']:+.3f}] | {v['middle']:+.3f} |")
    open('results/e20_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
