"""E47 post-review checks (not registered; asked for by the Codex review, results/review_e47.md): per-step lineage counts and the
AUCs on two sub-cohorts, (a) E37 and E43 lineages only (no hybrids or perturbed copies), (b) E44 hybrids only, each with the same
cluster bootstrap as e47_summ.py. usage: python e47_post.py -> results/e47_post.json"""
import json
import numpy as np

import e47_summ as S


def main():
    rng = np.random.default_rng(470)
    import glob, os
    L = []
    for f in sorted(glob.glob('results/e47/*.npz')):
        tag = os.path.basename(f)[:-4]; fp, cl = S.final_and_cluster(tag)
        if fp is None or not os.path.exists(fp) or float(np.load(fp)['gap']) >= 0.10:
            continue
        z = np.load(f); arm = tag.split('_')[1]
        kb = (0 if arm == 'br0' else int(arm[2:-1]) * 1000) if tag.startswith('e43_br') else -1
        m = {int(st): dict(lab=float(z['lab'][i]), xq=float(z['xq'][i]), kl=float(z['kl'][i]),
                           sign_share=float(z['sign_pc'][i] / z['sign_kl'][i]) if z['sign_kl'][i] > 0 else np.nan)
             for i, st in enumerate(z['step']) if int(st) > kb}
        L.append(dict(tag=tag, exp=tag[:3], cluster=cl, B=bool(float(np.load(fp)['rom_middle']) > 0), m=m))
    out = dict(counts={st: dict(n=sum(st in x['m'] for x in L), n_B=sum(st in x['m'] and x['B'] for x in L)) for st in S.STEPS}, cohorts={})
    for name, keep in (('e37_e43_only', ('e37', 'e43')), ('e44_only', ('e44',)), ('e46_only', ('e46',))):
        sub = [x for x in L if x['exp'] in keep]; cls = sorted({x['cluster'] for x in sub}); res = {}
        for st in S.STEPS:
            for mk in S.MARKERS:
                def stat(cs):
                    sel = [x for c in cs for x in sub if x['cluster'] == c and st in x['m'] and np.isfinite(x['m'][st][mk])]
                    return S.auc([x['m'][st][mk] for x in sel if x['B']], [x['m'][st][mk] for x in sel if not x['B']])
                bs = [b for b in (stat(list(rng.choice(cls, len(cls)))) for _ in range(1000)) if np.isfinite(b)]
                res['%s@%d' % (mk, st)] = [stat(cls), float(np.percentile(bs, 2.5)) if bs else np.nan, float(np.percentile(bs, 97.5)) if bs else np.nan]
        out['cohorts'][name] = dict(n=len(sub), n_B=sum(x['B'] for x in sub), auc=res)
        print(name, len(sub), {k: round(v[0], 2) for k, v in res.items() if k.split('@')[1] in ('3000', '5000', '10000')})
    print(out['counts'])
    json.dump(out, open('results/e47_post.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
