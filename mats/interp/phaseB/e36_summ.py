"""E36 analysis (PREREG-B amendment 44). Pooled projection shares of the single-source negation shift per route and group:
share_r = sum proj_r / sum full2; also the ratio of each hybrid's squared shift to the full one (off-axis size).
  P164  B bases: share(summ_own) >= 0.3.
  P165  A bases: share(summ_own) > share(lab2).
usage: python e36_summ.py -> results/e36.json, results/e36_tables.md
"""
import json, os
import numpy as np
import e26_summ as SM

R = ['summ_own', 'summ_all', 'query1', 'lab2']


def main():
    rows = json.load(open('results/e26_nets.json'))
    nets = {}
    for s in ('M', 'G'):
        for r in rows:
            if r['set'] == s:
                f = os.path.join('results/e36', s, r['tag'] + '.npz')
                if os.path.exists(f):
                    nets[r['tag']] = (s, {k: float(v) for k, v in np.load(f).items()})
    groups = dict(A=SM.A_BASES, B=SM.B_BASES, M=[t for t, (s, _) in nets.items() if s == 'M'],
                  G=[t for t, (s, _) in nets.items() if s == 'G'])
    res = dict(groups={}, per_net={t: dict(set=s, **{r: o['proj_' + r] / o['full2'] for r in R}) for t, (s, o) in nets.items()})
    for g, tags in groups.items():
        L = [nets[t][1] for t in tags if t in nets]
        if not L:
            continue
        F = sum(x['full2'] for x in L)
        res['groups'][g] = dict(n=len(L), **{'share_' + r: sum(x['proj_' + r] for x in L) / F for r in R},
                                **{'sq_' + r: sum(x['sq_' + r] for x in L) / F for r in R})
    pr = {}
    if 'A' in res['groups'] and 'B' in res['groups']:
        a, b = res['groups']['A'], res['groups']['B']
        pr['P164'] = dict(B_summ_own=b['share_summ_own'], holds=bool(b['share_summ_own'] >= 0.3))
        pr['P165'] = dict(A_summ_own=a['share_summ_own'], A_lab2=a['share_lab2'], holds=bool(a['share_summ_own'] > a['share_lab2']))
    res['predictions'] = pr
    md = ['# E36 tables', '', '| group | nets | share: own summary | all summaries | query token (block 1) | labelled, block-2 read | squared size / full: own summary | all summaries | query | labelled |',
          '|---' * 10 + '|']
    for g, o in res['groups'].items():
        md.append('| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (g, o['n'], *[o['share_' + r] for r in R], *[o['sq_' + r] for r in R]))
    md += ['', '| net | set | ' + ' | '.join(R) + ' |', '|---' * (2 + len(R)) + '|']
    for t, o in res['per_net'].items():
        md.append('| %s | %s | ' % (t, o['set']) + ' | '.join('%.2f' % o[r] for r in R) + ' |')
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e36.json', 'w'), indent=1)
    open('results/e36_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1))


if __name__ == '__main__':
    main()
