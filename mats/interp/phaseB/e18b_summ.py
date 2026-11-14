"""E18b analysis (PREREG-B amendment 32): single-pass LLMs on zero-mean matched pairs, both framings. Same statistics as E18.
usage: python e18b_summ.py -> results/e18b.json, results/e18b_tables.md
"""
import json
import numpy as np
import e18b_call as C

NBOOT = 1000
MODELS = ['haiku45', 'sonnet46', 'opus48', 'gpt56', 'qwen17']


def main():
    items = json.load(open('results/e18b_items.json')); by = {(it['pair'], it['version']): it for it in items}
    npair = max(it['pair'] for it in items) + 1
    calls = [json.loads(l) for l in open('results/e18b_calls.jsonl')]
    ans = {}
    for c in calls:
        a = C.parse(c.get('text'))
        if c['ok'] and a is not None:
            ans.setdefault((c['arm'], c['pair'], c['version']), {})[c['sample']] = a
    rng = np.random.default_rng(0); out = dict(arms={}, predictions={}, spend={})

    def rows_of(arm):
        rows = []
        for k in range(npair):
            b, s = ans.get((arm, k, 'big')), ans.get((arm, k, 'spread'))
            if not b or not s:
                continue
            ib, isp = by[(k, 'big')], by[(k, 'spread')]; src = isp['src']
            rows.append(dict(pair=k, mb=np.mean(list(b.values())), ms=np.mean(list(s.values())), L=isp['readings'][src] - isp['bayes_mean'],
                             W=isp['bayes_w'][src], db=ib['bayes_mean'] - isp['bayes_mean']))
        return rows

    def shift(rows):
        return sum((r['mb'] - r['ms']) * np.sign(r['L']) for r in rows) / sum(r['W'] * abs(r['L']) for r in rows)

    for arm in C.ARMS:
        rows = rows_of(arm)
        if not rows:
            continue
        bs = [shift([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(NBOOT)]
        x, y = [], []
        for (a_, k, v), dd in ans.items():
            if a_ == arm:
                it = by[(k, v)]; a = np.mean(it['readings']); x.append(it['bayes_mean'] - a); y.append(np.mean(list(dd.values())) - a)
        x, y = np.array(x), np.array(y)
        out['arms'][arm] = dict(n_pairs=len(rows), shift=float(shift(rows)), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                bayes_shift=float(sum(r['db'] * np.sign(r['L']) for r in rows) / sum(r['W'] * abs(r['L']) for r in rows)),
                                n_pos=int(sum((r['mb'] - r['ms']) * np.sign(r['L']) > 0 for r in rows)),
                                n_neg=int(sum((r['mb'] - r['ms']) * np.sign(r['L']) < 0 for r in rows)),
                                slope=float(np.cov(x, y)[0, 1] / np.var(x, ddof=1)),
                                mad=float(np.mean([abs(np.mean(list(dd.values())) - by[(k, v)]['bayes_mean']) for (a_, k, v), dd in ans.items() if a_ == arm])),
                                rows=rows)
    A = out['arms']
    for fr in ('S', 'U'):
        ev = [f'{fr}_{m}' for m in MODELS if f'{fr}_{m}' in A and A[f'{fr}_{m}']['slope'] >= 0.3]
        common = sorted(set.intersection(*[set(r['pair'] for r in A[a]['rows']) for a in ev])) if ev else []
        def pooled(ks):
            vals = []
            for a in ev:
                rr = {r['pair']: r for r in A[a]['rows']}; vals.append(shift([rr[k] for k in ks]))
            return float(np.mean(vals))
        pb = [pooled([common[i] for i in rng.integers(0, len(common), len(common))]) for _ in range(NBOOT)] if len(ev) else []
        out['predictions'][fr] = dict(evaluable=ev, pooled=pooled(common) if ev else None,
                                      ci=[float(np.quantile(pb, 0.025)), float(np.quantile(pb, 0.975))] if pb else None,
                                      n_pos=int(sum(A[a]['shift'] > 0 for a in ev)), n_neg=int(sum(A[a]['shift'] < 0 for a in ev)))
    S, U = out['predictions']['S'], out['predictions']['U']
    out['predictions']['P131'] = dict(holds=bool(len(S['evaluable']) >= 2 and S['ci'][0] > 0 and S['n_pos'] >= len(S['evaluable']) - 1),
                                      testable=len(S['evaluable']) >= 2)
    out['predictions']['P132'] = dict(holds=bool(len(U['evaluable']) >= 2 and U['ci'][1] < 0), testable=len(U['evaluable']) >= 2)
    for arm in C.ARMS:
        cc = [c for c in calls if c['arm'] == arm]
        if cc:
            out['spend'][arm] = dict(calls=len(cc), accepted=int(sum(c['ok'] and C.parse(c.get('text')) is not None for c in cc)),
                                     cost_usd=float(sum(c.get('cost') or 0 for c in cc)), gpt_tokens=int(sum(c.get('tokens') or 0 for c in cc)))
    json.dump(out, open('results/e18b.json', 'w'), indent=1, default=float)
    print({k: v for k, v in out['predictions'].items()})
    T = ['| arm | pairs | shift [95% CI] | pairs + / − | Bayes shift | reliability slope | mean abs dev from Bayes |', '|---|---|---|---|---|---|---|']
    for arm, v in A.items():
        T.append(f"| {arm} | {v['n_pairs']} | {v['shift']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['n_pos']} / {v['n_neg']} | {v['bayes_shift']:+.3f} | {v['slope']:.2f} | {v['mad']:.3f} |")
    T += ['', '| arm | calls | accepted | cost (USD, list) | GPT tokens |', '|---|---|---|---|---|']
    for arm, v in out['spend'].items():
        T.append(f"| {arm} | {v['calls']} | {v['accepted']} | {v['cost_usd']:.2f} | {v['gpt_tokens']} |")
    open('results/e18b_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
