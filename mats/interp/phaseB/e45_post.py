"""E45 post-review sensitivity checks (not registered; asked for by the Codex review, results/review_e45.md):
  (a) slope c and invariance ratio restricted to pairs with all four answers (two samples of each version);
  (b) leave-one-pair-out ranges of c and of the invariance ratio, and the pair contributing most to mean D^2.
usage: python e45_post.py -> results/e45_post.json
"""
import json
import numpy as np


def main():
    items = json.load(open('results/e45_items.json')); it = {(x['pair'], x['version']): x for x in items}
    pairs = sorted({x['pair'] for x in items})
    A = {}
    for l in open('results/e45_calls.jsonl'):
        r = json.loads(l)
        if r['ok'] and r['answer'] is not None:
            A[(r['arm'], r['pair'], r['version'], r['sample'])] = r['answer']
    out = {}
    for arm in sorted({k[0] for k in A}):
        if 'qwen' in arm:
            continue
        rec = {}
        for p in pairs:
            a = {v: [A[(arm, p, v, s)] for s in (0, 1) if (arm, p, v, s) in A] for v in ('orig', 'neg')}
            if a['orig'] and a['neg']:
                o = it[(p, 'orig')]
                rec[p] = dict(D=np.mean(a['neg']) - np.mean(a['orig']), X=2 * o['mean_err'] * o['bayes_w'][o['src']],
                              full=len(a['orig']) == 2 and len(a['neg']) == 2,
                              rt=[(x[0] - x[1]) ** 2 / 2 for x in a.values() if len(x) == 2])

        def stat(ps):
            D = np.array([rec[p]['D'] for p in ps]); X = np.array([rec[p]['X'] for p in ps])
            rt = [v for p in ps for v in rec[p]['rt']]
            return float((D * X).sum() / (X * X).sum()), float(np.mean(D ** 2) / np.mean(rt)) if rt else np.nan
        allp = sorted(rec); full = [p for p in allp if rec[p]['full']]
        c_all, inv_all = stat(allp); c_full, inv_full = stat(full)
        loo = [stat([q for q in allp if q != p]) for p in allp]
        d2 = {p: rec[p]['D'] ** 2 for p in allp}; top = max(d2, key=d2.get)
        out[arm] = dict(n=len(allp), n_full=len(full), c=c_all, inv=inv_all, c_full=c_full, inv_full=inv_full,
                        c_loo=[min(x[0] for x in loo), max(x[0] for x in loo)], inv_loo=[min(x[1] for x in loo), max(x[1] for x in loo)],
                        top_pair=int(top), top_share_of_D2=float(d2[top] / sum(d2.values())))
        print(arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[arm].items()})
    json.dump(out, open('results/e45_post.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
