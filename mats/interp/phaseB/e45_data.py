"""E45 items (PREREG-B amendment 53): sign invariance in single-pass LLM answers.
From each of E18's 60 base pairs, the 'spread' version's three sources (errors and readings as displayed in E18); the negated source t
is the one with the largest |mean error|. Two versions per base:
  orig  E18's displayed errors;
  neg   source t's four displayed errors multiplied by -1 (everything else unchanged).
Bayes depends on each source's errors only through their sum of squares, so both versions have the same Bayes mean (checked).
Prompts: E18's stated and unstated templates, with E18's source labels and display order.
usage: python e45_data.py -> results/e45_items.json
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import e18_data as D


def main():
    p = os.environ.get('E45_ITEMS', 'results/e45_items.json')   # override: plumbing checks only
    assert not os.path.exists(p)
    old = json.load(open('results/e18_items.json'))
    items = []
    for k in range(max(it['pair'] for it in old) + 1):
        sp = [it for it in old if it['pair'] == k and it['version'] == 'spread'][0]
        err0 = np.array(sp['errors']); read = np.array(sp['readings'])
        order_lines = [l for l in sp['stated'].split('\n') if l.startswith('Source ') and ',' in l]
        labels = [l.split(':')[0].split()[1] for l in order_lines]
        disp = {}
        for l, labl in zip(order_lines, labels):
            vals = np.array([float(x) for x in l.split(':')[1].split(',')])
            disp[labl] = [x for x in range(3) if np.allclose(err0[x], vals)][0]
        order = [disp[l] for l in labels]
        t = int(np.argmax(np.abs(err0.mean(1))))
        for ver in ('orig', 'neg'):
            err = err0.copy()
            if ver == 'neg':
                err[t] = -err[t]
            mb, wb = D.bayes(err, read)
            hist = '\n'.join(f"Source {l}: " + ', '.join(D.fmt(v) for v in err[x]) for l, x in zip(labels, order))
            rdt = '\n'.join(f"Source {l}: {read[x]:.2f}" for l, x in zip(labels, order))
            items.append(dict(pair=k, version=ver, src=t, src_label=[l for l, x in zip(labels, order) if x == t][0],
                              mean_err=float(err[t].mean()), errors=err.tolist(), readings=read.tolist(), bayes_mean=mb, bayes_w=wb,
                              stated=D.STATED.format(hist=hist, read=rdt),
                              unstated=D.UNSTATED.format(hist=hist.replace('Source', 'Forecaster'), read=rdt.replace('Source', 'Forecaster'))))
    json.dump(items, open(p, 'w'), indent=1)
    dm = max(abs(items[i]['bayes_mean'] - items[i + 1]['bayes_mean']) for i in range(0, len(items), 2))
    me = np.array([abs(items[i]['mean_err']) for i in range(0, len(items), 2)])
    print(len(items) // 2, 'pairs; max |Bayes mean difference within pair| %.2e; |mean error| of the negated source: median %.2f, max %.2f'
          % (dm, np.median(me), me.max()))
    print(items[1]['stated'])


if __name__ == '__main__':
    main()
