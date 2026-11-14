"""E18b items (PREREG-B amendment 32): E18's 60 base episodes with the modified source's errors rebuilt so that both versions
have the same sum of squares AND zero mean error (separating bias correction from reliability weighting):
  big miss  kappa 0.75: one error b = sqrt(0.75 SS) with random sign, the other three b/3 each with the opposite sign (mean 0);
  spread    four errors sqrt(SS/4), two positive and two negative in a random arrangement (mean 0).
Other sources, readings, labels and identities as in E18. Displayed numbers rounded to 2 decimals; Bayes from the displayed numbers.
usage: python e18b_data.py -> results/e18b_items.json
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import e18_data as D


def main():
    p = 'results/e18b_items.json'
    assert not os.path.exists(p)
    old = json.load(open('results/e18_items.json')); rng = np.random.default_rng(20_261_109)
    items = []
    for k in range(max(it['pair'] for it in old) + 1):
        sp = [it for it in old if it['pair'] == k and it['version'] == 'spread'][0]
        s = sp['src']; err0 = np.array(sp['errors']); read = np.array(sp['readings'])
        SS = float((err0[s] ** 2).sum())                      # E18's displayed SS for this source
        sgn = rng.choice([-1.0, 1.0]); pos = rng.integers(0, 4); arr = rng.permutation([1.0, 1.0, -1.0, -1.0])
        b = np.sqrt(0.75 * SS); big = np.full(4, -sgn * b / 3); big[pos] = sgn * b
        spread = arr * np.sqrt(SS / 4)
        for ver, r in (('spread', spread), ('big', big)):
            err = err0.copy(); err[s] = r; err = np.round(err, 2)
            mb, wb = D.bayes(err, read)
            # source labels: reuse E18's order (labels are in the prompt lines "Source X: ..."), rebuilt from E18's prompt
            order_lines = [l for l in sp['stated'].split('\n') if l.startswith('Source ') and ',' in l]
            labels = [l.split(':')[0].split()[1] for l in order_lines]                     # labels in display order
            # map display order back to sources by matching E18's displayed errors
            disp = {}
            for l, labl in zip(order_lines, labels):
                vals = np.array([float(x) for x in l.split(':')[1].split(',')])
                srcs = [x for x in range(3) if np.allclose(err0[x], vals)]
                disp[labl] = srcs[0]
            order = [disp[l] for l in labels]
            hist = '\n'.join(f"Source {l}: " + ', '.join(D.fmt(v) for v in err[x]) for l, x in zip(labels, order))
            rdt = '\n'.join(f"Source {l}: {read[x]:.2f}" for l, x in zip(labels, order))
            items.append(dict(pair=k, version=ver, src=s, src_label=[l for l, x in zip(labels, order) if x == s][0],
                              ss_displayed=float((err[s] ** 2).sum()), mean_err_displayed=float(err[s].mean()), errors=err.tolist(),
                              readings=read.tolist(), bayes_mean=mb, bayes_w=wb,
                              stated=D.STATED.format(hist=hist, read=rdt),
                              unstated=D.UNSTATED.format(hist=hist.replace('Source', 'Forecaster'), read=rdt.replace('Source', 'Forecaster'))))
    json.dump(items, open(p, 'w'), indent=1)
    dm = max(abs(items[i]['bayes_mean'] - items[i + 1]['bayes_mean']) for i in range(0, len(items), 2))
    me = max(abs(it['mean_err_displayed']) for it in items)
    print(len(items) // 2, 'pairs; max |Bayes mean difference within pair| %.4f; max |mean error of the modified source| %.4f' % (dm, me))
    print(items[1]['stated'])


if __name__ == '__main__':
    main()
