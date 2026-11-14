"""E18 items (PREREG-B amendment 25): matched big-miss / spread error histories rendered as LLM prompts.
Pairs: fresh episodes of the tiny world (default_rng(20_261_105)); per episode the first source with SS >= 3.182394 (E12's
large range) and |reading - Bayes mean| >= 0.75; 60 pairs. The modified source's four errors become a big miss (kappa 0.85)
or spread out (kappa 0.25) with the same SS, signs and position (E12's construction); other sources keep their errors.
All displayed numbers are rounded to two decimals; Bayes is computed from the displayed numbers.
usage: python e18_data.py -> results/e18_items.json
"""
import json, os
import numpy as np
import nnoise  # noqa: F401  (tiny-pfn path)
import world as W
import qweights as QWT

LO_LARGE, LEVER, N_PAIRS = 3.182394, 0.75, 60
LABELS = ['A', 'B', 'C']
STATED = ("You are estimating an unknown quantity Y from the readings of three sources.\n"
          "How the data were generated: Y is drawn from a normal distribution with mean 0 and standard deviation 1. Each source reports "
          "Y plus an error. A source's errors are independent draws from a normal distribution with mean 0 and a variance that is fixed "
          "for that source. Each source's variance was drawn at random from {{0.1, 0.5, 2.0}}, each equally likely, independently of the "
          "other sources. The variances are not shown.\n"
          "To judge reliability, here are each source's errors (reading minus true value) on four earlier questions whose answers are "
          "known (same sources, same generating process):\n{hist}\n"
          "On the new question, the sources read:\n{read}\n"
          "What is your best estimate of the new question's true value Y (the mean of your posterior distribution for Y)? "
          "Answer with a single number with two decimals and nothing else.")
UNSTATED = ("Three forecasters each predicted the value of an unknown quantity.\n"
            "Here are each forecaster's errors (forecast minus true value) on four earlier questions whose answers are known:\n{hist}\n"
            "On a new question, the forecasters predict:\n{read}\n"
            "What is your best estimate of the true value? Answer with a single number with two decimals and nothing else.")


def bayes(err, read):
    """err (3, 4), read (3,) -> Bayes mean and the three Bayes weights (central differences), from the displayed numbers."""
    o = np.zeros((1, 3, 5)); o[0, :, :4] = err; o[0, :, 4] = read
    d = dict(o=o, y=np.zeros((1, 5)), v=np.ones((1, 3)), vidx=np.zeros((1, 3), int))
    _, _, mom, _ = W.references(d)
    return float(mom['bayes'][0][0]), [float(x) for x in QWT.bayes_weights(d)[0]]


def fmt(x):
    return f'{x:+.2f}'


def main():
    p = 'results/e18_items.json'
    assert not os.path.exists(p)
    rng = np.random.default_rng(20_261_105); d = W.sample_np(400, rng)
    perm = np.array([rng.permutation(3) for _ in range(400)]); signs = rng.choice([-1.0, 1.0], size=(400, 4)); pos = rng.integers(0, 4, 400)
    _, _, mom, _ = W.references(d); mB = mom['bayes'][0]
    R = d['o'][:, :, :4] - d['y'][:, None, :4]; SS = (R ** 2).sum(2); rd = d['o'][:, :, 4]
    items = []
    for e in range(400):
        cand = [s for s in range(3) if SS[e, s] >= LO_LARGE and abs(rd[e, s] - mB[e]) >= LEVER]
        if not cand:
            continue
        s = cand[0]; k = len(items) // 2
        lab = {src: LABELS[perm[e][src]] for src in range(3)}
        for ver, kap in (('spread', 0.25), ('big', 0.85)):
            err = R[e].copy()
            r = np.full(4, np.sqrt((1 - kap) / 3 * SS[e, s])); r[pos[e]] = np.sqrt(kap * SS[e, s]); err[s] = r * signs[e]
            err = np.round(err, 2); read = np.round(rd[e], 2)
            mb, wb = bayes(err, read)
            order = sorted(range(3), key=lambda x: lab[x])
            hist = '\n'.join(f"Source {lab[x]}: " + ', '.join(fmt(v) for v in err[x]) for x in order)
            rdt = '\n'.join(f"Source {lab[x]}: {read[x]:.2f}" for x in order)
            items.append(dict(pair=k, episode=e, version=ver, src=s, src_label=lab[s], ss_displayed=float((err[s] ** 2).sum()),
                              errors=err.tolist(), readings=read.tolist(), bayes_mean=mb, bayes_w=wb,
                              stated=STATED.format(hist=hist, read=rdt),
                              unstated=UNSTATED.format(hist=hist.replace('Source', 'Forecaster'), read=rdt.replace('Source', 'Forecaster'))))
        if len(items) >= 2 * N_PAIRS:
            break
    json.dump(items, open(p, 'w'), indent=1)
    dm = [abs(items[i]['bayes_mean'] - items[i + 1]['bayes_mean']) for i in range(0, len(items), 2)]
    print(len(items) // 2, 'pairs; max |Bayes mean difference within pair| %.4f (rounding)' % max(dm))
    print(items[1]['stated'])


if __name__ == '__main__':
    main()
