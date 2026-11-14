"""Main table from runs/*/eval.json (and eval_init.json). usage: python summarize.py OUT"""
import glob, json, math, os, sys
import numpy as np

OUT = sys.argv[1]
Q = ('mu', 'logs', 'nll')


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


rows = {}
for f in glob.glob(os.path.join(OUT, 'runs', '*', 'eval.json')):
    name = os.path.basename(os.path.dirname(f))
    d, l, K, s, fr = name.split('_')
    key = (int(d[1:]), float(l[1:]), int(K[1:]), int(fr[1:]))
    rows.setdefault(key, {})[int(s[1:])] = load(f)
    if os.path.exists(f.replace('eval.json', 'eval_init.json')):
        rows.setdefault((int(d[1:]), 'init', 0, 0), {})[int(s[1:])] = load(f.replace('eval.json', 'eval_init.json'))


def eerr(e, K):
    return math.exp(np.mean([math.log(max(e[f'K{K}'][q]['nse'], 1e-12)) for q in Q]))


def gm(x):
    return math.exp(np.mean(np.log(np.maximum(x, 1e-12))))


def mse(x):
    x = np.array(x, float)
    return x.mean(), (x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float('nan'))


print('# estimability sweep summary\n')
for depth in (8, 4):
    base = rows.get((depth, 0.0, 4, 0), {})
    print(f'## depth {depth}\n')
    print('| arm | seeds | test gap to Bayes (nats) | Delta vs lambda=0 (paired) | K=2 E_err | K=4 E_err | K=4 improvement vs lambda=0 '
          '| N_eff(K=4) | N_match(K=4) | attn entropy | curved GELU | resid exkurt | weight change |')
    print('|' + '---|' * 13)
    order = [(depth, 'init', 0, 0), (depth, 0.0, 4, 0), (depth, 0.3, 4, 0), (depth, 3.0, 4, 0), (depth, 30.0, 4, 0),
             (depth, 3.0, 2, 0), (depth, 0.0, 4, 1)]
    for key in order:
        if key not in rows:
            continue
        ev = rows[key]
        seeds = sorted(s for s in ev if ev[s] is not None)
        if not seeds:
            continue
        gaps = [ev[s]['task']['gap'] for s in seeds]
        g, gse = mse(gaps)
        dl = [ev[s]['task']['gap'] - base[s]['task']['gap'] for s in seeds if s in base and base[s] is not None]
        dm, dse = mse(dl) if dl else (float('nan'), float('nan'))
        e2 = gm([eerr(ev[s], 2) for s in seeds]); e4 = gm([eerr(ev[s], 4) for s in seeds])
        e4b = gm([eerr(base[s], 4) for s in seeds if s in base and base[s] is not None]) if base else float('nan')
        nm = np.mean([ev[s]['K4']['N_match'] for s in seeds])
        ent = np.mean([np.mean(ev[s]['mech']['attn_entropy']) for s in seeds])
        cur = np.mean([np.mean(ev[s]['mech']['gelu_curved_frac']) for s in seeds])
        ku = np.mean([np.mean(ev[s]['mech']['resid_exkurt_rms']) for s in seeds])
        wc = np.mean([ev[s]['mech']['weight_change_mean'] for s in seeds])
        lab = {'init': 'untrained init'}.get(key[1], None) or (
            'random features (lambda=0)' if key[3] else f'lambda={key[1]:g} K={key[2]}')
        print(f'| {lab} | {len(seeds)} | {g:.4f} +- {gse:.4f} | {dm:+.4f} +- {dse:.4f} | {e2:.2e} | {e4:.2e} | '
              f'{e4b / e4:.2f}x | {1 / e4:.0f} | {nm:.0f} | {ent:.3f} | {cur:.3f} | {ku:.3f} | {wc:.3f} |')
    print()
    # per-quantity detail
    print('per quantity, K=4 nse (geo-mean over seeds) mu / logs / nll; K=2 likewise\n')
    for key in order:
        if key not in rows:
            continue
        ev = rows[key]; seeds = sorted(s for s in ev if ev[s] is not None)
        if not seeds:
            continue
        d4 = [gm([ev[s]['K4'][q]['nse'] for s in seeds]) for q in Q]
        d2 = [gm([ev[s]['K2'][q]['nse'] for s in seeds]) for q in Q]
        print(f'- {key}: K4 ' + ' / '.join(f'{x:.2e}' for x in d4) + ' ; K2 ' + ' / '.join(f'{x:.2e}' for x in d2))
    print()
