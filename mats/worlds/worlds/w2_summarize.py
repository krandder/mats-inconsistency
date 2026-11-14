"""Summarise the W2 pilot (PREREG.md, W2 section): gap-to-Bayes by N and policy for every arm and reference,
ability fractions, scaling slope, and the C4 / C7 diagnostic readings.

usage: python w2_summarize.py OUT      (OUT holds evalset dirs es_* and run dirs <arm>_s<seed>/eval_<es>/)"""
import glob, json, math, os, re, sys
import numpy as np
import harness as Hn
import world2_market as W

OUT = sys.argv[1]
REFS = [r for r in W.REFS if r != 'bayes']


def ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return float('nan'), float('nan'), 0
    return float(np.mean(v)), (float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0), len(v)


def fmt(m, s=None):
    if m is None or not np.isfinite(m):
        return '-'
    return f'{m:.3f}' if s is None or not np.isfinite(s) else f'{m:.3f}±{s:.3f}'


# ------------------------------------------------------------------ load
evalsets = {os.path.basename(d): json.load(open(os.path.join(d, 'index.json')))
            for d in sorted(glob.glob(os.path.join(OUT, 'es_*'))) if os.path.exists(os.path.join(d, 'index.json'))}
cells = {}                                            # (policy, N) -> (es, cell dict)
for es, idx in evalsets.items():
    for c in idx['cells']:
        cells[(c['policy'], c['N'])] = (es, c)
runs = {}                                             # arm -> seed -> run dir
for d in sorted(glob.glob(os.path.join(OUT, '*_s*'))):
    m = re.match(r'(.+)_s(\d+)$', os.path.basename(d))
    if m and os.path.isdir(d):
        runs.setdefault(m.group(1), {})[int(m.group(2))] = d
ARMS = sorted(runs)


def cell_data(pol, N):
    es, c = cells[(pol, N)]
    return dict(np.load(os.path.join(OUT, es, c['file']))), es, c


def pred(arm, seed, es, pol, N):
    f = os.path.join(runs[arm][seed], 'eval_' + es, f'pred_N{N}_{pol}.npz')
    return dict(np.load(f)) if os.path.exists(f) else None


lines = ['# W2 market world pilot — summary', '']
for arm in ARMS:
    for s, d in sorted(runs[arm].items()):
        mp = os.path.join(d, 'meta.json')
        if os.path.exists(mp):
            m = json.load(open(mp))
            lines.append(f'- {arm} s{s}: params {m["n_params"]}, best val {m["best_val"]:.4f} (Bayes {m["val_bayes_nll"]:.4f}, '
                         f'gap {m["best_val"] - m["val_bayes_nll"]:.4f}) @ {m["best_step"]}, '
                         f'last-quarter improvement {m["last_quarter_improvement"]}, {m["ms_per_step"]:.0f} ms/step')
lines.append('')

# ------------------------------------------------------------------ gap to Bayes by N / policy
gen_pols = [p for p in ('dir', 'conc', 'relay') if any(k[0] == p for k in cells)]
for split in ('all', 'unk', 'relay', 'hidden_relay', 'q_noSPOT'):
    lines += [f'## Excess NLL over Bayes (nats), split `{split}` — references, then arms (seed mean±se)', '',
              '| policy | N | n | Bayes NLL | ' + ' | '.join(REFS) + ' | ' + ' | '.join(ARMS) + ' |',
              '|' + '---|' * (4 + len(REFS) + len(ARMS))]
    for pol in gen_pols:
        for N in sorted(n for p, n in cells if p == pol):
            d, es, c = cell_data(pol, N)
            meta = {k[5:]: d[k] for k in d if k.startswith('meta_')}
            sel = W.MarketWorld.splits[split](meta)
            if sel.sum() < 5:
                continue
            row = [pol, str(N), str(int(sel.sum())), fmt(d['nll_bayes'][sel].mean())]
            row += [fmt((d['nll_' + r] - d['nll_bayes'])[sel].mean()) for r in REFS]
            for arm in ARMS:
                g = []
                for s in runs[arm]:
                    p = pred(arm, s, es, pol, N)
                    g.append(None if p is None else float((p['nll_model'] - d['nll_bayes'])[sel].mean()))
                row.append(fmt(*ms(g)[:2]))
            lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

# ------------------------------------------------------------------ KL and scaling slope
lines += ['## KL(Bayes || model) on `dir` by N, and scaling slope (KL vs log2 N, N >= 128)', '',
          '| arm | ' + ' | '.join(str(N) for N in sorted(n for p, n in cells if p == 'dir')) + ' | slope/doubling |',
          '|' + '---|' * (2 + len([1 for p, n in cells if p == 'dir']))]
for arm in ARMS:
    row, pts = [arm], {}
    for N in sorted(n for p, n in cells if p == 'dir'):
        d, es, c = cell_data('dir', N)
        k = []
        for s in runs[arm]:
            p = pred(arm, s, es, 'dir', N)
            k.append(None if p is None else float(p['kl_model'].mean()))
            if p is not None and N >= 128:
                pts.setdefault(s, []).append((math.log2(N), float(p['kl_model'].mean())))
        row.append(fmt(*ms(k)[:2]))
    sl = [np.polyfit(*zip(*v), 1)[0] for v in pts.values() if len(v) >= 3]
    row.append(fmt(*ms(sl)[:2]))
    lines.append('| ' + ' | '.join(row) + ' |')
lines.append('')

# ------------------------------------------------------------------ abilities
lines += ['## Ability fractions (weaker-ref -> Bayes NLL gain reproduced by the model; seed mean)', '',
          '| arm | policy | N | ' + ' | '.join(a[0] for a in W.MarketWorld.abilities) + ' |',
          '|' + '---|' * (3 + len(W.MarketWorld.abilities))]
for arm in ARMS:
    for pol in gen_pols:
        for N in sorted(n for p, n in cells if p == pol):
            es, c = cells[(pol, N)]
            vals = {a[0]: [] for a in W.MarketWorld.abilities}
            for s, rd in runs[arm].items():
                f = os.path.join(rd, 'eval_' + es, 'summary.json')
                if not os.path.exists(f):
                    continue
                for cc in json.load(open(f))['cells']:
                    if cc['N'] == N and cc['policy'] == pol:
                        for k, v in cc['abilities'].items():
                            vals[k].append(v['frac'])
            lines.append(f'| {arm} | {pol} | {N} | ' + ' | '.join(fmt(*ms(vals[a[0]])[:2]) for a in W.MarketWorld.abilities) + ' |')
lines.append('')


# ------------------------------------------------------------------ C4
def gauss_pred(mean, var):
    return np.ones((len(mean), 1)), mean[:, None], var[:, None]


for N in sorted(n for p, n in cells if p == 'c4'):
    d, es, c = cell_data('c4', N)
    pb = Hn._unragged(d)
    kl_f = Hn.kl_bayes_model(pb, gauss_pred(d['mean_freq_only'], d['var_freq_only']))
    kl_p = Hn.kl_bayes_model(pb, gauss_pred(d['mean_prior'], d['var_prior']))
    E = len(kl_f); pair = d['meta_pair']; mem = d['meta_member']
    i0 = np.flatnonzero(mem == 0); i1 = np.flatnonzero(mem == 1)
    shiftB = d['mean_bayes'][i0] - d['mean_bayes'][i1]
    lines += [f'## C4 frequency-matched calibration pairs, N={N} ({E} episodes, {len(i0)} pairs)', '',
              f'Pre-registered effect: KL(Bayes_0||Bayes_1) median {np.nanmedian(d["meta_delta"]):.2f} nats (all >= 0.3); '
              f'KL(Bayes||freq_only) median {np.median(kl_f):.3f}, min {kl_f.min():.3f} (must be >= 0.1: '
              f'{(kl_f >= 0.1).mean():.0%} of episodes); KL(Bayes||prior) median {np.median(kl_p):.3f}.', '',
              '| arm | seed | KL(B||model) median | closure median | frac episodes closure>=0.5 | verdict (>=80%) | '
              'mean-shift fraction (model/Bayes, median over pairs) | sign agreement |', '|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        for s in sorted(runs[arm]):
            p = pred(arm, s, es, 'c4', N)
            if p is None:
                continue
            clos = 1 - p['kl_model'] / kl_f
            fr = float((clos >= 0.5).mean())
            shiftM = p['mean_model'][i0] - p['mean_model'][i1]
            lines.append(f'| {arm} | {s} | {np.median(p["kl_model"]):.3f} | {np.median(clos):.2f} | {fr:.0%} | '
                         f'{"pass" if fr >= 0.8 else "FAIL"} | {np.median(shiftM / shiftB):.2f} | '
                         f'{np.mean(np.sign(shiftM) == np.sign(shiftB)):.0%} |')
    lines.append('')

# ------------------------------------------------------------------ C7
for N in sorted(n for p, n in cells if p == 'c7'):
    d, es, c = cell_data('c7', N)
    world = W.make_world(evalsets[es]['world_cfg'])
    ep = world.sample(np.random.default_rng(c['seed']), c['E'], N, 'c7')
    assert np.allclose(ep['target'], d['target'])
    arm_of = d['meta_arm'].astype(int); setid = d['meta_set'].astype(int)
    # naive references: double counting for diamonds (D, D4), provenance-naive for relay arms
    naive_var = d['var_prov_naive'].copy(); naive_mean = d['mean_prov_naive'].copy()
    for i, e in enumerate(ep['eps']):
        if W.C7_ARMS[arm_of[i]] in ('D', 'D4'):
            w_, m_, s_ = W.predictive(W.naive_double_count(e))
            mm = (w_ * m_).sum(); naive_mean[i] = mm; naive_var[i] = (w_ * (s_ ** 2 + m_ ** 2)).sum() - mm ** 2
    pb = Hn._unragged(d)
    kl_n = Hn.kl_bayes_model(pb, gauss_pred(naive_mean, naive_var))
    lines += [f'## C7 diamond / relay sets, N={N} ({int(setid.max()) + 1} sets)', '',
              'log sd ratio = 0.5 log(var_a / var_b) for the pair (a = shared-root arm, b = independent arm), mean over '
              'sets. Bayes > 0 (shared evidence discounted); naive (double-count / provenance-naive) ~ 0. '
              'dedup fraction = (model - naive) / (Bayes - naive).', '',
              '| pair | Bayes | naive | KL(B_a||naive_a) median | ' + ' | '.join(f'{a} ratio / dedup frac / KL_a' for a in ARMS) + ' |',
              '|' + '---|' * (4 + len(ARMS))]
    for a, b in W.C7_PAIRS:
        ia = np.flatnonzero(arm_of == W.C7_ARMS.index(a)); ib = np.flatnonzero(arm_of == W.C7_ARMS.index(b))
        assert np.array_equal(setid[ia], setid[ib])
        rB = 0.5 * np.log(d['var_bayes'][ia] / d['var_bayes'][ib]).mean()
        rN = 0.5 * np.log(naive_var[ia] / d['var_bayes'][ib]).mean()
        cols = []
        for arm in ARMS:
            rr, ff, kk = [], [], []
            for s in runs[arm]:
                p = pred(arm, s, es, 'c7', N)
                if p is None:
                    continue
                rM = 0.5 * np.log(p['var_model'][ia] / p['var_model'][ib]).mean()
                rr.append(rM); ff.append((rM - rN) / (rB - rN)); kk.append(float(np.median(p['kl_model'][ia])))
            cols.append(f'{fmt(*ms(rr)[:2])} / {fmt(*ms(ff)[:2])} / {fmt(ms(kk)[0])}')
        lines.append(f'| {a}/{b} | {rB:.3f} | {rN:.3f} | {np.median(kl_n[ia]):.3f} | ' + ' | '.join(cols) + ' |')
    lines.append('')

print('\n'.join(lines))
