"""W2c cue check on the C7 sets (PREREG-W2, W2c amendment): which token cues does a model's dedup rely on?
usage: python w2c_cue.py RUN_DIR ES_ROOT   -> prints a markdown table, writes RUN_DIR/cue.json

Ablations (model inputs only; the Bayes reference is unaffected by (a)-(c) because provenance is part of the data
the model is denied, and by (d) because C7 sets have no unknown noise levels, so Bayes variances do not depend on y):
  none     original tokens
  noprov   relay tokens get fresh unique root-event ids
  noflag   relay flag zeroed; origin-source block replaced by the quoting-source block
  both     noprov + noflag
  tie      (HI4 only) the four independent SPOT quotes of the root get one identical value (their mean)
Dedup fraction of a relay pair = model 0.5 log(var_shared / var_indep) / Bayes 0.5 log(var_shared / var_indep).
"""
import json, os, sys
import numpy as np
import harness as Hn
import world2_market as W

RUN, ES = sys.argv[1], sys.argv[2]
model, world, meta = Hn.load_run(RUN)
rng = np.random.default_rng(123)
out, lines = {}, []
for N in (96, 512):
    esd = os.path.join(ES, f'es_c7_{N}')
    if not os.path.exists(esd):
        continue
    idx = json.load(open(os.path.join(esd, 'index.json')))
    c = idx['cells'][0]
    d = dict(np.load(os.path.join(esd, c['file'])))
    ep = world.sample(np.random.default_rng(c['seed']), c['E'], N, 'c7')
    assert np.allclose(ep['target'], d['target'])
    X0, M = ep['X'], ep['mask']
    arm = d['meta_arm'].astype(int)
    rel = np.zeros(M.shape, bool)
    for i, e in enumerate(ep['eps']):
        rel[i, :e.n] = e.q_relay
    def ablate(kind):
        X = X0.copy()
        if kind in ('noprov', 'both'):
            v = rng.standard_normal((int(rel.sum()), W.D_EV)).astype(np.float32) / np.sqrt(W.D_EV)
            X[..., W.F_EV:W.F_EV + W.D_EV][rel] = v
        if kind in ('noflag', 'both'):
            X[..., W.F_RELAY][rel] = 0.0
            X[..., W.F_OSRC:W.F_OSRC + W.D_SRC + 2][rel] = X[..., W.F_SRC:W.F_SRC + W.D_SRC + 2][rel]
        if kind == 'tie':
            for i, e in enumerate(ep['eps']):
                if W.C7_ARMS[arm[i]] != 'HI4':
                    continue
                r1, S0 = e.K - 9, e.S - 4
                t = [k for k in range(e.n) if not e.q_relay[k] and e.ev_typ[e.q_ev[k]] == W.SPOT
                     and e.ev_items[e.q_ev[k], 0] == r1 and e.q_src[k] in (S0 + 1, S0 + 2)]
                X[i, t, W.F_Y] = X[i, t, W.F_Y].mean()
        return X
    var = {}
    for kind in ('none', 'noprov', 'noflag', 'both', 'tie'):
        p = Hn.predict(model, ablate(kind), M)
        var[kind] = Hn.moments(p)[1]
    res = {}
    for a, b in (('DR', 'DI'), ('HR', 'HI'), ('HR4', 'HI4')):
        ia = np.flatnonzero(arm == W.C7_ARMS.index(a)); ib = np.flatnonzero(arm == W.C7_ARMS.index(b))
        rB = 0.5 * np.log(d['var_bayes'][ia] / d['var_bayes'][ib]).mean()
        res[a + '/' + b] = {k: float(0.5 * np.log(var[k][ia] / var['none'][ib]).mean() / rB)
                            for k in ('none', 'noprov', 'noflag', 'both')}
    ib = np.flatnonzero(arm == W.C7_ARMS.index('HI4'))
    res['HI4 tie: 0.5 log(var_tied/var)'] = float(0.5 * np.log(var['tie'][ib] / var['none'][ib]).mean())
    res['HI4 tie / Bayes HR4-HI4 ratio'] = res['HI4 tie: 0.5 log(var_tied/var)'] / float(
        0.5 * np.log(d['var_bayes'][arm == W.C7_ARMS.index('HR4')] / d['var_bayes'][ib]).mean())
    out[N] = res
    lines += [f'### {os.path.basename(RUN.rstrip("/"))} N={N}', '', '| pair | none | noprov | noflag | both |', '|---|---|---|---|---|']
    for k, v in res.items():
        if isinstance(v, dict):
            lines.append(f'| {k} | ' + ' | '.join(f'{v[x]:.3f}' for x in ('none', 'noprov', 'noflag', 'both')) + ' |')
    lines += [f'| HI4 ties (model log-sd change; as fraction of Bayes HR4/HI4 ratio) | '
              f'{res["HI4 tie: 0.5 log(var_tied/var)"]:.3f} | {res["HI4 tie / Bayes HR4-HI4 ratio"]:.3f} | | |', '']
json.dump(out, open(os.path.join(RUN, 'cue.json'), 'w'), indent=1)
print('\n'.join(lines))
