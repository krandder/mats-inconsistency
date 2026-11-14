"""EXPLORATORY (PREREG Addendum B; enters no decision rule): the compute curve of the meta-trained reader.

Compares FR-small (1e15 FLOPs), FR seeds 0-2 (1e16) and the exploratory FR seed 0 at 4e16 (plan trainX, output tree
results/vm_out/x) on the same test episodes: AUROC by k per slice, final validation NLL, and C4 for reference.
  P16_OUT=results/vm_out python3 analyze_x.py  -> results/explore_compute.json, charts/compute_curve.json
"""
import json, os
import numpy as np
import analyze_final as AF

MAIN = AF.OUT
XO = os.path.join(MAIN, 'x')
RES = os.environ.get('P16_RESULTS', 'results')
CH = os.environ.get('P16_CHARTS', 'charts')


def val(tj):
    if not os.path.exists(tj):
        return None
    h = json.load(open(tj))
    return dict(flops=h['flops'], steps=h['steps'], wall_s=h['wall_s'], val_final=h['hist'][-1]['val'])


def diagnostics(te, sc, S):
    """Where the reader loses (reported only): ID AUROC at k = 16 by positive type, by CMI bin and by m, FR mean over
    seeds 0-2 against C4; and the ID AUROC by k on the tune split for every arm there."""
    j = AF.K16; typ, m, cmi = te['typ'], te['m'], te['cmi']
    P, N = S['ID']
    frs = [r for r in sc if r.startswith('FR_s')]
    arms = {'FR': frs, **{b: [b] for b in ('C4', 'C5', 'S1', 'S2', 'C1') if b in sc}}
    mean_au = lambda rs, pp, nn: AF.f(np.mean([AF.au(sc[r][:, j], pp, nn) for r in rs]))
    d = dict(by_pos_type={}, by_cmi_bin={}, by_m={})
    for a, rs in arms.items():
        d['by_pos_type'][a] = {t: mean_au(rs, P & (typ == t), N) for t in np.unique(typ[P])}
        d['by_cmi_bin'][a] = {f'{lo}-{hi}': mean_au(rs, P & (cmi >= lo) & (cmi < hi), N)
                              for lo, hi in ((.05, .1), (.1, .2), (.2, .4), (.4, 9))}
        d['by_m'][a] = {int(mm): mean_au(rs, P & (m == mm), N & (m == mm)) for mm in (2, 6, 12)}
    d['n_pos_by_type'] = {t: int((P & (typ == t)).sum()) for t in np.unique(typ[P])}
    # does FR carry information C4 lacks? rank-average of the two scores (per k), ID and primary slices
    if frs and 'C4' in sc:
        from scipy.stats import rankdata
        rk = lambda v: rankdata(v) / len(v)
        comb = {}
        for sl in ('ID', 'primary'):
            pp, nn = S[sl]
            comb[sl] = {'FR+C4': [AF.f(np.mean([AF.au(rk(sc[r][:, i]) + rk(sc['C4'][:, i]), pp, nn) for r in frs]))
                                  for i in range(len(AF.KS))],
                        'C4': [AF.f(AF.au(sc['C4'][:, i], pp, nn)) for i in range(len(AF.KS))]}
        d['rank_average_fr_c4'] = comb
    # C4's score is forward gain + lam * reverse CI term (lam = 1). Split it: FR's V is a forward gain only.
    zf = f'{MAIN}/eval/test/C4.npz'
    if frs and os.path.exists(zf):
        z = np.load(zf)
        pos = {int(v): i for i, v in enumerate(z['idx'])}
        o = np.array([pos[int(v)] for v in te['idx']])
        fwd, rev = z['x_fwd'][o], z['x_rev'][o]
        from scipy.stats import rankdata
        rk = lambda v: rankdata(v) / len(v)
        spl = {}
        for sl in ('ID', 'primary'):
            pp, nn = S[sl]
            spl[sl] = {'C4_forward_only': [AF.f(AF.au(fwd[:, i], pp, nn)) for i in range(len(AF.KS))],
                       'C4_reverse_only': [AF.f(AF.au(rev[:, i], pp, nn)) for i in range(len(AF.KS))],
                       'FR': [AF.f(np.mean([AF.au(sc[r][:, i], pp, nn) for r in frs])) for i in range(len(AF.KS))],
                       'FR+C4_reverse (rank avg)': [AF.f(np.mean([AF.au(rk(sc[r][:, i]) + rk(rev[:, i]), pp, nn) for r in frs]))
                                                    for i in range(len(AF.KS))]}
        d['c4_forward_vs_reverse'] = spl
    if os.path.exists(f'{MAIN}/eval/design_tune.json'):
        tu = AF.load_split('tune')
        Pt, Nt = tu['S']['ID']
        d['tune_id_auroc_by_k'] = {a: [AF.f(AF.au(v[:, i], Pt, Nt)) for i in range(len(AF.KS))]
                                   for a, v in tu['sc'].items() if getattr(v, 'ndim', 0) == 2}
    return d


def main():
    te = AF.load_split('test')
    sc, S, idx = te['sc'], te['S'], te['idx']
    arms = {}
    if 'FR-small' in sc:
        arms['FR-small_s0 (1e15)'] = (sc['FR-small'], val(f'{MAIN}/runs/FR-small_s0/train.json'))
    for r in sorted(r for r in sc if r.startswith('FR_s')):
        arms[f'{r} (1e16)'] = (sc[r], val(f'{MAIN}/runs/{r}/train.json'))
    AF.OUT = XO
    x = AF.merged('test', 'FR__FR_s0', idx) if os.path.exists(f'{XO}/eval/test') else None
    AF.OUT = MAIN
    if x is not None:
        arms['FR_s0 (4e16, exploratory)'] = (x, val(f'{XO}/runs/FR_s0/train.json'))
    for b in ('C4', 'C1'):
        if b in sc:
            arms[b] = (sc[b], None)
    out = dict(note='EXPLORATORY (PREREG Addendum B): not pre-registered, enters no decision rule', ks=list(AF.KS), arms={})
    for name, (s, v) in arms.items():
        out['arms'][name] = dict(train=v, auroc={sl: [AF.f(AF.au(s[:, j], pp, nn)) for j in range(len(AF.KS))]
                                                 for sl, (pp, nn) in S.items()})
    out['diagnostics'] = diagnostics(te, sc, S)
    os.makedirs(RES, exist_ok=True); os.makedirs(CH, exist_ok=True)
    json.dump(out, open(f'{RES}/explore_compute.json', 'w'), indent=1)
    json.dump(out, open(f'{CH}/compute_curve.json', 'w'), indent=1)
    j = AF.K16
    for name, d in out['arms'].items():
        a = d['auroc']
        print(f"{name:28s} ID@16 {a['ID'][j]:.3f}  primary@16 {a['primary'][j]:.3f}  "
              f"primary@64 {a['primary'][4]:.3f}  val {d['train']['val_final'] if d['train'] else '-'}")


if __name__ == '__main__':
    main()
