"""Final analysis for P16 (decision rules frozen in PREREG.md): python analyze.py final  (reads $P16_OUT, default results/vm_out)

Writes results/summary.json and charts/*.json. Every number named in PREREG's decision rules is computed here; arms that
did not run come out as null and the rule that needs them is marked 'not evaluable'.
Analysis choices not fixed by PREREG (stated in the summary under 'analysis_choices'):
  - AUROC drops episodes whose score is NaN (E1 outside its subsample).
  - The bootstrap resamples positives and negatives separately (class-stratified episode bootstrap).
  - k90(FR) is the worst (largest) of the FR seeds' k90; E1 has k <= 16 only and is left out of (c).
  - F4's FPR clause uses the mean over FR seeds of each seed's FPR at its own 80%-TPR threshold.
"""
import glob, json, os
import numpy as np
from scipy.stats import rankdata, spearmanr
import analyze as A

OUT = A.OUT
KS, SK = A.KS, A.SK
K16 = KS.index(16)
RES = os.environ.get('P16_RESULTS', 'results')
CH = os.environ.get('P16_CHARTS', 'charts')
BASE = ['C1', 'C4', 'C5', 'S1', 'S2']
NB = 2000


def f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def au(sc, pos, neg):
    """AUROC over episodes with finite scores."""
    sc = np.asarray(sc, float)
    ok = np.isfinite(sc)
    p, n = sc[pos & ok], sc[neg & ok]
    if len(p) == 0 or len(n) == 0:
        return float('nan')
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def au_idx(sc, ip, ineg):
    p, n = sc[ip], sc[ineg]
    r = rankdata(np.concatenate([p, n]))
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def merged(split, pattern, idx, key='preq'):
    fs = sorted(glob.glob(f'{OUT}/eval/{split}/{pattern}.npz'))
    if not fs:
        return None
    out = None
    for fn in fs:
        z = np.load(fn)
        if key not in z.files:
            continue
        pos = {int(i): r for r, i in enumerate(z['idx'])}
        if out is None:
            out = np.full((len(idx),) + z[key].shape[1:], np.nan)
        for r, i in enumerate(idx):
            if int(i) in pos:
                out[r] = z[key][pos[int(i)]]
    return out


def runs(prefix):
    return sorted(os.path.basename(p)[:-4].split('__')[1] for p in glob.glob(f'{OUT}/eval/test/{prefix}__*.npz'))


def load_split(split):
    D = A.design(split)
    idx = np.array(sorted(D))
    S, typ, lab = A.slice_masks(D, idx)
    cmi = np.array([D[i]['cmi'] for i in idx]); m = np.array([D[i]['m'] for i in idx])
    sc, st = {}, {}
    for a in BASE:
        v = merged(split, a, idx)
        if v is not None:
            sc[a] = v
    v = merged(split, 'C4', idx, 'settle')
    if v is not None:
        st['C4'] = v
    e = merged(split, 'E1@*', idx)
    if e is not None:
        sc['E1'] = e
    la, lt = merged(split, 'L@*', idx), merged(split, 'L@*', idx, 'x_ltrain')
    if la is not None:
        sc['L-all'], sc['L-train'] = la, lt
    def put(d, k, v):
        if v is not None:
            d[k] = v
    for r in runs('FR'):
        put(sc, r, merged(split, f'FR__{r}', idx)); put(st, r, merged(split, f'FR__{r}', idx, 'settle'))
    for r in runs('FR-small'):
        put(sc, 'FR-small', merged(split, f'FR-small__{r}', idx)); put(st, 'FR-small', merged(split, f'FR-small__{r}', idx, 'settle'))
    for r in runs('FR-direct'):
        put(sc, r + ':pred', merged(split, f'FR-direct__{r}', idx))
        put(sc, r + ':logit', merged(split, f'FR-direct__{r}', idx, 'x_logit'))
    c = merged(split, 'CONS', idx)
    if c is not None:
        sc['CONS'] = c
    return dict(D=D, idx=idx, S=S, typ=typ, lab=lab, cmi=cmi, m=m, sc=sc, st=st)


def boot_diff(fr_scores, base_scores, P, N, rng, nb=NB):
    """class-stratified bootstrap of mean_s AUROC(FR_s) - max_b AUROC(b) -> (point, lo, hi, draws)."""
    P = np.where(P)[0]; N = np.where(N)[0]
    pt = np.mean([au_idx(s, P, N) for s in fr_scores]) - max(au_idx(b, P, N) for b in base_scores)
    d = np.empty(nb)
    for i in range(nb):
        ip, ineg = rng.choice(P, len(P)), rng.choice(N, len(N))
        d[i] = np.mean([au_idx(s, ip, ineg) for s in fr_scores]) - max(au_idx(b, ip, ineg) for b in base_scores)
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), d


def k90(score_te, score_tu, te, tu):
    """smallest k with TPR >= .9 and FPR <= .1 on the test primary slice at a tune-ID 5%-FPR threshold; 256 if censored."""
    for j, k in enumerate(KS):
        thr = np.nanpercentile(score_tu[tu['S']['ID'][1], j], 95)
        p, n = te['S']['primary']
        s = score_te[:, j]
        tpr = np.mean(s[p] > thr); fpr = np.mean(s[n] > thr)
        if tpr >= .9 and fpr <= .1:
            return k
    return 256


def main():
    rng = np.random.default_rng(16)
    os.makedirs(RES, exist_ok=True); os.makedirs(CH, exist_ok=True)
    te = load_split('test')
    tu = load_split('tune') if os.path.exists(f'{OUT}/eval/design_tune.json') else None
    sc, st, S = te['sc'], te['st'], te['S']
    P, N = S['primary']
    FRS = [r for r in sc if r.startswith('FR_s')]
    FRD = sorted({r.split(':')[0] for r in sc if r.startswith('FR-direct_s')})
    base = [b for b in BASE if b in sc]
    summ = dict(n_test=len(te['idx']), n_primary=[int(P.sum()), int(N.sum())], fr_seeds=FRS, frd_seeds=FRD, baselines=base,
                analysis_choices=__doc__.split('Analysis choices not fixed by PREREG')[1].strip())
    summ['tuned'] = json.load(open(f'{OUT}/tuned.json')) if os.path.exists(f'{OUT}/tuned.json') else None
    summ['gate'] = json.load(open(f'{OUT}/gate.json')) if os.path.exists(f'{OUT}/gate.json') else None

    # FR-direct score: logit or predicted CMI, whichever wins on tune (ID slice, k = 16, mean over seeds)
    frd_key = 'pred'
    if FRD and tu is not None:
        ap = np.nanmean([au(tu['sc'][r + ':pred'][:, K16], *tu['S']['ID']) for r in FRD if r + ':pred' in tu['sc']] or [np.nan])
        al = np.nanmean([au(tu['sc'][r + ':logit'][:, K16], *tu['S']['ID']) for r in FRD if r + ':logit' in tu['sc']] or [np.nan])
        frd_key = 'logit' if al > ap else 'pred'
        summ['frd_score'] = dict(chosen=frd_key, tune_auroc_pred=f(ap), tune_auroc_logit=f(al))
    for r in FRD:
        sc[r] = sc[f'{r}:{frd_key}']

    # ---- AUROC vs k per slice / estimator
    arms = FRS + (['FR-small'] if 'FR-small' in sc else []) + FRD + base + [a for a in ('E1', 'L-all', 'L-train') if a in sc]
    table = {}
    for sl, (pp, nn) in S.items():
        table[sl] = {a: [f(au(sc[a][:, j], pp, nn)) for j in range(len(KS))] for a in arms}
        if FRS:
            table[sl]['FR'] = [f(np.mean([au(sc[r][:, j], pp, nn) for r in FRS])) for j in range(len(KS))]
        if FRD:
            table[sl]['FR-direct'] = [f(np.mean([au(sc[r][:, j], pp, nn) for r in FRD])) for j in range(len(KS))]
    settle_tab = {}
    for a in [x for x in st if st[x] is not None]:
        settle_tab[a] = [f(au(st[a][:, j], P, N)) for j in range(len(SK))]
    if FRS:
        settle_tab['FR'] = [f(np.mean([au(st[r][:, j], P, N) for r in FRS])) for j in range(len(SK))]
    strat = {}
    for mm in (2, 6, 12):
        sm = te['m'] == mm
        strat[mm] = {sl: {a: f(np.mean([au(sc[r][:, K16], S[sl][0] & sm, S[sl][1] & sm) for r in (FRS if a == 'FR' else [a])]))
                          for a in (['FR'] if FRS else []) + base + [x for x in ('E1', 'L-all', 'L-train') if x in sc]}
                     for sl in ('ID', 'primary')}
    summ['auroc_k16'] = {sl: {a: v[K16] for a, v in t.items()} for sl, t in table.items()}
    json.dump(dict(ks=KS, auroc=table, settlement_auroc_primary=dict(sks=SK, auroc=settle_tab), per_stratum_k16=strat),
              open(f'{CH}/auroc_vs_k.json', 'w'), indent=1)

    # ---- precondition
    pre = {}
    for r in FRS + (['FR-small_s0'] if 'FR-small' in sc else []) + FRD:
        tj = f'{OUT}/runs/{r}/train.json'
        if os.path.exists(tj):
            h = json.load(open(tj))
            vals = [x['val'] for x in h['hist']]; steps = [x['step'] for x in h['hist']]
            i80 = int(np.searchsorted(steps, 0.8 * steps[-1]))
            imp = (vals[i80] - vals[-1]) / abs(vals[i80]) if vals else float('nan')
            pre[r] = dict(steps=h['steps'], flops=h['flops'], wall_s=h['wall_s'], params=h['params'], val_first=vals[0],
                          val_final=vals[-1], val_improve_last20=imp, converged=bool(imp < .01))
    id_fr = [au(sc[r][:, K16], *S['ID']) for r in FRS]
    summ['precondition'] = dict(runs=pre, auroc_id_k16_per_seed=[f(x) for x in id_fr],
                                auroc_id_k16=f(np.mean(id_fr)) if id_fr else None)
    conv = all(pre.get(r, {}).get('converged', False) for r in FRS)
    summ['precondition']['met'] = bool(FRS and conv and np.mean(id_fr) >= .80)
    if 'FR-small' in sc and FRS:
        g = np.mean(id_fr) - au(sc['FR-small'][:, K16], *S['ID'])
        summ['precondition']['fr_small_gap_id'] = f(g); summ['precondition']['compute_limited'] = bool(g > .05)

    if not FRS or not base:
        summ['verdict'] = 'not evaluable: FR or baselines missing'
        json.dump(summ, open(f'{RES}/summary.json', 'w'), indent=1)
        print(json.dumps(summ, indent=1)[:3000])
        return

    # ---- (a)
    frk = [sc[r][:, K16] for r in FRS]; bk = [sc[b][:, K16] for b in base]
    a_full = boot_diff(frk, bk, P, N, rng)
    ra = dict(full=dict(diff=a_full[0], ci=[a_full[1], a_full[2]], max_baseline=max(base, key=lambda b: au(sc[b][:, K16], P, N))))
    ra['per_seed_full'] = {r: f(au(sc[r][:, K16], P, N) - max(au(b, P, N) for b in bk)) for r in FRS}
    ok = ra['full']['diff'] >= .05 and ra['full']['ci'][0] > 0 and all(v > 0 for v in ra['per_seed_full'].values())
    if 'E1' in sc:
        e = np.isfinite(sc['E1'][:, K16])
        a_sub = boot_diff(frk, bk + [sc['E1'][:, K16]], P & e, N & e, rng)
        ra['e1_subsample'] = dict(n=[int((P & e).sum()), int((N & e).sum())], diff=a_sub[0], ci=[a_sub[1], a_sub[2]])
        ra['per_seed_sub'] = {r: f(au(sc[r][:, K16], P & e, N & e) - max(au(b, P & e, N & e) for b in bk + [sc['E1'][:, K16]]))
                              for r in FRS}
        ok = ok and a_sub[0] >= .05 and a_sub[1] > 0 and all(v > 0 for v in ra['per_seed_sub'].values())
    else:
        ra['e1_subsample'] = 'not run'
    j16 = SK.index(16)
    if 'C4' in st:
        ds = np.mean([au(st[r][:, j16], P, N) for r in FRS]) - au(st['C4'][:, j16], P, N)
        ra['settlement_fr_minus_c4'] = f(ds); ok = ok and ds > 0
    ra['pass'] = bool(ok)
    ra['large_margin'] = bool(ok and a_full[1] >= .05 and (not isinstance(ra['e1_subsample'], dict) or ra['e1_subsample']['ci'][0] >= .05))
    summ['a'] = ra

    # ---- (b)
    fr_p = np.mean([au(s, P, N) for s in frk]); fr_id = np.mean(id_fr)
    rb = dict(fr_id=f(fr_id), fr_primary=f(fr_p))
    need_p = .80
    if 'L-all' in sc:
        la = au(sc['L-all'][:, K16], P, N)
        Pi, Ni = np.where(P & np.isfinite(sc['L-all'][:, K16]))[0], np.where(N & np.isfinite(sc['L-all'][:, K16]))[0]
        others = [s for s in frk] + bk
        wins = 0
        for i in range(500):
            ip, ineg = rng.choice(Pi, len(Pi)), rng.choice(Ni, len(Ni))
            wins += au_idx(sc['L-all'][:, K16], ip, ineg) >= max(au_idx(o, ip, ineg) for o in others)
        rb.update(l_all_primary=f(la), l_all_beats_all_frac=wins / 500)
        if la < .90 and wins / 500 >= .975:
            need_p = .5 + .8 * (la - .5); rb['exception_applied'] = True
    rb['need_primary'] = need_p
    rb['pass'] = bool(fr_id >= .85 and fr_p >= need_p)
    summ['b'] = rb

    # ---- (c) k90
    rc = {}
    if tu is not None:
        k_fr = [k90(sc[r], tu['sc'][r], te, tu) for r in FRS if r in tu['sc']]
        k_b = {b: k90(sc[b], tu['sc'][b], te, tu) for b in base if b in tu['sc']}
        if k_fr and k_b:
            kf = max(k_fr)
            rc = dict(fr_per_seed=k_fr, fr=kf, baselines=k_b, min_baseline=min(k_b.values()))
            rc['pass'] = bool(kf <= 128 and kf <= .5 * rc['min_baseline'])
    summ['c'] = rc or 'not evaluable (tune evals missing)'

    # ---- falsifiers
    F = {}
    sp_cmi = lambda s, msk: spearmanr(s[msk], te['cmi'][msk]).correlation if np.isfinite(s[msk]).all() else np.nan
    prim = P | N
    f1 = []
    for k in (8, 16, 32):
        j = KS.index(k)
        fa = np.mean([au(sc[r][:, j], P, N) for r in FRS]); fs = np.mean([sp_cmi(sc[r][:, j], prim) for r in FRS])
        cand = [b for b in ('C4', 'C5', 'S1', 'S2') if b in sc]
        ba = max(au(sc[b][:, j], P, N) for b in cand); bs = max(sp_cmi(sc[b][:, j], prim) for b in cand)
        f1.append(dict(k=k, fr=f(fa), best_b=f(ba), fr_spear=f(fs), best_b_spear=f(bs), within=bool(fa - ba <= .02 and fs - bs <= .05)))
    F['F1'] = dict(rows=f1, fires=all(x['within'] for x in f1))
    nov = summ['gate']['mean_novelty'] if summ['gate'] else None
    if 'L-train' in sc:
        lt = au(sc['L-train'][:, K16], P, N)
        F['F2'] = dict(fr=f(fr_p), l_train=f(lt), mean_novelty_pilot=nov,
                       fires=bool(summ['precondition']['met'] and nov is not None and nov >= .05 and fr_p <= lt + .02))
    if 'E1' in sc:
        e = np.isfinite(sc['E1'][:, K16])
        e1a = au(sc['E1'][:, K16], P & e, N & e); fra = np.mean([au(s, P & e, N & e) for s in frk])
        F['F3'] = dict(e1=f(e1a), fr=f(fra), fires=bool(e1a >= fra - .02))
    # F4 clause 1: FPR at 80% TPR
    negt = ['T0', 'T3a'] + A.HN
    fpr = {}
    for a in FRS + base + [x for x in ('E1', 'L-all') if x in sc] + FRD:
        s = sc[a][:, K16]
        ok_ = np.isfinite(s)
        if not (P & ok_).any():
            continue
        thr = np.percentile(s[P & ok_], 20)
        row = {'pooled_heldout': f(np.mean(s[N & ok_] > thr))}
        for t in negt:
            msk = (te['typ'] == t) & (te['lab'] == 0) & ok_
            row[t] = f(np.mean(s[msk] > thr)) if msk.any() else None
        fpr[a] = row
    fr_fpr = {k: f(np.mean([fpr[r][k] for r in FRS if fpr[r][k] is not None])) for k in fpr[FRS[0]]}
    c1 = fr_fpr['pooled_heldout'] > .15 or any(fr_fpr[t] is not None and fr_fpr[t] > .25 for t in negt)
    F4 = dict(fpr_at_80tpr_fr_mean=fr_fpr, clause1_fires=bool(c1))
    adv_f = f'{OUT}/eval/adv.npz'
    if os.path.exists(adv_f):
        z = np.load(adv_f)
        adv = {}
        for arm, stk in (('FR', 'fr_set'), ('C4', 'c4_set'), ('C5', 'c4_set')):
            if f'pick_{arm}' not in z.files:
                continue
            s = z[stk][np.arange(len(z[f'pick_{arm}'])), z[f'pick_{arm}']]
            bs = np.array([rng.choice(s, len(s)).mean() for _ in range(NB)])
            adv[arm] = dict(pick_settle_mean=f(s.mean()), ci=[f(np.percentile(bs, 2.5)), f(np.percentile(bs, 97.5))],
                            all_cand_mean=f(np.nanmean(z[stk])), pick_types=dict(zip(*np.unique(z['types'][np.arange(len(s)), z[f'pick_{arm}']], return_counts=True))))
            adv[arm]['pick_types'] = {k: int(v) for k, v in adv[arm]['pick_types'].items()}
        adv['n_episodes'] = int(len(z['n_cand'])); adv['mean_candidates'] = f(z['n_cand'].mean()); adv['max_cmi'] = f(np.nanmax(z['max_cmi']))
        F4['adversarial'] = adv
        F4['clause2_fires'] = bool('FR' in adv and adv['FR']['ci'][0] > 0)
        json.dump(adv, open(f'{CH}/adversarial.json', 'w'), indent=1)
    else:
        F4['adversarial'] = 'not run (judged on the FPR clause only)'; F4['clause2_fires'] = False
    F4['fires'] = bool(F4['clause1_fires'] or F4['clause2_fires'])
    F['F4'] = F4
    F['F5'] = 'untestable (S3 dropped)'
    if FRD:
        fd = np.mean([au(sc[r][:, K16], P, N) for r in FRD])
        F['F6'] = dict(fr_direct=f(fd), fr=f(fr_p), fires=bool(fd >= fr_p - .02))
    summ['falsifiers'] = F
    json.dump(fpr, open(f'{CH}/fpr_at_80tpr.json', 'w'), indent=1)

    # ---- reported, not part of PASS
    rep = {}
    inf = te['lab'] == 1
    rep['spearman_v64_cmi_informative'] = {a: f(np.mean([spearmanr(sc[r][inf, KS.index(64)] / 64, te['cmi'][inf]).correlation
                                                        for r in (FRS if a == 'FR' else [a])]))
                                           for a in ['FR'] + [b for b in base] + [x for x in ('L-all',) if x in sc and np.isfinite(sc[x][inf, 0]).all()]}
    if 'CONS' in sc:
        cg = sc['CONS'][:, 0]
        t3b = te['typ'] == 'T3b'
        rep['consumer_spearman_v16'] = {a: dict(all=f(np.mean([spearmanr(sc[r][:, K16], cg, nan_policy='omit').correlation for r in (FRS if a == 'FR' else [a])])),
                                                T3b=f(np.mean([spearmanr(sc[r][t3b, K16], cg[t3b], nan_policy='omit').correlation for r in (FRS if a == 'FR' else [a])])))
                                        for a in ['FR'] + base}
    if 'L-all' in sc:
        la = au(sc['L-all'][:, K16], P, N)
        rep['fraction_of_oracle_primary'] = f((fr_p - .5) / (la - .5))
        per_t = {}
        for t in A.HP + A.HN:
            pp, nn = (P & (te['typ'] == t), N) if t in A.HP else (P, N & (te['typ'] == t))
            per_t[t] = dict(fr=f(np.mean([au(s, pp, nn) for s in frk])), l_all=f(au(sc['L-all'][:, K16], pp, nn)),
                            l_train=f(au(sc['L-train'][:, K16], pp, nn)))
            per_t[t]['fr_minus_ltrain'] = f(per_t[t]['fr'] - per_t[t]['l_train'])
            per_t[t]['novelty_test'] = f(per_t[t]['l_all'] - per_t[t]['l_train'])
        rep['per_heldout_type'] = per_t
    if FRD:
        rep['fr_direct_minus_fr_primary'] = f(np.mean([au(sc[r][:, K16], P, N) for r in FRD]) - fr_p)
    cal = {}
    for a in FRS + (['C4'] if 'C4' in st else []):
        x = sc[a][:, K16] / 16; y = st[a][:, j16] / 16
        okc = np.isfinite(x) & np.isfinite(y)
        slope, icpt = np.polyfit(x[okc], y[okc], 1)
        r2 = np.corrcoef(x[okc], y[okc])[0, 1] ** 2
        cal[a] = dict(slope=f(slope), intercept=f(icpt), r2=f(r2))
        q = np.quantile(x[okc], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, x[okc], side='right') - 1, 0, 9)
        cal[a]['bins'] = [[f(x[okc][b == i].mean()), f(y[okc][b == i].mean()), int((b == i).sum())] for i in range(10) if (b == i).any()]
    rep['calibration_v16_vs_settle16'] = {a: {k: v for k, v in c.items() if k != 'bins'} for a, c in cal.items()}
    json.dump(cal, open(f'{CH}/calibration.json', 'w'), indent=1)
    comp = {}
    for fn in glob.glob(f'{OUT}/eval/test/*.npz'):
        z = np.load(fn)
        comp[os.path.basename(fn)[:-4]] = dict(n=int(len(z['idx'])), wall_s=f(float(z['wall_s'])), s_per_episode=f(float(z['wall_s']) / len(z['idx'])))
    e1f = sorted(glob.glob(f'{OUT}/eval/test/E1@*.npz'))
    if e1f:
        cs = sum(float(np.load(x)['x_calls_s'].max()) for x in e1f); cn = sum(int(np.load(x)['x_calls_n'].max()) for x in e1f)
        comp['_tabpfn'] = dict(calls=cn, s_per_call=f(cs / max(cn, 1)))
    comp['_f_meta'] = {r: v.get('flops') for r, v in pre.items()}
    rep['compute'] = comp
    summ['reported'] = rep

    # ---- verdict
    pas = bool((summ['gate'] or {}).get('gate_i_pass') and summ['precondition']['met'] and ra['pass'] and rb['pass']
               and isinstance(rc, dict) and rc.get('pass') and not any(F[k]['fires'] for k in ('F1', 'F2', 'F3', 'F4', 'F6') if isinstance(F.get(k), dict)))
    summ['PASS'] = pas
    fired = [k for k in ('F1', 'F2', 'F3', 'F4', 'F6') if isinstance(F.get(k), dict) and F[k]['fires']]
    summ['falsifiers_fired'] = fired
    if not summ['precondition']['met']:
        summ['verdict'] = 'inconclusive (undertrained): precondition not met'
    elif pas:
        summ['verdict'] = 'PASS' + (' (large margin)' if ra['large_margin'] else '')
    else:
        fails = [x for x, v in (('a', ra['pass']), ('b', rb['pass']), ('c', isinstance(rc, dict) and rc.get('pass'))) if not v]
        summ['verdict'] = 'FAIL: ' + ', '.join([f'({x}) fails' for x in fails] + [f'{x} fires' for x in fired])
    json.dump(summ, open(f'{RES}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps({k: summ[k] for k in ('verdict', 'PASS', 'falsifiers_fired', 'a', 'b', 'c')}, indent=1, default=str))


if __name__ == '__main__':
    main()
