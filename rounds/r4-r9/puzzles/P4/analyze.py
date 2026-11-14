"""P4 analysis: stage 1 (R1 fidelity), stage 2 (R2 curves), noise check, stage 3 (selection + fine-tunes).

  python analyze.py      # reads results/, writes results/summary.json, rows_r1.json, rows_ft.json, charts/*.json
Runs on partial results (missing parts are reported as missing).
"""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr, kendalltau, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
S3 = os.path.join(RES, 'stage3')
VM = os.path.join(S3, 'vm')
MR = os.path.join(HERE, '..', '..', 'market-resnet', 'results', 'pilot')
SHAP_K = (1, 2, 3, 4, 6, 8, 16, 32, 64)
AS_M = (5, 10, 20, 50)
ARMS = ['F0', 'F-AS-mkt', 'F-AS-aux', 'F-LOO-mkt', 'F-rand-mkt']


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).correlation)


def tau(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(kendalltau(a, b).statistic)


def top1(rule, ref, tol=0.01):
    rule, ref = np.asarray(rule, float), np.asarray(ref, float)
    return float(int(np.argmax(rule)) in set(np.flatnonzero(ref >= ref.max() - tol)))


def cboot(x, cl, B=4000, alpha=0.05, seed=0):
    """Mean of rows x with a cluster bootstrap CI (resampling clusters cl)."""
    x = np.nan_to_num(np.asarray(x, float))
    cl = np.asarray(cl)
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, n=0, nclu=0)
    u = sorted(set(cl.tolist()))
    idx = [np.flatnonzero(cl == c) for c in u]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        rows = np.concatenate([idx[i] for i in pick])
        bs.append(x[rows].mean())
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return dict(mean=round(float(x.mean()), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=len(x), nclu=len(u))


def tci(x, alpha=0.05):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    if len(x) == 1:
        return dict(mean=round(float(x[0]), 4), lo=None, hi=None, n=1)
    h = tdist.ppf(1 - alpha / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(float(x.mean()), 4), lo=round(float(x.mean() - h), 4), hi=round(float(x.mean() + h), 4),
                n=len(x), sd=round(float(x.std(ddof=1)), 4))


def same_game(rule):
    for g in ('SOFT', 'PAY'):
        if f'-{g}' in rule or rule.startswith(g):
            return g.lower()
    return 'flip'


def load_nets():
    out = []
    for f in sorted(glob.glob(os.path.join(RES, 'nets', '*.json'))):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- stage 1
def kmatched(n, budget=150):
    ks = [k for k in SHAP_K if k * (n - 1) + 2 <= budget]
    return max(ks) if ks else 1


def stage1(nets):
    r1 = [d for d in nets if d['kind'] in ('r1', 'h1', 'h3')]
    pilot = [d for d in r1 if d['kind'] == 'r1' and d['seed'] == 0 and d['circuit_seed'] < 5]
    acc = [d for d in r1 if d['accepted']]
    rows = []
    for d in acc:
        clu = f"{d['kind']}{d['circuit_seed']}"
        for L in d['levels']:
            ref = {g: np.array(L['ref'][g]) for g in ('flip', 'soft', 'pay')}
            prog = np.array(L['program']['shap'])
            rules = dict(L['rules'])
            km = kmatched(L['n'])
            rules['FLIP-SHAP-Kmatched'] = rules[f'FLIP-SHAP-K{km}']
            for name, v in rules.items():
                g = same_game(name)
                rows.append(dict(net=d['job'], cluster=clu, kind=d['kind'], level=L['level'], n=L['n'], rule=name,
                                 rho_flip=rho(v, ref['flip']), rho_same=rho(v, ref[g]), top1_flip=top1(v, ref['flip']),
                                 tau_flip=tau(v, ref['flip']), rho_prog=rho(v, prog)))
    by = {}
    for r in rows:
        by.setdefault(r['rule'], []).append(r)
    summ = {}
    for rule, rs in by.items():
        cl = [r['cluster'] for r in rs]
        summ[rule] = dict(rho_flip=cboot([r['rho_flip'] for r in rs], cl),
                          rho_same=cboot([r['rho_same'] for r in rs], cl),
                          top1_flip=round(float(np.mean([r['top1_flip'] for r in rs])), 4),
                          rho_prog=cboot([r['rho_prog'] for r in rs], cl),
                          nan=int(sum(np.isnan(r['rho_flip']) for r in rs)))
    out = dict(n_nets=len(r1), n_accepted=len(acc), n_rows_level=len({(r['net'], r['level']) for r in rows}),
               pilot_pass=f"{sum(d['accepted'] for d in pilot)}/{len(pilot)}",
               accepted={d['job']: [d['accepted'], d['iia']['iia'], d['iia']['strict_iia']] for d in r1},
               rules=summ)
    # S1 checks for AS-SOFT-m50 and AS-PAY-m50
    key = {(r['net'], r['level']): r for r in rows if r['rule'] == 'FLIP-SHAP-Kmatched'}
    s1 = {}
    for asr in ('AS-SOFT-m50', 'AS-PAY-m50'):
        rs = by.get(asr, [])
        if not rs:
            continue
        diff = [np.nan_to_num(r['rho_flip']) - np.nan_to_num(key[(r['net'], r['level'])]['rho_flip']) for r in rs]
        cl = [r['cluster'] for r in rs]
        d90 = cboot(diff, cl, alpha=0.10)
        m = summ[asr]
        s1[asr] = dict(rho=m['rho_flip'], top1=m['top1_flip'], diff_vs_matched_shap=d90,
                       pass_rho=bool(m['rho_flip']['mean'] >= 0.85 and m['rho_flip']['lo'] >= 0.80),
                       pass_top1=bool(m['top1_flip'] >= 0.80),
                       pass_tost=bool(d90['lo'] >= -0.05 and d90['hi'] <= 0.05),
                       falsifier1=bool(d90['mean'] < -0.10 or m['top1_flip'] < 0.60))
    # H1 level 3: p (3 copies) ranks first?
    h1 = []
    for d in acc:
        if d['kind'] != 'h1':
            continue
        L = [x for x in d['levels'] if x['level'] == 3][0]
        pl, ps = L['plants'], L['plant_scores']
        cp = pl['copies']
        oth = [k for k in range(L['n']) if k not in cp]
        e = dict(net=d['job'])
        for asr in ('AS-SOFT-m50', 'AS-PAY-m50', 'FLIP-SHAP-exact', 'FLIP-LOO'):
            v = np.array(L['rules'][asr])
            e[asr + '_summed_first'] = bool(v[cp].sum() > v[oth].max())
        # joint intervention on the net (exact joint flip-Shapley of the 3 copies as one player);
        # for AS the joint player's score equals the summed score (AS is additive over channels)
        for g in ('flip', 'soft', 'pay'):
            e[f'exact_{g}_joint_first'] = bool(ps[g]['joint_shap'] > max(ps[g]['others_shap']))
            e[f'exact_{g}_joint'] = ps[g]['joint_shap']
            e[f'exact_{g}_joint_loo'] = ps[g]['joint_loo']
        e['names'] = L['names']
        e['AS-SOFT-m50'] = L['rules']['AS-SOFT-m50']
        e['FLIP-LOO'] = L['rules']['FLIP-LOO']
        h1.append(e)
    out['h1_level3'] = h1
    s1['h1_pass'] = bool(h1) and all(e['AS-SOFT-m50_summed_first'] and e['exact_flip_joint_first'] for e in h1)
    s1['pass_any'] = any(s1[k]['pass_rho'] and s1[k]['pass_top1'] and s1[k]['pass_tost'] for k in s1 if k.startswith('AS')) \
        and s1['h1_pass']
    out['S1'] = s1
    # completeness, D(t), efficiency
    comp = {k: [L['completeness'][k] for d in acc for L in d['levels']] for k in ('soft-m50', 'pay-m50', 'soft-m20', 'pay-m20')}
    out['completeness'] = {k: dict(mean=round(float(np.mean(v)), 5), max=round(float(np.max(v)), 5)) for k, v in comp.items() if v}
    out['Dt'] = {g: [round(float(np.mean([L['Dt'][g][i] for d in acc for L in d['levels']])), 4) for i in range(3)]
                 for g in ('soft', 'pay')} if acc else None
    out['eff_gap_max'] = max((abs(v) for d in acc for L in d['levels'] for v in L['eff_gap'].values()), default=None)
    # roles: normalized share of |score| by role (gate/copy/epi/inert)
    roles = {}
    for rule in ('FLIP-SHAP-exact', 'AS-SOFT-m50', 'AS-PAY-m50', 'FLIP-LOO', 'GRADxACT-SOFT', 'MI-1BIT'):
        acc_r = {}
        for d in acc:
            for L in d['levels']:
                v = np.array(L['rules'][rule])
                s = np.abs(v).sum()
                if s == 0:
                    continue
                for k, ro in enumerate(L['roles']):
                    acc_r.setdefault(ro, []).append(v[k] / s)
        roles[rule] = {ro: round(float(np.mean(x)), 4) for ro, x in acc_r.items()}
    out['role_share'] = roles
    # noise floor: exact flip-Shapley of seed-0 vs seed-1 nets of the same circuit; net vs program
    pairs = {}
    for d in acc:
        if d['kind'] in ('r1', 'h1', 'h3'):
            pairs.setdefault((d['kind'], d['circuit_seed']), {})[d['seed']] = d
    nf = []
    for k, ds in pairs.items():
        if 0 in ds and 1 in ds:
            for L0, L1 in zip(ds[0]['levels'], ds[1]['levels']):
                nf.append(rho(L0['ref']['flip'], L1['ref']['flip']))
    out['noise_floor_seed_rho'] = tci(nf)
    out['net_vs_program_rho'] = summ.get('FLIP-SHAP-exact', {}).get('rho_prog')
    # accuracy vs cost ladder (rho to the exact flip-Shapley, P=2048), rule families
    lad = {}
    for d in acc:
        for L in d['levels']:
            ref = L['ref']['flip']
            for e in L['ladder']:
                if e['P'] != 2048:
                    continue
                lad.setdefault((e['method'], e['param'], e['cost_pair'] if e['method'].endswith('SHAP') else e['cost_pair']), []).append(
                    np.nan_to_num(rho(e['est'], ref)))
    out['ladder_P2048'] = sorted([dict(method=k[0], param=k[1], cost_pair_mean=k[2], rho_flip=round(float(np.mean(v)), 4), n=len(v))
                                  for k, v in lad.items()], key=lambda r: (r['method'], r['param']))
    return out, rows


# ---------------------------------------------------------------- noise check
def noise(nets):
    ns = [d for d in nets if d['kind'] == 'noise']
    out = dict(n=len(ns), accepted=sum(d['accepted'] for d in ns), calibrated=sum(d.get('calibrated', False) for d in ns),
               calib_mean_p={d['job']: d.get('calib_mean_p') for d in ns})
    ok = [d for d in ns if d['accepted'] and d.get('calibrated')]
    out['used'] = len(ok)
    comb = {}
    for d in (ok if ok else [d for d in ns if d['accepted']]):
        for L in d['levels']:
            u, r = L['program']['u'], L['program']['r']
            for rule in ('AS-PAY-m50', 'AS-SOFT-m50', 'PAY-SHAP-exact', 'FLIP-SHAP-exact', 'FLIP-LOO'):
                v = L['rules'][rule]
                comb.setdefault(f'{rule}~u', []).append((d['job'], np.nan_to_num(rho(v, u))))
                comb.setdefault(f'{rule}~r', []).append((d['job'], np.nan_to_num(rho(v, r))))
            comb.setdefault('u~r', []).append((d['job'], np.nan_to_num(rho(u, r))))
    out['rho'] = {k: cboot([x for _, x in v], [c for c, _ in v]) for k, v in comb.items()}
    if comb:
        dp = [a - b for (_, a), (_, b) in zip(comb['AS-PAY-m50~u'], comb['AS-PAY-m50~r'])]
        ds = [a - b for (_, a), (_, b) in zip(comb['AS-SOFT-m50~r'], comb['AS-SOFT-m50~u'])]
        cl = [c for c, _ in comb['AS-PAY-m50~u']]
        out['pay_tracks_u_minus_r'] = cboot(dp, cl)
        out['soft_tracks_r_minus_u'] = cboot(ds, cl)
        out['gate_ok'] = bool(ok)
    return out


# ---------------------------------------------------------------- stage 2
def s2_curve(asr, ladder, ref, n):
    """AS within 0.02 of its asymptotic rho: cost 3m; FLIP-SHAP cost for the same rho (ladder)."""
    ms = sorted(asr)
    r_as = {m: np.nan_to_num(rho(asr[m], ref)) for m in ms}
    asym = r_as[ms[-1]]
    mstar = min(m for m in ms if r_as[m] >= asym - 0.02)
    target = r_as[mstar]
    r_sh = [(e['K'], e['cost_pair'], np.nan_to_num(rho(e['est'], ref))) for e in ladder]
    hit = [c for K, c, r in r_sh if r >= target]
    kmax = max(K for K, _, _ in r_sh)
    cost_sh = min(hit) if hit else None
    bound = (kmax * 2) * (n - 1) + 2       # censored: needs more than the largest ladder K
    ratio = (3 * mstar) / (cost_sh if cost_sh else bound)
    return dict(rho_as={int(m): round(float(v), 4) for m, v in r_as.items()}, m_star=int(mstar), cost_as=3 * mstar,
                target=round(float(target), 4),
                rho_shap={int(K): round(float(r), 4) for K, _, r in r_sh}, cost_shap=cost_sh, censored=cost_sh is None,
                ratio=round(float(ratio), 4), pass_=bool(ratio <= 1 / 3))


def stage2(nets):
    r2 = [d for d in nets if d['kind'] == 'r2']
    nl = [d for d in nets if d['kind'] == 'null']
    out = dict(n_r2=len(r2), n_null=len(nl))
    if not r2:
        return out
    ka = lambda ds, k: [float(np.mean(d['align'][k])) if isinstance(d['align'][k], list) else d['align'][k] for d in ds]
    al = {}
    for k in ('code_best_kappa', 'share_codes_aligned', 'share_gates_recovered', 'share_codes_input'):
        a, b = ka(r2, k), ka(nl, k)
        rng = np.random.default_rng(0)
        bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(4000)] if a and b else [0]
        al[k] = dict(r2=round(float(np.mean(a)), 4), null=round(float(np.mean(b)), 4) if b else None,
                     diff_lo=round(float(np.quantile(bs, 0.025)), 4), diff_hi=round(float(np.quantile(bs, 0.975)), 4))
    out['alignment'] = al
    out['alignment_above_null'] = bool(al['code_best_kappa']['diff_lo'] > 0)
    out['train_acc'] = tci([d['train_acc'] for d in r2])
    cur = {}
    for d in r2:
        for rule, c in d['r2']['curves'].items():
            cur.setdefault(rule, []).append(c['mean4'])
    out['mean4'] = {k: tci(v) for k, v in cur.items()}
    pa = [d['r2']['curves']['AS-SOFT-m50']['mean4'] - d['r2']['curves']['FLIP-LOO']['mean4'] for d in r2]
    out['AS_minus_LOO_mean4'] = tci(pa)
    out['expected_AS_ge_LOO_plus_0.05'] = bool(np.mean(pa) >= 0.05)
    out['falsifier5_LOO_ge_AS'] = bool(np.mean(pa) <= 0)
    rr = {}
    for d in r2:
        R = d['r2']['rules']
        ref = R['FLIP-SHAP-ref']
        for rule in ('AS-SOFT-m50', 'AS-PAY-m50', 'FLIP-LOO', 'FLIP-SHAP-K2', 'FLIP-SHAP-K8', 'GRADxACT-SOFT', 'MI-1BIT', 'RANDOM'):
            rr.setdefault(rule, []).append(np.nan_to_num(rho(R[rule], ref)))
    out['rho_vs_flipshap_ref'] = {k: tci(v) for k, v in rr.items()}
    s2 = [s2_curve({m: d['r2']['rules'][f'AS-SOFT-m{m}'] for m in AS_M}, d['r2']['shap_ladder']['flip'],
                   d['r2']['rules']['FLIP-SHAP-ref'], d['r2']['n']) for d in r2]
    out['S2_r2'] = dict(per_net=s2, pass_share=round(float(np.mean([s['pass_'] for s in s2])), 3),
                        median_ratio=round(float(np.median([s['ratio'] for s in s2])), 4))
    out['completeness'] = {k: round(float(np.mean([d['r2']['completeness'][k] for d in r2])), 5)
                           for k in r2[0]['r2']['completeness']}
    return out


# ---------------------------------------------------------------- stage 3
def targets():
    t = {}
    for b in ('A', 'Bds'):
        f = os.path.join(MR, f'pilot_{b}.eval.json')
        if os.path.exists(f):
            t[b] = json.load(open(f))['sets']['L64']['fixed'][-1]['acc']
    return t


def stage3():
    out = {}
    tg = targets()
    base = {}
    for b in ('A', 'Bds'):
        f = os.path.join(VM, 'base', f'{b}.json')
        if os.path.exists(f):
            e = json.load(open(f))['eval']
            base[b] = {k: v['acc'] for k, v in e.items()}
    out['base'] = base
    out['repro'] = {b: dict(target=tg.get(b), got=base.get(b, {}).get('L64'),
                            ok=(abs(base[b]['L64'] - tg[b]) <= 0.01) if b in base and b in tg else None) for b in ('A', 'Bds')}
    cpu = os.path.join(S3, 'repro_cpu.json')
    if os.path.exists(cpu):
        out['repro_cpu'] = json.load(open(cpu))
    picks = json.load(open(os.path.join(S3, 'picks.json'))) if os.path.exists(os.path.join(S3, 'picks.json')) else {}
    out['picks_cpu'] = picks
    # selection (GPU, P=512)
    sel = {}
    for f in sorted(glob.glob(os.path.join(VM, 'select', '*.json'))):
        s = json.load(open(f))
        R = {k: np.array(v) for k, v in s['rules'].items()}
        rk = [k for k in R if k.startswith('FLIP-SHAP-ref')]
        kk = [k for k in R if k.startswith('FLIP-SHAP-K')]
        if not rk:
            continue
        Kref, Klad = int(rk[0].split('ref')[1]), int(kk[0].split('-K')[1])
        ref = R[rk[0]]
        R['FLIP-SHAP-512'] = (Klad * R[kk[0]] + Kref * ref) / (Klad + Kref)
        n = len(ref)
        tb = np.random.default_rng(7).permutation(n)
        pk = lambda v: sorted(range(n), key=lambda i: (-v[i], tb[i]))[:2]
        sels = {'AS-SOFTFLIP': 'AS-SOFT-m50', 'AS-PAY': 'AS-PAY-m50', 'FLIP-LOO': 'FLIP-LOO', 'FLIP-SHAP': 'FLIP-SHAP-512',
                'GRADxACT': 'GRADxACT-SOFT', 'RANDOM': 'RANDOM'}
        e = dict(t=s['t'], P=s['P'], completeness=s['completeness'], vC=s['vC'], v0=s['v0'], sec=s['sec'])
        e['picks'] = {k: [int(i) for i in pk(R[v])] for k, v in sels.items()}
        e['tied'] = {k: s['tied'].get(v) for k, v in sels.items()}
        # rank agreement with the 448 reference orders (independent of the 64 ladder orders)
        e['rho_vs_ref448'] = {k: round(np.nan_to_num(rho(R[v], ref)), 4) for k, v in sels.items() if v != 'FLIP-SHAP-512'}
        e['rho_vs_ref448']['FLIP-SHAP-K64'] = round(np.nan_to_num(rho(R[kk[0]], ref)), 4)
        e['top1_vs_ref448'] = {k: top1(R[v], ref, tol=0.0) for k, v in sels.items() if v != 'FLIP-SHAP-512'}
        e['scores'] = {k: [round(float(x), 5) for x in R[v]] for k, v in sels.items()}
        e['S2'] = s2_curve({int(k.split('-m')[1]): R[k] for k in R if k.startswith('AS-SOFT-m')}, s['shap_ladder']['flip'], ref, n)
        b = os.path.basename(f)[:-5]
        cb = b.split('_')[0]
        if cb in picks and s['t'] == 16:
            e['cpu_AS_picks'] = picks[cb]['AS']
            e['cpu_gpu_AS_overlap'] = len(set(picks[cb]['AS']) & set(e['picks']['AS-SOFTFLIP']))
        sel[b] = e
    out['select'] = sel
    if sel:
        out['S2_market'] = dict(pass_share=round(float(np.mean([e['S2']['pass_'] for e in sel.values()])), 3),
                                ratios={k: e['S2']['ratio'] for k, e in sel.items()})
    # fine-tunes
    rows = []
    for f in sorted(glob.glob(os.path.join(VM, 'ft', '*.json'))):
        d = json.load(open(f))
        ev = d['eval']
        vr = d['var_ratio']
        rows.append(dict(base=d['base_arm'], arm=d['arm'], seed=d['seed'], groups=d['groups'],
                         L64=ev.get('L64', {}).get('acc'), L128=ev.get('L128', {}).get('acc'),
                         L256=ev.get('L256', {}).get('acc'), id_8_32=ev.get('id_8_32', {}).get('acc'),
                         primary=d['primary'], collapse=d['collapse'],
                         min_var_ratio_chosen=min((vr[str(g)] for g in d['groups']), default=None),
                         final_loss=d['log'][-1].get('final') if d['log'] else None, tied_to=None, sec=d['sec']))
    # ties: an arm whose groups equal the AS groups was not rerun; it takes the F-AS-mkt result
    have = {(r['base'], r['arm'], r['seed']) for r in rows}
    for b, p in picks.items():
        for r in [r for r in rows if r['base'] == b and r['arm'] == 'F-AS-mkt']:
            s = r['seed']
            if sorted(p['LOO']) == sorted(p['AS']) and (b, 'F-LOO-mkt', s) not in have:
                rows.append(dict(r, arm='F-LOO-mkt', tied_to='F-AS-mkt'))
            if sorted(p['rand'][str(s)]) == sorted(p['AS']) and (b, 'F-rand-mkt', s) not in have:
                rows.append(dict(r, arm='F-rand-mkt', tied_to='F-AS-mkt'))
    out['n_ft'] = len(rows)
    tab = {(r['base'], r['arm'], r['seed']): r for r in rows}
    arm_s = {}
    for a in ARMS:
        for b in ('A', 'Bds'):
            rs = [r for r in rows if r['arm'] == a and r['base'] == b]
            arm_s[f'{b}/{a}'] = {k: tci([r[k] for r in rs]) for k in ('primary', 'L64', 'L128', 'L256', 'id_8_32')}
    out['arms'] = arm_s
    f0sd = {b: tci([r['L128'] for r in rows if r['arm'] == 'F0' and r['base'] == b]).get('sd') for b in ('A', 'Bds')}
    out['F0_L128_sd'] = f0sd
    out['F0_primary_sd'] = {b: tci([r['primary'] for r in rows if r['arm'] == 'F0' and r['base'] == b]).get('sd') for b in ('A', 'Bds')}

    def paired(a1, a2, key='primary'):
        ks = sorted({(b, s) for (b, a, s) in tab if a == a1} & {(b, s) for (b, a, s) in tab if a == a2})
        dd = [tab[(b, a1, s)][key] - tab[(b, a2, s)][key] for b, s in ks
              if tab[(b, a1, s)][key] is not None and tab[(b, a2, s)][key] is not None]
        r = tci(dd)
        r90 = tci(dd, alpha=0.10)
        r['lo90'], r['hi90'] = r90.get('lo'), r90.get('hi')
        return r

    cmp = {f'{x}-{y}': paired(x, y) for x, y in [('F-AS-mkt', 'F0'), ('F-AS-mkt', 'F-rand-mkt'), ('F-AS-aux', 'F-AS-mkt'),
                                                  ('F-LOO-mkt', 'F-AS-mkt'), ('F-rand-mkt', 'F0'), ('F-AS-aux', 'F0'),
                                                  ('F-LOO-mkt', 'F0')]}
    cmp.update({f'{x}-{y} [{k}]': paired(x, y, k) for x, y in [('F-AS-mkt', 'F0'), ('F-AS-mkt', 'F-rand-mkt')]
                for k in ('L64', 'L128', 'L256')})
    out['paired'] = cmp
    a0, ar = cmp['F-AS-mkt-F0'], cmp['F-AS-mkt-F-rand-mkt']
    if a0['n'] and ar['n'] and a0['lo'] is not None and ar['lo'] is not None:
        out['S3'] = dict(pass_=bool(a0['mean'] >= 0.05 and a0['lo'] > 0 and ar['mean'] >= 0.05 and ar['lo'] > 0))
        mk = [cmp[k]['mean'] for k in ('F-AS-mkt-F0', 'F-rand-mkt-F0', 'F-LOO-mkt-F0') if cmp[k]['mean'] is not None]
        out['falsifiers'] = dict(
            f2_rand_equiv_while_AS_beats_F0=bool(ar['lo90'] is not None and ar['lo90'] >= -0.02 and ar['hi90'] <= 0.02 and a0['lo'] > 0),
            f3_all_markets_le_F0_plus_2=bool(mk and max(mk) <= 0.02),
            f4_aux_ge_mkt=bool(cmp['F-AS-aux-F-AS-mkt']['mean'] is not None and cmp['F-AS-aux-F-AS-mkt']['mean'] >= 0),
            readout_LOO_within_2=bool(cmp['F-LOO-mkt-F-AS-mkt']['mean'] is not None and cmp['F-LOO-mkt-F-AS-mkt']['mean'] >= -0.02))
    out['collapse_any'] = [dict(base=r['base'], arm=r['arm'], seed=r['seed'], min_var=r['min_var_ratio_chosen'])
                           for r in rows if r['collapse']]
    out['invalid_F0_sd_gt_5'] = bool(any(v is not None and v > 0.05 for v in list(f0sd.values()) + list(out['F0_primary_sd'].values()))
                                     and len({r['seed'] for r in rows if r['arm'] == 'F0'}) >= 5)
    return out, rows


def charts(s1, s3):
    os.makedirs(os.path.join(HERE, 'charts'), exist_ok=True)
    rules = ['AS-SOFT-m50', 'AS-PAY-m50', 'AS-SOFT-m20', 'EG-SOFT-8', 'FLIP-SHAP-Kmatched', 'FLIP-SHAP-K4', 'FLIP-LOO',
             'FLIP-LOO-mean', 'GRADxACT-SOFT', 'MI-1BIT', 'VINFO-LOGISTIC', 'WNORM', 'RANDOM']
    R = s1.get('rules', {})
    c1 = dict(id='p4_r1_rho', kind='bar_ci', title='R1: rank agreement with the net\'s exact flip-Shapley (within level)',
              unit='Spearman rho (95% circuit-clustered CI)', source='puzzles/P4/results/summary.json',
              groups=[r for r in rules if r in R],
              series={'rho': [R[r]['rho_flip']['mean'] for r in rules if r in R],
                      'lo': [R[r]['rho_flip']['lo'] for r in rules if r in R],
                      'hi': [R[r]['rho_flip']['hi'] for r in rules if r in R],
                      'top1': [R[r]['top1_flip'] for r in rules if r in R]})
    json.dump(c1, open(os.path.join(HERE, 'charts', 'p4_r1_rho.json'), 'w'), indent=1)
    A = s3.get('arms', {})
    c2 = dict(id='p4_ft_primary', kind='grouped_bar', title='Market-resnet fine-tunes: extrapolation accuracy (mean of L64, L128)',
              unit='accuracy (mean over seeds)', source='puzzles/P4/results/summary.json', groups=ARMS,
              series={b: [A.get(f'{b}/{a}', {}).get('primary', {}).get('mean') for a in ARMS] for b in ('A', 'Bds')})
    json.dump(c2, open(os.path.join(HERE, 'charts', 'p4_ft_primary.json'), 'w'), indent=1)


def main():
    nets = load_nets()
    st0 = json.load(open(os.path.join(RES, 'stage0.json'))) if os.path.exists(os.path.join(RES, 'stage0.json')) else {}
    s0 = dict(theorems=st0.get('theorems'), agreement_gates_only=st0.get('agreement_gates_only'),
              top1_shap_vs_reach_gates=st0.get('top1_shap_vs_reach_gates'), H1_level3=st0.get('H1_level3'),
              H3_agree=st0.get('H3_agree'), n_circuits=st0.get('n_circuits'))
    s1, rows1 = stage1(nets)
    nz = noise(nets)
    s2 = stage2(nets)
    s3, rows3 = stage3()
    summ = dict(stage0=s0, stage1=s1, noise=nz, stage2=s2, stage3=s3)
    # invalidity conditions
    pp = s1['pilot_pass'].split('/')
    summ['invalid'] = dict(
        pilot_lt_4_of_5=bool(int(pp[1]) >= 5 and int(pp[0]) < 4),
        completeness_gt_2pct_m50=bool(s1.get('completeness') and min(s1['completeness']['soft-m50']['mean'],
                                                                       s1['completeness']['pay-m50']['mean']) > 0.02),
        r2_alignment_not_above_null=(not s2.get('alignment_above_null')) if 'alignment' in s2 else None,
        repro_miss=[b for b, v in s3['repro'].items() if v['ok'] is False],
        f0_sd_gt_5=s3.get('invalid_F0_sd_gt_5'))
    json.dump(summ, open(os.path.join(RES, 'summary.json'), 'w'), indent=1, default=float)
    json.dump(rows1, open(os.path.join(RES, 'rows_r1.json'), 'w'), default=float)
    json.dump(rows3, open(os.path.join(RES, 'rows_ft.json'), 'w'), indent=0, default=float)
    charts(s1, s3)
    return summ


if __name__ == '__main__':
    s = main()
    s1 = s['stage1']
    print('stage1', s1['n_accepted'], '/', s1['n_nets'], 'pilot', s1['pilot_pass'])
    for k in ('AS-SOFT-m50', 'AS-PAY-m50', 'FLIP-SHAP-Kmatched', 'FLIP-SHAP-K4', 'FLIP-LOO', 'GRADxACT-SOFT', 'MI-1BIT', 'RANDOM'):
        if k in s1['rules']:
            print(f"  {k:20s} rho {s1['rules'][k]['rho_flip']}  top1 {s1['rules'][k]['top1_flip']}")
    print('S1', json.dumps(s1['S1'])[:1500])
    print('completeness', s1.get('completeness'), 'Dt', s1.get('Dt'))
    print('noise', json.dumps(s['noise'].get('rho', {}))[:600])
    print('stage2', json.dumps({k: v for k, v in s['stage2'].items() if k not in ('S2_r2', 'mean4')})[:1200])
    print('stage3', json.dumps({k: v for k, v in s['stage3'].items() if k not in ('select', 'arms')})[:2000])
    print('invalid', s['invalid'])
