"""Tables for the phase-R report from results/*.json. usage: python analyze.py > results/tables.md
Post-review version (2026-10-29, PREREG-R amendment 2). Pre-review tables: results/prereview/tables.md."""
import glob, json, math, os
import numpy as np
from scipy.stats import norm as _n, beta as _B

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
LOG2 = math.log(2)


def load(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(RES, pat))):
        try:
            r = json.load(open(p)); r['_path'] = p; out.append(r)
        except Exception:
            pass
    return out


def f(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return '–'
    if isinstance(x, float) and math.isinf(x):
        return 'inf'
    return f'{x:.{nd}f}'


def regime(r):
    return f"{r['act']} {r['g']:g}"


def cp_lower(k, n, alpha):
    return 0.0 if k <= 0 else float(_B.ppf(alpha, k, n - k + 1))


def dinf_lcb(s, alpha=0.05):
    """Held-out D_inf lower confidence bound (bits) at level alpha from a stored rel_stats dict."""
    pl = cp_lower(s['Dinf_k'], s['Dinf_n'], alpha)
    return max(0.0, (s['Dinf_tau'] + math.log(pl)) / LOG2) if pl > 0 else 0.0


def eps_lcb(s, alpha):
    """Lower confidence bound on D_inf = c_l (bits) at level alpha: the larger of the bootstrap KL bound (D_inf >= KL) and
    the held-out Markov bound, each at alpha / 2 (union bound)."""
    q = s.get('KL_lcb_q', {})
    key = min(q, key=lambda k: abs(float(k) - alpha / 2)) if q else None
    kl = q[key] if key is not None and abs(float(key) - alpha / 2) < 1e-4 else None
    return max(kl if kl is not None else 0.0, dinf_lcb(s, alpha / 2), 0.0)


def seeds_str(vals, nd=1):
    return ' / '.join(f(v, nd) for v in vals)


# ---------------------------------------------------------------- Task 1
def score_tables():
    C = load('scorecb_*.json')
    print('## Task 1: forward chain, lower confidence bounds (review fix 1) and weight seeds (fix 4)\n')
    print('q* = copula code fitted to fresh samples of the same K q_{l-1}. KL LCB = one-sided 95% bootstrap lower bound on '
          'E_Kq[log q*/q] <= KL(Kq||q) (2,000 resamples of the sampled paths; approximate). D_inf LCB = the larger of the KL '
          'bound (D_inf >= KL) and the Markov bound tau + log of the exact Clopper-Pearson lower bound on Kq(log q*/q >= tau), '
          'tau chosen on the other half of the samples, each at 97.5%. env = rigorous upper bound on c_1 for the Gaussian chain. '
          'Values per weight seed (0 / 1 / 2). '
          'Bits per layer; per neuron in brackets (seed mean). Layer 1, where K q_0 = p_1 for every family.\n')
    print('| regime | d | Gaussian KL LCB | Gaussian D_inf LCB | product code KL LCB | know-nothing KL LCB | env (Gaussian) |')
    print('|---|---|---|---|---|---|---|')
    width = {}
    for reg in ('tanh 0.25', 'tanh 0.75', 'tanh 3', 'relu 1.414'):
        for d in (8, 16, 32, 64, 256):
            rs = sorted([r for r in C if regime(r) == reg and r['d'] == d and r['L'] <= 4 and r['sigma'] == 0.3], key=lambda r: r['seed'])
            if not rs: continue
            l1 = [r['layers'][0] for r in rs]
            G = [x['G']['KL_lcb'] for x in l1]; Gd = [eps_lcb(x['G'], 0.05) for x in l1]
            NC = [x['NC']['KL_lcb'] for x in l1]; KN = [x['KN']['KL_lcb'] for x in l1]; env = [x['env_c'] for x in l1]
            width[(reg, d)] = dict(G=G, Gd=Gd, NC=NC, KN=KN, seeds=[r['seed'] for r in rs])
            print(f"| {reg} | {d} | {seeds_str(G)} ({np.mean(G) / d:.3f}) | {seeds_str(Gd)} | {seeds_str(NC)} ({np.mean(NC) / d:.3f}) | "
                  f"{seeds_str(KN)} ({np.mean(KN) / d:.3f}) | {seeds_str(env, 0)} |")
    print()
    print('### Per-neuron slopes between widths (Gaussian KL LCB, seed means): (KL(d2) - KL(d1)) / (d2 - d1)\n')
    for reg in ('tanh 0.25', 'tanh 0.75', 'tanh 3', 'relu 1.414'):
        ds = [d for d in (8, 16, 32, 64, 256) if (reg, d) in width]
        sl = [f"{d1}->{d2}: {(np.mean(width[(reg, d2)]['G']) - np.mean(width[(reg, d1)]['G'])) / (d2 - d1):.3f}" for d1, d2 in zip(ds, ds[1:])]
        print(f"- {reg}: " + ', '.join(sl))
    print()
    print('### Depth profiles (Gaussian chain, KL LCB per layer; D_inf LCB range over layers)\n')
    for reg in ('tanh 0.75', 'tanh 3', 'relu 1.414', 'tanh 0.25'):
        for d, L in ((32, 16), (64, 8), (256, 4), (256, 3), (64, 4), (32, 4), (16, 4), (8, 4)):
            rs = sorted([r for r in C if regime(r) == reg and r['d'] == d and r['L'] == L], key=lambda r: r['seed'])
            if not rs or (L == 4 and d < 256 and reg not in ('tanh 0.75', 'tanh 3')): continue
            for r in rs:
                kl = [x['G']['KL_lcb'] for x in r['layers']]; dd = [eps_lcb(x['G'], 0.05) for x in r['layers']]
                print(f"- {reg}, d = {d}, L = {L}, seed {r['seed']}: KL LCB " + ', '.join(f(x, 1) for x in kl) +
                      f" (range {min(kl):.1f}-{max(kl):.1f}); D_inf LCB {min(dd):.1f}-{max(dd):.1f}")
    print()
    print('### Sum over layers 1-3 of the Gaussian chain\'s D_inf lower bound (per layer the larger of the KL and Markov bounds, '
          'each at 5%/6, so the sum holds jointly at >= 95%)\n')
    print('| regime | d | per seed | old in-sample sum (seed 0, pre-review) |')
    print('|---|---|---|---|')
    old = {}
    for r in load('score_*.json'):
        if '_smc' in r['_path'] or 'G' not in r['layers'] or len(r['layers']['G']) < 3 or r['sigma'] != 0.3: continue
        old[(regime(r), r['d'], r['seed'])] = sum(max(x.get('rel_Dinf', 0), 0) for x in r['layers']['G'][:3])
    for reg in ('tanh 0.75', 'tanh 3', 'relu 1.414', 'tanh 0.25'):
        for d in (8, 16, 32, 64, 256):
            rs = sorted([r for r in C if regime(r) == reg and r['d'] == d and r['L'] in (3, 4) and len(r['layers']) >= 3], key=lambda r: r['seed'])
            if not rs: continue
            sums = [sum(eps_lcb(x['G'], 0.05 / 3) for x in r['layers'][:3]) for r in rs]
            print(f"| {reg} | {d} | {seeds_str(sums, 0)} | {f(old.get((reg, d, 0)), 0)} |")
    print()
    smc = load('score_*_smc.json')
    print('### Absolute per-layer estimates by tempered SMC for log Kq (d <= 32, L = 4, one seed; unchanged by the review). '
          'KLc = KL(Kq||q), KLw = KL(q||Kq) (over-estimate), c = coverage search (lower estimate). Bits.\n')
    print('| regime | sigma | d | layer | G KLc / KLw / c | GC KLc / KLw / c | KN KLc / KLw / c |')
    print('|---|---|---|---|---|---|---|')
    for r in sorted(smc, key=lambda x: (regime(x), x['sigma'], x['d'])):
        for li in range(len(r['layers']['G'])):
            cells = []
            for fam in ('G', 'GC', 'KN'):
                rows = r['layers'].get(fam, [])
                row = rows[li] if li < len(rows) else {}
                cells.append(f"{f(row.get('smc_KLc'))} / {f(row.get('smc_KLw'))} / {f(row.get('smc_c'))}" if 'smc_KLc' in row else '–')
            print(f"| {regime(r)} | {r['sigma']:g} | {r['d']} | {li + 1} | " + ' | '.join(cells) + ' |')
    print()


# ---------------------------------------------------------------- Tasks 2-3: rare events
def netkey(p, prefix):
    return os.path.basename(p)[len(prefix):-5]


def rare_bundle():
    out = []
    for r in load('rare_*.json'):
        k = netkey(r['_path'], 'rare_')
        b = dict(rare=r, key=k, noln='_noLN' in k)
        for pre in ('truthfz_', 'chainfz_', 'epslb_'):
            p = os.path.join(RES, f'{pre}{k}.json')
            b[pre[:-1]] = json.load(open(p)) if os.path.exists(p) else None
        out.append(b)
    return sorted(out, key=lambda b: (b['noln'], regime(b['rare']), b['rare']['d'], b['rare']['seed']))


def name_of(b):
    r = b['rare']
    nm = {'tanh 0.75': 'critical', 'tanh 3': 'chaotic', 'relu 1.414': 'ReLU', 'tanh 0.25': 'ordered'}[regime(r)]
    return f"{nm} d={r['d']}" + (f" seed {r['seed']}" if r['seed'] else '') + (' noLN' if b['noln'] else '')


def truth_of(b, i):
    """New reference value (frozen-tuning splitting) for level i, with the amendment-2 provisional flags."""
    lev = b['rare']['levels'][i]
    pub = lev.get('truth_hi', lev['truth'])
    tf = b['truthfz']
    if tf is None:
        return dict(logp=pub['logp'], rel_se=pub['rel_se'], provisional=True, flags=['not rerun'], pub=pub['logp'], new=False)
    s = tf['per_level'][i]; assert abs(s['t'] - lev['t']) < 1e-12
    flags = []
    if not s['rel_se'] <= 0.25: flags.append('relSE')
    if not abs(s['half_bits']) <= 1: flags.append('halves')
    bulk = (lev['crosscheck']['bulk_logp'] - s['logp']) / LOG2
    if not abs(bulk) <= 1: flags.append('bulk')
    dpub = (pub['logp'] - s['logp']) / LOG2
    if not abs(dpub) <= 1: flags.append('published')
    return dict(logp=s['logp'], rel_se=s['rel_se'], half=s['half_bits'], max_share=s['max_share'], bulk=bulk,
                bulk_rel_sd=lev['crosscheck']['bulk_rel_sd'], dpub=dpub, pub=pub['logp'], pub_rel_se=pub['rel_se'],
                provisional=bool(flags), flags=flags, new=True, R=s['R'])


def chain_of(b, i):
    lev = b['rare']['levels'][i]; cf = b['chainfz']
    if cf is None:
        return lev['chain']['log_KqLm1'], lev['chain'].get('log_KqLm1_k2'), None, None
    s = cf['per_level'][i]; assert abs(s['t'] - lev['t']) < 1e-12
    return s['log_KqLm1'], s['log_KqLm1_k2'], s['rel_se_KqLm1'], s['rel_se_KqLm1_k2']


def truth_table(B):
    print('## Review fix 2: reference probabilities, frozen-tuning splitting against the earlier values\n')
    print('New = fixed levels and pCN scales frozen from an independent pilot, one run per repeat to the deepest level '
          '(N and repeats in the last column). Differences in bits (other minus new). Flags (PREREG-R amendment 2): relSE = '
          'relative SE > 0.25; halves = the two halves of the repeats differ by > 1 bit; bulk = bulk-twisted-SMC cross-check '
          'differs by > 1 bit; published = the published (adaptive-tuning) value differs by > 1 bit. A flagged level is '
          'provisional.\n')
    print('| net | k | new log2 p (rel SE) | halves | largest repeat share | published - new | bulk - new (bulk rel SD) | flags | N x R |')
    print('|---|---|---|---|---|---|---|---|---|')
    for b in B:
        tf = b['truthfz']
        for i, lev in enumerate(b['rare']['levels']):
            t = truth_of(b, i)
            if not t['new']:
                print(f"| {name_of(b)} | {lev['k']} | not rerun | | | | | | |"); continue
            print(f"| {name_of(b)} | {lev['k']} | {t['logp'] / LOG2:.2f} ({t['rel_se']:.2f}) | {t['half']:+.2f} | {t['max_share']:.2f} | "
                  f"{t['dpub']:+.2f} | {t['bulk']:+.2f} ({t['bulk_rel_sd']:.2f}) | {','.join(t['flags']) or 'ok'} | {tf['N']} x {tf['R']} |")
    print()


def eps_same_chain(b, alpha_tot=0.05):
    """Per-layer eps for layers 1-3 of the SAME chain: Bonferroni D_inf LCB, search (d <= 32), envelope. Bits."""
    e = b['epslb']
    if e is None: return None
    L3 = e['layers'][:3]
    lb = [eps_lcb(x, alpha_tot / 3) for x in L3]
    out = dict(lb=lb, env=[x['env_c'] for x in L3])
    if all('search_c' in x for x in L3):
        out['search'] = [x['search_c'] for x in L3]
        out['opt'] = [max(s, l, 0.0) for s, l in zip(out['search'], lb)]
    return out


def rare_tables(B):
    import rnet as RN
    print('## Tasks 2-3: rare events (L = 4, sigma = 0.3), against the new reference values\n')
    print('Bits lost = log2(bound / p). [last] bound = 2^(eps_1 + eps_2 + eps_3) (K q_3)(A) with every eps measured on the '
          'same fitted chain whose (K q_3)(A) is used: search = SMC coverage search (lower estimate of c_l, uncertified; '
          'd <= 32), taken as max(search, LCB); LCB = lower confidence bounds on D_inf (per layer the larger of the KL and Markov '
          'bounds, each at 5%/6; jointly 95%): every valid certificate on '
          'this chain loses at least this much (up to the error of the (K q_3)(A) estimate); env = rigorous envelope upper bound '
          '(a valid certificate). cap = free bound from the LN norm cap; samp = 95% sampling certificate at the chain\'s FLOPs. '
          '* = provisional reference value.\n')
    print('| net | k | log2 p | K2 tail err | chain q_L err | last-exact err: fitted / K2 chain | sum eps: search / LCB / env | '
          'bits lost [last]: search / LCB / env | cap | samp |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for b in B:
        r = b['rare']; d = r['d']; eps = eps_same_chain(b)
        netr = RN.RNet(d, 4, r['act'], r['g'], r['sigma'], r['seed'], use_ln=not b['noln'])
        capd = math.sqrt(d) * float((netr.w - netr.w.mean()).norm())
        for i, lev in enumerate(r['levels']):
            t = truth_of(b, i); lp = t['logp']; lpb = lp / LOG2; ch = lev['chain']
            lkq, lkq2, _, _ = chain_of(b, i)
            capb = None if b['noln'] else (((_n.logsf((lev['t'] - capd) / r['sigma']) / LOG2) if lev['t'] > capd else 0.0) - lpb)
            def lost(e):
                return None if e is None else min(0.0, sum(e) * LOG2 + lkq) / LOG2 - lpb
            se = eps or {}
            print(f"| {name_of(b)} | {lev['k']} | {lpb:.1f}{'*' if t['provisional'] else ''} | {(lev['k2']['logp'] - lp) / LOG2:+.1f} | "
                  f"{(ch['log_qL'] - lp) / LOG2:+.1f} | {(lkq - lp) / LOG2:+.1f} / {f((lkq2 - lp) / LOG2 if lkq2 is not None else None, 1)} | "
                  f"{f(sum(se['opt']) if 'opt' in se else None, 1)} / {f(sum(se['lb']) if 'lb' in se else None, 1)} / "
                  f"{f(sum(se['env']) if 'env' in se else None, 0)} | "
                  f"{f(lost(se.get('opt')), 1)} / {f(lost(se.get('lb')), 1)} / {f(lost(se.get('env')), 0)} | {f(capb, 1)} | "
                  f"{lev['sampling_cert']['log_bound'] / LOG2 - lpb:.1f} |")
    print()
    print('### Same-chain eps per layer (bits): Markov D_inf LCB (95%), KL LCB (95%), eps LCB used in sums (5%/3 per layer), search, envelope; pre-review values transferred from another fit\n')
    print('| net | layer | Markov D_inf LCB | KL LCB | eps LCB (95%/3) | search c | env | transferred pre-review: rel_Dinf / search |')
    print('|---|---|---|---|---|---|---|---|')
    for b in B:
        e = b['epslb']
        if e is None: continue
        old = b['rare']['eps_layers']
        for x in e['layers']:
            l = x['layer']
            o1 = old.get('rel_Dinf', [None] * 4)[l - 1] if 'rel_Dinf' in old else None
            o2 = old.get('smc_c', [None] * 4)[l - 1] if 'smc_c' in old else None
            print(f"| {name_of(b)} | {l} | {dinf_lcb(x, 0.05):.1f} | {x['KL_lcb']:.2f} | {eps_lcb(x, 0.05 / 3):.1f} | "
                  f"{f(x.get('search_c'), 1)} | {x['env_c']:.0f} | {f(o1, 1)} / {f(o2, 1)} |")
    print()
    print('### Chain estimates of (K q_3)(A): frozen-tuning recompute minus the published value (bits), relative SE\n')
    for b in B:
        cf = b['chainfz']
        if cf is None: continue
        print(f"- {name_of(b)}: " + ', '.join(f"k={s['k']}: {(s['log_KqLm1'] - s['old_log_KqLm1']) / LOG2:+.2f} ({s['rel_se_KqLm1']:.2f})"
                                               + (f" / K2 {(s['log_KqLm1_k2'] - s['old_log_KqLm1_k2']) / LOG2:+.2f} ({s['rel_se_KqLm1_k2']:.2f})"
                                                  if s['old_log_KqLm1_k2'] is not None else '') for s in cf['per_level']))
    print()


def estimator_table(B):
    print('### Estimators: FLOPs to reach 30% relative RMSE, recomputed against the new reference values (setup + run x '
          '(rel/0.3)^2, from the larger budget; "unreliable" if the error is > 0.5 or does not fall roughly as 1/sqrt(N)); '
          'plain MC analytic 1/(0.09 p) samples. * = provisional reference.\n')
    meths = ('mc', 'bulk', 'tail', 'iapf', 'ms', 'gs')
    print('| net | k | plain MC | ' + ' | '.join(meths) + ' |')
    print('|---|---|---|' + '---|' * len(meths))
    for b in B:
        r = b['rare']
        for i, lev in enumerate(r['levels']):
            t = truth_of(b, i); c = math.exp(lev['truth']['logp'] - t['logp'])   # estimates were scored against lev['truth']
            p = math.exp(t['logp']); cells = []
            for m in meths:
                es = sorted([e for e in lev['estimators'] if e['method'] == m], key=lambda x: x['N'])
                if not es: cells.append('–'); continue
                def rescale(e):
                    vb = 1 + e['rel_bias']; m2 = e['rel_rmse'] ** 2 + 2 * vb - 1
                    return math.sqrt(max(c * c * m2 - 2 * c * vb + 1, 0.0))
                e, e0 = es[-1], es[0]; r1, r0 = rescale(e), rescale(e0)
                healthy = r1 <= 0.5 and r0 / max(r1, 1e-12) >= 0.5 * math.sqrt(e['N'] / e0['N'])
                cells.append(f"{e['flops_setup'] + e['flops_run'] * (r1 / 0.3) ** 2:.1e} ({r0:.2f}→{r1:.2f})" if healthy
                             else f"unreliable ({r0:.2f}→{r1:.2f})")
            print(f"| {name_of(b)} | {lev['k']}{'*' if t['provisional'] else ''} | {r['flops_sample'] / (0.09 * p):.1e} | " + ' | '.join(cells) + ' |')
    print()


def gc_tables(B):
    G = {netkey(g['_path'], 'gctailfz_'): g for g in load('gctailfz_*.json')}
    O = {netkey(g['_path'], 'gctail_'): g for g in load('prereview/gctail_*.json')}
    if not G: return
    print('## Copula-code chain tails (review fix 6: exact-tailed sampler, frozen splitting, latent splitting for the defensive part)\n')
    print('Errors in bits against the new reference values; [last] bound with eps = max(same-chain coverage search, 0) for '
          'layers 1-3 (uncertified, d <= 32). Pre-review values (bounded sampler, crude-MC defensive part, layer-1 search of '
          'another chain used for all layers) in brackets. * = provisional reference.\n')
    print('| net | k | log2 p | q_L err | last-exact err (pre-review) | rel SE | sum search eps | bits lost [last] (pre-review) |')
    print('|---|---|---|---|---|---|---|---|')
    byk = {b['key']: b for b in B}
    for k, g in sorted(G.items(), key=lambda kv: (kv[1]['act'], kv[1]['g'], kv[1]['d'])):
        b = byk.get(k)
        if b is None: continue
        srch = [max(s['search_c'], 0.0) for s in g.get('search', [])]
        o = O.get(k)
        for i, row in enumerate(g['levels']):
            t = truth_of(b, i); lp = t['logp']; lpb = lp / LOG2
            e = (row['KqLm1']['logp'] - lp) / LOG2
            oe = (o['levels'][i]['KqLm1']['logp'] - lp) / LOG2 if o else None
            lost = (min(0.0, sum(srch) * LOG2 + row['KqLm1']['logp']) / LOG2 - lpb) if len(srch) == 3 else None
            print(f"| {name_of(b)} | {row['k']} | {lpb:.1f}{'*' if t['provisional'] else ''} | {(row['qL']['logp'] - lp) / LOG2:+.1f} | "
                  f"{e:+.1f} ({f(oe, 1)}) | {row['KqLm1']['rel_se']:.2f} | {f(sum(srch) if len(srch) == 3 else None, 1)} | {f(lost, 1)} |")
    print()


# ---------------------------------------------------------------- Task 3a: E[output]
def eout_tables():
    E = load('eout_*.json')
    if not E: return
    print('## Task 3a: E[output], normalised MSE (ARC metric), both K = 2 variants (review fix 3)\n')
    print('K=2 = default (mean-field LN with the delta-method mean correction; what eout.py calls k2); noLNcorr = the same '
          'without the correction. Advantage = plain sampling\'s MSE at K = 2\'s FLOPs / K = 2\'s MSE (seed mean of MSEs). '
          'floor = the truth\'s own noise, contained in every measured MSE; "-floor" columns subtract it.\n')
    print('| act | sigma | d | L | seeds | K=2 nMSE (adv) | noLNcorr nMSE (adv) | floor | K=2 adv - floor | noLNcorr adv - floor |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    keys = sorted({(e['act'], e['g'], e['sigma'], e['bias'], e['d'], e['L']) for e in E})
    for k in keys:
        es = [e for e in E if (e['act'], e['g'], e['sigma'], e['bias'], e['d'], e['L']) == k]
        k2 = np.mean([e['k2']['nmse'] for e in es]); k0 = np.mean([e['k2_nolncorr']['nmse'] for e in es])
        mc = np.mean([e['k2']['mc_nmse_at_equal_flops'] for e in es]); fl = np.mean([e['truth_nmse_floor'] for e in es])
        adv = lambda x: mc / x if x > 0 else float('inf')
        print(f"| {k[0]} {k[1]:g}{' b' + format(k[3], 'g') if k[3] else ''} | {k[2]:g} | {k[4]} | {k[5]} | {len(es)} | {k2:.1e} ({adv(k2):.0f}x) | "
              f"{k0:.1e} ({adv(k0):.0f}x) | {fl:.1e} | {adv(k2 - fl):.0f}x | {adv(k0 - fl):.0f}x |")
    print()


def e2e_tables():
    E = load('e2e_*.json')
    if not E: return
    print('## Theory check: end-to-end q_L vs real p_L (L = 4, sigma = 0.3), bits (unchanged by the review)\n')
    print('| net | family | KL(p_L||q_L) | sum_l KL(Kq||q) | max pts log p_L/q_L | sum_l c (pts) | KL(q_L||p_L) | sum_l KL(q||Kq) |')
    print('|---|---|---|---|---|---|---|---|')
    for r in sorted(E, key=lambda x: (regime(x), x['d'])):
        for fam, v in r['fams'].items():
            print(f"| {regime(r)} d={r['d']} | {fam} | {v['KL_p_q']:.2f} ± {v['KL_p_q_se']:.2f} | {v['sum_KLc']:.2f} | "
                  f"{v['c_end_pts']:.2f} | {v['sum_c_pts']:.2f} | {v['KL_q_p']:.2f} ± {v['KL_q_p_se']:.2f} | {v['sum_KLw']:.2f} |")
    print()


def surr_tables():
    S = load('surr_*.json')
    if not S: return
    O = {os.path.basename(o['_path']): o for o in load('prereview/surr_*.json')}
    print('## Addition: surrogate chains (sigma = 0.3)\n')
    print('KL upper bound for DV (review fix 5): one-sided 95% path-level bootstrap of the per-path sum over layers '
          '(pre-review: mean + 2 SE with the layers treated as independent).\n')
    print('| net | d | L | STAT | LR | QUAD | PL | MFLN | rho STAT | K2 nMSE | path KL (nats): mean / UCB (old) | DV width / CI: y; y^2/2 (old) | contains truth |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(S, key=lambda x: (x['act'], x['g'], x['d'], x['L'])):
        kl = {n: float(np.mean([v['mean'] for v in r['kl_bits'][n]])) for n in r['kl_bits']}
        o = O.get(os.path.basename(r['_path']), {}).get('dv', {})
        print(f"| {r['act']} {r['g']:g} | {r['d']} | {r['L']} | " + ' | '.join(f(kl[n], 2) for n in ('STAT', 'LR', 'QUAD', 'PL', 'MFLN')) +
              f" | {r['eout']['rho_stat']:.3f} | {r['eout']['k2_nmse']:.1e} | {r['dv']['KL_nats']:.2f} / {r['dv']['KL_upper_nats']:.2f} "
              f"({f(o.get('KL_upper_nats'))}) | {r['dv']['width_ratio_y_median']:.0f}; {r['dv']['width_ratio_F2_median']:.0f} "
              f"({f(o.get('width_ratio_y_median'), 0)}; {f(o.get('width_ratio_F2_median'), 0)}) | {r['dv']['contains_truth']} |")
    wy = [r['dv']['width_ratio_y_median'] for r in S]; wf = [r['dv']['width_ratio_F2_median'] for r in S]
    print(f"\nDV width / CI ranges: y {min(wy):.1f}-{max(wy):.1f}; y^2/2 {min(wf):.1f}-{max(wf):.1f}; all contain the truth: "
          f"{all(r['dv']['contains_truth'] for r in S)}\n")
    print('### E[output] (ARC metric): normalised MSE at matched FLOPs\n')
    print('| net | d | L | K2 FLOPs | K2 nMSE | MC at K2 FLOPs | SUR at 1000x K2 FLOPs | MC there | asymptotic SUR gain over MC | N* = rho / K2 nMSE |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(S, key=lambda x: (x['act'], x['g'], x['d'], x['L'])):
        e = r['eout']; fk2 = e['flops_k2']; fs = r['flops_sample']; d = r['d']; L = r['L']
        per = fs + L * (2 * d * d + 2 * d) + 16 * d
        sur = e['rho_stat'] / max(1.0, (1000 * fk2 - fk2) / per)
        print(f"| {r['act']} {r['g']:g} | {d} | {L} | {fk2:.1e} | {e['k2_nmse']:.1e} | {fs / fk2:.1e} | {sur:.1e} | {fs / (1000 * fk2):.1e} | "
              f"{fs / per / e['rho_stat']:.2f} | {e['rho_stat'] / max(e['k2_nmse'], 1e-300):.0f} |")
    print()
    M = load('skipmerge_*.json')
    if M:
        print('### Skip-merge of layers 2-3 (bits per merged pair, Gaussian lower bound; unchanged)\n')
        print('| net | d | merged sigma^2 (I + J J^T) | of which mean | naive sigma^2 I | floor | tr(C_real)/tr(C_merged) |')
        print('|---|---|---|---|---|---|---|')
        for r in sorted(M, key=lambda x: (x['act'], x['g'], x['d'])):
            print(f"| {r['act']} {r['g']:g} | {r['d']} | {r['kl_merged_bits']:.2f} | {r['mean_term_bits']:.2f} | {r['kl_naive_bits']:.2f} | "
                  f"{r['floor_bits']:.2f} | {r['trace_ratio']:.2f} |")
        print()


if __name__ == '__main__':
    print('# Phase R tables (post-review, PREREG-R amendment 2). Pre-review tables: results/prereview/tables.md\n')
    score_tables()
    B = rare_bundle()
    truth_table(B); rare_tables(B); estimator_table(B); gc_tables(B)
    eout_tables(); e2e_tables(); surr_tables()
