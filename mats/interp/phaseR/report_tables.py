"""Markdown snippets for the revised report (post-review), from the same data and rules as analyze.py.
usage: python report_tables.py > /tmp/report_snippets.md"""
import math
import numpy as np
import analyze as A

LOG2 = math.log(2)


def fmt(x, nd=1, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '–'
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def bundle_by_name(B):
    return {A.name_of(b): b for b in B}


def task2(B):
    import rnet as RN
    from scipy.stats import norm as _n
    sel = [('critical d=8', (30, 40, 60)), ('critical d=16', (30, 40, 60)), ('critical d=32', (30, 50, 60)),
           ('critical d=64', (30, 60)), ('ReLU d=16', (30, 60)), ('ReLU d=256', (30, 60)), ('chaotic d=64', (10, 30, 50, 60))]
    by = bundle_by_name(B)
    print('| network | p (reference) | eps = search | eps = LCB | eps = envelope | LN cap | sampling |')
    print('|---|---|---|---|---|---|---|')
    for nm, ks in sel:
        b = by.get(nm)
        if b is None: continue
        r = b['rare']; d = r['d']; eps = A.eps_same_chain(b) or {}
        netr = RN.RNet(d, 4, r['act'], r['g'], r['sigma'], r['seed'])
        capd = math.sqrt(d) * float((netr.w - netr.w.mean()).norm())
        first = True
        for i, lev in enumerate(r['levels']):
            if lev['k'] not in ks: continue
            t = A.truth_of(b, i); lp = t['logp']; lpb = lp / LOG2
            lkq, _, _, _ = A.chain_of(b, i)
            def lost(e):
                if e is None: return 'not run'
                v = min(0.0, sum(e) * LOG2 + lkq) / LOG2 - lpb
                return 'vacuous' if v >= -lpb - 1e-9 else f"{v:.1f}"
            capb = ((_n.logsf((lev['t'] - capd) / r['sigma']) / LOG2) if lev['t'] > capd else 0.0) - lpb
            cap = 'vacuous' if capb >= -lpb - 1e-9 else f"{capb:.1f}"
            print(f"| {nm if first else ''} | 2^{lpb:.1f}{'*' if t['provisional'] else ''} | {lost(eps.get('opt'))} | {lost(eps.get('lb'))} | "
                  f"{lost(eps.get('env'))} | {cap} | {lev['sampling_cert']['log_bound'] / LOG2 - lpb:.1f} |")
            first = False
    print()
    print('Sums over layers 1-3 (bits): search (max with LCB) / LCB (Bonferroni) / envelope:')
    for b in B:
        e = A.eps_same_chain(b)
        if e: print(f"- {A.name_of(b)}: {fmt(sum(e['opt']), 1, False) if 'opt' in e else '–'} / {sum(e['lb']):.1f} / {sum(e['env']):.0f}"
                    f"   per layer LCB {', '.join(f'{x:.1f}' for x in e['lb'])}" + (f"; search {', '.join(f'{x:.1f}' for x in e['search'])}" if 'search' in e else ''))
    print()


def task3_rare(B):
    print('| network | K = 2 output tail | Gaussian chain, last layer exact | same with the K = 2 chain |')
    print('|---|---|---|---|')
    allerr = []
    for b in B:
        r = b['rare']; c1, c2, c3 = [], [], []
        for i, lev in enumerate(r['levels']):
            t = A.truth_of(b, i); lp = t['logp']; star = '*' if t['provisional'] else ''
            lkq, lkq2, _, _ = A.chain_of(b, i)
            c1.append(fmt((lev['k2']['logp'] - lp) / LOG2) + star)
            e = (lkq - lp) / LOG2; c2.append(fmt(e) + star)
            c3.append(fmt((lkq2 - lp) / LOG2 if lkq2 is not None else None) + star)
            allerr.append((A.name_of(b), lev['k'], e, (lkq2 - lp) / LOG2 if lkq2 is not None else None, t['provisional'], b['noln'], t['flags']))
        ks = '/'.join(str(lev['k']) for lev in r['levels'])
        print(f"| {A.name_of(b)} (k = {ks}) | {' / '.join(c1)} | {' / '.join(c2)} | {' / '.join(c3)} |")
    print()
    ln = [x for x in allerr if not x[5]]
    for lab, sub in (('corroborated', [x for x in ln if not x[4]]), ('provisional', [x for x in ln if x[4]])):
        ex = [x for x in sub if not x[0].startswith('ReLU d=16')]
        if ex:
            print(f"{lab} levels on LN nets except ReLU d=16 ({len(ex)}): fitted chain error {min(x[2] for x in ex):+.1f} to {max(x[2] for x in ex):+.1f};"
                  f" K2 chain {min(x[3] for x in ex if x[3] is not None):+.1f} to {max(x[3] for x in ex if x[3] is not None):+.1f}")
        rl = [x for x in sub if x[0].startswith('ReLU d=16')]
        if rl:
            print(f"  ReLU d=16 {lab}: " + ', '.join(f"k={x[1]}: {x[2]:+.1f}" for x in rl))
    print('provisional levels:', ', '.join(f"{x[0]} k={x[1]} ({','.join(x[6])})" for x in allerr if x[4]))
    print()


def estimators(B):
    sel = [('critical d=16', (10, 20, 30, 40)), ('critical d=64', (10, 20, 40, 50)), ('chaotic d=64', (20,)), ('ReLU d=64', (10, 30)),
           ('ReLU d=256', (10, 20, 30, 40))]
    by = bundle_by_name(B)
    meths = ('mc', 'bulk', 'iapf', 'tail', 'ms', 'gs')
    print('| network | p | plain sampling | twisted SMC, chain psi | learned psi (iterated APF) | most-likely-path psi | mean-shift IS | splitting |')
    print('|---|---|---|---|---|---|---|---|')
    for nm, ks in sel:
        b = by.get(nm)
        if b is None: continue
        r = b['rare']; first = True
        for i, lev in enumerate(r['levels']):
            if lev['k'] not in ks: continue
            t = A.truth_of(b, i); c = math.exp(lev['truth']['logp'] - t['logp']); p = math.exp(t['logp'])
            cells = []
            for m in meths:
                if m == 'mc':
                    cells.append(f"{r['flops_sample'] / (0.09 * p):.1e}"); continue
                es = sorted([e for e in lev['estimators'] if e['method'] == m], key=lambda x: x['N'])
                def rescale(e):
                    vb = 1 + e['rel_bias']; m2 = e['rel_rmse'] ** 2 + 2 * vb - 1
                    return math.sqrt(max(c * c * m2 - 2 * c * vb + 1, 0.0))
                e, e0 = es[-1], es[0]; r1, r0 = rescale(e), rescale(e0)
                healthy = r1 <= 0.5 and r0 / max(r1, 1e-12) >= 0.5 * math.sqrt(e['N'] / e0['N'])
                cells.append(f"{e['flops_setup'] + e['flops_run'] * (r1 / 0.3) ** 2:.1e}" if healthy else 'no')
            print(f"| {nm if first else ''} | 2^-{lev['k']}{'*' if t['provisional'] else ''} | " + ' | '.join(cells) + ' |')
            first = False
    print()


def task1():
    C = A.load('scorecb_*.json')
    def vals(reg, d, fam, key, layer=0):
        rs = sorted([r for r in C if A.regime(r) == reg and r['d'] == d and r['L'] <= 4], key=lambda r: r['seed'])
        return [(A.eps_lcb(r['layers'][layer][fam], 0.05) if key == 'Dinf_lcb' else r['layers'][layer][fam][key]) for r in rs if layer < len(r['layers']) and fam in r['layers'][layer]]
    rows = [('ordered: KL(Kq‖Gaussian) >=', 'tanh 0.25', 'G', 'KL_lcb'), ('critical: KL(Kq‖Gaussian) >=', 'tanh 0.75', 'G', 'KL_lcb'),
            ('critical: D_inf(Kq‖Gaussian) >=', 'tanh 0.75', 'G', 'Dinf_lcb'), ('chaotic: KL(Kq‖Gaussian) >=', 'tanh 3', 'G', 'KL_lcb'),
            ('chaotic: D_inf(Kq‖Gaussian) >=', 'tanh 3', 'G', 'Dinf_lcb'), ('ReLU: KL(Kq‖Gaussian) >=', 'relu 1.414', 'G', 'KL_lcb'),
            ('ReLU: D_inf(Kq‖Gaussian) >=', 'relu 1.414', 'G', 'Dinf_lcb'), ('critical: KL(Kq‖product code) >=', 'tanh 0.75', 'NC', 'KL_lcb'),
            ('critical: KL(Kq‖know-nothing) >=', 'tanh 0.75', 'KN', 'KL_lcb')]
    print('| quantity (95% LCB; seeds 0 / 1 / 2) | d = 8 | 16 | 32 | 64 | 256 |')
    print('|---|---|---|---|---|---|')
    for lab, reg, fam, key in rows:
        cells = []
        for d in (8, 16, 32, 64, 256):
            v = vals(reg, d, fam, key)
            if not v: cells.append('not run'); continue
            s = ' / '.join(f"{x:.1f}" for x in v)
            if d == 256 and key == 'KL_lcb': s += f" ({np.mean(v) / d:.3f})"
            cells.append(s)
        print(f"| {lab} | " + ' | '.join(cells) + ' |')
    env = []
    for d in (8, 16, 32, 64, 256):
        rs = sorted([r for r in C if A.regime(r) == 'tanh 0.75' and r['d'] == d and r['L'] <= 4], key=lambda r: r['seed'])
        env.append(f"{rs[0]['layers'][0]['env_c']:,.0f}" if rs else '–')
    print('| critical: envelope (upper bound on c, seed 0) | ' + ' | '.join(env) + ' |')
    print()


if __name__ == '__main__':
    B = A.rare_bundle()
    print('## Task 1 width (layer 1)\n'); task1()
    print('## Task 2\n'); task2(B)
    print('## Task 3 rare\n'); task3_rare(B)
    print('## Estimators\n'); estimators(B)
