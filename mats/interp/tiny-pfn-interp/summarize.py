"""Tables for the step-1 report from out/*.json (decision rules: PREREG.md)."""
import json, glob
import numpy as np

SITES = ('emb', 'attn1', 'mlp1', 'attn2', 'mlp2')
G = {'main': [f'out/main_s{s}.json' for s in range(5)], 'small': ['out/small_s2.json', 'out/small_s4.json'],
     'plateau': ['out/plateau_s3.json']}
NEED = {'main': 4, 'small': 2, 'plateau': 1}
ROWS = [('r', 'LAB'), ('r2', 'LAB'), ('SS', 'LAB'), ('logSS', 'LAB'), ('SS', 'QQ'), ('logSS', 'QQ'),
        ('p_lab', 'LAB'), ('p_lab', 'QQ'), ('E_lab', 'LAB'), ('E_lab', 'QQ'),
        ('p_bayes', 'LAB'), ('p_bayes', 'QQ'), ('E_bayes', 'LAB'), ('E_bayes', 'QQ'), ('beta', 'LAB'), ('beta', 'QQ'),
        ('mean_bayes', 'QQ'), ('logvar_bayes', 'QQ'),
        ('mean_bayes', 'QT'), ('logvar_bayes', 'QT'), ('var_bayes', 'QT'), ('entropy_w27', 'QT'), ('mean_lab', 'QT'),
        ('model_mean', 'QT')]


def load(g):
    return [json.load(open(f)) for f in G[g]]


def cell(rs, key):
    tr = np.array([np.mean(r['probes_trained'][key][0]) for r in rs])
    un = np.array([np.mean(r['probes_untrained'][key][0]) for r in rs])
    sh = np.array([np.mean(r['probes_trained'][key][1]) for r in rs])
    pos = (tr >= 0.5) & (tr - np.maximum(un, sh) >= 0.10)
    return tr, un, sh, pos


def probe_table(g):
    rs = load(g); need = NEED[g]
    print(f'\n### Probes, {g} ({len(rs)} seed(s)): held-out R^2 trained (untrained); * = POSITIVE (>= {need}/{len(rs)} seeds), ** = also strong (mean R^2 >= 0.8)\n')
    print('| target | token | ' + ' | '.join(SITES) + ' | first appears |')
    print('|---|---|' + '---|' * len(SITES) + '---|')
    first = {}
    shmax = 0
    for name, kind in ROWS:
        cells, fa = [], None
        for site in SITES:
            tr, un, sh, pos = cell(rs, f'{name}|{kind}|{site}')
            shmax = max(shmax, sh.max())
            ok = pos.sum() >= need
            mark = ('**' if tr.mean() >= 0.8 else '*') if ok else ''
            if ok and fa is None: fa = site
            cells.append(f'{tr.mean():.2f}{mark} ({un.mean():.2f})')
        first[(name, kind)] = fa
        print(f'| {name} | {kind} | ' + ' | '.join(cells) + f' | {fa or "not found"} |')
    print(f'\nShuffled-target baseline: max held-out R^2 over all cells and seeds = {shmax:.3f}.')
    return first


def attn_table(g):
    rs = load(g)
    print(f'\n### Attention, {g}: mean over seeds (untrained in parentheses)\n')
    print('| head | from | self | own LAB /tok | other LAB /tok | own/other LAB | own QQ | other QQ /tok | corr(attn, r^2 of key) |')
    print('|---|---|---|---|---|---|---|---|---|')
    keys = list(rs[0]['attn_trained'].keys())
    for k in keys:
        for frm in ('LAB', 'QQ'):
            f = lambda nm, which='attn_trained': np.mean([r[which][k][frm][nm] for r in rs])
            fu = lambda nm: f(nm, 'attn_untrained')
            oq = f'{f("own_qq"):.3f}' if frm == 'LAB' else '-'
            ratios = [r['attn_trained'][k][frm]['ratio_lab'] for r in rs]
            print(f'| {k} | {frm} | {f("self"):.3f} | {f("own_lab_per_tok"):.3f} | {f("oth_lab_per_tok"):.3f} | '
                  f'{np.mean(ratios):.2f} [{min(ratios):.2f}-{max(ratios):.2f}] ({fu("ratio_lab"):.2f}) | {oq} | '
                  f'{f("oth_qq_per_tok"):.3f} | {f("corr_attn_r2"):+.2f} ({fu("corr_attn_r2"):+.2f}) |')
    print('\nQuery token (QT): mass on self / LAB / QQ, and Pearson r of per-source mass with Bayes weight beta_s and with E_s (held-out):\n')
    print('| head | self | LAB | QQ | r(all 5 tokens, beta) | r(QQ token, beta) | r(QQ, beta), normalised within episode | r(all, E_lab) | r(QQ->other QQ, beta) | seeds r(QQ,beta)>=0.5 & margin |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for k in keys:
        f = lambda nm, which='attn_trained', frm='QT': np.mean([r[which][k][frm][nm] for r in rs])
        fu = lambda nm, frm='QT': f(nm, 'attn_untrained', frm)
        npass = sum((r['attn_trained'][k]['QT']['r_qq_beta'] >= 0.5) and
                    (r['attn_trained'][k]['QT']['r_qq_beta'] - abs(r['attn_untrained'][k]['QT']['r_qq_beta']) >= 0.2) for r in rs)
        npa = sum((r['attn_trained'][k]['QT']['r_all_beta'] >= 0.5) and
                  (r['attn_trained'][k]['QT']['r_all_beta'] - abs(r['attn_untrained'][k]['QT']['r_all_beta']) >= 0.2) for r in rs)
        print(f'| {k} | {f("self"):.2f} | {f("lab"):.2f} | {f("qq"):.2f} | {f("r_all_beta"):+.2f} ({fu("r_all_beta"):+.2f}) | '
              f'{f("r_qq_beta"):+.2f} ({fu("r_qq_beta"):+.2f}) | {f("r_qqnorm_beta"):+.2f} ({fu("r_qqnorm_beta"):+.2f}) | '
              f'{f("r_all_E"):+.2f} | {f("r_qq_to_oth_qq_beta", frm="QQ"):+.2f} ({fu("r_qq_to_oth_qq_beta", frm="QQ"):+.2f}) | QQ {npass}/{len(rs)}, all {npa}/{len(rs)} |')
    per = {k: [round(r['attn_trained'][k]['QT']['r_qq_beta'], 2) for r in rs] for k in keys}
    print('\nPer-seed r(QQ token mass, beta):', per)


def sym_table():
    print('\n### Symmetry\n')
    print('| network | equivariance max abs (sites / head) | relabel: mean abs change of predictive mean | identity-redraw sd of mean | sd(model - Bayes mean) | relabel probe checks (R^2 orig -> relabelled own / vs identity source) |')
    print('|---|---|---|---|---|---|')
    for g in G:
        for f, r in zip(G[g], load(g)):
            s = r['symmetry']
            checks = '; '.join(f"{k[8:]}: {v['r2_orig']:.2f}->{v['r2_relabelled_own']:.2f} / {v['r2_relabelled_vs_identity_source']:.2f}"
                               for k, v in s.items() if k.startswith('relabel_') and isinstance(v, dict))
            print(f"| {r['tag']} | {s['equivariance_max_abs']:.1e} / {s['equivariance_head_max_abs']:.1e} | {s['relabel_mean_abs_dmean']:.4f} | "
                  f"{s['redraw_sd_dmean']:.4f} | {s['sd_model_minus_bayes']:.4f} | {checks} |")


if __name__ == '__main__':
    for g in G:
        print(f'\n{g}: NLL gaps', [round(r['nll_gap'], 4) for r in load(g)])
    for g in G:
        probe_table(g)
    for g in G:
        attn_table(g)
    sym_table()
