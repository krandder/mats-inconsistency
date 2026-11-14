"""Compact report tables from results/*.json (gate-entropy). usage: python report_tables.py > results/report_tables.md"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
l1 = json.load(open(os.path.join(HERE, 'results', 'layer1_bound.json')))


def load(tag):
    p = os.path.join(HERE, 'results', tag + '.json')
    if not os.path.exists(os.path.join(HERE, 'results', tag + '.done')):
        return None
    return json.load(open(p))


def lo_up(p):
    lo = p['plugin']
    if not p['H2_zero']: lo = max(lo, p['H2'])
    return lo, p['upper']


def br(lo, up, approx=None, point=None):
    if up - lo < 1.0:
        return f'{(point if point is not None else (lo + up) / 2):.1f}'
    s = f'{lo:.1f}-{up:.1f}'
    if approx is not None and approx > lo:
        s = f'{lo:.1f} [{approx:.0f}*]-{up:.1f}'
    return s


def e3s(e3, key):
    v = e3.get(key)
    if v is None:
        last = e3['path'][-1]
        return f'nr ({100 * last["change_rate"]:.1f}% at {last["H_leaves"]:.1f})' if key.startswith('H_merged_eps') else 'nr'
    return f'{abs(v):.1f}' if abs(v) < 0.05 else f'{v:.1f}'


def rnn_table(tag):
    r = load(tag)
    if r is None: return f'{tag}: pending\n'
    rows = [f'**{tag}** ({r["name"]}; accuracy {r["answer_ho_set"]["acc"]:.4f})', '',
            '| step | H(G) | H(G|X) | D(1%) | H(G_R) | k var99 | k agree99 | E3(1%) | E3(0.2%) |', '|---|---|---|---|---|---|---|---|---|']
    for l in range(1, 11):
        L = r['layers'][str(l)]
        lo, up = lo_up(L['E1']); e2 = L['E2']['tau0.01']
        lo2, up2 = lo_up(e2['panel']) if e2.get('panel') else (0, 0)
        s = e2.get('subspace', {})
        c = L.get('cond', {})
        rows.append(f'| {l} | {br(lo, up, point=L["E1"]["mm"])} | {c.get("plugin_mean", float("nan")):.2f} | {e2["D"]} | {br(lo2, up2, point=e2["panel"]["mm"] if e2.get("panel") else None)} | {s.get("k_var99", "-")} | '
                    f'{s.get("k_agree0.99", "-")} | {e3s(L["E3"], "H_merged_eps0.01")} | {e3s(L["E3"], "H_merged_eps0.002")} |')
    return '\n'.join(rows) + '\n'


def mlp_table(tag, family, width):
    r = load(tag)
    if r is None: return f'{tag}: pending\n'
    rows = [f'**{tag}** ({r["name"]})', '',
            '| layer | m | H(G) | H(G|X) | C_cov | D(1%) | median f | k var99 | k agree99 | E3(1%) | E3(bias<=1e-4) | E3(bias<=1e-5) |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for l, L in sorted(r['layers'].items(), key=lambda kv: int(kv[0])):
        lo, up = lo_up(L['E1'])
        approx = L['E1']['H2'] if L['E1']['H2_zero'] else None
        if l == '1':
            key = f'arc_n{width}' if family == 'arc' else (f'tanh_g3_n{width}' if family == 'tanhC' else f'tanh_g0.25_n{width}')
            if key in l1: lo = max(lo, l1[key]['lower'])
        f = np.array(L['flip_rate'])
        e2 = L['E2']['tau0.01']; s = e2.get('subspace', {})
        c = L.get('cond', {})
        cs = f'{c["plugin_mean"]:.1f}' + (f'-{c["indep_mean"]:.1f}' if c.get('indep_mean', 0) - c.get('plugin_mean', 0) > 1 else '') if c else '-'
        rows.append(f'| {l} | {L["m"]} | {br(lo, up, approx, point=L["E1"]["mm"])} | {cs} | {("%.0f" % L["cap_cov_bits"]) if L.get("cap_cov_bits") else "-"} | {e2["D"]} | {np.median(f):.2f} | '
                    f'{s.get("k_var99", "-")} | {s.get("k_agree0.99", "-")} | {e3s(L["E3"], "H_merged_eps0.01")} | '
                    f'{e3s(L["E3"], "H_merged_bias0.0001")} | {e3s(L["E3"], "H_merged_bias1e-05")} |')
    return '\n'.join(rows) + '\n'


if __name__ == '__main__':
    print('# Report tables (report_tables.py)\n')
    print('H(G) and H(G_R): one number = bracket narrower than 1 bit (Miller-Madow point estimate); a-b = [lower, upper]; [x*] = approximate '
          'zero-collision Renyi-2 bound. Layer-1 lower bounds of the MLPs include the rigorous Gaussian chain-rule bound. '
          'E3: nr = not reached (last checkpoint: change rate at leaf entropy). H(G|X): mean per-input plug-in over 256 '
          'noise draws (capped at 8 bits), with the per-unit sum when it is more than 1 bit higher.\n')
    print('## RNNs\n')
    for t in ('T16', 'W32s0', 'W64s0', 'W32s1', 'W64s1'):
        print(rnn_table(t))
    print('## Random MLPs\n')
    for fam, ws, ss in (('arc', (32, 64, 128, 256), (0.01,)), ('arc', (64,), (0.001, 0.1)), ('tanhC', (32, 64, 128, 256), (0.01,)),
                        ('tanhC', (64,), (0.001, 0.1)), ('tanhO', (64, 256), (0.01,)), ('tanhO', (64,), (0.001, 0.1))):
        for w in ws:
            for s in ss:
                print(mlp_table(f'{fam}{w}_s{s:g}', fam, w))
    r = load('PFN')
    if r:
        print('## PFN\n')
        for l in ('1', '2'):
            L = r['layers'][l]; lo, up = lo_up(L['E1']); e2 = L['E2']['tau0.01']; lo2, up2 = lo_up(e2['panel'])
            print(f'layer {l}: H(G) {br(lo, up, L["E1"]["H2"] if L["E1"]["H2_zero"] else None)}; D(1%) {e2["D"]}, D(0.1%) '
                  f'{L["E2"]["tau0.001"]["D"]}; H(G_R) {br(lo2, up2, e2["panel"]["H2"] if e2["panel"]["H2_zero"] else None)}; '
                  f'k var99 {e2["subspace"]["k_var99"]}, k agree99 {e2["subspace"]["k_agree0.99"]}; E3(1%) {e3s(L["E3"], "H_merged_eps0.01")}\n')
