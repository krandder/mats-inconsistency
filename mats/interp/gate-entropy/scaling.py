"""Width x depth scaling tables (gate-entropy). usage: python scaling.py > results/scaling.md"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L1 = os.path.join(HERE, 'results', 'layer1_bound.json')
l1 = json.load(open(L1)) if os.path.exists(L1) else {}


def load(tag):
    p = os.path.join(HERE, 'results', tag + '.json')
    if not os.path.exists(p) or not os.path.exists(os.path.join(HERE, 'results', tag + '.done')):
        return None
    return json.load(open(p))


def lo_up(p):
    lo = p['plugin']
    if not p['H2_zero']: lo = max(lo, p['H2'])
    return lo, p['upper'], p['H2_zero'], p['H2']


def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def cell(r, l, family, width):
    L = r['layers'].get(str(l))
    if L is None: return '-'
    lo, up, zero, H2 = lo_up(L['E1'])
    if l == 1 and family in ('arc', 'tanh'):
        g = r['name'].split('_')[1][1:] if family == 'tanh' else None          # e.g. tanh_g3_d64... -> '3'
        key = f'arc_n{width}' if family == 'arc' else f'tanh_g{g}_n{width}'
        if key in l1: lo = max(lo, l1[key]['lower'])
    e3 = L['E3']
    s = f'{lo:.0f}-{up:.0f}' if up - lo >= 1 else f'{L["E1"]["mm"]:.1f}'
    if zero and lo < H2: s = f'{lo:.0f}(H2>{H2:.0f}*)-{up:.0f}'
    s += ' / ' + (f'{e3["H_merged_eps0.01"]:.1f}' if e3.get('H_merged_eps0.01') is not None else 'nr')
    if 'H_merged_bias0.0001' in e3:
        s += ' / ' + (f'{e3["H_merged_bias0.0001"]:.1f}' if e3.get('H_merged_bias0.0001') is not None else 'nr')
    f = np.array(L['flip_rate']); D = int((f >= 0.01).sum())
    s += f' / D={D}'
    cond = L.get('cond')
    if cond: s += f' / n={cond["plugin_mean"]:.1f}'
    return s


def fano_lower(r, l, eps=0.01, l1key=None):
    """Heuristic E3 lower bound: if every substitution changes the answer at least as often as the least sensitive
    relevant single flip (rho = min_i f_i over R), then P(G != rep) <= eps / rho and by Fano
    E3 >= H(G_R) - h2(Pe) - Pe log2(K), K = 2^D (3^D/... for tanh: 2^D upper)."""
    L = r['layers'][str(l)]
    f = np.array(L['flip_rate']); R = f >= 0.01
    if not R.any(): return None
    rho = f[R].min(); Pe = min(eps / rho, 1.0)
    p = L['E2']['tau0.01'].get('panel') or L['E1']
    lo, up, zero, H2 = lo_up(p)
    if l == 1 and l1key in l1 and R.sum() == len(f): lo = max(lo, l1[l1key]['lower'])
    out = dict(rho=float(rho), Pe=float(Pe), lower=float(lo - h2(Pe) - Pe * R.sum()), H_lo=float(lo))
    last = L['E3']['path'][-1]
    cgc = last.get('change_given_changed')
    if cgc is None and last.get('frac_any_gate_changed') is None and last['change_rate'] > 0.9:
        cgc = last['change_rate']            # older runs: nearly every sample substituted, so rho ~ change rate
    if cgc:
        Pe2 = min(eps / cgc, 1.0)
        out.update(rho_tree=float(cgc), lower_tree=float(lo - h2(Pe2) - Pe2 * R.sum()))
    return out


def table(family, sigma, widths, layers, tagfmt):
    rows = ['| width | ' + ' | '.join(f'layer {l}' for l in layers) + ' |', '|---|' + '---|' * len(layers)]
    for w in widths:
        r = load(tagfmt.format(w=w, s=sigma))
        if r is None:
            rows.append(f'| {w} | ' + ' | '.join('pending' for _ in layers) + ' |'); continue
        rows.append(f'| {w} | ' + ' | '.join(cell(r, l, family, w) for l in layers) + ' |')
    return '\n'.join(rows)


if __name__ == '__main__':
    print('# Scaling tables (scaling.py)\n')
    print('Cell: H(G) bracket [lower-upper] (lower includes the layer-1 Gaussian chain-rule bound) / E3 at 1% per-input '
          'change (nr = not reached) / E3 at mean-bias nMSE <= 1e-4 / D(1%) / n = noise-driven H(G|X) (plug-in).\n')
    layers = [1, 2, 4, 8, 16, 24, 32]
    print('## ARC He-ReLU, sigma = 0.01, depth 32\n')
    print(table('arc', 0.01, [32, 64, 128, 256], layers, 'arc{w}_s{s:g}'))
    print('\n## tanh chaotic (g = 3, LN, bias 0.5), sigma = 0.01, depth 32\n')
    print(table('tanh', 0.01, [32, 64, 128, 256], layers, 'tanhC{w}_s{s:g}'))
    print('\n## tanh ordered (g = 0.25), sigma = 0.01\n')
    print(table('tanh', 0.01, [64, 256], layers, 'tanhO{w}_s{s:g}'))
    for s in (0.001, 0.1):
        print(f'\n## sigma = {s:g}: ARC (64, 256), tanh chaotic and ordered (64)\n')
        print(table('arc', s, [64, 256], layers, 'arc{w}_s{s:g}'))
        print()
        print(table('tanh', s, [64], layers, 'tanhC{w}_s{s:g}'))
        print()
        print(table('tanh', s, [64], layers, 'tanhO{w}_s{s:g}'))
    print('\n## RNNs (per step): H(G_t) / E3(1%) / D / noise-driven\n')
    for tag in ('T16', 'W32s0', 'W64s0', 'W32s1', 'W64s1'):
        r = load(tag)
        if r is None: print(f'{tag}: pending\n'); continue
        cells = []
        for l in range(1, 11):
            L = r['layers'][str(l)]; lo, up, zero, H2 = lo_up(L['E1']); e3 = L['E3']
            cells.append((f'{L["E1"]["mm"]:.1f}' if up - lo < 1 else f'{lo:.1f}-{up:.1f}') + ' / ' + (f'{e3["H_merged_eps0.01"]:.1f}' if e3.get('H_merged_eps0.01') is not None else 'nr')
                         + ' / ' + (f'{e3["H_merged_eps0.002"]:.1f}' if e3.get('H_merged_eps0.002') is not None else 'nr'))
        print(f'{tag}: ' + ' | '.join(f't{t + 1}: {c}' for t, c in enumerate(cells)) + '\n')
    print('\n## Heuristic Fano lower bound on E3(1%) (assumes rho = min relevant single-flip rate holds for every substitution)\n')
    for tag in ('arc32_s0.01', 'arc64_s0.01', 'arc128_s0.01', 'arc256_s0.01', 'tanhC32_s0.01', 'tanhC64_s0.01', 'tanhC128_s0.01', 'tanhC256_s0.01'):
        r = load(tag)
        if r is None: continue
        out = []
        for l in sorted(r['layers'], key=int):
            w = int(''.join(ch for ch in tag.split('_')[0] if ch.isdigit()) or 0)
            key = (f'arc_n{w}' if tag.startswith('arc') else f'tanh_g3_n{w}')
            fl = fano_lower(r, int(l), l1key=key)
            if fl: out.append(f'l{l}: H_lo={fl["H_lo"]:.1f}; min-f rho={fl["rho"]:.2f} -> {fl["lower"]:.1f}' + (f'; tree rho={fl["rho_tree"]:.2f} -> {fl["lower_tree"]:.1f}' if 'rho_tree' in fl else ''))
        print(f'{tag}: ' + '; '.join(out) + '\n')
