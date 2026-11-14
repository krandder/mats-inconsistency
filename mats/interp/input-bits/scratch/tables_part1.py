"""Part 1 tables for the report -> results/part1_tables.md (medians over the 4 evaluation nets unless stated)."""
import glob, json, math, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

D_ = '/home/agent/workspaces/mats/work/input-bits'
S = [4000, 4001, 4002, 4003]
med = lambda v: float(np.median(v))
out = []


def J2(t):
    return 3 * math.sin(t) * math.cos(t) + (math.pi - t) * (1 + 2 * math.cos(t) ** 2)


rho = 0.0; ker = {}; Lam = 0.0; b1 = 1.0
for l in range(1, 2049):
    th = math.acos(rho); Lam += 6 - (2 / math.pi) * J2(th); b1 *= (math.pi - th) / math.pi
    ham = th / math.pi; rho = float(ib.kappa(rho))
    ker[l] = dict(delta=1 - rho, C=Lam / 4, hamfrac=ham, b1=b1)
out.append('## Table 1. Three-term split of tr Var_x(a): medians over 4 nets (min-max), exact chain (annealed), kernel + 1/n')
out.append('| n | L | s_rad | s_amp | s_dir | chain s_amp / s_dir | V (median) | chain V | V_dir/V_amp: nets / chain / kernel 4n delta/Lambda | noise share at sigma 1e-3 (max over nets) |')
out.append('|---|---|---|---|---|---|---|---|---|---|')
for n in (32, 64, 128, 256):
    A = [json.load(open(f'{D_}/runs/p1a_n{n}_s{s}.json')) for s in S]
    ch = json.load(open(f'{D_}/runs/p1chain_n{n}.json'))
    for L in sorted(A[0]['split'], key=int):
        v = [a['split'][L] for a in A]
        c = ch['chainA'].get(L)
        rat = med([x['V_dir'] / x['V_amp'] for x in v if x['V_amp'] > 0])
        kr = 4 * n * ker[int(L)]['delta'] / (4 * ker[int(L)]['C'])
        nz = max(a['noise'][L]['coupled_diff_var'] for a in A)
        f = lambda k: f"{med([x[k] for x in v]):.3f} ({min(x[k] for x in v):.2f}-{max(x[k] for x in v):.2f})"
        out.append(f"| {n} | {L} | {f('s_rad')} | {f('s_amp')} | {f('s_dir')} | {c['s_amp']:.3f} / {c['s_dir']:.3f} | {med([x['V'] for x in v]):.3g} | "
                   f"{c['V']:.3g} | {rat:.3g} / {c['ratio_dir_amp_chain']:.3g} / {kr:.3g} | {nz:.2g} |")
out.append('')
out.append('## Table 2. Input gate bits by layer: Hamming count between two independent inputs (median nets / chain / kernel n theta/pi), and H_marg / Hamming')
out.append('| n | l=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | H/Ham at l = 8, 16, 32 |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|')
for n in (32, 64, 128, 256):
    A = [json.load(open(f'{D_}/runs/p1a_n{n}_s{s}.json')) for s in S]
    ch = {x['l']: x['ham'] for x in json.load(open(f'{D_}/runs/p1chain_n{n}.json'))['ham_layers']}
    m = min(len(a['layers']) for a in A)
    hm = np.median([[x['hamming'] for x in a['layers'][:m]] for a in A], 0)
    hr = np.median([[x['H_marg'] / max(x['hamming'], 1e-9) for x in a['layers'][:m]] for a in A], 0)
    cells = []
    for l in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        cells.append(f'{hm[l - 1]:.1f} / {ch[l]:.1f} / {n * ker[l]["hamfrac"]:.1f}' if l <= m else '-')
    out.append(f'| {n} | ' + ' | '.join(cells) + f' | {hr[7]:.2f}, {hr[15]:.2f}, {hr[31] if m >= 32 else float("nan"):.2f} |')
out.append('')
out.append('## Table 3. Interaction structure over input coordinates (medians over 4 nets)')
out.append('| n | L | W1 | S1 (deg <= 4) | W2 (off-diag) | s_even | mean dim D | chain D_ann | rotated S1 / W2_off | |a|: W1, S1 |')
out.append('|---|---|---|---|---|---|---|---|---|---|')
for n in (32, 64, 128, 256):
    B = [json.load(open(f'{D_}/runs/p1b_n{n}_s{s}.json')) for s in S]
    ch = json.load(open(f'{D_}/runs/p1chain_n{n}.json'))
    for L in sorted(B[0]['by_depth'], key=int):
        v = [b['by_depth'][L] for b in B]
        g = lambda k: med([x[k] for x in v])
        cd = ch['chainA'].get(L, {}).get('D_ann', float('nan'))
        out.append(f"| {n} | {L} | {g('W1'):.3f} | {g('S1'):.3f} | {g('W2'):.3f} ({g('W2_off'):.3f}) | {g('s_even'):.3f} | {g('D'):.2f} "
                   f"({min(x['D'] for x in v):.1f}-{max(x['D'] for x in v):.1f}) | {cd:.2f} | {g('rot_S1'):.3f} / {g('rot_W2_off'):.3f} | {g('amp_W1'):.3f}, {g('amp_S1'):.3f} |")
out.append('')
out.append('## Table 4. Input information by Lyapunov direction (medians over 4 nets; Gamma from the bug-fixed rerun if present)')
out.append('| n | l | finite angular fraction / kernel delta_{l} | finite angular PR | tangent angular PR | n/(2l)+1 | finite radial share | Gamma (infinitesimal angle growth) | Gamma / finite fraction |')
out.append('|---|---|---|---|---|---|---|---|---|')
for n in (64, 256):
    C = [json.load(open(f'{D_}/runs/p1c_n{n}_s{s}.json')) for s in S]
    fixf = sorted(glob.glob(f'{D_}/runs/p1cfix_n{n}_s*.json'))
    Cf = [json.load(open(f)) for f in fixf] if len(fixf) == 4 else None
    for l in sorted(C[0]['layers'], key=int):
        v = [c['layers'][l] for c in C if l in c['layers']]
        gsrc = [c['layers'][l] for c in Cf if l in c['layers']] if Cf else [dict(x, Gamma_mean=x['Gamma_mean'] * 64 / n) for x in v]
        fa = med([x['finite']['ang_rel'] for x in v])
        gm = med([x['Gamma_mean'] for x in gsrc])
        out.append(f"| {n} | {l} | {fa:.3g} / {ker[int(l)]['delta'] if int(l) > 0 else 1:.3g} | {med([x['finite']['ang_PR'] for x in v]):.1f} | "
                   f"{med([x['tangent']['ang_PR'] for x in v]):.1f} | {n / (2 * int(l)) + 1:.1f} | {med([x['finite']['radial_share'] for x in v]):.3f} | "
                   f"{gm:.3g} | {gm / max(fa, 1e-300):.3g} |")
open(f'{D_}/results/part1_tables.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
