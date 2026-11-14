"""Tables for the random-phases report: python analyze.py > results/tables.md"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MFP = json.load(open(os.path.join(HERE, 'mf_predictions.json')))
REGS = ['tanh-ord2', 'tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he', 'relu-ln']
SIG = [0.001, 0.003, 0.01, 0.03, 0.1]
LAM_MF = {'tanh-ord2': -0.218, 'tanh-ord': -0.138, 'tanh-crit': 0.0, 'tanh-chaos': 0.147, 'tanh-chaos2': 0.306,
          'relu-he': 0.0, 'relu-ln': 0.192}
CHI_C = {'tanh-chaos': 0.867, 'tanh-chaos2': 0.800, 'tanh-ord': 0.759, 'tanh-ord2': 0.647}


def crossing(R, level=0.5):
    for l in range(1, len(R)):
        a, b = R[l - 1], R[l]
        if a is None or b is None:
            continue
        if np.isfinite(a) and np.isfinite(b) and a >= level > b:
            return l + (a - level) / (a - b)
    return float('nan')


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '-'
    if isinstance(x, (int, np.integer)):
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.1e}"
    return f"{x:.{nd}g}"


def load_branch():
    D = defaultdict(list)
    for f in glob.glob(os.path.join(HERE, 'results', 'branch', '*.json')):
        r = json.load(open(f))
        D[(r['regime'], r['n'], r['sigma'])].append(r)
    return D


def load_backward():
    D = {}
    for f in glob.glob(os.path.join(HERE, 'results', 'backward', '*.json')):
        r = json.load(open(f))
        D[(r['regime'], r['n'], r['sigma'], r['seed'])] = r
    return D


def branch_tables(B):
    out = []
    P = lambda s: out.append(s)
    P('## 1. Forgetting: input share R(0 -> L) and L* (depth where R crosses 0.5)\n')
    P('Seed means over the seeds present (see the run manifest at the end); mean-field (MF) value in brackets.\n')
    summ = {}
    for n in (256, 64):
        P(f'### width {n}\n')
        P('| regime | lambda_1 (QR) | lambda_typ | lambda_mf | ' + ' | '.join(f'L* s={s}' for s in SIG) +
          ' | slope dL*/dln(1/s) | 1/|lambda_1| | 1/|lambda_mf| |')
        P('|---|---|---|---|' + '---|' * len(SIG) + '---|---|---|')
        for reg in REGS:
            Ls = []; lam1 = []; lamt = []
            cells = []
            for s in SIG:
                rs = B.get((reg, n, s), [])
                if not rs:
                    cells.append('-'); Ls.append(float('nan')); continue
                ls = [crossing(r['R']['0']) for r in rs]
                lam1 += [r['lyap']['lam'][0] for r in rs]; lamt += [r['lyap']['lam_typ'] for r in rs]
                m = float(np.nanmean(ls)) if np.any(np.isfinite(ls)) else float('nan')
                Ls.append(m)
                mfL = MFP[reg]['by_sigma'][str(s)]['Lstar']
                spread = (f"{np.nanmin(ls):.1f}-{np.nanmax(ls):.1f}" if len(ls) > 1 and np.all(np.isfinite(ls)) else '')
                cells.append(f"{fmt(m)} ({fmt(mfL)})" + (f" [{spread}]" if spread else ''))
            x = np.log(1 / np.array(SIG)); y = np.array(Ls)
            ok = np.isfinite(y) & (y <= 62)
            slope = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else float('nan')
            l1 = float(np.mean(lam1)) if lam1 else float('nan'); lt = float(np.mean(lamt)) if lamt else float('nan')
            summ[(reg, n)] = dict(Ls=Ls, slope=slope, lam1=l1, lamt=lt)
            P(f"| {reg} | {fmt(l1)} | {fmt(lt)} | {LAM_MF[reg]:+.3f} | " + ' | '.join(cells) +
              f" | {fmt(slope)} | {fmt(1 / abs(l1)) if l1 == l1 and abs(l1) > 1e-3 else '-'} | "
              f"{fmt(1 / abs(LAM_MF[reg])) if LAM_MF[reg] else '-'} |")
        P('')
    # increments for critical
    P('### Increments of L* per sigma step (sigma 0.1 -> 0.03 -> 0.01 -> 0.003 -> 0.001), seed means\n')
    P('| regime | n | L* (0.1, 0.03, 0.01, 0.003, 0.001) | increments |')
    P('|---|---|---|---|')
    for n in (256, 64):
        for reg in REGS:
            if (reg, n) not in summ:
                continue
            Ls = summ[(reg, n)]['Ls'][::-1]
            inc = [Ls[i + 1] - Ls[i] for i in range(len(Ls) - 1)]
            P(f"| {reg} | {n} | {', '.join(fmt(v) for v in Ls)} | {', '.join(fmt(v) for v in inc)} |")
    P('')
    P('### R(0 -> L) at L = 8, 16, 32, 48, 64 (n = 256, seed mean; MF in brackets at 32 and 64; jackknife SE of seed 0 at 64)\n')
    P('| regime | sigma | R8 | R16 | R32 (MF) | R48 | R64 (MF) | SE64 |')
    P('|---|---|---|---|---|---|---|---|')
    for reg in REGS:
        for s in SIG:
            rs = B.get((reg, 256, s), [])
            if not rs:
                continue
            Rm = np.mean([r['R']['0'] for r in rs], axis=0)
            mf = MFP[reg]['by_sigma'][str(s)]
            P(f"| {reg} | {s} | {fmt(Rm[7])} | {fmt(Rm[15])} | {fmt(Rm[31])} ({fmt(mf['R32'])}) | {fmt(Rm[47])} | "
              f"{fmt(Rm[63])} ({fmt(mf['R64'])}) | {fmt(next((r for r in rs if r['seed'] == 0), rs[0])['R_se']['0'][63])} |")
    P('')
    # TV proxy
    P('### Inputs still distinguishable? Projected equal-covariance Gaussian diagnostic TV_G = 2 Phi(sqrt(S/2)) - 1, S = tr(Sigma_w^+ Sigma_b) (n = 256, seed 0)\n')
    P('TV floor: about 0.1-0.2 (S has sampling SD about 0.05-0.1 with M = 256, K = 32; deep chaotic layers with R ~ 0 read 0.1-0.18).\n')
    P('| regime | sigma | L* (R = 0.5) | first L with TV < 0.5 | TV at L* | min R with TV >= 0.5 | max R with TV < 0.5 | TV at L = 64 | S at 64 |')
    P('|---|---|---|---|---|---|---|---|---|')
    for reg in REGS:
        for s in SIG:
            rs = [r for r in B.get((reg, 256, s), []) if r['seed'] == 0]
            if not rs:
                continue
            r = rs[0]; TV = r['S']['TV']; R0 = r['R']['0']; S = r['S']['S']
            Ls = crossing(R0)
            # layer 1 has no noise inside the branch (noise enters after a_1), so TV is defined from L = 2
            ltv = next((l + 1 for l in range(1, 64) if TV[l] is not None and TV[l] < 0.5), None)
            tvL = TV[int(round(Ls)) - 1] if np.isfinite(Ls) else None
            minR = min([R0[l] for l in range(1, 64) if TV[l] is not None and TV[l] >= 0.5] or [float('nan')])
            maxR_lowTV = max([R0[l] for l in range(1, 64) if TV[l] is not None and TV[l] < 0.5] or [float('nan')])
            P(f"| {reg} | {s} | {fmt(Ls)} | {ltv if ltv else '>64'} | {fmt(tvL)} | {fmt(minR)} | {fmt(maxR_lowTV)} | {fmt(TV[63])} | {fmt(S[63])} |")
    P('')
    # backward share R(d -> L) stationarity
    P('### Backward share R(m -> 64): share of Var(a_64) explained by the state at depth m (n = 256, seed mean)\n')
    ms = ['0', '16', '32', '48', '56']
    P('| regime | sigma | ' + ' | '.join(f'm={m}' for m in ms) + ' | MF R(32 -> 64) |')
    P('|---|---|' + '---|' * len(ms) + '---|')
    for reg in REGS:
        for s in SIG:
            rs = B.get((reg, 256, s), [])
            if not rs:
                continue
            vals = [np.mean([r['R'][m][63] for r in rs]) for m in ms]
            P(f"| {reg} | {s} | " + ' | '.join(fmt(v) for v in vals) + f" | {fmt(MFP[reg]['by_sigma'][str(s)]['Rback32_to_64'])} |")
    P('')
    return out, summ


def struct_tables(B, BW):
    out = []
    P = lambda s: out.append(s)
    P('## 3. Structure: gates, margins, collapse (relu-he; relu-ln for contrast)\n')
    P('From the truth runs (n = 256, seed 0; n = 64 seeds 0-1): per-neuron statistics over inputs and noise.\n')
    for n in (256, 64):
        P(f'### width {n}, relu-he\n')
        P('| sigma | layer | mixed gates (p in (0,1)) | min(p,1-p) > 0.01 | > 0.1 | dead | margin < 1 | < 2 | < 3 | median margin | PR | k90 | k99 | mean share | cos(PC1, mean) |')
        P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for s in (0.0, 0.001, 0.01, 0.1):
            r = BW.get(('relu-he', n, s, 0))
            if r is None:
                continue
            for l in (1, 2, 4, 8, 16, 32, 48, 64):
                st = r['struct'][l - 1]
                P(f"| {s} | {l} | {fmt(st['mixed_any'])} | {fmt(st['mixed_01'])} | {fmt(st['mixed_10'])} | {fmt(st['p_dead'])} | "
                  f"{fmt(st['margin_lt1'])} | {fmt(st['margin_lt2'])} | {fmt(st['margin_lt3'])} | {fmt(st['margin_median'])} | "
                  f"{fmt(st['pr'])} | {st['k90']} | {st['k99']} | {fmt(st['mean_share'])} | {fmt(st['cos_pc1_mean'])} |")
        P('')
    # power-law fit of margin < 1 fraction vs depth
    P('### Near-boundary fraction (total margin < 1) versus depth: fit frac ~ c l^a over l = 4..64\n')
    P('| regime | n | sigma | a | c | frac at 16 / 32 / 64 | count n*frac at 32 / 64 |')
    P('|---|---|---|---|---|---|---|')
    for reg in ('relu-he',):
        for n in (64, 256):
            for s in (0.0, 0.001, 0.01, 0.1):
                rr = [BW.get((reg, n, s, sd)) for sd in (0, 1)]
                rr = [r for r in rr if r is not None]
                if not rr:
                    continue
                fr = np.mean([[r['struct'][l - 1]['margin_lt1'] for l in range(1, 65)] for r in rr], axis=0)
                ls = np.arange(4, 65); y = fr[3:64]
                ok = y > 0
                a, lc = np.polyfit(np.log(ls[ok]), np.log(y[ok]), 1) if ok.sum() > 5 else (float('nan'), float('nan'))
                P(f"| {reg} | {n} | {s} | {fmt(a)} | {fmt(math.exp(lc))} | {fmt(fr[15])} / {fmt(fr[31])} / {fmt(fr[63])} | "
                  f"{fmt(n * fr[31])} / {fmt(n * fr[63])} |")
    P('')
    # noise-only gate stats from branch runs
    P('### Noise-only gate statistics (branch runs, m = 0 cloud: K = 32 noise draws per input), relu-he, seed mean\n')
    P('| n | sigma | layer | noise-flip fraction | noise margin < 1 | < 2 | < 3 | PR(Sigma_b) | PR(Sigma_w) | PR(total) |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for n in (256, 64):
        for s in SIG:
            rs = B.get(('relu-he', n, s), [])
            if not rs:
                continue
            for l in (8, 16, 32, 64):
                g = lambda k: np.mean([r['gate'][k][l - 1] for r in rs])
                c = lambda k: np.mean([r['collapse'][k][l - 1]['pr'] for r in rs])
                P(f"| {n} | {s} | {l} | {fmt(g('noise_flip'))} | {fmt(g('noise_margin_lt1'))} | {fmt(g('noise_margin_lt2'))} | "
                  f"{fmt(g('noise_margin_lt3'))} | {fmt(c('b'))} | {fmt(c('w'))} | {fmt(c('tot'))} |")
    P('')
    P('### Chaotic and other tanh nets: saturation (fraction |a| > 0.9) and collapse, truth runs n = 256 seed 0\n')
    P('| regime | sigma | sat@16 | sat@32 | sat@64 | PR@16 | PR@64 | mean share@64 |')
    P('|---|---|---|---|---|---|---|---|')
    for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2'):
        for s in (0.0, 0.001, 0.01, 0.1):
            r = BW.get((reg, 256, s, 0))
            if r is None:
                continue
            st = r['struct']
            P(f"| {reg} | {s} | {fmt(st[15]['sat'])} | {fmt(st[31]['sat'])} | {fmt(st[63]['sat'])} | {fmt(st[15]['pr'])} | "
              f"{fmt(st[63]['pr'])} | {fmt(st[63]['mean_share'])} |")
    P('')
    return out


def kstar(row, L, d_list, tau):
    """smallest k = L - d with nMSE <= tau at that k and at every larger k in the grid."""
    ks = sorted([(L - d, row.get(f'{d}')) for d in d_list if d < L and row.get(f'{d}') is not None])
    best = None
    for i in range(len(ks) - 1, -1, -1):
        k, v = ks[i]
        if v is not None and np.isfinite(v) and v <= tau:
            best = k
        else:
            break
    return best


def backward_tables(BW):
    out = []
    P = lambda s: out.append(s)
    P('## 2. Backward-only estimation of E[a_L]\n')
    for n in (256, 64):
        for L in (64, 32):
            P(f'### n = {n}, L = {L}: nMSE of stand-ins at depth d = L - k pushed exactly (sampling) through the last k layers\n')
            P('| regime | sigma | stand-in | ' + ' | '.join(f'k={k}' for k in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 60, 62, 63) if k < L) +
              ' | k*(1e-2) | k*(1e-3) | k*(1e-4) |')
            ks_all = [k for k in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 60, 62, 63) if k < L]
            P('|---|---|---|' + '---|' * len(ks_all) + '---|---|---|')
            for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he'):
                for s in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
                    r = BW.get((reg, n, s, 0))
                    if r is None:
                        continue
                    depths = r['depths']
                    for kind in ('MF', 'MM'):
                        row = {}
                        for d in depths:
                            v = r['est'].get(f'{kind}_{d}', {}).get(str(L))
                            if v is not None:
                                row[str(d)] = v
                        cells = [fmt(row.get(str(L - k))) for k in ks_all]
                        ksx = [kstar(row, L, depths, t) for t in (1e-2, 1e-3, 1e-4)]
                        P(f"| {reg} | {s} | {kind} | " + ' | '.join(cells) + ' | ' +
                          ' | '.join(str(k) if k is not None else 'none' for k in ksx) + ' |')
            P('')
    P('### Absolute accuracy: relative L2 error of the mean vector, sqrt(nMSE x mean Var / mean E[a]^2), MF stand-in, L = 64 (seed 0)\n')
    P('| regime | n | sigma | k=8 | k=16 | k=32 | k=48 | Var/E[a]^2 at L=64 |')
    P('|---|---|---|---|---|---|---|---|')
    for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he'):
        for n in (256, 64):
            for s in (0.0, 0.001, 0.01, 0.1):
                r = BW.get((reg, n, s, 0))
                if r is None:
                    continue
                st = r['struct'][63]; ratio = st['var_mean'] / st['mean_sq'] if st['mean_sq'] > 0 else float('nan')
                vals = [r['est'].get(f'MF_{64 - k}', {}).get('64') for k in (8, 16, 32, 48)]
                P(f"| {reg} | {n} | {s} | " + ' | '.join(fmt(math.sqrt(max(v * ratio, 0))) if v is not None else '-' for v in vals) + f" | {fmt(ratio)} |")
    P('')
    P('### Scale-matched mean-field stand-in MFr (relu-he, amendment 1) vs MF, L = 64 (seed 0)\n')
    P('| n | sigma | stand-in | k=1 | k=4 | k=8 | k=16 | k=24 | k=32 | k=40 | k=48 | k=56 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in (256, 64):
        for s in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
            r = BW.get(('relu-he', n, s, 0))
            if r is None:
                continue
            for kind in ('MF', 'MFr', 'MM'):
                P(f"| {n} | {s} | {kind} | " + ' | '.join(fmt(r['est'].get(f'{kind}_{64 - k}', {}).get('64')) for k in (1, 4, 8, 16, 24, 32, 40, 48, 56)) + ' |')
    P('')
    P('### Output law beyond the mean: relative RMS error of per-neuron Var(a_L) (amendment 3; floor ~ sqrt(2/NS)), L = 64, seed 0\n')
    P('| regime | n | sigma | MF k=8 | MF k=16 | MF k=32 | MF k=48 | MM k=8 | MM k=32 | aniso | point | gauss control |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he'):
        for n in (256, 64):
            for s in (0.0, 0.001, 0.01, 0.1):
                r = BW.get((reg, n, s, 0))
                if r is None or 'law_aniso_var' not in r['est']:
                    continue
                e = r['est']
                g = lambda key: fmt(e.get(key, {}).get('64'))
                P(f"| {reg} | {n} | {s} | {g('MF_56_var')} | {g('MF_48_var')} | {g('MF_32_var')} | {g('MF_16_var')} | "
                  f"{g('MM_56_var')} | {g('MM_32_var')} | {g('law_aniso_var')} | {g('law_point_var')} | {g('law_gauss_control_var')} |")
    P('')
    # MF decay rate fit (chaotic)
    P('### MF stand-in decay with k in the chaotic nets (L = 64): fitted rate of -ln nMSE per layer vs ln(1/chi_c)\n')
    P('| regime | n | sigma | fitted rate | ln(1/chi_c) | points used |')
    P('|---|---|---|---|---|---|')
    for reg in ('tanh-chaos', 'tanh-chaos2', 'tanh-ord'):
        for n in (256, 64):
            for s in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
                r = BW.get((reg, n, s, 0))
                if r is None:
                    continue
                pts = []
                floor = 5.0 / r['NS']
                for d in r['depths']:
                    v = r['est'].get(f'MF_{d}', {}).get('64')
                    if v is not None and v > floor:
                        pts.append((64 - d, math.log(v)))
                if len(pts) >= 3:
                    k, y = np.array(pts).T
                    rate = -np.polyfit(k, y, 1)[0]
                else:
                    rate = float('nan')
                P(f"| {reg} | {n} | {s} | {fmt(rate)} | {fmt(-math.log(CHI_C[reg]))} | {len(pts)} |")
    P('')
    # input-law swaps
    P('### Input-law swaps: nMSE of E[a_L] under another input law vs the truth (seed 0)\n')
    P('| regime | n | sigma | law | L=4 | L=8 | L=16 | L=24 | L=32 | L=48 | L=64 | control (gauss) at 64 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he'):
        for n in (256, 64):
            for s in (0.0, 0.001, 0.01, 0.1):
                r = BW.get((reg, n, s, 0))
                if r is None:
                    continue
                for law in ('aniso', 'point'):
                    e = r['est'][f'law_{law}']
                    P(f"| {reg} | {n} | {s} | {law} | " + ' | '.join(fmt(e[str(L)]) for L in (4, 8, 16, 24, 32, 48, 64)) +
                      f" | {fmt(r['est']['law_gauss_control']['64'])} |")
    P('')
    # deductive vs sampling
    P('### Deductive: ARC kprop K = 2 started from the input-independent MF stand-in at depth L - k, or from the input (ARC standard), vs plain sampling at equal FLOPs\n')
    P('gain = (1/N_eq) / nMSE_kprop with N_eq = kprop FLOPs (ARC counter) / flopscope cost of one depth-L path; > 1 means kprop wins. '
      'nMSE below the resolution floor 2/NT is shown as < floor and the gain as a lower bound. MM-started kprop uses the true moments '
      '(oracle) and is listed separately as a closure check.\n')
    P('| regime | n | sigma | L | start | k | nMSE kprop | N_eq | MC nMSE | gain (ARC count) | gain (raw count, /1.34) |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    best = {}
    for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he'):
        for n in (256, 64):
            for s in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
                r = BW.get((reg, n, s, 0))
                if r is None:
                    continue
                floor = 2.0 / r['NT']
                for L in (64, 32):
                    fp = r['flops_path'][str(L)]
                    for start in ('input', 'MF', 'MM'):
                        if start == 'input':
                            cand = [(L, r['est'].get('kprop_input', {}).get(str(L)), r['est'].get('kprop_input_flops', {}).get(str(L)))]
                        else:
                            cand = []
                            for d in sorted(r['depths'], reverse=True):
                                e = r['est'].get(f'kprop{start}_{d}')
                                if e is None or d >= L:
                                    continue
                                cand.append((L - d, e.get(str(L)), r['est'][f'kprop{start}_{d}_flops'].get(str(L))))
                        for k, v, fl in cand:
                            if v is None or fl is None:
                                continue
                            Neq = fl / fp; mc = 1.0 / Neq
                            if v < floor:
                                vs, gs, gr = f'< {floor:.0e}', f'> {mc / floor:.3g}', f'> {mc / floor / 1.34:.3g}'
                            else:
                                vs, gs, gr = fmt(v), fmt(mc / v), fmt(mc / v / 1.34)
                            best.setdefault((reg, n, s, L, start), []).append((k, v, Neq, mc))
                            lab = start if start != 'MM' else 'MM (oracle)'
                            P(f"| {reg} | {n} | {s} | {L} | {lab} | {k} | {vs} | {fmt(Neq)} | {fmt(mc)} | {gs} | {gr} |")
    P('')
    P('### ARC K = 2 from the input: nMSE by depth (seed 0)\n')
    P('| regime | n | sigma | L=1 | L=4 | L=8 | L=16 | L=32 | L=48 | L=64 |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for reg in ('tanh-ord', 'tanh-crit', 'tanh-chaos', 'tanh-chaos2', 'relu-he'):
        for n in (256, 64):
            for s in (0.0, 0.001, 0.01, 0.1):
                r = BW.get((reg, n, s, 0))
                if r is None or 'kprop_input' not in r['est']:
                    continue
                e = r['est']['kprop_input']
                P(f"| {reg} | {n} | {s} | " + ' | '.join(fmt(e[str(L)]) for L in (1, 4, 8, 16, 32, 48, 64)) + ' |')
    P('')
    return out, best


def resolution_table(BW, B):
    out = []
    P = lambda s: out.append(s)
    P('## Resolution and run manifest\n')
    P('Gaussian control: a second, independent plain Monte Carlo run from the true input law, scored like a stand-in; its '
      'debiased nMSE should be 0 up to noise. RMS and max |.| over depths 2-64 give the resolution of every nMSE in that run.\n')
    P('| regime | n | sigma | seed | NT | NS | control RMS | control max abs | control at L = 64 |')
    P('|---|---|---|---|---|---|---|---|---|')
    for key in sorted(BW):
        r = BW[key]; c = r['est']['law_gauss_control']
        v = np.array([c[str(L)] for L in range(2, 65)])
        P(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {r['NT']} | {r['NS']} | {fmt(float(np.sqrt((v ** 2).mean())))} | "
          f"{fmt(float(np.abs(v).max()))} | {fmt(c['64'])} |")
    P('')
    P('Branch runs present (regime, width, sigma: seeds):\n')
    for key in sorted(B):
        P(f"- {key[0]}, n = {key[1]}, sigma = {key[2]}: seeds {sorted(r['seed'] for r in B[key])}")
    P('')
    return out


if __name__ == '__main__':
    B = load_branch(); BW = load_backward()
    print(f'# Random-net phases: tables (regenerated by analyze.py; {sum(len(v) for v in B.values())} branch runs, {len(BW)} backward runs)\n')
    o1, summ = branch_tables(B)
    print('\n'.join(o1))
    o2, best = backward_tables(BW)
    print('\n'.join(o2))
    print('\n'.join(struct_tables(B, BW)))
    print('\n'.join(resolution_table(BW, B)))
