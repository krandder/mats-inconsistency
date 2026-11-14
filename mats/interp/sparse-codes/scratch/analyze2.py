"""Tables for the sparse-codes report (v2 outputs): python scratch/analyze2.py > results/tables.md
Reads results/gpu/<run>/<net>/{e1_L*.json, e2_L*.json, e3.json, truth.npz, e4.json, truth_<law>.npz} (v2 runs only:
directories listed in V2RUNS), and results/kprop*/*.json. Verdicts follow PREREG amendment 2."""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results')
V2RUNS = ['sc-a3big', 'sc-b2', 'sc-c2', 'sc-b2_cut24']
NETDIRS = {}
for run in V2RUNS:
    for d in sorted(glob.glob(os.path.join(R, 'gpu', run, '*'))):
        if os.path.isdir(d):
            NETDIRS.setdefault(os.path.basename(d), []).append(d)


def load(name, fn):
    for d in NETDIRS.get(name, []):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return json.load(open(p)) if fn.endswith('.json') else np.load(p)
    return None


def nn(fam, n, sigma, seed=65):
    return f"{fam}_n{n}_s{sigma:g}_seed{seed}"


def f3(x):
    if x is None:
        return '-'
    if isinstance(x, str):
        return x
    if x == 0:
        return '0'
    a = abs(x)
    if a >= 1000 or a < 0.01:
        return f"{x:.2g}"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def e(x):
    return '-' if x is None else f"{x:.1e}"


LAYERS = {'arc': [1, 2, 4, 8, 16, 24, 31], 'tord': [1, 2, 4, 8, 31], 'tcha': [1, 8, 31]}
WIDTHS = [64, 128, 256]
CONFIGS = [('arc', 0.01, 65), ('arc', 0.01, 66), ('arc', 0.001, 65), ('tord', 0.01, 65), ('tcha', 0.01, 65)]
out = []
P = out.append

# ======================================================================================================== E1
P("# E1. Dictionaries in bits\n")
P("Noise units = E||a - a_hat||^2 / (n s^2) per coordinate; below 1 = below the noise. Bits per layer = noise units x n /"
  " (2 ln 2). Held out. 'mean' = TopK SAE with learned b_dec (initialised at the mean); 'zero' = b_dec fixed at 0, first n"
  " atoms initialised at the neuron basis (arc only, F = n, 4n). Cells: best over F of raw (refit) [F/n].\n")
KT1 = {}
for fam, sig, seed in CONFIGS:
    for n in WIDTHS:
        name = nn(fam, n, sig, seed)
        rows = []
        for l in LAYERS[fam]:
            d = load(name, f'e1_L{l}.json')
            if d is not None:
                rows.append((l, d))
        if not rows:
            continue
        KS = [n // 16, n // 8, n // 4, n // 2]
        P(f"\n## {name}\n")
        P("| layer | s | var/coord [noise u.] | PR | eig>s^2 | k_PCA below noise | mean only | " + " | ".join(
            f"k={k}: SAE mean / SAE zero / PCA / neuron" for k in KS) + " | KT1 |")
        P("|" + "---|" * (7 + len(KS) + 1))
        for l, d in rows:
            cells = []
            for k in KS:
                def best(origin):
                    c = [r for r in d['sae'] if r['k'] == k and r.get('origin', 'mean') == origin]
                    if not c:
                        return '-'
                    b = min(c, key=lambda r: min(r['noise_units'], r['noise_units_refit']))
                    return f"{f3(b['noise_units'])} ({f3(b['noise_units_refit'])}) [{b['F'] // n}]"
                neu = d['neuron_topk'][str(k)]['noise_units']
                if 'neuron_topk_zero' in d:
                    neu = min(neu, d['neuron_topk_zero'][str(k)]['noise_units'])
                cells.append(f"{best('mean')} / {best('zero')} / {f3(d['pca'][str(k)]['noise_units'])} / {f3(neu)}")
            cand = [min(r['noise_units'], r['noise_units_refit']) for r in d['sae'] if r['k'] <= n // 4]
            # controls also count (same F, other training) for KT1
            for v in d.get('controls', {}).values():
                cand += [min(r['noise_units'], r['noise_units_refit']) for r in v['res'] if r['k'] <= n // 4]
            bestv = min(cand) if cand else None
            fires = bestv is None or bestv >= 1
            near = bestv is not None and 1 <= bestv < 3
            KT1[(fam, sig, seed, n, l)] = (fires, bestv)
            P(f"| {l} | {d['s']:.2g} | {f3(d['var_per_coord'] / d['s'] ** 2)} | {d['PR']:.1f} | {d['eig_above_noise']} | "
              f"{d['k_pca_below_noise']} | {f3(d['pca']['0']['noise_units'])} | " + " | ".join(cells) +
              f" | {'fires' if fires else 'no'}{' (near)' if near else ''} ({f3(bestv)}) |")
        if fam == 'arc':
            P("\nNeuron code (exact at L0 = #active): active mean / sd / min / max / ever active: " + '; '.join(
                f"L{l}: {d['active']['mean']:.1f} / {d['active']['sd']:.1f} / {d['active']['min']} / {d['active']['max']} / {d['active']['ever_active']}"
                for l, d in rows))
        P("\nError vs F at k = n/8 and n/4 (mean origin, raw noise units; F/n = 1, 2, 4, 8, 16) and dead fraction at 16n:")
        for l, d in rows:
            parts = []
            for k in (n // 8, n // 4):
                byF = {r['F'] // n: r for r in d['sae'] if r['k'] == k and r.get('origin', 'mean') == 'mean'}
                parts.append(f"k={k}: " + ', '.join(f3(byF[fm]['noise_units']) if fm in byF else '-' for fm in (1, 2, 4, 8, 16))
                             + f" (dead {f3(byF[16]['dead_frac']) if 16 in byF else '-'})")
            # learning curve: last vs middle checkpoint, F = 4n mean origin, k = n/4
            cv = d.get('curves', {}).get(f'mean_F{4 * n}')
            lc = ''
            if cv:
                ki = [n // 16, n // 8, n // 4, n // 2].index(n // 4)
                mid = cv[len(cv) // 2][1][ki]; last = cv[-1][1][ki]
                lc = f"; curve F=4n k=n/4: mid {f3(mid)} -> end {f3(last)}"
            P(f"- layer {l}: " + ' | '.join(parts) + lc)
        for l, d in rows:
            if 'controls' not in d:
                continue
            base = {(r.get('origin', 'mean'), r['k']): r for r in d['sae'] if r['F'] == 4 * n}
            P(f"\nControls at layer {l} (F = 4n; raw (refit) noise units by k):")
            for nm_, v in d['controls'].items():
                origin = 'zero' if '_zero_' in nm_ else 'mean'
                P(f"- {nm_}: " + '; '.join(
                    f"k={r['k']}: base {f3(base[(origin, r['k'])]['noise_units']) if (origin, r['k']) in base else '-'} -> "
                    f"{f3(r['noise_units'])} ({f3(r['noise_units_refit'])})" for r in v['res']))

# ======================================================================================================== E2
P("\n\n# E2. One step: latent-conditional Gaussians (all carry the exact mean and full covariance)\n")
P("nMSE of the next layer's E[act(z')] (ARC units, debiased, CONF set, +- jackknife SE). G = Gaussian = one step of K=2."
  " Best sparse = best of SPARSE (mean origin) / SPARSEZ (zero origin) over F, selected on SEL. cap3/cap4 = share of the"
  " per-neuron 3rd/4th cumulants of z' reproduced (CONF). H = support entropy upper bound (sum of feature binary"
  " entropies, bits).\n")
KT2c, KT3c = [], []


def pick(rows, pred):
    c = [r for r in rows if pred(r)]
    if not c:
        return None
    return min(c, key=lambda r: r['sel']['mean_nmse'])


for fam, sig, seed in CONFIGS:
    for n in WIDTHS:
        name = nn(fam, n, sig, seed)
        blocks = [(l, load(name, f'e2_L{l}.json')) for l in LAYERS[fam]]
        blocks = [(l, d) for l, d in blocks if d is not None]
        if not blocks:
            continue
        KS = [n // 16, n // 8, n // 4]
        P(f"\n## {name}\n")
        P("| layer | G | " + " | ".join(f"k={k}: dense (DENSE/DENSEQ) | k={k}: best sparse [model]" for k in KS) + " | kappa3/kappa4 norm |")
        P("|" + "---|" * (2 + 2 * len(KS) + 1))
        for l, d in blocks:
            rows = d['rows']
            G = next(r for r in rows if r['model'] == 'G')
            cells = []
            for k in KS:
                dn = pick(rows, lambda r, k=k: r['model'] in (f'DENSE{k}', f'DENSEQ{k}'))
                sp = pick(rows, lambda r, k=k: r['model'].startswith('SPARSE') and r['k'] == k)
                cd = dn['conf'] if dn else None; cs = sp['conf'] if sp else None
                cells.append(f"{e(cd['mean_nmse']) if cd else '-'}+-{e(cd['se']) if cd else '-'}; c3 {f3(cd['cap3']) if cd else '-'}, c4 {f3(cd['cap4']) if cd else '-'} [{dn['model'] if dn else '-'}]")
                cells.append(f"{e(cs['mean_nmse']) if cs else '-'}+-{e(cs['se']) if cs else '-'}; c3 {f3(cs['cap3']) if cs else '-'}, c4 {f3(cs['cap4']) if cs else '-'}; "
                             f"H {f3(cs.get('support_H_indep_bits')) if cs else '-'} [{sp['model'] if sp else '-'}]")
                if fam == 'arc' and sig == 0.01 and seed == 65 and l in (8, 16, 24, 31) and k in (n // 16, n // 8):
                    win = (cs is not None and cd is not None and cs['mean_nmse'] + 2 * cs['se'] < cd['mean_nmse'])
                    KT2c.append((n, l, k, cs['mean_nmse'] if cs else None, cs['se'] if cs else None,
                                 cd['mean_nmse'] if cd else None, win))
            P(f"| {l} | {e(G['conf']['mean_nmse'])}+-{e(G['conf']['se'])} | " + " | ".join(cells) +
              f" | {f3(d['k3T_norm'])} / {f3(d['k4T_norm'])} |")
            if fam == 'arc' and sig == 0.01 and seed == 65 and l in (8, 16, 24, 31):
                sp = pick(rows, lambda r: r['model'].startswith('SPARSE') and r['k'] <= n // 4)
                cs = sp['conf'] if sp else None
                win = cs is not None and cs['mean_nmse'] + 2 * cs['se'] < G['conf']['mean_nmse']
                KT3c.append((n, l, sp['model'] if sp else None, cs['mean_nmse'] if cs else None, cs['se'] if cs else None,
                             G['conf']['mean_nmse'], win))
            if fam != 'arc':
                pl = [(r['model'], r['conf'].get('postLN_nmse')) for r in rows if r['conf'].get('postLN_nmse') is not None]
                if pl:
                    best = {}
                    for m_, v in pl:
                        key = 'G' if m_ == 'G' else ('dense' if m_.startswith('DENSE') else 'sparse')
                        if key not in best or v < best[key][1]:
                            best[key] = (m_, v)
                    P(f"  post-LN next-state mean nMSE (layer {l}): " + '; '.join(f"{k_}: {e(v[1])} [{v[0]}]" for k_, v in best.items()))
        # residual diagnostics
        P("\nResidual given the latent (CONF): |skew| median; excess kurtosis median (|.| p90); after conditional-variance"
          " normalisation: kurtosis median (|.| p90), variance-regression R^2; neuron space -> projected by W_{l+1}; residual size:")
        for l, d in blocks:
            parts = []
            for r in d['rows']:
                c = r['conf']
                if 'resid_neuron_cond_kurt_med' not in c:
                    continue
                if not (r['model'].startswith('SPARSE_F') or r['model'].startswith('SPARSEZ_F') or r['model'].startswith('DENSE')):
                    continue
                parts.append(f"{r['model']}: neuron sk {f3(c['resid_neuron_skew_absmed'])}, ku {f3(c['resid_neuron_kurt_med'])} ({f3(c['resid_neuron_kurt_absp90'])}), "
                             f"cond ku {f3(c['resid_neuron_cond_kurt_med'])} ({f3(c['resid_neuron_cond_kurt_absp90'])}), R2 {f3(c['resid_neuron_var_R2_med'])} -> "
                             f"proj sk {f3(c['resid_proj_skew_absmed'])}, ku {f3(c['resid_proj_kurt_med'])}, cond ku {f3(c['resid_proj_cond_kurt_med'])} "
                             f"({f3(c['resid_proj_cond_kurt_absp90'])}); size {f3(c['resid_noise_units'])} nu")
            if parts:
                P(f"- layer {l}:\n  - " + '\n  - '.join(parts))
if KT2c:
    wins = sum(1 for c in KT2c if c[6])
    P(f"\n**KT2** (arc 0.01 seed 65; sparse wins a cell if nMSE + 2 SE < dense; fires if wins < 12 of 24): {wins} wins of "
      f"{len(KT2c)} cells present (24 registered) -> {'FIRES' if wins < 12 else 'does not fire'}"
      + ('' if len(KT2c) == 24 else ' [INCOMPLETE: missing cells count as not won]'))
    P("Cells (n, layer, k: sparse +- SE vs dense, W = sparse wins): " + '; '.join(
        f"({c[0]},{c[1]},{c[2]}: {e(c[3])}+-{e(c[4])} vs {e(c[5])}{' W' if c[6] else ''})" for c in KT2c))
if KT3c:
    wins = sum(1 for c in KT3c if c[6])
    P(f"\n**KT3** (best sparse k <= n/4 vs G; wins if nMSE + 2 SE < G; fires if wins < 6 of 12): {wins} wins of {len(KT3c)} "
      f"cells present (12 registered) -> {'FIRES' if wins < 6 else 'does not fire'}" + ('' if len(KT3c) == 12 else ' [INCOMPLETE]'))
    P("Cells (n, layer: model, sparse +- SE vs G): " + '; '.join(
        f"({c[0]},{c[1]}: {c[2]}, {e(c[3])}+-{e(c[4])} vs {e(c[5])}{' W' if c[6] else ''})" for c in KT3c))


# ======================================================================================================== kprop
def kprop_nmse(n, sigma, v, L, law='std', seed=65):
    if law == 'std':
        p = os.path.join(R, 'kprop', f'kprop_n{n}_seed{seed}_s{sigma:g}_{v}.json')
        T = load(nn('arc', n, sigma, seed), 'truth.npz')
        if T is None:
            T = load(nn('arc', n, sigma, seed), 'truth_std.npz')
    else:
        p = os.path.join(R, 'kprop_law', f'kprop_n{n}_seed{seed}_s{sigma:g}_{law}_{v}.json')
        T = load(nn('arc', n, sigma, seed), f'truth_{law}.npz')
    if not os.path.exists(p) or T is None:
        return None
    m = np.array(json.load(open(p))['means'][L - 1])
    N = float(T['N']); tm = T['S1'][L - 1] / N; tv = T['S2'][L - 1] / N - tm ** 2
    return float(((m - tm) ** 2).mean() / tv.mean() - 1.0 / N)


# ======================================================================================================== E3
P("\n\n# E3. Oracle representation at a cut, pushed through the real layers (arc)\n")
P("nMSE of E[a_L] (common random numbers; debiased with the model's own variance and the truth's). TRUE = no replacement."
  " K=2 / K3f = ARC's kprop end to end (same net, same truth). Best sparse over F and both origins.\n")
KT4 = {}
for fam, sig, seed in CONFIGS:
    if fam != 'arc':
        continue
    for n in WIDTHS:
        name = nn('arc', n, sig, seed)
        d = load(name, 'e3.json')
        if d is None:
            continue
        rows = d['rows']
        dd = load(name, 'e3dense.json')
        chk = ''
        if dd is not None:
            # amendment 5: dense rows from the rerun; TRUE/G of the rerun must match run_main2's
            old_tg = {(r['cut'], r['L'], r['model']): r['nmse'] for r in rows if r['model'] in ('TRUE', 'G')}
            new_tg = {(r['cut'], r['L'], r['model']): r['nmse'] for r in dd['rows'] if r['model'] in ('TRUE', 'G')}
            diffs = [abs(old_tg[k_] - new_tg[k_]) for k_ in old_tg if k_ in new_tg]
            chk = f"; rerun check: max |TRUE/G difference| = {max(diffs):.1e}" if diffs else ''
            rows = [r for r in rows if not r['model'].startswith('DENSE')] + [r for r in dd['rows'] if r['model'].startswith('DENSE')]
        else:
            rows = [r for r in rows if not r['model'].startswith('DENSE')]      # v2 dense rows are invalid (amendment 5)
            chk = '; DENSE rows pending the rerun (amendment 5)'
        KS = [n // 16, n // 8, n // 4, n // 2]
        P(f"\n## {name} (N3 = {d['N3']}, truth {d['NT']}{chk})\n")
        P("| cut -> L | TRUE | G | " + " | ".join(f"k={k}: DENSE / DENSEQ / best sparse" for k in KS) + " | K=2 e2e | K3f e2e |")
        P("|" + "---|" * (3 + len(KS) + 2))
        for (m, L) in sorted(set((r['cut'], r['L']) for r in rows)):
            rr = {r['model']: r['nmse'] for r in rows if r['cut'] == m and r['L'] == L}
            cells = []
            for k in KS:
                sp = {k_: v for k_, v in rr.items() if k_.startswith('SPARSE') and k_.endswith(f'_k{k}')}
                b = min(sp.items(), key=lambda t: t[1]) if sp else None
                cells.append(f"{e(rr.get(f'DENSE{k}'))} / {e(rr.get(f'DENSEQ{k}'))} / {e(b[1]) if b else '-'} [{b[0] if b else '-'}]")
            k2 = kprop_nmse(n, sig, 'k2', L, seed=seed); k3 = kprop_nmse(n, sig, 'k3f', L, seed=seed)
            P(f"| {m} -> {L} | {e(rr.get('TRUE'))} | {e(rr.get('G'))} | " + " | ".join(cells) + f" | {e(k2)} | {e(k3)} |")
            if sig == 0.01 and seed == 65 and L == 32 and m in (8, 16, 24):
                sp_ok = [v for k_, v in rr.items() if k_.startswith('SPARSE') and int(k_.split('_k')[1]) <= n // 4]
                KT4.setdefault(n, []).append((m, min(sp_ok) if sp_ok else None, k3))
for n, cells in sorted(KT4.items()):
    k3 = cells[0][2]
    best = min(c[1] for c in cells if c[1] is not None)
    P(f"\n**KT4** n={n}: best sparse (k <= n/4) oracle representation error at 32 over cuts 8/16/24 = {e(best)} vs K3f "
      f"{e(k3)} -> {'FIRES' if (k3 is not None and best >= k3) else 'does not fire'}" + (" (registered for n = 256)" if n == 256 else " (descriptive)"))

# ======================================================================================================== E4
P("\n\n# E4. Transcoder chains (arc, sigma = 0.01, seed 65)\n")
for n in WIDTHS:
    name = nn('arc', n, 0.01)
    d = load(name, 'e4.json')
    if d is None:
        continue
    P(f"\n## {name}\n")
    P("Memorisation reference (std-law truth used as the estimate): " + '; '.join(
        f"{law}: " + ', '.join(f"L{L} {e(v)}" for L, v in d['std_mean_as_estimate'][law].items()) for law in d['laws']))
    P("\n| setting | path KL bits (l=1..31) | median noise units / layer | law | chain sampled L8/16/32 | closure L8/16/32 | "
      "closure vs own chain | neuron K=2 (ours) | ARC K=2 | ARC K3f |")
    P("|" + "---|" * 10)
    g = lambda D, L: D.get(str(L), D.get(L)) if D else None   # noqa: E731
    for law in d['laws']:
        idc = d['identity_closure'][law]
        P(f"| identity | - | - | {law} | - | - | - | {', '.join(e(g(idc, L)) for L in (8, 16, 32))} | "
          f"{', '.join(e(kprop_nmse(n, 0.01, 'k2', L, law)) for L in (8, 16, 32))} | {', '.join(e(kprop_nmse(n, 0.01, 'k3f', L, law)) for L in (8, 16, 32))} |")
    for st in d['settings']:
        for law in d['laws']:
            P(f"| F={st['F']} k={st['k']} | {f3(st['path_kl_bits_l1_31'])} | {f3(float(np.median(st['noise_units_per_layer'])))} | {law} | "
              f"{', '.join(e(g(st['chain_sampled'][law], L)) for L in (8, 16, 32))} | {', '.join(e(g(st['chain_closure'][law], L)) for L in (8, 16, 32))} | "
              f"{', '.join(e(g(st['closure_vs_chain'][law], L)) for L in (8, 16, 32))} | - | - | - |")
    p = os.path.join(R, 'kprop', f'kprop_n{n}_seed65_s0.01_k2.json')
    if os.path.exists(p) and d.get('identity_means'):
        km = json.load(open(p))['means']
        diffs = [float(np.max(np.abs(np.array(g(d['identity_means'], L)) - np.array(km[L - 1])))) for L in (8, 16, 32)]
        rel = [float(np.max(np.abs(np.array(g(d['identity_means'], L)) - np.array(km[L - 1]))) / np.abs(np.array(km[L - 1])).mean()) for L in (8, 16, 32)]
        P(f"\nCode check, identity-transcoder closure vs ARC's K=2 means, max abs diff (relative to mean |E a|) L8/16/32: "
          + ', '.join(f"{x:.1e} ({y:.1e})" for x, y in zip(diffs, rel)))
print('\n'.join(out))


# ======================================================================================================== E5
out2 = []
Q = out2.append
Q("\n\n# E5. Feature families with exact marginals + full covariance vs the exact latent (arc, sigma = 0.01)\n")
Q("All models except *_raw are recoloured to the exact mean and covariance. COP = Gaussian copula over the features' exact"
  " zero-inflated marginals; IND = independent features; SPARSE = exact latent (oracle); NEUCOP = neuron copula."
  " Best over F, origin and k <= n/4. One step: nMSE of the next E[ReLU] (debiased); rep: representation at the cut pushed"
  " to depth 32 (16 from cut 8).\n")
KT6 = []


def best_of(d, prefixes, kmax):
    c = {k_: v for k_, v in d.items() if any(k_.startswith(p_) for p_ in prefixes) and int(k_.split('_k')[1]) <= kmax}
    if not c:
        return None, None
    k_ = min(c, key=lambda z: c[z])
    return k_, c[k_]


for seed in (65, 66):
    for n in WIDTHS:
        name = nn('arc', n, 0.01, seed)
        d = None
        for dd_ in NETDIRS.get(name, []):                     # merge cuts across E5 runs (cut 24 at n = 256 was rerun)
            pth = os.path.join(dd_, 'e5.json')
            if os.path.exists(pth):
                j = json.load(open(pth))
                if d is None:
                    d = j
                else:
                    d['cuts'].update(j['cuts'])
        if d is None:
            continue
        Q(f"\n## {name}\n")
        Q("| cut | metric | G | NEUCOP (raw) | DENSE k=n/16, n/8, n/4 | exact SPARSE | SPARSECOP | SPARSEIND | SPARSEIND_raw (indep. product) |")
        Q("|" + "---|" * 9)
        for m, cd in sorted(d['cuts'].items(), key=lambda t: int(t[0])):
            m = int(m)
            for metric, src in (('one step', {k_: v['nmse'] for k_, v in cd['one_step'].items()}),
                                ('rep @32', {k_: v.get('32', v.get(32)) for k_, v in cd['rep'].items()})):
                ex = best_of(src, ['SPARSE_F', 'SPARSEZ_F'], n // 4)
                cp = best_of(src, ['SPARSECOP_F', 'SPARSEZCOP_F'], n // 4)
                ind = best_of(src, ['SPARSEIND_F', 'SPARSEZIND_F'], n // 4)
                ind0 = best_of(src, ['SPARSEIND_raw_F', 'SPARSEZIND_raw_F'], n // 4)
                dens = ', '.join(e(src.get(f'DENSE{k}')) for k in (n // 16, n // 8, n // 4))
                Q(f"| {m} | {metric} | {e(src.get('G'))} | {e(src.get('NEUCOP'))} ({e(src.get('NEUCOP_raw'))}) | {dens} | "
                  f"{e(ex[1])} [{ex[0]}] | {e(cp[1])} [{cp[0]}] | {e(ind[1])} [{ind[0]}] | {e(ind0[1])} |")
                if metric == 'rep @32' and seed == 65 and ex[1] is not None and cp[1] is not None:
                    G_ = src.get('G')
                    fires_cell = (cp[1] > 3 * ex[1]) and not (G_ is not None and cp[1] < G_ / 2)   # registered rule, no floor
                    KT6.append((n, m, ex[1], cp[1], G_, fires_cell))
if KT6:
    nf = sum(1 for c in KT6 if c[5])
    Q(f"\n**KT6** (couplings beyond the covariance matter: COP > 3x exact and COP not below G/2, in >= 5 of 9 cells): {nf} of "
      f"{len(KT6)} cells present fire -> {'FIRES' if nf >= 5 else ('does not fire' if len(KT6) - nf > 4 else 'INCOMPLETE')}")
    Q("Cells (n, cut: exact, COP, G): " + '; '.join(f"({c[0]},{c[1]}: {e(c[2])}, {e(c[3])}, {e(c[4])}{' X' if c[5] else ''})" for c in KT6))
print('\n'.join(out2))

# ======================================================================================================== KT5
out3 = []
fires = {}
for n in (64, 256):
    d = load(nn('arc', n, 0.01), 'e4.json')
    if d is None:
        continue
    g = lambda D, L: D.get(str(L), D.get(L)) if D else None   # noqa: E731
    beats = []
    for st in d['settings']:
        b = all(g(st['chain_closure'][law], 8) < g(d['identity_closure'][law], 8) for law in ('shift', 'aniso'))
        beats.append((st['F'], st['k'], b, [g(st['chain_closure'][law], 8) for law in ('shift', 'aniso')]))
    fires[n] = not any(b[2] for b in beats)
    out3.append(f"KT5 n={n}: settings beating neuron K=2 at depth 8 under both shift and aniso: "
                f"{[(b[0], b[1]) for b in beats if b[2]]}; closure L8 (shift, aniso) per setting: " +
                '; '.join(f"F={b[0]} k={b[1]}: {e(b[3][0])}, {e(b[3][1])}" for b in beats) +
                f"; neuron K=2: {e(g(d['identity_closure']['shift'], 8))}, {e(g(d['identity_closure']['aniso'], 8))}")
if fires:
    out3.append(f"\n**KT5** -> {'FIRES' if len(fires) == 2 and all(fires.values()) else ('does not fire' if len(fires) == 2 else 'incomplete')} ({fires})")
print('\n'.join(out3))
