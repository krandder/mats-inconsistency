"""Tables for the sparse-codes report: python scratch/analyze.py > results/tables.md
Reads results/gpu/<run>/<net>/{e1_L*.json, e2_L*.json, e3.json, truth.npz, e4.json, truth_<law>.npz} and
results/kprop*/*.json."""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results')
NETDIRS = {}
for d in sorted(glob.glob(os.path.join(R, 'gpu', '*', '*'))):
    if os.path.isdir(d):
        NETDIRS.setdefault(os.path.basename(d), []).append(d)


def load(name, fn):
    for d in NETDIRS.get(name, []):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return json.load(open(p)) if fn.endswith('.json') else np.load(p)
    return None


def netname(fam, n, sigma, seed=65):
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
out = []
P = out.append

# ======================================================================================================== E1
P("# E1. Dictionaries in bits\n")
P("Noise units = E||a - a_hat||^2 / (n s^2) per coordinate (below 1 = below the noise); bits = n/(2 ln 2) x noise units.")
P("Best SAE = lowest held-out error over F in {1,2,4,8,16} x n at that k (raw TopK reconstruction; LS refit in brackets).\n")
KT1 = {}
for fam, sig in (('arc', 0.01), ('arc', 0.001), ('tord', 0.01), ('tcha', 0.01)):
    for n in WIDTHS:
        nm = netname(fam, n, sig)
        rows = []
        for l in LAYERS[fam]:
            d = load(nm, f'e1_L{l}.json')
            if d is None:
                continue
            KS = [n // 16, n // 8, n // 4, n // 2]
            best = {}
            for k in KS:
                cand = [r for r in d['sae'] if r['k'] == k]
                if cand:
                    b = min(cand, key=lambda r: r['noise_units'])
                    bf = min(cand, key=lambda r: r['noise_units_refit'])
                    best[k] = (b['noise_units'], bf['noise_units_refit'], b['F'] // n)
            # KT1: any SAE with L0 <= n/4 below the noise (raw or refit)?
            alive = any(min(r['noise_units'], r['noise_units_refit']) < 1 for r in d['sae'] if r['k'] <= n // 4)
            KT1[(fam, sig, n, l)] = not alive
            rows.append((l, d, best, alive))
        if not rows:
            continue
        P(f"\n## {nm}\n")
        P("| layer | s | var/coord (noise units) | PR | eig > s^2 | k_PCA below noise | mean only | "
          + " | ".join(f"k={k}: SAE best (refit) [F/n] / PCA / neuron" for k in [n // 16, n // 8, n // 4, n // 2]) + " | KT1 (L0<=n/4) |")
        P("|" + "---|" * (7 + 4 + 1))
        for l, d, best, alive in rows:
            cells = []
            for k in [n // 16, n // 8, n // 4, n // 2]:
                b = best.get(k)
                pca = d['pca'].get(str(k), {}).get('noise_units')
                if fam == 'arc':
                    neu = min(d['neuron_topk'][str(k)]['noise_units'], d['neuron_topk_zero'][str(k)]['noise_units'])
                else:
                    neu = d['neuron_topk'][str(k)]['noise_units']
                cells.append(f"{f3(b[0]) if b else '-'} ({f3(b[1]) if b else '-'}) [{b[2] if b else '-'}] / {f3(pca)} / {f3(neu)}")
            P(f"| {l} | {d['s']:.2g} | {f3(d['var_per_coord'] / d['s'] ** 2)} | {d['PR']:.1f} | {d['eig_above_noise']} | "
              f"{d['k_pca_below_noise']} | {f3(d['pca']['0']['noise_units'])} | " + " | ".join(cells) +
              f" | {'fires' if not alive else 'no'} |")
        # scaling with F at k = n/8
        P("\nError vs dictionary size at k = n/8 (raw, noise units), F/n = 1, 2, 4, 8, 16; and the 2x-epoch control at F = 4n:")
        for l, d, best, alive in rows:
            k = n // 8
            byF = {r['F'] // n: r['noise_units'] for r in d['sae'] if r['k'] == k}
            s_ = ', '.join(f3(byF.get(fm)) for fm in (1, 2, 4, 8, 16))
            lc = d.get('long_control')
            lcs = ''
            if lc:
                base = {r['k']: r['noise_units'] for r in d['sae'] if r['F'] == 4 * n}
                lcs = '; 2x epochs F=4n: ' + ', '.join(f"k={r['k']}: {f3(base.get(r['k']))} -> {f3(r['noise_units'])}" for r in lc)
            dead = {r['F'] // n: r['dead_frac'] for r in d['sae'] if r['k'] == k}
            P(f"- layer {l}: {s_} (dead fraction at 16n: {f3(dead.get(16))}){lcs}")
        if fam == 'arc':
            P("\nActive neurons (mean / max / ever active): " + '; '.join(
                f"L{l}: {d['active']['mean']:.1f} / {d['active']['max']} / {d['active']['ever_active']}" for l, d, _, _ in rows))

# ======================================================================================================== E2
P("\n\n# E2. One step: higher-order structure carried by latent-conditional Gaussians\n")
P("All models carry the exact mean and full covariance. mean = nMSE of E[act(z')] (ARC units), cap3/cap4 = fraction of the"
  " per-neuron 3rd/4th cumulants of z' reproduced. TRUE_indep = independent fresh truth (sampling floor for independent"
  " samples; the models share latent and noise with the truth, so their floor is lower).\n")
KT2c, KT3c = [], []
for fam, sig in (('arc', 0.01), ('arc', 0.001), ('tord', 0.01), ('tcha', 0.01)):
    for n in WIDTHS:
        nm = netname(fam, n, sig)
        blocks = []
        for l in LAYERS[fam]:
            d = load(nm, f'e2_L{l}.json')
            if d is None:
                continue
            rows = {r['model']: r for r in d['rows']}
            blocks.append((l, d, rows))
        if not blocks:
            continue
        P(f"\n## {nm}\n")
        KS = [n // 16, n // 8, n // 4]
        P("| layer | median abs g3 / g4 of z' | TRUE_indep | G (K=2) | EW3 oracle | EW4 oracle | " +
          " | ".join(f"k={k}: DENSE / DENSEQ / best SPARSE [F/n] (mean nMSE; cap3)" for k in KS) + " |")
        P("|" + "---|" * (6 + len(KS)))
        for l, d, rows in blocks:
            cells = []
            for k in KS:
                dn = rows.get(f'DENSE{k}'); dq = rows.get(f'DENSEQ{k}')
                sp = [r for r in d['rows'] if r['model'].startswith('SPARSE') and r.get('k') == k]
                b = min(sp, key=lambda r: r['mean_nmse']) if sp else None
                cells.append(f"{e(dn['mean_nmse']) if dn else '-'}; {f3(dn['cap3']) if dn else '-'} / "
                             f"{e(dq['mean_nmse']) if dq else '-'}; {f3(dq['cap3']) if dq else '-'} / "
                             f"{e(b['mean_nmse']) if b else '-'}; {f3(b['cap3']) if b else '-'} [{b['F'] // n if b else '-'}]")
                if fam == 'arc' and sig == 0.01 and l in (8, 16, 24, 31) and b is not None:
                    if k in (n // 16, n // 8):
                        dens = [x['mean_nmse'] for x in (dn, dq) if x is not None]
                        KT2c.append((n, l, k, b['mean_nmse'], min(dens), b['mean_nmse'] >= min(dens)))
                    KT3c.append((n, l, k, b['mean_nmse'], rows['G']['mean_nmse'], b['mean_nmse'] >= rows['G']['mean_nmse']))
            P(f"| {l} | {d['z_g3_med']:.2f} / {d['z_g4_med']:.2f} | {e(rows['TRUE_indep']['mean_nmse'])} | {e(rows['G']['mean_nmse'])} | "
              f"{e(rows['EW3_oracle']['mean_nmse'])} | {e(rows['EW4_oracle']['mean_nmse'])} | " + " | ".join(cells) + " |")
        # residual Gaussianity
        P("\nResidual given the latent, per-neuron excess kurtosis (median [p90]): marginal / conditional-variance-normalised,"
          " in neuron space and projected by W_{l+1}; residual size in noise units:")
        for l, d, rows in blocks:
            parts = []
            for k in KS:
                sp = [r for r in d['rows'] if r['model'].startswith('SPARSE') and r.get('k') == k and r.get('F') == 4 * n]
                dn = rows.get(f'DENSE{k}')
                for tag, r in (('SPARSE4n', sp[0] if sp else None), ('DENSE', dn)):
                    if r is None or 'resid_kurt_med' not in r:
                        continue
                    parts.append(f"k={k} {tag}: {f3(r['resid_kurt_med'])} [{f3(r['resid_kurt_p90'])}] / "
                                 f"{f3(r.get('resid_neuron_condkurt_med'))} [{f3(r.get('resid_neuron_condkurt_p90'))}]; proj "
                                 f"{f3(r['resid_proj_kurt_med'])} / {f3(r.get('resid_proj_condkurt_med'))}"
                                 + (f"; size {f3(r.get('resid_noise_units'))}" if r.get('resid_noise_units') is not None else ''))
            P(f"- layer {l}: " + ' | '.join(parts))
if KT2c:
    nf = sum(1 for c in KT2c if c[5])
    P(f"\n**KT2** (sparse no better than dense at matched k in {{n/16, n/8}}; arc 0.01, layers 8-31): {nf} of {len(KT2c)} cells "
      f"-> {'FIRES' if nf >= len(KT2c) / 2 else 'does not fire'}")
    P("Cells (n, layer, k, best SPARSE, best dense, sparse >= dense): " + '; '.join(
        f"({c[0]},{c[1]},{c[2]}: {e(c[3])} vs {e(c[4])}{' X' if c[5] else ''})" for c in KT2c))
if KT3c:
    # KT3 uses k <= n/4: best SPARSE over k <= n/4 per (n, layer)
    cells = {}
    for c in KT3c:
        key = (c[0], c[1])
        if key not in cells or c[3] < cells[key][0]:
            cells[key] = (c[3], c[4])
    nf = sum(1 for v in cells.values() if v[0] >= v[1])
    P(f"\n**KT3** (best SPARSE k<=n/4 one-step mean nMSE >= G; arc 0.01, layers 8-31): {nf} of {len(cells)} cells -> "
      f"{'FIRES' if nf >= len(cells) / 2 else 'does not fire'}")
    P("Cells (n, layer: best SPARSE vs G): " + '; '.join(f"({k[0]},{k[1]}: {e(v[0])} vs {e(v[1])})" for k, v in sorted(cells.items())))

# ======================================================================================================== kprop scoring
def kprop_nmse(n, sigma, v, L, law='std', seed=65):
    if law == 'std':
        p = os.path.join(R, 'kprop', f'kprop_n{n}_seed{seed}_s{sigma:g}_{v}.json')
        T = load(netname('arc', n, sigma, seed), 'truth.npz')
    else:
        p = os.path.join(R, 'kprop_law', f'kprop_n{n}_seed{seed}_s{sigma:g}_{law}_{v}.json')
        T = load(netname('arc', n, sigma, seed), f'truth_{law}.npz')
    if not os.path.exists(p) or T is None:
        return None
    m = np.array(json.load(open(p))['means'][L - 1])
    N = float(T['N']); tm = T['S1'][L - 1] / N; tv = T['S2'][L - 1] / N - tm ** 2
    return float(((m - tm) ** 2).mean() / tv.mean() - 1.0 / N)


# ======================================================================================================== E3
P("\n\n# E3. Representation at a cut, pushed through the real layers (arc)\n")
P("nMSE of E[a_L] (both Monte Carlo floors subtracted). TRUE = no replacement (pipeline control). K=2 / K3f = ARC's kprop"
  " end to end from the input (scored against the same truth).\n")
KT4 = {}
for sig in (0.01, 0.001):
    for n in WIDTHS:
        nm = netname('arc', n, sig)
        d = load(nm, 'e3.json')
        if d is None:
            continue
        rows = d['rows']
        P(f"\n## {nm} (N3 = {d['N3']}, truth {d['NT']})\n")
        KS = [n // 16, n // 8, n // 4, n // 2]
        P("| cut -> L | TRUE | G | " + " | ".join(f"k={k}: DENSE / DENSEQ / best SPARSE [F/n]" for k in KS) + " | K=2 e2e | K3f e2e |")
        P("|" + "---|" * (3 + len(KS) + 2))
        keys = sorted(set((r['cut'], r['L']) for r in rows))
        for (m, L) in keys:
            rr = {r['model']: r['nmse'] for r in rows if r['cut'] == m and r['L'] == L}
            cells = []
            for k in KS:
                sp = {nm_: v for nm_, v in rr.items() if nm_.startswith('SPARSE') and nm_.endswith(f'_k{k}')}
                b = min(sp.items(), key=lambda t: t[1]) if sp else None
                bF = int(b[0].split('_F')[1].split('_')[0]) // n if b else None
                cells.append(f"{e(rr.get(f'DENSE{k}'))} / {e(rr.get(f'DENSEQ{k}'))} / {e(b[1]) if b else '-'} [{bF}]")
            k2 = kprop_nmse(n, sig, 'k2', L); k3 = kprop_nmse(n, sig, 'k3f', L)
            P(f"| {m} -> {L} | {e(rr.get('TRUE'))} | {e(rr.get('G'))} | " + " | ".join(cells) + f" | {e(k2)} | {e(k3)} |")
            if sig == 0.01 and L == 32 and m in (8, 16, 24):
                sp_ok = [v for nm_, v in rr.items() if nm_.startswith('SPARSE') and int(nm_.split('_k')[1]) <= n // 4]
                KT4.setdefault(n, []).append((m, min(sp_ok) if sp_ok else None, k3))
for n, cells in KT4.items():
    k3 = cells[0][2]
    if k3 is None:
        continue
    best = min(c[1] for c in cells if c[1] is not None)
    P(f"\n**KT4** n={n}: best SPARSE (k<=n/4) representation error at 32 over cuts 8/16/24 = {e(best)} vs K3f e2e {e(k3)} -> "
      f"{'FIRES' if best >= k3 else 'does not fire'}" + (" (pre-registered for n=256)" if n == 256 else " (descriptive)"))

# ======================================================================================================== E4
P("\n\n# E4. Transcoder chains (arc, sigma = 0.01)\n")
for n in WIDTHS:
    nm = netname('arc', n, 0.01)
    d = load(nm, 'e4.json')
    if d is None:
        continue
    P(f"\n## {nm}\n")
    P("Moving the input law: the std-law truth used as the estimate for each law (memorisation reference): " +
      '; '.join(f"{law}: " + ', '.join(f"L{L} {e(v)}" for L, v in d['std_mean_as_estimate'][law].items()) for law in d['laws']))
    P("\n| setting | path KL bits (l=1..31) | median noise units per layer | law | chain sampled L8/16/32 | closure L8/16/32 | "
      "closure vs own chain L8/16/32 | neuron K=2 (ours) L8/16/32 | ARC K=2 / K3f L8/16/32 |")
    P("|" + "---|" * 9)
    for law in d['laws']:
        idc = d['identity_closure'][law]
        arc = ' / '.join(', '.join(e(kprop_nmse(n, 0.01, v, L, law)) for L in (8, 16, 32)) for v in ('k2', 'k3f'))
        P(f"| identity | - | - | {law} | - | - | - | {', '.join(e(idc[str(L)] if str(L) in idc else idc.get(L)) for L in (8, 16, 32))} | {arc} |")
    for st in d['settings']:
        for law in d['laws']:
            cs = st['chain_sampled'][law]; cl = st['chain_closure'][law]; cv = st['closure_vs_chain'][law]
            g = lambda D, L: D.get(str(L), D.get(L))
            P(f"| F={st['F']} k={st['k']} | {f3(st['path_kl_bits_l1_31'])} | {f3(float(np.median(st['noise_units_per_layer'])))} | {law} | "
              f"{', '.join(e(g(cs, L)) for L in (8, 16, 32))} | {', '.join(e(g(cl, L)) for L in (8, 16, 32))} | "
              f"{', '.join(e(g(cv, L)) for L in (8, 16, 32))} | - | - |")
    # identity check against ARC's K=2 means (std law)
    p = os.path.join(R, 'kprop', f'kprop_n{n}_seed65_s0.01_k2.json')
    if os.path.exists(p) and d.get('identity_means'):
        km = json.load(open(p))['means']
        diffs = [float(np.max(np.abs(np.array(d['identity_means'][str(L)] if str(L) in d['identity_means'] else d['identity_means'][L])
                                       - np.array(km[L - 1])))) for L in (8, 16, 32)]
        P(f"\nCode check, our identity-transcoder closure vs ARC's K=2 means (max abs diff) L8/16/32: " + ', '.join(f"{x:.1e}" for x in diffs))
print('\n'.join(out))
