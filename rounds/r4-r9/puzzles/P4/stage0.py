"""Stage 0: exact program-level tables and theorem checks (numpy, no training).
Writes results/stage0.json and results/stage0_rows.json."""
import json
import sys
import time
import numpy as np
from scipy.stats import spearmanr, kendalltau
import gen

X = gen.allx()
NPAIR = 4096


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(spearmanr(a, b).correlation)


def tau(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(kendalltau(a, b).statistic)


def pairs(seed, nx=len(X)):
    rng = np.random.default_rng(10_000 + seed)
    return rng.integers(0, nx, NPAIR), rng.integers(0, nx, NPAIR)


def circuit_rows(prog, seed):
    xi, xp = pairs(seed)
    nlev = len(prog['levels']) - 2
    rows = []
    for l in range(1, nlev + 1):
        t = gen.program_tables(prog, l, X, xi, xp)
        rows.append(dict(circuit=prog['name'], level=l, roles=t['roles'], names=t['names'], types=t['types'],
                         r=t['r'].tolist(), marg=t['flip']['marg'].tolist(), shap=t['flip']['shap'].tolist(),
                         m=t['m'].tolist(), u=t['u'].tolist(),
                         pay_shap=t['pay']['shap'].tolist(), pay_marg=t['pay']['marg'].tolist(),
                         mi_shap=t['mi']['shap'].tolist(), mi_loo=t['mi']['loo'].tolist(),
                         mi_marg=t['mi']['marg'].tolist(),
                         eff_gap={g: t[g]['eff_gap'] for g in ('flip', 'pay', 'mi')},
                         plants={k: (v if not isinstance(v, np.integer) else int(v))
                                 for k, v in prog.get('plants', {}).get(l, {}).items()}))
    return rows


KEYS = ['r', 'marg', 'shap', 'm', 'u', 'mi_shap', 'mi_loo', 'pay_shap']


def agreement(rows, gates_only=False):
    """Mean within-level and pooled (per circuit) Spearman / Kendall between the ground truths."""
    out = {}
    circuits = sorted(set(r['circuit'] for r in rows))
    for i, a in enumerate(KEYS):
        for b in KEYS[i + 1:]:
            wl_r, wl_t, pl_r = [], [], []
            for r in rows:
                sel = [k for k, ro in enumerate(r['roles']) if (ro == 'gate' or not gates_only)]
                A, B = np.array(r[a])[sel], np.array(r[b])[sel]
                wl_r.append(rho(A, B))
                wl_t.append(tau(A, B))
            for c in circuits:
                A, B = [], []
                for r in rows:
                    if r['circuit'] != c:
                        continue
                    sel = [k for k, ro in enumerate(r['roles']) if (ro == 'gate' or not gates_only)]
                    A += list(np.array(r[a])[sel])
                    B += list(np.array(r[b])[sel])
                pl_r.append(rho(A, B))
            out[f'{a}~{b}'] = dict(within_rho=float(np.nanmean(wl_r)), within_tau=float(np.nanmean(wl_t)),
                                   within_nan=int(np.isnan(wl_r).sum()), pooled_rho=float(np.nanmean(pl_r)))
    return out


def top1(rows, a, b, gates_only=True, tol=0.01):
    hits = []
    for r in rows:
        sel = [k for k, ro in enumerate(r['roles']) if (ro == 'gate' or not gates_only)]
        A, B = np.array(r[a])[sel], np.array(r[b])[sel]
        hits.append(int(np.argmax(A)) in set(np.flatnonzero(B >= B.max() - tol)))
    return float(np.mean(hits))


def tied_share(rows, gates_only=True):
    t = []
    for r in rows:
        sel = [k for k, ro in enumerate(r['roles']) if (ro == 'gate' or not gates_only)]
        R = np.round(np.array(r['r'])[sel], 3)
        t.append(len(np.unique(R)) < len(R))
    return float(np.mean(t))


def theorem_checks(rows, h1rows, h2mi):
    chk = {}
    copy_loo = [r['r'][k] for r in rows + h1rows for k, ro in enumerate(r['roles']) if ro == 'copy']
    chk['copy_LOO_max'] = float(np.max(np.abs(copy_loo)))
    epi = []
    for r in rows + h1rows:
        p = r['plants']
        if p.get('epi') is None:
            continue
        e, s = p['epi'], p['epi_src']
        epi.append((abs(r['m'][e] - r['m'][s]), abs(r['r'][e]), abs(r['shap'][e])))
    epi = np.array(epi)
    chk['epi_MI_minus_source_MI_max'] = float(epi[:, 0].max())
    chk['epi_flip_LOO_max'] = float(epi[:, 1].max())
    chk['epi_flip_shap_max'] = float(epi[:, 2].max())
    h1l3 = [r for r in h1rows if r['level'] == 3][0]
    chk['H1_I(p;Y)'] = float(max(h1l3['m'][k] for k in h1l3['plants']['copies']))
    chk['H2_balanced_candidate_MI_min'] = float(min(h2mi))
    chk['H2_balanced_candidate_MI_max'] = float(max(h2mi))
    chk['shapley_efficiency_gap_max'] = float(max(abs(v) for r in rows + h1rows for v in r['eff_gap'].values()))
    chk['pass'] = bool(chk['copy_LOO_max'] < 1e-12 and chk['epi_MI_minus_source_MI_max'] < 1e-9
                       and chk['epi_flip_LOO_max'] < 1e-12 and chk['epi_flip_shap_max'] < 1e-9
                       and chk['H1_I(p;Y)'] < 1e-9 and abs(chk['H2_balanced_candidate_MI_min'] - 1) < 1e-9
                       and abs(chk['H2_balanced_candidate_MI_max'] - 1) < 1e-9
                       and chk['shapley_efficiency_gap_max'] < 1e-9)
    return chk


def h2_checks():
    prog = gen.h2()
    X16 = gen.allx(16)
    V = gen.values(prog, X16)
    Y = V[-1]
    mis = []
    for l in (1, 2):
        for k in range(V[l].shape[0]):
            if abs(V[l][k].mean() - 0.5) < 1e-9:
                mis.append(gen.mi_bits(V[l][k:k + 1], Y))
    # reach of H2 candidates (graded; MI cannot grade them)
    rng = np.random.default_rng(5)
    xi, xp = rng.integers(0, len(X16), 4096), rng.integers(0, len(X16), 4096)
    reach = {l: gen.flip_loo_program(prog, l, X16, xi, xp).tolist() for l in (1, 2)}
    return mis, reach


def main():
    t0 = time.time()
    rows = []
    for s in range(30):
        prog = gen.random_circuit(s)
        rows += circuit_rows(prog, s)
        print('circuit', s, round(time.time() - t0, 1), flush=True)
    h1rows = circuit_rows(gen.h1(), 101)
    h3rows = circuit_rows(gen.h3(), 103)
    h2mi, h2reach = h2_checks()
    # separated-cone circuits: generate until 10 have mean within-level Spearman(u, r) < 0.3 (gates only)
    sep, sep_all, s = [], [], 1000
    while len(sep) < 10 and s < 1400:
        prog = gen.random_circuit(s, split_cones=True)
        rr = circuit_rows(prog, s)
        su = np.nanmean([rho(np.array(r['u'])[[k for k, ro in enumerate(r['roles']) if ro == 'gate']],
                             np.array(r['r'])[[k for k, ro in enumerate(r['roles']) if ro == 'gate']]) for r in rr])
        sep_all.append(dict(seed=s, rho_u_r=float(su)))
        if su < 0.3:
            sep.append(dict(seed=s, rho_u_r=float(su)))
            rows_sep = rr
        s += 1
    out = dict(
        n_circuits=30, npairs=NPAIR,
        agreement_all_candidates=agreement(rows), agreement_gates_only=agreement(rows, gates_only=True),
        top1_shap_vs_reach_gates=top1(rows, 'shap', 'r'), top1_shap_vs_reach_all=top1(rows, 'shap', 'r', False),
        tied_reach_share_gates=tied_share(rows), tied_reach_share_all=tied_share(rows, False),
        theorems=theorem_checks(rows, h1rows, h2mi),
        H1=h1rows, H3=h3rows, H2=dict(balanced_MI=h2mi, reach=h2reach),
        H3_agree=dict((k, v) for k, v in agreement(h3rows).items() if k in ('r~m', 'r~shap', 'shap~m', 'r~u')),
        H3_r_top={l: dict(reach_top=int(np.argmax(r['r'])) == 0, mi_top=int(np.argmax(r['m'])) == 0)
                  for l, r in zip((1, 2, 3), h3rows)},
        H1_level3=dict(p_copies_reach=[h1rows[2]['r'][k] for k in (2, 3, 4)],
                       p_copies_shap=[h1rows[2]['shap'][k] for k in (2, 3, 4)],
                       p_copies_mi=[h1rows[2]['m'][k] for k in (2, 3, 4)]),
        separated_cones=dict(chosen=sep, tried=len(sep_all), all=sep_all),
        outxor_mean=float(np.mean([sum(nd['type'].startswith('XOR') for nd in gen.random_circuit(s)['levels'][-1])
                                   for s in range(30)])),
        sec=round(time.time() - t0, 1))
    json.dump(out, open('results/stage0.json', 'w'), indent=1)
    json.dump(rows, open('results/stage0_rows.json', 'w'))
    print(json.dumps({k: v for k, v in out.items() if k not in ('H1', 'H3', 'agreement_all_candidates', 'separated_cones')}, indent=1)[:4000])
    print('agreement all', json.dumps(out['agreement_all_candidates'])[:1500])
    print('sep chosen', sep, 'tried', len(sep_all))


if __name__ == '__main__':
    main()
