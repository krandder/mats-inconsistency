"""Run B analysis (PREREG Q1b, Q1c, covering): on the held-out paths B, scores of the fixed frames (marginal-gate,
optimal fixed; raw and D_c-projected), of a random projected frame, and of the nearest reference among the first R
paths of A by angle between states (D_c-projected and raw); at the gate layer c = L/2 also the covering statistics
(greedy centres chosen on half of B and scored on the other half; ball masses; random-centre count R99).
nice -n 19 ionice -c3 python r2b_analyze.py --n 64 --seed 400 --L 32 [--cover-layers 16]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf, cover

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--L', type=int, default=32); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--cover-layers', default='')
ap.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
A = ap.parse_args()
torch.set_num_threads(A.threads)
n, L, seed = A.n, A.L, A.seed
tag = f"n{n}_s{seed}_L{L}"
MAIN = kf.MAIN_LAYERS[L]
CL = [int(c) for c in A.cover_layers.split(',')] if A.cover_layers else [L // 2]
DELTAS = (0.3, 0.2, 0.1, 0.05)
t0 = time.time()
out = {}
for c in MAIN:
    d = np.load(os.path.join(A.runs, f'B_{tag}_c{c}.npz'))
    k = int(d['k'])
    zA, zB = torch.from_numpy(d['zA']).double().to(kf.DEV), torch.from_numpy(d['zB']).double().to(kf.DEV)
    VB, sB = torch.from_numpy(d['VB']).double().to(kf.DEV), torch.from_numpy(d['sB']).double().to(kf.DEV)
    EB = torch.from_numpy(d['EB']).to(kf.DEV)
    DB = (zB > 0).double()
    NB, NA = zB.shape[0], zA.shape[0]
    res = {'k': k, 'NA': NA, 'NB': NB, 'chance': float((k / DB.sum(-1)).mean()),
           'trunc_max': float(max(d['truncA'].max(), d['truncB'].max()))}

    def sc(G, name):
        O, CS, T = kf.scores(G, VB, sB, k, EB)
        res[name] = {'O': float(O.mean()), 'CS': float(CS.mean()), 'T': float(T.mean()),
                     'T_energy': float((T * EB).sum() / EB.sum()), 'CS_q10': float(CS.quantile(0.1)),
                     'CS_q50': float(CS.median()), 'frac_CS_ge_0.9': float((CS >= 0.9).double().mean())}
        return O, CS, T

    sc(VB[:, :, :k], 'true')
    for nm in ('Gb', 'GM'):
        G = torch.from_numpy(d[nm]).double().to(kf.DEV)
        sc(G.expand(NB, n, k), f'{nm}_raw')
        sc(kf.project_active(G.expand(NB, n, k), DB), f'{nm}_proj')
    sc(kf.random_frame(DB, k, kf.gen(60_000 + seed)), 'rand_proj')
    # nearest reference by state angle among the first R of A
    zAn = zA / zA.norm(dim=-1, keepdim=True)
    zBn = zB / zB.norm(dim=-1, keepdim=True)
    cos = zBn @ zAn.T                                   # (NB, NA)
    VA = torch.from_numpy(d['VA'][:, :, :k]).double().to(kf.DEV)
    Rs = [R for R in (1, 4, 16, 64, 256, 1024, 4096, 16384) if R <= NA]
    if NA not in Rs:
        Rs.append(NA)
    res['nearest'] = {}
    for R in Rs:
        cm, idx = cos[:, :R].max(1)
        F = VA[idx]
        Op, CSp, Tp = kf.scores(kf.project_active(F, DB), VB, sB, k, EB)
        Or, CSr, Tr = kf.scores(F, VB, sB, k, EB)
        res['nearest'][R] = {'O_proj': float(Op.mean()), 'CS_proj': float(CSp.mean()), 'T_proj': float(Tp.mean()),
                             'O_raw': float(Or.mean()), 'CS_raw': float(CSr.mean()),
                             'angle_median': float(torch.arccos(cm.clamp(-1, 1)).median()),
                             'frac_CS_ge_0.9': float((CSp >= 0.9).double().mean())}
    print(f"[{tag}] c={c} k={k} chance={res['chance']:.3f} | " +
          " ".join(f"{nm}: O={res[nm]['O']:.3f} CS={res[nm]['CS']:.3f}" for nm in
                   ('Gb_raw', 'Gb_proj', 'GM_raw', 'GM_proj', 'rand_proj')) +
          " | nearest R: " + " ".join(f"{R}:CS={v['CS_proj']:.3f},O={v['O_proj']:.3f},ang={v['angle_median']:.3f}"
                                       for R, v in res['nearest'].items()), flush=True)
    if c in CL:
        CSm, Om = cover.pair_scores(VB.float(), sB.float(), DB.float(), VA.float(), k)
        loss = 1 - CSm
        half = NB // 2
        cv = {}
        for dl in DELTAS:
            covm = loss <= dl
            p = covm.double().mean(1).numpy()                      # ball masses p(x), x in B
            ch, curve = cover.greedy_cover(covm[:half])            # centres chosen on B1
            sel = torch.tensor(ch, dtype=torch.long)
            covB2 = covm[half:][:, sel]
            # held-out coverage on B2 after the first j centres
            reached = {}
            run = torch.zeros(NB - half, dtype=torch.bool)
            cov_after = []
            for j in range(len(ch)):
                run |= covB2[:, j]
                cov_after.append(float(run.double().mean()))
            for fr in (0.5, 0.9, 0.99):
                js = [j + 1 for j, v in enumerate(cov_after) if v >= fr]
                reached[fr] = js[0] if js else None
            cv[dl] = {'p_mean': float(p.mean()), 'p_median': float(np.median(p)), 'frac_p0': float((p == 0).mean()),
                      'R99_random': cover.r99(p), 'greedy_inB1_final': curve[-1] if curve else 0.0,
                      'greedy_centres_inB1': len(ch), 'heldout_B2_final': cov_after[-1] if cov_after else 0.0,
                      'heldout_centres_for': {str(fr): v for fr, v in reached.items()},
                      'union_all_A_B': float(covm.any(1).double().mean())}
            print(f"[{tag}] c={c} cover delta={dl}: p mean {p.mean():.2e} median {np.median(p):.2e} p=0 {np.mean(p == 0):.3f}"
                  f" R99 {cv[dl]['R99_random']:.3g} | greedy {len(ch)} centres -> B1 {cv[dl]['greedy_inB1_final']:.3f},"
                  f" B2 {cv[dl]['heldout_B2_final']:.3f}; centres for 50/90/99% of B2: {reached}; all of A covers "
                  f"{cv[dl]['union_all_A_B']:.3f}", flush=True)
        res['cover'] = cv
        res['pair_CS_hist'] = np.histogram(CSm.numpy().ravel(), bins=40, range=(0, 1))[0].tolist()
        np.save(os.path.join(A.runs, f'Bcov_{tag}_c{c}.npy'), CSm.numpy().astype(np.float16))
    out[c] = res
json.dump(out, open(os.path.join(A.runs, f'Bres_{tag}.json'), 'w'), indent=1)
print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)
