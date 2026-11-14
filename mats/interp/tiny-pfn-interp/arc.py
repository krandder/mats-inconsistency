"""B2 Part 5 (PREREG Amendment 4): mechanistic vs sampling estimates of the expected excess loss.
usage: python arc.py OUT_JSON   (pool cached in /tmp/tinypfn_arc_pool.npz)
"""
import json, os, sys, time
import numpy as np
import torch
from locate import TP, W, P

torch.set_num_threads(1)
NPOOL, SEED = 100_000, 30_000
SEEDS = range(5)
POOL = '/tmp/tinypfn_arc_pool.npz'


def pool():
    if os.path.exists(POOL):
        return dict(np.load(POOL))
    d, X, m = P.fixed_set(SEED, NPOOL, W.DS)
    t0 = time.time()
    out, nll, mom, w = W.references(d)
    tb = time.time() - t0
    means, vars_, _ = out['bayes']
    mb, vb = mom['bayes']
    bcv = (w * (means - mb[:, None]) ** 2).sum(1)
    wl = out['lab'][2]
    coup = (w * (np.log(np.maximum(w, 1e-300)) - np.log(np.maximum(wl, 1e-300)))).sum(1)
    spread = d['o'][:, :, W.QL].max(1) - d['o'][:, :, W.QL].min(1)
    P_ = dict(X=X, valid=m, ystar=d['y'][:, W.QL], comp_means=means, comp_vars=vars_, w_bayes=w, mean_bayes=mb, var_bayes=vb,
              nll_bayes=nll['bayes'], bcv=bcv, coup=coup, spread=spread, t_bayes=tb)
    np.savez(POOL, **P_)
    return P_


def main():
    outp = sys.argv[1]
    D = pool()
    T = {k: D[k] for k in ('comp_means', 'comp_vars', 'w_bayes')}
    vb = D['var_bayes']
    phi_m = np.stack([np.ones_like(vb), D['spread'] ** 2 / vb, D['bcv'] / vb, D['coup']], 1)
    phi_g = np.stack([np.ones_like(vb), vb, np.log(vb)], 1)
    KL, NLLG = {}, {}
    res = dict(t_bayes_per_ep=float(D['t_bayes']) / NPOOL)
    for s in SEEDS:
        rd = os.path.join(TP, f'runs/w32_h2_L2_mix4_ws_s{s}')
        meta = json.load(open(os.path.join(rd, 'meta.json')))
        model = P.build(meta); model.load_state_dict(torch.load(os.path.join(rd, 'model.pt')))
        t0 = time.time(); head = P.predict(model, D['X'], D['valid']); res[f't_net_per_ep_s{s}'] = (time.time() - t0) / NPOOL
        KL[s] = P.kl_grid(T, head, chunk=200)
        NLLG[s] = P.head_nll_np(head, D['ystar']) - D['nll_bayes']
        print('seed', s, 'G', KL[s].mean(), flush=True)
    rng = np.random.default_rng(0)
    Ns = [2 ** k for k in range(2, 14)]
    REP = 4000
    for s in SEEDS:
        others = [t for t in SEEDS if t != s]
        th_m = np.linalg.lstsq(np.concatenate([phi_m] * 4), np.concatenate([KL[t] for t in others]), rcond=None)[0]
        th_g = np.linalg.lstsq(np.concatenate([phi_g] * 4), np.concatenate([KL[t] for t in others]), rcond=None)[0]
        cm, cg = phi_m @ th_m, phi_g @ th_g
        k = KL[s]; G = float(k.mean())
        M0, G0 = float(cm.mean()), float(cg.mean())
        r = dict(G=G, G_nll_mean=float(NLLG[s].mean()), sd_kl=float(k.std()), sd_nllgap=float(NLLG[s].std()),
                 M0=M0, M0_bias=M0 - G, G0=G0, G0_bias=G0 - G,
                 var_ratio_M1=float(k.var() / (k - cm).var()), var_ratio_G1=float(k.var() / (k - cg).var()),
                 corr_mech=float(np.corrcoef(k, cm)[0, 1]), corr_gen=float(np.corrcoef(k, cg)[0, 1]),
                 theta_mech=th_m.tolist(), theta_gen=th_g.tolist(), mse={})
        for N in Ns:
            idx = rng.integers(0, len(k), (REP, N))
            S = k[idx].mean(1); Snll = NLLG[s][idx].mean(1)
            M1 = M0 + (k - cm)[idx].mean(1); G1 = G0 + (k - cg)[idx].mean(1)
            r['mse'][N] = dict(F=G ** 2, S=float(((S - G) ** 2).mean()), S_nll=float(((Snll - G) ** 2).mean()),
                               M0=(M0 - G) ** 2, G0=(G0 - G) ** 2, M1=float(((M1 - G) ** 2).mean()), G1=float(((G1 - G) ** 2).mean()))
        Nstar = next((N for N in Ns if r['mse'][N]['S'] < r['mse'][N]['F']), None)
        r['N_star_flat'] = Nstar
        if Nstar:
            N2 = min(2 * Nstar, Ns[-1])
            r['M0_beats_S_at_2Nstar'] = bool(r['mse'][N2]['M0'] < r['mse'][N2]['S'])
            r['M1_beats_S_all_N_ge_Nstar'] = bool(all(r['mse'][N]['M1'] < r['mse'][N]['S'] for N in Ns if N >= Nstar))
            r['G1_beats_S_all_N_ge_Nstar'] = bool(all(r['mse'][N]['G1'] < r['mse'][N]['S'] for N in Ns if N >= Nstar))
        r['M0_beats_S_up_to_N'] = max([N for N in Ns if r['mse'][N]['M0'] < r['mse'][N]['S']], default=0)
        res[f's{s}'] = r
    json.dump(res, open(outp, 'w'))
    np.savez_compressed(outp.replace('.json', '_kl.npz'), **{f's{s}': KL[s].astype(np.float32) for s in SEEDS})


if __name__ == '__main__':
    main()
