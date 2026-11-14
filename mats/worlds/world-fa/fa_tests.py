"""World FA aggregation tests on the binding-given main setup (PREREG-FA.md Amendment A7, part 2).

usage: python fa_tests.py --run RUN [--ckpt best.pt] [--E 1000] --out RUN/tests.json

1. scaling: model vs Bayes vs T in {0, 8, 32, 64, 128} at N = 8 and vs N in {4, 8, 16, 32, 64} at T = 32
   (gap to Bayes, KL(Bayes || model), fraction of the history value recovered).
2. in-context structure inference: (a) history gate (identical query reports; histories from a shared-source or an
   anti-correlated-pair structure vs from independent traders): tracking = mean KL(model_A || model_B) / mean
   KL(Bayes_A || Bayes_B), slope and correlation of the model's mean shift on Bayes's; (b) fresh structures (a shared
   source among 4; two anti-correlated pairs), against Bayes under the training family ('bayes_train'), Bayes whose
   prior also contains the fresh structure ('bayes_full') and plug-in GLS.
3. discounting shared information: paired episodes, 3 traders on one shared source (rho 0.8) vs 3 independent traders,
   same marginals, common random numbers: model vs Bayes log predictive-variance ratio and mean response.
"""
import argparse, json, math
import numpy as np
import torch
import harness as H
import fa_core as C

FRESH = {'shared4': {'blocks': [[4, 0.8, 0]], 'marg': 1.0},
         'two_anti_pairs': {'blocks': [[2, 0.8, 1], [2, 0.8, 1]], 'marg': 1.0}}
B_IND = np.eye(3)
B_SHARED = 0.2 * np.eye(3) + 0.8
B_ANTI = np.array([[1, -0.8, 0], [-0.8, 1, 0], [0, 0, 1.0]])


def npp(p):
    return tuple(x.detach().cpu().numpy().astype(np.float64) for x in p)


def predict(model, b, c):
    X, M = C.tokens(b, c)
    return H.predict(model, X.numpy(), M.numpy())


def gls_refs(b, c):
    """oracle (true Sigma) and plug-in GLS, without the heavy diag-Bayes reference."""
    r = C.references(b, c, with_bayes=False, with_diag=False)
    return {k: npp(r[k]) for k in ('oracle', 'gls_plugin', 'prior')}


def scaling_cell(model, c, N, T, E, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = C.gen(E, N, T, c, 'cpu')
    y = b['v'].numpy()
    pm = predict(model, b, c)
    pb, p0 = npp(C.bayes(b, c, True)), npp(C.bayes(b, c, False))
    g = gls_refs(b, c)
    nm, nb, n0 = H.nll(pm, y), H.nll(pb, y), H.nll(p0, y)
    hv = float((n0 - nb).mean())
    return dict(N=N, T=T, E=E, nll_model=float(nm.mean()), nll_bayes=float(nb.mean()), nll_bayes_T0=float(n0.mean()),
                gap=float((nm - nb).mean()), se_gap=float((nm - nb).std() / math.sqrt(E)), history_value=hv,
                frac=float((n0 - nm).mean() / hv) if hv > 1e-3 else None,
                kl_bayes_model=float(H.kl_bayes_model(pb, pm).mean()),
                gap_oracle=float((H.nll(g['oracle'], y) - nb).mean()), gap_gls_plugin=float((H.nll(g['gls_plugin'], y) - nb).mean()),
                gap_prior=float((H.nll(g['prior'], y) - nb).mean()))


def fresh_cell(model, c, name, T, E, seed, N=8):
    full = dict(c, f_templates=tuple(dict(t, train=True) for t in c['f_templates']) + (dict(FRESH[name], train=False),))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = C.gen(E, N, T, dict(full, _part='fresh'), 'cpu')
    y = b['v'].numpy()
    pm = predict(model, b, c)
    pbt, p0t, pbf = npp(C.bayes(b, c, True)), npp(C.bayes(b, c, False)), npp(C.bayes(b, full, True))
    g = gls_refs(b, c)
    nm, nbt, n0t, nbf = H.nll(pm, y), H.nll(pbt, y), H.nll(p0t, y), H.nll(pbf, y)
    hv = float((n0t - nbt).mean())
    return dict(structure=name, N=N, T=T, E=E, nll_model=float(nm.mean()), nll_bayes_train=float(nbt.mean()),
                nll_bayes_T0_train=float(n0t.mean()), nll_bayes_full=float(nbf.mean()),
                gap_to_bayes_train=float((nm - nbt).mean()), gap_to_bayes_full=float((nm - nbf).mean()),
                se=float((nm - nbf).std() / math.sqrt(E)),
                frac_of_train_history_value=float((n0t - nm).mean() / hv) if hv > 1e-3 else None,
                nll_gls_plugin=float(H.nll(g['gls_plugin'], y).mean()), nll_oracle=float(H.nll(g['oracle'], y).mean()))


def paired(model, c, blocks, T, G, seed, gate, N=8):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = C.pair_sets_f(G, N, T, blocks, c, 'cpu', gate=gate)
    A = len(blocks)
    pm = predict(model, b, c)
    pb = npp(C.bayes(b, c, True))
    mm, vm = H.moments(pm); mb, vb = H.moments(pb)
    sel = lambda p, a: tuple(x[a::A] for x in p)
    out = dict(T=T, G=G, N=N)
    # arm 1 relative to arm 0
    dmm, dmb = mm[1::A] - mm[0::A], mb[1::A] - mb[0::A]
    out.update(log_var_ratio_model=float(np.log(vm[1::A] / vm[0::A]).mean()),
               log_var_ratio_bayes=float(np.log(vb[1::A] / vb[0::A]).mean()),
               mean_shift_slope=float((dmm * dmb).sum() / (dmb ** 2).sum()) if (dmb ** 2).sum() > 0 else None,
               mean_shift_corr=float(np.corrcoef(dmm, dmb)[0, 1]) if dmb.std() > 0 and dmm.std() > 0 else None,
               rms_shift_ratio=float(np.sqrt((dmm ** 2).mean() / max((dmb ** 2).mean(), 1e-300))))
    klm = H.kl_bayes_model(sel(pm, 0), sel(pm, 1)); klb = H.kl_bayes_model(sel(pb, 0), sel(pb, 1))
    out.update(kl_model=float(klm.mean()), kl_bayes=float(klb.mean()), kl_tracking=float(klm.mean() / klb.mean()),
               frac_bayes_kl_ge_0p3=float((klb >= 0.3).mean()))
    y = b['v'].numpy()
    out.update(gap_arm0=float((H.nll(sel(pm, 0), y[0::A]) - H.nll(sel(pb, 0), y[0::A])).mean()),
               gap_arm1=float((H.nll(sel(pm, 1), y[1::A]) - H.nll(sel(pb, 1), y[1::A])).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--E', type=int, default=1000); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    model, world, meta = H.load_run(a.run, a.ckpt)
    c = world.c
    res = dict(run=a.run, ckpt=a.ckpt, scaling_T=[], scaling_N=[], fresh=[], gate={}, dedup=[])
    dump = lambda: json.dump(res, open(a.out, 'w'), indent=1)
    for T in (0, 8, 32, 64, 128):
        res['scaling_T'].append(scaling_cell(model, c, 8, T, a.E, 100 + T)); print(json.dumps(res['scaling_T'][-1]), flush=True)
        dump()
    for N in (4, 8, 16, 32, 64):
        res['scaling_N'].append(scaling_cell(model, c, N, 32, a.E if N <= 16 else max(200, a.E // 4), 200 + N))
        print(json.dumps(res['scaling_N'][-1]), flush=True); dump()
    for name in FRESH:
        for T in (8, 32, 64):
            res['fresh'].append(fresh_cell(model, c, name, T, a.E, 300 + T)); print(json.dumps(res['fresh'][-1]), flush=True)
            dump()
    for name, B in (('shared_vs_independent', B_SHARED), ('anti_vs_independent', B_ANTI)):
        res['gate'][name] = paired(model, c, [B, B_IND], 32, a.E // 2, 400, gate=True)
        print(name, json.dumps(res['gate'][name]), flush=True); dump()
    for T in (0, 8, 32):
        r = paired(model, c, [B_IND, B_SHARED], T, a.E // 2, 500 + T, gate=False)
        res['dedup'].append(r); print('dedup', json.dumps(r), flush=True); dump()


if __name__ == '__main__':
    main()
