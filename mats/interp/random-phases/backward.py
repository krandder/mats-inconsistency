"""Backward-only estimation of E[a_L]: truth, input-independent stand-ins pushed through the last k layers, input-law
swaps, and the deductive version (stand-in, then ARC's kprop K = 2 through the last k layers).

usage: python backward.py REGIME N SIGMA SEED [NT NS NA]
Truth: NT plain Monte Carlo paths from x ~ N(0, I); per-layer means and per-neuron variances of a_l, centred
second moments (for the moment-matched stand-in and the collapse statistics) and, for ReLU kinds, gate statistics.
Stand-ins for the state h_d (the input of layer d + 1), d in DEPTHS, each pushed by NS sampled paths through the real
noisy layers d+1..64 (one push gives every output depth L > d, so k = L - d):
  MF  N(m 1, v I) with the mean-field per-neuron mean and variance of h_d (input-independent; knows only the scale)
  MM  N(mu_d, Sigma_d), the true mean and covariance of h_d (not input-independent; the best Gaussian, reference)
Input-law swaps (whole net, NA paths): 'aniso' N(0, C) with tr C = n and eigenvalues ~ 1/i; 'point' x0 + 0.1 N(0, I).
Deductive: ARC's mlp_kprop machinery (linear_kprop + nonlin_kprop, K = 2 simple, avg metric, noise added to the
diagonal of K2 after every layer but the last; ext/kprop_noisy_copy.py's recipe plus biases and tanh) started from MF
or MM at depth d; FLOPs from ARC's NamedFlopCounter. Plain sampling at equal FLOPs has nMSE 1/N, N = FLOPs / path cost.
Error metric (ARC's): nMSE = mean_j (est_j - truth_j)^2 / mean_j Var(a_L,j), debiased by subtracting both Monte Carlo
variances (est's and truth's).
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnets

DEPTHS = (1, 2, 4, 8, 16, 24, 28, 30, 31, 32, 40, 48, 56, 60, 62, 63)
KPROP_DEPTHS = {256: (24, 32, 48, 56, 60, 63), 64: (1, 8, 16, 24, 32, 40, 48, 56, 60, 63)}
F32, F64 = torch.float32, torch.float64
CH = 1 << 13


def pr_stats(C, mu, e2):
    ev, U = np.linalg.eigh(C)
    ev = np.clip(ev[::-1], 0, None); U = U[:, ::-1]
    tot = ev.sum()
    cs = np.cumsum(ev) / tot if tot > 0 else np.ones_like(ev)
    return dict(pr=float(tot ** 2 / (ev ** 2).sum()) if tot > 0 else float('nan'),
                k90=int(np.searchsorted(cs, 0.90) + 1), k99=int(np.searchsorted(cs, 0.99) + 1),
                mean_share=float((mu ** 2).sum() / e2) if e2 > 0 else float('nan'),
                cos_pc1_mean=float(abs(U[:, 0] @ mu) / (np.linalg.norm(mu) + 1e-300)),
                var_share=float(tot / e2) if e2 > 0 else float('nan'))


@torch.no_grad()
def truth_run(net, NT, gen):
    n, L = net.n, net.L
    S1 = torch.zeros(L, n, dtype=F64); S2 = torch.zeros(L, n, dtype=F64)
    C = torch.zeros(L, n, n, dtype=F64); ref = []
    relu = net.kind in ('relu', 'relu_ln')
    zon = torch.zeros(L, n, dtype=F64); z1 = torch.zeros(L, n, dtype=F64); z2 = torch.zeros(L, n, dtype=F64)
    sat = torch.zeros(L, dtype=F64)
    done = 0
    while done < NT:
        m = min(CH, NT - done)
        h = net.inputs(m, gen)
        for l in range(L):
            z = net.pre(l, h); a = net.act(z)
            if len(ref) <= l:
                ref.append(a.double().mean(0).to(F32))      # centring reference from the first chunk
            ad = a.double()
            S1[l] += ad.sum(0); S2[l] += (ad ** 2).sum(0)
            ac = a - ref[l]
            C[l] += (ac.T @ ac).double()
            if relu:
                zd = z.double(); zon[l] += (zd > 0).sum(0); z1[l] += zd.sum(0); z2[l] += (zd ** 2).sum(0)
            else:
                sat[l] += (a.abs() > 0.9).double().sum()
            h = net.noise(l, a, gen)
        done += m
    mean = S1 / NT; var = S2 / NT - mean ** 2
    refm = torch.stack([r.double() for r in ref])
    dm = mean - refm
    cov = C / NT - dm[:, :, None] * dm[:, None, :]
    out = dict(mean=mean, var=var.clamp_min(0), cov=cov)
    if relu:
        p_on = zon / NT; zm = z1 / NT; zv = (z2 / NT - zm ** 2).clamp_min(1e-300)
        out.update(p_on=p_on, margin=zm.abs() / zv.sqrt())
    else:
        out.update(sat=sat / (NT * n))
    return out


@torch.no_grad()
def push(net, d, sampler, NS, gen):
    """Sample h_d from sampler(m, gen) (state after layer d, noise included), run layers d+1..L.
    Returns per-depth mean and variance of a_L for L = d+1..net.L (index L-1)."""
    n, L = net.n, net.L
    S1 = torch.zeros(L, n, dtype=F64); S2 = torch.zeros(L, n, dtype=F64)
    done = 0
    while done < NS:
        m = min(CH, NS - done)
        h = sampler(m, gen)
        for l in range(d, L):
            a = net.act(net.pre(l, h))
            ad = a.double(); S1[l] += ad.sum(0); S2[l] += (ad ** 2).sum(0)
            h = net.noise(l, a, gen)
        done += m
    mean = S1 / NS
    return mean, (S2 / NS - mean ** 2).clamp_min(0)


def nmse(est, est_var, Ns, T, l):
    """Debiased nMSE of an estimate of E[a_{l+1}] (0-based l)."""
    m, v = T['mean'][l], T['var'][l]
    vm = float(v.mean())
    if vm <= 0:
        return float('nan')
    e = float(((est - m) ** 2).mean())
    e -= float(v.mean()) / T['NT']
    if est_var is not None and Ns:
        e -= float(est_var.mean()) / Ns
    return e / vm


def var_err(est_var, T, l):
    """Relative RMS error of the per-neuron output variances (amendment 3; floor about sqrt(2/NS))."""
    v = T['var'][l]
    den = float((v ** 2).mean())
    return float(((est_var - v) ** 2).mean() / den) ** 0.5 if den > 0 else float('nan')


# ------------------------------------------------------------------------------------------------ ARC's kprop
def kprop_from(net, d, mu, Sig, Lmax):
    ARC_SRC = os.path.join(pnets.HERE, '..', 'arc-noisy', 'ext', 'mlp_cumulant_propagation', 'src')
    if ARC_SRC not in sys.path:
        sys.path.insert(0, ARC_SRC)
    import logging
    logging.getLogger('mlp_kprop').setLevel(logging.ERROR)
    from mlp_kprop import kprop_harmonic as KH
    from mlp_kprop.harmonic import HTensor
    from mlp_kprop.flop_utils import NamedFlopCounter, flop_name
    from mlp_kprop.wick import WICK_COEF_D
    nonlin = 'tanh' if net.kind == 'tanh' else 'relu'
    counter = NamedFlopCounter()
    means = {}; flops = {}
    counter.__enter__()
    try:
        with flop_name('kprop'):
            K = KH.coerce_input({1: mu.clone(), 2: Sig.clone()}, k_max=2, kind=KH.SIMPLE)
        base = counter.total()
        for l in range(d, Lmax):
            with flop_name('kprop'):
                W = net.W64[l]
                b = net.b64[l] if net.b is not None else None
                K = KH.linear_kprop(K, W, k_max=2, set_metric=net.sw2, bias=b)
                K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D[nonlin], k_max=2, kind=KH.SIMPLE,
                                    use_pK=True, factor=False)
                means[l] = K[1].to_tensor().clone()
                if l < Lmax - 1 and net.sigma > 0:
                    K2 = K[2]
                    core = K2.core.clone(); core.diagonal().add_(float(net.s[l]) ** 2)
                    K[2] = HTensor(core=core, r=0, metric=K2.metric)
            flops[l] = counter.total() - base                # ARC's discounted count, layers d+1..l+1
    finally:
        counter.__exit__(None, None, None)
    tot = counter.total()
    return means, flops, tot


def main():
    regime, n, sigma, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    NT = int(sys.argv[5]) if len(sys.argv) > 5 else 1 << 18
    NS = int(sys.argv[6]) if len(sys.argv) > 6 else 1 << 15
    NA = int(sys.argv[7]) if len(sys.argv) > 7 else 1 << 15
    do_kprop = (len(sys.argv) <= 8 or sys.argv[8] != 'nokprop')
    t0 = time.time()
    net = pnets.PNet(regime, n, sigma, seed)
    L = net.L
    gen = torch.Generator().manual_seed(9_100_000 + 1000 * seed + n)
    T = truth_run(net, NT, gen); T['NT'] = NT
    t_truth = time.time() - t0
    res = dict(regime=regime, n=n, sigma=sigma, seed=seed, NT=NT, NS=NS, NA=NA, L=L, depths=list(DEPTHS),
               r=net.r, s=net.s, flops_path={str(l): net.flops_path(l) for l in range(1, L + 1)})
    # per-layer structure from the truth
    struct = []
    for l in range(L):
        mu = T['mean'][l].numpy(); C = T['cov'][l].numpy(); e2 = float((T['var'][l] + T['mean'][l] ** 2).sum())
        st = pr_stats(C, mu, e2)
        if 'p_on' in T:
            p = T['p_on'][l]; mg = T['margin'][l]
            st.update(p_dead=float((p == 0).double().mean()), p_alive_always=float((p == 1).double().mean()),
                      mixed_any=float(((p > 0) & (p < 1)).double().mean()),
                      mixed_01=float((torch.minimum(p, 1 - p) > 0.01).double().mean()),
                      mixed_10=float((torch.minimum(p, 1 - p) > 0.10).double().mean()),
                      margin_lt05=float((mg < 0.5).double().mean()), margin_lt1=float((mg < 1).double().mean()),
                      margin_lt2=float((mg < 2).double().mean()), margin_lt3=float((mg < 3).double().mean()),
                      margin_median=float(mg.median()))
        else:
            st.update(sat=float(T['sat'][l]))
        st['var_mean'] = float(T['var'][l].mean()); st['mean_sq'] = float((T['mean'][l] ** 2).mean())
        struct.append(st)
    res['struct'] = struct
    # stand-ins
    est = {}
    for kind in ('MF', 'MM'):
        for d in DEPTHS:
            if kind == 'MF':
                mm_, vv_ = pnets.mf_standin(net, d)
                mu_t = torch.full((n,), mm_, dtype=F32); sd = math.sqrt(vv_)

                def sampler(m, g, mu_t=mu_t, sd=sd):
                    return mu_t[None, :] + sd * torch.randn(m, n, generator=g, dtype=F32)
                Kmu = torch.full((n,), mm_, dtype=F64); KS = torch.eye(n, dtype=F64) * vv_
            else:
                mu = T['mean'][d - 1]; Sig = T['cov'][d - 1] + (net.s[d - 1] ** 2) * torch.eye(n, dtype=F64)
                ev, U = torch.linalg.eigh(Sig)
                A = (U * ev.clamp_min(0).sqrt()[None, :]).to(F32); mu_t = mu.to(F32)

                def sampler(m, g, mu_t=mu_t, A=A):
                    return mu_t[None, :] + torch.randn(m, n, generator=g, dtype=F32) @ A.T
                Kmu = mu.clone(); KS = Sig.clone()
            mean, var = push(net, d, sampler, NS, gen)
            row = {}
            for l in range(d, L):
                row[str(l + 1)] = nmse(mean[l], var[l], NS, T, l)
            est[f'{kind}_{d}'] = row
            est[f'{kind}_{d}_var'] = {str(l + 1): var_err(var[l], T, l) for l in range(d, L)}
            if do_kprop and net.kind != 'relu_ln' and d in KPROP_DEPTHS.get(n, DEPTHS):
                try:
                    means, fl, tot = kprop_from(net, d, Kmu, KS, L)
                    rowk = {}; rowf = {}
                    for l in range(d, L):
                        rowk[str(l + 1)] = nmse(means[l], None, 0, T, l)
                        rowf[str(l + 1)] = fl[l]
                    est[f'kprop{kind}_{d}'] = rowk; est[f'kprop{kind}_{d}_flops'] = rowf
                except Exception as e:                              # report, do not hide
                    est[f'kprop{kind}_{d}_error'] = repr(e)
    # full kprop from the true input law (ARC's standard use)
    if do_kprop and net.kind != 'relu_ln':
        means, fl, tot = kprop_from(net, 0, torch.zeros(n, dtype=F64), torch.eye(n, dtype=F64), L)
        est['kprop_input'] = {str(l + 1): nmse(means[l], None, 0, T, l) for l in range(L)}
        est['kprop_input_flops'] = {str(l + 1): fl[l] for l in range(L)}
    # input-law swaps
    aux = net.alt_aux()
    for law in ('aniso', 'point'):
        mean, var = push(net, 0, lambda m, g, law=law: net.inputs(m, g, law, aux), NA, gen)
        est[f'law_{law}'] = {str(l + 1): nmse(mean[l], var[l], NA, T, l) for l in range(L)}
        est[f'law_{law}_var'] = {str(l + 1): var_err(var[l], T, l) for l in range(L)}
    # control: a second independent Gaussian-input run of NA paths (should be ~0 after debiasing)
    mean, var = push(net, 0, lambda m, g: net.inputs(m, g), NA, gen)
    est['law_gauss_control'] = {str(l + 1): nmse(mean[l], var[l], NA, T, l) for l in range(L)}
    est['law_gauss_control_var'] = {str(l + 1): var_err(var[l], T, l) for l in range(L)}
    # Amendment 1 (2026-10-29 14:29Z): scale-matched mean-field stand-in for the ReLU kinds (MFr): the MF shape rescaled
    # so that the activation RMS equals the net's true per-layer RMS (one scalar per layer, from the truth run), noise
    # added. Removes the overall-gain error of MF; run after everything above so earlier random streams are unchanged.
    if net.kind == 'relu':
        rms = (T['var'] + T['mean'] ** 2).mean(1).sqrt()
        for d in (8, 16, 24, 32, 40, 48, 56, 60, 63):
            rd = float(rms[d - 1]); sd2 = float(net.s[d - 1]) ** 2 if net.sigma > 0 else 0.0
            mm_ = rd / math.sqrt(math.pi); vv_ = rd ** 2 * (1 - 1 / math.pi) + sd2
            mu_t = torch.full((n,), mm_, dtype=F32); sdv = math.sqrt(vv_)

            def sampler(m, g, mu_t=mu_t, sdv=sdv):
                return mu_t[None, :] + sdv * torch.randn(m, n, generator=g, dtype=F32)
            mean, var = push(net, d, sampler, NS, gen)
            est[f'MFr_{d}'] = {str(l + 1): nmse(mean[l], var[l], NS, T, l) for l in range(d, L)}
    res['est'] = est
    res['seconds'] = time.time() - t0; res['seconds_truth'] = t_truth
    os.makedirs(os.path.join(pnets.HERE, 'results', 'backward'), exist_ok=True)
    fn = os.path.join(pnets.HERE, 'results', 'backward', net.name() + '.json')
    json.dump(res, open(fn, 'w'))
    np.savez_compressed(fn.replace('.json', '_truth.npz'), mean=T['mean'].numpy(), var=T['var'].numpy(), NT=NT)
    print(fn, f"{res['seconds']:.0f}s (truth {t_truth:.0f}s)")


if __name__ == '__main__':
    main()
