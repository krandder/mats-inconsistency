"""Part 2 (PREREG P1, P3, P4, P5): per-path Jacobians, fixed blocks, and weight-derived (gate-independent) subspaces.
v2 (after the GPT-6 review): M includes every path with a nonzero Jacobian (noise can revive a path that once died),
the random-mask PR ignores zero products, and the reference path is one with a nonzero Jacobian at the cut.
usage: python p2_jac.py FAM N L SIGMA SEEDS(comma) OUTJSON
Plain: J_x = d a_L / d z_c = D_L W_L ... D_{c+1} W_{c+1} D_c (z_c pre-activation at the cut, noise on).
Residual: J_x = d h_L / d h_c = prod_{l=c+1..L} (I + V_l D_l W_l / sqrt(L)).
M = E_x J_x^T J_x; Sigma = Cov(z_c) (residual: Cov(h_c)); S = Sigma^(1/2) M Sigma^(1/2).
Candidate subspaces U (n x k): lin (unmasked product), rand (E over iid Bernoulli(1/2) masks), avg (marginal gate
probabilities), att (one reference path), pcs (top PCs of Sigma; reference), opt (Sigma^-1/2 V_k(S)).
Scores: tail_U = tr(M Schur_U) / tr(M Sigma) (first-order error share, Theorem B), capM_U = tr(P_U M) / tr M.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]
OUT = sys.argv[6]
P = 1024; NRAND = 1024
LPS = sorted(set([lp for lp in (4, 8, 16, 32, 64, 128, 192, 248) if lp <= L - 4] + [L - 4]))
DELTAS = (math.exp(-1), math.exp(-2), 0.01, 0.001)
DT = nd.DT


def step_J(J, l, D, net):
    """Apply layer l (0-based, the layer after the current J's output side) to J (B, n, n) with masks D (B, n)."""
    if FAM == 'plain':
        return D[:, :, None] * (net.W[l] @ J)
    return J + net.c * (net.V[l] @ (D[:, :, None] * (net.W[l] @ J)))


def chain(J0, masks, lo, hi, net):
    """Push J0 (B, n, n) through layers lo..hi-1 (0-based) with masks[l] (B, n); renormalise, return J, logscale."""
    J = J0.clone(); ls = torch.zeros(J.shape[0], dtype=DT)
    for l in range(lo, hi):
        J = step_J(J, l, masks[l], net)
        nr = J.flatten(1).norm(dim=1).clamp_min(1e-300)
        J = J / nr[:, None, None]; ls = ls + torch.log(nr)
    return J, ls


def topk_right(A, k):
    U, s, Vh = torch.linalg.svd(A)
    return Vh[:k].T


def schur_tail(M, Sig, U):
    G = U.T @ Sig @ U
    Sch = Sig - Sig @ U @ torch.linalg.pinv(G, hermitian=True) @ U.T @ Sig
    return float(torch.trace(M @ Sch) / torch.trace(M @ Sig))


def capM(M, U):
    Q, _ = torch.linalg.qr(U)
    return float(torch.trace(Q.T @ M @ Q) / torch.trace(M))


def psd_sqrt(A):
    ev, V = torch.linalg.eigh(A)
    return V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T


out = {}
for seed in SEEDS:
    t0 = time.time()
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(6_000_011 + seed)
    x = torch.randn(P, N, generator=g, dtype=DT)
    h = x; masks = []; zs = {}; hs = {}; alive = torch.ones(P, dtype=torch.bool)
    for l in range(L):
        z = h @ net.W[l].T
        D = (z > 0).to(DT)
        masks.append(D)
        if FAM == 'plain':
            alive &= D.sum(1) > 0
        zs[l + 1] = z; hs[l] = h                       # hs[l]: residual stream entering layer l+1 (h_l)
        a = torch.relu(z)
        e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIGMA > 0 else 0.0
        h = a + e if FAM == 'plain' else h + ((a + e) @ net.V[l].T) * net.c
    out_x = h if FAM == 'res' else torch.relu(zs[L])
    res = dict(alive_frac=float(alive.double().mean()), cuts={})
    if FAM == 'plain' and alive.sum() < 8:
        res['dead_net'] = True
        out[seed] = res
        print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: DEAD net (alive {alive.sum().item()})", flush=True)
        continue
    pbar = [m.mean(0) for m in masks]
    ref = int(torch.nonzero(alive)[0]) if FAM == 'plain' else 0
    grand = torch.Generator().manual_seed(7_000_003 + seed)
    for lp in LPS:
        c = L - lp                                     # cut after layer c: plain J w.r.t. z_c (layer c pre-activation)
        I = torch.eye(N, dtype=DT).expand(P, N, N)
        if FAM == 'plain':
            J0 = masks[c - 1][:, :, None] * I          # D_c
            Sig = torch.cov(zs[c].T)
        else:
            J0 = I.clone()
            Sig = torch.cov(hs[c].T)                   # h_c, the stream entering layer c+1
        J, ls = chain(J0, masks, c, L, net)
        s = torch.linalg.svdvals(J)                    # per path, relative (renormalised) singular values
        okp = s[:, 0] > 0          # v2 (review): every path with a nonzero Jacobian, incl. noise-revived ones
        if not bool(okp[ref]):
            ref_c = int(torch.nonzero(okp & alive)[0]) if bool((okp & alive).any()) else int(torch.nonzero(okp)[0])
        else:
            ref_c = ref
        sv = s[okp]
        rel = sv / sv[:, :1]
        ns = {f"{d:.4g}": dict(mean=float(nd.n_surv(sv, d).double().mean()), median=float(nd.n_surv(sv, d).double().median()))
              for d in DELTAS}
        rank1 = float((rel[:, 1] < 1e-12).double().mean()) if N > 1 else 1.0
        r2 = rel[:, 1]; nz = r2 > 1e-12
        slope = float((torch.log(r2[nz]) / lp).mean()) if nz.sum() > 0 else None
        kF = [nd.kstar_F((s_ ** 2).numpy(), 1e-4) for s_ in sv[:256]]
        # M with true scales (common shift)
        w = torch.exp(2 * (ls - ls[okp].max()))
        w[~okp] = 0
        M = torch.einsum('p,pij,pik->jk', w, J, J) / P
        Ms = psd_sqrt(Sig)
        S = Ms @ M @ Ms
        evM = torch.linalg.eigvalsh(M).flip(0).clamp_min(0).numpy()
        evS, VS = torch.linalg.eigh(S); evS = evS.flip(0).clamp_min(0); VS = VS.flip(1)
        prM = float(evM.sum() ** 2 / (evM ** 2).sum()); prS = float(evS.sum() ** 2 / (evS ** 2).sum())
        # candidates
        if FAM == 'plain':
            Plin = torch.eye(N, dtype=DT)
            for l in range(c, L): Plin = net.W[l] @ Plin; Plin = Plin / Plin.norm()
            Pavg = torch.diag(pbar[c - 1])
            for l in range(c, L): Pavg = pbar[l][:, None] * (net.W[l] @ Pavg); Pavg = Pavg / Pavg.norm()
            Rm = [(torch.rand(NRAND, N, generator=grand, dtype=DT) < 0.5).to(DT) for _ in range(L)]
            J0r = Rm[c - 1][:, :, None] * torch.eye(N, dtype=DT).expand(NRAND, N, N)
        else:
            Plin = torch.eye(N, dtype=DT)
            for l in range(c, L): Plin = Plin + net.c * net.V[l] @ (net.W[l] @ Plin)
            Pavg = torch.eye(N, dtype=DT)
            for l in range(c, L): Pavg = Pavg + net.c * net.V[l] @ (pbar[l][:, None] * (net.W[l] @ Pavg))
            Rm = [(torch.rand(NRAND, N, generator=grand, dtype=DT) < 0.5).to(DT) for _ in range(L)]
            J0r = torch.eye(N, dtype=DT).expand(NRAND, N, N).clone()
        Jr, lsr = chain(J0r, Rm, c, L, net)
        wr = torch.exp(2 * (lsr - lsr.max()))
        Mr = torch.einsum('p,pij,pik->jk', wr, Jr, Jr) / NRAND
        _, sr, Vhr = torch.linalg.svd(Jr)
        nzr = sr[:, 0] > 0
        v1 = Vhr[nzr, 0, :]
        Pv = (v1[:, :, None] * v1[:, None, :]).mean(0)
        evPv = torch.linalg.eigvalsh(Pv).clamp_min(0)
        pr_worst = float(evPv.sum() ** 2 / evPv.pow(2).sum())
        # gains (log): real paths' top singular value vs random masks' max, and the product of operator norms
        lg_real = (ls[okp] + torch.log(sv[:, 0])).numpy()
        lg_rand = (lsr[nzr] + torch.log(sr[nzr, 0])).numpy()
        lg_opnorm = float(sum(math.log(float(torch.linalg.matrix_norm(net.W[l], 2))) for l in range(c, L))) if FAM == 'plain' else \
            float(sum(math.log(float(torch.linalg.matrix_norm(torch.eye(N, dtype=DT) + net.c * net.V[l] @ net.W[l], 2)) +
                               0.0) for l in range(c, L)))
        Jref = J[ref_c] * torch.exp(ls[ref_c] - ls[okp].max())
        cands = {}
        for k in (1, 2, 4):
            if k >= N: continue
            Uopt = torch.linalg.pinv(Ms) @ VS[:, :k]
            evSig, VSig = torch.linalg.eigh(Sig); Upcs = VSig.flip(1)[:, :k]
            Us = dict(opt=Uopt, pcs=Upcs, lin=topk_right(Plin, k), rand=torch.linalg.eigh(Mr)[1].flip(1)[:, :k],
                      avg=topk_right(Pavg, k), att=topk_right(Jref, k))
            cands[k] = {nm: dict(tail=schur_tail(M, Sig, U), capM=capM(M, U)) for nm, U in Us.items()}
            cands[k]['opt_tail_exact'] = float(evS[k:].sum() / evS.sum())
            cands[k]['capM_opt'] = float(evM[:k].sum() / evM.sum())
        res['cuts'][lp] = dict(c=c, n_paths=int(okp.sum()), nsurv=ns, rank1_frac=rank1, slope_log_s2s1=slope,
                               kF_path_median=float(np.median(kF)), kF_path_mean=float(np.mean(kF)),
                               kM={str(e): nd.kstar_F(evM, e) for e in (1e-1, 1e-2, 1e-4)},
                               kS={str(e): nd.kstar_F(evS.numpy(), e) for e in (1e-1, 1e-2, 1e-4)},
                               prM=prM, prS=prS, pr_worst=pr_worst, frac_zero_rand=float((~nzr).double().mean()),
                               gain_real_median=float(np.median(lg_real)), gain_rand_max=float(lg_rand.max()),
                               gain_rand_median=float(np.median(lg_rand)), gain_opnorm=lg_opnorm, cands=cands)
        cc = res['cuts'][lp]
        print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed} L'={lp}: Nsurv(.01) mean={ns['0.01']['mean']:.2f} med={ns['0.01']['median']:.0f}"
              f" rank1={rank1:.2f} slope={slope if slope is None else round(slope,4)} kM(1e-2)={cc['kM']['0.01']} kS(1e-2)={cc['kS']['0.01']}"
              f" prM={prM:.2f} prWorst={pr_worst:.2f} gain(real med/rand max/opnorm)={cc['gain_real_median']:.1f}/{cc['gain_rand_max']:.1f}/{lg_opnorm:.1f}"
              f" k1 tail opt/lin/rand/avg/att/pcs=" + '/'.join(f"{cands[1][nm]['tail']:.3f}" for nm in ('opt', 'lin', 'rand', 'avg', 'att', 'pcs'))
              + f" k1 capM topM/lin/rand/avg/att/pcs={cands[1]['capM_opt']:.3f}/" + '/'.join(f"{cands[1][nm]['capM']:.3f}" for nm in ('lin', 'rand', 'avg', 'att', 'pcs')),
              flush=True)
    res['sec'] = time.time() - t0
    out[seed] = res
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, seeds=SEEDS, P=P, LPS=LPS, res=out), open(OUT, 'w'))
