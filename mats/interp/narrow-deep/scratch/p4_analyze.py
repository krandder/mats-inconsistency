"""Part 4: structure of trained narrow nets (PREREG Part 4): collapse/gates per step or layer, per-path Jacobians
of the output w.r.t. the state at a cut, fixed blocks, weight-derived subspaces, variance split.
usage: python p4_analyze.py KIND NPZ OUTJSON      (KIND = rnn | mlp)
RNN: state h_t (after noise); output logits W_oh h_T; J = W_oh D_T W_hh ... D_{c+1} W_hh (w.r.t. h_c).
MLP res: J = prod (I + V_l D_l W_l) w.r.t. h_c; plain: W_out D_L W_L ... D_{c+1} W_{c+1} w.r.t. h_c.
Candidates as in p2_jac.py: lin (masks dropped), rand (iid Bernoulli(1/2) masks), avg (marginal gate
probabilities), att (one reference path), pcs (top PCs of Sigma), opt (Sigma^-1/2 V_k(S)); capM, tail.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd, tnets

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
depth = net.T if KIND == 'rnn' else net.L
width = net.d if KIND == 'rnn' else net.n
nin = net.n if KIND == 'rnn' else net.n
P = 2048; NRAND = 1024
g = torch.Generator().manual_seed(21_000_007)
t0 = time.time()


def draw(Pn):
    return torch.randn(Pn, nin, generator=g, dtype=DT)


# ---------------------------------------------------------------- collapse / gates
x = draw(P)
xn = draw(512).repeat_interleave(2, 0)
_, kept, masks = net.forward(torch.cat([x, xn]), gen=g, keep=set(range(1, depth + 1)), return_masks=True)
rows = []
for l in range(1, depth + 1):
    h = kept[l][:P]; D = masks[l - 1]
    A, B = h[0::2], h[1::2]
    cos = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-300)
    th = torch.arccos(cos.clamp(-1, 1))
    gd = (D[:P][0::2] != D[:P][1::2]).double().mean()
    gn = (D[P:][0::2] != D[P:][1::2]).double().mean()
    C = torch.cov(h.T); ev = torch.linalg.eigvalsh(C).clamp_min(0)
    rows.append(dict(l=l, th_med=float(th.median()), gd_pair=float(gd), gd_noise=float(gn), act=float(D[:P].mean()),
                     pr_cov=float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-300))))


# ---------------------------------------------------------------- Jacobians at cuts
def jstep(J, l, D):
    """Apply layer/step index l (0-based) with masks D (B, width) to J (B, width, k)."""
    if KIND == 'rnn':
        return D[:, :, None] * (net.W_hh @ J)
    if net.fam == 'res':
        return J + net.V[l] @ (D[:, :, None] * (net.W[l] @ J))
    return D[:, :, None] * (net.W[l] @ J)


def readout(J):
    if KIND == 'rnn':
        return net.W_oh @ J
    if net.fam == 'plain':
        return net.W_out @ J
    return J


def chain(J0, Ms, lo, hi):
    J = J0.clone(); ls = torch.zeros(J.shape[0], dtype=DT)
    for l in range(lo, hi):
        J = jstep(J, l, Ms[l])
        nr = J.flatten(1).norm(dim=1).clamp_min(1e-300); J = J / nr[:, None, None]; ls = ls + torch.log(nr)
    return readout(J), ls


def topk_right(A, k):
    return torch.linalg.svd(A)[2][:k].T


def psd_sqrt(A):
    ev, V = torch.linalg.eigh(A)
    return V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T


def schur_tail(M, Sig, U):
    G = U.T @ Sig @ U
    Sch = Sig - Sig @ U @ torch.linalg.pinv(G, hermitian=True) @ U.T @ Sig
    return float(torch.trace(M @ Sch) / torch.trace(M @ Sig))


def capM(M, U):
    Q, _ = torch.linalg.qr(U)
    return float(torch.trace(Q.T @ M @ Q) / torch.trace(M))


masksP = [m[:P] for m in masks]
pbar = [m.mean(0) for m in masksP]
grand = torch.Generator().manual_seed(22_000_009)
Rm = [(torch.rand(NRAND, width, generator=grand, dtype=DT) < 0.5).to(DT) for _ in range(depth)]
cuts = {}
for lp in sorted(set([lp for lp in (4, 8, 16, 32) if lp < depth] + [depth // 2, depth - 4])):
    c = depth - lp
    I = torch.eye(width, dtype=DT)
    J, ls = chain(I.expand(P, width, width), masksP, c, depth)
    s = torch.linalg.svdvals(J)
    ok = s[:, 0] > 0
    sv = s[ok]
    w = torch.exp(2 * (ls - ls[ok].max())); w[~ok] = 0
    M = torch.einsum('p,pij,pik->jk', w, J, J) / P
    Sig = torch.cov(kept[c][:P].T)
    Ms = psd_sqrt(Sig); S = Ms @ M @ Ms
    evM = torch.linalg.eigvalsh(M).flip(0).clamp_min(0).numpy()
    evS, VS = torch.linalg.eigh(S); evS = evS.flip(0).clamp_min(0); VS = VS.flip(1)
    # candidates
    Plin = I.clone()
    for l in range(c, depth):
        if KIND == 'rnn': Plin = net.W_hh @ Plin
        elif net.fam == 'res': Plin = Plin + net.V[l] @ (net.W[l] @ Plin)
        else: Plin = net.W[l] @ Plin
        Plin = Plin / Plin.norm()
    Plin = readout(Plin)
    Pavg = I.clone()
    for l in range(c, depth):
        Pavg = jstep(Pavg[None], l, pbar[l][None])[0]; Pavg = Pavg / Pavg.norm()
    Pavg = readout(Pavg)
    Jr, lsr = chain(I.expand(NRAND, width, width), Rm, c, depth)
    wr = torch.exp(2 * (lsr - lsr.max()))
    Mr = torch.einsum('p,pij,pik->jk', wr, Jr, Jr) / NRAND
    v1 = torch.linalg.svd(Jr)[2][:, 0, :]
    Pv = (v1[:, :, None] * v1[:, None, :]).mean(0); evPv = torch.linalg.eigvalsh(Pv).clamp_min(0)
    ref = int(torch.nonzero(ok)[0])
    cands = {}
    for k in (1, 2, 4):
        Us = dict(opt=torch.linalg.pinv(Ms) @ VS[:, :k], pcs=torch.linalg.eigh(Sig)[1].flip(1)[:, :k],
                  lin=topk_right(Plin, k), rand=torch.linalg.eigh(Mr)[1].flip(1)[:, :k], avg=topk_right(Pavg, k),
                  att=topk_right(J[ref], k))
        cands[k] = {nm: dict(tail=schur_tail(M, Sig, U), capM=capM(M, U)) for nm, U in Us.items()}
        cands[k]['capM_opt'] = float(evM[:k].sum() / evM.sum()); cands[k]['tail_opt'] = float(evS[k:].sum() / evS.sum())
    cuts[lp] = dict(c=c, nsurv={f'{d:.4g}': float(nd.n_surv(sv, d).double().mean()) for d in (math.exp(-1), 0.01, 0.001)},
                    kM={str(e): nd.kstar_F(evM, e) for e in (1e-1, 1e-2, 1e-4)},
                    kS={str(e): nd.kstar_F(evS.numpy(), e) for e in (1e-1, 1e-2, 1e-4)},
                    prM=float(evM.sum() ** 2 / (evM ** 2).sum()), pr_worst=float(evPv.sum() ** 2 / evPv.pow(2).sum()),
                    cands=cands)
    cc = cuts[lp]
    print(f"{os.path.basename(NPZ)} L'={lp} (c={c}): Nsurv(.01)={cc['nsurv']['0.01']:.2f} kM={cc['kM']} kS={cc['kS']} prM={cc['prM']:.2f} "
          f"prWorst={cc['pr_worst']:.1f} k1 capM top/lin/rand/avg/att/pcs={cands[1]['capM_opt']:.2f}/" +
          '/'.join(f"{cands[1][nm]['capM']:.2f}" for nm in ('lin', 'rand', 'avg', 'att', 'pcs')) +
          f" k2 tail opt/lin/rand/avg/att/pcs=" + '/'.join(f"{cands[2][nm]['tail']:.3f}" for nm in ('opt', 'lin', 'rand', 'avg', 'att', 'pcs')), flush=True)


# ---------------------------------------------------------------- variance split
def anova(Y):
    gm = Y.mean(1); within = Y.var(1, unbiased=True).sum(-1).mean(); between = gm.var(0, unbiased=True).sum()
    tot = Y.reshape(-1, Y.shape[-1]).var(0, unbiased=True).sum()
    return float((between - within / Y.shape[1]) / tot)


Mv, K = 2048, 8
xx = draw(Mv).repeat_interleave(K, 0)
vs = dict(input=anova(net.forward(xx, gen=g).reshape(Mv, K, -1)))
u = draw(Mv); u = u / u.norm(dim=1, keepdim=True)
rad = draw(Mv * K).norm(dim=1, keepdim=True)
vs['direction'] = anova(net.forward(u.repeat_interleave(K, 0) * rad, gen=g).reshape(Mv, K, -1))
rad = draw(Mv).norm(dim=1, keepdim=True).repeat_interleave(K, 0)
u = draw(Mv * K); u = u / u.norm(dim=1, keepdim=True)
vs['norm'] = anova(net.forward(u * rad, gen=g).reshape(Mv, K, -1))
print(f"{os.path.basename(NPZ)}: variance shares input={vs['input']:.3f} direction={vs['direction']:.3f} norm={vs['norm']:.3f}; "
      f"gate disagreement pairs at l=1/{depth//4}/{depth//2}/{depth}: " +
      '/'.join(f"{rows[l-1]['gd_pair']:.3f}" for l in (1, depth // 4, depth // 2, depth)) +
      f"; median angle: " + '/'.join(f"{rows[l-1]['th_med']:.3f}" for l in (1, depth // 4, depth // 2, depth)) +
      f"; noise-only gd at {depth}: {rows[-1]['gd_noise']:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(npz=NPZ, kind=KIND, rows=rows, cuts=cuts, varsplit=vs), open(OUT, 'w'))
