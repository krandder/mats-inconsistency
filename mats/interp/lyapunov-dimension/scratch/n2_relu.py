"""N2 + N3: downstream Jacobians of ARC's He-ReLU nets at a cut.
Per path x (noisy forward pass to the cut c, then L' more layers with the path's own noise):
  J_x = d a_{c+L'} / d z_c = D_{c+L'} W_{c+L'} ... D_{c+1} W_{c+1} D_c.
N2 (single path): singular values of J_x, and of the compression Jhat = P_{e_out-perp} J P_{e_in-perp}
  (e_in = a_c/|a_c|, e_out = a_{c+L'}/|a_{c+L'}|), against (i) a simulated product of independent Gaussian matrices of
  sizes (m_l - 1) x (m_{l-1} - 1), entries N(0, 2/n), with the path's own active counts m_l (Theorem C), and
  (ii) Fuss-Catalan with m = n/2 (lib/fc.py, run separately).
N3 (input-averaged): M = E_x J_x^T J_x; tr M, tr M^2, participation ratio, eigenvalues; pairwise mask overlaps per
  layer (fraction of neurons active on both paths) for the two-replica recursion; state covariance Sigma = Cov(z_c);
  S = Sigma^1/2 M Sigma^1/2; k*_F(delta) for J^T J (per path, averaged), M, P M P, Sigma, S.
python n2_relu.py --seed 113 --n 256 --L 64 --cut 32 --Lps 8,16,32 --paths 2048 --sigma 0.001"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import anet

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=113); ap.add_argument('--n', type=int, default=256)
ap.add_argument('--L', type=int, default=64); ap.add_argument('--cut', type=int, default=32)
ap.add_argument('--Lps', default='8,16,32'); ap.add_argument('--paths', type=int, default=2048)
ap.add_argument('--svpaths', type=int, default=64); ap.add_argument('--statepaths', type=int, default=1 << 15)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--threads', type=int, default=2); ap.add_argument('--out', default='runs')
ap.add_argument('--tag', default='')
A = ap.parse_args()
torch.set_num_threads(A.threads)
DT = torch.float64
n, L, c = A.n, A.L, A.cut
Lps = sorted(int(v) for v in A.Lps.split(',') if c + int(v) <= L)
rtab = json.load(open(os.path.join(HERE, '..', 'lib', 'rtable_ng.json')))
key = f"{n}_{L}_{A.seed}_0.001"
r = rtab[key] if A.sigma > 0 else None
net = anet.Net(A.seed, n, L, A.sigma, r=r)
W = [w.to(DT) for w in net.W]
s = net.s.to(DT) if A.sigma > 0 else torch.zeros(L, dtype=DT)
DELTAS = [math.exp(-1), math.exp(-2), 1e-2, 1e-3]
g = torch.Generator().manual_seed(4242 + 17 * A.seed + c)


def kF_counts(ev):
    """ev: eigenvalues (of a PSD matrix) descending. k*_F(delta): smallest k with tail beyond k <= delta^2 total."""
    ev = np.clip(np.asarray(ev, dtype=float), 0, None)
    tail = np.cumsum(ev[::-1])[::-1]
    tot = tail[0]
    out = {}
    for d in DELTAS:
        nxt = np.append(tail[1:], 0.0)
        out[f'{d:.3g}'] = int(np.argmax(nxt <= d * d * tot)) + 1
    return out


def nsurv(sv):
    sv = np.asarray(sv, dtype=float)
    return {f'{d:.3g}': int((sv >= d * sv[0]).sum()) for d in DELTAS}


t0 = time.time()
# ---------------- state covariance at the cut (forward only)
Zs = []
done = 0
while done < A.statepaths:
    b = min(4096, A.statepaths - done)
    h = torch.randn(b, n, generator=g, dtype=DT)
    for l in range(c):
        z = h @ W[l].T
        if l < c - 1:
            h = torch.relu(z) + s[l] * torch.randn(b, n, generator=g, dtype=DT)
    Zs.append(z); done += b
Z = torch.cat(Zs)
mu = Z.mean(0); Sigma = torch.cov(Z.T)
evS, VS = torch.linalg.eigh(Sigma)
evS = evS.flip(0).clamp_min(0); VS = VS.flip(1)
Sig_half = (VS * evS.sqrt()) @ VS.T
u = mu / mu.norm(); P = torch.eye(n, dtype=DT) - torch.outer(u, u)
print(f"state at cut {c}: {A.statepaths} paths, PR {float(evS.sum()**2/(evS**2).sum()):.2f}, "
      f"top PC cos with mean {float(abs(VS[:, 0] @ u)):.3f}  {time.time()-t0:.0f}s", flush=True)

# ---------------- Jacobians along paths
Ms = {Lp: torch.zeros(n, n, dtype=DT) for Lp in Lps}
sv_full = {Lp: [] for Lp in Lps}; sv_hat = {Lp: [] for Lp in Lps}; sv_sim = {Lp: [] for Lp in Lps}
mcount = {Lp: [] for Lp in Lps}
masks_keep = []   # masks at layers c..L for the first 256 paths (overlap statistics)
hdirs_keep = []   # directions of h_{l-1} at layers c+1..L for the same paths
trJJ = {Lp: [] for Lp in Lps}; trJJ2 = {Lp: [] for Lp in Lps}
done = 0
while done < A.paths:
    b = min(A.batch, A.paths - done)
    h = torch.randn(b, n, generator=g, dtype=DT)
    for l in range(c):
        z = h @ W[l].T
        if l < c - 1:
            h = torch.relu(z) + s[l] * torch.randn(b, n, generator=g, dtype=DT)
    # z = z_c ; build J over the next layers
    Dc = (z > 0).to(DT)
    a = torch.relu(z)
    e_in = a / a.norm(dim=1, keepdim=True)
    J = torch.diag_embed(Dc)
    keepm = [Dc.clone()]; keeph = []
    h = a + s[c - 1] * torch.randn(b, n, generator=g, dtype=DT)
    act = []
    for k in range(1, max(Lps) + 1):
        lw = c + k - 1                     # 0-based index of W_{c+k}
        keeph.append(h / h.norm(dim=1, keepdim=True))
        z = h @ W[lw].T
        D = (z > 0).to(DT)
        keepm.append(D.clone()); act.append(D.sum(1))
        J = D[:, :, None] * (W[lw] @ J)
        a = torch.relu(z)
        if k in Lps:
            Lp = k
            JtJ = J.transpose(1, 2) @ J
            Ms[Lp] += JtJ.sum(0)
            trJJ[Lp].append(torch.diagonal(JtJ, dim1=1, dim2=2).sum(1))
            trJJ2[Lp].append((JtJ * JtJ).sum((1, 2)))
            if len(sv_full[Lp]) * b < A.svpaths or done < A.svpaths:
                sv_full[Lp].append(torch.linalg.svdvals(J))
                e_out = a / a.norm(dim=1, keepdim=True)
                Po = torch.eye(n, dtype=DT)[None] - e_out[:, :, None] * e_out[:, None, :]
                Pi = torch.eye(n, dtype=DT)[None] - e_in[:, :, None] * e_in[:, None, :]
                sv_hat[Lp].append(torch.linalg.svdvals(Po @ J @ Pi))
                # simulated product of independent Gaussians with this path's active counts
                ms = [int(v) for v in Dc.sum(1)]
                sims = []
                for i in range(b):
                    dims = [int(Dc[i].sum()) - 1] + [int(act[j][i]) - 1 for j in range(Lp)]
                    G = torch.eye(dims[0], dtype=DT)
                    for j in range(Lp):
                        G = (torch.randn(dims[j + 1], dims[j], generator=g, dtype=DT) * math.sqrt(2.0 / n)) @ G
                    sv = torch.zeros(n, dtype=DT)
                    v = torch.linalg.svdvals(G)
                    sv[:v.shape[0]] = v
                    sims.append(sv)
                sv_sim[Lp].append(torch.stack(sims))
                mcount[Lp].append(torch.stack([Dc.sum(1)] + act[:Lp], 1))
        if k < max(Lps):
            h = a + s[lw] * torch.randn(b, n, generator=g, dtype=DT)
    if done < 256:
        masks_keep.append(torch.stack(keepm, 1)); hdirs_keep.append(torch.stack(keeph, 1))
    done += b
    if done % 512 == 0:
        print(f"  paths {done}  {time.time()-t0:.0f}s", flush=True)

res = dict(seed=A.seed, n=n, L=L, cut=c, sigma=A.sigma, paths=A.paths, Lps=Lps,
           state=dict(pr=float(evS.sum()**2/(evS**2).sum()), kF=kF_counts(evS.numpy()),
                      ev_top=[float(v) for v in evS[:64]], tr=float(evS.sum())))
# overlap statistics (first 256 paths): per layer, mean over pairs of q = |D cap D'|/n and of (pi - theta)/(2 pi)
Mk = torch.cat(masks_keep)[:256]; Hk = torch.cat(hdirs_keep)[:256]
npair = Mk.shape[0]
iu = torch.triu_indices(npair, npair, 1)
q_meas, q_angle, p_act = [], [], []
for j in range(Mk.shape[1]):
    Dm = Mk[:, j, :]
    O = (Dm @ Dm.T) / n
    q_meas.append(float(O[iu[0], iu[1]].mean())); p_act.append(float(Dm.mean()))
    if j >= 1:
        Hd = Hk[:, j - 1, :]
        C = (Hd @ Hd.T).clamp(-1, 1)
        th = torch.arccos(C[iu[0], iu[1]])
        q_angle.append(float(((math.pi - th) / (2 * math.pi)).mean()))
res['overlap'] = dict(q_meas=q_meas, q_angle=[None] + q_angle, p_active=p_act)
print("pairwise both-active fraction q per layer c..: " + " ".join(f"{v:.3f}" for v in q_meas[:9]) + " ... "
      + " ".join(f"{v:.3f}" for v in q_meas[-3:]), flush=True)
print("angle prediction (pi-theta)/(2pi):          " + " ".join(f"{v:.3f}" for v in q_angle[:8]) + " ... "
      + " ".join(f"{v:.3f}" for v in q_angle[-3:]), flush=True)

res['Lp'] = {}
for Lp in Lps:
    M = Ms[Lp] / A.paths
    evM = torch.linalg.eigvalsh(M).flip(0).clamp_min(0)
    PMP = P @ M @ P
    evPMP = torch.linalg.eigvalsh(PMP).flip(0).clamp_min(0)
    S = Sig_half @ M @ Sig_half
    evSS = torch.linalg.eigvalsh(0.5 * (S + S.T)).flip(0).clamp_min(0)
    PSP = (P @ Sig_half @ P) @ M @ (P @ Sig_half @ P)
    evPSP = torch.linalg.eigvalsh(0.5 * (PSP + PSP.T)).flip(0).clamp_min(0)
    trj = torch.cat(trJJ[Lp]); trj2 = torch.cat(trJJ2[Lp])
    svf = torch.cat(sv_full[Lp]).numpy(); svh = torch.cat(sv_hat[Lp]).numpy(); svs = torch.cat(sv_sim[Lp]).numpy()
    pr_single = float((trj ** 2 / trj2).mean())
    rec = dict(
        trM=float(evM.sum()), trM2=float((evM ** 2).sum()), pr_M=float(evM.sum() ** 2 / (evM ** 2).sum()),
        pr_PMP=float(evPMP.sum() ** 2 / (evPMP ** 2).sum()), pr_S=float(evSS.sum() ** 2 / (evSS ** 2).sum()),
        pr_PSP=float(evPSP.sum() ** 2 / (evPSP ** 2).sum()),
        pr_single_mean=pr_single, mean_trJJ=float(trj.mean()), mean_trJJ2=float(trj2.mean()),
        kF=dict(M=kF_counts(evM.numpy()), PMP=kF_counts(evPMP.numpy()), S=kF_counts(evSS.numpy()),
                PSP=kF_counts(evPSP.numpy()),
                single=[kF_counts(v ** 2) for v in svf], single_hat=[kF_counts(v ** 2) for v in svh],
                sim=[kF_counts(v ** 2) for v in svs]),
        nsurv=dict(single=[nsurv(v) for v in svf], single_hat=[nsurv(v) for v in svh], sim=[nsurv(v) for v in svs]),
        logratio=dict(single=np.log(svf[:, :129] / svf[:, :1]).mean(0).tolist(),
                      single_hat=np.log(np.maximum(svh[:, :129], 1e-300) / svh[:, :1]).mean(0).tolist(),
                      sim=np.log(np.maximum(svs[:, :129], 1e-300) / svs[:, :1]).mean(0).tolist()),
        log_s1=dict(single=float(np.log(svf[:, 0]).mean()), single_hat=float(np.log(svh[:, 0]).mean()),
                    sim=float(np.log(svs[:, 0]).mean())),
        ev_M=[float(v) for v in evM[:160]], ev_PMP=[float(v) for v in evPMP[:160]], ev_S=[float(v) for v in evSS[:160]],
        ev_PSP=[float(v) for v in evPSP[:160]])
    res['Lp'][Lp] = rec

    def avg(lst):
        return {d: float(np.mean([x[d] for x in lst])) for d in lst[0]}
    print(f"L' {Lp}: PR single path {pr_single:.2f}; PR(M) {rec['pr_M']:.2f}, PR(PMP) {rec['pr_PMP']:.2f}, "
          f"PR(S) {rec['pr_S']:.2f}, PR(PSP) {rec['pr_PSP']:.2f}; trM/n {rec['trM']/n:.3f}", flush=True)
    for name, d in (('single N_surv', avg(rec['nsurv']['single'])), ('Jhat N_surv', avg(rec['nsurv']['single_hat'])),
                    ('sim N_surv', avg(rec['nsurv']['sim'])), ('single k*_F', avg(rec['kF']['single'])),
                    ('Jhat k*_F', avg(rec['kF']['single_hat'])), ('sim k*_F', avg(rec['kF']['sim'])),
                    ('M k*_F', rec['kF']['M']), ('PMP k*_F', rec['kF']['PMP']), ('S k*_F', rec['kF']['S']),
                    ('PSP k*_F', rec['kF']['PSP'])):
        print(f"   {name:14s} " + " ".join(f"d={k}: {v:6.1f}" for k, v in d.items()), flush=True)
    ks = [k for k in (1, 3, 7, 15, 31, 63, 127) if k < min(n, 129)]
    print("   log(s_k/s_1) k=" + ",".join(str(k + 1) for k in ks) + " single " + " ".join(f"{rec['logratio']['single'][k]:+.2f}" for k in ks)
          + " | Jhat " + " ".join(f"{rec['logratio']['single_hat'][k]:+.2f}" for k in ks)
          + " | sim " + " ".join(f"{rec['logratio']['sim'][k]:+.2f}" for k in ks), flush=True)
os.makedirs(A.out, exist_ok=True)
np.savez_compressed(f"{A.out}/n2mat_{n}_{L}_c{c}_s{A.seed}_sig{A.sigma:g}{A.tag}.npz", Sigma=Sigma.numpy(), mu=mu.numpy(),
                    **{f'M{Lp}': (Ms[Lp] / A.paths).numpy() for Lp in Lps})
fn = f"{A.out}/n2_{n}_{L}_c{c}_s{A.seed}_sig{A.sigma:g}{A.tag}.json"
json.dump(res, open(fn, 'w'))
print('wrote', fn, f'{time.time()-t0:.0f}s', flush=True)
