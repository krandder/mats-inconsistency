"""Amendment-1 variant (cuts 0..8; projected meet with S-directions from sampled Jacobians and from deterministic
probes; at c = 0 the dropped input directions are set to 0). Oracle structure study (dev nets only; sampling is the oracle here, nothing below is an estimator).
usage: python oracle_structure.py SEED SIGMA OUTJSON [LOG2P=16]
For cuts c in CUTS, with P paths of the noisy net (common random numbers throughout):
 A. projected meet: replace h_c by Ph_c + (I-P)E[h_c | P h_c] (linear regression, with intercept), P = projection
    onto the top-k eigenvectors of M_c = E J^T J (J = d a_L / d h_c along the path) or of S_c = C^1/2 M C^1/2,
    run the rest of the net with the same noise, and report nMSE of the change of the mean, k = 1..8.
 B. downstream-distinct classes at c: distinct noiseless future gate patterns (layers c+1..c+D, D = 8, 16, 32, all)
    of the noiseless states, against distinct upstream histories (layers 1..c); counts covering 90/99% of the
    radius-weighted mass |a_L| (noiseless).
 C. linearity: R^2 of the best homogeneous linear map h_c -> a_L, and the nMSE of the "linear meet" Lambda E[h_c].
 D. rank collapse: per noiseless path, first layer with exactly one active unit, first all-off layer; share of the
    noiseless E|a_L| carried by paths whose first collapse is at layer <= c.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
import nd

torch.set_num_threads(1)
SEED, SIG, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
LOG2P = int(sys.argv[4]) if len(sys.argv) > 4 else 16
N, L = 8, 256
P = 1 << LOG2P
DT = nd.DT
CUTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
t0 = time.time()
net = nd.Net('plain', SEED, N, L, SIG)
g = torch.Generator().manual_seed(41_000_007 + SEED)
x = torch.randn(P, N, generator=g, dtype=DT)


def EPSf(l):
    """common noise of layer l (0-based) for every continuation, regenerated from a per-layer seed"""
    gg = torch.Generator().manual_seed(43_000_019 + 1009 * SEED + l)
    return torch.randn(P, N, generator=gg, dtype=DT)


def run(h, l_from, noise=True):
    """layers l_from+1..L from state h (after noise of layer l_from); returns a_L and gates (L-l_from, P, N)."""
    gates = []
    for l in range(l_from, L):
        z = h @ net.W[l].T
        a = torch.relu(z)
        gates.append(z > 0)
        if l < L - 1:
            h = a + (EPSf(l) * net.s[l] if (noise and SIG > 0) else 0.0)
        else:
            h = a
    return h, gates


# forward with noise, keep states at cuts and gates
h = x; Hc = {0: x.clone()}; G = []
for l in range(L):
    z = h @ net.W[l].T
    a = torch.relu(z); G.append(z > 0)
    h = a + EPSf(l) * net.s[l] if (l < L - 1 and SIG > 0) else a
    if (l + 1) in CUTS:
        Hc[l + 1] = h.clone()
out = h
mean = out.mean(0); var = ((out - mean) ** 2).mean(0); trv = float(var.sum())
res = dict(seed=SEED, sigma=SIG, P=P, trvar=trv, cuts={})
# Jacobians J_c = d a_L / d h_c by a backward sweep (normalised per path to avoid underflow; M uses the true scale
# through log-norms)
A = torch.eye(N, dtype=DT).expand(P, N, N).clone()
logs = torch.zeros(P, dtype=DT)
Jc = {}
for l in range(L - 1, -1, -1):
    D = G[l].to(DT)
    A = A * D[:, None, :]                       # A D_l  (columns masked)
    A = A @ net.W[l]                            # (A D_l) W_l
    nrm = A.flatten(1).norm(dim=1).clamp_min(1e-300)
    A = A / nrm[:, None, None]; logs = logs + torch.log(nrm)
    # after processing layer l (1-based l+1), A = d a_L / d h_l (state after layer l's noise, 0-based l)
    if l in CUTS:
        Jc[l] = (A.clone(), logs.clone())
for c in CUTS:
    if c not in Jc:
        continue
    J, lg = Jc[c]
    hc = Hc[c]
    w = torch.exp(2 * (lg - lg.max()))           # relative J scale^2
    M = (w[:, None, None] * (J.transpose(1, 2) @ J)).mean(0)
    M = M / M.trace()
    mc = hc.mean(0); C = torch.cov(hc.T)
    ev, U = torch.linalg.eigh(0.5 * (C + C.T)); Ch = U @ torch.diag(ev.clamp_min(0).sqrt()) @ U.T
    S = Ch @ M @ Ch; S = S / S.trace()
    eM, VM = torch.linalg.eigh(M); eM, VM = eM.flip(0), VM.flip(1)
    eS, VS = torch.linalg.eigh(S); eS, VS = eS.flip(0), VS.flip(1)
    # S eigvecs live in whitened coords; map back to state coords: directions Ch^-1 v (span of relevant state dirs)
    row = dict(kM=[nd.kstar_F(eM.numpy(), e) for e in (1e-1, 1e-2, 1e-3)], kS=[nd.kstar_F(eS.numpy(), e) for e in (1e-1, 1e-2, 1e-3)],
               eigM=eM.tolist(), eigS=eS.tolist())
    # A. projected meet
    pm = {}
    # sample-free probe directions: Jacobians (noiseless, c -> L) along deterministic probes
    if c == 0:
        import itertools as _it
        Ei = torch.eye(N, dtype=DT)
        prb = [Ei[i] * sg for i in range(N) for sg in (1, -1)]
        prb += [(Ei[i] + sg * Ei[j]) / math.sqrt(2) * tt for i, j in _it.combinations(range(N), 2) for sg in (1, -1) for tt in (1, -1)]
        prb = torch.stack(prb)
    else:
        evp, Up = torch.linalg.eigh(0.5 * (C + C.T)); Sp = Up * evp.clamp_min(0).sqrt()
        prb = torch.cat([mc[None]] + [torch.stack([mc + math.sqrt(N) * Sp[:, i], mc - math.sqrt(N) * Sp[:, i]]) for i in range(N)])
    Mp = torch.zeros(N, N, dtype=DT)
    for pv in prb:
        hh_ = pv.clone(); Jp = torch.eye(N, dtype=DT)
        for l in range(c, L):
            zz = net.W[l] @ hh_; Dd = (zz > 0).to(DT); hh_ = torch.relu(zz); Jp = Dd[:, None] * (net.W[l] @ Jp)
            nj = Jp.norm()
            if nj > 0: Jp = Jp / nj
        Mp += Jp.T @ Jp
    Mp = Mp / Mp.trace().clamp_min(1e-300)
    Sp_ = Ch @ Mp @ Ch; eSp, VSp = torch.linalg.eigh(0.5 * (Sp_ + Sp_.T)); eSp, VSp = eSp.flip(0), VSp.flip(1)
    VP = torch.linalg.qr(Ch @ VSp)[0]
    row['kP'] = [nd.kstar_F(eSp.numpy(), e) for e in (1e-1, 1e-2, 1e-3)]
    for tag, V in (('P', VP), ('S', None)):
        if tag == 'S':
            Chi = U @ torch.diag(1.0 / ev.clamp_min(1e-30 * ev.max()).sqrt()) @ U.T
            V = torch.linalg.qr(Chi @ VS)[0] if False else None
            # relevant state directions for S: v_state = Ch v (columns), orthonormalised in order
            Vst = Ch @ VS
            V = torch.linalg.qr(Vst)[0]
        errs = []
        for k in range(1, N + 1):
            Uk = V[:, :k]
            y = hc @ Uk                                       # kept coordinates
            if k < N and c == 0:
                ht = y @ Uk.T                                 # input: dropped directions at their conditional mean 0
            elif k < N:
                perp = hc - y @ Uk.T                          # dropped part
                X = torch.cat([y, torch.ones(P, 1, dtype=DT)], 1)
                B = torch.linalg.lstsq(X, perp).solution
                ht = y @ Uk.T + X @ B
            else:
                ht = hc
            o2, _ = run(ht, c)
            d = (o2 - out).mean(0)
            se = ((o2 - out) - d).pow(2).mean(0).sum() / P
            errs.append(dict(k=k, nmse=float(d.pow(2).sum() / trv), se_nmse=float(se / trv)))
        pm[tag] = errs
    row['projmeet'] = pm
    # C. linearity: homogeneous linear map h_c -> a_L
    Lam = torch.linalg.lstsq(hc, out).solution.T          # a_L ~ Lam h_c
    pred = hc @ Lam.T
    r2 = 1 - float(((out - pred) ** 2).sum() / ((out - mean) ** 2).sum())
    lin_meet = Lam @ mc
    row['lin_r2'] = r2
    row['lin_meet_nmse'] = float((lin_meet - mean).pow(2).sum() / trv)
    res['cuts'][c] = row
    print(f"c={c}: kM={row['kM']} kS={row['kS']} R2={r2:.4f} linmeet={row['lin_meet_nmse']:.3g} "
          f"projP(k=1..8)={[f"{e['nmse']:.2g}" for e in pm['P']]} projS(k=1..8)={[f"{e['nmse']:.2g}" for e in pm['S']]} "
          f"({time.time()-t0:.0f}s)", flush=True)
# B and D on the noiseless net
net0 = nd.Net('plain', SEED, N, L, 0.0)
h = x; codes = []
first1 = torch.full((P,), -1, dtype=torch.long); first0 = torch.full((P,), -1, dtype=torch.long)
pw = torch.tensor([1 << i for i in range(N)], dtype=torch.long)
for l in range(L):
    z = h @ net0.W[l].T
    gt = z > 0
    codes.append((gt.long() * pw).sum(1))
    m = gt.sum(1)
    first1 = torch.where((first1 < 0) & (m == 1), torch.full_like(first1, l + 1), first1)
    first0 = torch.where((first0 < 0) & (m == 0), torch.full_like(first0, l + 1), first0)
    h = torch.relu(z)
w0 = h.norm(dim=1)                                    # noiseless |a_L| as the mass weight
codes = torch.stack(codes, 1)                         # (P, L)
MOD = (1 << 61) - 1


def hash_rows(cols):
    k = torch.zeros(P, dtype=torch.long)
    for j in range(cols.shape[1]):
        k = (k * 1_000_003 + cols[:, j] * 7_919 + j) % MOD
    return k


def count(keys, wt):
    u, inv, cnt = torch.unique(keys, return_inverse=True, return_counts=True)
    ws = torch.zeros(len(u), dtype=DT).index_add_(0, inv, wt)
    ws = ws.sort(descending=True).values
    cum = torch.cumsum(ws, 0) / ws.sum().clamp_min(1e-300)
    return dict(distinct=int(len(u)), singletons=int((cnt == 1).sum()),
                w90=int((cum < 0.9).sum()) + 1, w99=int((cum < 0.99).sum()) + 1)


cls = {}
for c in [1, 2, 4, 8, 12, 16, 24, 32]:
    up = count(hash_rows(codes[:, :c]), w0)
    dn = {}
    for D in (8, 16, 32, L):
        hi = min(L, c + D)
        dn[str(D)] = count(hash_rows(codes[:, c:hi]), w0)
    cls[c] = dict(up=up, down=dn)
    print(f"classes c={c}: upstream {up} downstream(8/16/32/all) " +
          ' '.join(f"{D}:{v['distinct']}/{v['w99']}" for D, v in dn.items()), flush=True)
res['classes'] = cls
f1 = first1.numpy(); f0 = first0.numpy(); wn = w0.numpy()
res['collapse'] = dict(
    frac_first1_by=[float(((f1 > 0) & (f1 <= c)).mean()) for c in CUTS],
    mass_first1_by=[float(wn[(f1 > 0) & (f1 <= c)].sum() / max(wn.sum(), 1e-300)) for c in CUTS],
    frac_dead_by=[float(((f0 > 0) & (f0 <= c)).mean()) for c in CUTS],
    frac_never1=float((f1 < 0).mean()), mass_never1=float(wn[f1 < 0].sum() / max(wn.sum(), 1e-300)))
print('collapse', res['collapse'], flush=True)
res['sec'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
