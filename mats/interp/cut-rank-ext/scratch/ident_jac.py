"""NC3 (PREREG.md): differential rank of theta_f -> X(f) = (M(f, b_k)_j)_{k, j} (identifiability conjecture I1).

usage: ident_jac.py n c d [--K 64] [--N 65536] [--fronts 3]
M_N(theta_f, b) = E|x| mean over N unit directions of g_b(h_c(u)); Jacobian by exact backpropagation (float64) for two
independent direction sets A, B. Known symmetries: input rotations dW_1 = W_1 A and hidden-unit rescalings.
Restricted spectrum: singular values of ((J_A + J_B)/2) Q, Q = orthonormal complement of the symmetry directions;
noise nu = ||(J_A - J_B) Q||_op / 2. Writes runs/nc3_n{n}_c{c}_d{d}.json and prints a summary.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cut-rank/lib')
import crnet as cn

ap = argparse.ArgumentParser()
for k in ('n', 'c', 'd'):
    ap.add_argument(k, type=int)
ap.add_argument('--K', type=int, default=64)
ap.add_argument('--N', type=int, default=65536)
ap.add_argument('--fronts', type=int, default=3)
a = ap.parse_args()
n, c, d, K, N = a.n, a.c, a.d, a.K, a.N
tag = cn.cfg_tag(n, c, d, 'nc3')
Er = cn.mean_radius(n)
pf = c * n * n
backs = [[W.astype(np.float64) for W in cn.back_weights(tag, n, d, b)] for b in range(K)]
HERE = os.path.dirname(os.path.abspath(__file__))


def front_pass(U, Wf):
    hs, Ds = [U], []
    h = U
    for W in Wf:
        z = h @ W.T
        D = (z > 0).astype(np.float64)
        h = z * D
        hs.append(h); Ds.append(D)
    return hs, Ds


def jacobian(U, Wf):
    """J (K n, pf): d M_N / d theta_f, theta_f = concat of W_1..W_c (row-major)."""
    hs, Ds = front_pass(U, Wf)
    hc = hs[-1]
    Jrows = np.zeros((K, n, pf))
    for k in range(K):
        Wb = backs[k]
        h = hc; G = None                                  # G (N, n_out, n): d a_L / d h_c per sample
        for W in Wb:
            z = h @ W.T
            Dl = (z > 0).astype(np.float64)
            if G is None:
                G = Dl[:, :, None] * W[None, :, :]
            else:
                G = Dl[:, :, None] * np.einsum('ab,sbc->sac', W, G)
            h = z * Dl
        delta = G * Ds[-1][:, None, :]                    # d a / d z_c  (N, j, unit)
        grads = [None] * c
        for l in range(c - 1, -1, -1):                    # layer l+1 in 1-based terms
            grads[l] = np.einsum('sjk,sm->jkm', delta, hs[l]) / len(U) * Er
            if l > 0:
                delta = np.einsum('sjk,km->sjm', delta, Wf[l]) * Ds[l - 1][:, None, :]
        Jrows[k] = np.concatenate([g.reshape(n, -1) for g in grads], axis=1)
    return Jrows.reshape(K * n, pf), [h for h in hs[1:]]


def symmetry_dirs(Wf):
    rot, resc = [], []
    for p in range(n):
        for q in range(p + 1, n):
            A = np.zeros((n, n)); A[p, q] = 1; A[q, p] = -1
            v = [np.zeros((n, n)) for _ in range(c)]
            v[0] = Wf[0] @ A
            rot.append(np.concatenate([x.ravel() for x in v]))
    for l in range(c - 1):
        for i in range(n):
            v = [np.zeros((n, n)) for _ in range(c)]
            E = np.zeros((n, n)); E[i, i] = 1
            v[l] = E @ Wf[l]
            v[l + 1] = -Wf[l + 1] @ E
            resc.append(np.concatenate([x.ravel() for x in v]))
    return np.array(rot).T, (np.array(resc).T if resc else np.zeros((pf, 0)))


t0 = time.time()
res = dict(n=n, c=c, d=d, K=K, N=N, pf=pf, n_rot=n * (n - 1) // 2, n_resc=n * (c - 1), fronts=[], rejected=[])
f = 0
while len(res['fronts']) < a.fronts and f < 200:
    Wf = [W.astype(np.float64) for W in cn.front_weights(tag, n, c, f)]
    UA = cn.directions(n, N, tag, 31, f, 0).astype(np.float64)
    UB = cn.directions(n, N, tag, 31, f, 1).astype(np.float64)
    JA, hsA = jacobian(UA, Wf)
    dead = [int((h.max(0) <= 0).sum()) for h in hsA]
    if sum(dead) > 0:
        res['rejected'].append(dict(front=f, dead_per_layer=dead)); f += 1
        continue
    JB, _ = jacobian(UB, Wf)
    Srot, Sresc = symmetry_dirs(Wf)
    S = np.concatenate([Srot, Sresc], axis=1)
    rankS = int(np.linalg.matrix_rank(S))
    Us, ss, _ = np.linalg.svd(S, full_matrices=True)
    Q = Us[:, rankS:]                                     # orthonormal complement
    Jm = 0.5 * (JA + JB)
    nu = float(np.linalg.norm((JA - JB) @ Q, 2) / 2)
    sv = np.linalg.svd(Jm @ Q, compute_uv=False)
    top = float(np.linalg.svd(Jm, compute_uv=False)[0])
    qr, _ = np.linalg.qr(Srot); sv_rot = np.linalg.svd(Jm @ qr, compute_uv=False)
    if Sresc.shape[1]:
        qs, _ = np.linalg.qr(Sresc); sv_resc = np.linalg.svd(Jm @ qs, compute_uv=False)
    else:
        sv_resc = np.zeros(0)
    fr = dict(front=f, rank_sym=rankS, m=int(Q.shape[1]), nu=nu, top=top, sv_min=float(sv.min()), sv_max=float(sv.max()),
              n_above_10nu=int((sv >= 10 * nu).sum()), sv_min_over_nu=float(sv.min() / nu),
              rot_max_over_nu=float(sv_rot.max() / nu), resc_max_over_top=float(sv_resc.max() / top) if len(sv_resc) else 0.0,
              sv_restricted=[float(x) for x in sv])
    res['fronts'].append(fr)
    print(f"front {f}: m={fr['m']} (rank_sym {rankS}) top {top:.3g} nu {nu:.3g} | restricted sv min {fr['sv_min']:.3g} "
          f"(min/nu {fr['sv_min_over_nu']:.1f}; {fr['n_above_10nu']}/{fr['m']} >= 10 nu) | rot max/nu {fr['rot_max_over_nu']:.2f} "
          f"| resc max/top {fr['resc_max_over_top']:.2e} | {time.time() - t0:.0f}s", flush=True)
    f += 1
res['P7_pass'] = all(fr['n_above_10nu'] == fr['m'] for fr in res['fronts'])
res['P8_pass'] = all(fr['resc_max_over_top'] <= 1e-8 and fr['rot_max_over_nu'] <= 3 for fr in res['fronts'])
res['secs'] = round(time.time() - t0)
with open(os.path.join(HERE, '..', 'runs', f'nc3_n{n}_c{c}_d{d}.json'), 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'fronts'}), flush=True)
