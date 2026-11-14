"""Per-layer gate entropies of a noisy network for one question (gate-entropy, 2026-10-29).

For each analysed layer l:
  E1  H(G_l), all gates: plug-in, Miller-Madow, NSB, collision (Renyi-2, lower bound), independent and MADE/FVSBN
      cross-entropy upper bounds, on N_ent fresh samples; noise-only part H(G_l | X) on replicated inputs.
  E2  flip rates f_i (fraction of held-out samples whose answer changes when gate i is flipped; actual downstream
      network, common random numbers); D(tau) = #{i : f_i >= tau}; H(G_R) with the same estimator panel.
  E3  merged entropy: greedy decision tree over gate values, leaves = merged classes, each leaf replaces the gates of
      its members by the leaf's majority pattern; built on the flip set from single-flip 'care' indicators; every
      checkpoint verified on held-out samples by actual substitution; H(leaves) on held-out samples.
  Subspace: PCA of the relevant gates' pre-activations on the data: components for 99% variance and for 99% / 99.9%
      mean gate-sign agreement; log2 of the number of cells of D hyperplanes in general position in k dimensions.
  Capacity: C_l = (d/2) log2(1 + 1/sigma^2) and the covariance bound (1/2) log2 det(I + Cov(signal)/noise var).
"""
import math, time, json, sys
import numpy as np
import torch
import ent

torch.set_num_threads(1)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------------------------- flips
def flip_matrix(net, st, l, idx, chunk_rows=1 << 15):
    """(len(idx), m) bool: answer changed when gate i alone is flipped; plus base answers via the same code path."""
    G = torch.from_numpy(net.gates(st, l))
    idx = np.asarray(idx); n = len(idx); m = G.shape[1]
    base = net.sub(st, l, idx, G[idx])
    F = np.zeros((n, m), dtype=bool)
    per = max(1, chunk_rows // m)
    eye = torch.eye(m, dtype=torch.bool)
    for i in range(0, n, per):
        ii = idx[i:i + per]; k = len(ii)
        Gr = G[ii][:, None, :] ^ eye[None]                         # (k, m, m)
        rep = np.repeat(ii, m)
        a1 = net.sub(st, l, rep, Gr.reshape(k * m, m))
        b0 = base[i:i + k]
        b0r = b0.repeat_interleave(m, 0) if b0.dim() > 1 else b0.repeat_interleave(m)
        F[i:i + k] = net.changed(b0r, a1).reshape(k, m).numpy()
    return F, base


# ----------------------------------------------------------------------------------------------- merging tree
class Tree:
    def __init__(self, m):
        self.gate = [-1]; self.child = [[-1, -1]]; self.rep = [None]; self.m = m

    def assign(self, G):
        node = np.zeros(len(G), dtype=np.int64)
        gate = np.array(self.gate); child = np.array(self.child)
        while True:
            internal = gate[node] >= 0
            if not internal.any(): break
            ii = np.nonzero(internal)[0]
            node[ii] = child[node[ii], G[ii, gate[node[ii]]].astype(np.int64)]
        return node


def build_tree_path(G, C, max_splits=4000, min_leaf=2, checkpoints=None, on_checkpoint=None, w=None):
    """Greedy rate-distortion tree. G, C (n, m) bool. Phase 1 (hard): a sample is 'violated' in a leaf if some gate that
    matters for it (C, single-flip care) disagrees with the leaf representative (majority); apply the split with the
    largest drop in violated samples per bit of leaf entropy added. Phase 2 (soft, Amendment 1), used once no leaf has
    hard violations: the drop in flip-rate-weighted mismatches sum_x sum_j w_j [G_xj != rep_j], w_j = training flip
    rate. Every checkpoint is verified elsewhere on held-out samples."""
    n, m = G.shape
    Gf = G.astype(np.float32); A = (C & G).astype(np.float32); B = (C & ~G).astype(np.float32)
    wv = (C.mean(0) if w is None else np.asarray(w)).astype(np.float64)
    tree = Tree(m)
    def rep_of(S):
        return (Gf[S].mean(0) > 0.5) if len(S) else np.zeros(m, bool)
    def viol(S, rep):
        if len(S) == 0: return 0
        v = A[S] @ (1 - rep.astype(np.float32)) + B[S] @ rep.astype(np.float32)
        return int((v > 0).sum())
    def best_split(S):
        S = np.asarray(S); k = len(S)
        if k < 2 * min_leaf: return (None, None)
        rep = rep_of(S); v0 = viol(S, rep)
        GS = Gf[S]; n1 = GS.sum(0); n0 = k - n1
        ok = (n1 >= min_leaf) & (n0 >= min_leaf)
        if not ok.any(): return (None, None)
        co = GS.T @ GS                                              # (s, j)
        col = GS.sum(0)
        rep1 = (co / np.maximum(n1[:, None], 1)) > 0.5               # (s, j) majority in child g_s = 1
        c0 = col[None, :] - co
        rep0 = (c0 / np.maximum(n0[:, None], 1)) > 0.5
        p1 = n1 / k
        dH = (k / n) * ent.h2(p1)
        hard = None
        if v0 > 0:
            AS, BS = A[S], B[S]
            v1 = (AS @ (1 - rep1.T.astype(np.float32)) + BS @ rep1.T.astype(np.float32)) > 0    # (k, s)
            v0m = (AS @ (1 - rep0.T.astype(np.float32)) + BS @ rep0.T.astype(np.float32)) > 0
            after = (GS * v1).sum(0) + ((1 - GS) * v0m).sum(0)
            gain = v0 - after
            score = np.where(ok & (gain > 0), gain / np.maximum(dH, 1e-9), -np.inf)
            s = int(np.argmax(score))
            if np.isfinite(score[s]): hard = (float(score[s]), s, int(gain[s]))
        mb = float((wv * np.where(rep, k - col, col)).sum())
        m1 = (wv[None, :] * np.where(rep1, n1[:, None] - co, co)).sum(1)
        m0 = (wv[None, :] * np.where(rep0, n0[:, None] - c0, c0)).sum(1)
        gs = mb - m1 - m0
        sc = np.where(ok & (gs > 1e-9), gs / np.maximum(dH, 1e-9), -np.inf)
        s2 = int(np.argmax(sc))
        soft = (float(sc[s2]), s2, float(gs[s2])) if np.isfinite(sc[s2]) else None
        return (hard, soft)
    leaves = {0: np.arange(n)}
    tree.rep[0] = rep_of(leaves[0])
    cand = {0: best_split(leaves[0])}
    splits = 0
    tree.phase = []
    checkpoints = sorted(set(checkpoints or []))
    if on_checkpoint and 0 in checkpoints: on_checkpoint(tree, 0)
    while splits < max_splits:
        best = None
        for k, c in cand.items():
            if c[0] is not None and (best is None or c[0][0] > best[1][0]): best = (k, c[0])
        phase = 'hard'
        if best is None:
            phase = 'soft'
            for k, c in cand.items():
                if c[1] is not None and (best is None or c[1][0] > best[1][0]): best = (k, c[1])
        if best is None: break
        node, (score, s, gain) = best
        S = leaves.pop(node); cand.pop(node)
        tree.gate[node] = s
        tree.phase.append(phase)
        for val in (0, 1):
            Sc = S[G[S, s] == bool(val)]
            nid = len(tree.gate); tree.gate.append(-1); tree.child.append([-1, -1]); tree.rep.append(rep_of(Sc))
            tree.child[node][val] = nid
            leaves[nid] = Sc; cand[nid] = best_split(Sc)
        splits += 1
        if on_checkpoint and splits in checkpoints: on_checkpoint(tree, splits)
    if on_checkpoint and splits not in checkpoints: on_checkpoint(tree, splits)
    return tree, splits


def merged_entropy(net, st_tr, st_ho, l, F_tr, eps_list, max_splits=4000, log_every=True):
    G_tr = net.gates(st_tr, l); G_ho = net.gates(st_ho, l)
    n_ho = len(G_ho); idx_ho = np.arange(n_ho)
    Gt_ho = torch.from_numpy(G_ho)
    base_ho = net.sub(st_ho, l, idx_ho, Gt_ho)
    rows = []
    def on_cp(tree, splits):
        leaf = tree.assign(G_ho)
        reps = np.stack([tree.rep[k] if tree.rep[k] is not None else np.zeros(tree.m, bool) for k in range(len(tree.rep))])
        Gn = torch.from_numpy(reps[leaf])
        a1 = net.sub(st_ho, l, idx_ho, Gn)
        ch = net.changed(base_ho, a1).numpy()
        c = np.bincount(leaf); c = c[c > 0]
        row = dict(splits=splits, leaves=splits + 1, H_leaves=ent.plugin(c), H_leaves_mm=ent.miller_madow(c),
                   change_rate=float(ch.mean()), change_se=float(math.sqrt(max(ch.mean() * (1 - ch.mean()), 1e-12) / n_ho)),
                   gates_changed_mean=float((Gn.numpy() != G_ho).sum(1).mean()), frac_any_gate_changed=float((Gn.numpy() != G_ho).any(1).mean()),
                   change_given_changed=float(ch[(Gn.numpy() != G_ho).any(1)].mean()) if (Gn.numpy() != G_ho).any(1).any() else None, hard_splits=int(sum(p == 'hard' for p in getattr(tree, 'phase', []))))
        if hasattr(net, 'bias_nmse'):
            row['bias_nmse'] = net.bias_nmse(base_ho, a1, n_ho)
        if hasattr(net, 'bias_abs'):
            row['bias_abs'] = net.bias_abs(base_ho, a1, n_ho)
        rows.append(row)
        if log_every: log(f'  tree l={l} splits={splits} H={row["H_leaves"]:.2f} change={row["change_rate"]:.4f}')
    cps = sorted(set([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]))
    cps = [c for c in cps if c <= max_splits]
    t0 = time.time()
    build_tree_path(G_tr, F_tr, max_splits=max_splits, checkpoints=cps, on_checkpoint=on_cp)
    res = dict(path=rows, seconds=time.time() - t0)
    for eps in eps_list:
        ok = [r for r in rows if r['change_rate'] <= eps]
        res[f'H_merged_eps{eps:g}'] = (min(r['H_leaves'] for r in ok) if ok else None)
        res[f'leaves_eps{eps:g}'] = (min(r['leaves'] for r in ok) if ok else None)
    for thr in (1e-3, 1e-4, 1e-5):
        ok = [r for r in rows if r.get('bias_nmse') is not None and r['bias_nmse'] <= thr]
        if rows and rows[0].get('bias_nmse') is not None:
            res[f'H_merged_bias{thr:g}'] = (min(r['H_leaves'] for r in ok) if ok else None)
    res['final_change_rate'] = rows[-1]['change_rate']; res['final_H'] = rows[-1]['H_leaves']
    return res


# ----------------------------------------------------------------------------------------------- subspace
def subspace(Z, G, R_cols_neuron, agree_levels=(0.99, 0.999)):
    """Z (n, k_all) pre-activations; columns R (neurons) and their gate thresholds handled via G sign agreement of
    the reconstructed pre-activations. Returns k for 99% variance and for mean sign agreement."""
    if len(R_cols_neuron) == 0:
        return dict(D_neurons=0)
    X = Z[:, R_cols_neuron].astype(np.float64)
    mu = X.mean(0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2; cum = np.cumsum(var) / var.sum()
    k99 = int(np.searchsorted(cum, 0.99) + 1); k999 = int(np.searchsorted(cum, 0.999) + 1)
    pr = float(var.sum() ** 2 / (var ** 2).sum())
    out = dict(D_neurons=len(R_cols_neuron), k_var99=k99, k_var999=k999, participation=pr)
    return out, (mu, Vt)


def gate_agreement_k(Z, R_neur, thresholds_fn, mu, Vt, levels=(0.99, 0.999), kmax=None):
    X = Z[:, R_neur].astype(np.float64)
    Gtrue = thresholds_fn(X)
    D = X.shape[1]; kmax = kmax or D
    res = {}
    grid = list(range(0, 33)) + [40, 48, 56, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024]
    ks = [k for k in grid if k <= min(kmax, Vt.shape[0])]
    agree = []
    for k in ks:
        Xr = mu + ((X - mu) @ Vt[:k].T) @ Vt[:k] if k > 0 else np.broadcast_to(mu, X.shape)
        agree.append(float((thresholds_fn(Xr) == Gtrue).mean()))
    for lv in levels:
        kk = [k for k, a in zip(ks, agree) if a >= lv]
        res[f'k_agree{lv:g}'] = kk[0] if kk else None
    res['agree_curve'] = agree[:65]
    return res


def log2_cells(D, k):
    """log2 of the number of cells of D affine hyperplanes in general position in R^k: sum_{j<=k} C(D, j)."""
    k = min(k, D)
    return float(math.log2(sum(math.comb(D, j) for j in range(k + 1))))


# ----------------------------------------------------------------------------------------------- capacity bounds
def cov_capacity(signal, noise_sd):
    """(1/2) log2 det(I + Cov(signal) / noise_sd^2), noise_sd scalar or per-coordinate vector."""
    if signal is None: return None
    S = signal.double()
    S = S - S.mean(0)
    C = (S.T @ S) / (len(S) - 1)
    ns = torch.as_tensor(noise_sd, dtype=torch.float64)
    if ns.dim() == 0: ns = ns.expand(C.shape[0])
    keep = ns > 0                                   # dead units (zero noise scale) carry no signal either
    C = C[keep][:, keep]; ns = ns[keep]
    Cw = C / ns[:, None] / ns[None, :]
    ev = torch.linalg.eigvalsh(Cw).clamp_min(0)
    return float(0.5 * torch.log2(1 + ev).sum())
