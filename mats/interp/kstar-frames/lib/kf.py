"""kstar-frames core: noisy forward passes, downstream Jacobians and their right frames at every layer, past products,
frame transports, and scores (PREREG.md). float64 torch for the net, numpy for small per-path linear algebra.

Conventions (layers 1..L, python index l-1):
  z_l = W_l h_{l-1}, a_l = ReLU(z_l), h_l = a_l + s_l eps_l (l < L), output a_L.  D_l = 1[z_l > 0].
  J_c = d a_L / d z_c = D_L W_L D_{L-1} ... W_{c+1} D_c  (n x n, rows: output units, columns: z_c coordinates).
  Right frame F_c = top-k right singular vectors of J_c (columns), in range(D_c).
  Past product P_c = d z_c / d x = W_c D_{c-1} W_{c-1} ... D_1 W_1.
Scores of an orthonormal frame G (n x k) on a path with right singular vectors V (n x K, descending) and singular
values s (K) of J_c, and total energy E = |J_c|_F^2:
  O = |V[:, :k]^T G|_F^2 / k;  CS = sum_i s_i^2 |v_i^T G|^2 / sum_{i<=k} s_i^2;  T = 1 - sum_i s_i^2 |v_i^T G|^2 / E.
"""
import math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nd  # unchanged copy of work/narrow-deep/lib/nd.py

DT = torch.float64
DEV = torch.device(os.environ.get('KF_DEVICE', 'cpu'))   # 'cuda' on the GPU VM
SIGMA = 1e-3
SEEDS = (400, 401, 402)
WIDTHS = (32, 64, 128)
MAIN_LAYERS = {32: (8, 16, 24), 16: (4, 8, 12)}


def make_net(seed, n, L=32, sigma=SIGMA):
    """Depth-32 net from nd.py; depth 16 = its first 16 layers with the same noise scales."""
    net32 = nd.Net('plain', seed, n, 32, sigma)
    net = net32 if L == 32 else nd.Net('plain', seed, n, L, sigma, r=net32.r[:L].tolist())
    net.W = [w.to(DEV) for w in net.W]
    net.r, net.s = net.r.to(DEV), net.s.to(DEV)
    return net


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(int(seed))


def draw_x(P, n, g):
    return torch.randn(P, n, generator=g, dtype=DT, device=DEV)


@torch.no_grad()
def forward(net, x, g=None, noise=None, start=0, h0=None):
    """Noisy forward pass. Returns zs (list of L tensors (P, n): z_1..z_L) and eps (list of L-1 noise draws).
    noise: optional list of L-1 tensors to reuse (coupled paths). start/h0: begin at layer start+1 from h_start = h0
    (then zs[:start] and eps[:start] are None)."""
    L = net.L
    zs = [None] * L
    eps = [None] * (L - 1)
    h = x if h0 is None else h0
    for l in range(start, L):
        z = h @ net.W[l].T
        zs[l] = z
        a = torch.relu(z)
        if l < L - 1:
            if noise is not None and noise[l] is not None:
                e = noise[l]
            else:
                e = torch.randn(a.shape, generator=g, dtype=DT, device=DEV)
            eps[l] = e
            h = a + net.s[l] * e
    return zs, eps


def masks(zs):
    return [None if z is None else (z > 0).to(DT) for z in zs]


@torch.no_grad()
def downstream_jacobians(net, D, layers, lowest=None):
    """J_c for c in layers (1-based), from the masks D (list of (P, n) 0/1 tensors, D[l-1] = D_l).
    Returns dict c -> (P, n, n). Accumulates backward from J_L = diag(D_L)."""
    L = net.L
    want = set(layers)
    lowest = min(want) if lowest is None else lowest
    P, n = D[L - 1].shape
    J = torch.diag_embed(D[L - 1])                    # J_L
    out = {}
    if L in want:
        out[L] = J.clone()
    for c in range(L - 1, lowest - 1, -1):           # J_c = J_{c+1} W_{c+1} D_c ; W_{c+1} = net.W[c]
        J = (J @ net.W[c]) * D[c - 1][:, None, :]
        if c in want:
            out[c] = J.clone()
    return out


@torch.no_grad()
def past_products(net, D, layers):
    """P_c = d z_c / d x for c in layers. Returns dict c -> (P, n, n)."""
    want = set(layers)
    P, n = D[0].shape
    Pc = net.W[0].expand(P, n, n).clone()             # P_1 = W_1
    out = {}
    if 1 in want:
        out[1] = Pc.clone()
    for c in range(2, max(want) + 1):                 # P_c = W_c D_{c-1} P_{c-1}
        Pc = net.W[c - 1] @ (D[c - 2][:, :, None] * Pc)
        if c in want:
            out[c] = Pc.clone()
    return out


def svd_right(J, use_eigh=None):
    """Batched SVD; returns s (P, n) descending and V (P, n, n) with right singular vectors as columns.
    On CUDA (or use_eigh=True) from eigh of J^T J: singular values below ~1e-8 s_1 lose relative accuracy, which does
    not affect frames with s_k >= 0.01 s_1 or energies."""
    if J.is_cuda if use_eigh is None else use_eigh:
        ev, U = torch.linalg.eigh(J.transpose(-1, -2) @ J)
        return ev.clamp_min(0).sqrt().flip(-1), U.flip(-1)
    U, s, Vh = torch.linalg.svd(J)
    return s, Vh.transpose(-1, -2)


def svdvals(J, use_eigh=None):
    if J.is_cuda if use_eigh is None else use_eigh:
        return torch.linalg.eigvalsh(J.transpose(-1, -2) @ J).clamp_min(0).sqrt().flip(-1)
    return torch.linalg.svdvals(J)


def left_frame(A, k, use_eigh=None):
    """Top-k left singular vectors of A (P, n, n)."""
    if A.is_cuda if use_eigh is None else use_eigh:
        ev, U = torch.linalg.eigh(A @ A.transpose(-1, -2))
        return U.flip(-1)[:, :, :k]
    U, _, _ = torch.linalg.svd(A)
    return U[:, :, :k]


def pinvT_apply(B, G, tol=1e-12):
    """(B^T)^+ G = B (B^T B)^+ G, with the pseudo-inverse from eigh (eigenvalues below tol * max dropped: the masked
    zero rows and columns give exact zeros; genuine small singular values of the Gaussian block are far above)."""
    ev, U = torch.linalg.eigh(B.transpose(-1, -2) @ B)
    inv = torch.where(ev > tol * ev[..., -1:].clamp_min(1e-300), 1.0 / ev, torch.zeros_like(ev))
    return B @ (U @ (inv[..., :, None] * (U.transpose(-1, -2) @ G)))


def n_surv(s, delta=0.01):
    return (s >= delta * s[..., :1]).sum(-1)


def orth(A):
    """Batched QR orthonormalisation of the columns of A (P, n, k)."""
    Q, R = torch.linalg.qr(A)
    return Q


def scores(G, V, s, k, E=None):
    """G (P, n, kg) orthonormal; V (P, n, K) right singular vectors; s (P, K); k = true frame size.
    Returns O, CS, T as (P,) tensors. If E (P,) is None, E = sum s^2 over the given K."""
    proj = V.transpose(-1, -2) @ G                     # (P, K, kg): v_i^T G
    w = (proj ** 2).sum(-1)                           # (P, K): |v_i^T G|^2
    s2 = s ** 2
    cap = (s2 * w).sum(-1)
    top = s2[:, :k].sum(-1)
    if E is None:
        E = s2.sum(-1)
    O = w[:, :k].sum(-1) / k
    return O, cap / top, 1.0 - cap / E


def project_active(G, Dc):
    """orth(D_c G): restrict a frame to the path's active coordinates at layer c."""
    return orth(Dc[:, :, None] * G)


def push_forward(net, F, D, c_from, c_to):
    """Push-forward by the layers' own Jacobians: G <- orth(D_l W_l G) for l = c_from+1..c_to (QR each step)."""
    G = F
    for l in range(c_from + 1, c_to + 1):
        G = orth(D[l - 1][:, :, None] * (net.W[l - 1] @ G))
    return G


def dual_forward_loop(net, F, D, c_from, c_to):
    """Dual (inverse-transpose) transport, one path at a time (numpy): G <- orth(pinv(B_l)^T G), B_l = D_l W_l D_{l-1}
    restricted to the active rows (layer l) and columns (layer l-1), for l = c_from+1..c_to. If m_l < k the frame has
    rank m_l (its other columns are zero)."""
    dev = F.device
    G = F.detach().cpu().clone()
    P, n, k = G.shape
    for l in range(c_from + 1, c_to + 1):
        Wl = net.W[l - 1].cpu().numpy()
        newG = np.zeros((P, n, k))
        Dl = D[l - 1].cpu().numpy() > 0.5
        Dp = D[l - 2].cpu().numpy() > 0.5
        Gn = G.numpy()
        for p in range(P):
            ra, ca = np.where(Dl[p])[0], np.where(Dp[p])[0]
            Bs = Wl[np.ix_(ra, ca)]                   # m_l x m_{l-1}
            g = np.linalg.pinv(Bs).T @ Gn[p][ca]      # (m_l x m_{l-1}) @ (m_{l-1} x k)
            q, _ = np.linalg.qr(g)                    # (m_l, min(m_l, k)); if m_l < k the frame is rank m_l
            newG[p][np.ix_(ra, np.arange(q.shape[1]))] = q
        G = torch.from_numpy(newG)
    return G.to(dev)


def dual_forward(net, F, D, c_from, c_to, use_eigh=None):
    """Batched dual transport (same subspaces as dual_forward_loop): the pinv of the masked n x n layer map is the
    embedded pinv of its active block. Paths whose active count drops below k on the way use the loop."""
    k = F.shape[-1]
    small = torch.zeros(F.shape[0], dtype=torch.bool, device=F.device)
    for l in range(c_from, c_to + 1):
        small |= D[l - 1].sum(-1) < k
    G = F
    for l in range(c_from + 1, c_to + 1):
        B = D[l - 1][:, :, None] * net.W[l - 1][None] * D[l - 2][:, None, :]
        G = orth(pinvT_apply(B, G) if (B.is_cuda if use_eigh is None else use_eigh)
                 else torch.linalg.pinv(B).transpose(-1, -2) @ G)
    if bool(small.any()):
        idx = torch.where(small)[0]
        G = G.clone()
        G[idx] = dual_forward_loop(net, F[idx], [None if d is None else d[idx] for d in D], c_from, c_to)
    return G


def backward_one(net, F_next, D, c):
    """One backward step from the next layer's frame: orth(D_c W_{c+1}^T F_{c+1})."""
    return orth(D[c - 1][:, :, None] * (net.W[c].T @ F_next))


def random_frame(Dc, k, g):
    P, n = Dc.shape
    A = torch.randn(P, n, k, generator=g, dtype=DT, device=DEV)
    return project_active(A, Dc)


def angle(a, b):
    """Angle between rows of a and b (P, n)."""
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    return torch.arccos(cos.clamp(-1.0, 1.0))
