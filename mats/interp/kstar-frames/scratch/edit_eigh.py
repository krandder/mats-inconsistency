"""One-off source edit (2026-10-31 ~00:40Z): on CUDA, batched SVD of the rank-deficient masked Jacobians is ~170x slower
than eigh of J^T J (benchmark on the L4: 7.9 s vs 46 ms per batch of 64 at n = 128; cusolver Jacobi fails to converge
and falls back). On CUDA use eigh for right frames, singular values, left frames and the pseudo-inverse of the dual
transport; on CPU keep LAPACK SVD. Run once from the lane directory: python3 scratch/edit_eigh.py"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def sub(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:70], s.count(old))
    open(path, 'w').write(s.replace(old, new))


p = 'lib/kf.py'
sub(p, '''def svd_right(J):
    """Batched SVD; returns s (P, n) descending and V (P, n, n) with right singular vectors as columns."""
    U, s, Vh = torch.linalg.svd(J)
    return s, Vh.transpose(-1, -2)
''', '''def svd_right(J, use_eigh=None):
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
''')
sub(p, '''        B = D[l - 1][:, :, None] * net.W[l - 1][None] * D[l - 2][:, None, :]
        G = orth(torch.linalg.pinv(B).transpose(-1, -2) @ G)''', '''        B = D[l - 1][:, :, None] * net.W[l - 1][None] * D[l - 2][:, None, :]
        G = orth(pinvT_apply(B, G) if (B.is_cuda if use_eigh is None else use_eigh)
                 else torch.linalg.pinv(B).transpose(-1, -2) @ G)''')
sub(p, "def dual_forward(net, F, D, c_from, c_to):", "def dual_forward(net, F, D, c_from, c_to, use_eigh=None):")
p = 'scratch/r1_layers.py'
sub(p, "        s = torch.linalg.svdvals(Js[c])", "        s = kf.svdvals(Js[c])")
sub(p, """        U, _, _ = torch.linalg.svd(D[c - 1][:, :, None] * Ps[c])
        frames['blv'] = kf.orth(U[:, :, :k])""", """        frames['blv'] = kf.orth(kf.left_frame(D[c - 1][:, :, None] * Ps[c], k))""")
print('edits done')
