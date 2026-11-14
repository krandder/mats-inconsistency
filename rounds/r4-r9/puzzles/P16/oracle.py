"""Exact oracle: p(B, b, y) = sum_x p(x) p(B, b | x) p(y | x), and I(b; Y | B) in bits.

Torch, batched over episodes of the same (m, w, nx): fp64 on CPU for labels/tests, fp32 on GPU for FR-direct targets.
p(B | x) is built column by column (each column's table is indexed by its parents' values), so columns that are
functions of earlier listed columns (copies, XORs, gates) are exact. Leak candidates (functions of y) and Markov
noise (marginally Bern(.5), independent of everything) are handled separately.
"""
import numpy as np
import torch


def pack(eps, device='cpu', dtype=torch.float64):
    """Stack episodes (same m, w, nx) into tensors."""
    E = len(eps); m = eps[0].m; ncol = m + 1
    X = 2 ** eps[0].nx; Y = 2 ** eps[0].w
    P1 = np.full((E, ncol, X, 8), .5); par = np.full((E, ncol, 3), -1, np.int64)
    yd = np.full((E, Y), .5); isy = np.zeros(E, bool)
    for e, ep in enumerate(eps):
        assert ep.m == m and 2 ** ep.w == Y and 2 ** ep.nx == X
        for c, col in enumerate(ep.cols):
            if col.ydep is not None:
                assert c == m; yd[e] = col.ydep; isy[e] = True
            else:
                P1[e, c] = col.P1
                par[e, c, :len(col.par)] = col.par
    py = np.stack([ep.py for ep in eps])
    t = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=device)
    return dict(P1=t(P1), par=t(par, torch.long), py=t(py), yd=t(yd), isy=t(isy, torch.bool), m=m)


def _cfg(par_c, n):
    """par_c (E,3) parent indices (-1 none) -> (E, n) parent config for every joint config index s < n."""
    s = torch.arange(n, device=par_c.device)
    cfg = torch.zeros(par_c.shape[0], n, dtype=torch.long, device=par_c.device)
    for i in range(3):
        p = par_c[:, i]
        bit = (s[None] >> p.clamp(min=0)[:, None]) & 1
        cfg += (bit * (p[:, None] >= 0)) << i
    return cfg


def add_col(J, P1c, par_c):
    """J (E, X, n) joint over existing columns given x -> (E, X, 2n) with the new column as the top bit."""
    E, X, n = J.shape
    cfg = _cfg(par_c, n)
    P = torch.gather(P1c, 2, cfg[:, None, :].expand(E, X, n))
    return torch.cat([J * (1 - P), J * P], 2)


def listed_joint(T):
    P1, par, m = T['P1'], T['par'], T['m']
    E, _, X, _ = P1.shape
    J = torch.ones(E, X, 1, dtype=P1.dtype, device=P1.device)
    for c in range(m):
        J = add_col(J, P1[:, c], par[:, c])
    return J


def joint_BbY(T, J=None):
    """-> (E, 2^m, 2, Y) exact p(B, b, y)."""
    if J is None: J = listed_joint(T)
    E, X, n = J.shape
    m = T['m']
    py = T['py']
    out = torch.empty(E, n, 2, py.shape[-1], dtype=J.dtype, device=J.device)
    isy = T['isy']
    if (~isy).any():
        idx = (~isy).nonzero()[:, 0]
        J2 = add_col(J[idx], T['P1'][idx, m], T['par'][idx, m])
        pj = torch.bmm(J2.transpose(1, 2), py[idx]) / X          # (e, 2n, Y)
        out[idx] = pj.view(len(idx), 2, n, -1).transpose(1, 2)
    if isy.any():
        idx = isy.nonzero()[:, 0]
        pBy = torch.bmm(J[idx].transpose(1, 2), py[idx]) / X     # (e, n, Y)
        q = T['yd'][idx][:, None, :]
        out[idx] = torch.stack([pBy * (1 - q), pBy * q], 2)
    return out


def xlogy(a, b):
    return torch.where(a > 0, a * torch.log2(torch.where(b > 0, b, torch.ones_like(b))), torch.zeros_like(a))


def cmi_from_joint(p):
    """p (E, nB, 2, Y) -> I(b; Y | B) bits (E,)."""
    pB = p.sum((2, 3), keepdim=True)
    pBb = p.sum(3, keepdim=True)
    pBy = p.sum(2, keepdim=True)
    num = p * pB
    den = pBb * pBy
    t = torch.where(p > 0, p * (torch.log2(torch.where(num > 0, num, torch.ones_like(num)))
                                - torch.log2(torch.where(den > 0, den, torch.ones_like(den)))), torch.zeros_like(p))
    return t.sum((1, 2, 3))


def cmi_batch(eps, device='cpu', dtype=torch.float64):
    T = pack(eps, device, dtype)
    return cmi_from_joint(joint_BbY(T)).clamp(min=0)


_DEV = None


def best_device():
    global _DEV
    if _DEV is None:
        _DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _DEV


def cmi_episode(ep):
    return float(cmi_batch([ep], best_device())[0])


def joint_cols(ep, cols):
    """Exact p(C, y) over an arbitrary list of x/parent columns (parents index into `cols`) -> (2^len, Y) numpy fp64."""
    X = 2 ** ep.nx
    J = torch.ones(1, X, 1, dtype=torch.float64)
    for c, col in enumerate(cols):
        par = torch.full((1, 3), -1, dtype=torch.long); par[0, :len(col.par)] = torch.tensor(col.par, dtype=torch.long)
        J = add_col(J, torch.as_tensor(col.P1, dtype=torch.float64)[None], par)
    return (J[0].T @ torch.as_tensor(ep.py)).numpy() / X


def cmi_subset(pCy, ncond, sel):
    """pCy (2^n, Y): I(C_sel; Y | C_0..ncond-1) with the conditioning columns the low bits."""
    n = int(np.log2(pCy.shape[0]))
    keep = list(range(ncond)) + list(sel)
    p = pCy.reshape([2] * n + [pCy.shape[1]])
    # numpy axes: index s = sum bit_c << c -> reshape with C order gives axis 0 = highest bit
    axes_bits = list(range(n))[::-1]           # axis a holds bit axes_bits[a]
    drop = tuple(a for a, b in enumerate(axes_bits) if b not in keep)
    p = p.sum(drop) if drop else p
    kept_bits = [b for b in axes_bits if b in keep]
    cond_ax = tuple(a for a, b in enumerate(kept_bits) if b < ncond)
    sel_ax = tuple(a for a, b in enumerate(kept_bits) if b >= ncond)
    y_ax = p.ndim - 1
    pB = p.sum(sel_ax + (y_ax,), keepdims=True)
    pBs = p.sum((y_ax,), keepdims=True)
    pBy = p.sum(sel_ax, keepdims=True) if sel_ax else p
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(p > 0, p * np.log2(p * pB / (pBs * pBy)), 0)
    return float(t.sum())
