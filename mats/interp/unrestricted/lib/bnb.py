"""Certified branch-and-bound over input cones for the mean of a bias-free ReLU net (sigma = 0).

Cells. R^n is split into the 2n cones over the faces of the cube: face (k, s) is {x : s x_k > 0, |x_i| <= |x_k|}.
A cell on face (k, s) is the cone C = {t v(y) : t > 0, y in box [a, b]}, with v(y)_k = s, v(y)_free = y
(free = the n - 1 other coordinates). Its Gaussian mass and first moment are 1-D integrals in t = |x_k|:
  P(C)          = int_0^inf phi(t) prod_i D_i(t) dt,            D_i(t) = Phi(b_i t) - Phi(a_i t)
  E[x_i 1_C]    = int_0^inf phi(t) E_i(t) prod_{j != i} D_j(t) dt, E_i(t) = phi(a_i t) - phi(b_i t)   (i free)
  E[s x_k 1_C]  = int_0^inf t phi(t) prod_i D_i(t) dt
computed by Gauss-Legendre in t on [0, T] (q nodes; accuracy is tested in tests/test_bnb.py).

Bounds. Forward symbolic linear bounds in the slice coordinates y (ReluVal/Neurify style, mid/rad form):
Z_hi = W Hm + |W| Hr, Z_lo = W Hm - |W| Hr; relu upper relaxation by the chord on [min, max] of Z_hi over the box,
lower relaxation alpha Z_lo (alpha in {0, 1}). If every unit of every layer is stable on the box the bounds are the
exact linear map, and the cell's integral is exact. By positive homogeneity the slice bounds extend to the cone:
for x = t v(y), f(x) = t f(v(y)), and a slice-linear c.(y, 1) becomes the x-linear c_free . x_free + c_const s x_k.

Estimates. Certified cells: the exact integral. Other cells: (a) "point": the local linear map at the cell centre
applied to the cell's first moment (exact if the cell lies in the centre's linear region); (b) "mid": the midpoint
of the integrated bounds (uses only the bounds). Certificate: sum over cells of the integrated bounds.
Refinement: split the cell with the largest certified gap (summed over outputs) at the midpoint of the free
coordinate that explains most of its unstable units' ranges.
Floating-point rounding is not controlled (float64), so the certificate is exact arithmetic up to rounding.
"""
import heapq
import math
import numpy as np
from scipy.special import ndtr

try:
    from .nets import mm_flops, pass_flops, F_CDF, F_PDF
except ImportError:  # run as a script
    from nets import mm_flops, pass_flops, F_CDF, F_PDF

SQ2PI = math.sqrt(2.0 * math.pi)


def phi(z):
    return np.exp(-0.5 * z * z) / SQ2PI


def gl_nodes(n, q):
    T = math.sqrt(n) + 10.0
    x, w = np.polynomial.legendre.leggauss(q)
    t = 0.5 * T * (x + 1.0)
    return t, 0.5 * T * w


def cell_moments(lo, hi, t, w):
    """lo, hi (C, m) boxes of the m = n-1 free gnomonic coordinates. Returns mass (C,), Mf (C, m) = E[x_free 1_C],
    Mt (C,) = E[s x_k 1_C] (>= 0)."""
    C, m = lo.shape
    at = lo[:, None, :] * t[None, :, None]  # (C, q, m)
    bt = hi[:, None, :] * t[None, :, None]
    # D = Phi(bt) - Phi(at), computed on the side that avoids cancellation near 1
    pos = at >= 0
    D = np.where(pos, ndtr(-at) - ndtr(-bt), ndtr(bt) - ndtr(at))
    E = phi(at) - phi(bt)
    pre = np.ones((C, len(t), m + 1))
    np.cumprod(D, axis=2, out=pre[:, :, 1:])
    suf = np.ones((C, len(t), m + 1))
    suf[:, :, :m] = np.cumprod(D[:, :, ::-1], axis=2)[:, :, ::-1]
    Pi = pre[:, :, m]
    excl = pre[:, :, :m] * suf[:, :, 1:]  # prod_{j != i} D_j
    wt = w * phi(t)  # (q,)
    mass = Pi @ wt
    Mf = np.einsum('cqi,q->ci', excl * E, wt)
    Mt = Pi @ (wt * t)
    return mass, Mf, Mt


def default_q(n):
    """Gauss-Legendre nodes in t; relative moment error <= 1e-11 on random cells (scratch/quad_check.py)."""
    return 32 if n <= 4 else 48 if n <= 8 else 64 if n <= 16 else 96


def moments_flops(n, q):
    m = n - 1
    return q * m * (2 * F_CDF + 2 * F_PDF) + q * m * 10 + q * (m + 4) * 2


class CertifiedBnB:
    def __init__(self, W, q=None, batch=128, priority='gap', split='mid'):
        self.W = [np.asarray(Wl, dtype=np.float64) for Wl in W]
        self.absW = [np.abs(Wl) for Wl in self.W]
        self.n = self.W[0].shape[0]
        self.L = len(self.W)
        if q is None:
            q = default_q(self.n)
        self.t, self.w = gl_nodes(self.n, q)
        self.q = q
        self.batch = batch
        self.priority = priority  # 'gap': certified gap; 'instab': mass x summed unstable range fractions
        self.split = split        # 'mid': bisect the smear coordinate; 'kink': cut where the worst unit's kink crosses
        self.spos = []
        self.flops = 0.0
        self.pf = pass_flops(self.n, self.L)
        n, m = self.n, self.n - 1
        # cell storage (lists of arrays; leaves tracked by id)
        self.k, self.s, self.lo, self.hi = [], [], [], []
        self.est, self.lbI, self.ubI, self.gap, self.sdim, self.exact = [], [], [], [], [], []
        self.mass = []
        self.heap = []
        self.alive = []
        self.sum_point = np.zeros(n)
        self.sum_mid = np.zeros(n)
        self.sum_lb = np.zeros(n)
        self.sum_ub = np.zeros(n)
        self.mass_exact = 0.0
        self.n_exact = 0
        self.n_leaves = 0
        # initial cells: 2n faces, y in [-1, 1]^(n-1)
        ks, ss = [], []
        for k in range(n):
            for s in (1, -1):
                ks.append(k)
                ss.append(s)
        lo0 = -np.ones((2 * n, m))
        hi0 = np.ones((2 * n, m))
        self._add_cells(np.array(ks), np.array(ss), lo0, hi0)

    # ---------- per-batch evaluation ----------
    def _free_index(self, k):
        return [i for i in range(self.n) if i != k]

    def _bounds(self, ks, ss, lo, hi):
        """Symbolic bounds for a batch of cells. Returns output H_hi, H_lo (B, n, n) in slice coords (n-1 free +
        const), exact flags (B,), smear (B, n-1), and FLOPs."""
        B = len(ks)
        n, m = self.n, self.n - 1
        mid = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo)
        Hm = np.zeros((B, n, n))
        for b in range(B):
            fr = self._free_index(ks[b])
            Hm[b, fr, np.arange(m)] = 1.0
            Hm[b, ks[b], m] = ss[b]
        Hr = np.zeros((B, n, n))
        exact = np.ones(B, dtype=bool)
        smear = np.zeros((B, m))
        best = np.zeros(B)
        brow = np.zeros((B, n))
        fl = 0.0
        mm = mm_flops(n, n, n)
        for l in range(self.L):
            Zm = np.matmul(self.W[l], Hm)
            any_inexact = not exact.all()
            if any_inexact:
                Zr = np.matmul(self.absW[l], Hr)
                fl += np.sum(~exact) * mm
            else:
                Zr = np.zeros_like(Zm)
            fl += B * mm
            # concretise: centre value and box spread of Z_hi = Zm + Zr and Z_lo = Zm - Zr
            Zhi = Zm + Zr
            Zlo = Zm - Zr
            c_hi = np.einsum('bjm,bm->bj', Zhi[:, :, :m], mid) + Zhi[:, :, m]
            s_hi = np.einsum('bjm,bm->bj', np.abs(Zhi[:, :, :m]), rad)
            c_lo = np.einsum('bjm,bm->bj', Zlo[:, :, :m], mid) + Zlo[:, :, m]
            s_lo = np.einsum('bjm,bm->bj', np.abs(Zlo[:, :, :m]), rad)
            u_h, l_h = c_hi + s_hi, c_hi - s_hi
            u_l, l_l = c_lo + s_lo, c_lo - s_lo
            fl += B * 12 * n * n
            # upper relaxation of relu from Z_hi
            Hhi = np.zeros_like(Zhi)
            act_h = l_h >= 0
            una_h = (u_h > 0) & (l_h < 0)
            Hhi[act_h] = Zhi[act_h]
            if una_h.any():
                lam = u_h[una_h] / (u_h[una_h] - l_h[una_h])
                rows = Zhi[una_h] * lam[:, None]
                rows[:, m] -= lam * l_h[una_h]
                Hhi[una_h] = rows
            # lower relaxation from Z_lo
            Hlo = np.zeros_like(Zlo)
            act_l = l_l >= 0
            una_l = (u_l > 0) & (l_l < 0)
            Hlo[act_l] = Zlo[act_l]
            keep = una_l & (u_l > -l_l)
            Hlo[keep] = Zlo[keep]
            unstable = ~(act_l | (u_h <= 0))  # neither certified active nor certified inactive
            if unstable.any():
                exact &= ~unstable.any(axis=1)
                # smear: fraction of each unstable unit's range explained by each free coordinate
                rng_ = np.maximum(u_h - l_l, 1e-300)
                contrib = np.abs(Zm[:, :, :m]) * rad[:, None, :] / rng_[:, :, None]
                smear += np.einsum('bjm,bj->bm', contrib, unstable.astype(float))
                sc = np.where(unstable, np.maximum(u_h, 0) * np.maximum(-l_l, 0) / rng_, 0.0)
                j = np.argmax(sc, axis=1)
                scj = sc[np.arange(B), j]
                upd = scj > best
                if upd.any():
                    best[upd] = scj[upd]
                    brow[upd] = Zm[np.where(upd)[0], j[upd], :]
                fl += B * 6 * n * n
            Hm = 0.5 * (Hhi + Hlo)
            Hr = 0.5 * (Hhi - Hlo)
            Hr[exact] = 0.0
        self._last_brow = brow
        self._last_best = best
        return Hhi, Hlo, exact, smear, fl

    def _point(self, ks, ss, lo, hi, M):
        """Local linear map at the cell centre applied to the first moment M (B, n) in x coordinates."""
        B = len(ks)
        n = self.n
        mid = 0.5 * (lo + hi)
        xc = np.zeros((B, n))
        for b in range(B):
            fr = self._free_index(ks[b])
            xc[b, fr] = mid[b]
            xc[b, ks[b]] = ss[b]
        h = xc
        v = M
        for l in range(self.L):
            z = h @ self.W[l].T
            g = z > 0
            h = np.where(g, z, 0.0)
            v = np.where(g, v @ self.W[l].T, 0.0)
        return v, B * 2 * self.pf

    def _add_cells(self, ks, ss, lo, hi):
        B = len(ks)
        n, m = self.n, self.n - 1
        mass, Mf, Mt = cell_moments(lo, hi, self.t, self.w)
        self.flops += B * moments_flops(n, self.q)
        Hhi, Hlo, exact, smear, fl = self._bounds(ks, ss, lo, hi)
        self.flops += fl
        # integrated bounds: c_free . Mf + c_const * Mt
        ubI = np.einsum('bjm,bm->bj', Hhi[:, :, :m], Mf) + Hhi[:, :, m] * Mt[:, None]
        lbI = np.einsum('bjm,bm->bj', Hlo[:, :, :m], Mf) + Hlo[:, :, m] * Mt[:, None]
        self.flops += B * 4 * n * n
        # first moment in x coordinates
        M = np.zeros((B, n))
        for b in range(B):
            fr = self._free_index(ks[b])
            M[b, fr] = Mf[b]
            M[b, ks[b]] = ss[b] * Mt[b]
        est = 0.5 * (ubI + lbI)
        if (~exact).any():
            idx = np.where(~exact)[0]
            v, fl2 = self._point(ks[idx], ss[idx], lo[idx], hi[idx], M[idx])
            est[idx] = v
            self.flops += fl2
        est[exact] = ubI[exact]
        gap = np.sum(ubI - lbI, axis=1)
        gap[exact] = 0.0
        mid = 0.5 * (lo + hi)
        if self.split == 'kink':
            brow = self._last_brow
            c = brow[:, :m]
            sd = np.argmax(np.abs(c) * (hi - lo), axis=1)
            sp = mid[np.arange(B), sd].copy()
            ci = c[np.arange(B), sd]
            ok = (self._last_best > 0) & (np.abs(ci) > 0)
            rest = np.einsum('bm,bm->b', c, mid) - ci * mid[np.arange(B), sd] + brow[:, m]
            ystar = np.where(ok, -rest / np.where(ok, ci, 1.0), sp)
            w = hi[np.arange(B), sd] - lo[np.arange(B), sd]
            sp = np.clip(ystar, lo[np.arange(B), sd] + 0.15 * w, hi[np.arange(B), sd] - 0.15 * w)
        else:
            sd = np.argmax(smear, axis=1)
            nosmear = smear.max(axis=1) <= 0
            if nosmear.any():
                sd[nosmear] = np.argmax(hi[nosmear] - lo[nosmear], axis=1)
            sp = mid[np.arange(B), sd]
        mid_est = 0.5 * (ubI + lbI)
        for b in range(B):
            cid = len(self.k)
            self.k.append(int(ks[b])); self.s.append(int(ss[b]))
            self.lo.append(lo[b].copy()); self.hi.append(hi[b].copy())
            self.est.append(est[b]); self.lbI.append(lbI[b]); self.ubI.append(ubI[b])
            self.gap.append(float(gap[b])); self.sdim.append(int(sd[b])); self.exact.append(bool(exact[b]))
            self.spos.append(float(sp[b]))
            self.mass.append(float(mass[b]))
            self.alive.append(True)
            self.sum_point += est[b]
            self.sum_mid += mid_est[b]
            self.sum_lb += lbI[b]
            self.sum_ub += ubI[b]
            self.n_leaves += 1
            if exact[b]:
                self.mass_exact += mass[b]
                self.n_exact += 1
            else:
                key = float(gap[b]) if self.priority == 'gap' else float(mass[b] * smear[b].sum())
                heapq.heappush(self.heap, (-key, cid))

    def step(self):
        """Split up to `batch` uncertified leaves with the largest gaps. Returns False when none is left."""
        if not self.heap:
            return False
        pops = []
        while self.heap and len(pops) < self.batch:
            _, cid = heapq.heappop(self.heap)
            pops.append(cid)
        m = self.n - 1
        ks, ss, los, his = [], [], [], []
        for cid in pops:
            self.alive[cid] = False
            self.n_leaves -= 1
            e = self.est[cid]
            self.sum_point -= e
            self.sum_mid -= 0.5 * (self.ubI[cid] + self.lbI[cid])
            self.sum_lb -= self.lbI[cid]
            self.sum_ub -= self.ubI[cid]
            d = self.sdim[cid]
            lo, hi = self.lo[cid], self.hi[cid]
            c = self.spos[cid]
            lo1, hi1 = lo.copy(), hi.copy()
            hi1[d] = c
            lo2, hi2 = lo.copy(), hi.copy()
            lo2[d] = c
            for (a, b) in ((lo1, hi1), (lo2, hi2)):
                ks.append(self.k[cid]); ss.append(self.s[cid]); los.append(a); his.append(b)
            # free the parent's arrays
            self.est[cid] = self.lbI[cid] = self.ubI[cid] = None
            self.lo[cid] = self.hi[cid] = None
        self.flops += len(pops) * 10 * self.n
        self._add_cells(np.array(ks), np.array(ss), np.array(los), np.array(his))
        return True

    def state(self):
        return dict(flops=self.flops, passes=self.flops / self.pf, point=self.sum_point.copy(),
                    mid=self.sum_mid.copy(), lb=self.sum_lb.copy(), ub=self.sum_ub.copy(),
                    leaves=self.n_leaves, exact_cells=self.n_exact, mass_exact=self.mass_exact,
                    open=len(self.heap))

    def run(self, max_passes, checkpoints=None, max_leaves=None):
        """Run until max_passes forward-pass equivalents, recording the state at each checkpoint (in passes)."""
        out = []
        cps = sorted(checkpoints or [])
        ci = 0
        while True:
            while ci < len(cps) and self.flops / self.pf >= cps[ci]:
                st = self.state(); st['checkpoint'] = cps[ci]; out.append(st); ci += 1
            if self.flops / self.pf >= max_passes:
                break
            if max_leaves is not None and self.n_leaves >= max_leaves:
                break
            if not self.step():
                break
        st = self.state(); st['checkpoint'] = 'final'; out.append(st)
        return out
