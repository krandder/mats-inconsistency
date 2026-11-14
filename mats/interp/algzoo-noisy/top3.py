"""Amendment 7, step 2: TOP3, a deductive error-rate estimator with no closure in the algorithmic variables.

The ordered positions (i, j, k) of the max, 2nd and 3rd max are uniform over the n(n-1)(n-2) triples, independent of
the values. Coordinates (c_s, c_lo, t) uniform on (0,1)^3 map exactly to the top-3 values: s = x(3) = F3^{-1}(c_s);
given s the top two are two iid N(0,1) truncated to (s, inf), the smaller at quantile q_lo = 1 - sqrt(1 - c_lo), the
larger at q_hi = q_lo + (1 - q_lo) t. Boxes: KS equal intervals in c_s, KG intervals with edges (i/KG)^2 in c_lo and
t; exact box probabilities; each box represented by its centre.

Per node (cell, box), from the weights alone: representative input with the top three at (i, j, k) and the 7 others
at E[Z | Z < s]; one noiseless forward pass fixes the ReLU pattern; the logit differences D_c = l_j - l_c of the 3
competitors with the smallest D_c are linearised (Jacobians w.r.t. the 7 lower inputs, variance Var(Z | Z < s) each,
and w.r.t. the post-ReLU activation at every step, variance (sigma r_t,i)^2); P(error | node) =
1 - Phi2(z_a, z_b; rho_ab) Phi(z_c), the two most dangerous competitors exactly.
"""
import math
import numpy as np
from scipy.special import ndtr, ndtri

GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def phi2(h, k, rho):
    """Bivariate standard normal CDF P(Y1 < h, Y2 < k), corr rho; vectorised; theta-substitution + Gauss-Legendre."""
    h = np.clip(h, -40, 40); k = np.clip(k, -40, 40)
    rho = np.clip(rho, -0.9999, 0.9999); th1 = np.arcsin(rho)
    th = 0.5 * th1[..., None] * (GL_X + 1)                       # (..., 24) nodes on [0, th1]
    c2 = np.cos(th) ** 2
    ex = np.exp(-(h[..., None] ** 2 + k[..., None] ** 2 - 2 * h[..., None] * k[..., None] * np.sin(th)) / (2 * c2))
    integ = 0.5 * th1 * (ex * GL_W).sum(-1)
    return np.clip(ndtr(h) * ndtr(k) + integ / (2 * math.pi), 0.0, 1.0)


class Top3:
    def __init__(self, W, r, sigma, ncomp=3):
        self.Whi, self.Whh, self.Woh = [np.asarray(w, np.float64) for w in W]
        self.r = np.asarray(r, np.float64); self.sigma = float(sigma)
        self.d = self.Whi.shape[0]; self.n = self.Woh.shape[0]; self.nc = ncomp
        n = self.n
        self.cells = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n) if len({i, j, k}) == 3])
        # inverse CDF of the 3rd largest of n iid N(0,1) on a fine grid
        g = np.linspace(-6, 7, 130001); P = ndtr(g); Q = ndtr(-g)
        F3 = sum(math.comb(n, kk) * Q ** kk * P ** (n - kk) for kk in range(3))
        self.g3, self.F3 = g, F3
        d = self.d
        self.fl_node = (n * (2 * d * d + 3 * d) + 2 * n * d + n * math.log2(n) + ncomp * n * (2 * d * d + 3 * d)
                        + n * ncomp * ncomp * 2 * d + ncomp * ncomp * 2 * 7 + 24 * 25 + 300)

    # ---------------- exact law of the top-3 values ----------------
    def s_of(self, cs):
        return np.interp(cs, self.F3, self.g3)

    def cs_of(self, s):
        return np.interp(s, self.g3, self.F3)

    @staticmethod
    def values(s, clo, t):
        sf_s = ndtr(-s)
        sf2 = sf_s * np.sqrt(1 - clo)                 # survival of u2: sf(s) (1 - q_lo), 1 - q_lo = sqrt(1 - c_lo)
        sf1 = sf2 * (1 - t)                           # survival of u1: sf(s) (1 - q_hi)
        return -ndtri(np.clip(sf2, 1e-300, 1)), -ndtri(np.clip(sf1, 1e-300, 1))

    def grid(self, KS, KG):
        es = np.arange(KS + 1) / KS; eg = (np.arange(KG + 1) / KG) ** 2
        cs = 0.5 * (es[:-1] + es[1:]); cg = 0.5 * (eg[:-1] + eg[1:])
        CS, CL, CT = np.meshgrid(cs, cg, cg, indexing='ij')
        PB = np.einsum('a,b,c->abc', np.diff(es), np.diff(eg), np.diff(eg))
        s = self.s_of(CS.ravel()); u2, u1 = self.values(s, CL.ravel(), CT.ravel())
        return dict(KS=KS, KG=KG, es=es, eg=eg, s=s, u2=u2, u1=u1, pb=PB.ravel())

    def box_of(self, x, G):
        """cell index (into self.cells) and box index of each input x (B, n) for grid G."""
        n = self.n
        idx = np.argsort(-x, 1); srt = np.take_along_axis(x, idx, 1)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        cell = i * (n - 1) * (n - 2) + (j - (j > i)) * (n - 2) + (k - (k > i) - (k > j))
        s, u2, u1 = srt[:, 2], srt[:, 1], srt[:, 0]
        cs = self.cs_of(s); sfs = ndtr(-s); sf2 = ndtr(-u2); sf1 = ndtr(-u1)
        clo = 1 - (sf2 / sfs) ** 2; t = 1 - sf1 / np.maximum(sf2, 1e-300)
        a = np.clip(np.searchsorted(G['es'], cs, 'right') - 1, 0, G['KS'] - 1)
        b = np.clip(np.searchsorted(G['eg'], clo, 'right') - 1, 0, G['KG'] - 1)
        c = np.clip(np.searchsorted(G['eg'], t, 'right') - 1, 0, G['KG'] - 1)
        return cell, (a * G['KG'] + b) * G['KG'] + c

    # ---------------- per-node error model ----------------
    def node_perr(self, X0, y, low, s):
        """X0 (B, n) representative inputs, y (B,) target positions, low (B, n) mask of the lower inputs, s (B,)."""
        Whi, Whh, Woh, n, d, nc = self.Whi, self.Whh, self.Woh, self.n, self.d, self.nc
        B = X0.shape[0]; h = np.zeros((B, d)); masks = np.empty((n, B, d), bool)
        for t in range(n):
            z = h @ Whh.T + X0[:, t:t + 1] * Whi
            masks[t] = z > 0; h = np.maximum(z, 0)
        lo = h @ Woh.T
        D = lo[np.arange(B), y][:, None] - lo; D[np.arange(B), y] = np.inf
        comp = np.argsort(D, 1)[:, :nc]; Dm = np.take_along_axis(D, comp, 1)
        G = Woh[y][:, None, :] - Woh[comp]                                   # (B, nc, d) d D / d h_n
        C = np.zeros((B, nc, nc)); Jx = np.zeros((B, nc, n))
        for t in range(n - 1, -1, -1):
            C += np.einsum('bci,bki,i->bck', G, G, (self.sigma * self.r[t]) ** 2)   # noise added to h_{t+1}
            Gz = G * masks[t][:, None, :]
            Jx[:, :, t] = Gz @ Whi
            G = Gz @ Whh
        ml = -np.exp(-0.5 * s ** 2) / math.sqrt(2 * math.pi) / np.maximum(ndtr(s), 1e-300)
        vlo = np.maximum(1 - s * (-ml) - ml ** 2, 0.0)                       # Var(Z | Z < s)
        Jl = Jx * low[:, None, :]
        C += np.einsum('bcm,bkm->bck', Jl, Jl) * vlo[:, None, None]
        sd = np.sqrt(np.maximum(np.einsum('bcc->bc', C), 1e-300))
        z = Dm / sd
        o = np.argsort(z, 1); za = np.take_along_axis(z, o, 1)
        a, b2 = o[:, 0], o[:, 1]
        rho = C[np.arange(B), a, b2] / (sd[np.arange(B), a] * sd[np.arange(B), b2])
        pok = phi2(za[:, 0], za[:, 1], rho)
        for q in range(2, nc):
            pok = pok * ndtr(np.clip(za[:, q], -40, 40))
        return 1 - pok

    def rep_inputs(self, cells, s, u2, u1):
        """Representative inputs for arrays of cells (B, 3) and values (B,)."""
        n = self.n; B = len(s)
        ml = -np.exp(-0.5 * s ** 2) / math.sqrt(2 * math.pi) / np.maximum(ndtr(s), 1e-300)
        X0 = np.repeat(ml[:, None], n, 1); low = np.ones((B, n), bool)
        ar = np.arange(B)
        X0[ar, cells[:, 0]] = u1; X0[ar, cells[:, 1]] = u2; X0[ar, cells[:, 2]] = s
        low[ar, cells[:, 0]] = False; low[ar, cells[:, 1]] = False; low[ar, cells[:, 2]] = False
        return X0, low

    def estimate(self, KS, KG, chunk=16384):
        """Deductive average over all cells and boxes. Returns (p_hat, flops, table (n_cells, n_boxes), grid)."""
        G = self.grid(KS, KG); nb = len(G['s']); nC = len(self.cells)
        cel = np.repeat(np.arange(nC), nb); box = np.tile(np.arange(nb), nC)
        P = np.empty(nC * nb)
        for i0 in range(0, nC * nb, chunk):
            sl = slice(i0, i0 + chunk); cc = self.cells[cel[sl]]; bb = box[sl]
            X0, low = self.rep_inputs(cc, G['s'][bb], G['u2'][bb], G['u1'][bb])
            P[sl] = self.node_perr(X0, cc[:, 1], low, G['s'][bb])
        T = P.reshape(nC, nb)
        p_hat = float((T * G['pb'][None, :]).sum() / nC)
        return p_hat, float(nC * nb * self.fl_node), T, G

    def predict_inputs(self, x, chunk=16384):
        """Per-input prediction at each input's own top-3 values and positions (diagnostic)."""
        out = np.empty(x.shape[0])
        for i0 in range(0, x.shape[0], chunk):
            xc = x[i0:i0 + chunk]; idx = np.argsort(-xc, 1); srt = np.take_along_axis(xc, idx, 1)
            X0, low = self.rep_inputs(idx[:, :3], srt[:, 2], srt[:, 1], srt[:, 0])
            out[i0:i0 + chunk] = self.node_perr(X0, idx[:, 1], low, srt[:, 2])
        return out

    def sample_box(self, cell_ids, box_ids, G, rng):
        """Exact samples x from the input law restricted to the given (cell, box) units (one sample per entry)."""
        n = self.n; B = len(cell_ids); KG = G['KG']
        a = box_ids // (KG * KG); b = (box_ids // KG) % KG; c = box_ids % KG
        cs = G['es'][a] + rng.random(B) * (G['es'][a + 1] - G['es'][a])
        clo = G['eg'][b] + rng.random(B) * (G['eg'][b + 1] - G['eg'][b])
        t = G['eg'][c] + rng.random(B) * (G['eg'][c + 1] - G['eg'][c])
        s = self.s_of(cs); u2, u1 = self.values(s, clo, t)
        cells = self.cells[cell_ids]
        # 7 lower inputs: iid N(0,1) truncated to (-inf, s), by inverse CDF
        lowv = ndtri(np.clip(rng.random((B, n)) * ndtr(s)[:, None], 1e-300, 1))
        x = lowv; ar = np.arange(B)
        x[ar, cells[:, 0]] = u1; x[ar, cells[:, 1]] = u2; x[ar, cells[:, 2]] = s
        return x
