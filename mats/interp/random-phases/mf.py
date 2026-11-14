"""Mean-field (infinite-width) theory for the random-phases study (work/random-phases, 2026-10-29).

Nets (per-coordinate input variance 1, x ~ N(0, I_n)):
  tanh:   z_l = W_l h_{l-1} + b_l, W iid N(0, sw2/n), b iid N(0, sb2);  a_l = tanh(z_l);  h_l = a_l + sigma r_l eps
  relu:   ARC's family, W iid N(0, 2/n), no bias, a_l = relu(z_l), h_l = a_l + sigma r_l eps
  relu_ln: s_l = LN(relu(z_l)) (centre, scale to RMS 1), h_l = s_l + sigma eps (gain drops out)
r_l^2 = E[a_l^2] (RMS over neurons and paths), so the noise is a fixed fraction sigma of the activation scale.

Recursions (all exact in the infinite-width limit):
  q_{l+1}   = sw2 (1 + sigma^2) Qa(q_l) + sb2                 (diagonal; noise adds sigma^2 r^2 = sigma^2 Qa)
  q12_{l+1} = sw2 Ca(q_l, q12_l) + sb2                         (two copies with independent noise)
Copies that share h_m (branch at depth m) start layer m+1 with q12 = q. Two different inputs start with
q12_1 = sb2 (x.x'/n ~ 0). For the output a_L (post-activation, before its noise):
  K_same = Ca(q_L, q12_same), K_diff = Ca(q_L, q12_diff), K_diag = Qa(q_L)
  R(m -> L) = (K_same - K_diff) / (K_diag - K_diff)   = share of Var(a_L) explained by h_m (m = 0: the input).
chi_1 = sw2 E[phi'(z)^2] at the (noisy) fixed point; lambda_mf = 0.5 ln chi_1 per layer.
chi_c = sw2 E[phi'(u) phi'(v)] at the different-input fixed point q12* (contraction of the cross-covariance).
"""
import math
import numpy as np

_X, _W = np.polynomial.hermite_e.hermegauss(96)
_W = _W / _W.sum()
G1, G2 = np.meshgrid(_X, _X, indexing='ij')
WW = np.outer(_W, _W)


def E1(f, q, mu=0.0):
    return float((f(mu + math.sqrt(max(q, 0.0)) * _X) * _W).sum())


def E2(f, g, q1, q2, q12):
    s1, s2 = math.sqrt(q1), math.sqrt(q2)
    c = float(np.clip(q12 / (s1 * s2), -1.0, 1.0))
    u = s1 * G1
    v = s2 * (c * G1 + math.sqrt(max(1 - c * c, 0.0)) * G2)
    return float((f(u) * g(v) * WW).sum())


tanh = np.tanh
dtanh = lambda z: 1.0 / np.cosh(z) ** 2


def relu_C(q1, q2, q12):
    """E[relu(u) relu(v)] (arc-cosine kernel of degree 1)."""
    s = math.sqrt(q1 * q2)
    c = min(max(q12 / s, -1.0), 1.0)
    return s / (2 * math.pi) * (math.sqrt(max(1 - c * c, 0.0)) + (math.pi - math.acos(c)) * c)


class MF:
    def __init__(self, kind, sw2=2.0, sb2=0.0, sigma=0.0):
        self.kind, self.sw2, self.sb2, self.sigma = kind, sw2, sb2, sigma

    # ---- one-layer maps on (q, q12) of the pre-activations
    def Qa(self, q):
        if self.kind == 'tanh':
            return E1(lambda z: tanh(z) ** 2, q)
        if self.kind == 'relu':
            return q / 2
        return 1.0                                                   # relu_ln: s has RMS 1

    def Ca(self, q, q12):
        if self.kind == 'tanh':
            return E2(tanh, tanh, q, q, q12)
        if self.kind == 'relu':
            return relu_C(q, q, q12)
        c = min(max(q12 / q, -1.0), 1.0)                             # relu_ln: centred, normalised arc-cosine
        return ((math.sqrt(max(1 - c * c, 0.0)) + (math.pi - math.acos(c)) * c) - 1.0) / (math.pi - 1.0)

    def next_q(self, q):
        if self.kind == 'relu_ln':
            return 1.0 + self.sigma ** 2                             # gain drops out under LN; z-variance = |h|^2/n
        return self.sw2 * (1 + self.sigma ** 2) * self.Qa(q) + self.sb2

    def next_q12(self, q, q12):
        if self.kind == 'relu_ln':
            return self.Ca(q, q12)
        return self.sw2 * self.Ca(q, q12) + self.sb2

    def q1(self):
        return 1.0 if self.kind == 'relu_ln' else self.sw2 + self.sb2   # x ~ N(0, I): first-layer z-variance

    def chi1(self, q):
        if self.kind == 'tanh':
            return self.sw2 * E1(lambda z: dtanh(z) ** 2, q)
        if self.kind == 'relu':
            return self.sw2 * 0.5
        return (math.pi / (math.pi - 1.0)) / (1 + self.sigma ** 2)

    # ---- curves
    def run(self, L, m_list=(0,)):
        """Returns dict with q[l], R[m][l] for l = 1..L (index l-1), chi1[l], c_diff[l]."""
        q = [self.q1()]
        for l in range(1, L):
            q.append(self.next_q(q[-1]))
        q12d = [self.sb2 if self.kind == 'tanh' else 0.0]
        for l in range(1, L):
            q12d.append(self.next_q12(q[l - 1], q12d[-1]))
        Kdiag = [self.Qa(qq) if self.kind != 'relu_ln' else 1.0 for qq in q]
        Kdiff = [self.Ca(q[l], q12d[l]) for l in range(L)]
        out = dict(q=q, c_diff=[q12d[l] / q[l] for l in range(L)], chi1=[self.chi1(qq) for qq in q], R={})
        for m in m_list:
            R = [float('nan')] * L
            if m == 0:
                q12s = q[0]                                          # same input: identical layer-1 pre-activations
                start = 0
            else:
                start = m                                            # layer m+1 (index m) starts identical
                q12s = q[m] if m < L else None
            if start >= L:
                out['R'][m] = R
                continue
            for l in range(start, L):
                if l > start:
                    q12s = self.next_q12(q[l - 1], q12s)
                Ks = self.Ca(q[l], q12s)
                den = Kdiag[l] - Kdiff[l]
                R[l] = (Ks - Kdiff[l]) / den if den > 0 else float('nan')
            out['R'][m] = R
        return out


def fixed_point(sw2, sb2, sigma=0.0, iters=2000):
    m = MF('tanh', sw2, sb2, sigma)
    q = sw2 + sb2
    for _ in range(iters):
        q = m.next_q(q)
    q12 = sb2
    for _ in range(iters):
        q12 = m.next_q12(q, q12)
    chi1 = m.chi1(q)
    chic = sw2 * E2(dtanh, dtanh, q, q, q12)
    return dict(q=q, c_star=q12 / q, chi1=chi1, chi_c=chic, lam=0.5 * math.log(chi1))


def critical_sw2(sb2, sigma=0.0):
    lo, hi = 0.5, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if fixed_point(mid, sb2, sigma, iters=600)['chi1'] < 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crossing(R, level=0.5):
    """First depth (1-based, linear interpolation) at which R falls below level."""
    for l in range(1, len(R)):
        a, b = R[l - 1], R[l]
        if np.isfinite(a) and np.isfinite(b) and a >= level > b:
            return l + (a - level) / (a - b)
    return float('nan')


if __name__ == '__main__':
    import sys, json
    sb2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.09
    print('critical sw2 at sb2 =', sb2, 'sigma=0:', critical_sw2(sb2))
    for sw2 in (0.5, 0.8, 1.0, 1.2, 1.5, 1.76, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0):
        fp = fixed_point(sw2, sb2)
        print(f"sw2={sw2:5.2f} sw={math.sqrt(sw2):.3f} q*={fp['q']:.4f} c*={fp['c_star']:.4f} chi1={fp['chi1']:.4f} "
              f"lam={fp['lam']:+.4f} chi_c={fp['chi_c']:.4f}")
