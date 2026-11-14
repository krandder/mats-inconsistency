"""Closed-form error rate of a noisy M_{2,2} (2 neurons, 2 steps), extending ARC's region analysis with the noise.

Input x = rho (cos th, sin th). Without noise the logit difference D = l_1 - l_0 is rho * m0(th), piecewise linear in
(cos th, sin th), with pieces bounded by rays where a gate changes. The noise: step-2 noise enters D linearly; step-1
noise is propagated through step 2 with the gates of the noiseless path (linearisation, exact unless the noise flips a
step-2 gate). On a piece, D = rho * (A cos th + B sin th) + s Z with s constant, so with k = R / s,
  E_rho P(error | th) = (1 - k' cos u / sqrt(1 + k'^2 cos^2 u)) / 2   (Rayleigh rho; u = th - phi; k' signed),
and its integral over the piece is closed form: (u2 - u1)/2 - [arcsin(k sin u / sqrt(1 + k^2))]/2.
Correct sign: the target is the 2nd argmax = 0 if x_0 < x_1, so prediction 1 (D > 0) is correct iff x_0 > x_1.
The target line x_0 = x_1 is added as a breakpoint (the correct sign flips there).
FLOPs are counted.
"""
import math
import numpy as np


def error_rate(W, r, sigma):
    W_hi, W_hh, W_oh = [np.asarray(w, float) for w in W]
    fl = 0
    # breakpoints: x_0 = 0 (step-1 gates), x_0 = x_1 (target), and step-2 gate rays for each step-1 sign pattern
    bps = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, math.pi / 4, 5 * math.pi / 4, 2 * math.pi]
    for s0 in (+1, -1):
        g1 = (W_hi * s0 > 0).astype(float)           # step-1 gates for sign of x_0
        # z2 = W_hh (g1 * W_hi x0) + W_hi x1 = a x0 + c x1
        a = W_hh @ (g1 * W_hi); c = W_hi; fl += 2 * 4 + 8
        for i in range(2):
            # a_i cos + c_i sin = 0  -> th = atan2(-a_i, c_i) (+ pi)
            th = math.atan2(-a[i], c[i]) % (2 * math.pi)
            for t in (th, (th + math.pi) % (2 * math.pi)):
                bps.append(t)
    bps = sorted(set([b % (2 * math.pi) for b in bps] + [2 * math.pi]))
    total = 0.0
    for lo, hi in zip(bps[:-1], bps[1:]):
        if hi - lo < 1e-15: continue
        mid = 0.5 * (lo + hi); x0, x1 = math.cos(mid), math.sin(mid)
        g1 = (W_hi * x0 > 0).astype(float)
        a = W_hh @ (g1 * W_hi); c = W_hi
        z2 = a * x0 + c * x1; g2 = (z2 > 0).astype(float)
        # D = (W_oh[1] - W_oh[0]) . h2 ; h2 = g2 * (a x0 + c x1) + noise2 ; noise1 -> g2 * W_hh noise1
        dw = W_oh[1] - W_oh[0]
        A = dw @ (g2 * a); B = dw @ (g2 * c)
        J1 = (g2[:, None] * W_hh)                        # dh2/dh1
        v = ((dw @ J1) ** 2 * (sigma * r[0]) ** 2).sum() + ((dw) ** 2 * (sigma * r[1]) ** 2).sum()
        fl += 60
        sgn = 1.0 if x0 > x1 else -1.0                   # correct sign of D on this piece
        A *= sgn; B *= sgn                                  # margin m0(th) = A cos + B sin
        R = math.hypot(A, B); phi = math.atan2(B, A)
        if v <= 0:
            # noiseless: error iff margin < 0 on the piece (constant sign inside a piece only if no root; add roots)
            k = 1e300
        s = math.sqrt(max(v, 1e-300)); k = R / s
        u1, u2 = lo - phi, hi - phi
        F = lambda u: math.asin(max(-1.0, min(1.0, k * math.sin(u) / math.sqrt(1 + k * k))))
        total += 0.5 * (hi - lo) - 0.5 * (F(u2) - F(u1))
        fl += 40
    return total / (2 * math.pi), fl


def flip_bound(W, r, sigma):
    """Certified half-width for error_rate(): the linear-response computation is exact unless the step-1 noise flips a
    step-2 gate, so |p - p_lin| <= P(some step-2 gate flips) <= sum_i E_x Phi(-|z_{2,i}(x)| / sd_i(x)), with
    sd_i = sigma * sqrt(sum_j (W_hh[i, j] r_0[j])^2) (the step-1 noise reaches z_2 linearly, before any ReLU).
    The radial expectation is closed form as in error_rate(), on pieces where z_{2,i} has a fixed sign."""
    W_hi, W_hh, W_oh = [np.asarray(w, float) for w in W]
    fl = 0
    bps = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    for s0 in (+1, -1):
        g1 = (W_hi * s0 > 0).astype(float)
        a = W_hh @ (g1 * W_hi); c = W_hi
        for i in range(2):
            th = math.atan2(-a[i], c[i]) % (2 * math.pi)
            bps += [th, (th + math.pi) % (2 * math.pi)]
    bps = sorted(set([b % (2 * math.pi) for b in bps] + [2 * math.pi]))
    sd = sigma * np.sqrt(((W_hh * r[0][None, :]) ** 2).sum(1)); fl += 12
    total = 0.0
    for lo, hi in zip(bps[:-1], bps[1:]):
        if hi - lo < 1e-15: continue
        mid = 0.5 * (lo + hi); x0, x1 = math.cos(mid), math.sin(mid)
        g1 = (W_hi * x0 > 0).astype(float)
        a = W_hh @ (g1 * W_hi); c = W_hi
        for i in range(2):
            A, B = a[i], c[i]
            sgn = 1.0 if (A * x0 + B * x1) > 0 else -1.0
            A *= sgn; B *= sgn                                  # |z_2,i| = rho (A cos + B sin) on this piece
            R = math.hypot(A, B); phi = math.atan2(B, A)
            if sd[i] <= 0: continue
            k = R / sd[i]
            F = lambda u: math.asin(max(-1.0, min(1.0, k * math.sin(u) / math.sqrt(1 + k * k))))
            total += 0.5 * (hi - lo) - 0.5 * (F(hi - phi) - F(lo - phi))
            fl += 40
    return total / (2 * math.pi), fl
