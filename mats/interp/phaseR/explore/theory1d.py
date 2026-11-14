# 1-D theory numbers used to write the predictions (no network chain involved):
# per-neuron non-Gaussianity of h = (phi(z) - E phi)/sd(phi) + sigma*eps, z ~ N(0, g^2 (1+sigma^2)) (mean-field LN),
# against the moment-matched Gaussian N(0, 1 + sigma^2); and the chaos factor chi = g^2 E[phi'(z)^2] / Var(phi(z))
# (mean squared singular value of one LN layer's Jacobian on its input).
# Density of s by a fine histogram (bin << sigma), noise convolution by FFT.
import numpy as np, math

LOG2 = math.log(2)


def per_neuron(phi, dphi, g, sig, lim=12.0):
    v = g * g * (1 + sig * sig)
    zz = np.linspace(-12, 12, 400001); wz = np.exp(-zz ** 2 / 2); wz /= wz.sum()
    a = phi(zz * math.sqrt(v))
    m = wz @ a; var = wz @ (a - m) ** 2
    s = (a - m) / math.sqrt(var)
    # histogram of s with bins << sigma, then exact log-domain convolution (no FFT floor in the tails)
    nb = int(min(6000, max(800, 40 * (s.max() - s.min()) / sig)))
    cnt, edges = np.histogram(s, bins=nb, weights=wz); cen = 0.5 * (edges[1:] + edges[:-1]); keep = cnt > 0
    cen, lw = cen[keep], np.log(cnt[keep])
    H = np.linspace(-lim, lim, 7001); dh = H[1] - H[0]
    from scipy.special import logsumexp
    lr = logsumexp(lw[None] - (H[:, None] - cen[None]) ** 2 / (2 * sig * sig), axis=1) - 0.5 * math.log(2 * math.pi * sig * sig)
    r = np.exp(lr)
    vg = 1 + sig * sig
    lg = -H ** 2 / (2 * vg) - 0.5 * math.log(2 * math.pi * vg); gq = np.exp(lg)
    ok = np.ones_like(H, bool)
    kl_g_r = (gq * (lg - lr))[ok].sum() * dh / LOG2   # waste direction KL(q||Kq) per neuron (bits; lower if tails cut)
    kl_r_g = (r * (lr - lg)).sum() * dh / LOG2        # KL(Kq||q)
    dinf = (lr - lg)[ok].max() / LOG2                 # coverage D_inf(Kq||q)
    arg = H[ok][np.argmax((lr - lg)[ok])]
    chi = g * g * (wz @ dphi(zz * math.sqrt(v)) ** 2) / var
    return kl_g_r, kl_r_g, dinf, arg, chi


acts = {"tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2), "relu": (lambda z: np.maximum(z, 0), lambda z: (z > 0) * 1.0)}
print("phi g sigma | KL(q||Kq) KL(Kq||q) Dinf(Kq||q) [bits/neuron] argmax | chi")
for name, gs in (("tanh", (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)), ("relu", (1.414,))):
    phi, dphi = acts[name]
    for sig in (0.1, 0.3, 1.0):
        for g in gs:
            a, b, c, arg, chi = per_neuron(phi, dphi, g, sig)
            print(f"{name} g={g:4.2f} s={sig:3.1f} | {a:7.4f} {b:7.4f} {c:7.4f} at h={arg:6.2f} | chi={chi:5.3f}", flush=True)
