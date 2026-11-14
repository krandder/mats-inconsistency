s = open('prop.py').read()
helper = '''

def _psd(P, log=None):
    """Nearest PSD matrix (eigenvalue clipping) of a [C,n,n] covariance; straight-through (the correction is
    treated as a constant in the backward pass). log: list receiving the clipped negative mass relative to the trace."""
    Ps = 0.5 * (P + P.transpose(-1, -2))
    with torch.no_grad():
        L, U = torch.linalg.eigh(Ps.to(F64))
        neg = L.clamp_max(0)
        corr = -(U * neg[..., None, :]) @ U.transpose(-1, -2)
        if log is not None:
            log.append(float((-neg.sum(-1) / L.clamp_min(0).sum(-1).clamp_min(1e-30)).max()))
    return Ps + corr.to(P.dtype)

'''
anchor = "\nclass Propagator:"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
# residual add
old = "        A = self.pad(A).clone()\n        A[:, :N] = A[:, :N] + Ab\n        return m, Pn, A\n"
assert old in s
s = s.replace(old, "        A = self.pad(A).clone()\n        A[:, :N] = A[:, :N] + Ab\n        if self.psd:\n            Pn = _psd(Pn, self.psd_log)\n        return m, Pn, A\n")
# layernorm output
old = "        Ey = Ey * g + b\n        Py = Py * g[None, None, :, None, None] * g[None, None, None, None, :]\n"
assert old in s
s = s.replace(old, "        if self.psd:\n            Py = _psd(Py.reshape(C, T * d, T * d), self.psd_log).reshape(C, T, d, T, d)\n" + old)
s = s.replace("        self.av_k3 = True\n", "        self.av_k3 = True\n        self.psd = True\n        self.psd_log = []\n", 1)
s = s.replace("    def _reset(self, C):\n        self.c3, self.c4, self.J, self.C = [], [], 0, C\n",
              "    def _reset(self, C):\n        self.c3, self.c4, self.J, self.C = [], [], 0, C\n        self.psd_log = []\n")
open('prop.py', 'w').write(s)
print('ok')
