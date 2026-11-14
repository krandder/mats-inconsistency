s = open('prop.py').read()
old = """    c1 = (wt * f * h1).sum(-1) / sd
    c2 = (wt * f * h2).sum(-1) / var
    c3 = (wt * f * h3).sum(-1) / (sd * var)
    return [t.to(dt) for t in (E, V, m3, m4 - 3 * V ** 2, c1, c2, c3)]
"""
new = """    # Hermite coefficients as E[f^(k)] (equal to E[f He_k]/sd^k by Stein; numerically stable as sd -> 0)
    d1, d2, d3 = DERIVS[fn](z)
    c1 = (wt * d1).sum(-1); c2 = (wt * d2).sum(-1); c3 = (wt * d3).sum(-1)
    return [t.to(dt) for t in (E, V, m3, m4 - 3 * V ** 2, c1, c2, c3)]
"""
assert old in s
s = s.replace(old, new)
s = s.replace('Returns mean, var, kappa3, kappa4 of fn(z) and Hermite coefficients h_k = E[fn(z) He_k(x)]/sd^k, k=1..3."""',
              'Returns mean, var, kappa3, kappa4 of fn(z) and Hermite coefficients h_k = E[fn^(k)(z)], k=1..3."""')
# derivative table, defined after gelu
anchor = "def gelu(z):\n    return 0.5 * z * (1 + torch.erf(z / math.sqrt(2)))\n"
assert anchor in s
deriv = anchor + '''

def _phi(z):
    return torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _gelu_d(z):
    p = _phi(z)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * p, (2 - z * z) * p, (z ** 3 - 4 * z) * p


def _softplus_d(z):
    s_ = torch.sigmoid(z); q = s_ * (1 - s_)
    return s_, q, q * (1 - 2 * s_)


def _signeg(z):
    return torch.sigmoid(-z)


def _signeg_d(z):
    p = torch.sigmoid(-z); q = p * (1 - p)
    return -q, q * (1 - 2 * p), -q * (1 - 6 * q)


DERIVS = {gelu: _gelu_d, Fn.softplus: _softplus_d, _signeg: _signeg_d}
'''
s = s.replace(anchor, deriv)
s = s.replace("        sig = lambda z: torch.sigmoid(-z)\n        Ea, Va, k3a, k4a, h1, h2, h3 = elementwise(sig, ml, Vl, k3l, k4l, self.K4)",
              "        Ea, Va, k3a, k4a, h1, h2, h3 = elementwise(_signeg, ml, Vl, k3l, k4l, self.K4)")
assert "elementwise(_signeg" in s
open('prop.py', 'w').write(s)
print('ok')
