s = open('prop.py').read()
old = "        Ea = Ea * al\n        Ra = (h1 * al)[..., None] * Rl\n        Pav = (h1 * al)[..., None, None, None] * Plv\n        Aa = (h1 * al)[..., None] * Al\n"
new = '''        Ea = Ea * al
        Ra = (h1 * al)[..., None] * Rl
        Pav = (h1 * al)[..., None, None, None] * Plv
        Aa = (h1 * al)[..., None] * Al
        if self.rowproj:
            # exact constraint sum_s a_ts = 1: orthogonal projection of means and of every covariance/factor row
            nk = al.sum(2, keepdim=True)                                                # [1,T,1,1]
            Ea = Ea + (1 - Ea.sum(2, keepdim=True)) / nk * al
            proj = lambda X, dim: X - al.view(al.shape + (1,) * (X.dim() - 4)) * X.sum(dim, keepdim=True) / nk.view(nk.shape + (1,) * (X.dim() - 4))
            Ra = proj(Ra, 2); Pav = proj(Pav, 2); Aa = proj(Aa, 2)
            Paa = proj(Paa, 2)
            Paa = Paa.permute(0, 4, 5, 6, 1, 2, 3)
            Paa = proj(Paa, 2).permute(0, 4, 5, 6, 1, 2, 3)
'''
assert old in s
s = s.replace(old, new)
s = s.replace("        self.ln2nd = True\n", "        self.ln2nd = True\n        self.rowproj = True\n", 1)
open('prop.py', 'w').write(s)
print('ok')
