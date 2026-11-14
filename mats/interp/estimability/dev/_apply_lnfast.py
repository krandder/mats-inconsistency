s = open('prop.py').read()
old_start = s.index("        res = {}\n        for k in (1, 2, 3, 4, 5):")
old_end = s.index("        wt1, ms1, Ss1 = res[1]")
new = '''        # one Gauss-Laguerre(alpha=-1/2) node set for all orders k: int s^(k/2-1) e^(-cs) G ds with weights x^((k-1)/2)
        x, w = _laguerre(-0.5, self.ln_n)
        x = torch.tensor(x, dtype=F64, device=self.dev); w = torch.tensor(w, dtype=F64, device=self.dev)
        sn = x / Ev[..., None]                                                   # [C,T,n]
        Bk = I + (2 * sn / d)[..., None, None] * Sd[:, :, None]                  # [C,T,n,d,d]
        Binv = torch.linalg.inv(Bk)
        logdet = torch.linalg.slogdet(Bk)[1]
        ms = (Binv @ m64[:, :, None, :, None])[..., 0]                           # [C,T,n,d]
        Ss = Binv @ Sd[:, :, None]
        logZ = -0.5 * logdet - (sn / d) * (m64[:, :, None] * ms).sum(-1) - sn * LN_EPS
        base = w * torch.exp(logZ + x)
        res = {}
        for k in (1, 2, 3, 4, 5):
            if k == 5 and not self.ln2nd:
                continue
            al = k / 2 - 1
            wt = base * x ** (al + 0.5) * Ev[..., None] ** (-al - 1) / math.gamma(k / 2)
            res[k] = (wt, ms, Ss)
'''
s = s[:old_start] + new + s[old_end:]
s = s.replace("        self.ln_pair_n = 6\n", "        self.ln_pair_n = 6\n        self.ln_n = 16\n", 1)
open('prop.py', 'w').write(s)
print('ok')
