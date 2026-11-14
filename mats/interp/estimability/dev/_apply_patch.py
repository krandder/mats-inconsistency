import sys
s = open('prop.py').read()
ln = open('_patch_ln.txt').read()
sm = open('_patch_sm.txt').read()
a = s.index("    def layernorm(self, blk, g, b):")
b = s.index("    def gelu_op(self, blk):")
s = s[:a] + ln + s[b:]
a = s.index("        # ---- log-normaliser L[t,h]")
b = s.index("        # ---- o[t,h,b] = sum_s a[t,s,h] v[s,h,b]")
s = s[:a] + sm + s[b:]
s = s.replace("        self.dbg = dict(ms=ms, Pss=Pss)\n        # ---- softmax", "        # ---- softmax")
s = s.replace("def gelu(z):", '''_LAG = {}


def _laguerre(al, n=32):
    if al not in _LAG:
        from scipy.special import roots_genlaguerre
        _LAG[al] = roots_genlaguerre(n, al)
    return _LAG[al]


def gelu(z):''', 1)
s = s.replace("        self.K4 = (K == 4)\n", "        self.K4 = (K == 4)\n        self.ln2nd = True\n", 1)
open('prop.py', 'w').write(s)
print('ok')
