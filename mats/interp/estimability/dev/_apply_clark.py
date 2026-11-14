s = open('prop.py').read()
a = s.index("        alw = allow.to(self.dt)\n")
b = s.index("        k3l, k4l = self.marg(Al)")
s = s[:a] + open('_patch_clark.txt').read() + s[b:]
# elementwise must accept None cumulants in Gaussian mode
s = s.replace("    if K4:\n        g3 = (k3.to(F64)", "    if K4 and k3 is not None:\n        g3 = (k3.to(F64)")
open('prop.py', 'w').write(s)
print('ok')
