s = open('prop.py').read()
anchor = "        eyeT = torch.eye(T, dtype=dt, device=self.dev)\n        Py = Py * (1 - eyeT)"
assert anchor in s
s = s.replace(anchor, open('_patch_ln2d.txt').read() + anchor)
s = s.replace("        self.ln2nd = True\n", "        self.ln2nd = False\n        self.ln_pair = True\n        self.ln_pair_n = 10\n", 1)
s = s.replace("        Py = torch.einsum('ctia,ctasb,csjb->ctisj', Jd, P, Jd)\n", "        Py = torch.einsum('ctia,ctasb,csjb->ctisj', Jd, P, Jd) if not self.ln_pair else None\n")
open('prop.py', 'w').write(s)
print('ok')
