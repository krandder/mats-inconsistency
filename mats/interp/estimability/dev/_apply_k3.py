s = open('prop.py').read()
# track Clark linear coefficients beta[t,s,h,w]
old = "        started = torch.zeros(T, T, dtype=torch.bool, device=self.dev)\n"
assert old in s
s = s.replace(old, old + "        bZ = torch.zeros(C, T, T, H, T, dtype=self.dt, device=self.dev)\n")
old = "            started = started | valid\n"
assert old in s
s = s.replace(old, "            oh = torch.zeros(T, dtype=self.dt, device=self.dev); oh[w] = 1.0\n"
                   "            bZ = (1 - p)[..., None] * bZ + p[..., None] * oh\n" + old)
old = "        Plv = vZ - Psv\n"
assert old in s
s = s.replace(old, old + "        bl = bZ - torch.eye(T, dtype=self.dt, device=self.dev)[None, None, :, None, :].permute(0, 1, 2, 3, 4)\n")
# add kappa3(v, l, v') term to Po in the Stein branch
old = "            Ro = (ein('ctsh,cshb,ctshr->cthbr', F1, mv, Rl)"
assert old in s
add = '''            if self.av_k3:
                # kappa3(v_x, l_b, v_y) = sum_sigma beta[b,sigma] kappa3(v_x, s_sigma, v_y), with the exact Wick form for
                # bilinear scores: kappa3(v_x, q.k, v_y) = sum_a Cov(v_x,q_a)Cov(k_a,v_y) + Cov(v_x,k_a)Cov(q_a,v_y)
                Pkv_ = Pkv; Pqv_ = Pqv                                                     # [C,T,H,A,U,G,B]
                PkvD = torch.diagonal(Pkv_, 0, 2, 5).permute(0, 1, 5, 2, 3, 4)            # [C,Z,G,A,W,E]
                PqvD = torch.diagonal(Pqv_, 0, 2, 5).permute(0, 1, 5, 2, 3, 4)            # [C,U,G,A,W,E]
                blz = bl.permute(0, 1, 2, 3, 4)                                            # [C,U,W,G,Z]
                M1 = ein('cuwgz,czgawe->cuwgae', blz, PkvD)
                K3 = ein('cugashb,cuwgae->cshbuwge', Pqv_, M1)
                M2 = ein('cuwgz,czgashb->cuwgashb', blz, Pkv_)
                K3 = sc * (K3 + ein('cuwgashb,cugawe->cshbuwge', M2, PqvD))              # [C,S,H,B,U,W,G,E]
                Tk = ein('ctsh,cuwg,cshbuwge->cthbuge', F0, F1, K3)
                Po = Po + Tk + Tk.permute(0, 4, 5, 6, 1, 2, 3)
'''
s = s.replace(old, add + old)
s = s.replace("        self.av_stein = True\n", "        self.av_stein = True\n        self.av_k3 = True\n", 1)
open('prop.py', 'w').write(s)
print('ok')
