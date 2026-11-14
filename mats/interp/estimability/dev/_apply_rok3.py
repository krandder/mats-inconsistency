s = open('prop.py').read()
anchor = "            Alp = self.pad(Al)\n"
assert anchor in s
add = '''            if self.av_k3:
                # Cov(o, r) gains sum_s E[f'] kappa3(l_tsh, v_shb, r), kappa3(l, v, r) = sum_z beta[tsh,z] kappa3(s_tzh, v, r),
                # kappa3(s_tzh, v, r) = sc * sum_a [Cov(q_tha, v) Cov(k_zha, r) + Cov(k_zha, v) Cov(q_tha, r)]
                PqvD2 = torch.diagonal(Pqv, 0, 2, 5)                                      # [C,T,A,S,B,H]
                PkvD2 = torch.diagonal(Pkv, 0, 2, 5)                                      # [C,Z,A,S,B,H]
                Fb = F1[..., None] * bl                                                   # [C,T,S,H,Z]
                Wq = ein('ctshz,ctasbh->cthabz', Fb, PqvD2)
                Vk = ein('ctshz,czasbh->cthab', Fb, PkvD2)

                def k3ref(Rk_, Rq_):
                    return sc * (ein('cthabz,czhar->cthbr', Wq, Rk_) + ein('cthab,cthar->cthbr', Vk, Rq_))
'''
s = s.replace(anchor, add + anchor)
old = ("            Ao = (ein('ctsh,cshb,ctshj->cthbj', F1, mv, Alp) + ein('ctsh,ctshb,ctshj->cthbj', F2, c1, Alp)\n"
       "                  + ein('ctsh,cshbj->cthbj', F0, self.pad(Av)))\n")
assert old in s
new = old + ("            if self.av_k3:\n                Ro = Ro + k3ref(Rk, Rq)\n"
             "                if self.K4:\n                    Ao = Ao + k3ref(self.pad(Ak), self.pad(Aq))\n")
s = s.replace(old, new)
open('prop.py', 'w').write(s)
print('ok')
