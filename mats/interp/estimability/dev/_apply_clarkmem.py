s = open('prop.py').read()
old = """            rZ = (1 - pe) * rZ + pe * Rs[:, :, w][:, :, None]
            vZ = (1 - p)[..., None, None, None] * vZ + p[..., None, None, None] * Psv[:, :, w][:, :, None]
            if self.K4:
                AZ = (1 - pe) * AZ + pe * Asp[:, :, w][:, :, None]
"""
assert old in s
s = s.replace(old, "")
old2 = "        rowok = started[None]                                                          # [1,T,S]\n"
assert old2 in s
s = s.replace(old2, old2 + """        # the cross-covariances with the reference / values and the factors follow the same linear recursion as
        # beta, so they are formed once from beta (identical result, far less memory than updating them per step)
        rZ = torch.einsum('ctshw,ctwhr->ctshr', bZ, Rs)
        vZ = torch.einsum('ctshw,ctwhugb->ctshugb', bZ, Psv)
        AZ = torch.einsum('ctshw,ctwhj->ctshj', bZ, Asp) if self.K4 else None
""")
s = s.replace("        rZ = torch.zeros_like(Rs); vZ = torch.zeros_like(Psv)\n        AZ = torch.zeros_like(self.pad(As)) if self.K4 else None\n", "")
open('prop.py', 'w').write(s)
print('ok')
