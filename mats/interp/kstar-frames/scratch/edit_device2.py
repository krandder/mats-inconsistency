"""One-off source edit (2026-10-31 ~00:25Z): device support in r2b_analyze.py and r4_mlp.py.
Run once from the lane directory: python3 scratch/edit_device2.py"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def sub(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:70], s.count(old))
    open(path, 'w').write(s.replace(old, new))


p = 'scratch/r2b_analyze.py'
sub(p, "    zA, zB = torch.from_numpy(d['zA']).double(), torch.from_numpy(d['zB']).double()",
    "    zA, zB = torch.from_numpy(d['zA']).double().to(kf.DEV), torch.from_numpy(d['zB']).double().to(kf.DEV)")
sub(p, "    VB, sB, EB = torch.from_numpy(d['VB']).double(), torch.from_numpy(d['sB']).double(), torch.from_numpy(d['EB'])",
    "    VB, sB = torch.from_numpy(d['VB']).double().to(kf.DEV), torch.from_numpy(d['sB']).double().to(kf.DEV)\n"
    "    EB = torch.from_numpy(d['EB']).to(kf.DEV)")
sub(p, "        G = torch.from_numpy(d[nm]).double()", "        G = torch.from_numpy(d[nm]).double().to(kf.DEV)")
sub(p, "    VA = torch.from_numpy(d['VA'][:, :, :k]).double()", "    VA = torch.from_numpy(d['VA'][:, :, :k]).double().to(kf.DEV)")

p = 'scratch/r4_mlp.py'
sub(p, """    z = torch.from_numpy(z).to(f32)
    return (z / z.norm(dim=-1, keepdim=True), (z > 0).to(f32), torch.from_numpy(V).to(f32),
            torch.from_numpy(s).to(f32))""", """    z = torch.from_numpy(z).to(f32).to(kf.DEV)
    return (z / z.norm(dim=-1, keepdim=True), (z > 0).to(f32), torch.from_numpy(V).to(f32).to(kf.DEV),
            torch.from_numpy(s).to(f32).to(kf.DEV))""")
sub(p, "EB = torch.from_numpy(d['EB'])", "EB = torch.from_numpy(d['EB']).to(kf.DEV)")
sub(p, "Gfix = torch.from_numpy(d['GM']).to(f32)", "Gfix = torch.from_numpy(d['GM']).to(f32).to(kf.DEV)")
sub(p, "model = Net()", "model = Net().to(kf.DEV)")
sub(p, "idx_fit = torch.arange(nfit)", "idx_fit = torch.arange(nfit, device=kf.DEV)")
sub(p, "    perm = idx_fit[torch.randperm(nfit)]", "    perm = idx_fit[torch.randperm(nfit, device=kf.DEV)]")
print('edits done')
