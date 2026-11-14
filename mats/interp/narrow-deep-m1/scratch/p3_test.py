"""PREREG P3 on evaluation nets: the deterministic downstream (noiseless middle c -> c_d, D0 deep part) applied to the
exact (sampled) law at c = 8, without projection (Amendment 2's direct meet) and with projection onto the top-k
probe directions (k = 3, 5; dropped part by the law's linear regression), nMSE against the truth; plus the downstream's
cost per evaluated state. usage: python p3_test.py SEED SIGMA TRUTHJSON OUTJSON"""
import sys, json, torch, math
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf
torch.set_num_threads(1)
DT = torch.float64
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
T = json.load(open(TJ))
net = nd.Net('plain', SEED, 8, 256, SIG, r=T['r']); n, L = 8, 256
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']; fp = net.flops_path()
g = torch.Generator().manual_seed(29)
P = 1 << 17
res = dict(seed=SEED, sigma=SIG, rows=[])
h = torch.randn(P, n, generator=g, dtype=DT)
for c in (8,):
    for l in range(c):
        a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
    for c_d in (32, 64):
        fl = bvf.Flops()
        cand = bvf.noiseless_pass(net, h[:256], c, c_d, fl)
        deep = bvf.Deep(net, c_d, cand, torch.ones(256, dtype=DT), fl)
        f_setup = fl.n
        fq = bvf.Flops()
        V = deep.value(bvf.noiseless_pass(net, h, c, c_d, fq), fq)
        e = float((V.mean(0) - truth).pow(2).sum() / trv)
        row = dict(c=c, c_d=c_d, nmse_direct=e, setup_paths=f_setup / fp, per_state_paths=fq.n / P / fp,
                   class_share=deep.class_share)
        mh = h.mean(0); M2 = h.T @ h / P; Ch = M2 - torch.outer(mh, mh)
        pr, pw = bvf.unscented(mh, Ch)
        U8, eS = bvf.relevant_directions(net, c, c_d, deep, pr, pw.abs() + 1e-3, None, Ch, n, bvf.Flops())
        for k in (3, 5):
            U = U8[:, :k]; t = h @ U
            X = torch.cat([t, torch.ones(P, 1, dtype=DT)], 1)
            B = torch.linalg.lstsq(X, h - t @ U.T).solution
            hr = t @ U.T + X @ B
            Vp = deep.value(bvf.noiseless_pass(net, hr, c, c_d, bvf.Flops()), bvf.Flops())
            row[f'nmse_proj_k{k}'] = float((Vp.mean(0) - truth).pow(2).sum() / trv)
        res['rows'].append(row)
        print(f"s{SEED} c={c} c_d={c_d}: direct downstream on the exact law nMSE {e:.3g}; projected k=3 {row['nmse_proj_k3']:.3g}, "
              f"k=5 {row['nmse_proj_k5']:.3g}; setup {row['setup_paths']:.0f} paths, {row['per_state_paths']:.3f} paths/state, "
              f"ref class share {deep.class_share:.3f}", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
