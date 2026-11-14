"""E1 (single cut) and E2 (re-projection) for the nongauss-family lane. Device-agnostic (CUDA if present).
For each net: truth (load truth/truth_{n}_{L}_s{sigma}_seed{seed}.npz if it has >= 2^NT paths, else compute and save),
one fit set of NFIT true paths of z_l at every needed layer, then
  E1: for each cut c and family F: fit F to the fit set at c, draw NR samples, run the exact noisy layers c+1..LMAX,
      nMSE of E[a_L] for every L = c..LMAX (minus 1/N_truth + 1/NR); the error vectors at LMAX are saved too;
  E2: (if --compose) re-projection from layer C0 at every layer, and at the cuts in --two only.
python scratch/e1.py --nets 113 --cuts 16,32,48 --fams EMP,G,... [--nfit 20 --nr 21 --nt 24 --lmax 64 --n 256]
       [--compose EMP,G,PG,PI --c0 32 --nc 20 --two 32,48] --out runs/e1"""
import argparse, json, os, sys, time, zlib
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng

ap = argparse.ArgumentParser()
ap.add_argument('--nets', required=True); ap.add_argument('--cuts', default='16,32,48')
ap.add_argument('--fams', default=','.join(ng.FAMILIES)); ap.add_argument('--nfit', type=int, default=20)
ap.add_argument('--nr', type=int, default=21); ap.add_argument('--nt', type=int, default=24)
ap.add_argument('--lmax', type=int, default=64); ap.add_argument('--n', type=int, default=256)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--out', default='runs/e1')
ap.add_argument('--compose', default=''); ap.add_argument('--c0', type=int, default=32)
ap.add_argument('--nc', type=int, default=20); ap.add_argument('--two', default='32,48')
ap.add_argument('--truthdir', default='truth')
ap.add_argument('--threads', type=int, default=int(os.environ.get('TH', '3')))
A = ap.parse_args()
torch.set_num_threads(A.threads)
dev = ng.device()
if dev.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
os.chdir(os.path.join(HERE, '..'))
os.makedirs(A.out, exist_ok=True); os.makedirs(A.truthdir, exist_ok=True)
cuts = [int(c) for c in A.cuts.split(',') if c]
fams = [f for f in A.fams.split(',') if f]
comp = [f for f in A.compose.split(',') if f]
NFIT, NR, NT, NC = 1 << A.nfit, 1 << A.nr, 1 << A.nt, 1 << A.nc
print(f'device {dev}  nets {A.nets}  cuts {cuts}  fams {fams}  compose {comp}', flush=True)


def fseed(*parts):
    return zlib.crc32('/'.join(str(p) for p in parts).encode()) & 0x7fffffff


for seed in [int(s) for s in A.nets.split(',')]:
    t0 = time.time()
    net = bf.get_net(seed, A.n, A.lmax, A.sigma)
    dn = ng.DNet(net, dev)
    tf = f'{A.truthdir}/truth_{A.n}_{A.lmax}_s{A.sigma:g}_seed{seed}.npz'
    if os.path.exists(tf) and int(np.load(tf)['N']) >= NT:
        d = np.load(tf); Nt = float(d['N'])
        TM = torch.tensor(d['S1'] / Nt, device=dev); TV = torch.tensor(d['S2'] / Nt, device=dev) - TM ** 2
        print(f'net {seed}: truth loaded ({int(Nt)} paths)', flush=True)
    else:
        M1, M2 = ng.truth_moments(dn, NT, seed=910_000 + seed)
        Nt = float(NT)
        np.savez(tf, S1=(M1 * Nt).cpu().numpy(), S2=(M2 * Nt).cpu().numpy(), N=NT, seed=seed, n=A.n, L=A.lmax,
                 sigma=A.sigma, r=net.r.numpy(), device=str(dev))
        TM = M1; TV = M2 - M1 ** 2
        print(f'net {seed}: truth computed ({NT} paths) {time.time() - t0:.0f}s', flush=True)
    need = set(cuts)
    for f in fams:
        if f.startswith('ATT'):
            need |= {c - int(f[3:]) for c in cuts}
    if comp:
        need.add(A.c0)
    need = sorted(x for x in need if x >= 2)
    t1 = time.time()
    X = ng.true_paths(dn, NFIT, need, seed=120_000 + seed)
    print(f'net {seed}: fit set {NFIT} paths at layers {need} {time.time() - t1:.0f}s', flush=True)
    res = dict(seed=seed, n=A.n, L=A.lmax, sigma=A.sigma, NFIT=NFIT, NR=NR, NT=int(Nt), device=str(dev), e1={}, e2={})
    errvec = {}
    for c in cuts:
        for f in fams:
            t2 = time.time()
            g = torch.Generator(device=dev).manual_seed(fseed(seed, c, f))
            ref = X.get(c - int(f[3:])) if f.startswith('ATT') else None
            try:
                fam = ng.build(f, X[c], g, Xref=ref)
            except ValueError as e:
                print(f'net {seed} cut {c} {f}: skipped ({e})', flush=True)
                continue
            est = ng.propagate(dn, fam, NR, c, A.lmax, g)
            del fam
            errs = ng.nmse_layers(est, TM, TV, c, Nt, NR)
            res['e1'][f'{c}/{f}'] = errs
            errvec[(c, f)] = (est[-1] - TM[A.lmax - 1]).cpu()
            show = [c, c + 1, c + 2, c + 4, c + 8, c + 16, A.lmax]
            print(f'net {seed} cut {c} {f:5s} ' + ' '.join(f'L{L}:{errs[L - c]:.2e}' for L in show if L <= A.lmax)
                  + f'  {time.time() - t2:.0f}s', flush=True)
        json.dump(res, open(f'{A.out}/e1_{A.n}_{A.lmax}_{seed}.json', 'w'))
    # cosine between the output error vectors of different cuts (coherence of single-cut errors)
    res['cos'] = {}
    for f in fams:
        for c1 in cuts:
            for c2 in cuts:
                if c1 < c2 and (c1, f) in errvec and (c2, f) in errvec:
                    a, b = errvec[(c1, f)], errvec[(c2, f)]
                    res['cos'][f'{c1}-{c2}/{f}'] = float((a @ b) / (a.norm() * b.norm()))
    if res['cos']:
        print(f'net {seed} cosines of output error vectors: ' + json.dumps({k: round(v, 3) for k, v in res['cos'].items()}),
              flush=True)
    two = [int(x) for x in A.two.split(',') if x]
    for f in comp:
        for tag, proj in (('every', set(range(A.c0, A.lmax))), ('two', set(two))):
            t2 = time.time()
            g = torch.Generator(device=dev).manual_seed(fseed(seed, 'compose', tag, f))
            est, npj = ng.compose(dn, f, X[A.c0], A.c0, A.lmax, proj, NC, g)
            errs = ng.nmse_layers(est, TM, TV, A.c0, Nt, NC)
            res['e2'][f'{tag}/{f}'] = dict(err=errs, nproj=npj)
            show = [A.c0 + 1, A.c0 + 2, A.c0 + 4, A.c0 + 8, A.c0 + 16, A.lmax]
            print(f'net {seed} compose-{tag} {f:5s} ' + ' '.join(f'L{L}:{errs[L - A.c0]:.2e}' for L in show if L <= A.lmax)
                  + f'  {time.time() - t2:.0f}s', flush=True)
            json.dump(res, open(f'{A.out}/e1_{A.n}_{A.lmax}_{seed}.json', 'w'))
    json.dump(res, open(f'{A.out}/e1_{A.n}_{A.lmax}_{seed}.json', 'w'))
    print(f'net {seed} done {time.time() - t0:.0f}s', flush=True)
    del X
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
