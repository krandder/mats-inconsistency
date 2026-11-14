"""nongauss-family runner for Amendment 1 (copy of scratch/run.py v2; families from lib/ng2.build2, which adds
PBA{k} / PBS{k} and any PB{k}).
Original docstring:
nongauss-family runner (v2, after the design critique). Device-agnostic; on the GPU VM it runs from the staged lane
directory. Phases (each net processed independently, outputs per net):

truth  : Monte Carlo truth (2^NT paths, all layers) -> truth/truth_{n}_{L}_s{sigma}_seed{seed}.npz (skip if present)
e1     : single cuts. For each replicate r, one stream of NFIT true paths is walked through the layers; at each cut c
         every family in --fams is fitted to the paths' z_c and NR family samples are run through the exact noisy
         layers c+1..L. Stored: E[a_L] estimates for every L >= c, scores (nMSE minus truth floor and the family's own
         sampling floor), floors. ATT8 / ATT16 use the same paths at c - 8 / c - 16; ATTX uses net --xref's paths at c.
lin    : single cuts at EVERY layer c in [--lin0, L-1] for --linfams (NR_lin samples), estimates at L only (for the
         linearised sum of per-layer projection errors).
e2     : re-projection chains from --c0 on NC fresh true paths: projection at every layer ('every') and at the cuts in
         --two only ('two'); 'MC' = no projection (plain Monte Carlo on the same NC paths), 'EMP' = bootstrap.
Outputs: {out}/{phase}_{n}_{L}_{seed}.json (scores) and .npz (estimate vectors).
python scratch/run.py PHASE --n 256 --L 64 --nets 113,114 [options]"""
import argparse, json, os, sys, time, zlib
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng, ng2

ap = argparse.ArgumentParser()
ap.add_argument('phase', choices=['truth', 'e1', 'lin', 'e2'])
ap.add_argument('--n', type=int, default=256); ap.add_argument('--L', type=int, default=64)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--nets', required=True)
ap.add_argument('--nt', type=int, default=22)
ap.add_argument('--cuts', default=''); ap.add_argument('--fams', default='')
ap.add_argument('--fams1', default='', help='families run only in replicate 0')
ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--nfit', type=int, default=20); ap.add_argument('--nr', type=int, default=21)
ap.add_argument('--xref', type=int, default=112)
ap.add_argument('--lin0', type=int, default=32); ap.add_argument('--linfams', default='EMP,G,PI')
ap.add_argument('--nrlin', type=int, default=22)
ap.add_argument('--c0', type=int, default=32); ap.add_argument('--two', default='32,48')
ap.add_argument('--cfams', default='MC,EMP,G,PG,PI,PB16'); ap.add_argument('--tfams', default='G,PI,PB16')
ap.add_argument('--ncs', default='20,21')
ap.add_argument('--out', default='runs/main'); ap.add_argument('--truthdir', default='truth')
ap.add_argument('--threads', type=int, default=int(os.environ.get('TH', '3')))
A = ap.parse_args()
torch.set_num_threads(A.threads)
dev = ng.device()
if dev.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
os.chdir(os.path.join(HERE, '..'))
os.makedirs(A.out, exist_ok=True); os.makedirs(A.truthdir, exist_ok=True)
NT = 1 << A.nt
nets = [int(s) for s in A.nets.split(',') if s]
cuts = [int(c) for c in A.cuts.split(',') if c]
fams = [f for f in A.fams.split(',') if f]; fams1 = [f for f in A.fams1.split(',') if f]
for f in fams + fams1:
    assert f in ng.FAMILIES or f[:3] in ('PBA', 'PBS', 'PBW') or (f.startswith('PB') and f[2:].isdigit()), f
print(f'{A.phase} device {dev} n {A.n} L {A.L} nets {nets}', flush=True)


def fseed(*parts):
    return zlib.crc32('/'.join(str(p) for p in parts).encode()) & 0x7fffffff


def load_truth(seed, dn, net):
    tf = f'{A.truthdir}/truth_{A.n}_{A.L}_s{A.sigma:g}_seed{seed}.npz'
    if os.path.exists(tf) and int(np.load(tf)['N']) >= NT:
        d = np.load(tf); Nt = float(d['N'])
        TM = torch.tensor(d['S1'] / Nt, device=dev); TV = torch.tensor(d['S2'] / Nt, device=dev) - TM ** 2
        return TM, TV, Nt
    t0 = time.time()
    M1, M2 = ng.truth_moments(dn, NT, seed=910_000 + seed)
    np.savez(tf, S1=(M1 * NT).cpu().numpy(), S2=(M2 * NT).cpu().numpy(), N=NT, seed=seed, n=A.n, L=A.L,
             sigma=A.sigma, r=net.r.numpy(), device=str(dev))
    print(f'net {seed}: truth {NT} paths {time.time() - t0:.0f}s', flush=True)
    return M1, M2 - M1 ** 2, float(NT)


@torch.no_grad()
def walk(dn, N, seed, stops, fn):
    """Walk N true paths through the layers; call fn(l, Z_l) at every l in stops (Z_l float32 (N, n) of z_l)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(N, dn.n, generator=g, device=dev)
    top = max(stops)
    for l in range(1, top + 1):
        Z = torch.empty(N, dn.n, device=dev)
        for a in range(0, N, ng.CH):
            Z[a:a + ng.CH] = X[a:a + ng.CH] @ dn.W[l - 1].T
        del X
        if l in stops:
            fn(l, Z)
        if l == top:
            break
        X = Z
        for a in range(0, N, ng.CH):
            X[a:a + ng.CH] = torch.relu(Z[a:a + ng.CH]) + dn.s[l - 1] * torch.randn(
                min(ng.CH, N - a), dn.n, generator=g, device=dev)
        del Z


xref_paths = {}
for seed in nets:
    t0 = time.time()
    net = bf.get_net(seed, A.n, A.L, A.sigma)
    dn = ng.DNet(net, dev)
    if A.phase == 'truth':
        load_truth(seed, dn, net)
        continue
    TM, TV, Nt = load_truth(seed, dn, net)
    res = dict(seed=seed, n=A.n, L=A.L, sigma=A.sigma, NT=int(Nt), device=str(dev), phase=A.phase, runs={})
    est = {}
    jf = f'{A.out}/{A.phase}_{A.n}_{A.L}_{seed}.json'

    def save():
        json.dump(res, open(jf, 'w'))
        np.savez_compressed(jf[:-5] + '.npz', TM=TM.cpu().numpy().astype(np.float64),
                            TV=TV.cpu().numpy().astype(np.float64),
                            **{k: v.cpu().numpy().astype(np.float32) for k, v in est.items()})

    if A.phase == 'e1':
        NFIT, NR = 1 << A.nfit, 1 << A.nr
        res.update(NFIT=NFIT, NR=NR, cuts=cuts)
        for rep in range(1 if seed == A.xref else A.reps):
            fl = fams + (fams1 if rep == 0 else [])
            refs = {}
            stops = set(cuts)
            for f in fl:
                if f in ('ATT8', 'ATT16'):
                    stops |= {c - int(f[3:]) for c in cuts}
            stops = sorted(x for x in stops if x >= 1)

            def at(l, Z, rep=rep, fl=fl):
                if any(l == c - 8 for c in cuts) or any(l == c - 16 for c in cuts):
                    refs[l] = Z.clone()
                if seed == A.xref:
                    if rep == 0 and l in cuts:
                        xref_paths[l] = Z.cpu().clone()
                    return                                   # the reference net only provides paths
                if l not in cuts:
                    return
                for f in fl:
                    t2 = time.time()
                    g = torch.Generator(device=dev).manual_seed(fseed(seed, rep, l, f))
                    if f.startswith('ATT'):
                        if f == 'ATTX':
                            if seed == A.xref or l not in xref_paths:
                                continue
                            Xr = xref_paths[l].to(dev)
                        else:
                            Xr = refs.get(l - int(f[3:]))
                            if Xr is None:
                                continue
                        fam = ng.build(f, Z, g, Xref=Xr)
                    else:
                        fam = ng2.build2(f, Z, g, dn, l, A.L)
                    M1, M2 = ng.propagate(dn, fam, NR, l, A.L, g)
                    del fam
                    sc = ng.scores(M1, M2, TM, TV, l, Nt, NR)
                    res['runs'][f'{rep}/{l}/{f}'] = sc
                    est[f'{rep}/{l}/{f}'] = M1
                    e = sc['nmse']
                    show = [l, l + 1, l + 2, l + 4, l + 8, l + 16, A.L]
                    print(f'net {seed} rep {rep} cut {l} {f:5s} ' +
                          ' '.join(f'L{L}:{e[L - l]:.2e}' for L in sorted(set(show)) if L <= A.L) +
                          f'  {time.time() - t2:.1f}s', flush=True)
                save()
                for k in [k for k in refs if k < l - 16]:
                    del refs[k]

            walk(dn, NFIT, 120_000 + 1000 * rep + seed, stops, at)
            del refs
    elif A.phase == 'lin':
        NFIT, NR = 1 << A.nfit, 1 << A.nrlin
        lf = [f for f in A.linfams.split(',') if f]
        res.update(NFIT=NFIT, NR=NR, lin0=A.lin0, linfams=lf)

        def at(l, Z):
            for f in lf:
                t2 = time.time()
                g = torch.Generator(device=dev).manual_seed(fseed(seed, 'lin', l, f))
                fam = ng.build(f, Z, g)
                M1, M2 = ng.propagate(dn, fam, NR, l, A.L, g)
                del fam
                sc = ng.scores(M1[-1:], M2[-1:], TM, TV, A.L, Nt, NR)
                res['runs'][f'{l}/{f}'] = sc
                est[f'{l}/{f}'] = M1[-1]
                print(f'net {seed} lin cut {l} {f:5s} L{A.L}:{sc["nmse"][0]:.2e}  {time.time() - t2:.1f}s', flush=True)
            save()

        walk(dn, NFIT, 130_000 + seed, list(range(A.lin0, A.L)), at)
    elif A.phase == 'e2':
        cf = [f for f in A.cfams.split(',') if f]; tf_ = [f for f in A.tfams.split(',') if f]
        two = set(int(x) for x in A.two.split(',') if x)
        res.update(c0=A.c0, two=sorted(two))
        for lnc in [int(x) for x in A.ncs.split(',') if x]:
            NC = 1 << lnc
            box = {}
            walk(dn, NC, 140_000 + 100 * lnc + seed, [A.c0], lambda l, Z: box.__setitem__('Z', Z))
            for tag, fl, proj in (('every', cf, set(range(A.c0, A.L))), ('two', tf_, two)):
                for f in fl:
                    if tag == 'two' and f in ('MC',):
                        continue
                    t2 = time.time()
                    g = torch.Generator(device=dev).manual_seed(fseed(seed, 'e2', lnc, tag, f))
                    M1, M2, npj = ng.compose(dn, f, box['Z'], A.c0, A.L, proj, NC, g)
                    sc = ng.scores(M1, M2, TM, TV, A.c0, Nt, NC)
                    sc['nproj'] = npj
                    key = f'{lnc}/{tag}/{f}'
                    res['runs'][key] = sc; est[key] = M1
                    e = sc['nmse']
                    show = [A.c0 + 1, A.c0 + 2, A.c0 + 4, A.c0 + 8, A.c0 + 16, A.L]
                    print(f'net {seed} e2 NC 2^{lnc} {tag} {f:5s} ' +
                          ' '.join(f'L{L}:{e[L - A.c0]:.2e}' for L in sorted(set(show)) if L <= A.L) +
                          f'  {time.time() - t2:.1f}s', flush=True)
                    save()
            del box
    save()
    print(f'net {seed} {A.phase} done {time.time() - t0:.0f}s', flush=True)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
