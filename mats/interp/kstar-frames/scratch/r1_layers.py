"""Run A (PREREG): frames at every layer for P paths of one net and depth; transports (Q1a), past frame, random frame,
noise ceiling at the main layers; k-table (k*(c) from this run's paths, used by all later runs); window check.
nice -n 19 ionice -c3 python r1_layers.py --n 64 --seed 400 --L 32 [--paths 512 --batch 64 --threads 3]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--L', type=int, default=32); ap.add_argument('--paths', type=int, default=512)
ap.add_argument('--batch', type=int, default=64); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--out', default=os.path.join(HERE, '..', 'runs'))
ap.add_argument('--ktable-only', action='store_true', help='pass 1 only: k*(c), active counts, window angles')
A = ap.parse_args()
torch.set_num_threads(A.threads)
n, L, seed = A.n, A.L, A.seed
net = kf.make_net(seed, n, L)
MAIN = kf.MAIN_LAYERS[L]
TS = (1, 2, 4, 8)
PRED = ['true', 'fwd_1', 'fwd_2', 'fwd_4', 'fwd_8', 'dual_1', 'dual_2', 'dual_4', 'dual_8', 'bwd_1', 'blv', 'rand',
        'noise_1', 'noise_avg4']
tag = f"n{n}_s{seed}_L{L}"
t0 = time.time()
XSEED = 10_000 + 1000 * seed + n + 7 * L   # input/noise stream of run A for this net and depth


def batches():
    g = kf.gen(XSEED)
    for b in range(A.paths // A.batch):
        x = kf.draw_x(A.batch, n, g)
        zs, eps = kf.forward(net, x, g)
        yield b, x, zs, eps


# pass 1: k*(c) from this run's paths
nsurv = np.zeros((A.paths, L + 1)); mact = np.zeros((A.paths, L + 1))
ang1 = np.full((A.paths // 2, L + 1), np.nan)
for b, x, zs, eps in batches():
    D = kf.masks(zs)
    Js = kf.downstream_jacobians(net, D, range(1, L + 1))
    sl = slice(b * A.batch, (b + 1) * A.batch)
    half = slice(b * A.batch // 2, (b + 1) * A.batch // 2)
    for c in range(1, L + 1):
        ang1[half, c] = kf.angle(zs[c - 1][0::2], zs[c - 1][1::2]).cpu().numpy()
    for c in range(1, L + 1):
        s = kf.svdvals(Js[c])
        nsurv[sl, c] = kf.n_surv(s).cpu().numpy()
        mact[sl, c] = D[c - 1].sum(-1).cpu().numpy()
K = {c: int(round(nsurv[:, c].mean())) for c in range(1, L + 1)}
print(f"[{tag}] k*(c): " + " ".join(f"{c}:{K[c]}" for c in range(1, L + 1)), flush=True)
print(f"[{tag}] mean active m_c: " + " ".join(f"{c}:{mact[:, c].mean():.1f}" for c in range(1, L + 1)), flush=True)
if A.ktable_only:
    os.makedirs(A.out, exist_ok=True)
    json.dump({'n': n, 'seed': seed, 'L': L, 'paths': A.paths, 'ktable_only': True, 'k': K,
               'chance': {c: float(np.mean(K[c] / mact[:, c])) for c in range(1, L + 1)},
               'nsurv_mean': {c: float(nsurv[:, c].mean()) for c in range(1, L + 1)},
               'm_mean': {c: float(mact[:, c].mean()) for c in range(1, L + 1)},
               'median_angle': {c: float(np.median(ang1[:, c])) for c in range(1, L + 1)}},
              open(os.path.join(A.out, f'A_{tag}.json'), 'w'), indent=1)
    print(f"[{tag}] k-table only, done {time.time() - t0:.0f}s", flush=True)
    sys.exit(0)

# pass 2: everything else on the same paths
sc = np.full((A.paths, L + 1, len(PRED), 3), np.nan)
ang = np.full((A.paths // 2, L + 1), np.nan)
topt = np.full((A.paths, L + 1), np.nan)
gnoise = kf.gen(XSEED + 555)
grand = kf.gen(XSEED + 777)
for b, x, zs, eps in batches():
    D = kf.masks(zs)
    sl = slice(b * A.batch, (b + 1) * A.batch)
    half = slice(b * A.batch // 2, (b + 1) * A.batch // 2)
    for c in range(1, L + 1):
        ang[half, c] = kf.angle(zs[c - 1][0::2], zs[c - 1][1::2]).cpu().numpy()
    Js = kf.downstream_jacobians(net, D, range(1, L + 1))
    SV = {c: kf.svd_right(Js[c]) for c in range(1, L + 1)}
    Ps = kf.past_products(net, D, range(1, L + 1))
    for c in range(2, L):
        k = K[c]
        s, V = SV[c]
        F = V[:, :, :k]
        E = (s ** 2).sum(-1)
        topt[sl, c] = (1 - (s[:, :k] ** 2).sum(-1) / E).cpu().numpy()
        frames = {'true': F}
        for t in TS:
            if c - t >= 1:
                Ft = SV[c - t][1][:, :, :k]
                frames[f'fwd_{t}'] = kf.push_forward(net, Ft, D, c - t, c)
                frames[f'dual_{t}'] = kf.dual_forward(net, Ft, D, c - t, c)
        frames['bwd_1'] = kf.backward_one(net, SV[c + 1][1][:, :, :k], D, c)
        frames['blv'] = kf.orth(kf.left_frame(D[c - 1][:, :, None] * Ps[c], k))
        frames['rand'] = kf.random_frame(D[c - 1], k, grand)
        if c in MAIN:
            h_c = torch.relu(zs[c - 1])
            Macc = torch.zeros(A.batch, n, n, dtype=kf.DT, device=kf.DEV)
            for r in range(5):
                h0 = h_c + net.s[c - 1] * torch.randn(h_c.shape, generator=gnoise, dtype=kf.DT, device=kf.DEV)
                zs2, _ = kf.forward(net, None, gnoise, start=c, h0=h0)
                D2 = D[:c] + kf.masks(zs2)[c:]
                J2 = kf.downstream_jacobians(net, D2, (c,))[c]
                if r == 0:
                    frames['noise_1'] = kf.svd_right(J2)[1][:, :, :k]
                else:
                    Macc += J2.transpose(-1, -2) @ J2 / (J2 ** 2).sum((-1, -2))[:, None, None]
            ev, evec = torch.linalg.eigh(Macc)
            frames['noise_avg4'] = evec[:, :, -k:]
        for name, G in frames.items():
            O, CS, T = kf.scores(G, V, s, k, E)
            sc[sl, c, PRED.index(name)] = torch.stack([O, CS, T], -1).cpu().numpy()
    print(f"[{tag}] batch {b} done {time.time() - t0:.0f}s", flush=True)

chance = {c: float(np.mean(K[c] / mact[:, c])) for c in range(1, L + 1)}
summ = {'n': n, 'seed': seed, 'L': L, 'paths': A.paths, 'k': K, 'chance': chance,
        'nsurv_mean': {c: float(nsurv[:, c].mean()) for c in range(1, L + 1)},
        'm_mean': {c: float(mact[:, c].mean()) for c in range(1, L + 1)},
        'median_angle': {c: float(np.median(ang[:, c])) for c in range(1, L + 1)},
        'T_true_mean': {c: float(np.nanmean(topt[:, c])) for c in range(2, L)},
        'scores': {name: {c: [float(np.nanmean(sc[:, c, i, j])) for j in range(3)] for c in range(2, L)}
                   for i, name in enumerate(PRED)}}
os.makedirs(A.out, exist_ok=True)
np.savez_compressed(os.path.join(A.out, f'A_{tag}.npz'), sc=sc, ang=ang, topt=topt, nsurv=nsurv, mact=mact,
                    pred=np.array(PRED))
json.dump(summ, open(os.path.join(A.out, f'A_{tag}.json'), 'w'), indent=1)
for c in MAIN:
    print(f"[{tag}] c={c} L'={L-c} k={K[c]} chance={chance[c]:.3f} angle={summ['median_angle'][c]:.3f} | " +
          " ".join(f"{name}: O={summ['scores'][name][c][0]:.3f} CS={summ['scores'][name][c][1]:.3f}" for name in PRED),
          flush=True)
print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)
