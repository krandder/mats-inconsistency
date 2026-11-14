"""Run B data (PREREG): at the main layers, N_A reference/training paths and N_B held-out paths with their states z_c,
top-(k+16) right singular vectors and values of J_c, total energy |J_c|_F^2; the marginal activation probabilities
p_l (4096 separate paths) and the marginal-gate frame; the optimal fixed frame (top-k of Mt = mean over A of
J^T J / sum_{i<=k} s_i^2). k*(c) from run A (runs/A_<tag>.json).
nice -n 19 ionice -c3 python r2_big.py --n 64 --seed 400 --L 32 [--NA 16384 --NB 2048]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--L', type=int, default=32); ap.add_argument('--NA', type=int, default=0)
ap.add_argument('--NB', type=int, default=2048); ap.add_argument('--batch', type=int, default=0)
ap.add_argument('--threads', type=int, default=3); ap.add_argument('--out', default=os.path.join(HERE, '..', 'runs'))
A = ap.parse_args()
torch.set_num_threads(A.threads)
n, L, seed = A.n, A.L, A.seed
NA = A.NA or (8192 if n >= 128 else 16384)
bs = A.batch or (64 if n >= 128 else 256)
net = kf.make_net(seed, n, L)
MAIN = kf.MAIN_LAYERS[L]
tag = f"n{n}_s{seed}_L{L}"
K = {int(c): v for c, v in json.load(open(os.path.join(A.out, f'A_{tag}.json')))['k'].items()}
KK = {c: min(n, K[c] + 16) for c in MAIN}
t0 = time.time()

# marginal activation probabilities and the marginal-gate frame
g = kf.gen(30_000 + 1000 * seed + n + 7 * L)
cnt = [torch.zeros(n, dtype=kf.DT, device=kf.DEV) for _ in range(L)]
for _ in range(4096 // 512):
    zs, _ = kf.forward(net, kf.draw_x(512, n, g), g)
    for l in range(L):
        cnt[l] += (zs[l] > 0).to(kf.DT).sum(0)
p = [c_ / 4096.0 for c_ in cnt]
Gb = {}
for c in MAIN:
    J = torch.diag(p[L - 1])
    for l in range(L - 1, c - 1, -1):                 # J <- J W_{l+1} diag(p_l), down to layer c
        J = (J @ net.W[l]) * p[l - 1][None, :]
    _, s, Vh = torch.linalg.svd(J)
    Gb[c] = Vh[:K[c]].T.contiguous()                  # (n, k)


def run_split(N, gseed, accumulate):
    g = kf.gen(gseed)
    st = {c: {'z': np.zeros((N, n), np.float32), 'V': np.zeros((N, n, KK[c]), np.float32),
              's': np.zeros((N, KK[c]), np.float32), 'E': np.zeros(N), 'trunc': np.zeros(N)} for c in MAIN}
    Mt = {c: torch.zeros(n, n, dtype=kf.DT, device=kf.DEV) for c in MAIN}
    for b in range(N // bs):
        x = kf.draw_x(bs, n, g)
        zs, _ = kf.forward(net, x, g)
        D = kf.masks(zs)
        Js = kf.downstream_jacobians(net, D, MAIN)
        sl = slice(b * bs, (b + 1) * bs)
        for c in MAIN:
            s, V = kf.svd_right(Js[c])
            k, kk = K[c], KK[c]
            s2 = s ** 2
            E = s2.sum(-1)
            top = s2[:, :k].sum(-1)
            st[c]['z'][sl] = zs[c - 1].cpu().numpy()
            st[c]['V'][sl] = V[:, :, :kk].cpu().numpy()
            st[c]['s'][sl] = s[:, :kk].cpu().numpy()
            st[c]['E'][sl] = E.cpu().numpy()
            st[c]['trunc'][sl] = ((E - s2[:, :kk].sum(-1)) / top).cpu().numpy()
            if accumulate:
                Mt[c] += (Js[c].transpose(-1, -2) @ Js[c] / top[:, None, None]).sum(0)
        if b % 16 == 0:
            print(f"[{tag}] split seed {gseed} batch {b}/{N // bs} {time.time() - t0:.0f}s", flush=True)
    return st, Mt


stA, MtA = run_split(NA, 40_000 + 1000 * seed + n + 7 * L, True)
stB, _ = run_split(A.NB, 50_000 + 1000 * seed + n + 7 * L, False)
os.makedirs(A.out, exist_ok=True)
for c in MAIN:
    ev, evec = torch.linalg.eigh(MtA[c] / NA)
    GM = evec[:, -K[c]:].flip(-1).contiguous()
    np.savez(os.path.join(A.out, f'B_{tag}_c{c}.npz'), k=K[c], kk=KK[c], NA=NA, NB=A.NB,
             zA=stA[c]['z'], VA=stA[c]['V'], sA=stA[c]['s'], EA=stA[c]['E'], truncA=stA[c]['trunc'],
             zB=stB[c]['z'], VB=stB[c]['V'], sB=stB[c]['s'], EB=stB[c]['E'], truncB=stB[c]['trunc'],
             Gb=Gb[c].cpu().numpy(), GM=GM.cpu().numpy(), Mt_evals=ev.flip(-1).cpu().numpy(), p=np.stack([q.cpu().numpy() for q in p]))
    print(f"[{tag}] c={c} k={K[c]} saved; max truncation bound A {stA[c]['trunc'].max():.2e} B {stB[c]['trunc'].max():.2e}",
          flush=True)
print(f"[{tag}] done {time.time() - t0:.0f}s", flush=True)
