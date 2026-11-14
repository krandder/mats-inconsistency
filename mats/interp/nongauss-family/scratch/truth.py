"""Monte Carlo truth for the nongauss-family lane: E[a_l] and E[a_l^2] per neuron at every layer l = 1..L of an ARC
He-ReLU net (x ~ N(0, I), W iid N(0, 2/n), noise s_l = sigma r_l after each ReLU, r_l from anet's seeded pilot).
float32 matmuls, float64 accumulation. Saves truth/truth_{n}_{L}_s{sigma}_seed{seed}.npz with S1 (L, n), S2 (L, n), N.
python scratch/truth.py SEED N_LOG2 [n L sigma]"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf

seed = int(sys.argv[1]); N = 1 << int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
L = int(sys.argv[4]) if len(sys.argv) > 4 else 64
sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001
net = bf.get_net(seed, n, L, sigma)
W = [w.float() for w in net.W]; s = [float(v) for v in net.s]
g = torch.Generator().manual_seed(900_000 + 17 * seed + n)
S1 = torch.zeros(L, n, dtype=torch.float64); S2 = torch.zeros(L, n, dtype=torch.float64)
chunk = 1 << 16; done = 0; t0 = time.time()
out = f'truth/truth_{n}_{L}_s{sigma:g}_seed{seed}.npz'
with torch.no_grad():
    while done < N:
        b = min(chunk, N - done)
        h = torch.randn(b, n, generator=g)
        for l in range(L):
            a = torch.relu(h @ W[l].T)
            ad = a.double(); S1[l] += ad.sum(0); S2[l] += (ad * ad).sum(0)
            h = a + s[l] * torch.randn(b, n, generator=g) if l < L - 1 else a
        done += b
        if done % (1 << 20) == 0 or done == N:
            np.savez(out + '.tmp.npz', S1=S1.numpy(), S2=S2.numpy(), N=done, seed=seed, n=n, L=L, sigma=sigma,
                     r=net.r.numpy())
            os.replace(out + '.tmp.npz', out)
            print(f'{done} paths {time.time() - t0:.0f}s', flush=True)
print('done', flush=True)
