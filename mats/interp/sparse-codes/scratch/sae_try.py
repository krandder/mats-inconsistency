"""Dev-net SAE smoke/timing (seed 64). usage: sae_try.py FAMILY N SIGMA LAYER FMULT EPOCHS"""
import math, sys, os, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nets, sae
torch.set_num_threads(int(os.environ.get('NT', '4')))
fam, n, sigma, l, fm, ep = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
net = nets.SNet(fam, n, 64, sigma)
g = torch.Generator().manual_seed(4242)
N = 1 << 16
X = net.sample(N + (1 << 13), g, [l])[l]
Xtr, Xte = X[:N], X[N:]
s = net.noise_sd(l)
ks = [n // 16, n // 8, n // 4, n // 2]
t0 = time.time()
model, c, hist = sae.train_batched(Xtr, fm * n, ks, epochs=ep, batch=1024, log=print)
print(f"train {time.time()-t0:.0f}s")
for r in sae.evaluate(model, c, Xte, s):
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
