"""GPU micro-benchmark of the splitting inner loop (depth 32, 64k particles)."""
import time, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs
torch.backends.cuda.matmul.allow_tf32 = False
net = TNet(100, 32, 0.001, device='cuda')
gen = torch.Generator(device='cuda').manual_seed(0)
B, N = 64, 1000
Z = bgs.draw(net, B, N, gen)
idx = torch.arange(B, device='cuda') % 256
sf = bgs.neuron_scores(net, idx)
def t(f, k=5):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(k): r = f()
    torch.cuda.synchronize(); return (time.time() - t0) / k
print('hidden+score %.4f s' % t(lambda: sf(Z)))
print('draw %.4f s' % t(lambda: bgs.draw(net, B, N, gen)))
print('pcn combine %.4f s' % t(lambda: 0.99 * Z + 0.1 * bgs.draw(net, B, N, gen)))
acc = torch.rand(B, N, device='cuda') > 0.5
ZP = Z.clone()
print('where %.4f s' % t(lambda: torch.where(acc[:, :, None, None], ZP, Z)))
X = torch.randn(B * N, 256, device='cuda')
W = net.W[0]
print('one matmul 64k x 256 x 256: %.5f s -> %.1f TFLOPS' % (t(lambda: X @ W.T, 20), 2 * B * N * 256 * 256 / t(lambda: X @ W.T, 20) / 1e12))
torch.backends.cuda.matmul.allow_tf32 = True
print('TF32 matmul: %.5f s' % t(lambda: X @ W.T, 20))
