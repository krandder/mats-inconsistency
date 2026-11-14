"""Amendment 2, test T6 (Kelvin via main, 29 Oct ~18:00Z): a family-agnostic learned compressor at the cut.
At cut l of net (n, L): the state z_l -> code c = E(w) in R^k (w = z_l / |z_l|; the radius |z_l| is kept exactly, by
positive homogeneity) -> w_hat = D(c) -> z_hat = |z_l| w_hat / |w_hat|. The true noisy net runs from z_hat and from
z_l with the SAME noise draws to the output L; the loss is the pathwise output error
    E || a_L(z_hat) - a_L(z_l) ||^2 / (n mean_j Var a_L,j)     ("pathwise nMSE"),
trained end to end through the true layers (fresh true paths every step). Models:
- 'mlp': E = MLP(n -> 2n -> 2n -> k), D = MLP(k -> 2n -> 2n -> n), SiLU, inputs centred and scaled by the direction
  law's mean and RMS spread;
- 'lin': E, D linear (a trained linear bottleneck);
- 'pca': no training, projection on the top-k principal components of the direction law.
Evaluation on 2^17 held-out paths (shared noise): pathwise nMSE; the channel floor (same z_l, independent noise,
i.e. what the activation noise alone does to the output); and the induced bias of the mean, ||mean(a_L(z_hat) -
a_L(z_l))||^2 / (n mean Var) (paired), with its noise level pathwise / N; and the same bias for the stochastic-residual
version (code of path i plus the reconstruction residual w_j - w_hat_j of an independent path j, normalised and scaled
by |z_i|), a learned-code analogue of the PB families.
python scratch/comp.py --n 256 --L 64 --net 113 --cut 48 --ks 1,4,16,64 --models mlp,lin,pca [--steps 3000]"""
import argparse, json, math, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=256); ap.add_argument('--L', type=int, default=64)
ap.add_argument('--net', type=int, required=True); ap.add_argument('--cut', type=int, required=True)
ap.add_argument('--ks', default='1,4,16,64'); ap.add_argument('--models', default='mlp,lin,pca')
ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--neval', type=int, default=17)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--out', default='runs/a2')
ap.add_argument('--threads', type=int, default=int(os.environ.get('TH', '3')))
A = ap.parse_args()
torch.set_num_threads(A.threads)
dev = ng.device()
if dev.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
os.chdir(os.path.join(HERE, '..')); os.makedirs(A.out, exist_ok=True)
net = bf.get_net(A.net, A.n, A.L, A.sigma); dn = ng.DNet(net, dev)
l, L, n = A.cut, A.L, A.n


def true_z(B, g):
    with torch.no_grad():
        h = torch.randn(B, n, generator=g, device=dev)
        for k in range(1, l + 1):
            z = h @ dn.W[k - 1].T
            if k < l:
                h = torch.relu(z) + dn.s[k - 1] * torch.randn(B, n, generator=g, device=dev)
    return z


def downstream(z, noise):
    """a_L from z_l with given noise list (noise[i] for the layer l + i)."""
    h = torch.relu(z)
    for i, k in enumerate(range(l + 1, L + 1)):
        h = h + dn.s[k - 2] * noise[i]
        h = torch.relu(h @ dn.W[k - 1].T)
    return h


def draw_noise(B, g):
    return [torch.randn(B, n, generator=g, device=dev) for _ in range(l + 1, L + 1)]


# direction-law statistics (unweighted; used only for input / output scaling and the PCA baseline)
g0 = torch.Generator(device=dev).manual_seed(190_000 + A.net)
Z0 = torch.cat([true_z(1 << 14, g0) for _ in range(8)])
W0 = Z0 / Z0.norm(dim=1, keepdim=True)
mw, Cw = ng.mean_cov(W0); evw, Vw = ng.eig_desc(Cw)
mw = mw.float(); scale = float(evw.sum().sqrt() / math.sqrt(n))
with torch.no_grad():
    nz = draw_noise(1 << 14, g0)
    aL = downstream(Z0[:1 << 14], nz)
    VAR = float(aL.double().var(0).mean())                      # mean_j Var(a_L,j) (normalisation)
    del nz, aL


class MLP(torch.nn.Module):
    def __init__(self, a, b, hidden):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(a, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                     torch.nn.SiLU(), torch.nn.Linear(hidden, b))

    def forward(self, x):
        return self.f(x)


class Comp(torch.nn.Module):
    def __init__(self, kind, k):
        super().__init__()
        self.kind = kind
        if kind == 'mlp':
            self.E = MLP(n, k, 2 * n); self.D = MLP(k, n, 2 * n)
        elif kind == 'lin':
            self.E = torch.nn.Linear(n, k); self.D = torch.nn.Linear(k, n)
        else:
            self.register_buffer('V', Vw[:, :k].float())

    def forward(self, z):
        rho = z.norm(dim=1, keepdim=True); w = z / rho
        x = (w - mw) / scale
        if self.kind == 'pca':
            xh = (x @ self.V) @ self.V.T
        else:
            xh = self.D(self.E(x))
        wh = mw + scale * xh
        return rho * wh / wh.norm(dim=1, keepdim=True)


out = dict(n=n, L=L, net=A.net, cut=l, VAR=VAR, steps=A.steps, batch=A.batch, runs={})
gev = torch.Generator(device=dev).manual_seed(200_000 + A.net)
NE = 1 << A.neval
# channel floor: same z, two independent noise draws
with torch.no_grad():
    fl = 0.0
    for a in range(0, NE, 8192):
        z = true_z(8192, gev)
        d = downstream(z, draw_noise(8192, gev)) - downstream(z, draw_noise(8192, gev))
        fl += float((d.double() ** 2).sum())
    out['channel_floor'] = fl / (NE * n * VAR) / 2           # per single draw: Var of a_L given z_l, normalised
print(f'n {n} L {L} net {A.net} cut {l}: VAR {VAR:.3e}, channel floor (pathwise nMSE of the noise alone) '
      f'{out["channel_floor"]:.3e}', flush=True)
for kind in A.models.split(','):
    for k in [int(x) for x in A.ks.split(',')]:
        t0 = time.time()
        torch.manual_seed(1000 * k + len(kind))
        model = Comp(kind, k).to(dev)
        g = torch.Generator(device=dev).manual_seed(210_000 + 13 * k + A.net)
        hist = []
        if kind != 'pca':
            opt = torch.optim.Adam(model.parameters(), lr=A.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.steps, eta_min=A.lr / 100)
            for step in range(A.steps):
                z = true_z(A.batch, g)
                nz = draw_noise(A.batch, g)
                with torch.no_grad():
                    target = downstream(z, nz)
                pred = downstream(model(z), nz)
                loss = ((pred - target) ** 2).mean() / VAR
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                if step % 250 == 0 or step == A.steps - 1:
                    hist.append((step, float(loss)))
        # evaluation on held-out paths (fresh generator, shared noise)
        ge = torch.Generator(device=dev).manual_seed(220_000 + A.net)
        se, sd, ss = 0.0, torch.zeros(n, dtype=ng.F64, device=dev), torch.zeros(n, dtype=ng.F64, device=dev)
        with torch.no_grad():
            for a in range(0, NE, 8192):
                z = true_z(8192, ge); nz = draw_noise(8192, ge)
                zh = model(z)
                base = downstream(z, nz)
                d = (downstream(zh, nz) - base).double()
                se += float((d ** 2).sum()); sd += d.sum(0)
                # stochastic-residual version: code of path i + reconstruction residual of an independent path j
                rho = z.norm(dim=1, keepdim=True); w = z / rho; wh = zh / rho
                perm = torch.randperm(8192, generator=ge, device=dev)
                ws = wh + (w - wh)[perm]
                zs = rho * ws / ws.norm(dim=1, keepdim=True)
                ss += (downstream(zs, nz) - base).double().sum(0)
        pw = se / (NE * n * VAR)
        bias = float((sd / NE).pow(2).mean()) / VAR
        bias_s = float((ss / NE).pow(2).mean()) / VAR
        rec = dict(pathwise=pw, bias=bias, bias_shuffled_residual=bias_s, bias_noise=pw / NE, hist=hist,
                   secs=time.time() - t0)
        out['runs'][f'{kind}/{k}'] = rec
        print(f'  {kind:3s} k {k:3d}: pathwise nMSE {pw:.3e} (x floor {pw / out["channel_floor"]:.1f}); mean bias '
              f'{bias:.2e}, with shuffled residual {bias_s:.2e} (noise ~{pw / NE:.1e}); last train loss {hist[-1][1] if hist else float("nan"):.3e}  '
              f'{time.time() - t0:.0f}s', flush=True)
        json.dump(out, open(f'{A.out}/comp_{n}_{L}_{A.net}_c{l}.json', 'w'), indent=1)
print('done', flush=True)
