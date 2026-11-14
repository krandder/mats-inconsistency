"""Core: AlgZoo 2nd-argmax RNNs, noiseless and noisy (per-step, per-neuron Gaussian noise at a fixed fraction of the
activation scale). Weights are loaded from ext/weights (ARC's public GCS files, state dicts only, weights_only=True).

Noisy model (definition used everywhere in this directory):
  h_0 = 0;  z_{t+1} = W_hh h_t + W_hi x_t;  h_{t+1} = ReLU(z_{t+1}) + sigma * r_{t+1} * eps_{t+1},  eps ~ N(0, I_d)
  logits = W_oh h_n;  prediction = argmax logits;  target = 2nd argmax of x (positions 0..n-1).
  r_{t,i} = RMS of ReLU(z_{t,i}) under the noisy model itself (inputs ~ N(0, I_n), earlier noise on), frozen after training.
  Per-neuron scales make the noisy model equivariant under ARC's neuron-rescaling symmetry (h -> D h, D > 0 diagonal).
"""
import math, os, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, 'ext', 'weights')


def load_arc(d, n, seed=None, example=False):
    """Return (W_hi (d,), W_hh (d,d), W_oh (n,d)) as float64 numpy arrays."""
    if example:
        p = os.path.join(WDIR, 'one_layer_16_hidden_94_acc_2nd_argmax.pth')
    else:
        p = os.path.join(WDIR, f'2nd_argmax_{d}_{n}_1073741824_{seed}.pth')
    sd = torch.load(p, weights_only=True, map_location='cpu')
    W_hi = sd['rnn.weight_ih_l0'].double().numpy()[:, 0]
    W_hh = sd['rnn.weight_hh_l0'].double().numpy()
    W_oh = sd['linear.weight'].double().numpy()
    return W_hi, W_hh, W_oh


BEST = {(2, 2): 2, (4, 3): 0, (16, 10): 2, (32, 10): 0, (8, 10): 4, (16, 5): 2, (8, 5): 3, (4, 4): 1, (16, 8): 1}


def target(x):
    """2nd argmax along last axis (numpy or torch)."""
    if isinstance(x, torch.Tensor):
        return x.argsort(-1)[..., -2]
    return np.argsort(x, -1)[..., -2]


class RNN(torch.nn.Module):
    """Trainable noisy RNN. r is a buffer (n, d); sigma a float. Noise scale during training: EMA of batch RMS."""

    def __init__(self, d, n, W=None, sigma=0.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.n = d, n
        if W is None:  # torch.nn.RNN default init: U(-1/sqrt(d), 1/sqrt(d)); Linear: U(-1/sqrt(d), 1/sqrt(d))
            k = 1 / math.sqrt(d)
            W = ((torch.rand(d, generator=g) * 2 - 1) * k, (torch.rand(d, d, generator=g) * 2 - 1) * k,
                 (torch.rand(n, d, generator=g) * 2 - 1) * k)
            W = [w.double().numpy() for w in W]
        self.W_hi = torch.nn.Parameter(torch.as_tensor(np.asarray(W[0]), dtype=torch.float32).clone())
        self.W_hh = torch.nn.Parameter(torch.as_tensor(np.asarray(W[1]), dtype=torch.float32).clone())
        self.W_oh = torch.nn.Parameter(torch.as_tensor(np.asarray(W[2]), dtype=torch.float32).clone())
        self.sigma = float(sigma)
        self.register_buffer('r', torch.ones(n, d))
        self.ema = 0.99

    def forward(self, x, noise=True, gen=None, update_r=False, return_states=False):
        B = x.shape[0]
        h = torch.zeros(B, self.d, dtype=x.dtype, device=x.device)
        hs = []
        for t in range(self.n):
            a = torch.relu(h @ self.W_hh.T + x[:, t:t + 1] * self.W_hi)
            if update_r:
                with torch.no_grad():
                    rms = a.pow(2).mean(0).sqrt()
                    self.r[t] = self.ema * self.r[t] + (1 - self.ema) * rms
            if noise and self.sigma > 0:
                e = torch.randn(B, self.d, generator=gen, dtype=x.dtype, device=x.device)
                h = a + self.sigma * self.r[t] * e
            else:
                h = a
            if return_states: hs.append(h)
        out = h @ self.W_oh.T
        return (out, hs) if return_states else out

    def numpy_weights(self):
        return (self.W_hi.detach().double().numpy().copy(), self.W_hh.detach().double().numpy().copy(),
                self.W_oh.detach().double().numpy().copy())

    @torch.no_grad()
    def calibrate_r(self, n_samples=2 ** 20, seed=12345, iters=3, bs=2 ** 16):
        """Fixed-point r: RMS of pre-noise activation under the noisy model with the current r (a few iterations)."""
        g = torch.Generator().manual_seed(seed)
        for _ in range(iters if self.sigma > 0 else 1):
            acc = torch.zeros(self.n, self.d, dtype=torch.float64)
            cnt = 0
            for i in range(0, n_samples, bs):
                x = torch.randn(bs, self.n, generator=g)
                h = torch.zeros(bs, self.d)
                for t in range(self.n):
                    a = torch.relu(h @ self.W_hh.T + x[:, t:t + 1] * self.W_hi)
                    acc[t] += a.double().pow(2).sum(0)
                    h = a + self.sigma * self.r[t] * torch.randn(bs, self.d, generator=g) if self.sigma > 0 else a
                cnt += bs
            self.r.copy_((acc / cnt).sqrt().float())
        return self.r.clone()


def np_forward(W, x, sigma=0.0, r=None, rng=None, return_states=False):
    """float64 numpy forward. x (B, n). Returns logits (B, n) [and list of h_t]."""
    W_hi, W_hh, W_oh = W
    B, n = x.shape
    d = W_hi.shape[0]
    h = np.zeros((B, d))
    hs = []
    for t in range(n):
        h = np.maximum(h @ W_hh.T + x[:, t:t + 1] * W_hi, 0.0)
        if sigma > 0:
            h = h + sigma * r[t] * rng.standard_normal((B, d))
        if return_states: hs.append(h)
    lo = h @ W_oh.T
    return (lo, hs) if return_states else lo


@torch.no_grad()
def accuracy(model, n_samples=2 ** 22, seed=777, noise=True, bs=2 ** 16):
    """Single-draw accuracy (noise on if noise=True), with binomial standard error. Also returns mean CE loss."""
    g = torch.Generator().manual_seed(seed)
    correct = 0; tot = 0; ce = 0.0
    for i in range(0, n_samples, bs):
        x = torch.randn(bs, model.n, generator=g)
        lo = model(x, noise=noise, gen=g)
        y = target(x)
        correct += (lo.argmax(-1) == y).sum().item(); tot += bs
        ce += torch.nn.functional.cross_entropy(lo.double(), y, reduction='sum').item()
    p = correct / tot
    return dict(acc=p, se=math.sqrt(p * (1 - p) / tot), n=tot, ce=ce / tot)


def save_model(model, path, meta):
    W = model.numpy_weights()
    np.savez(path, W_hi=W[0], W_hh=W[1], W_oh=W[2], r=model.r.double().numpy(), sigma=model.sigma)
    json.dump(meta, open(path.replace('.npz', '.json'), 'w'), indent=1)


def load_npz(path):
    z = np.load(path)
    return (z['W_hi'], z['W_hh'], z['W_oh']), z['r'], float(z['sigma'])


def model_from(W, r=None, sigma=0.0):
    d = W[0].shape[0]; n = W[2].shape[0]
    m = RNN(d, n, W=W, sigma=sigma)
    if r is not None:
        m.r.copy_(torch.as_tensor(r, dtype=torch.float32))
    return m
