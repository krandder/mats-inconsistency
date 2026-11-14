"""E20 minimal model (PREREG-B amendment 27): one source's reliability from its four labelled errors, with the transformer's
layer-1 constraint isolated.
Episode: v ~ uniform{0.1, 0.5, 2.0}; y_q ~ N(0, 1); r_q ~ N(0, v); o_q = y_q + r_q (q = 1..4); a source identity vector (shared
by the four tokens) and a question identity vector per token, each N(0, I/16) in R^16, as in the PFN's tokens.
Token features: [source id (16), question id (16), o, y, 1] (35).
Target: the class of v (3-way cross-entropy); the Bayes posterior depends on the errors only through SS = sum r_q^2.
Models (all end in the same readout MLP d -> 64 -> 3, GELU):
  ln    tokens -> linear (d = 32) -> LayerNorm -> attention pool (one learned query, softmax over the 4 tokens) -> readout
        (the transformer's layer-1 situation: the only per-token nonlinearity before pooling is LayerNorm);
  pre   tokens -> linear -> per-token MLP (residual, hidden 64) -> LayerNorm -> attention pool -> readout;
  ss    the oracle: log SS -> readout (no departure by construction).
  lnmean as ln but pooled by the plain mean over the 4 tokens (amendment 27 addendum 1: no error-dependent attention weights).
Optional noise sigma: N(0, sigma^2) added to the normalised tokens (after LayerNorm), as in the noisy nets' channels.
Training: AdamW (lr 1e-3, weight decay 0.01), batch 512, fresh episodes, STEPS steps, gradient clipping 1.
usage: python minimal.py MODEL SEED [--sigma S] [--steps N] -> results/e20/MODEL_s{SEED}_sig{S}.pt and .json
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
VLEV = np.array([0.1, 0.5, 2.0])


def sample(n, rng):
    k = rng.integers(0, 3, n); v = VLEV[k]
    y = rng.standard_normal((n, 4)); r = rng.standard_normal((n, 4)) * np.sqrt(v)[:, None]
    sid = rng.standard_normal((n, 1, 16)) / 4; qid = rng.standard_normal((n, 4, 16)) / 4
    return k, y, r, sid, qid


def tokens(y, r, sid, qid):
    n = len(y)
    X = np.concatenate([np.repeat(sid, 4, 1), qid, (y + r)[:, :, None], y[:, :, None], np.ones((n, 4, 1))], 2)
    return torch.as_tensor(X, dtype=torch.float32)


def bayes_post(r):
    ss = (r ** 2).sum(-1)
    ll = -0.5 * ss[:, None] / VLEV[None] - 2.0 * np.log(VLEV)[None]
    ll -= ll.max(1, keepdims=True); p = np.exp(ll); return p / p.sum(1, keepdims=True)


class Net(nn.Module):
    def __init__(self, kind, d=32, sigma=0.0):
        super().__init__()
        self.kind, self.sigma = kind, sigma
        self.read = nn.Sequential(nn.Linear(d if kind != 'ss' else 1, 64), nn.GELU(), nn.Linear(64, 3))
        if kind != 'ss':
            self.inp = nn.Linear(35, d); self.ln = nn.LayerNorm(d); self.q = nn.Parameter(torch.zeros(d)); self.k = nn.Linear(d, d)
            if kind == 'pre':
                self.mlp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))

    def tok(self, X):
        h = self.inp(X)
        if self.kind == 'pre':
            h = h + self.mlp(h)
        h = self.ln(h)
        if self.sigma > 0:
            h = h + self.sigma * torch.randn_like(h)
        return h

    def forward(self, X=None, logss=None):
        if self.kind == 'ss':
            return self.read(logss[:, None])
        h = self.tok(X)
        if self.kind == 'lnmean':
            return self.read(h.mean(1))
        a = torch.softmax(self.k(h) @ self.q / math.sqrt(h.shape[-1]), 1)        # (n, 4)
        return self.read((a[:, :, None] * h).sum(1))


def run(kind, seed, sigma, steps, out):
    torch.manual_seed(seed); rng = np.random.default_rng([seed, 2027]); net = Net(kind, sigma=sigma)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    vr = np.random.default_rng(99); vk, vy, vrr, vs, vq = sample(20000, vr); vX = tokens(vy, vrr, vs, vq)
    vlog = torch.as_tensor(np.log((vrr ** 2).sum(1)), dtype=torch.float32); vb = bayes_post(vrr)
    vbayes = float(-np.log(vb[np.arange(len(vk)), vk]).mean())
    curve, t0 = [], time.time()
    for st in range(1, steps + 1):
        k, y, r, s, q = sample(512, rng)
        logit = net(tokens(y, r, s, q), torch.as_tensor(np.log((r ** 2).sum(1)), dtype=torch.float32))
        loss = nn.functional.cross_entropy(logit, torch.as_tensor(k))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if st % 1000 == 0 or st == steps:
            net.eval()
            with torch.no_grad():
                lv = nn.functional.cross_entropy(net(vX, vlog), torch.as_tensor(vk)).item()
            net.train(); curve.append((st, lv - vbayes))
    torch.save(net.state_dict(), out + '.pt')
    json.dump(dict(kind=kind, seed=seed, sigma=sigma, steps=steps, val_gap=curve[-1][1], curve=curve, seconds=time.time() - t0),
              open(out + '.json', 'w'))
    print(kind, seed, sigma, 'final val gap %.4f' % curve[-1][1], '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['ln', 'pre', 'ss', 'lnmean']); ap.add_argument('seed', type=int)
    ap.add_argument('--sigma', type=float, default=0.0); ap.add_argument('--steps', type=int, default=30000)
    a = ap.parse_args(); os.makedirs('results/e20', exist_ok=True)
    run(a.kind, a.seed, a.sigma, a.steps, f'results/e20/{a.kind}_s{a.seed}_sig{a.sigma}')
