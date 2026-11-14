"""Question-driven DAG over (neuron, step) for a noisy 2nd-argmax RNN.

Edges: rec (t, i, j): h_{t,j} -> z_{t+1,i} for t = 1..n-1;  inp (t, i): x_t -> z_{t+1,i} for t = 0..n-1;
       out (i): h_{n,i} -> logits.  An ablated edge is replaced by its mean contribution (W_ij * E h_{t,j}; 0 for x).
A surrogate = the network with a set of edges ablated, run on the same inputs and the same noise (common random numbers).

Question divergence (the question = "is the output the 2nd argmax?"):
  flip = P(error indicator of surrogate != error indicator of network), coupled;  dp = p_sur - p_net.
Forward-only cost of the same surrogate: the path KL by the chain rule,
  KL(net path || sur path) = sum_{t,i} E_net (f_{t,i}(h) - g_{t,i}(h))^2 / (2 sigma^2 r_{t,i}^2)   [nats],
  f, g = network / surrogate pre-noise activation given the network's previous state.
Saliency (first order): E[ k(m) (dm/dz_{t+1,i} * W_ij * (h_{t,j} - mu_{t,j}))^2 ], m = target-logit margin,
  k(m) = exp(-m^2 / (2 s^2)) with s = 0.1 x sd(m): weight on inputs near the decision boundary.
"""
import numpy as np


class Net:
    def __init__(self, W, r, sigma):
        self.W_hi, self.W_hh, self.W_oh = [np.asarray(w, float) for w in W]
        self.d = self.W_hi.shape[0]; self.n = self.W_oh.shape[0]
        self.r = np.asarray(r, float); self.sigma = float(sigma)

    def edges(self):
        d, n = self.d, self.n
        E = [('rec', t, i, j) for t in range(1, n) for i in range(d) for j in range(d)]
        E += [('inp', t, i, -1) for t in range(n) for i in range(d)]
        E += [('out', n, i, -1) for i in range(d)]
        return E

    def run(self, x, eps, keep=None, mu=None, real_states=None):
        """Forward with ablation. keep: dict with arrays rec (n, d, d) bool (index t = source step), inp (n, d) bool,
        out (d,) bool; None = full network. eps (B, n, d) standard normal noise. If real_states is given, also return
        the per-node chain-rule KL terms (B-mean) using the network's own previous states."""
        d, n = self.d, self.n; B = x.shape[0]
        h = np.zeros((B, d)); hs = [h]; kl = np.zeros((n, d)) if real_states is not None else None
        for t in range(n):
            if keep is None:
                z = h @ self.W_hh.T + x[:, t:t + 1] * self.W_hi
            else:
                Wm = self.W_hh * keep['rec'][t]
                bias = (self.W_hh * (~keep['rec'][t])) @ mu[t]
                z = h @ Wm.T + bias + x[:, t:t + 1] * (self.W_hi * keep['inp'][t])
                if real_states is not None:
                    hr = real_states[t]
                    zr_net = hr @ self.W_hh.T + x[:, t:t + 1] * self.W_hi
                    zr_sur = hr @ Wm.T + bias + x[:, t:t + 1] * (self.W_hi * keep['inp'][t])
                    diff = np.maximum(zr_net, 0) - np.maximum(zr_sur, 0)
                    kl[t] = (diff ** 2).mean(0) / (2 * (self.sigma * self.r[t]) ** 2 + 1e-300)
            a = np.maximum(z, 0)
            h = a + self.sigma * self.r[t] * eps[:, t]
            hs.append(h)
        if keep is None:
            lo = h @ self.W_oh.T
        else:
            lo = h @ (self.W_oh * keep['out']).T + (self.W_oh * (~keep['out'])) @ mu[n]
        return lo, hs, kl

    def full_keep(self):
        d, n = self.d, self.n
        rec = np.ones((n, d, d), bool); rec[0] = False   # h_0 = 0: no recurrent edges from step 0
        return dict(rec=rec, inp=np.ones((n, d), bool), out=np.ones(d, bool))

    def saliency(self, x, eps, mu):
        """First-order edge saliencies (dict of arrays shaped like keep) and the margin kernel width."""
        d, n = self.d, self.n; B = x.shape[0]
        lo, hs, _ = self.run(x, eps)
        y = np.argsort(x, -1)[:, -2]
        lo2 = lo.copy(); lo2[np.arange(B), y] = -np.inf; jst = lo2.argmax(1)
        m = lo[np.arange(B), y] - lo[np.arange(B), jst]
        s = 0.1 * m.std(); k = np.exp(-0.5 * (m / s) ** 2)
        g_h = self.W_oh[y] - self.W_oh[jst]                       # dm/dh_n (B, d)
        sal = dict(rec=np.zeros((n, d, d)), inp=np.zeros((n, d)), out=np.zeros(d))
        sal['out'] = (k[:, None] * (g_h * (hs[n] - mu[n])) ** 2).mean(0)
        for t in range(n - 1, -1, -1):                            # z_{t+1} = W h_t + w x_t ; hs[t+1] = relu(z) + noise
            zt = hs[t] @ self.W_hh.T + x[:, t:t + 1] * self.W_hi
            g_z = g_h * (zt > 0)
            sal['inp'][t] = (k[:, None] * (g_z * x[:, t:t + 1] * self.W_hi) ** 2).mean(0)
            if t >= 1:
                dev = hs[t] - mu[t]                               # (B, d) source deviations
                # contribution of edge (i <- j): g_z[:, i] * W_ij * dev[:, j]
                sal['rec'][t] = np.einsum('b,bi,bj->ij', k, g_z ** 2, dev ** 2) / B * self.W_hh ** 2
            g_h = g_z @ self.W_hh
        return sal, m, s


def question_metrics(net, x, eps, keep, mu, base=None):
    """flip rate, dp, path KL (nats), for a surrogate vs the network on (x, eps)."""
    y = np.argsort(x, -1)[:, -2]
    if base is None:
        lo, hs, _ = net.run(x, eps); base = dict(err=lo.argmax(1) != y, hs=hs)
    lo_s, _, kl = net.run(x, eps, keep=keep, mu=mu, real_states=base['hs'])
    es = lo_s.argmax(1) != y
    return dict(flip=float((es != base['err']).mean()), dp=float(es.mean() - base['err'].mean()),
                p_net=float(base['err'].mean()), p_sur=float(es.mean()), pathKL=float(kl.sum()),
                pathKL_steps=kl.sum(1).tolist()), base


def n_edges(keep):
    return int(keep['rec'].sum() + keep['inp'].sum() + keep['out'].sum())
