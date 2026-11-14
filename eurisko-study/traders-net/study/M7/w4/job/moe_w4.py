"""M7 worker 4: global wealth prior on a top-2 MoE router (and the per-family comparator).

Model: wealth-net-style transformer over Q question tokens (S0 substrate streams), L=2, d=32,
H=4. Block 0 has a dense ReLU MLP (the shared core); block 1 replaces the MLP with E experts
(ReLU, hidden 32 each), top-2 routing, gates = softmax over the selected router logits
(selection may add a bias; gate values never do, DeepSeek convention). The MoE sits in the last
block, so removing one expert's output from one token changes only that token's logit: the
leave-one-out settlement is exact and cheap.

Settlement (CURRENCY.lock committed-trader rule): for token t routed to expert e,
  s[t,e] = log p(y_t) - log p_{-e}(y_t),
p_{-e} = the prediction with g_e*E_e(h_t) removed from the residual (no renormalisation: the
router is a broker and is paid nothing). Computed at TEST time (prequential), before the step
trains on that label.

Arms (router selection score = router logit + bias; the bias carries no gradient):
  none     : no balancing.
  deepseek : b_e += u*sign(mean_load - load_e), u=1e-3 (arXiv 2408.15664).
  global   : b_e = beta*(R_e - mean_e R), R_e = EMA (half-life H steps) of expert e's settlement
             per live token, summed over ALL families.
  perfam   : b_{e,f(t)} = beta*(R_{e,f} - mean_e R_{e,f}), R_{e,f} = EMA of settlement per
             family-f token.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "substrate"))
from model import LN  # noqa: E402
import streams_ext  # noqa: E402

torch.set_num_threads(1)


class Attn(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.H, self.dh = H, d // H
        self.ln = LN(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)

    def forward(self, x):
        B, Q, d = x.shape
        q, k, v = self.qkv(self.ln(x)).view(B, Q, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), -1)
        return self.o((A @ v).transpose(1, 2).reshape(B, Q, d))


class MoENet(nn.Module):
    def __init__(self, Q, W, E, d=32, H=4, he=32, k=2):
        super().__init__()
        self.Q, self.E, self.k, self.d, self.he = Q, E, k, d, he
        self.win = nn.Linear(W, d)
        self.qid = nn.Parameter(torch.randn(Q, d) * 0.1)
        self.a0, self.a1 = Attn(d, H), Attn(d, H)
        self.ln_m0 = LN(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
        self.ln_m1 = LN(d)
        self.router = nn.Linear(d, E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.w1 = nn.Parameter(torch.randn(E, d, he) / math.sqrt(d)); self.b1 = nn.Parameter(torch.zeros(E, he))
        self.w2 = nn.Parameter(torch.randn(E, he, d) / math.sqrt(he) * 0.5); self.b2 = nn.Parameter(torch.zeros(E, d))
        self.lnf = LN(d)
        self.rw = nn.Parameter(torch.randn(Q, d) / math.sqrt(d)); self.rb = nn.Parameter(torch.zeros(Q))

    def active_flops_per_token(self):
        """Multiply-adds x2 of the per-token forward, MoE counted at k experts (attention
        score/value terms excluded; identical across arms)."""
        d, he, k, E = self.d, self.he, self.k, self.E
        lin = 2 * (self.win.in_features * d + 2 * (3 * d * d + d * d) + 2 * 4 * d * d + d * E + d)
        return lin + k * 2 * (2 * d * he)

    def forward(self, win, bias_fn, want_loo=False):
        x = self.win(2.0 * win - 1.0) + self.qid.unsqueeze(0)
        x = x + self.a0(x)
        x = x + self.fc2(F.relu(self.fc1(self.ln_m0(x))))
        x = x + self.a1(x)
        h = self.ln_m1(x)                                          # [B,Q,d]
        logits = self.router(h)                                    # [B,Q,E]
        with torch.no_grad():
            sel = logits + bias_fn(logits)
            idx = sel.topk(self.k, -1).indices                     # [B,Q,k]
        g = torch.softmax(logits.gather(-1, idx), -1)              # [B,Q,k]
        # only the k selected experts are evaluated (gather their weights per token)
        w1 = self.w1[idx]; b1 = self.b1[idx]; w2 = self.w2[idx]; b2 = self.b2[idx]
        hh = F.relu(torch.einsum("bqd,bqkdh->bqkh", h, w1) + b1)
        eo = torch.einsum("bqkh,bqkhd->bqkd", hh, w2) + b2         # [B,Q,k,d]
        contrib = g.unsqueeze(-1) * eo
        x = x + contrib.sum(2)
        z = (self.lnf(x) * self.rw).sum(-1) + self.rb              # [B,Q]
        out = {"z": z, "idx": idx, "g": g}
        if want_loo:
            xl = x.unsqueeze(2) - contrib                          # [B,Q,k,d]
            out["z_loo"] = (self.lnf(xl) * self.rw.unsqueeze(1)).sum(-1) + self.rb.view(1, -1, 1)
        return out


def logp(z, y):
    return -(F.softplus(z) - y * z)   # log p(y) for a Bernoulli logit


def run(arm, E, seed, T, B, beta, half_life, out, u=1e-3, lr=1e-3, Q=64):
    torch.manual_seed(seed); np.random.seed(seed)
    s = streams_ext.make_streams(Q=Q, T=T, seed=seed, regime_every=10000, regime_frac=0.25)
    st = streams_ext.to_array_stream(s)
    Qt, nf = st.Q, st.n_fam
    fam = torch.as_tensor(st.fam, dtype=torch.long)
    net = MoENet(Qt, st.W, E)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rho = 0.5 ** (1.0 / half_life)
    R_g = torch.zeros(E)
    R_f = torch.zeros(nf, E)
    b_ds = torch.zeros(E)
    fam_oh = F.one_hot(fam, nf).float()

    def bias_fn(logits):
        if arm == "none":
            return torch.zeros(E)
        if arm == "deepseek":
            return b_ds
        if arm == "global":
            return beta * (R_g - R_g.mean())
        if arm == "perfam":
            Rc = R_f - R_f.mean(1, keepdim=True)
            return beta * Rc[fam].unsqueeze(0)
        raise ValueError(arm)

    n_pos = st.n_pos; n_steps = n_pos // B
    Z = np.zeros((n_steps * B, Qt), np.float32); M = np.zeros((n_steps * B, Qt), bool)
    Y = np.zeros((n_steps * B, Qt), np.uint8)
    cnt_fe = np.zeros((n_steps, nf, E), np.float32)       # routed slots per (family, expert)
    pay_fe = np.zeros((n_steps, nf, E), np.float32)       # settlement sums per (family, expert)
    ntok_f = np.zeros((n_steps, nf), np.float32)
    bias_abs = np.zeros(n_steps, np.float32)
    trained = np.zeros((n_pos,), bool)
    t0 = time.time()
    for step in range(n_steps):
        P = np.arange(step * B, (step + 1) * B)
        win, y, m = st.window(P), st.label(P), st.label_mask(P)
        net.eval()
        with torch.no_grad():
            o = net(win, bias_fn, want_loo=True)
            z = o["z"]
            Z[P], Y[P], M[P] = z.numpy(), y.numpy().astype(np.uint8), m.numpy()
            s_pay = logp(z, y).unsqueeze(-1) - logp(o["z_loo"], y.unsqueeze(-1))   # [B,Q,k]
            mf = m.float()
            oh = F.one_hot(o["idx"], E).float() * mf[..., None, None]             # [B,Q,k,E]
            fq = fam_oh.unsqueeze(0).expand(len(P), -1, -1)
            c_fe = torch.einsum("bqf,bqke->fe", fq, oh)
            p_fe = torch.einsum("bqf,bqke->fe", fq, oh * s_pay.unsqueeze(-1))
            n_f = (fq * mf.unsqueeze(-1)).sum((0, 1))
            cnt_fe[step], pay_fe[step], ntok_f[step] = c_fe.numpy(), p_fe.numpy(), n_f.numpy()
            bb = bias_fn(None) if arm != "none" else torch.zeros(E)
            bias_abs[step] = float(bb.abs().mean())
            # settle (delay 0: label revealed now); the new wealth prices the next step
            ntot = n_f.sum().clamp_min(1.0)
            R_g.mul_(rho).add_((1 - rho) * p_fe.sum(0) / ntot)
            R_f.mul_(rho).add_((1 - rho) * p_fe / n_f.clamp_min(1.0).unsqueeze(-1))
            if arm == "deepseek":
                load = c_fe.sum(0)
                b_ds.add_(u * torch.sign(load.mean() - load))
        assert not trained[P].any(), "label trained twice"
        trained[P] = True
        net.train()
        opt.zero_grad(set_to_none=True)
        zt = net(win, bias_fn)["z"]
        loss = (F.binary_cross_entropy_with_logits(zt, y, reduction="none") * m.float()).sum() / m.float().sum()
        loss.backward(); opt.step()
    np.savez_compressed(os.path.join(out, "raw.npz"), Z=Z, Y=Y, M=M, cnt_fe=cnt_fe, pay_fe=pay_fe,
                        ntok_f=ntok_f, fam=st.fam, bias_abs=bias_abs)
    json.dump(s.events, open(os.path.join(out, "events.json"), "w"))
    return {"wall": time.time() - t0, "n_steps": n_steps, "flops_tok": net.active_flops_per_token(),
            "n_params": sum(p.numel() for p in net.parameters())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--T", type=int, default=40032)
    ap.add_argument("--B", type=int, default=8); ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--half_life", type=float, default=1000.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = run(a.arm, a.E, a.seed, a.T, a.B, a.beta, a.half_life, a.out)
    info.update(vars(a))
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    print(json.dumps(info))
