"""M6 worker 3: (e) log- vs exp-utility pool and (f) bounded bets, plus in-harness references.

Usage: python m6w3.py ARM SEED [T]   -> results/<arm>-s<seed>.json
Arms: a, b, kwd, kboost (references), elog, eexp, fb01, fb05 (w3). See PREREG.md.
"""
import json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(STUDY, "S0", "substrate"))
torch.set_num_threads(1)
from streams_ext import make_streams, to_array_stream, FAMILIES  # noqa: E402
from model import Net, Logistic  # noqa: E402
import panel  # noqa: E402

N_TR = 16
LEARNABLE = ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]
NOISE = ["prng", "ca30"]


class Trunk(Net):
    def features(self, win):
        B = win.shape[0]
        x = self.win(2.0 * win - 1.0) + self.qid.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).view(B, self.Q, 3, self.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            x = x + blk.o((A @ v).transpose(1, 2).reshape(B, self.Q, self.d))
            x = x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))))
        return self.lnf(x)                                   # [B,Q,d]


class MarketLayer(nn.Module):
    def __init__(self, arm, Q, W, d=32):
        super().__init__()
        self.arm = arm
        self.trunk = Trunk(Q, W, d=d, H=4, L=1)
        self.lm = Logistic(Q, W)
        self.Wd = nn.Parameter(torch.randn(Q, N_TR, d + 1) * 0.01)
        self.bd = nn.Parameter(torch.zeros(Q, N_TR))
        if arm in ("elog", "eexp"):
            self.Wc = nn.Parameter(torch.randn(Q, N_TR, d + 1) * 0.01)
            self.bc = nn.Parameter(torch.full((Q, N_TR), -3.0))

    def parts(self, win):
        l0 = self.lm(win)                                    # [B,Q]
        f = torch.cat([self.trunk.features(win), l0.detach().unsqueeze(-1) / 4.0], -1)  # [B,Q,d+1]
        d = torch.einsum("bqf,qnf->bqn", f, self.Wd) + self.bd
        k = None
        if self.arm in ("elog", "eexp"):
            k = F.softplus(torch.einsum("bqf,qnf->bqn", f, self.Wc) + self.bc)
        return l0, d, k

    @staticmethod
    def combine(arm, l0, d, k, drop=None):
        """Final logit. drop: index of a trader to leave out (leave-one-out settlement)."""
        if drop is not None:
            keep = torch.ones(d.shape[-1], dtype=d.dtype); keep[drop] = 0
            d = d * keep
            k = None if k is None else k * keep
        L0 = l0.detach()
        if arm in ("a", "kwd"):
            return d.sum(-1)
        if arm == "b":
            return L0 + d.sum(-1)
        if arm == "kboost":
            return L0 + 0.1 * d.sum(-1)
        if arm == "fb01":
            return L0 + (0.1 * torch.tanh(d / 0.1)).sum(-1)
        if arm == "fb05":
            return L0 + (0.5 * torch.tanh(d / 0.5)).sum(-1)
        if arm == "eexp":
            return L0 + (k * d).sum(-1) / (1 + k.sum(-1))
        if arm == "elog":
            p = (torch.sigmoid(L0) + (k * torch.sigmoid(L0.unsqueeze(-1) + d)).sum(-1)) / (1 + k.sum(-1))
            p = p.clamp(1e-6, 1 - 1e-6)
            return torch.log(p) - torch.log1p(-p)
        raise ValueError(arm)

    def forward(self, win):
        l0, d, k = self.parts(win)
        return self.combine(self.arm, l0, d, k), l0, d, k


def bce_np(z, y):
    z = z.astype(np.float64)
    return np.logaddexp(0, z) - y * z


def run(arm, seed, T=24000, B=4, out=None):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    s = make_streams(Q=64, T=T, seed=seed, heldout=8, release=T // 2, regime_every=0)
    st = to_array_stream(s)
    Q, n_pos = st.Q, st.n_pos
    m = MarketLayer(arm, Q, st.W)
    lm_params = list(m.lm.parameters())
    rest = [p for n_, p in m.named_parameters() if not n_.startswith("lm.")]
    if arm == "kwd":
        opt = torch.optim.AdamW(rest, 1e-3, weight_decay=0.1)
    else:
        opt = torch.optim.Adam(rest, 1e-3)
    opt_l = torch.optim.Adam(lm_params, 1e-2)
    Z = np.full((n_pos, Q), np.nan, np.float32); Z0 = Z.copy()
    Y = np.zeros((n_pos, Q), np.uint8); M = np.zeros((n_pos, Q), bool)
    trained = np.zeros((n_pos, Q), bool)
    wealth = np.zeros((Q, N_TR))                  # LOO committed settlement, second half only
    wealth_all = np.zeros((Q, N_TR))
    kmean = []
    half = n_pos // 2
    n_steps = math.ceil(n_pos / B)
    for sidx in range(n_steps):
        P = np.arange(sidx * B, min((sidx + 1) * B, n_pos))
        win, y, mk = st.window(P), st.label(P), st.label_mask(P)
        with torch.no_grad():                     # TEST at the trade price
            z, l0, d, k = m(win)
            Z[P], Z0[P] = z.numpy(), l0.numpy()
            Y[P], M[P] = y.numpy().astype(np.uint8), mk.numpy()
            assert not trained[P].any(), "label trained before scored"
            # committed-trader settlement (CURRENCY.lock rule, leave-one-out at trade time)
            if sidx % 4 == 0:
                lp = -F.binary_cross_entropy_with_logits(z, y, reduction="none")
                mf = mk.float()
                for i in range(N_TR):
                    zi = MarketLayer.combine(arm, l0, d, k, drop=i)
                    si = (lp + F.binary_cross_entropy_with_logits(zi, y, reduction="none")) * mf
                    w = si.sum(0).numpy()
                    wealth_all[:, i] += w
                    if P[0] >= half:
                        wealth[:, i] += w
                if k is not None:
                    kmean.append(float(k.sum(-1).mean()))
        # TRAIN on this block (delay 0)
        tm = mk.float()
        opt.zero_grad(set_to_none=True); opt_l.zero_grad(set_to_none=True)
        z, l0, _, _ = m(win)
        loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * tm).sum() / tm.sum().clamp_min(1)
        loss_l = (F.binary_cross_entropy_with_logits(l0, y, reduction="none") * tm).sum() / tm.sum().clamp_min(1)
        (loss + loss_l).backward()
        opt.step(); opt_l.step()
        trained[P] = mk.numpy()
    # metrics on second half
    fam = st.fam
    sel = np.zeros_like(M); sel[half:] = True
    sel &= M
    bz, b0 = bce_np(Z, Y), bce_np(Z0, Y)
    fam_bce, fam_bce0 = {}, {}
    for f, name in enumerate(FAMILIES):
        mm = sel & (fam[None, :] == f)
        if mm.any():
            fam_bce[name], fam_bce0[name] = float(bz[mm].mean()), float(b0[mm].mean())
    lmask = sel & np.isin(fam, [FAMILIES.index(n) for n in LEARNABLE])[None, :]
    res = {"arm": arm, "seed": seed, "T": T, "B": B, "n_traders": N_TR, "steps": n_steps,
           "learnable_bce": float(bz[lmask].mean()), "learnable_bce_market": float(b0[lmask].mean()),
           "noise_excess": max(fam_bce[n] - math.log(2) for n in NOISE),
           "noise_excess_market": max(fam_bce0[n] - math.log(2) for n in NOISE),
           "fam_bce": fam_bce, "fam_bce_market": fam_bce0}
    # AND/OR coherence (prices at trade time, second half)
    cons = []
    for q, ds in enumerate(s.desc):
        mt = re.match(r"(and|or)\(q(\d+) .*, q(\d+) ", ds)
        if mt:
            cons.append((mt.group(1), q, int(mt.group(2)), int(mt.group(3))))
    Pp = 1 / (1 + np.exp(-Z[half:].astype(np.float64)))
    v = panel.violations(Pp, cons, tol=0.01)
    res["viol"] = {k_: v[k_]["mean"] for k_ in v}
    res["viol_frac"] = {k_: v[k_]["frac_gt_tol"] for k_ in v}
    # wealth diagnostics (committed LOO settlement, sampled every 4th step)
    wv = wealth[:s.n_main].ravel()
    res["wealth_gini_abs"] = float(panel.gini(np.abs(wv)))
    res["wealth_frac_negative"] = float((wv < 0).mean())
    res["wealth_total_per_q"] = float(wv.sum() / s.n_main)
    res["wealth_by_family"] = {FAMILIES[f]: float(wealth[fam == f].sum() / max(1, (fam == f).sum()))
                               for f in range(len(FAMILIES)) if (fam == f).any()}
    res["conf_sum_mean_last"] = float(np.mean(kmean[-100:])) if kmean else None
    res["prequential_assert"] = "each (pos,q) scored before trained, trained once"
    res["wall_s"] = time.time() - t0
    if out:
        with open(out + ".tmp", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(out + ".tmp", out)
        np.savez_compressed(os.path.join("/tmp/traders-net/M6/w3", f"{arm}-s{seed}.npz"),
                            wealth=wealth_all.astype(np.float32), Z=Z[::8], Z0=Z0[::8])
    return res


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
    tag = "" if T == 24000 else f"-T{T}"
    out = os.path.join(HERE, "results", f"{arm}-s{seed}{tag}.json")
    r = run(arm, seed, T=T, out=out)
    print(json.dumps({k: r[k] for k in ("arm", "seed", "learnable_bce", "learnable_bce_market", "noise_excess", "wall_s")}))
