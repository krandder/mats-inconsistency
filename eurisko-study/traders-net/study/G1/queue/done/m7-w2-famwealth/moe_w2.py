"""M7 w2: top-2 MoE on the S0 substrate; DeepSeek bias + per-family wealth prior (see PREREG.md).

Usage: python moe_w2.py --arm dsfw --E 8 --seed 1 [--T 30000] --out results/
Arms: none, ds3, ds2, dsfw, dsfw_fast, dsrw.
"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("M7_SUBSTRATE", os.path.join(HERE, "../../S0/substrate")))
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/wealth-net/.pylib")
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)
import streams_ext
from model import LN
from streams_ext import FAMILIES

ARMS = {  # u (DeepSeek step; 0 = off), gamma (wealth discount; None = no prior), groups
    "none": (0.0, None, None), "ds3": (1e-3, None, None), "ds2": (1e-2, None, None),
    "dsfw": (1e-3, 0.999, "family"), "dsfw_fast": (1e-3, 0.99, "family"), "dsrw": (1e-3, 0.999, "random"),
    "dsfw_mean": (1e-3, 0.999, "family"),   # bonus = per-token mean payoff only (no N/E evidence scaling)
}


class MoE(nn.Module):
    def __init__(self, d, E, h):
        super().__init__()
        self.E = E
        self.router = nn.Linear(d, E)
        self.W1 = nn.Parameter(torch.randn(E, d, h) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(E, h))
        self.W2 = nn.Parameter(torch.randn(E, h, d) / math.sqrt(h))
        self.b2 = nn.Parameter(torch.zeros(E, d))

    def forward(self, x, sel_bias):
        """x [N,d]; sel_bias [N,E] or None (selection only). Returns slot outputs [N,2,d] (gated), idx [N,2]."""
        logits = self.router(x)
        sel = logits if sel_bias is None else logits + sel_bias
        idx = sel.detach().topk(2, dim=-1).indices                     # [N,2]
        g = torch.softmax(logits.gather(1, idx), dim=-1)               # renormalised top-2 gates
        hdn = F.relu(torch.einsum("nd,nkdh->nkh", x, self.W1[idx]) + self.b1[idx])
        y = torch.einsum("nkh,nkhd->nkd", hdn, self.W2[idx]) + self.b2[idx]
        return g.unsqueeze(-1) * y, idx


class MoENet(nn.Module):
    def __init__(self, Q, W, E, d=64, H=4, L=2, h=64):
        super().__init__()
        self.Q, self.W, self.d, self.H, self.L, self.E, self.dh = Q, W, d, H, L, E, d // H
        self.win = nn.Linear(W, d)
        self.qid = nn.Parameter(torch.randn(Q, d) * 0.1)
        self.ln1 = nn.ModuleList([LN(d) for _ in range(L)])
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(L)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.ln2 = nn.ModuleList([LN(d) for _ in range(L)])
        self.moe = nn.ModuleList([MoE(d, E, h) for _ in range(L)])
        self.lnf = LN(d)
        self.rw = nn.Parameter(torch.randn(Q, d) / math.sqrt(d))
        self.rb = nn.Parameter(torch.zeros(Q))

    def forward(self, win, sel_bias=None, remove=None):
        """sel_bias: list per layer of [B*Q,E] or None. remove=(layer, slot): drop that slot's gated
        output at that layer for every token (settlement counterfactual). Returns z [B,Q], idx list."""
        B, Q, d = win.shape[0], self.Q, self.d
        x = self.win(2.0 * win - 1.0) + self.qid.unsqueeze(0)
        idxs = []
        for l in range(self.L):
            h = self.ln1[l](x)
            q, k, v = self.qkv[l](h).view(B, Q, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)
            x = x + self.o[l]((A @ v).transpose(1, 2).reshape(B, Q, d))
            slots, idx = self.moe[l](self.ln2[l](x).reshape(B * Q, d), None if sel_bias is None else sel_bias[l])
            if remove is not None and remove[0] == l:
                keep = torch.ones(2, 1, device=x.device); keep[remove[1]] = 0.0
                slots = slots * keep
            x = x + slots.sum(1).view(B, Q, d)
            idxs.append(idx)
        z = (self.lnf(x) * self.rw).sum(-1) + self.rb
        return z, idxs


def run(arm, E, seed, T=24000, B=8, out="results", lr=1e-3, device="cpu"):
    dev = torch.device(device)
    u, gamma, groups = ARMS[arm]
    t0 = time.time()
    s = streams_ext.make_streams(Q=64, T=T, seed=seed, heldout=8, release=T // 2, regime_every=10000)
    st = streams_ext.to_array_stream(s)
    Q, n_main, L = st.Q, s.n_main, 2
    fam0 = np.asarray(s.fam0)
    n_grp = len(FAMILIES)
    if groups == "random":   # same group sizes, questions shuffled across groups (fixed per seed)
        grp = np.random.default_rng(1000 + seed).permutation(fam0)
    else:
        grp = fam0.copy()
    torch.manual_seed(seed)
    net = MoENet(Q, st.W, E).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_steps = st.n_pos // B
    n_pos = n_steps * B
    Z = np.full((n_pos, Q), np.nan, np.float32); Y = np.zeros((n_pos, Q), np.uint8); M = np.zeros((n_pos, Q), bool)
    loads = np.zeros((n_steps, L, E), np.int32)
    famcnt = np.zeros((L, n_grp, E), np.int64)       # routed slots by (layer, fam0, expert), last 2000 steps
    bias = torch.zeros(L, E, device=dev)
    S = torch.zeros(L, n_grp, E, device=dev); C = torch.zeros(L, n_grp, E, device=dev); N = torch.zeros(L, n_grp, device=dev)
    grp_t = torch.as_tensor(np.tile(grp, B), dtype=torch.long, device=dev)          # token -> group, tokens = b*Q+q
    fam_t = torch.as_tensor(np.tile(fam0, B), dtype=torch.long)
    payoff_log = []   # per step: mean payoff per layer (diagnostic)
    n_fwd = 0
    for step in range(n_steps):
        P = np.arange(step * B, (step + 1) * B)
        win, y, m = st.window(P).to(dev), st.label(P).to(dev), st.label_mask(P).to(dev)
        if gamma is not None:
            bonus = S / (C + 1.0) if arm == "dsfw_mean" else (N / E).unsqueeze(-1) * S / (C + 1.0)   # [L,G,E]
            sb = [bias[l].unsqueeze(0) + bonus[l][grp_t] for l in range(L)]
        elif u > 0:
            sb = [bias[l].unsqueeze(0).expand(B * Q, E) for l in range(L)]
        else:
            sb = None
        net.eval()
        with torch.no_grad():                                   # TEST at the trade price
            z, idxs = net(win, sb); n_fwd += 1
            Z[P] = z.cpu().numpy(); Y[P] = y.cpu().numpy().astype(np.uint8); M[P] = m.cpu().numpy()
            for l in range(L):
                loads[step, l] = torch.bincount(idxs[l].reshape(-1), minlength=E).cpu().numpy()
                if step >= n_steps - 2000:
                    np.add.at(famcnt[l], (np.repeat(fam_t.cpu().numpy(), 2), idxs[l].reshape(-1).cpu().numpy()), 1)
            if gamma is not None:                               # SETTLE committed trades (label revealed now)
                lp = -F.binary_cross_entropy_with_logits(z, y, reduction="none").reshape(-1)
                live = m.reshape(-1)
                S.mul_(gamma); C.mul_(gamma); N.mul_(gamma)
                pl = []
                for l in range(L):
                    for k in range(2):
                        zr, _ = net(win, sb, remove=(l, k)); n_fwd += 1
                        pay = lp + F.binary_cross_entropy_with_logits(zr, y, reduction="none").reshape(-1)
                        e = idxs[l][:, k]
                        g_ = grp_t[live]; e_ = e[live]
                        S[l].index_put_((g_, e_), pay[live], accumulate=True)
                        C[l].index_put_((g_, e_), torch.ones_like(pay[live]), accumulate=True)
                        N[l].index_add_(0, g_, torch.ones_like(pay[live]))
                        pl.append(float(pay[live].mean()))
                payoff_log.append(pl)
        net.train()                                             # TRAIN on the revealed labels
        opt.zero_grad(set_to_none=True)
        zt, idt = net(win, sb); n_fwd += 1
        mf = m.float()
        loss = (F.binary_cross_entropy_with_logits(zt, y, reduction="none") * mf).sum() / mf.sum().clamp_min(1)
        loss.backward(); opt.step()
        if u > 0:                                               # DeepSeek bias update on the training batch load
            for l in range(L):
                ld = torch.bincount(idt[l].reshape(-1), minlength=E).float()
                bias[l] += u * torch.sign(ld.mean() - ld)
    # ---- metrics
    bce = F.binary_cross_entropy_with_logits(torch.tensor(Z), torch.tensor(Y, dtype=torch.float32), reduction="none").numpy()
    main = np.zeros(Q, bool); main[:n_main] = True
    def mbce(rows, cols):
        mm = M[rows][:, cols]; b = bce[rows][:, cols]
        return float(b[mm].mean()) if mm.any() else float("nan")
    allr = np.arange(n_pos); last = np.arange(max(0, n_pos - 10000), n_pos)
    res = {"arm": arm, "E": E, "seed": seed, "T": T, "B": B, "steps": n_steps,
           "bce_all": mbce(allr, main), "bce_last10k": mbce(last, main)}
    # per-family (fam0) BCE, main questions
    for f in range(n_grp):
        cols = np.where(fam0 == f)[0]
        if len(cols):
            res[f"bce_fam_{FAMILIES[f]}"] = mbce(allr, cols)
    # after regime shift: switched questions, 1000 positions after each boundary vs 1000 before
    post, pre = [], []
    for ev in s.events:
        if ev["kind"] != "switch":
            continue
        i0 = ev["t"] - st.W
        a = np.arange(i0, min(i0 + 1000, n_pos)); b = np.arange(max(0, i0 - 1000), i0)
        post.append(bce[a, ev["q"]][M[a, ev["q"]]].mean()); pre.append(bce[b, ev["q"]][M[b, ev["q"]]].mean())
    res["bce_post_shift"] = float(np.mean(post)); res["bce_pre_shift"] = float(np.mean(pre))
    ho = np.where(fam0 == 8)[0]
    ra = np.arange(s.release - st.W, min(s.release - st.W + 1000, n_pos))
    res["bce_heldout_1k"] = mbce(ra, ho); res["bce_heldout_all"] = mbce(allr, ho)
    lw = loads[-1000:].sum(0).astype(float)                                 # [L,E]
    share = lw / lw.sum(1, keepdims=True)
    res["max_load"] = float(share.max()); res["max_load_by_layer"] = share.max(1).tolist()
    res["dead_frac"] = float((share < 0.1 / E).mean())
    mis = []
    for l in range(L):
        pj = famcnt[l] / famcnt[l].sum(); pf = pj.sum(1, keepdims=True); pe = pj.sum(0, keepdims=True)
        nz = pj > 0
        mis.append(float((pj[nz] * np.log2(pj[nz] / (pf @ pe)[nz])).sum()))
    res["mi_expert_family_bits"] = float(np.mean(mis)); res["mi_by_layer"] = mis
    # FLOPs per token: expert (2 slots x 2 matmuls d x 64) + router; fwd counts incl. settlement
    ff_tok = 2 * 2 * (2 * 64 * 64) + 2 * 64 * E
    res["expert_router_flops_per_token_fwd"] = ff_tok
    res["forwards_per_step"] = n_fwd / n_steps
    if payoff_log:
        pa = np.array(payoff_log)
        res["mean_payoff_by_layer_slot_last1k"] = pa[-1000:].mean(0).tolist()
        bonus = S / (C + 1.0) if arm == "dsfw_mean" else (N / E).unsqueeze(-1) * S / (C + 1.0)
        res["final_bonus_absmax"] = float(bonus.abs().max()); res["final_bonus_sd_over_experts"] = float(bonus.std(-1).mean())
        np.savetxt(os.path.join(out, f"{arm}-E{E}-s{seed}-bonus.csv"), bonus.cpu().numpy().reshape(-1, E), delimiter=",", fmt="%.5g", header="rows = layer*9 + fam0 (streams_ext.FAMILIES), cols = expert; final prior bonus (nats)")
    res["wall_s"] = time.time() - t0
    os.makedirs(out, exist_ok=True)
    tag = f"{arm}-E{E}-s{seed}"
    with open(os.path.join(out, tag + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    # per-step load trace (max share per layer, per 100 steps) for plots
    ls = loads.reshape(-1, 100, L, E).sum(1) if n_steps % 100 == 0 else loads[: n_steps // 100 * 100].reshape(-1, 100, L, E).sum(1)
    np.savetxt(os.path.join(out, tag + "-maxload.csv"), (ls.max(-1) / ls.sum(-1)), delimiter=",", fmt="%.4f", header="layer0,layer1 (per 100 steps)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS)); ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--out", default=os.path.join(HERE, "results")); ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    r = run(a.arm, a.E, a.seed, a.T, B=a.B, out=a.out, device=a.device)
    print(json.dumps({k: r[k] for k in ("arm", "E", "seed", "bce_all", "bce_last10k", "max_load", "dead_frac", "mi_expert_family_bits", "wall_s")}))
