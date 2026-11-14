"""Exact market settlement at the output instead of gradient credit (no training).

A neuron's direct logit contribution: its output vector act_i * W2[:, i] read out through the
final LayerNorm with the per-example scale held fixed, c_i[q] = act_i[q] * u[q, i] / std[q].
Settlement uses the log score lp(z) = log p(y) = -softplus(-(2y-1) z):
  loo   s_i = lp(z) - lp(z - c_i)                                   (leave one out)
  seq   MLP layer l earns lp(lens after l) - lp(lens before l), logit lens = final LN + readout
        on the residual stream at that cut (fresh LN). Inside the layer: leave-one-out at the
        lens after l; the leftover (layer earning minus the sum of leave-one-outs) is split
        equally, so the neurons of a layer sum exactly to the layer's earning.
  shap  sampled Shapley over the 512 neurons' direct contributions, v(S) = lp(base + sum_S c),
        8 random orders per example (the same order for all questions of that example).
Each is summed over (a) the held-out segment (the same positions the oracle and curves use:
labels the oracle also sees) and (b) 4096 training positions (what a market would have seen).
Compared with the single-neuron ablation oracle (results/decay_oracle_s*.npy), plus pruning
curves on the same held-out subsample as decay.py. Usage: python3 settle.py SEED [results]
"""
import csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from streams import make_questions, FAMILIES
from model import Net
from decay import tail_fns

FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
KINDS = ["loo", "seq", "shap"]


def lp(z, y):
    return -F.softplus(-(2 * y - 1) * z)


def settle(m, wins, ys, gen):
    """Per-neuron settlement totals [512] for each kind, and per-layer earnings, over the given examples."""
    L, N = m.L, 4 * m.d
    tot = {k: torch.zeros(L * N, dtype=torch.float64) for k in KINDS}
    layer_earn = torch.zeros(L, dtype=torch.float64)
    eff_err = [0.0]
    g, rw = m.lnf.g, m.rw                                    # [d], [Q,d]
    U = []
    for blk in m.blocks:
        W2c = blk.fc2.weight - blk.fc2.weight.mean(0, keepdim=True)   # [d,N], centered over d
        U.append(torch.einsum("dn,d,qd->qn", W2c, g, rw))             # [Q,N]
    for i in range(0, wins.shape[0], 512):
        win, y = wins[i:i + 512], ys[i:i + 512]
        rec = {}
        z = m(win, rec=rec)                                  # [B,Q]
        cuts = rec["cuts"]                                   # x0, after attn0, after mlp0, after attn1, after mlp1
        def std(x):
            xc = x - x.mean(-1, keepdim=True)
            return torch.sqrt(xc.pow(2).mean(-1) + m.lnf.eps)   # [B,Q]
        def lens(x):
            return (m.lnf(x) * rw).sum(-1) + m.rb
        s_fin = std(cuts[-1])
        cs = [rec["act"][l] * U[l] / s_fin.unsqueeze(-1) for l in range(L)]   # [B,Q,N] final-lens direct contributions
        c = torch.cat(cs, -1)                                               # [B,Q,LN]
        lz = lp(z, y)
        tot["loo"] += (lz.unsqueeze(-1) - lp(z.unsqueeze(-1) - c, y.unsqueeze(-1))).sum((0, 1)).double()
        for l in range(L):
            pre, post = cuts[2 * l + 1], cuts[2 * l + 2]
            zpost, zpre = lens(post), lens(pre)
            earn = lp(zpost, y) - lp(zpre, y)                               # [B,Q]
            cl = rec["act"][l] * U[l] / std(post).unsqueeze(-1)
            loo = lp(zpost, y).unsqueeze(-1) - lp(zpost.unsqueeze(-1) - cl, y.unsqueeze(-1))
            left = (earn - loo.sum(-1)) / N
            tot["seq"][l * N:(l + 1) * N] += (loo + left.unsqueeze(-1)).sum((0, 1)).double()
            layer_earn[l] += earn.sum().double()
        base = z - c.sum(-1)
        B = c.shape[0]
        sh = torch.zeros(L * N, dtype=torch.float64)
        for _ in range(8):                                   # a fresh random order for every example
            perm = torch.argsort(torch.rand(B, L * N, generator=gen), -1)          # [B,LN]
            pq = perm.unsqueeze(1).expand(-1, c.shape[1], -1)                     # same order for every question
            cum = base.unsqueeze(-1) + torch.cumsum(c.gather(-1, pq), -1)
            v = lp(cum, y.unsqueeze(-1))
            marg = v - torch.cat([lp(base, y).unsqueeze(-1), v[..., :-1]], -1)    # marginal of the unit at each position
            sh.scatter_add_(0, pq.reshape(-1), marg.reshape(-1).double())
        tot["shap"] += sh / 8
        eff_err[0] += float((sh / 8).sum() - (lz - lp(base, y)).sum())
    print('shapley efficiency error', round(eff_err[0], 3), 'of', round(float(tot['shap'].sum()), 1), flush=True)
    return {k: v.numpy() for k, v in tot.items()}, layer_earn.numpy()


def main(seed, res="results"):
    torch.set_num_threads(int(os.environ.get("THREADS", "5")))
    t0 = time.time(); tag = f"train-s{seed}"
    meta = json.load(open(os.path.join(res, f"{tag}.json")))
    Q, W, T = meta["Q"], meta["W"], meta["T"]; E = 4096
    bits, fam, _ = make_questions(Q, T + E, seed)
    b = torch.tensor(bits, dtype=torch.float32)
    wins = b.unfold(1, W, 1)
    ev = torch.arange(T - W, T + E - W)
    ev_win, ev_y = wins[:, ev].permute(1, 0, 2).contiguous(), b[:, ev + W].T.contiguous()
    tr = torch.randint(0, int((T - W) * 0.8), (E,), generator=torch.Generator().manual_seed(seed + 99))
    tr_win, tr_y = wins[:, tr].permute(1, 0, 2).contiguous(), b[:, tr + W].T.contiguous()
    famt = torch.tensor(fam)
    final = meta["steps_done"]
    oracle = np.load(os.path.join(res, f"decay_oracle_s{seed}.npy"))
    abl, spear = [], []
    gen = torch.Generator().manual_seed(seed)
    for step, wf in ((final, os.path.join(res, f"{tag}.pt")), (20000, os.path.join(res, "weights", f"{tag}-step20000.pt"))):
        m = Net(Q, W, meta["d"], meta["H"], meta["L"])
        m.load_state_dict(torch.load(wf, map_location="cpu")); m.eval()
        L, N = m.L, 4 * m.d; n = L * N
        head, tail, *_ = tail_fns(m)
        with torch.no_grad():
            cur = {}
            for seg, (wv, yv) in (("heldout", (ev_win, ev_y)), ("train", (tr_win, tr_y))):
                tot, le = settle(m, wv, yv, gen)
                for k, v in tot.items():
                    cur[f"{k}_{seg}"] = v
                if step == final and seg == "heldout":
                    # sanity: seq neurons sum to the MLP layers' earnings
                    print(seed, "seq conservation", float(tot["seq"].sum() - le.sum()), "layer earnings", le.round(1), flush=True)
            np.savez(os.path.join(res, f"settle_currencies_s{seed}_step{step}.npz"), **cur)
            if step == final:
                top = np.argsort(oracle)[::-1][:int(round(0.05 * n))]
                for k, v in cur.items():
                    rk = np.argsort(np.argsort(v))                      # 0 = lowest
                    spear.append(dict(seed=seed, step=step, currency=k,
                                      spearman_vs_oracle=round(float(spearmanr(v, oracle).correlation), 4),
                                      spearman_layer0=round(float(spearmanr(v[:N], oracle[:N]).correlation), 4),
                                      spearman_layer1=round(float(spearmanr(v[N:], oracle[N:]).correlation), 4),
                                      top5_in_top10=round(float((rk[top] >= n - n // 10).mean()), 3),
                                      top5_in_bottom10=round(float((rk[top] < n // 10).mean()), 3),
                                      frac_negative=round(float((v < 0).mean()), 4)))
            print(seed, step, "settled", round(time.time() - t0), "s", flush=True)
            sub = torch.arange(0, E, 4)
            x1, y = head(ev_win[sub]), ev_y[sub]
            ones = torch.ones(L, N)
            for k, v in cur.items():
                od = np.argsort(v, kind="stable")
                for fr in FRACS:
                    kk = int(round(fr * n))
                    mk = ones.clone().view(-1); mk[torch.tensor(od[:kk])] = 0
                    bce = F.binary_cross_entropy_with_logits(tail(x1, mk.view(L, N)), y, reduction="none").mean(0)
                    abl.append(dict(seed=seed, step=step, final=int(step == final), criterion=k, frac=fr, k_removed=kk,
                                    n_negative=int((v < 0).sum()), bce=round(bce.mean().item(), 6),
                                    **{f"bce_{f}": round(bce[famt == i].mean().item(), 6) for i, f in enumerate(FAMILIES)}))
            print(seed, step, "pruned", round(time.time() - t0), "s", flush=True)
    for name, rows in (("settle_ablation", abl), ("settle_spearman", spear)):
        with open(os.path.join(res, f"{name}_s{seed}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(seed, "all done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "results")
