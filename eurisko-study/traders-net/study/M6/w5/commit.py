"""M6 w5: committed vs uncommitted output trader heads (see PREREG.md).

usage: python3 commit.py --arm U|Cm|Cw|L0 --seed 1 [--T 20000] --out results/
Writes results/<arm>-s<seed>.json (aggregates only).
"""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/wealth-net/.pylib")
sys.path.insert(0, os.path.join(STUDY, "S0", "substrate"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
torch.set_num_threads(1)
import streams_ext
from model import Net, Logistic
import panel

LEARN = ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]
NOISE = ["prng", "ca30"]


class Market(nn.Module):
    def __init__(self, Q, W, d, nh):
        super().__init__()
        self.net = Net(Q, W, d=d, H=4, L=2)
        self.R = nn.Parameter(torch.randn(nh, Q, d) / math.sqrt(d) / math.sqrt(nh))
        self.b = nn.Parameter(torch.zeros(nh, Q))

    def heads(self, win):
        n = self.net
        B = win.shape[0]
        x = n.win(2.0 * win - 1.0) + n.qid.unsqueeze(0)
        for blk in n.blocks:
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).view(B, n.Q, 3, n.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            x = x + blk.o((A @ v).transpose(1, 2).reshape(B, n.Q, n.d))
            x = x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))))
        y = n.lnf(x)                                        # [B,Q,d]
        return torch.einsum("bqd,hqd->bhq", y, self.R) + self.b   # [B,H,Q]


def bce_np(z, y):
    return np.logaddexp(0, z) - y * z


def logp_np(z, y):  # log p(y) for logit z
    return -bce_np(z, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["U", "Cm", "Cw", "Cs", "L0"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--nh", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    t0 = time.time()
    s = streams_ext.make_streams(Q=64, T=a.T, seed=a.seed, heldout=8, regime_every=0)
    st = streams_ext.to_array_stream(s)
    Q, n_pos, nh = st.Q, st.n_pos, a.nh
    torch.manual_seed(a.seed)
    mk = Market(Q, st.W, a.d, nh)
    lg = Logistic(Q, st.W)
    opt = torch.optim.Adam(mk.parameters(), lr=a.lr)
    opt_l = torch.optim.Adam(lg.parameters(), lr=1e-2)
    C = np.zeros((n_pos, nh, Q), np.float32)   # head corrections at trade time
    L0 = np.zeros((n_pos, Q), np.float32)
    ST = np.zeros((n_pos, nh, Q), np.float32)  # stakes at trade time
    Y = np.zeros((n_pos, Q), np.float32)
    M = np.zeros((n_pos, Q), bool)
    K = np.zeros((nh, Q), np.float64)          # running committed wealth (for Cw stakes)
    scored = np.zeros(n_pos, bool)
    for s0 in range(0, n_pos, a.B):
        P = np.arange(s0, min(s0 + a.B, n_pos))
        win, y, m = st.window(P), st.label(P), st.label_mask(P)
        with torch.no_grad():
            c = mk.heads(win).numpy()
            l0 = lg(win).numpy()
        if a.arm in ("U", "Cs"):
            w = np.ones((nh, Q))
        elif a.arm == "Cm":
            w = np.full((nh, Q), 1.0 / nh)
        elif a.arm == "Cw":
            w = np.exp(K - K.max(0, keepdims=True)); w /= w.sum(0, keepdims=True)
        else:
            w = np.zeros((nh, Q))
        C[P], L0[P], ST[P] = c, l0, w[None]
        yn, mn = y.numpy(), m.numpy()
        Y[P], M[P] = yn, mn
        scored[P] = True
        # book committed settlement (prequential; labels of this block are now revealed)
        ks = logp_np(l0[:, None, :] + c, yn[:, None, :]) - logp_np(l0, yn)[:, None, :]
        K += (ks * mn[:, None, :]).sum(0)
        # TRAIN on the same block (test-then-train)
        assert scored[P].all()
        mf = m.float()
        if a.arm != "L0":
            ct = mk.heads(win)
            l0t = torch.as_tensor(l0)
            if a.arm == "U":
                z = l0t + ct.sum(1)
                l = F.binary_cross_entropy_with_logits(z, y, reduction="none")
            elif a.arm == "Cs":
                pre = torch.cumsum(ct, 1) - ct                 # sum_{j<i} c_j
                z = l0t[:, None, :] + pre.detach() + ct
                l = F.binary_cross_entropy_with_logits(z, y[:, None, :].expand_as(z), reduction="none").sum(1)
            else:
                z = l0t[:, None, :] + ct
                l = F.binary_cross_entropy_with_logits(z, y[:, None, :].expand_as(z), reduction="none").mean(1)
            loss = (l * mf).sum() / mf.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        opt_l.zero_grad(set_to_none=True)
        ll = F.binary_cross_entropy_with_logits(lg(win), y, reduction="none")
        ((ll * mf).sum() / mf.sum().clamp_min(1)).backward(); opt_l.step()
    t_train = time.time() - t0
    fam = np.asarray(s.fam0)
    names = streams_ext.FAMILIES
    cut = int(0.75 * n_pos)
    zc = L0 + (ST * C).sum(1)                       # market logit at trade time
    pz = 1 / (1 + np.exp(-zc))
    res = {"arm": a.arm, "seed": a.seed, "T": a.T, "Q": Q, "nh": nh, "d": a.d, "n_pos": n_pos,
           "cut": cut, "train_s": round(t_train, 1)}
    # performance on the future window
    fw = slice(cut, n_pos)
    b = bce_np(zc[fw], Y[fw]); b0 = bce_np(L0[fw], Y[fw])
    perf = {}
    for f, nm in enumerate(names):
        mm = M[fw] & (fam[None, :] == f)
        if mm.any():
            perf[nm] = {"bce": float(b[mm].mean()), "bce_l0": float(b0[mm].mean())}
    lm = M[fw] & np.isin(fam, [names.index(x) for x in LEARN])[None, :]
    res["perf"] = perf
    res["learn_bce"] = float(b[lm].mean()); res["learn_bce_l0"] = float(b0[lm].mean())
    res["noise_excess"] = float(max(perf[x]["bce"] - math.log(2) for x in NOISE))
    res["all_bce_prequential"] = float(bce_np(zc, Y)[M].mean())
    if a.arm == "L0":
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, f"{a.arm}-s{a.seed}.json"), "w"), indent=1)
        print(json.dumps({k: res[k] for k in ("arm", "seed", "learn_bce", "noise_excess")})); return

    def loo_logit(Cx, Sx, L0x):
        # market logit with head i removed, [n,H,Q]
        tot = (Sx * Cx).sum(1, keepdims=True)
        if a.arm in ("U", "Cs"):
            return L0x[:, None, :] + tot - Cx
        den = 1 - Sx
        return L0x[:, None, :] + (tot - Sx * Cx) / np.where(den > 1e-12, den, 1.0) * (den > 1e-12)

    def measures(sl):
        Cx, Sx, L0x, Yx, Mx = C[sl], ST[sl], L0[sl], Y[sl], M[sl]
        z = L0x + (Sx * Cx).sum(1)
        p = 1 / (1 + np.exp(-z))
        G = (-Cx * Sx * (p - Yx)[:, None, :] * Mx[:, None, :]).sum(0)
        zl = loo_logit(Cx, Sx, L0x)
        S = ((logp_np(z, Yx)[:, None, :] - logp_np(zl, Yx[:, None, :])) * Mx[:, None, :]).sum(0)
        Kc = ((logp_np(L0x[:, None, :] + Cx, Yx[:, None, :]) - logp_np(L0x, Yx)[:, None, :]) * Mx[:, None, :]).sum(0)
        n = Mx.sum(0)[None, :].clip(1)
        if a.arm == "Cs":
            pre = np.cumsum(Cx, 1) - Cx
            Kc = ((logp_np(L0x[:, None, :] + pre + Cx, Yx[:, None, :]) - logp_np(L0x[:, None, :] + pre, Yx[:, None, :])) * Mx[:, None, :]).sum(0)
        return G, S, Kc, n
    G, S, Kb, _ = measures(slice(0, cut))
    _, Dsum, _, nf = measures(fw)
    Delta = Dsum / nf                                # mean BCE increase when i is removed, future window
    _, _, Kf, _ = measures(fw)
    Cf, L0f, Yf, Mf = C[fw], L0[fw], Y[fw], M[fw]
    Dalone = ((bce_np(L0f, Yf)[:, None, :] - bce_np(L0f[:, None, :] + Cf, Yf[:, None, :])) * Mf[:, None, :]).sum(0) / nf
    live = (M[fw].sum(0) > 0)                        # questions scored in the future window
    out = {}
    for nm, X in (("G", G), ("S", S), ("K", Kb)):
        x, dl = X[:, live].ravel(), Delta[:, live].ravel()
        useful = dl > 0
        out[nm] = {"spearman": float(spearmanr(x, dl).correlation),
                   "sign_agree_useful": float((x[useful] > 0).mean()) if useful.any() else None,
                   "frac_positive": float((x > 0).mean())}
    res["measures_vs_oracle"] = out

    def within(X, D):
        rs = []
        for q in np.nonzero(live)[0]:
            if np.std(X[:, q]) > 0 and np.std(D[:, q]) > 0:
                rs.append(spearmanr(X[:, q], D[:, q]).correlation)
        return {"mean": float(np.nanmean(rs)) if rs else None, "n_q": len(rs)}
    wq = {}
    for nm, X in (("G", G), ("S", S), ("K", Kb)):
        wq[nm] = {"vs_loo": within(X, Delta), "vs_alone": within(X, Dalone)}
    wq["K_future_vs_loo"] = within(Kf, Delta)
    res["within_question_spearman"] = wq
    res["pooled_vs_alone"] = {nm: float(spearmanr(X[:, live].ravel(), Dalone[:, live].ravel()).correlation) for nm, X in (("G", G), ("S", S), ("K", Kb))}
    res["dalone_mean"] = float(Dalone[:, live].mean())
    res["n_traders"] = int(live.sum() * nh)
    res["frac_useful"] = float((Delta[:, live] > 0).mean())
    res["delta_mean"] = float(Delta[:, live].mean())
    # Gini of committed wealth K vs a shuffled-label null (predictions held fixed)
    bw = slice(0, cut)
    Cb, L0b, Mb = C[bw], L0[bw], M[bw]

    def kfn(yy):
        return (((logp_np(L0b[:, None, :] + Cb, yy[:, None, :]) - logp_np(L0b, yy)[:, None, :]) * Mb[:, None, :]).sum(0))[:, live]
    groups = np.broadcast_to(np.arange(Q)[None, :], Y[bw].shape)
    res["gini_K"] = panel.gini_null(kfn, Y[bw], n_null=20, seed=a.seed, groups=groups)
    # stake concentration (Cw) at the end: mean over questions of max stake
    res["final_max_stake_mean"] = float(ST[-1].max(0)[live].mean())
    # head diversity: mean pairwise corr of head corrections over the future window (per question avg)
    cf = C[fw][:, :, live]
    cc = []
    for q in range(cf.shape[2]):
        X = cf[:, :, q]
        if X.std(0).min() > 1e-8:
            r = np.corrcoef(X.T); cc.append(r[np.triu_indices(nh, 1)].mean())
    res["head_corr_mean"] = float(np.mean(cc)) if cc else None
    res["corr_sum_magnitude"] = float(np.abs((ST * C).sum(1)[fw][M[fw]]).mean())
    os.makedirs(a.out, exist_ok=True)
    zb = L0[bw] + (ST[bw] * C[bw]).sum(1)
    res["book_market_gain"] = float(((logp_np(zb, Y[bw]) - logp_np(L0[bw], Y[bw])) * M[bw]).sum())
    res["book_sumK"] = float(Kb.sum())
    np.savez_compressed(os.path.join(a.out, f"{a.arm}-s{a.seed}.npz"), G=G, S=S, K=Kb, Delta=Delta,
                        Dalone=Dalone, live=live, fam=fam)
    json.dump(res, open(os.path.join(a.out, f"{a.arm}-s{a.seed}.json"), "w"), indent=1)
    print(json.dumps({"arm": a.arm, "seed": a.seed, "learn_bce": res["learn_bce"], "noise": res["noise_excess"],
                      "G": out["G"]["spearman"], "S": out["S"]["spearman"], "K": out["K"]["spearman"], "Kwq": wq["K"]["vs_loo"]["mean"], "Gwq": wq["G"]["vs_loo"]["mean"],
                      "train_s": res["train_s"]}))


if __name__ == "__main__":
    main()
