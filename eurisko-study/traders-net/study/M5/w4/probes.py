"""M5 worker 4: online MLP probes as class-C shadow traders (see PREREG.md).

Usage:
  python3 probes.py --source logistic --seed 1
  python3 probes.py --source wn --ckpt 8000|16000|24000|final|random --seed 1
Writes out/<tag>.json (summary) and out/<tag>-ec.csv (E_T(t) per probe, every 50 steps).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
WN = "/home/agent/workspaces/eurisko-study/wealth-net"
S0 = "/home/agent/workspaces/eurisko-study/traders-net/study/S0/substrate"
sys.path.insert(0, WN)
from streams import make_questions, FAMILIES, N_FAM  # noqa: E402
from model import Net, Logistic  # noqa: E402
sys.path.insert(1, S0)
import panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {"S": 8, "M": 32, "L": 128}
EPS = 1e-3
CLIP = 8.0


class Probe(nn.Module):
    def __init__(self, nin, h):
        super().__init__()
        self.f1, self.f2, self.f3 = nn.Linear(nin, h), nn.Linear(h, h), nn.Linear(h, 1)
        nn.init.zeros_(self.f3.weight); nn.init.zeros_(self.f3.bias)

    def forward(self, feat, lp, fam_oh):
        x = torch.cat([feat, lp.unsqueeze(-1) / 4.0, fam_oh], -1)
        return self.f3(F.relu(self.f2(F.relu(self.f1(x))))).squeeze(-1)  # correction d


def logloss(z, y):
    return F.binary_cross_entropy_with_logits(z, y, reduction="none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["logistic", "wn", "preq"])
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--feat", default="win", choices=["win", "hid"])
    ap.add_argument("--member", type=int, default=1)
    ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--maxn", type=int, default=0, help="smoke test: stop after this many steps")
    ap.add_argument("--preq_n", type=int, default=32000, help="preq source: first N positions (PREREG A4)")
    a = ap.parse_args()
    Q, W, T = 64, 32, a.T
    tag = ("smoke-" if a.maxn else "") + f"{a.source}{'' if a.source == 'logistic' else '-' + a.ckpt}-{a.feat}-s{a.seed}"
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bits, fam, _ = make_questions(Q, T, a.seed)
    bits_t = torch.tensor(bits, dtype=torch.float32)
    wins = bits_t.unfold(1, W, 1)  # [Q, T-W+1, W]
    n_pos = T - W
    split = int(n_pos * 0.8)
    fam_t = torch.tensor(fam)
    fam_oh = F.one_hot(fam_t, N_FAM).float()

    # ---- price source
    if a.source == "preq":
        f = np.load(f"/tmp/traders-net/S0/w1/{a.ckpt}-s{a.seed}.npz")
        Zp = torch.tensor(f["z"], dtype=torch.float32)
        Yp = torch.tensor(f["y"], dtype=torch.float32)
        fam = f["fam"].astype(int); fam_oh = F.one_hot(torch.tensor(fam), N_FAM).float()
        n = Zp.shape[0]
        pad = torch.cat([torch.full((W, Q), 0.5), Yp])      # [W+n, Q]
        preq_win = pad.unfold(0, W, 1)[:n]                  # [n, Q, W]: y[t-W..t-1]
        times = torch.arange(n); lm = None; HIDp = None
    elif a.source == "wn":
        pos = torch.arange(split, n_pos)
        path = {"final": f"{WN}/results/train-s{a.seed}.pt", "random": f"{WN}/results/random-s{a.seed}.pt"}.get(
            a.ckpt, f"{WN}/results/weights/train-s{a.seed}-step{a.ckpt}.pt")
        net = Net(Q, W, 64, 4, 2)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        Z, HID = [], []
        with torch.no_grad():
            for i in range(0, len(pos), 512):
                idx = pos[i:i + 512]
                w = wins[:, idx].permute(1, 0, 2).contiguous()
                rec = {}
                Z.append(net(w, rec=rec))
                # final LN output = what the readout reads
                HID.append(net.lnf(rec["cuts"][-1]))
        Zp = torch.cat(Z)  # [n,Q]
        HIDp = torch.cat(HID)
        times = pos
        lm = None
    else:
        times = torch.arange(0, n_pos)
        lm = Logistic(Q, W); opt_l = torch.optim.Adam(lm.parameters(), lr=1e-2)
        Zp = HIDp = None
    Y = Yp if a.source == "preq" else bits_t[:, times + W].T.contiguous()  # [n,Q]
    n = len(times) if not a.maxn else min(a.maxn, len(times))
    if a.source == "preq":
        n = min(n, a.preq_n)

    # ---- probes
    feat_dim = W if a.feat == "win" else 64
    nin = feat_dim + 1 + N_FAM
    arms = {k: Probe(nin, h) for k, h in SIZES.items()}
    arms["P"] = Probe(nin, SIZES["M"])  # price-only control
    opts = {k: torch.optim.Adam(m.parameters(), lr=1e-3) for k, m in arms.items()}
    E = {k: torch.zeros(Q) for k in arms}          # per-question cumulative E_T (shadow)
    E_fam = {k: np.zeros(N_FAM) for k in arms}
    # member market (probe L at eps, fixed-point clearing), one market per question
    mem = {"logw_T": torch.full((Q,), math.log(EPS)), "E_clear": torch.zeros(Q), "E_traded": torch.zeros(Q),
           "ll_market": 0.0, "ll_p0": 0.0}
    mprobe = Probe(nin, SIZES["L"]); mopt = torch.optim.Adam(mprobe.parameters(), lr=1e-3)
    torch.manual_seed(a.seed + 777); mprobe.load_state_dict(arms["L"].state_dict())
    rows, ll_price = [], 0.0
    t0 = time.time()
    for i in range(n):
        y = Y[i]
        if a.source == "preq":
            zp = Zp[i]; feat = preq_win[i] * 2 - 1
        elif a.source == "wn":
            zp = Zp[i]; feat = wins[:, times[i]] * 2 - 1 if a.feat == "win" else HIDp[i]
        else:
            w = wins[:, times[i]]
            zl = lm(w.unsqueeze(0)).squeeze(0)
            zp = zl.detach(); feat = w * 2 - 1
        lp = zp.clamp(-CLIP, CLIP)
        lpp = logloss(lp, y)
        ll_price += float(lpp.sum())
        for k, m in arms.items():
            f_in = torch.zeros_like(feat) if k == "P" else feat
            d = m(f_in, lp, fam_oh)
            lq = logloss(lp + d, y)
            gain = (lpp - lq).detach()
            E[k] += gain
            E_fam[k] += (fam_oh.T @ gain).numpy()
            opts[k].zero_grad(); lq.mean().backward(); opts[k].step()
        if a.member:
            with torch.no_grad():
                share = torch.exp(mem["logw_T"])
                p0 = torch.sigmoid(lp)
                lo, hi = torch.zeros(Q), torch.ones(Q)
                for _ in range(40):
                    mid = (lo + hi) / 2
                    qm = torch.sigmoid(torch.logit(mid.clamp(1e-6, 1 - 1e-6)).clamp(-CLIP, CLIP)
                                       + mprobe(feat, torch.logit(mid.clamp(1e-6, 1 - 1e-6)).clamp(-CLIP, CLIP), fam_oh))
                    f = (1 - share) * p0 + share * qm - mid
                    lo = torch.where(f > 0, mid, lo); hi = torch.where(f > 0, hi, mid)
                ps = ((lo + hi) / 2).clamp(1e-7, 1 - 1e-7)
                zs = torch.logit(ps).clamp(-CLIP, CLIP)
                q_clear = torch.sigmoid(zs + mprobe(feat, zs, fam_oh)).clamp(1e-7, 1 - 1e-7)
                q_trad = torch.sigmoid(lp + mprobe(feat, lp, fam_oh)).clamp(1e-7, 1 - 1e-7)
                lik = lambda p: torch.where(y > 0.5, p, 1 - p)
                mem["E_clear"] += torch.log(lik(q_clear) / lik(ps))
                mem["E_traded"] += torch.log(lik(q_trad) / lik(ps))
                mem["logw_T"] = (mem["logw_T"] + torch.log(lik(q_clear) / lik(ps))).clamp(max=0.0)
                mem["ll_market"] += float(-torch.log(lik(ps)).sum()); mem["ll_p0"] += float(-torch.log(lik(p0.clamp(1e-7, 1 - 1e-7))).sum())
            dm = mprobe(feat, zs, fam_oh)
            lqm = logloss(zs + dm, y)
            mopt.zero_grad(); lqm.mean().backward(); mopt.step()
        if lm is not None:
            opt_l.zero_grad(); logloss(zl, y).mean().backward(); opt_l.step()
        if i % 50 == 0 or i == n - 1:
            rows.append([i + 1] + [float(E[k].sum()) for k in arms] + [float(mem["E_clear"].sum())])
    # ---- summary
    arr = np.array(rows)
    t = arr[:, 0]
    out = {"tag": tag, "source": a.source, "ckpt": a.ckpt, "seed": a.seed, "feat": a.feat, "n_steps": n, "Q": Q,
           "price_bce": ll_price / (n * Q), "seconds": round(time.time() - t0, 1), "probes": {}}
    fam_n = np.bincount(fam, minlength=N_FAM) * n
    for j, k in enumerate(arms):
        Ek = arr[:, 1 + j]
        sl = panel.ec_slopes(t, Ek)
        out["probes"][k] = {"E_final": float(Ek[-1]), "E_per_qstep": float(Ek[-1]) / (n * Q),
                            "phi_last_half_per_qstep": sl["lin_last_half"] / Q, "loglog": sl["loglog"],
                            "E_fam_per_qstep": {FAMILIES[f]: float(E_fam[k][f] / fam_n[f]) for f in range(N_FAM)}}
    Ec = arr[:, 1:1 + len(SIZES)].max(1)
    sl = panel.ec_slopes(t, Ec)
    out["E_C"] = {"final": float(Ec[-1]), "per_qstep": float(Ec[-1]) / (n * Q), "phi_last_half_per_qstep": sl["lin_last_half"] / Q,
                  "loglog": sl["loglog"], "argmax": list(SIZES)[int(np.argmax(arr[-1, 1:1 + len(SIZES)]))]}
    if a.member:
        out["member_L"] = {"eps": EPS, "bound": math.log(1 / EPS),
                           "E_clear_max_q": float(mem["E_clear"].max()), "E_clear_sum": float(mem["E_clear"].sum()),
                           "E_traded_max_q": float(mem["E_traded"].max()),
                           "final_share_mean": float(torch.exp(mem["logw_T"]).mean()),
                           "final_share_gt_half": int((mem["logw_T"] > math.log(0.5)).sum()),
                           "market_bce": mem["ll_market"] / (n * Q), "p0_bce": mem["ll_p0"] / (n * Q)}
    json.dump(out, open(os.path.join(HERE, "out", tag + ".json"), "w"), indent=1)
    np.savetxt(os.path.join(HERE, "out", tag + "-ec.csv"), arr, delimiter=",", fmt="%.6g",
               header="t," + ",".join(f"E_{k}" for k in arms) + ",E_member_clear", comments="")
    print(json.dumps({k: out[k] for k in ["tag", "price_bce", "seconds", "E_C"]}))


if __name__ == "__main__":
    main()
