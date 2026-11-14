"""Train the question transformer and keep the books.

modes:
  train   train N steps (Adam), books every step, checkpoint every --ckpt steps
  frozen  load the final weights of a train run, no updates, books for N steps
  random  fresh random init, no updates, books for N steps
Output: <out>/<tag>.npz (cumulative wealth tables per checkpoint etc.) and <tag>.pt weights.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from streams import make_questions, N_FAM, FAMILIES
from model import Net, Logistic
from books import account, grad_norms, account_logistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "frozen", "random"])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--Q", type=int, default=64)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--ckpt", type=int, default=100)
    ap.add_argument("--cost", type=float, default=0.0, help="compute cost per activation per example")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-min", type=float, default=58.0, help="stop after this many minutes")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--keep-every", type=int, default=5000, help="keep a weights copy every N steps (weights/<tag>-stepN.pt)")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tag = a.tag or f"{a.mode}-s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    bits, fam, desc = make_questions(a.Q, a.T, a.seed)
    bits_t = torch.tensor(bits, dtype=torch.float32, device=dev)
    wins = bits_t.unfold(1, a.W, 1)                      # [Q, T-W+1, W]; wins[:, i] = bits[i:i+W]
    n_pos = a.T - a.W                                    # predict bits[i+W] from wins[:, i]
    split = int(n_pos * 0.8)
    fam_oh = F.one_hot(torch.tensor(fam, device=dev), N_FAM).float()
    g = torch.Generator(device="cpu").manual_seed(a.seed + 1000 * {"train": 0, "frozen": 1, "random": 2}[a.mode])

    def batch(idx):
        idx = idx.to(dev)
        return wins[:, idx].permute(1, 0, 2).contiguous(), bits_t[:, idx + a.W].T.contiguous()

    ev_idx = torch.arange(split, n_pos, max(1, (n_pos - split) // 2048))
    ev_win, ev_y = batch(ev_idx)

    model = Net(a.Q, a.W, a.d, a.H, a.L).to(dev)
    if a.mode == "frozen":
        model.load_state_dict(torch.load(a.init_from, map_location=dev))
    lm = Logistic(a.Q, a.W).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    opt_l = torch.optim.Adam(lm.parameters(), lr=1e-2)
    train = a.mode == "train"

    L, H, N = a.L, a.H, 4 * a.d
    nb = len(model.bias_params())
    W_ = {"out": torch.zeros(a.Q, N_FAM, device=dev), "head": torch.zeros(L, H, N_FAM, device=dev),
          "neuron": torch.zeros(L, N, N_FAM, device=dev), "comp": torch.zeros(2 + 2 * L, N_FAM, device=dev),
          "cut": torch.zeros(2 * L + 1, N_FAM, device=dev), "bias": torch.zeros(nb, N_FAM, device=dev),
          "R": torch.zeros(N_FAM, device=dev)}
    GN = {"out": torch.zeros(a.Q, device=dev), "head": torch.zeros(L, H, device=dev),
          "neuron": torch.zeros(L, N, device=dev), "comp": torch.zeros(2 + 2 * L, device=dev)}
    ACT = torch.zeros(L, N, device=dev)
    LW = {"lag": torch.zeros(a.Q, a.W, device=dev), "bias": torch.zeros(a.Q, device=dev)}
    hist = {k: [] for k in ["step", "out", "head", "neuron", "comp", "cut", "bias", "R", "gn_out", "gn_head",
                            "gn_neuron", "gn_comp", "act", "lag", "lbias", "ev_bce", "ev_brier", "ev_bce_log",
                            "ev_brier_log", "frozen_attn_resid", "frozen_attn_sum_resid"]}
    per_step = {"R": [], "loss": [], "cons_resid": [], "sum_resid": [], "log_resid": []}

    def fam_metric(z, y):
        p = torch.sigmoid(z)
        bce = F.binary_cross_entropy_with_logits(z, y, reduction="none").mean(0)
        br = ((p - y) ** 2).mean(0)
        return (fam_oh.T @ bce / fam_oh.sum(0)).cpu().numpy(), (fam_oh.T @ br / fam_oh.sum(0)).cpu().numpy()

    snapdir = os.path.join(a.out, tag, "snapshots")
    os.makedirs(snapdir, exist_ok=True)
    prev = {"flat": None, "win_flat": None, "hist_flat": []}

    def flat_units():
        # every unit at every granularity, total over families
        return np.concatenate([W_["out"].sum(-1).cpu().numpy(), W_["head"].sum(-1).cpu().numpy().ravel(),
                               W_["neuron"].sum(-1).cpu().numpy().ravel(), W_["comp"].sum(-1).cpu().numpy()])

    def gran_slices():
        n = [a.Q, L * H, L * N, 2 + 2 * L]
        o = np.cumsum([0] + n)
        return {"out": slice(o[0], o[1]), "head": slice(o[1], o[2]), "neuron": slice(o[2], o[3]), "comp": slice(o[3], o[4])}

    def gini_pos(x):
        x = np.sort(np.clip(x, 0, None))
        if x.sum() <= 0:
            return 0.0
        n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

    def spearman(x, y):
        if x is None or y is None or len(x) < 3:
            return None
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        c = np.corrcoef(rx, ry)[0, 1]
        return None if np.isnan(c) else float(c)

    def r5(x):
        return np.round(np.asarray(x, dtype=np.float64), 6).tolist()

    def snapshot(step):
        fl = flat_units()
        win_fl = fl - (prev["flat"] if prev["flat"] is not None else 0)
        sl = gran_slices()
        hist_fl = prev["hist_flat"]
        old = hist_fl[-10] if len(hist_fl) >= 10 else None
        k0 = max(0, len(per_step["loss"]) - a.ckpt)
        snap = {"tag": tag, "mode": a.mode, "seed": a.seed, "step": step, "elapsed_s": round(time.time() - t0, 1),
                "loss": float(torch.stack(per_step["loss"][k0:]).mean()),
                "R_per_step": r5(torch.stack(per_step["R"][k0:]).cpu().numpy()),
                "cons_resid_rel_mean": r5((torch.stack(per_step["cons_resid"][k0:]) /
                                          torch.stack(per_step["R"][k0:]).abs().clamp_min(1e-9).unsqueeze(1)).mean(0).cpu().numpy()),
                "sum_resid_max": float(torch.stack(per_step["sum_resid"][k0:]).abs().max()),
                "frozen_attn_resid_rel": r5(hist["frozen_attn_resid"][-1]),
                "ev_bce": r5(hist["ev_bce"][-1]), "ev_brier": r5(hist["ev_brier"][-1]),
                "ev_bce_log": r5(hist["ev_bce_log"][-1]), "ev_brier_log": r5(hist["ev_brier_log"][-1]),
                "wealth": {k: r5(fl[v]) for k, v in sl.items()},
                "income_window": {k: r5(win_fl[v]) for k, v in sl.items()},
                "wealth_by_family": {k: r5(W_[k].sum(tuple(range(W_[k].dim() - 1))).cpu().numpy()) for k in ["out", "head", "neuron", "comp"]},
                "comp_by_family": r5(W_["comp"].cpu().numpy()), "head_by_family": r5(W_["head"].cpu().numpy()),
                "cut": r5(W_["cut"].sum(-1).cpu().numpy()), "R_total": float(W_["R"].sum()),
                "bias": r5(W_["bias"].sum(-1).cpu().numpy()),
                "gini_pos": {k: gini_pos(fl[v]) for k, v in sl.items()},
                "n_negative": {k: int((fl[v] < 0).sum()) for k, v in sl.items()},
                "turnover_income_rank_corr": {k: spearman(win_fl[v], prev["win_flat"][v] if prev["win_flat"] is not None else None) for k, v in sl.items()},
                "wealth_rank_corr_vs_1000_steps_ago": {k: spearman(fl[v], old[v] if old is not None else None) for k, v in sl.items()},
                "act_per_example_cum": r5(ACT.cpu().numpy()) if step % (a.ckpt * 10) == 0 else None,
                "gradnorm_cum": {k: r5(GN[k].cpu().numpy()) for k in GN},
                "cost": a.cost}
        if step % (a.ckpt * 10) == 0 or step == a.steps:
            snap["neuron_by_family"] = r5(W_["neuron"].cpu().numpy())
        prev["win_flat"] = win_fl; prev["flat"] = fl; hist_fl.append(fl)
        js = json.dumps(snap)
        with open(os.path.join(snapdir, f"step-{step:05d}.json"), "w") as f:
            f.write(js)
        snap["neuron_by_family"] = r5(W_["neuron"].cpu().numpy())
        snap["act_per_example_cum"] = r5(ACT.cpu().numpy())
        tmp = os.path.join(a.out, tag, "latest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, os.path.join(a.out, tag, "latest.json"))

    def checkpoint(step):
        hist["step"].append(step)
        for k in ["out", "head", "neuron", "comp", "cut", "bias", "R"]:
            hist[k].append(W_[k].cpu().numpy().copy())
        for k in GN:
            hist["gn_" + k].append(GN[k].cpu().numpy().copy())
        hist["act"].append(ACT.cpu().numpy().copy())
        hist["lag"].append(LW["lag"].cpu().numpy().copy()); hist["lbias"].append(LW["bias"].cpu().numpy().copy())
        with torch.no_grad():
            b1, b2 = fam_metric(model(ev_win), ev_y); b3, b4 = fam_metric(lm(ev_win), ev_y)
        hist["ev_bce"].append(b1); hist["ev_brier"].append(b2); hist["ev_bce_log"].append(b3); hist["ev_brier_log"].append(b4)
        # exact check of the implementation: frozen attention => conservation up to float error
        chk = account(model, ev_win[:256], ev_y[:256], fam_oh, freeze_attn=True)
        hist["frozen_attn_resid"].append(chk["cons_resid"].cpu().numpy() / max(1e-12, abs(float(chk["R"].sum()))))
        hist["frozen_attn_sum_resid"].append(chk["sum_resid"].cpu().numpy())
        snapshot(step)
        if step % (a.ckpt * 10) == 0:
            save(step)
        if train and a.keep_every and step % a.keep_every == 0:
            os.makedirs(os.path.join(a.out, "weights"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(a.out, "weights", f"{tag}-step{step}.pt"))

    def save(last):
        torch.save(model.state_dict(), os.path.join(a.out, f"{tag}.pt"))
        arr = {k: np.array(v) for k, v in hist.items()}
        for k, v in per_step.items():
            arr["ps_" + k] = torch.stack(v).cpu().numpy()
        arr["fam"] = fam
        arr["bias_names"] = np.array([n for n, _, _ in model.bias_params()])
        np.savez_compressed(os.path.join(a.out, f"{tag}.npz"), **arr)
        meta = dict(vars(a), tag=tag, steps_done=last, seconds=time.time() - t0, desc=desc,
                    device_name=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
        json.dump(meta, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)

    t0 = time.time()
    last = 0
    for step in range(a.steps):
        idx = torch.randint(0, split, (a.B,), generator=g)
        win, y = batch(idx)
        if train:
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(win), y)
            loss.backward()
            gn = grad_norms(model)
            for k in GN:
                GN[k] += gn[k]
        acc = account(model, win, y, fam_oh)          # books on the pre-update weights
        if train:
            opt.step()
        for k in ["out", "head", "neuron", "comp", "cut", "bias", "R"]:
            W_[k] += acc[k]
        ACT.add_(acc["act_count"])
        # baseline
        zl = lm(win)
        lacc = account_logistic(lm, win, y, fam_oh)
        LW["lag"] += lacc["lag"]; LW["bias"] += lacc["bias"]
        if train:
            opt_l.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(zl, y).backward()
            opt_l.step()
        per_step["R"].append(acc["R"].sum()); per_step["loss"].append(acc["loss"])
        per_step["cons_resid"].append(acc["cons_resid"]); per_step["sum_resid"].append(acc["sum_resid"])
        per_step["log_resid"].append(lacc["cons_resid"])
        last = step + 1
        if last % a.ckpt == 0:
            checkpoint(last)
            if last % (a.ckpt * 5) == 0:
                el = time.time() - t0
                print(f"[{tag}] step {last} loss {float(acc['loss']):.4f} R {float(acc['R'].sum()):.4f} "
                      f"resid {acc['cons_resid'].cpu().numpy().round(4)} {el/last*1000:.1f} ms/step", flush=True)
            if (time.time() - t0) / 60 > a.max_min:
                print(f"[{tag}] time budget reached at step {last}", flush=True)
                break
    if last % a.ckpt:
        checkpoint(last)
    el = time.time() - t0
    save(last)
    print(f"[{tag}] done {last} steps in {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
