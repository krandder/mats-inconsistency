"""Evaluate a checkpoint: full trajectories to T_cap, then halting policies
applied post hoc (halting only freezes the belief, so a policy is a choice of
stopping step per example). Writes results/<name>.eval.json."""
import argparse, json, math
import numpy as np
import torch
from mresnet import MarketResNet, prefix_batch, score

EVALSETS = {  # name: (lmin, lmax, n, T_cap)
    "id_8_32": (8, 32, 2000, 64),
    "L32": (32, 32, 1000, 64),
    "L48": (48, 48, 1000, 96),
    "mix_8_64": (8, 64, 2000, 128),
    "L64": (64, 64, 1000, 128),
    "L128": (128, 128, 500, 256),
}


@torch.no_grad()
def trajectories(model, x, y, m, T, chunk=250):
    out = {k: [] for k in ("S", "exact", "g", "lam", "bits")}
    for i in range(0, len(x), chunk):
        xs, ys, ms = x[i:i + chunk], y[i:i + chunk], m[i:i + chunk]
        bs, gs, lams, _ = model(xs, ms, T)
        out["S"].append(torch.stack([score(b, ys, ms) for b in bs], 1).cpu())
        out["exact"].append(torch.stack([(((b > 0).float() == ys) | (ms == 0)).all(1) for b in bs], 1).cpu())
        out["g"].append(torch.stack(gs, 1).cpu())
        out["lam"].append(torch.stack(lams, 1).cpu())
        out["bits"].append(ms.sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def at(tr, tau):
    i = np.arange(len(tau))
    return {"steps": float(tau.mean()), "acc": float(tr["exact"][i, tau].mean()),
            "nll_bit": float((-tr["S"][i, tau] / tr["bits"]).mean())}


def priced_tau(tr, c, Tc):
    g = tr["g"][:, :Tc]              # g[:, t] predicts pay_{t+1} from state t
    stop = g <= c
    stop[:, 0] = False               # always take step 1
    stop = np.concatenate([stop, np.ones((len(g), 1), bool)], 1)
    return stop.argmax(1)


def ponder_tau(tr, q, Tc):
    lam = tr["lam"][:, :Tc].copy()    # lam[:, t-1]: halt prob after step t
    surv = np.cumprod(np.concatenate([np.ones((len(lam), 1)), 1 - lam[:, :-1]], 1), 1)
    cdf = np.cumsum(lam * surv, 1)
    hit = np.concatenate([cdf >= q, np.ones((len(lam), 1), bool)], 1)
    return np.minimum(hit.argmax(1) + 1, Tc)   # stop after step argmax+1; never reached q: stop at Tc


def evaluate(ck, device, credit=False, sets=None):
    s = torch.load(ck, map_location=device)
    a = s["args"]
    model = MarketResNet(w=a["width"]).to(device).eval()
    model.load_state_dict(s["model"])
    res = {"args": a, "sets": {}}
    for name, (lmin, lmax, n, Tc) in EVALSETS.items():
        if sets and name not in sets:
            continue
        gen = torch.Generator().manual_seed(777 + lmin * 1000 + lmax)
        x, y, m = prefix_batch(n, lmin, lmax, gen, device)
        tr = trajectories(model, x, y, m, Tc)
        pay = np.diff(tr["S"], axis=1)                   # (n, Tc)
        r = {"T_cap": Tc, "n": n}
        r["fixed"] = [at(tr, np.full(n, k)) for k in range(Tc + 1)]
        Ttr = a["T"]
        r["monotone_frac_train_T"] = float((pay[:, :Ttr] >= -1e-3).all(1).mean())
        r["monotone_frac_cap"] = float((pay >= -1e-3).all(1).mean())
        r["monotone_frac_cap_tol01"] = float((pay >= -1e-2).all(1).mean())
        r["neg_pay_mass_train_T"] = float(np.clip(-pay[:, :Ttr], 0, None).sum(1).mean())
        r["neg_pay_mass_cap"] = float(np.clip(-pay, 0, None).sum(1).mean())
        r["mean_pay_by_step"] = pay.mean(0).tolist()
        if a["arm"].startswith("C"):
            curve = []
            for c in np.concatenate([[-1e9], np.logspace(-4, 1.5, 34)]):
                tau = priced_tau(tr, c, Tc)
                d = at(tr, tau)
                d["cost"] = float(c)
                i = np.arange(n)
                d["post_halt_gain"] = float((tr["S"][:, -1] - tr["S"][i, tau]).mean())
                # E: random halting with the same step histogram
                rng = np.random.default_rng(0)
                d["E_perm"] = at(tr, rng.permutation(tau))
                p = 1.0 / max(d["steps"], 1.0)
                d["E_geom"] = at(tr, np.minimum(rng.geometric(p, n), Tc))
                d["tau_by_len"] = {int(L): float(tau[tr["bits"] == L].mean())
                                   for L in np.unique(tr["bits"])[::4]}
                curve.append(d)
            r["priced"] = curve
            # operating points at the train-time cost levels of the sweep (0.05 = default Ctt runs)
            r["priced_at"] = {}
            for c in [0.01, 0.05, 0.2]:
                tau = priced_tau(tr, c, Tc)
                d = at(tr, tau)
                rng = np.random.default_rng(0)
                d["E_perm"] = at(tr, rng.permutation(tau))
                d["post_halt_gain"] = float((tr["S"][:, -1] - tr["S"][np.arange(n), tau]).mean())
                r["priced_at"][str(c)] = d
            g = tr["g"][:, :Tc].ravel(); rp = pay.ravel()
            qs = np.quantile(g, np.linspace(0, 1, 11))
            bins = np.clip(np.searchsorted(qs, g, side="right") - 1, 0, 9)
            r["calib"] = {"pred": [float(g[bins == k].mean()) for k in range(10)],
                          "real": [float(rp[bins == k].mean()) for k in range(10)],
                          "corr": float(np.corrcoef(g, rp)[0, 1]),
                          "mae": float(np.abs(g - rp).mean())}
            gt = tr["g"][:, :Ttr].ravel(); pt = pay[:, :Ttr].ravel()
            r["calib"]["corr_within_train_T"] = float(np.corrcoef(gt, pt)[0, 1])
        if a["arm"] == "D":
            curve = []
            for q in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]:
                tau = ponder_tau(tr, q, Tc)
                d = at(tr, tau); d["q"] = q
                d["tau_by_len"] = {int(L): float(tau[tr["bits"] == L].mean())
                                   for L in np.unique(tr["bits"])[::4]}
                curve.append(d)
            r["ponder"] = curve
        # oracle: earliest step that reaches the example's best exact / log score
        best = tr["S"].max(1, keepdims=True)
        r["oracle_first_best"] = at(tr, (tr["S"] >= best - 1e-3).argmax(1))
        res["sets"][name] = r
    if credit:
        res["credit"] = credit_ig(model, a, device)
    return res


def credit_ig(model, a, device, n=64, nalpha=64):
    """Aumann-Shapley / integrated-gradients credit. Players: the h-write and
    the b-write of each step (2T players). Value of payment t:
    pay_t(alpha) = S(b_t) - S(b_{t-1}) with every write scaled by alpha.
    Straight path alpha: 0 -> 1 (midpoint Riemann). pay_t(0) = 0, so credits
    sum to the realized payment (completeness). Returns M_h[s,t], M_b[s,t]:
    credit of step s's h-/b-write for step t's payment (mean over examples)."""
    T = a["T"]
    gen = torch.Generator().manual_seed(4242)
    x, y, m = prefix_batch(n, a["lmin"], a["lmax"], gen, device)
    Mh = torch.zeros(T, T); Mb = torch.zeros(T, T)
    for k in range(nalpha):
        s = (k + 0.5) / nalpha
        ah = torch.full((T,), s, device=device, requires_grad=True)
        ab = torch.full((T,), s, device=device, requires_grad=True)
        bs, _, _, _ = model(x, m, T, alpha_h=ah, alpha_b=ab)
        S = torch.stack([score(b, y, m).mean() for b in bs])
        pays = S[1:] - S[:-1]
        for t in range(T):
            gh, gb = torch.autograd.grad(pays[t], [ah, ab], retain_graph=t < T - 1)
            Mh[:, t] += gh.cpu() / nalpha   # times (alpha_end - alpha_start) = 1
            Mb[:, t] += gb.cpu() / nalpha
    with torch.no_grad():
        bs, _, _, _ = model(x, m, T)
        S = torch.stack([score(b, y, m).mean() for b in bs]).cpu()
    realized = (S[1:] - S[:-1])
    tot_h, tot_b = Mh.sum().item(), Mb.sum().item()
    # lag profile of h-credit: credit of h-write s for payment at t, lag = t - s
    lag = [float(sum(Mh[s, s + L] for s in range(T - L))) for L in range(T)]
    return {"Mh": Mh.tolist(), "Mb": Mb.tolist(), "realized_pay": realized.tolist(),
            "completeness_err": float((Mh.sum(0) + Mb.sum(0) - realized).abs().max()),
            "share_h": tot_h / (tot_h + tot_b), "share_b": tot_b / (tot_h + tot_b),
            "h_credit_by_lag": lag,
            "own_b_write_share": float(torch.diag(Mb).sum() / realized.sum()),
            "h_credit_by_writer": Mh.sum(1).tolist(), "b_credit_by_writer": Mb.sum(1).tolist()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--credit", action="store_true")
    p.add_argument("--sets", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    for ck in a.ckpts:
        r = evaluate(ck, a.device, credit=a.credit, sets=a.sets.split(",") if a.sets else None)
        out = ck.replace(".pt", ".eval.json")
        json.dump(r, open(out, "w"))
        for k, v in r["sets"].items():
            f = v["fixed"]; Tt = r["args"]["T"]
            print(ck, k, "acc@Ttrain", round(f[Tt]["acc"], 4), "acc@cap", round(f[-1]["acc"], 4),
                  "best", round(max(d["acc"] for d in f), 4), "mono", round(v["monotone_frac_cap"], 3), flush=True)
        if "credit" in r:
            c = r["credit"]
            print("credit share_h", round(c["share_h"], 3), "own_b", round(c["own_b_write_share"], 3),
                  "compl_err", c["completeness_err"])
