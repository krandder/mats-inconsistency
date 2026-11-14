"""Experiment B: the rescaling check.

For a relu unit, (in *= c, bias *= c, out /= c) computes exactly the same
function.  Split each hidden unit h into an in-agent (W1[h], b1[h]) and an
out-agent (v[h]); 2H players.  For unit h rescaled by c in {0.1, 1, 10}:

 (a) counterfactual pay over the 2H in/out agents, on held-out data:
     Shapley (K=200 sampled orders, same orders for every c), LOO, and the
     mean-ablation variant.  Expected invariant.
 (b) magnitude schemes:
     act_share  the in-agents' pot (their total Shapley at c=1, held fixed) is
                split in proportion to delivered activation E|a_h|
     reader_pays  out-agent h pays in-agent h a fixed price per unit E|a_h|
                (price set so the transfer equals the act_share pay at c=1)
     l2_in / l2_out  L2 compute cost lam*||w||^2 on each agent's weights
                (lam set so the total cost at c=1 is 10% of the pot)
 (c) fix: normalized goods.  a_hat = a_h / s_h with s_h the std of a_h on a
     reference batch, v' = v_h * s_h (function unchanged).  Magnitude schemes
     recomputed on a_hat and v'.  Two cases: stats recomputed after the
     rescale ("fresh"), and stats frozen from before it ("frozen").
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from core import make_data, mean_ci
from exp_mlp import Net1, score, shap_from_orders

HERE = os.path.dirname(os.path.abspath(__file__))
H, T, BATCH, LR = 16, 1500, 250, 0.1
CS = (0.1, 1.0, 10.0)


def train(seed):
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(1000 + seed)
    net = Net1(12, H, rng, "relu", np.ones((H, 12)))
    net.v = rng.standard_normal(H) * 0.3
    for _ in range(T):
        x, y, _ = make_data(BATCH, drng)
        c, a, da = net.contrib(x)
        r = y - 1 / (1 + np.exp(-(net.b0 + c.sum(1))))
        gpre = r[:, None] * net.v * da
        net.v += LR * (r @ a) / BATCH
        net.W1 += LR * gpre.T @ x / BATCH
        net.b1 += LR * gpre.sum(0) / BATCH
    return net


def game(a, v, y, b0, cbar=None):
    """value of in/out-agent coalitions (2H players: 0..H-1 in, H..2H-1 out)."""
    c = a * v

    def vf(M):
        U = M[:, :H] * M[:, H:]                          # unit works iff both halves present
        z = b0 + c @ U.T
        if cbar is not None:
            z = z + (1 - U) @ cbar
        return (score(z, y[:, None]) - score(b0, y)[:, None]).sum(0) / len(y)
    return vf


def pays_at(net, h, cscale, xt, yt, xref, orders, stats_ref=None):
    W1, b1, v = net.W1.copy(), net.b1.copy(), net.v.copy()
    W1[h] *= cscale
    b1[h] *= cscale
    v[h] /= cscale
    a = np.maximum(xt @ W1.T + b1, 0)
    ar = np.maximum(xref @ W1.T + b1, 0)
    z = net.b0 + (a * v).sum(1)
    vf = game(a, v, yt, net.b0)
    shap = shap_from_orders(vf, orders, 2 * H)
    full = vf(np.ones((1, 2 * H)))[0]
    loo = full - vf(1 - np.eye(2 * H))
    vfm = game(a, v, yt, net.b0, cbar=ar.mean(0) * v)
    shap_mean = shap_from_orders(vfm, orders[:50], 2 * H)
    s_fresh = ar.std(0) + 1e-12
    s = s_fresh if stats_ref is None else stats_ref
    return dict(z=z, shap=shap, loo=loo, shap_mean=shap_mean, full=full,
                act=np.abs(a).mean(0), l2_in=(W1 ** 2).sum(1) + b1 ** 2, l2_out=v ** 2,
                act_n=np.abs(a / s).mean(0), l2_out_n=(v * s) ** 2, s=s)


def run(seed):
    net = train(seed)
    xt, yt, _ = make_data(3000, np.random.default_rng(4242 + seed))
    xref, _, _ = make_data(2000, np.random.default_rng(77 + seed))
    orng = np.random.default_rng(5)
    orders = [orng.permutation(2 * H) for _ in range(200)]
    base = pays_at(net, 0, 1.0, xt, yt, xref, orders)
    pot_in = base["shap"][:H].sum()
    price = pot_in / base["act"].sum()
    price_n = pot_in / base["act_n"].sum()
    lam = 0.1 * base["full"] / (base["l2_in"].sum() + base["l2_out"].sum())
    lam_n = 0.1 * base["full"] / (base["l2_in"].sum() + base["l2_out_n"].sum())
    alive = np.flatnonzero(base["act"] > 1e-6)
    units = alive[np.argsort(-base["shap"][alive] - base["shap"][H + alive])][:6]   # top-6 live units
    rows = []
    for h in units:
        rec = {"unit": int(h)}
        for cs in CS:
            for mode in ("fresh", "frozen"):
                if mode == "frozen" and cs == 1.0:
                    continue
                p = pays_at(net, h, cs, xt, yt, xref, orders, stats_ref=base["s"] if mode == "frozen" else None)
                key = f"c={cs}" if mode == "fresh" else f"c={cs}|frozen"
                act_share = pot_in * p["act"] / p["act"].sum()
                act_share_n = pot_in * p["act_n"] / p["act_n"].sum()
                others = np.arange(H) != h
                rec[key] = dict(
                    max_logit_diff=float(np.abs(p["z"] - base["z"]).max()),
                    shap_in=p["shap"][h], shap_out=p["shap"][H + h],
                    shapmean_in=p["shap_mean"][h], shapmean_out=p["shap_mean"][H + h],
                    loo_in=p["loo"][h], loo_out=p["loo"][H + h],
                    actshare_in=act_share[h], actshare_others=act_share[others].sum(),
                    readerpays_transfer=price * p["act"][h],
                    l2_in=lam * p["l2_in"][h], l2_out=lam * p["l2_out"][h],
                    n_actshare_in=act_share_n[h], n_actshare_others=act_share_n[others].sum(),
                    n_readerpays_transfer=price_n * p["act_n"][h],
                    n_l2_in=lam_n * p["l2_in"][h], n_l2_out=lam_n * p["l2_out_n"][h])
        rows.append(rec)
    return dict(seed=seed, units=rows, pot_in=pot_in, full=base["full"],
                in_out_share=float(base["shap"][:H].sum() / base["shap"].sum()))


def main():
    t0 = time.time()
    with Pool(5) as pool:
        res = pool.map(run, range(5))
    # ratios pay(c)/pay(1), per unit and seed
    metrics = ["shap_in", "shap_out", "shapmean_in", "shapmean_out", "loo_in", "loo_out",
               "actshare_in", "actshare_others", "readerpays_transfer", "l2_in", "l2_out",
               "n_actshare_in", "n_actshare_others", "n_readerpays_transfer", "n_l2_in", "n_l2_out"]
    summ = {"design": __doc__, "seeds": 5, "units_per_seed": 6, "H": H,
            "in_agents_share_of_value": mean_ci([r["in_out_share"] for r in res])}
    for key in ("c=0.1", "c=10.0", "c=0.1|frozen", "c=10.0|frozen"):
        d = {"max_logit_diff": float(max(u[key]["max_logit_diff"] for r in res for u in r["units"]))}
        for m in metrics:
            rat = [u[key][m] / u["c=1.0"][m] for r in res for u in r["units"] if abs(u["c=1.0"][m]) > 1e-9]
            # per-seed mean ratio, then CI over seeds
            per_seed = [np.mean([u[key][m] / u["c=1.0"][m] for u in r["units"] if abs(u["c=1.0"][m]) > 1e-9]) for r in res]
            d[m] = {"ratio_mean_ci": mean_ci(per_seed), "ratio_min": float(np.min(rat)), "ratio_max": float(np.max(rat))}
        summ[key] = d
    # a concrete gaming example: seed 0, top unit, c=10
    u = res[0]["units"][0]
    ex = {"unit": u["unit"]}
    for key in ("c=1.0", "c=10.0", "c=0.1"):
        ex[key] = {k: round(float(v), 5) for k, v in u[key].items()}
    summ["example_seed0_top_unit"] = ex
    summ["secs"] = round(time.time() - t0)
    json.dump(summ, open(os.path.join(HERE, "results", "rescale_summary.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # chart
    bars = []
    for m, lab in (("shap_in", "Shapley pay, in-agent"), ("loo_in", "LOO pay, in-agent"),
                   ("actshare_in", "Activation-share pay, in-agent"), ("l2_in", "L2 cost, in-agent"),
                   ("l2_out", "L2 cost, out-agent"), ("n_actshare_in", "Activation-share, normalized goods"),
                   ("n_l2_out", "L2 cost out-agent, normalized goods"), ("n_l2_in", "L2 cost in-agent, normalized goods")):
            mc = summ["c=10.0"][m]["ratio_mean_ci"]
            bars.append({"label": lab, "value": round(mc[0], 4), "ci": round(mc[1], 4),
                         "highlight": m.startswith("n_") and m != "n_l2_in"})
    json.dump({"id": "mlp_rescale_ratio", "kind": "bar",
               "title": "Rescale unit by c=10 (in x10, out /10, same function): pay or cost at c=10 / at c=1",
               "unit": "ratio (1 = invariant)", "source": "trader-economy/results/rescale_summary.json",
               "note": "relu, 16 units, top-6 units x 5 seeds; normalized goods = activation / reference std",
               "bars": bars}, open(os.path.join(HERE, "charts", "mlp_rescale_ratio.json"), "w"), indent=1)
    print("done", summ["secs"])


if __name__ == "__main__":
    main()
