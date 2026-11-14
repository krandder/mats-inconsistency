"""Numerical checks for T1 lemma 3 (shared-trunk cost sharing) + FLOP budget tables for M7 and K1.
Run: python3 check_L3.py -> out/L3.json, out/flops_M7.json, out/flops_K1.json. Seeds 1,2,3."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import costshare as cs

OUT = os.path.join(HERE, "out")
SEEDS = [1, 2, 3]


def msd(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1))]


def main():
    res = {"a": {}, "b": {}, "c": {}, "d": {}}
    # (a) Euler for costs: degree-r homogeneous C -> marginal collects r*C; AS collects C - C(0)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        row = {}
        for r in [0.5, 1.0, 1.5]:
            mr, ar = [], []
            for _ in range(100):
                K = 8; a = rng.uniform(0.1, 2, K); p = rng.uniform(0.5, 3); x = rng.uniform(0.1, 5, K)
                C = lambda x: (a * x ** p).sum() ** (r / p)
                grad = lambda x: (r / p) * (a * x ** p).sum() ** (r / p - 1) * a * p * np.maximum(x, 1e-300) ** (p - 1)
                mr.append(cs.marginal(grad, x).sum() / C(x))
                ar.append(cs.aumann_shapley(grad, x, n=4096).sum() / C(x))
            row[str(r)] = {"marginal_sum_over_C": msd(mr), "AS_sum_over_C": msd(ar)}
        res["a"][seed] = row
    # (b) concave shared trunk + separable own costs, K = 8: AS vs Shapley vs marginal
    for seed in SEEDS:
        rng = np.random.default_rng(10 + seed)
        K = 8; F = 100.0; beta = 0.7
        c = rng.uniform(0.5, 2, K); x = rng.lognormal(0, 1, K)
        C = lambda x: F * (np.sum(x) ** beta if np.sum(x) > 0 else 0.0) + (c * x).sum()
        grad = lambda x: F * beta * max(np.sum(x), 1e-300) ** (beta - 1) * np.ones(K) + c
        AS = cs.aumann_shapley(grad, x, n=20000)
        prop = c * x + F * np.sum(x) ** beta * x / x.sum()          # closed form: average-cost split of the trunk
        SH = cs.shapley(lambda S: C(np.array([x[i] if i in S else 0.0 for i in range(K)])), K)
        MC = cs.marginal(grad, x)
        tot = C(x)
        # union (public-good) game: 12 trunk components, random user sets
        users = [list(rng.choice(K, size=rng.integers(1, K + 1), replace=False)) for _ in range(12)]
        Fu = rng.uniform(1, 10, 12)
        Cu = lambda S: sum(f for u, f in zip(users, Fu) if set(u) & set(S))
        SHu = cs.shapley(Cu, K); SHc = cs.union_shapley(users, Fu)
        res["b"][seed] = {
            "AS_budget_resid": float(AS.sum() - tot), "AS_vs_closed_form_maxabs": float(np.abs(AS - prop).max()),
            "shapley_budget_resid": float(SH.sum() - tot),
            "marginal_recovered_frac": float(MC.sum() / tot), "trunk_frac_of_cost": float(F * np.sum(x) ** beta / tot),
            "L1_AS_vs_shapley_over_total": float(np.abs(AS - SH).sum() / tot),
            "L1_marginal_vs_AS_over_total": float(np.abs(MC - AS).sum() / tot),
            "union_game_shapley_closed_form_maxabs": float(np.abs(SHu - SHc).max()),
            "union_budget_resid": float(SHc.sum() - sum(Fu))}
    # (c) M7 FLOP tables + budget balance on simulated skewed routing
    tables = {}
    for E in [8, 64]:
        for kind, h_e in [("fine_grained", None), ("full_mlp_experts", 256)]:
            fl = cs.moe_token_flops(E=E, k=2, h_e=h_e)
            active = fl["trunk"] + fl["L"] * (fl["E"] * fl["router_col"] + fl["k"] * fl["expert"])
            tables[f"E{E}_{kind}"] = dict(fl, active_flops_per_token=active,
                                          trunk_share_of_active=fl["trunk"] / active,
                                          charge_per_routed_token_shapley=fl["expert"] + fl["trunk"] / (fl["L"] * fl["k"]),
                                          rent_per_token_seen_per_expert=fl["router_col"],
                                          charge_per_routed_token_marginal=fl["expert"])
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                N = 4096; L = fl["L"]
                counts = np.zeros((L, E))
                for l in range(L):
                    pr = rng.dirichlet(np.full(E, 0.3))
                    for _ in range(N):
                        counts[l, rng.choice(E, 2, replace=False, p=pr)] += 1
                ch, tot = cs.moe_charges(counts, N, fl, "shapley")
                chm, _ = cs.moe_charges(counts, N, fl, "marginal")
                res["c"][f"E{E}_{kind}_s{seed}"] = {"budget_resid_rel": float((ch.sum() - tot) / tot),
                                                    "marginal_recovered_frac": float(chm.sum() / tot),
                                                    "dead_experts": int((counts == 0).sum()),
                                                    "max_over_mean_charge": float(ch.max() / ch.mean())}
    with open(os.path.join(OUT, "flops_M7.json"), "w") as f:
        json.dump(tables, f, indent=1)
    # (d) K1: population of pattern-k and logistic traders; budget balance, crowding effect
    k1 = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        N, Q = 1000, 64
        kinds = [("pattern", int(k)) for k in rng.choice([2, 4, 8], 800)] + [("logistic", 32)] * 200
        niche = rng.integers(0, Q, N)
        trades = np.zeros((N, Q), bool)
        trades[np.arange(N), niche] = True                         # niches on: one question each
        trades |= rng.random((N, Q)) < 0.02                        # some generalists
        ch, spent = cs.k1_charges(kinds, trades)
        per_trade = ch / np.maximum(trades.sum(1), 1)
        pat = np.array([kd == "pattern" for kd, _ in kinds])
        k1[seed] = {"budget_resid_rel": float((ch.sum() - spent) / spent),
                    "pattern_charge_per_trade_mean": float(per_trade[pat].mean()),
                    "logistic_charge_per_trade_mean": float(per_trade[~pat].mean()),
                    "logistic_full_cost_per_trade": float(sum(cs.k1_trader_flops("logistic", W=32)[k] for k in ("shared", "own")))}
    res["d"] = k1
    ktab = {"pattern2": cs.k1_trader_flops("pattern", k=2), "pattern4": cs.k1_trader_flops("pattern", k=4),
            "pattern8": cs.k1_trader_flops("pattern", k=8), "logistic32": cs.k1_trader_flops("logistic", W=32),
            "breakeven_c_nats_per_flop_at_0.1_nats_per_trade": {
                "pattern4_alone": cs.breakeven_price(0.1, 8 + 14), "logistic32_alone": cs.breakeven_price(0.1, 32 + 144),
                "logistic32_with_100_sharers": cs.breakeven_price(0.1, 32 / 100 + 144)},
            "note": "shared part is split equally among the traders that consume the same (feature, question) on a step; own part is paid in full"}
    with open(os.path.join(OUT, "flops_K1.json"), "w") as f:
        json.dump(ktab, f, indent=1)
    with open(os.path.join(OUT, "L3.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["a"][1]), "\n", json.dumps(res["b"][1]), "\n", json.dumps({k: v for k, v in res["c"].items() if k.endswith("s1")}), "\n", json.dumps(res["d"][1]))
    print(json.dumps({k: {m: v[m] for m in ("active_flops_per_token", "trunk_share_of_active", "charge_per_routed_token_shapley", "rent_per_token_seen_per_expert", "expert", "trunk")} for k, v in tables.items()}, indent=0))


if __name__ == "__main__":
    main()
