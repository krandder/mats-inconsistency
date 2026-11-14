"""T1 hawk, env-grounded check on the harness's own ShipHunt env (harness/env/shiphunt.py).
Questions: (1) is there headroom for multi-candidate search over plans at all, beyond the greedy
top-cells plan that exact cell beliefs give? (2) under a realistic two-part forecaster error
(belief-level misbelief shared by all candidates + per-plan forecast noise), how big is the curse
(gap argmax - gap random) vs n, and how correlated are candidate errors? (3) how many distinct
plans does a forecast-driven proposer produce? Single process, numpy only. Output T1_env_sim.json."""
import json, os, sys, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "harness"))
from env.shiphunt import ShipHunt, PRESETS, placements
rng = np.random.default_rng(1)
NS = [1, 2, 4, 8, 16]; NSTATE = 400; MAXSETS = 3000
out = {"nstate": NSTATE, "ns": NS}

def run_preset(name, s_belief, t_plan, temp=0.05):
    cfg = PRESETS[name]; P = placements(cfg).astype(np.int16)
    sets = list(itertools.combinations(range(cfg.n_cells), cfg.plan_len))
    if len(sets) > MAXSETS:
        sets = [sets[i] for i in rng.choice(len(sets), MAXSETS, replace=False)]
    S = np.zeros((cfg.n_cells, len(sets)), np.int16)
    for j, s in enumerate(sets): S[list(s), j] = 1
    succ = ((P @ S) >= cfg.need_hits).astype(float)            # [placements, plans]
    stats = dict(oracle=[], greedy=[], greedy_is_opt=[], median_plan=[], n_placements=int(len(P)), n_plans=len(sets))
    gap = {m: {n: [] for n in NS} for m in ("iid", "policy")}
    gapr = {m: {n: [] for n in NS} for m in ("iid", "policy")}
    tv = {m: {n: [] for n in NS} for m in ("iid", "policy")}
    distinct = {n: [] for n in NS}; corr = []
    for k in range(NSTATE):
        env = ShipHunt(cfg, seed=10_000 + k); w = env.posterior()
        pstar = w @ succ
        cp = w @ P.astype(float); top = tuple(sorted(np.argsort(-cp)[:cfg.plan_len]))
        g = pstar[sets.index(top)] if top in sets else float((w * ((P[:, list(top)].sum(1)) >= cfg.need_hits)).sum())
        stats["oracle"].append(pstar.max()); stats["greedy"].append(g)
        stats["greedy_is_opt"].append(float(g >= pstar.max() - 1e-9)); stats["median_plan"].append(float(np.median(pstar)))
        # forecaster: misbelief shared across candidates (perturbed posterior) + per-plan noise
        logw = np.log(np.maximum(w, 1e-12)) + s_belief * rng.normal(size=len(w))
        wf = np.exp(logw - logw.max()); wf /= wf.sum()
        F = np.clip(wf @ succ + t_plan * rng.normal(size=len(sets)), 0, 1)
        E = F - pstar
        # error correlation between two random plans in this state is not identifiable from one state;
        # collect per-state mean error (shared part) for a variance decomposition below
        corr.append((E.mean(), E.var()))
        for mode in ("iid", "policy"):
            if mode == "iid":
                cand = rng.choice(len(sets), 16, replace=True)
            else:  # policy proposes plans it believes are good: softmax over its own forecast
                p = np.exp((F - F.max()) / temp); p /= p.sum(); cand = rng.choice(len(sets), 16, p=p)
            for n in NS:
                c = cand[:n]; i = c[np.argmax(F[c])]; r = c[rng.integers(n)]
                gap[mode][n].append(E[i]); gapr[mode][n].append(E[r]); tv[mode][n].append(pstar[i])
                if mode == "policy": distinct[n].append(len(set(c.tolist())))
    m = lambda a: float(np.mean(a)); se = lambda a: float(np.std(a) / np.sqrt(len(a)))
    shared = np.array(corr)
    res = dict(cfg=name, s_belief=s_belief, t_plan=t_plan, temp=temp,
               n_placements=stats["n_placements"], n_plans=stats["n_plans"],
               oracle_mean=m(stats["oracle"]), greedy_mean=m(stats["greedy"]),
               greedy_is_optimal_frac=m(stats["greedy_is_opt"]), median_plan_mean=m(stats["median_plan"]),
               shared_error_var_frac=float(shared[:, 0].var() / (shared[:, 0].var() + shared[:, 1].mean())),
               curves={mode: {n: dict(gap_argmax=m(gap[mode][n]), gap_random=m(gapr[mode][n]),
                                      curse=m(np.array(gap[mode][n]) - np.array(gapr[mode][n])),
                                      curse_se=se(np.array(gap[mode][n]) - np.array(gapr[mode][n])),
                                      trueV=m(tv[mode][n])) for n in NS} for mode in ("iid", "policy")},
               distinct_policy={n: m(distinct[n]) for n in NS})
    return res
out["runs"] = []
for name in ("easy", "medium", "hard"):
    for sb, tp in ((0.0, 0.05), (1.0, 0.0), (1.0, 0.05)):
        r = run_preset(name, sb, tp); out["runs"].append(r)
        print(name, sb, tp, "oracle %.3f greedy %.3f greedy_opt %.2f median %.3f shared %.2f" % (
            r["oracle_mean"], r["greedy_mean"], r["greedy_is_optimal_frac"], r["median_plan_mean"], r["shared_error_var_frac"]))
        for mode in ("iid", "policy"):
            print("  ", mode, " ".join("n%d curse %.3f±%.3f tv %.3f" % (n, r["curves"][mode][n]["curse"], r["curves"][mode][n]["curse_se"], r["curves"][mode][n]["trueV"]) for n in NS))
        print("   distinct(policy)", r["distinct_policy"], flush=True)
json.dump(out, open(os.path.join(HERE, "T1_env_sim.json"), "w"), indent=1)
