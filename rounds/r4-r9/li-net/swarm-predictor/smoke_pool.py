"""CPU smoke test of pool.py and settle.py on ShipHunt with SYNTHETIC chains (no language model).

Checks the pooling and settlement math before any GPU is rented. Claims under test, fixed before
the first run:
  C1  When chains carry partially independent information (evidence overlap a < 1), the world pool
      with learned stakes (W-fit, arm d) has lower Brier vs exact Bayes than the mean (a) and the vote
      share (b) at n = 4, 16 and 64 chains: paired bootstrap 95% CI of the difference below 0.
  C2  Settlement credit tracks informativeness: the mean within-episode Spearman correlation between
      a chain's settlement s_i (paid against W-fit) and its oracle informativeness info_i is above 0
      (95% CI) at n = 4, 16 and 64, in every scenario.
  C3  The settlement is minus the stake gradient of the pool's log loss (finite differences, < 1e-6).
Added after a 40/60-board trial run, before the full run (see RESULTS-SMOKE.md): that trial showed the
naive world pool W-fit cannot reach exact Bayes even with perfect chains, because it adds the shifts
of correlated statements. So a second candidate for arm (d) was added, D-fit = the LL-fit prices
projected onto legal boards (pool.project), with these claims:
  C1b C1 for D-fit.
  C4  With full overlap (a = 1) the arm-(d) pool does not lose to the mean at n = 16 and 64 (lower end
      of the 95% CI of "pool minus mean" at or below 0); checked for both W-fit and D-fit.

Boards. Preset `medium` (5x4, ships 3 and 2, 6 noisy probes, eps 0.15) from the harness, plus
K_SHOTS noiseless shots at random unprobed cells (the revealed cells). Statements: 20 cells and
"ship of length L horizontal in row r" (8); revealed cells are not statements. Exact Bayes =
ShipHunt.posterior() with all evidence (pool.Board.bayes).

Synthetic chain i (a stand-in for one short reasoning trace):
  * reads each evidence item (probe or revealed cell) with probability a (the overlap knob:
    a = 1, every chain sees everything; a = 0.25, chains see small, partly different subsets);
  * forms the exact posterior given only what it read (the same harness oracle on a subset);
  * claims K_CLAIMS open statements, chosen with probability ∝ |its posterior - prior| + 0.02 (it
    talks about what its evidence bears on);
  * states q = sigmoid(logit(its posterior) + noise), noise ~ N(0, sigma_i^2) on the log-odds,
    sigma_i = noise_level * U(0.5, 1.5);
  * states one confidence w_i = clip(exp(N(0, 0.3^2)) / (1 + sigma_i^2), 0.05, 1): partly
    informative about its own noise, blind to how much it shares with other chains.
Chains 0..63 are drawn once per episode; the n-chain pool uses chains 0..n-1 (paired across n).
Learned stakes are fit on N_TRAIN training boards (realised outcomes only), scored on N_TEST other
boards. Outputs: smoke_results.json and smoke_tables.md next to this file.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from pool import (METHODS, Board, Claims, fit_stakes, logit, price, project, sigmoid, stakes, theta, world)
from settle import grad_check, informativeness, loo_world, settle, spearman
from env.shiphunt import PRESETS, ShipHunt

HERE = os.path.dirname(os.path.abspath(__file__))
NS = (1, 4, 16, 64)
OVERLAPS = (1.0, 0.5, 0.25)
NOISES = (0.5, 1.5)


def ece(p, y, bins=10):
    """Equal-mass binned ECE (same definition as harness metrics/calib.py: ece)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    o = np.argsort(p, kind="stable")
    return float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(o, bins) if len(c)) / len(p))


def boot(x, n=1000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def make_episode(board, cfg, seed, k_shots):
    env = ShipHunt(cfg, seed)
    rng = np.random.default_rng([seed, 11])
    probed = {c for c, _ in env.probes}
    cand = [c for c in range(cfg.n_cells) if c not in probed]
    for c in rng.choice(cand, size=k_shots, replace=False):
        env.fire(int(c))
    zstar = board.truth(env)
    opn = np.ones(board.J, bool)
    opn[[c for c, _ in env.fired]] = False          # statement index of cell c is c
    return {"env": env, "bayes": board.bayes(env) @ board.F, "zstar": zstar, "y": board.F[zstar].copy(), "open": opn}


def make_chains(board, ep, n, a, noise, k, rng):
    env = ep["env"]
    n_items = len(env.probes) + len(env.fired)
    open_idx = np.flatnonzero(ep["open"])
    cache, parts, lat = {}, [], []
    for i in range(n):
        mask = rng.random(n_items) < a
        key = mask.tobytes()
        if key not in cache:
            pi = [t for t in range(len(env.probes)) if mask[t]]
            fi = [t for t in range(len(env.fired)) if mask[len(env.probes) + t]]
            cache[key] = board.bayes_subset(env, pi, fi) @ board.F
        m = cache[key]
        sig = noise * rng.uniform(0.5, 1.5)
        pr = np.abs(m[open_idx] - board.p0_marg[open_idx]) + 0.02
        js = rng.choice(open_idx, size=k, replace=False, p=pr / pr.sum())
        q = np.clip(sigmoid(logit(np.clip(m[js], 0.02, 0.98)) + rng.normal(0, sig, k)), 0.01, 0.99)
        w = float(np.clip(np.exp(rng.normal(0, 0.3)) / (1 + sig ** 2), 0.05, 1.0))
        parts.append(Claims(np.full(k, i), js, q, np.full(k, w)))
        lat.append((sig, mask.mean()))
    return Claims.concat(parts), lat


def run(args):
    t0 = time.time()
    cfg = PRESETS[args.preset]
    board = Board(cfg)
    train = [make_episode(board, cfg, 100_000 + e, args.k_shots) for e in range(args.n_train)]
    test = [make_episode(board, cfg, 200_000 + e, args.k_shots) for e in range(args.n_test)]
    nmax = max(NS)
    floor = {"W-add": [], "D_c20": []}
    for ep in test:
        o = ep["open"]
        js = np.flatnonzero(o)
        cl = Claims(np.zeros(len(js), int), js, np.clip(ep["bayes"][js], 1e-4, 1 - 1e-4), np.ones(len(js)))
        p, _ = price(board, cl, "W-add")
        floor["W-add"].append(float(np.mean((p[o] - ep["bayes"][o]) ** 2)))
        mu, _ = project(board, ep["bayes"], o, 20.0)
        floor["D_c20"].append(float(np.mean(((mu @ board.F)[o] - ep["bayes"][o]) ** 2)))
    out_floor = {k: boot(v) for k, v in floor.items()}
    out = {"floor_one_exact_chain_all_open_statements": out_floor,"config": {"preset": args.preset, "env": cfg.to_dict(), "k_shots": args.k_shots, "k_claims": args.k_claims,
                      "n_train": args.n_train, "n_test": args.n_test, "ns": NS, "overlaps": OVERLAPS, "noises": NOISES,
                      "methods": METHODS},
           "board": {"layouts": board.Z, "patterns": board.U, "statements": board.J, "names": board.names,
                     "patterns_with_2plus_layouts": int((board.mult > 1).sum())},
           "scenarios": {}}
    for si, (a, noise) in enumerate([(a, s) for a in OVERLAPS for s in NOISES]):
        name = f"a{a:.2f}_noise{noise:.1f}"
        rng = np.random.default_rng([si, 1])
        for ep in train + test:
            ep["claims"], ep["lat"] = make_chains(board, ep, nmax, a, noise, args.k_claims, rng)
        sc = {"overlap_a": a, "noise": noise, "fit": {}, "brier_vs_bayes": {}, "logloss": {}, "ece": {},
              "diff": {}, "settle": {}}
        for n in NS:
            tr = [dict(claims=e["claims"].first_chains(n), y=e["y"], zstar=e["zstar"], open=e["open"]) for e in train]
            fits = {"LL": fit_stakes(board, tr, "stmt"), "W": fit_stakes(board, tr, "world")}
            sc["fit"][n] = fits
            se = {m: [] for m in METHODS}
            ll = {m: [] for m in METHODS}
            py = {m: ([], []) for m in METHODS}
            st = {"s": [], "s_raw": [], "pi": [], "loo": [], "s_vs_Dfit": []}
            pooled = {"s": [], "info": []}
            gmax = 0.0
            for ti, ep in enumerate(test):
                cl = ep["claims"].first_chains(n)
                o = ep["open"]
                for m in METHODS:
                    par = fits["LL"] if m in ("LL-fit", "D-fit") else fits["W"] if m == "W-fit" else None
                    p, _ = price(board, cl, m, par)
                    if m == "D-fit":
                        pD = p
                    se[m].append(float(np.mean((p[o] - ep["bayes"][o]) ** 2)))
                    pc = np.clip(p[o], 1e-3, 1 - 1e-3)
                    yo = ep["y"][o]
                    ll[m].append(float(-np.mean(yo * np.log(pc) + (1 - yo) * np.log(1 - pc))))
                    py[m][0].extend(p[o]); py[m][1].extend(yo)
                if n >= 4:
                    f = fits["W"]
                    b = stakes(board, cl, "add", f["kappa"], f["alpha"])
                    mu = world(board, theta(board, cl, b))
                    r = settle(board, cl, b, mu @ board.F, ep["y"], n)
                    info = informativeness(board, cl, ep["bayes"], n)
                    loo = loo_world(board, cl, ep["zstar"], f["kappa"], f["alpha"], n)
                    rD = settle(board, cl, b, pD, ep["y"], n)
                    for key, v in (("s", r["s"]), ("s_raw", r["s_raw"]), ("pi", r["pi"]), ("loo", loo), ("s_vs_Dfit", rD["s"])):
                        st[key].append(spearman(v, info))
                    pooled["s"].extend(r["s"]); pooled["info"].extend(info)
                    if n == 16 and ti < args.grad_eps:
                        gmax = max(gmax, grad_check(board, cl, b, ep["zstar"]))
            sc["brier_vs_bayes"][n] = {m: boot(se[m]) for m in METHODS}
            sc["logloss"][n] = {m: boot(ll[m]) for m in METHODS}
            sc["ece"][n] = {m: ece(*py[m]) for m in METHODS}
            sc["diff"][n] = {"W-fit_minus_mean": boot(np.array(se["W-fit"]) - np.array(se["mean"])),
                             "W-fit_minus_vote": boot(np.array(se["W-fit"]) - np.array(se["vote"])),
                             "W-fit_minus_LL-fit": boot(np.array(se["W-fit"]) - np.array(se["LL-fit"])),
                             "W-fit_minus_W-avg": boot(np.array(se["W-fit"]) - np.array(se["W-avg"])),
                             "D-fit_minus_mean": boot(np.array(se["D-fit"]) - np.array(se["mean"])),
                             "D-fit_minus_vote": boot(np.array(se["D-fit"]) - np.array(se["vote"])),
                             "D-fit_minus_W-fit": boot(np.array(se["D-fit"]) - np.array(se["W-fit"])),
                             "D-fit_minus_LL-fit": boot(np.array(se["D-fit"]) - np.array(se["LL-fit"]))}
            if n >= 4:
                sc["settle"][n] = {f"spearman_{k}_vs_info": boot(v) for k, v in st.items()}
                sc["settle"][n]["pooled_spearman_s_vs_info"] = spearman(pooled["s"], pooled["info"])
                sc["settle"][n]["mean_s"] = float(np.mean(pooled["s"]))
                if n == 16:
                    sc["settle"][n]["grad_check_max_abs_err"] = gmax
            print(f"{name} n={n:2d} D-fit {sc['brier_vs_bayes'][n]['D-fit'][0]:.4f} W-fit {sc['brier_vs_bayes'][n]['W-fit'][0]:.4f} mean {sc['brier_vs_bayes'][n]['mean'][0]:.4f}"
                  f" vote {sc['brier_vs_bayes'][n]['vote'][0]:.4f} kappa {fits['W']['kappa']:.3f} alpha {fits['W']['alpha']:.3f}"
                  f" t={time.time() - t0:.0f}s", flush=True)
        out["scenarios"][name] = sc
    # verdicts on the pre-stated claims
    c1, c1b, c2, c2d, c4w, c4d = [], [], [], [], [], []
    for name, sc in out["scenarios"].items():
        for n in (4, 16, 64):
            d = sc["diff"][n]
            if sc["overlap_a"] < 1:
                c1.append(d["W-fit_minus_mean"][2] < 0 and d["W-fit_minus_vote"][2] < 0)
                c1b.append(d["D-fit_minus_mean"][2] < 0 and d["D-fit_minus_vote"][2] < 0)
            elif n >= 16:
                c4w.append(d["W-fit_minus_mean"][1] <= 0)
                c4d.append(d["D-fit_minus_mean"][1] <= 0)
            c2.append(sc["settle"][n]["spearman_s_vs_info"][1] > 0)
            c2d.append(sc["settle"][n]["spearman_s_vs_Dfit_vs_info"][1] > 0)
    gm = max(sc["settle"][16]["grad_check_max_abs_err"] for sc in out["scenarios"].values())
    frac = lambda v: f"{sum(v)}/{len(v)} cells"
    out["verdicts"] = {"C1_W-fit_beats_mean_and_vote_when_a_lt_1": frac(c1),
                       "C1b_D-fit_beats_mean_and_vote_when_a_lt_1": frac(c1b),
                       "C2_settlement_vs_W-fit_spearman_ci_above_0": frac(c2),
                       "C2_settlement_vs_D-fit_spearman_ci_above_0": frac(c2d),
                       "C3_grad_check_max_abs_err": gm,
                       "C4_W-fit_no_loss_to_mean_at_a1": frac(c4w),
                       "C4_D-fit_no_loss_to_mean_at_a1": frac(c4d)}
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def tables(out):
    L = []
    f3 = lambda v: f"{1000 * v[0]:.2f}"
    ci = lambda v: f"{1000 * v[0]:+.2f} [{1000 * v[1]:+.2f}, {1000 * v[2]:+.2f}]"
    L.append("Brier vs exact Bayes, x1000 (mean over test boards of the mean squared gap to the exact posterior"
             " on open statements; lower is better; 0 = exact Bayes).\n")
    for name, sc in out["scenarios"].items():
        L.append(f"Scenario {name} (overlap a = {sc['overlap_a']}, noise = {sc['noise']})\n")
        L.append("| pool | " + " | ".join(f"n={n}" for n in NS) + " |")
        L.append("|---|" + "---|" * len(NS))
        for m in METHODS:
            L.append(f"| {m} | " + " | ".join(f3(sc["brier_vs_bayes"][n][m]) for n in NS) + " |")
        L.append("| W-fit minus mean [95% CI] | " + " | ".join(ci(sc["diff"][n]["W-fit_minus_mean"]) for n in NS) + " |")
        L.append("| W-fit minus vote [95% CI] | " + " | ".join(ci(sc["diff"][n]["W-fit_minus_vote"]) for n in NS) + " |")
        L.append("| D-fit minus mean [95% CI] | " + " | ".join(ci(sc["diff"][n]["D-fit_minus_mean"]) for n in NS) + " |")
        L.append("| D-fit minus vote [95% CI] | " + " | ".join(ci(sc["diff"][n]["D-fit_minus_vote"]) for n in NS) + " |")
        L.append("| D-fit minus W-fit [95% CI] | " + " | ".join(ci(sc["diff"][n]["D-fit_minus_W-fit"]) for n in NS) + " |")
        L.append("| learned stakes W: kappa, alpha | " + " | ".join(
            f"{sc['fit'][n]['W']['kappa']:.2f}, {sc['fit'][n]['W']['alpha']:.3g}" for n in NS) + " |")
        L.append("| learned stakes LL (used by D-fit): kappa, alpha | " + " | ".join(
            f"{sc['fit'][n]['LL']['kappa']:.2f}, {sc['fit'][n]['LL']['alpha']:.3g}" for n in NS) + " |")
        L.append("| ECE: mean / vote / W-fit / D-fit | " + " | ".join(
            f"{sc['ece'][n]['mean']:.3f} / {sc['ece'][n]['vote']:.3f} / {sc['ece'][n]['W-fit']:.3f} / {sc['ece'][n]['D-fit']:.3f}" for n in NS) + " |")
        L.append("")
    L.append("Settlement credit vs oracle informativeness: mean within-board Spearman [95% CI] over test boards.\n")
    L.append("Columns 3-6 settle against W-fit; column 7 settles s_i against D-fit prices.\n")
    L.append("| scenario | n | s_i (paid) | s_raw | stake-weighted pi_i | exact leave-one-out | s_i vs D-fit | pooled Spearman s_i |")
    L.append("|---|---|---|---|---|---|---|---|")
    sp = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    for name, sc in out["scenarios"].items():
        for n in (4, 16, 64):
            s = sc["settle"][n]
            L.append(f"| {name} | {n} | {sp(s['spearman_s_vs_info'])} | {sp(s['spearman_s_raw_vs_info'])} | "
                     f"{sp(s['spearman_pi_vs_info'])} | {sp(s['spearman_loo_vs_info'])} | {sp(s['spearman_s_vs_Dfit_vs_info'])} | {s['pooled_spearman_s_vs_info']:+.3f} |")
    L.append("")
    fl = out["floor_one_exact_chain_all_open_statements"]
    L.append(f"Floor check (one noiseless chain stating exact Bayes on every open statement, {out['config']['n_test']} test boards):"
             f" W-add {fl['W-add'][0]:.5f} [{fl['W-add'][1]:.5f}, {fl['W-add'][2]:.5f}],"
             f" projection c=20 {fl['D_c20'][0]:.2e} [{fl['D_c20'][1]:.2e}, {fl['D_c20'][2]:.2e}] Brier vs Bayes.\n")
    L.append("Verdicts: " + json.dumps(out["verdicts"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--k-shots", type=int, default=4)
    ap.add_argument("--k-claims", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--grad-eps", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "smoke_results.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "smoke_tables.md"))
    args = ap.parse_args()
    res = run(args)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    with open(args.md, "w") as fh:
        fh.write(tables(res))
    print(json.dumps(res["verdicts"]), "runtime_s", res["runtime_s"])
