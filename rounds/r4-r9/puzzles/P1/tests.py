"""Phase-0 unit tests (a)-(e).  python tests.py  -> writes results/unit_tests.json"""
import json
import os
import sys
import tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from heads import perpos_targets, HORIZONS, KL_HORIZONS, ent_bern, ARMS_T2M, features  # noqa: E402
from halting import Obs, stops_for, rule_stat, SIMPLE_RULES  # noqa: E402

RES = {}


def logit(p):
    return torch.log(p) - torch.log1p(-p)


def mix_G(Z, t, T, theta):
    """G integrated over y ~ Bern(theta) given theta (per path)."""
    g1 = perpos_targets(Z, t, T, y=torch.ones_like(Z[0]).float(), kinds=("G",))["G"]
    g0 = perpos_targets(Z, t, T, y=torch.zeros_like(Z[0]).float(), kinds=("G",))["G"]
    th = theta.view(-1, 1, 1)
    return th * g1 + (1 - th) * g0


def test_a(n=400_000, t0=2, seed=0):
    """Beta-Bernoulli predictive path: G, K and entropy drop agree within 2%."""
    g = torch.Generator().manual_seed(seed)
    T = t0 + 64
    theta = torch.rand(n, generator=g, dtype=torch.float64)
    x = (torch.rand(n, T, generator=g, dtype=torch.float64) < theta[:, None]).double()
    s = torch.cat([torch.zeros(n, 1, dtype=torch.float64), x.cumsum(1)], 1)
    b = (1 + s) / (2 + torch.arange(T + 1, dtype=torch.float64))          # b_t, t = 0..T
    Z = logit(b).T.unsqueeze(2)                                             # (T+1, n, 1) float64
    G = mix_G(Z, t0, T, theta)[:, :, 0].mean(0)
    K = perpos_targets(Z, t0, T, kinds=("K",))["K"][:, :, 0].mean(0)
    out = {}
    for H in (1, 16, 64):
        j = HORIZONS.index(H)
        pt, pH = b[:, t0], b[:, t0 + H]
        kl = (pH * torch.log(pH / pt) + (1 - pH) * torch.log((1 - pH) / (1 - pt))).mean().item()
        ed = (ent_bern(pt) - ent_bern(pH)).mean().item()
        kH = K[KL_HORIZONS.index(H)].item() if H in KL_HORIZONS else kl
        gH = G[j].item()
        rel = max(abs(gH - kl), abs(kH - kl), abs(ed - kl)) / kl
        out[str(H)] = {"G": gH, "K": kH, "ent_drop": ed, "max_rel_dev": rel, "pass": rel < 0.02}
    RES["a"] = out
    return all(v["pass"] for v in out.values())


def test_a2(n=200_000, seed=1, rho=0.9):
    """Forgetful path: b_{t+H} = 0.5 + rho^H (b_t - 0.5), b_t calibrated -> G < 0 <= K."""
    g = torch.Generator().manual_seed(seed)
    T = 64
    bt = torch.rand(n, generator=g, dtype=torch.float64) * 0.98 + 0.01
    path = torch.stack([0.5 + rho ** k * (bt - 0.5) for k in range(T + 1)], 0)   # (T+1, n)
    Z = logit(path).unsqueeze(2)
    G = mix_G(Z, 0, T, bt)[:, :, 0].mean(0)
    K = perpos_targets(Z, 0, T, kinds=("K",))["K"][:, :, 0].mean(0)
    ok = bool((G[1:] < 0).all() and (K >= 0).all())
    RES["a2"] = {"G": G.tolist(), "K": K.tolist(), "pass": ok}
    return ok


def test_a3(n=200_000, seed=2, delta=1.0):
    """Biased b_t (logit shift delta), calibrated b_{t+H} = q: G ~= K, entropy drop != K."""
    g = torch.Generator().manual_seed(seed)
    q = torch.rand(n, generator=g, dtype=torch.float64) * 0.9 + 0.05
    bt = torch.sigmoid(logit(q) + delta)
    T = 4
    path = torch.stack([bt] + [q] * T, 0)
    Z = logit(path).unsqueeze(2)
    G = mix_G(Z, 0, T, q)[:, :, 0].mean(0)[HORIZONS.index(2)].item()
    K = perpos_targets(Z, 0, T, kinds=("K",))["K"][:, :, 0].mean(0)[0].item()
    ed = (ent_bern(bt) - ent_bern(q)).mean().item()
    ok = abs(G - K) / K < 0.02 and abs(ed - K) / K > 0.2
    RES["a3"] = {"G": G, "K": K, "ent_drop": ed, "pass": bool(ok)}
    return ok


def test_b():
    """Known stops on hand-built trajectories."""
    Tc = 10
    ht = np.arange(1, Tc + 1)
    ent = np.array([[9, 8, 7, 3, 5, 1, 1, 1, 1, 1, 1.0]])
    kl1 = np.array([[0, 5, 4, 0.1, 3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    d = {"ent": ent, "kl1": kl1, "kl2": kl1, "conf_sum": -ent, "conf_min": 1 - ent / 10}
    # R7 head: V (1, G, 8); V_1 = 0.5 at t<=4 else 0.01; others zero
    V = np.zeros((1, Tc, 8)); V[0, :4, 0] = 0.5; V[0, 4:, 0] = 0.01
    d["h_R7"] = V
    bits = np.zeros((1, Tc, 3), bool); bits[0, 6, 0] = True; bits[0, 2, 1] = True
    d["h_R13_bits"] = bits
    ob = Obs(d, Tc, ht)
    got = {
        "R3ent@4": int(stops_for(ob, "R3ent", [4.0])[0, 0]),       # first t with ent <= 4 -> t=3
        "R3ent@0.5": int(stops_for(ob, "R3ent", [0.5])[0, 0]),     # never -> Tc
        "R2:1:1@0.2": int(stops_for(ob, "R2:1:1", [0.2])[0, 0]),   # kl1[3]=0.1 -> 3
        "R2:1:2@0.2": int(stops_for(ob, "R2:1:2", [0.2])[0, 0]),   # max(kl1[t-1:t+1]) <= .2 first at t=6
        "R7@0.1": int(stops_for(ob, "R7", [0.1])[0, 0]),           # V1/1 <= 0.1 from t=5
        "R0@7": int(stops_for(ob, "R0", [7])[0, 0]),
        "R13": stops_for(ob, "R13", None)[0].tolist(),             # [7, 3, 10]
    }
    want = {"R3ent@4": 3, "R3ent@0.5": 10, "R2:1:1@0.2": 3, "R2:1:2@0.2": 6, "R7@0.1": 5, "R0@7": 7,
            "R13": [7, 3, 10]}
    ok = got == want
    RES["b"] = {"got": got, "want": want, "pass": ok}
    return ok


def test_c():
    """The unlabelled loader never yields labels and never opens solutions.npy."""
    from data import UnlabelledMazes, LabelledMazes, maze_dir
    with tempfile.TemporaryDirectory() as root:
        d = maze_dir(root, 13)
        os.makedirs(d)
        np.save(os.path.join(d, "inputs.npy"), np.zeros((1000, 3, 32, 32), np.uint8))
        ys = [y for _, y, _ in UnlabelledMazes(root, 13).batches(64)]
        ok = len(ys) > 0 and all(y is None for y in ys)
        try:
            LabelledMazes(root, 13, "pool")
            ok = False     # labelled loader must fail without solutions.npy
        except FileNotFoundError:
            pass
    RES["c"] = {"pass": ok}
    return ok


def test_d_e(pilot, n=20, L=24, seed=0):
    """(d) online stop == post-hoc stop; (e) leak test. Pilot backbone + random-init heads."""
    from models import load_pilot
    from mresnet import prefix_batch
    from traj import rollout
    from train_heads import ArmSet
    from heads import kl_bern
    torch.manual_seed(seed)
    model = load_pilot(pilot)
    g = torch.Generator().manual_seed(seed)
    x, y, _ = prefix_batch(n, L, L, g)
    Tc = 2 * L
    A = ArmSet({k: ARMS_T2M[k] for k in ("R4", "R7", "R13")}, model.w + 5, 1, 0, "cpu")
    logc = torch.log(torch.tensor(np.geomspace(1e-3, 10, 20), dtype=torch.float32))
    heads = {"h_R4": (A.heads["R4"], "value"), "h_R7": (A.heads["R7"], "value"), "h_R13": (A.heads["R13"], "qact")}
    ht = list(range(1, Tc + 1))
    R = rollout(model, x, Tc, head_times=ht, heads=heads, qact_logc=logc)
    d = {k: v.numpy().astype(np.float64) for k, v in R["obs"].items()}
    for k, v in R["hout"].items():
        d[k] = v.numpy() if k.endswith("_bits") else v.numpy().astype(np.float64)
    rules = ["R3ent", "R2:2:8", "R7", "R4e", "R13"]
    kn = {}
    for r in rules[:-1]:
        s = rule_stat(Obs(d, Tc, ht), r)
        fin = s[:, 1:Tc][np.isfinite(s[:, 1:Tc])]
        kn[r] = np.quantile(fin, [0.1, 0.3, 0.6])
    post = {r: stops_for(Obs(d, Tc, ht), r, kn.get(r)) for r in rules}
    # ---- online loop (independent code path)
    online = {r: np.full((n, 3 if r != "R13" else len(logc)), -1) for r in rules}
    with torch.no_grad():
        st = model.init_state(x)
        p0 = torch.full((n, L), 0.5)
        hist = [p0, p0]
        klh = {1: [], 2: []}
        for t in range(1, Tc + 1):
            st, lg, h = model.step_once(st)
            p = torch.sigmoid(lg.float())
            k2 = kl_bern(p, hist[-2]).sum(1).double().numpy()
            klh[2].append(k2)
            f = features(h, p, hist[-1], hist[-2])
            ent = ent_bern(p).sum(1).double().numpy()
            V7 = A.heads["R7"].totals(f)[0].double().numpy()
            V4 = A.heads["R4"].totals(f)[0].double().numpy()[:, 0]
            he = np.array([min(H, Tc - t) if H != "cap" else Tc - t for H in HORIZONS], float)
            v7 = np.max(np.where(he > 0, V7 / np.maximum(he, 1e-9), -np.inf), 1) if t < Tc else np.full(n, -np.inf)
            v4e = np.maximum(V4, ent / (Tc - t)) if t < Tc else np.full(n, -np.inf)
            win = np.max(np.stack(klh[2][-8:], 1), 1) if t >= 8 else np.full(n, np.inf)
            tot, pooled = A.heads["R13"].totals(f)
            adv = (A.heads["R13"].qact_adv(pooled, L, logc) + tot[:, 1:2]).numpy()
            stat = {"R3ent": ent if t < Tc else np.full(n, -np.inf), "R2:2:8": win if t < Tc else np.full(n, -np.inf),
                    "R7": v7, "R4e": v4e}
            for r in rules[:-1]:
                hit = (stat[r][:, None] <= kn[r][None]) & (online[r] < 0)
                online[r][hit] = t
            hit = ((adv <= 0) | (t == Tc)) & (online["R13"] < 0)
            online["R13"][hit] = t
            hist = [hist[-1], p]
    dres = {r: int((online[r] == post[r]).all()) for r in rules}
    ok_d = all(dres.values())
    # ---- (e) leak test: corrupt everything after a random cut tau (and no eval is passed at all)
    rng = np.random.default_rng(seed)
    tau = rng.integers(1, Tc, n)
    d2 = {}
    for k, v in d.items():
        v2 = v.copy()
        for i in range(n):
            if k.startswith("h_"):
                cut = np.array(ht) > tau[i]
                v2[i, cut] = rng.random(v2[i, cut].shape) > 0.5 if k.endswith("_bits") else rng.normal(0, 5, v2[i, cut].shape)
            else:
                v2[i, tau[i] + 1:] = np.abs(rng.normal(0, 5, v2[i, tau[i] + 1:].shape))
        d2[k] = v2
    eres = {}
    for r in rules:
        s1, s2 = post[r], stops_for(Obs(d2, Tc, ht), r, kn.get(r))
        early = s1 <= tau[:, None]
        same = np.where(early, s1 == s2, s2 > tau[:, None])
        eres[r] = int(same.all())
    ok_e = all(eres.values())
    RES["d"] = {"per_rule": dres, "pass": ok_d, "n": n}
    RES["e"] = {"per_rule": eres, "pass": ok_e}
    return ok_d and ok_e


if __name__ == "__main__":
    torch.set_num_threads(4)
    pilot = os.path.join(HERE, "..", "..", "market-resnet", "results", "pilot", "pilot_Bds.pt")
    res = {"a": test_a(), "a2": test_a2(), "a3": test_a3(), "b": test_b(), "c": test_c(), "d_e": test_d_e(pilot)}
    RES["all_pass"] = all(res.values())
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(RES, open(os.path.join(HERE, "results", "unit_tests.json"), "w"), indent=1, default=str)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in RES.items()}))
    print(json.dumps(RES, default=str)[:3000])
