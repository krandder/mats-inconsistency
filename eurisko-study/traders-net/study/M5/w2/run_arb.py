"""Run the M5-w2 arbitrageur accounting on one price source. Usage:
  python3 run_arb.py li SEED            (reads /tmp/traders-net/M5/w2/li-prices-s<SEED>.npz; markets net + logistic)
  python3 run_arb.py wn SEED STEP       (wealth-net frozen checkpoint, STEP = final | 8000 ...; held-out positions)
Writes results/<tag>.json (summary per trader/kind) and results/<tag>-curves.csv (E_C(t) per kind, grid of 200)."""
import json, os, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "..", "S0", "substrate")
WN = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "wealth-net"))
sys.path.insert(0, HERE); sys.path.insert(0, SUB)
import arb
from panel import ec_slopes
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)


def li_cons(st):
    cons = []
    for q in range(st.Q):
        k = st.kind[q]; a, b = int(st.ops[q, 0]), int(st.ops[q, 1])
        if k in ("and", "or"):
            cons.append(dict(kind=k, legs=[a, b, q]))
        elif k in ("taut_or", "impl"):
            cons.append(dict(kind=k, legs=[q], fixed=1))
        elif k == "contra":
            cons.append(dict(kind=k, legs=[q], fixed=0))
        elif k == "twin":
            cons.append(dict(kind=k, legs=[a, q]))
    for g in np.unique(st.group[st.group >= 0]):
        cons.append(dict(kind="ex3", legs=[int(x) for x in np.where(st.group == g)[0]]))
    return cons


def analyse(tag, cons, P, y, mask, t, delay, T_end, meta, do_member=True):
    t0 = time.time()
    P = np.clip(P, 1e-7, 1 - 1e-7)  # price floor: a market never quotes exactly 0/1
    sh = arb.shadow(cons, P, y, mask, t, delay, T_end)
    grid = np.linspace(t.min(), T_end, 200)
    rows, per = [], []
    kinds = sorted(set(c["kind"] for c in cons))
    curves = {}
    for kind in kinds + ["all"]:
        sel = [s for s in sh if kind == "all" or s["kind"] == kind]
        for form in ("K", "C", "A", "B"):
            cs = [arb.curve(s["s" + form], s["e" + form], grid) for s in sel]
            Emax = np.max(np.stack(cs), 0); Esum = np.sum(np.stack(cs), 0)
            curves[f"{kind}:{form}:max"] = Emax; curves[f"{kind}:{form}:sum"] = Esum
    summ = {}
    for key, E in curves.items():
        sl = ec_slopes(grid - grid[0] + 1, E)
        h = len(grid) // 2
        first = float(np.polyfit(grid[:h], E[:h], 1)[0]); second = float(np.polyfit(grid[h:], E[h:], 1)[0])
        summ[key] = dict(final=float(E[-1]), loglog=sl["loglog"], slope_first_half=first, slope_second_half=second)
    for c, s in zip(cons, sh):
        d = dict(kind=c["kind"], legs=c["legs"], n_live=s["n_live"], E_K=float(s["eK"].sum()),
                 E_B=float(s["eB"].sum()), E_clip=float(s["eC"].sum()), E_A=float(s["eA"].sum()), n_B=int(len(s["eB"])), minpnl_B=float(s["eB"].min()) if len(s["eB"]) else None,
                 phi_K=float(s["eK"].sum() / max(1, T_end - t.min())))
        tmid = t.min() + (T_end - t.min()) // 2
        d["invasion_eps_h2"] = float(s["eI"][s["tI"] >= tmid].sum() / max(1, T_end - tmid)) if len(s["eI"]) else 0.0
        d["invasion_eps_h1"] = float(s["eI"][s["tI"] < tmid].sum() / max(1, tmid - t.min())) if len(s["eI"]) else 0.0
        if do_member:
            d.update(arb.member(c, P, y, mask, t, delay, T_end))
        per.append(d)
    res = dict(tag=tag, meta=meta, curves_summary=summ, traders=per, wall_s=time.time() - t0,
               ln_1_over_eps=float(np.log(1 / arb.EPS)))
    json.dump(res, open(os.path.join(HERE, "results", f"{tag}.json"), "w"), indent=1)
    import csv
    with open(os.path.join(HERE, "results", f"{tag}-curves.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["t"] + list(curves)); 
        for i, g in enumerate(grid):
            wr.writerow([f"{g:.0f}"] + [f"{curves[k][i]:.5g}" for k in curves])
    print(tag, f"{res['wall_s']:.0f}s", {k: round(v["final"], 2) for k, v in summ.items() if k.startswith("all")})


if sys.argv[1] == "li":
    seed = int(sys.argv[2])
    from li_families import make_li_stream, violations
    d = np.load(f"/tmp/traders-net/M5/w2/li-prices-s{seed}.npz")
    st = make_li_stream(seed, int(d["T"]))
    cons = li_cons(st); t = d["t"]; y = d["y"].astype(int); mask = d["mask"].astype(bool)
    T_end = st.T - 1
    for name, zk in (("net", "z"), ("logistic", "zb")):
        P = 1 / (1 + np.exp(-d[zk].astype(np.float64)))
        v = violations(P, st, t=t, tol=0.05)
        meta = dict(source=f"li-{name}", seed=seed, T=int(d["T"]), Q=st.Q,
                    viol={k: vv["mean"] for k, vv in v.items()})
        analyse(f"li-{name}-s{seed}", cons, P, y, mask, t, st.delay.astype(int), T_end, meta)
elif sys.argv[1] == "wn":
    import torch
    torch.set_num_threads(1)
    seed, step = int(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, WN)
    import importlib.util
    def load(name):
        spec = importlib.util.spec_from_file_location("wn_" + name, os.path.join(WN, name + ".py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    S = load("streams"); M = load("model")
    cfg = json.load(open(os.path.join(WN, "results", f"train-s{seed}.json")))
    bits, fam, desc = S.make_questions(cfg["Q"], cfg["T"], seed)
    Q, W = cfg["Q"], cfg["W"]
    net = M.Net(Q, W, cfg["d"], cfg["H"], cfg["L"])
    path = os.path.join(WN, "results", f"train-s{seed}.pt") if step == "final" else \
        os.path.join(WN, "results", "weights", f"train-s{seed}-step{step}.pt")
    net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()
    n_pos = cfg["T"] - W; split = int(n_pos * 0.8)
    idx = np.arange(split, n_pos)
    bt = torch.tensor(bits, dtype=torch.float32)
    wins = bt.unfold(1, W, 1)
    Z = []
    with torch.no_grad():
        for s0 in range(0, len(idx), 512):
            ii = torch.as_tensor(idx[s0:s0 + 512])
            Z.append(net(wins[:, ii].permute(1, 0, 2).contiguous()).numpy())
    Z = np.concatenate(Z); P = 1 / (1 + np.exp(-Z.astype(np.float64)))
    y = bits[:, idx + W].T.astype(int); t = idx + W; mask = np.ones_like(y, bool)
    cons = []
    for q, dsc in enumerate(desc):
        m = re.match(r"(and|or)\(q(\d+) .*, q(\d+) ", dsc)
        if m:
            cons.append(dict(kind=m.group(1), legs=[int(m.group(2)), int(m.group(3)), q]))
    assert all((y[:, c["legs"][2]] == ((y[:, c["legs"][0]] & y[:, c["legs"][1]]) if c["kind"] == "and"
               else (y[:, c["legs"][0]] | y[:, c["legs"][1]]))).all() for c in cons)
    # violation means (same formulas as S0 li_families.violations, AND/OR only)
    viol = {}
    for kind in ("and", "or"):
        vs = []
        for c in cons:
            if c["kind"] != kind: continue
            a, b, cc = (P[:, l] for l in c["legs"])
            v = (np.maximum(0, cc - np.minimum(a, b)) + np.maximum(0, a + b - 1 - cc)) if kind == "and" else \
                (np.maximum(0, np.maximum(a, b) - cc) + np.maximum(0, cc - a - b))
            vs.append(v)
        viol[kind] = float(np.mean(vs))
    bce = float(-(y * np.log(np.clip(P, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - P, 1e-7, 1))).mean())
    meta = dict(source=f"wealth-net-{step}", seed=seed, ckpt=path, n_pos_eval=len(idx), viol=viol, bce_all=bce,
                steps_done=cfg.get("steps_done"))
    analyse(f"wn-{step}-s{seed}", cons, P, y, mask, t, np.zeros(Q, int), int(t.max()), meta)
