"""M2 item 3: pruning frontier. Zero-ablate the k least important units by each predictor and
measure held-out BCE on an independent segment (bit positions T+6144 .. T+8191, disjoint from worker 2 _val sample T+4096..T+6143,
the oracle's T .. T+4095 and from training). Rows are cached per (seed, step, gran, predictor),
so predictors can be added later without recomputing.

Usage: python3 frontier.py SEED [--preds FILE.csv ...] [--threads 2]
Predictor CSV (long): seed,step,gran,unit_flat,predictor,score  (higher = more important);
built-in predictors: random (5 orders), oracle_zero / oracle_mean (worker 1's single-unit
oracle on the other segment), wealth_v1 (wealth-net cumulative wealth), gradnorm_v1.
Output: /tmp/traders-net/M2/w3/frontier-sS.csv (cache) and results/frontier-sS.csv.
"""
import csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WN = os.path.abspath(os.path.join(HERE, "../../../../wealth-net"))
sys.path.insert(0, WN)
import torch
import torch.nn.functional as F
from streams import make_questions, FAMILIES
from model import Net

FRACS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60]
CKPT = {1: [20000, 31100], 2: [20000, 31200], 3: [20000, 31100]}   # frontier at 20k + final (compute; see PREREG amendment)
FINAL = {1: 31100, 2: 31200, 3: 31100}
NPOS = 2048
TMP = "/tmp/traders-net/M2/w3"
ABS = {"v1_wealth", "v2_lmsr_int", "v2_tangent_int", "taylor", "cond", "grad_live_cum", "settle_loo_train"}
ORACLE = os.path.join(HERE, "../w1/results/oracle_units.csv")
GRAN_N = {"neuron": 512, "head": 8, "readout": 64}


def weights(seed, step):
    final = FINAL[seed]
    return os.path.join(WN, "results", f"train-s{seed}.pt") if step == final else \
        os.path.join(WN, "results", "weights", f"train-s{seed}-step{step}.pt")


def load_eval(seed, meta):
    Q, W, T = meta["Q"], meta["W"], meta["T"]
    bits, fam, _ = make_questions(Q, T + 6144 + NPOS, seed)
    b = torch.tensor(bits, dtype=torch.float32)
    wins = b.unfold(1, W, 1)
    ev = torch.arange(T + 6144 - W, T + 6144 + NPOS - W)    # predicts bits[T+6144 .. T+6144+NPOS-1]
    return wins[:, ev].permute(1, 0, 2).contiguous(), b[:, ev + W].T.contiguous(), torch.tensor(fam)


def builtin_scores(seed, step):
    out = {}
    sn = json.load(open(os.path.join(WN, "results", f"train-s{seed}", "snapshots", f"step-{step:05d}.json")))
    for g, key in (("neuron", "neuron"), ("head", "head"), ("readout", "out")):
        out[(g, "wealth_v1")] = np.array(sn["wealth"][key], dtype=float).ravel()
        out[(g, "gradnorm_v1")] = np.array(sn["gradnorm_cum"][key], dtype=float).ravel()
    import glob
    fs = sorted(glob.glob(os.path.join(os.path.dirname(ORACLE), f"oracle_units_s{seed}_step{step}.csv")))
    if os.path.exists(ORACLE):
        fs.append(ORACLE)
    for f in fs:
        for r in csv.DictReader(open(f)):
            if int(r["seed"]) != seed or int(r["step"]) != step:
                continue
            g = r["gran"]; i = int(r["flat"])
            for ab in ("zero", "mean"):
                out.setdefault((g, f"oracle_{ab}"), np.full(GRAN_N[g], np.nan))[i] = float(r[f"dloss_{ab}"])
    return out


def ext_scores(files, seed, step):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if int(r["seed"]) != seed or int(r["step"]) != step:
                continue
            g = r["gran"]
            out.setdefault((g, r["predictor"]), np.full(GRAN_N[g], np.nan))[int(r.get("unit_flat", r.get("flat")))] = float(r.get("score", r.get("value")))
    return out


def main():
    args = sys.argv[1:]
    seed = int(args[0])
    preds = []; threads = 2
    if "--preds" in args:
        i = args.index("--preds") + 1
        while i < len(args) and not args[i].startswith("--"):
            preds.append(args[i]); i += 1
    if "--threads" in args:
        threads = int(args[args.index("--threads") + 1])
    torch.set_num_threads(threads)
    import glob
    preds += sorted(glob.glob(os.path.join(HERE, "../w2/results", f"pred-s{seed}-*-step*.csv")))
    preds += sorted(glob.glob(os.path.join(HERE, "results", f"pred-s{seed}-*-step*.csv")))
    os.makedirs(TMP, exist_ok=True)
    cache_f = os.path.join(TMP, f"frontier-s{seed}.csv")
    done = set(); rows = []
    if os.path.exists(cache_f):
        for r in csv.DictReader(open(cache_f)):
            rows.append(r); done.add((int(r["step"]), r["gran"], r["predictor"]))
    meta = json.load(open(os.path.join(WN, "results", f"train-s{seed}.json")))
    ev_win, ev_y, famt = load_eval(seed, meta)
    fields = ["seed", "step", "gran", "predictor", "order", "frac", "k", "n_units", "n_nan", "bce", "brier"] + \
             [f"bce_{f}" for f in FAMILIES]
    new = []
    for step in CKPT[seed]:
        memo = {}
        m = Net(meta["Q"], meta["W"], meta["d"], meta["H"], meta["L"])
        m.load_state_dict(torch.load(weights(seed, step), map_location="cpu")); m.eval()
        L, H, N = m.L, m.H, 4 * m.d
        with torch.no_grad():
            z0 = torch.cat([m(ev_win[i:i + 1024]) for i in range(0, ev_win.shape[0], 1024)])
        sc = builtin_scores(seed, step); sc.update(ext_scores(preds, seed, step))
        for (g, p) in list(sc):                      # |signed| variants: does the sign or the size carry the signal?
            if p in ABS:
                sc[(g, f"abs_{p}")] = np.abs(sc[(g, p)])
        for g in ("neuron", "head", "readout"):
            n = GRAN_N[g]
            names = ["random"] + sorted(p for (gg, p) in sc if gg == g)
            for p in names:
                if (step, g, p) in done:
                    continue
                t0 = time.time()
                if p == "random":
                    rng = np.random.default_rng(seed * 1000 + step)
                    orders = [rng.permutation(n) for _ in range(5)]; nnan = 0
                else:
                    s = sc[(g, p)].copy(); nnan = int(np.isnan(s).sum())
                    s[np.isnan(s)] = np.inf                         # unknown units are never pruned first
                    orders = [np.argsort(s, kind="stable")]         # least important first
                ks = sorted(set(list(range(n + 1)) if g == "head" else [int(round(f * n)) for f in FRACS]))
                for oi, od in enumerate(orders):
                    for k in ks:
                        key = (g, frozenset(od[:k].tolist()))
                        if key in memo:
                            bce, br = memo[key]
                        else:
                          with torch.no_grad():
                            if g == "readout":
                                z = z0.clone(); z[:, torch.tensor(od[:k], dtype=torch.long)] = 0.0
                            else:
                                mask = {"neuron": torch.ones(L, N), "head": torch.ones(L, H)}
                                mask[g].view(-1)[torch.tensor(od[:k], dtype=torch.long)] = 0
                                z = torch.cat([m(ev_win[i:i + 1024], mask=mask) for i in range(0, ev_win.shape[0], 1024)])
                            bce = F.binary_cross_entropy_with_logits(z, ev_y, reduction="none").mean(0)
                            br = ((torch.sigmoid(z) - ev_y) ** 2).mean(0)
                          memo[key] = (bce, br)
                        r = {"seed": seed, "step": step, "gran": g, "predictor": p, "order": oi, "frac": round(k / n, 6),
                             "k": k, "n_units": n, "n_nan": nnan, "bce": round(bce.mean().item(), 6),
                             "brier": round(br.mean().item(), 6)}
                        for fi, fn in enumerate(FAMILIES):
                            r[f"bce_{fn}"] = round(bce[famt == fi].mean().item(), 6)
                        new.append(r)
                print(seed, step, g, p, f"{time.time() - t0:.1f}s", flush=True)
                with open(cache_f, "a", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=fields)
                    if f.tell() == 0:
                        wr.writeheader()
                    wr.writerows([x for x in new if (x["step"], x["gran"], x["predictor"]) == (step, g, p)])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.system(f"cp {cache_f} {os.path.join(HERE, 'results', f'frontier-s{seed}.csv')}")


if __name__ == "__main__":
    main()
