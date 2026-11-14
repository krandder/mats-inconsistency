"""M6 w1 runs: python3 run_w1.py REGIME ARM SEED   (REGIME P|ME, ARM A|B|WD|BOOST). See PREREG.md."""
import json, math, os, re, sys, time
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m6layer import MarketNet, masked_bce, make_opts, SUB  # noqa: E402
import streams_ext, panel  # noqa: E402
from prequential import assert_disjoint  # noqa: E402

FAM = streams_ext.FAMILIES[:8]
LEARN = ["periodic", "lfsr16", "ca110", "prime", "and", "or"]
NOISE = ["prng", "ca30"]
LN2 = math.log(2)
Q, W, T = 64, 32, 24032
OUT = os.path.join(HERE, "results")
TMP = "/tmp/traders-net/M6/w1"


def constraints(s):
    cons = []
    for q, d in enumerate(s.desc):
        m = re.match(r"(and|or)\(q(\d+) \S+, q(\d+) \S+\)", d)
        if m:
            cons.append((m.group(1), q, int(m.group(2)), int(m.group(3))))
    return cons


def metrics(z, y, m, fam, cons):
    """z,y,m [n,Q] numpy. Returns per-family scores + primaries."""
    p = 1 / (1 + np.exp(-z.astype(np.float64)))
    F = np.broadcast_to(fam[None, :], z.shape)
    sc = panel.scores(p[m], y[m], F[m], names=FAM)
    fb = {f: sc[f]["bce"] for f in FAM if f in sc}
    out = {"fam": sc,
           "learn_bce": float(np.mean([fb[f] for f in LEARN])),
           "humility": float(max(fb[f] - LN2 for f in NOISE)),
           "lfsr_learnt": int(fb["lfsr16"] < 0.35)}
    v = panel.violations(p, cons, tol=0.01)
    out["viol_and"] = v.get("and", {}).get("mean"); out["viol_or"] = v.get("or", {}).get("mean")
    out["viol_all"] = v["all"]["mean"]
    return out


def run_P(model, st, o, om, B=4):
    n_pos = st.n_pos
    Z = np.zeros((n_pos, Q), np.float32); ZL = np.zeros_like(Z)
    Y = np.zeros((n_pos, Q), np.float32); M = np.zeros((n_pos, Q), bool)
    pred = np.full(n_pos, -1); trained = np.full(n_pos, -1)
    for s in range(math.ceil(n_pos / B)):
        P = np.arange(s * B, min((s + 1) * B, n_pos))
        win, y, m = st.window(P), st.label(P), st.label_mask(P)
        model.eval()
        with torch.no_grad():
            out = model(win)
        Z[P], ZL[P] = out["z"].numpy(), out["l0"].numpy()
        Y[P], M[P] = y.numpy(), m.numpy(); pred[P] = s
        assert np.all(pred[P] >= 0) and np.all(trained[P] < 0), "prequential violation"
        trained[P] = s
        model.train()
        out = model(win); mf = m.float()
        loss = masked_bce(out["z"], y, mf) + masked_bce(out["l0"], y, mf)
        o.zero_grad(set_to_none=True); om.zero_grad(set_to_none=True)
        loss.backward(); o.step(); om.step()
    return Z, ZL, Y, M


def evalP(model, st, P, chunk=600):
    zs, zl = [], []
    model.eval()
    with torch.no_grad():
        for k in range(0, len(P), chunk):
            out = model(st.window(P[k:k + chunk])); zs.append(out["z"].numpy()); zl.append(out["l0"].numpy())
    return np.concatenate(zs), np.concatenate(zl), st.label(P).numpy(), st.label_mask(P).numpy()


def main():
    regime, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    torch.manual_seed(seed); np.random.seed(seed)
    s = streams_ext.make_streams(Q, T, seed, heldout=0, regime_every=0, W=W)
    st = streams_ext.to_array_stream(s)
    fam, cons = np.asarray(s.fam0), constraints(s)
    assert len(cons) == 16, len(cons)
    model = MarketNet(Q, W, arm="A" if arm == "WD" else arm)
    o, om = make_opts(model, arm_wd=0.1 if arm == "WD" else 0.0)
    t0 = time.time()
    tag = f"{regime}-{arm}-s{seed}"
    res = {"regime": regime, "arm": arm, "seed": seed, "Q": Q, "W": W, "T": T, "n_cons": len(cons)}
    if regime == "P":
        Z, ZL, Y, M = run_P(model, st, o, om)
        sl = slice(st.n_pos // 2, st.n_pos)
        res["score"] = metrics(Z[sl], Y[sl], M[sl], fam, cons)
        res["score_L0"] = metrics(ZL[sl], Y[sl], M[sl], fam, cons)
        np.savez_compressed(f"{TMP}/{tag}.npz", Z=Z, ZL=ZL)
    else:
        n = st.n_pos
        tr, va, te = np.arange(0, 16800), np.arange(16800, 19200), np.arange(19200, n)
        assert_disjoint(st, tr, va); assert_disjoint(st, tr, te)
        g = np.random.default_rng(seed + 7919)
        curve = []
        for e in range(21):
            zv, zlv, yv, mv = evalP(model, st, va)
            zt, zlt, yt, mt = evalP(model, st, te)
            rv, rt = metrics(zv, yv, mv, fam, cons), metrics(zt, yt, mt, fam, cons)
            row = {"epoch": e, "val_bce": rv["fam"]["all"]["bce"], "val_learn": rv["learn_bce"],
                   "test": rt, "test_L0": metrics(zlt, yt, mt, fam, cons) if e in (0, 20) else None}
            curve.append(row)
            print(tag, e, round(row["val_bce"], 4), round(rt["learn_bce"], 4), round(rt["humility"], 4), flush=True)
            if e == 20:
                break
            model.train()
            perm = g.permutation(tr)
            for k in range(0, len(perm), 128):
                P = perm[k:k + 128]
                win, y, m = st.window(P), st.label(P), st.label_mask(P).float()
                out = model(win)
                loss = masked_bce(out["z"], y, m) + masked_bce(out["l0"], y, m)
                o.zero_grad(set_to_none=True); om.zero_grad(set_to_none=True)
                loss.backward(); o.step(); om.step()
        best = min(range(len(curve)), key=lambda i: curve[i]["val_bce"])
        res["score"] = curve[-1]["test"]; res["score_L0"] = curve[-1]["test_L0"]
        res["es_epoch"] = best; res["score_es"] = curve[best]["test"]
        res["curve"] = [{k: (v if k != "test" else {kk: v[kk] for kk in ("learn_bce", "humility", "viol_all")})
                         for k, v in r.items() if k != "test_L0"} for r in curve]
    res["wall_s"] = time.time() - t0
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(tag, "done", round(res["wall_s"]), res["score"]["learn_bce"], res["score"]["humility"])


if __name__ == "__main__":
    main()
