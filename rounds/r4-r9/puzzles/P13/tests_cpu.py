"""P13 CPU unit tests (a)-(h). python tests_cpu.py [a b c d e f g h]; writes results/unit_tests.json.
Tiny widths and batches (General's CPU is shared); float64 where equality within 1e-5 is asserted."""
import copy, json, os, subprocess, sys, time
from types import SimpleNamespace
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import econ_maze as E  # noqa: E402
PY = sys.executable
OUTD = os.path.join(HERE, "results", "unit_tests")
TMP = "/tmp/p13t"
os.makedirs(TMP, exist_ok=True); os.makedirs(OUTD, exist_ok=True)


def args(**kw):
    a = E.get_args(["--out", "/tmp/p13t/x", "--device", "cpu"])
    a.width, a.N, a.D, a.bs = 8, 4, 6, 3
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def batch(bs=3, seed=0):
    X, Y, tr, va = DATA
    g = torch.Generator().manual_seed(seed)
    bi = tr[torch.randperm(len(tr), generator=g)[:bs]]
    x, y = X[bi].double(), Y[bi].double()
    return x, y, (x.max(1)[0] > 0).double()


def clone_weights(src, dst):
    for s, d in zip(src.traders, dst.traders):
        d.load_state_dict(s.state_dict())


def maxdiff_params(M1, M2):
    return max((p1 - p2).abs().max().item() for t1, t2 in zip(M1.traders, M2.traders)
               for p1, p2 in zip(t1.parameters(), t2.parameters()))


def test_a():
    out = {}
    t = torch.arange(2 * 5 * 6, dtype=torch.float64).view(2, 5, 6)
    for dy, dx in [(1, 0), (0, 2), (-1, 3), (4, -5), (7, 0)]:
        s = E.shift2d(t, dy, dx)
        for r in range(5):
            for c in range(6):
                ref = t[:, r + dy, c + dx] if 0 <= r + dy < 5 and 0 <= c + dx < 6 else torch.zeros(2, dtype=t.dtype)
                assert torch.equal(s[:, r, c], ref), (dy, dx, r, c)
    out["shift2d"] = "ok"
    lv = {s: E.nlevels2d(s) for s in [24, 32, 44, 72, 124, 204, 408]}
    assert lv == {24: 5, 32: 5, 44: 6, 72: 7, 124: 7, 204: 8, 408: 9}, lv
    out["nlevels"] = lv
    g = torch.Generator().manual_seed(1)
    y = (torch.rand(3, 10, 12, generator=g) > 0.5).double()
    m = (torch.rand(3, 10, 12, generator=g) > 0.2).double()
    eq, meq = E.pair_truth2d(y, m, 3)
    assert eq.shape == (3, 3, 2, 10, 12)
    for j in range(3):
        d = 2 ** j
        for r in range(10):
            for c in range(12):
                yv = y[:, r + d, c] if r + d < 10 else torch.zeros(3, dtype=y.dtype)
                yh = y[:, r, c + d] if c + d < 12 else torch.zeros(3, dtype=y.dtype)
                mv = m[:, r + d, c] if r + d < 10 else torch.zeros(3, dtype=y.dtype)
                mh = m[:, r, c + d] if c + d < 12 else torch.zeros(3, dtype=y.dtype)
                assert torch.equal(eq[:, j, 0, r, c], (y[:, r, c] == yv).double())
                assert torch.equal(eq[:, j, 1, r, c], (y[:, r, c] == yh).double())
                assert torch.equal(meq[:, j, 0, r, c], m[:, r, c] * mv)
                assert torch.equal(meq[:, j, 1, r, c], m[:, r, c] * mh)
    out["pair_truth2d"] = "ok"
    # Hillis-Steele composition on real 9x9 mazes: eq_j(p) = XNOR(eq_(j-1)(p), eq_(j-1)(p + 2^(j-1) e_a))
    x, y, m = batch(8, seed=5)
    J = E.nlevels2d(24)
    eq, _ = E.pair_truth2d(y, m, J)
    bad = 0
    for j in range(1, J):
        h = 2 ** (j - 1)
        for ax, (dy, dx) in enumerate([(h, 0), (0, h)]):
            comp = 1 - (eq[:, j - 1, ax] - E.shift2d(eq[:, j - 1, ax], dy, dx)).abs()
            # the shifted term leaves the image where p + 2^j e_a is outside: compare only inside
            H = 24
            inside = torch.ones(H, H, dtype=torch.bool)
            if dy:
                inside[H - 2 * h:, :] = False
            else:
                inside[:, H - 2 * h:] = False
            bad += int((comp[:, inside] != eq[:, j, ax][:, inside]).sum())
    assert bad == 0, bad
    out["xnor_composition_9x9_mazes"] = "ok (8 mazes, levels 1..4, both axes)"
    return out


def run_pair(a1, a2, iters=3, seed=0, check_traj=False):
    """train two models from identical weights on identical batches; returns per-iteration max param diff"""
    torch.manual_seed(seed)
    M1 = E.Model(a1, "cpu")
    M2 = E.Model(a2, "cpu")
    clone_weights(M1, M2)
    init = [{k: v.detach().clone() for k, v in t.named_parameters()} for t in M1.traders]
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    diffs, out = [], {}
    for it in range(iters):
        x, y, m = batch(a1.bs, seed=100 + it)
        l1 = M1.train_step(x, y, m, a1.D, g1)
        l2 = M2.train_step(x, y, m, a2.D, g2)
        diffs.append(maxdiff_params(M1, M2))
    # non-vacuity: how far the updates moved the weights (whole net and the v / bid heads)
    M1.moved = {"all": max((v - i0[k]).abs().max().item() for t, i0 in zip(M1.traders, init) for k, v in t.named_parameters()),
                "out.weight": max((dict(t.named_parameters())["out.weight"] - i0["out.weight"]).abs().max().item() for t, i0 in zip(M1.traders, init)),
                "gain.2.weight": max((dict(t.named_parameters())["gain.2.weight"] - i0["gain.2.weight"]).abs().max().item() for t, i0 in zip(M1.traders, init))}
    return diffs, M1, M2


def test_b():
    torch.set_default_dtype(torch.float64)
    a1 = args(mode="econ", wealth_cap=0, train_pick="random")
    a2 = args(mode="bptt", tbptt=1, ds=1, covsup=1, vocsup=1, bidsup=1, ptopt=1, train_pick="random")
    diffs, M1, M2 = run_pair(a1, a2)
    heads = {}
    for name in ["out.weight", "gain.2.weight"]:
        heads[name] = max((dict(t1.named_parameters())[name] - dict(t2.named_parameters())[name]).abs().max().item()
                          for t1, t2 in zip(M1.traders, M2.traders))
    torch.set_default_dtype(torch.float32)
    assert max(diffs) < 1e-5, diffs
    assert min(M1.moved.values()) > 1e-6, M1.moved
    return {"max_param_diff_per_iter": diffs, "params_moved_by_updates": M1.moved, "head_diffs(v rows are out.weight[3:], bid is gain)": heads,
            "iters": 3, "note": "ECON --wealth_cap 0 vs bptt --tbptt 1 --ds 1 --covsup 1 --vocsup 1 --bidsup 1 --ptopt 1, "
                                "random pick, float64, after per-trader clip 1.0 and Adam"}


def test_c():
    torch.set_default_dtype(torch.float64)
    a1 = args(mode="econ", wealth_cap=1, W0=1e9)
    a2 = args(mode="econ", wealth_cap=0)
    diffs, M1, M2 = run_pair(a1, a2)
    torch.set_default_dtype(torch.float32)
    assert max(diffs) < 1e-5, diffs
    assert min(M1.moved.values()) > 1e-6, M1.moved
    return {"max_param_diff_per_iter": diffs, "params_moved_by_updates": M1.moved}


FIXES = {"F0": {}, "F-TR1 dbcap1": {"dbcap": 1.0}, "F-TR2 liq sqrt": {"liq": "sqrt"}, "liq t": {"liq": "t"},
         "F-LB bcap10": {"bcap": 10.0}, "bcap 0.3 (binding)": {"bcap": 0.3}, "dbcap 0.05 (binding)": {"dbcap": 0.05},
         "F-HF l2b1e-3": {"l2b": 1e-3}, "F-FL floor0.01": {"floor": 0.01}, "F-IN insq": {"insq": 1}}


def test_d():
    """for each fix flag: (1) econ rollout vs bptt rollout (with grad) from identical weights: b, c, v over 30 steps;
    (2) the econ settlement's recomputed post-trade state equals the rollout's next state at every step;
    (3) eval_maze.trader_step (round robin) equals the rollout at pick=roundrobin.
    Extra case 'wealth_cap W0=0.02': econ with binding wealth scale (checks (2) and (3) only)."""
    torch.set_default_dtype(torch.float64)
    import eval_maze as EV
    out = {}
    x, y, m = batch(2, seed=9)
    cases = [(k, v, 0) for k, v in FIXES.items()] + [("wealth_cap W0=0.02", {"W0": 0.02}, 1)]
    for name, kw, wc in cases:
        a1 = args(mode="econ", wealth_cap=wc, D=30, bs=2, **kw)
        a2 = args(mode="bptt", ds=1, covsup=1, vocsup=1, D=30, bs=2, **kw)
        torch.manual_seed(3)
        M1 = E.Model(a1, "cpu"); M2 = E.Model(a2, "cpu"); clone_weights(M1, M2)
        with torch.no_grad():   # scale the out head so trades are large and the caps bind
            for M in (M1, M2):
                for t in M.traders:
                    t.out.weight.mul_(20.0)
        g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
        recs, bs1 = M1.rollout(x, m, 30, pick="random", gen=g1, keep=True)
        c1s = [r["c"] for r in recs[1:]] + [M1.last[1]]
        v1s = [r["v"] for r in recs[1:]] + [M1.last[2]]
        r = {}
        if not wc:
            _, bs2 = M2.rollout(x, m, 30, pick="random", gen=g2, grad=True, rec=("c", "v"))
            r["rollout_econ_vs_bptt"] = [max((p - q.detach()).abs().max().item() for p, q in zip(bs1[1:], bs2[1:])),
                                         max((p - q.detach()).abs().max().item() for p, q in zip(c1s, M2.cs)),
                                         max((p - q.detach()).abs().max().item() for p, q in zip(v1s, M2.vs))]
        else:
            S = torch.stack([q["s"] for q in recs])
            r["frac_steps_s<1"] = float((S < 1).double().mean())
        M1.debug, M1.dbg = True, {}
        M1c = copy.deepcopy(M1)
        M1.train_step(x, y, m, 30, torch.Generator().manual_seed(4))
        r["settlement_vs_rollout"] = [max((M1.dbg[t][0] - bs1[t + 1]).abs().max().item() for t in range(30)),
                                      max((M1.dbg[t][1] - c1s[t]).abs().max().item() for t in range(30)),
                                      max((M1.dbg[t][2] - v1s[t]).abs().max().item() for t in range(30))]
        _, bs3 = M1c.rollout(x, m, 30, pick="roundrobin")
        st = (torch.zeros_like(x[:, 0]), torch.zeros(2, 5, 2, 24, 24), torch.zeros(2, 3, 24, 24))
        de = 0.0
        with torch.no_grad():
            for t in range(1, 31):
                st, b = EV.trader_step(M1c, st, t, x, m, 1.0)
                de = max(de, (b - bs3[t]).abs().max().item())
        r["eval_vs_rollout_rr"] = de
        r["max|b|"] = round(max(q.abs().max().item() for q in bs1), 3)
        out[name] = r
        worst = max([de] + r["settlement_vs_rollout"] + r.get("rollout_econ_vs_bptt", []))
        assert worst < 1e-9, (name, r)
        if "bcap" in kw:
            assert r["max|b|"] <= kw["bcap"] + 1e-9
        if wc:
            assert r["frac_steps_s<1"] > 0, "wealth cap case must bind"
    torch.set_default_dtype(torch.float32)
    return out


def sh(cmd, ok=(0,)):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode not in ok:
        raise RuntimeError(f"{cmd} -> {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r


def test_e():
    base = [PY, "econ_maze.py", "--mode", "econ", "--width", "8", "--D", "4", "--bs", "2", "--iters", "50",
            "--W0", "0.05", "--frac", "1.0", "--log_every", "5", "--diag_every", "10", "--ckpt_every", "0", "--device", "cpu"]
    for f in ["e_full", "e_res"]:
        for ext in [".pt", ".json", ".ckpt.pt"]:
            if os.path.exists(f"{TMP}/{f}{ext}"):
                os.remove(f"{TMP}/{f}{ext}")
    sh(base + ["--out", f"{TMP}/e_full"])
    sh(base + ["--out", f"{TMP}/e_res", "--kill_at", "23"], ok=(3,))
    assert "resumed_at_it" in sh(base + ["--out", f"{TMP}/e_res"]).stdout
    A = torch.load(f"{TMP}/e_full.pt", weights_only=False); B = torch.load(f"{TMP}/e_res.pt", weights_only=False)
    same_w = all(torch.equal(A["traders"][i][k], B["traders"][i][k]) for i in range(4) for k in A["traders"][i])
    la = [{k: v for k, v in l.items() if k != "sec"} for l in A["log"]]
    lb = [{k: v for k, v in l.items() if k != "sec"} for l in B["log"]]
    ok = same_w and A["wealth"] == B["wealth"] and A["stats"] == B["stats"] and la == lb
    out = {"econ_bit_identical_50_iters_kill_at_23": ok, "respawns": A["stats"]["respawns"], "wealth": A["wealth"]}
    assert ok, out
    assert A["stats"]["respawns"] > 0, "test must cover respawns"
    # bptt path (joint optimizer is one Adam; ptopt MultiOpt) resume
    base2 = [PY, "econ_maze.py", "--mode", "bptt", "--ds", "1", "--covsup", "1", "--vocsup", "1", "--ptopt", "1",
             "--prog", "1", "--alpha", "0.01", "--width", "8", "--D", "4", "--bs", "2", "--iters", "20",
             "--log_every", "5", "--ckpt_every", "0", "--device", "cpu"]
    for f in ["b_full", "b_res"]:
        for ext in [".pt", ".json", ".ckpt.pt"]:
            if os.path.exists(f"{TMP}/{f}{ext}"):
                os.remove(f"{TMP}/{f}{ext}")
    sh(base2 + ["--out", f"{TMP}/b_full"]); sh(base2 + ["--out", f"{TMP}/b_res", "--kill_at", "9"], ok=(3,))
    assert "resumed_at_it" in sh(base2 + ["--out", f"{TMP}/b_res"]).stdout
    A = torch.load(f"{TMP}/b_full.pt", weights_only=False); B = torch.load(f"{TMP}/b_res.pt", weights_only=False)
    ok2 = all(torch.equal(A["traders"][i][k], B["traders"][i][k]) for i in range(4) for k in A["traders"][i])
    out["bptt_progcov_bit_identical_20_iters_kill_at_9"] = ok2
    assert ok2
    # DTR resume
    for f in ["d_full.pt", "d_res.pt", "d_full.partial", "d_res.partial"]:
        if os.path.exists(f"{TMP}/dtr/{f}"):
            os.remove(f"{TMP}/dtr/{f}")
    b3 = [PY, "dtr_maze.py", "--width", "8", "--steps", "40000", "--val_n", "50", "--device", "cpu", "--out", f"{TMP}/dtr"]
    sh(b3 + ["--name", "d_full", "--stop_at", "30"])
    r = subprocess.Popen(b3 + ["--name", "d_res", "--stop_at", "30", "--ckpt_sec", "0"], cwd=HERE, stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    # --ckpt_sec 0 checkpoints every step; kill the process after step 12's checkpoint appears
    t0 = time.time()
    while time.time() - t0 < 900:
        if os.path.exists(f"{TMP}/dtr/d_res.partial"):
            s = torch.load(f"{TMP}/dtr/d_res.partial", weights_only=False)
            if s["step"] >= 12:
                r.kill(); break
        time.sleep(0.5)
    r.wait()
    assert "resumed_at" in sh(b3 + ["--name", "d_res", "--stop_at", "30"]).stdout, "dtr run finished before the kill"
    A = torch.load(f"{TMP}/dtr/d_full.pt", weights_only=False); B = torch.load(f"{TMP}/dtr/d_res.pt", weights_only=False)
    ok3 = all(torch.equal(A["model"][k], B["model"][k]) for k in A["model"])
    out["dtr_bit_identical_30_steps_killed_mid_run"] = ok3
    assert ok3
    return out


def test_f():
    sh([PY, "eval_maze.py", "--kind", "mref", "--ckpt", os.path.join(HERE, "..", "P0", "src", "dt_net.pth"),
        "--sets", "m13", "--limit", "20", "--device", "cpu", "--out", f"{TMP}/mref_m13"])
    s = json.load(open(f"{TMP}/mref_m13.json"))["sets"]["m13"]
    assert s["exact@12n"] == 1.0, s
    return {"mref_m13_20_exact@12n": s["exact@12n"], "ll_raw@12n": s["ll_raw@12n"]}


def test_g():
    import flops
    t = flops.table()
    json.dump(t, open(os.path.join(HERE, "results", "flops.json"), "w"), indent=1)
    r = t["sizes"][9]["none"]["ratio_to_dtr_step"]
    assert 0.75 <= r <= 1.25, r
    return {"trader_vs_dtr_step_macs_at_24px": r, "t_eq@99": {k: t["sizes"][99][k]["t_eq"] for k in ["none", "scale", "cap"]},
            "t_eq@201": {k: t["sizes"][201][k]["t_eq"] for k in ["none", "scale", "cap"]}, "file": "results/flops.json"}


def test_h():
    p0 = os.path.join(HERE, "..", "P0")
    for f in [f"{TMP}/h_p0/h.pt", f"{TMP}/h_p13/h.pt"]:
        if os.path.exists(f):
            os.remove(f)
    sh([PY, os.path.join(p0, "train.py"), "--name", "h", "--task", "maze", "--mode", "bansal", "--alpha", "0.01",
        "--width", "8", "--steps", "200", "--seed", "0", "--frozen", os.path.join(p0, "frozen"), "--out", f"{TMP}/h_p0",
        "--device", "cpu", "--probe_at", ""])
    sh([PY, "dtr_maze.py", "--name", "h", "--width", "8", "--steps", "40000", "--warmup_frac", "0.2", "--stop_at", "200",
        "--seed", "0", "--out", f"{TMP}/h_p13", "--device", "cpu"])
    A = torch.load(f"{TMP}/h_p0/h.pt", weights_only=False); B = torch.load(f"{TMP}/h_p13/h.pt", weights_only=False)
    d = max((A["model"][k] - B["model"][k]).abs().max().item() for k in A["model"])
    la = [(l["step"], l["loss"]) for l in A["log"]]; lb = [(l["step"], l["loss"]) for l in B["log"]]
    assert d <= 1e-6 and la == lb, (d, la, lb)
    return {"max_param_diff_after_200_steps": d, "loss@200": la, "val9@200": [A["log"][-1].get("val9_acc30"), B["log"][-1].get("val9_acc30")],
            "width": 8, "note": "P0 train.py --steps 200 vs dtr_maze.py --steps 40000 --warmup_frac 0.2 --stop_at 200"}


if __name__ == "__main__":
    which = sys.argv[1:] or list("abcdefgh")
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    DATA = E.load_train(os.path.join(HERE, "frozen_p0"), "cpu")
    res = {}
    for w in which:
        t0 = time.time()
        try:
            r = globals()[f"test_{w}"]()
            res[w] = {"pass": True, **r, "sec": round(time.time() - t0, 1)}
        except Exception as ex:   # noqa: BLE001
            import traceback
            res[w] = {"pass": False, "error": repr(ex)[:3000], "tb": traceback.format_exc()[-3000:]}
        res[w]["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(w, json.dumps(res[w])[:1500], flush=True)
        json.dump(res[w], open(os.path.join(OUTD, f"{w}.json"), "w"), indent=1)
