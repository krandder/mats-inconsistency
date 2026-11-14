"""P18 CPU unit tests (results/unit_tests/*.json). Run: python tests_cpu.py [a b c d e f]

(a) --mode prog is bit-identical to P13 dtr_maze.py (kept as p13_dtr_maze_ref.py): alpha 0.5 and 1, width 16, 6 steps.
(b) DTRecallK(K=1) initial weights equal DTRecall's under the same seed (after the recur_block -> recur_blocks.0 rename).
(c) Gradients of --mode ds (first step, width 8, K 4) match an independent reference:
    detach 1: sum over t of the gradient of loss_t / T through step t alone, from a no-grad h_(t-1) (t >= 2);
    detach 0: full autograd through all T steps; and the two differ. Picks: rr = 0,1,2,3,0,...; rand = the
    random.Random(seed) sequence.
(d) Kill and resume is bit-identical (prog alpha 0.5, ds rand detach 1, ds rr detach 0; killed at step 4 of 8).
(e) eval18.py: M-ref scores 1.0 exact on m13[0:10] at 12n; a K = 4 checkpoint evaluates with round robin; an eval
    killed after one batch and resumed gives the same arrays as an uninterrupted eval.
(f) Divergence gate: a run forced to a non-finite loss stops, is saved, and is marked diverged.
"""
import json, os, random, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable
OUT = os.path.join(HERE, "results", "unit_tests")
os.makedirs(OUT, exist_ok=True)
os.makedirs("/tmp/p18gpu", exist_ok=True)
TMP = tempfile.mkdtemp(prefix="tests_", dir="/tmp/p18gpu")


def run(args, ok_codes=(0,)):
    r = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True)
    if r.returncode not in ok_codes:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise RuntimeError(f"rc={r.returncode}: {' '.join(args)}")
    return r


def sd_equal(p1, p2, key="model"):
    a = torch.load(p1, map_location="cpu", weights_only=False)[key]
    b = torch.load(p2, map_location="cpu", weights_only=False)[key]
    assert a.keys() == b.keys(), (list(a)[:3], list(b)[:3])
    return all(torch.equal(a[k], b[k]) for k in a)


def report(name, res):
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(name, json.dumps(res)[:400], flush=True)


def t_a():
    res = {"test": "prog mode bit-identical to P13 dtr_maze.py", "cases": {}}
    for al in ["0.5", "1"]:
        d1, d2 = os.path.join(TMP, f"a_ref_{al}"), os.path.join(TMP, f"a_new_{al}")
        com = ["--width", "16", "--steps", "6", "--seed", "3", "--alpha", al, "--device", "cpu", "--val_every", "1000000", "--val_n", "100"]
        run(["p13_dtr_maze_ref.py", "--name", "r", "--out", d1] + com)
        run(["train18.py", "--name", "r", "--mode", "prog", "--out", d2] + com)
        eq = sd_equal(os.path.join(d1, "r.pt"), os.path.join(d2, "r.pt"))
        l1 = json.load(open(os.path.join(d1, "r.json")))["log"]; l2 = json.load(open(os.path.join(d2, "r.json")))["log"]
        res["cases"][al] = {"weights_equal": eq, "final_loss": [l1[-1]["loss"], l2[-1]["loss"]]}
        assert eq and l1[-1]["loss"] == l2[-1]["loss"]
    res["pass"] = True
    report("a_prog_equiv", res)


def t_b():
    from dtnet import DTRecall
    from dtmod import DTRecallK, dtr_to_k1_state
    ok = True
    for s in [0, 1, 7]:
        torch.manual_seed(s); m1 = DTRecall(16, 2, 3)
        torch.manual_seed(s); m2 = DTRecallK(16, K=1)
        a, b = dtr_to_k1_state(m1.state_dict()), m2.state_dict()
        ok &= a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    torch.manual_seed(0); m4 = DTRecallK(16, K=4)
    st, hd = m4.macs_per_pos(); st1, hd1 = DTRecall(16, 2, 3).macs_per_pos()
    res = {"test": "K=1 init equals DTRecall", "pass": bool(ok), "macs_step_equal": st == st1 and hd == hd1}
    assert ok and st == st1
    report("b_k1_init", res)


def ref_grads(seed, K, order, detach, width=8, T=30):
    """Independent recomputation of train18 --mode ds's first-step gradient."""
    from dtmod import DTRecallK
    from check import expand
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = DTRecallK(width, K=K)
    z = np.load(os.path.join(HERE, "frozen_p0", "maze_train9.npz"))
    X, Y = expand(torch.as_tensor(z["train9/code"]), torch.as_tensor(np.unpackbits(z["train9/sol"], axis=-1, count=9)))
    tr = torch.as_tensor(z["train9/train_idx"]).long()
    perm = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed))]
    bi = perm[:50]
    x, y = X[bi], Y[bi].long()
    mask = x.max(1)[0] > 0
    picks = [rng.randrange(K) if order == "rand" else (t - 1) % K for t in range(1, T + 1)]
    net.zero_grad()
    if detach:
        with torch.no_grad():
            hs = [net.init(x)]
            for t in range(1, T + 1):
                hs.append(net.step(hs[-1], x, picks[t - 1]))
        for t in range(1, T + 1):
            hin = net.init(x) if t == 1 else hs[t - 1].clone()
            h = net.step(hin, x, picks[t - 1])
            l = F.cross_entropy(net.readout(h), y, reduction="none")[mask].mean() / T
            l.backward()
    else:
        h = net.init(x); L = 0
        for t in range(1, T + 1):
            h = net.step(h, x, picks[t - 1])
            L = L + F.cross_entropy(net.readout(h), y, reduction="none")[mask].mean()
        (L / T).backward()
    return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}, bi, picks


def t_c():
    res = {"test": "ds gradients match independent reference", "cases": {}}
    g = {}
    for order in ["rand", "rr"]:
        for det in [1, 0]:
            f = os.path.join(TMP, f"c_{order}_{det}.pt")
            run(["train18.py", "--name", "c", "--mode", "ds", "--K", "4", "--order", order, "--detach", str(det), "--width", "8",
                 "--steps", "10", "--seed", "5", "--device", "cpu", "--out", os.path.join(TMP, "c"), "--grad_dump", f])
            d = torch.load(f, weights_only=False)
            rg, bi, picks = ref_grads(5, 4, order, det)
            assert torch.equal(d["bi"], bi)
            assert d["info"]["_picks"] == picks, (d["info"]["_picks"][:8], picks[:8])
            mx = max((d["grads"][k] - rg[k]).abs().max().item() / (rg[k].abs().max().item() + 1e-12) for k in rg)
            same_keys = set(d["grads"]) == set(rg)
            res["cases"][f"{order}_det{det}"] = {"max_rel_diff": mx, "same_param_set": same_keys, "picks8": picks[:8]}
            assert same_keys and mx < 1e-4, mx
            g[(order, det)] = d["grads"]
        dd = max((g[(order, 1)][k] - g[(order, 0)][k]).abs().max().item() for k in g[(order, 1)])
        res["cases"][f"{order}_det1_vs_det0_maxdiff"] = dd
        assert dd > 1e-6
    assert res["cases"]["rr_det1"]["picks8"] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert len(set(res["cases"]["rand_det1"]["picks8"])) > 1
    res["pass"] = True
    report("c_ds_grads", res)


def t_d():
    res = {"test": "kill and resume bit-identical", "cases": {}}
    cases = {"prog05": ["--mode", "prog", "--alpha", "0.5"],
             "ds_rand_det": ["--mode", "ds", "--K", "4", "--order", "rand", "--detach", "1"],
             "ds_rr_bptt": ["--mode", "ds", "--K", "4", "--order", "rr", "--detach", "0"]}
    for nm, extra in cases.items():
        com = ["--width", "8", "--steps", "8", "--seed", "2", "--device", "cpu", "--val_every", "4", "--val_n", "100"] + extra
        d1, d2 = os.path.join(TMP, f"d1_{nm}"), os.path.join(TMP, f"d2_{nm}")
        run(["train18.py", "--name", "r", "--out", d1] + com)
        r = run(["train18.py", "--name", "r", "--out", d2, "--kill_at", "4"] + com, ok_codes=(3,))
        assert os.path.exists(os.path.join(d2, "r.partial"))
        run(["train18.py", "--name", "r", "--out", d2] + com)
        eq = sd_equal(os.path.join(d1, "r.pt"), os.path.join(d2, "r.pt"))
        l1 = json.load(open(os.path.join(d1, "r.json"))); l2 = json.load(open(os.path.join(d2, "r.json")))
        same_log = [(e["step"], e["loss"]) for e in l1["log"]] == [(e["step"], e["loss"]) for e in l2["log"]]
        res["cases"][nm] = {"weights_equal": eq, "log_equal": same_log, "cnt_equal": l1["cnt"] == l2["cnt"], "cnt": l1["cnt"]}
        assert eq and same_log and l1["cnt"] == l2["cnt"]
    res["pass"] = True
    report("d_resume", res)


def t_e():
    res = {"test": "eval18"}
    o = os.path.join(TMP, "e_mref")
    run(["eval18.py", "--kind", "mref", "--ckpt", "mref/dt_net.pth", "--sets", "m13", "--limit", "10", "--out", o, "--device", "cpu"])
    s = json.load(open(o + ".json"))["sets"]["m13"]
    res["mref_m13_10_exact12n"] = s["exact@12n"]
    assert s["exact@12n"] == 1.0
    # K=4 checkpoint from (d)
    ck = os.path.join(TMP, "d1_ds_rand_det", "r.pt")
    o1, o2 = os.path.join(TMP, "e_k4a"), os.path.join(TMP, "e_k4b")
    com = ["eval18.py", "--ckpt", ck, "--sets", "v9,e19", "--limit", "12", "--bs", "4", "--device", "cpu", "--ck_every", "0"]
    run(com + ["--out", o1])
    run(com + ["--out", o2, "--kill_after", "1"], ok_codes=(3,))
    assert os.path.exists(o2 + ".evck.npz")
    run(com + ["--out", o2])
    z1, z2 = np.load(o1 + ".npz"), np.load(o2 + ".npz")
    same = sorted(z1.files) == sorted(z2.files) and all(np.array_equal(z1[k], z2[k]) for k in z1.files)
    j = json.load(open(o1 + ".json"))
    res.update({"k4_eval_resume_equal": same, "k4_K": j["K"], "k4_sets": list(j["sets"])})
    assert same and j["K"] == 4
    # round robin check: eval's exact at t=30 on v9 equals a manual round-robin rollout
    from dtmod import DTRecallK
    from eval18 import load_set
    c = torch.load(ck, weights_only=False); net = DTRecallK(8, K=4); net.load_state_dict(c["model"]); net.eval()
    X, Y, n, _ = load_set("v9", os.path.join(HERE, "frozen_p0"), os.path.join(HERE, "frozen"))
    x, y = X[:12], Y[:12]
    with torch.no_grad():
        h = net.init(x)
        for t in range(1, 31):
            h = net.step(h, x, (t - 1) % 4)
        b = net.readout(h); b = b[:, 1] - b[:, 0]
    m = x.max(1)[0] > 0
    ex = (((b > 0) & m).float() == y) | (~m)
    manual = ex.flatten(1).all(1).numpy().astype(np.uint8)
    ti = list(z1["v9/t"]).index(30)
    res["rr_manual_equal"] = bool(np.array_equal(manual, z1["v9/exact"][ti]))
    assert res["rr_manual_equal"]
    res["pass"] = True
    report("e_eval", res)


def t_f():
    d = os.path.join(TMP, "f")
    # alpha = nan makes the loss non-finite at step 1
    run(["train18.py", "--name", "r", "--out", d, "--mode", "prog", "--alpha", "nan", "--width", "8", "--steps", "5",
         "--device", "cpu"])
    j = json.load(open(os.path.join(d, "r.json")))
    ck = torch.load(os.path.join(d, "r.pt"), weights_only=False)
    res = {"test": "divergence gate", "diverged": j["diverged"], "steps": j["steps"], "pt_marked": ck["diverged"] is not None}
    assert j["diverged"] is not None and ck["diverged"] is not None
    res["pass"] = True
    report("f_diverge", res)


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d", "e", "f"]
    for w in which:
        t0 = time.time()
        globals()["t_" + w]()
        print(f"test {w} ok in {time.time() - t0:.0f}s", flush=True)
