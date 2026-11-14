"""Stage-0 unit tests (CPU): ledger, pairing until the first exit, replay, PASS identity, resume, numerics flags.
Run: python test_p6.py  (tiny sizes; ~1-3 min on one thread)."""
import os, sys, json, tempfile, argparse
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6_entry as P


def args(arm, seed, outdir, **kw):
    a = P.parser().parse_args(["--arm", arm, "--seed", str(seed), "--outdir", outdir, "--device", "cpu"])
    a.iters, a.hash_every, a.ck_sec, a.L0, a.lmin = 40, 1, 1e9, 8, 8
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def load(outdir, name):
    return json.load(open(os.path.join(outdir, "train", name + ".json")))


def first_diff(h1, h2):
    for (i1, a), (i2, b) in zip(h1, h2):
        assert i1 == i2
        if a != b:
            return i1
    return None


def test_numerics():
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cudnn.deterministic is True
    print("ok numerics flags (TF32 off, deterministic)")


def test_ledger_pairing_replay(tmp):
    # forced churn: heavy tax, short maturity -> floor exits within tens of iterations
    kw = dict(tau=0.08, m=6, entry_stop=10_000)
    for arm in ("F", "EC"):
        P.run(args(arm, 0, tmp, **kw))
    ec, f = load(tmp, "EC_s0"), load(tmp, "F_s0")
    for arm in ("PBT", "EMA"):
        P.run(args(arm, 0, tmp, **kw))
    pbt, ema = load(tmp, "PBT_s0"), load(tmp, "EMA_s0")
    sched = {int(k): v for k, v in ec["exit_schedule"].items()}
    assert len(sched) >= 1, "no exits forced"
    first = min(sched)
    # ledger: asserted every iteration inside the run; check the final error and that money is conserved
    for r in (ec, f, pbt):
        assert r["max_ledger_err"] < 1e-6, r["max_ledger_err"]
    nb = sum(1 for e in f["events"] if e["kind"] == "bailout")
    assert nb >= 1 and not any(e["kind"] == "replace" for e in f["events"])
    # exits carry their causes
    ex = [e for e in ec["events"] if e["kind"] == "replace"]
    assert all(e["wealth"] < 2.0 and e["age"] >= 6 for e in ex)
    # pairing: bitwise identical parameters until the first exit, different at it
    d_f = first_diff(ec["hashes"], f["hashes"])
    d_p = first_diff(ec["hashes"], pbt["hashes"])
    assert d_f == first, (d_f, first)
    assert d_p is None or d_p >= first, (d_p, first)
    # replay: PBT and EMA reproduce EC's exit iterations and counts
    for r in (pbt, ema):
        got = {}
        for e in r["events"]:
            if e["kind"] == "replace":
                got[e["it"]] = got.get(e["it"], 0) + 1
        short = [e for e in r["events"] if e["kind"] == "replay_short"]
        assert got == sched or short, (got, sched)
    d_e = first_diff(ec["hashes"], ema["hashes"])
    s_small = sum(l.get("win", {}).get("s_small", 0) for l in ec["log"])
    print(f"ok ledger (max rel err {max(ec['max_ledger_err'], f['max_ledger_err'], pbt['max_ledger_err']):.1e}), "
          f"{len(ex)} exits, {nb} F bailouts; pairing F/PBT identical until first exit it={first} "
          f"(F diverges at {d_f}, PBT at {d_p}); EMA diverges at {d_e}; replay ok")
    return first, d_e


def test_resume(tmp):
    a = args("EC", 1, tmp, tau=0.08, m=6, iters=30)
    P.run(a)
    full = load(tmp, "EC_s1")
    os.remove(os.path.join(tmp, "train", "EC_s1.json"))
    a2 = args("EC", 1, tmp, tau=0.08, m=6, iters=30, stop_at=13)
    P.run(a2)
    a3 = args("EC", 1, tmp, tau=0.08, m=6, iters=30)
    P.run(a3)
    res = load(tmp, "EC_s1")
    assert res["hashes"] == full["hashes"], first_diff(res["hashes"], full["hashes"])
    assert res["exit_schedule"] == full["exit_schedule"]
    print("ok resume: preempted-at-13 run is bitwise identical to the straight run")


def test_pass_identity():
    a = args("F", 0, "/tmp")
    M = P.Econ("F", 0, "cpu", a)
    with torch.no_grad():
        M.P["q_w2"].zero_(); M.P["q_b2"].fill_(-5.0)     # every bid is -5: PASS everywhere
    g = torch.Generator().manual_seed(1)
    x, y, m = P.prefix_batch(16, 32, 32, g, "cpu")
    b0 = torch.randn(16, 32); h0 = torch.randn(16, P.H, 32)
    out, info = M.rollout(x, m, y, [4, 64], b0=b0, h0=h0, want_h=True)
    assert torch.equal(out[64][0], b0) and torch.equal(out[64][1], h0)
    assert (info["halt_step"] == 0).all() and info["applied"].sum() == 0
    # training: passes are free, cost is charged per executed trade only
    gp = torch.Generator().manual_seed(2)
    M.train_iter(x, y, m, 32, 0, gp)
    w = M.win
    c = 0.1 / 32
    assert w["passes"] > 0 and w["trades"] + w["passes"] == w["rowchunks"]
    assert abs(M.sink - w["trades"] * c / 16) < 1e-9, (M.sink, w["trades"] * c / 16)
    # PASS rows are exactly the non-exploration rows here (all bids <= 0)
    print(f"ok PASS: identity step, zero cost ({w['passes']} passes, {w['trades']} exploration trades)")


def test_plain_and_single(tmp):
    for arm in ("R0", "DTL", "R1"):
        P.run(args(arm, 0, tmp, iters=12))
    import eval_ladder as E
    E.DMAX = 128        # CPU test: short ladder
    for name, var in (("R0_s0", "plain"), ("DTL_s0", "plain"), ("R1_s0", "auction"), ("EC_s0", "auction"), ("EC_s0", "single")):
        arm, M = E.load_model(os.path.join(tmp, "models", name + ".pt"), "cpu")
        slot = E.richest(arm, M) if var == "single" else None
        rec = E.eval_cell(arm, M, 32, var, "cpu", slot)
        assert rec["grid"][-1][0] == 128 and rec["D4"] == 128 and 0 <= rec["exact4"] <= 1, rec
        pi = E.eval_pi(arm, M, 32, var, "cpu", slot)
        assert 0 <= pi["all4_agree"] <= 1
    print("ok R0 / DT-L / R1 train + eval cells (auction, single, plain, pi)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_numerics()
    test_pass_identity()
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        test_ledger_pairing_replay(tmp)
        test_resume(tmp)
        if "--quick" not in sys.argv:
            test_plain_and_single(tmp)
    print("ALL TESTS PASSED")
