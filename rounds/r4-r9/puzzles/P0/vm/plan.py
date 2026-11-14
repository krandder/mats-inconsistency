"""P0 job plans per VM lane. plan(name, out, py, cuts) -> (jobs, pools).

Cuts (env P0_CUTS, comma list, applied in the spec's order when the pilot timing does not fit):
  stretch  -> stretch set for M-ref only;  p6 -> drop P6;  m5 -> M5 one seed;
  m1long30 -> M1-long stops at 30k;       pl2 -> P-L two seeds.
"""
import json, os

MAZE_TEST = "v33,m9,m13,m33,m33[0:100]@3000=m33h,m59"
_MAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frozen", "manifest.json")
SHIFT = ",".join(sorted(k for k in json.load(open(_MAN))["sets"] if k.startswith("sh_"))) if os.path.exists(_MAN) else ""
PREFIX_TEST = "vL128,vL512,fL128,fL256,fL512,e2h512"
LEGACY = ",".join("legacy_" + k for k in ("id_8_32", "L32", "L48", "mix_8_64", "L64", "L128"))


def plan(name, out, py, cuts):
    ck, ev = os.path.join(out, "ckpt"), os.path.join(out, "eval")
    J = []

    def tr(jid, pool, args, **kw):
        J.append({"id": jid, "pool": pool, "cmd": [py, "train.py", "--out", ck] + args, **kw})

    def evj(jid, pool, model, sets, probe="", extra=(), **kw):
        cmd = [py, "evalrun.py", "--ckpt", ck, "--model", model, "--sets", sets, "--out", os.path.join(ev, jid[3:])]
        if probe:
            cmd += ["--probe", probe]
        J.append({"id": jid, "pool": pool, "cmd": cmd + list(extra), **kw})

    maze = ["--task", "maze", "--width", "64", "--T", "30"]
    pre = ["--task", "prefix", "--T", "30", "--steps", "10000"]

    if name == "pilot-a":
        pools = {"gpu": 1}
        # smoke first: unit tests, kill/resume equality, every save/val/split/load path at toy length
        J.append({"id": "unit_tests", "pool": "gpu", "tries": 1, "cmd": ["bash", "-c",
                  f"P0_KNUTSON={os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/knutson "
                  f"P0_TESTS_OUT={out}/unit_tests.json P0_THREADS=4 {py} tests.py"]})
        J.append({"id": "resume_check", "pool": "gpu", "tries": 1, "cmd": [py, "vm/resume_check.py", out]})
        tr("tr_SM1_s0", "gpu", ["--name", "SM1_s0"] + maze + ["--mode", "bansal", "--alpha", "0.01", "--steps", "400",
                                "--val_every", "200", "--probe_at", "400", "--split_at", "200", "--split_name", "SM1h_s0"])
        evj("ev_SM1_s0_best", "gpu", "SM1:0:best", "m9[0:50],m33[0:20]@100=m33s,sh_n9_p0_de0[0:20]", probe="v9",
            after=["tr_SM1_s0"])
        evj("ev_SM1h_s0", "gpu", "SM1h:0", "m9[0:20]", after=["tr_SM1_s0"])
        tr("tr_SP1_s0", "gpu", ["--name", "SP1_s0", "--task", "prefix", "--T", "30", "--steps", "200", "--mode", "bansal"])
        evj("ev_SP1_s0", "gpu", "SP1:0", "fL128[0:64]@100=fL128s", probe="v_id", after=["tr_SP1_s0"])
        J.append({"id": "tr_SMRP_s0", "pool": "gpu", "cmd": [py, "mr_train.py", "--arm", "P", "--T", "24", "--iters", "50",
                  "--seed", "0", "--out", os.path.join(ck, "SMRP_s0.pt")]})
        evj("ev_SMRP_s0", "gpu", f"MRfile:{ck}/SMRP_s0.pt", "legacy_L32[0:64]", probe="v_id", after=["tr_SMRP_s0"])
        # R1 on M-ref: shift sets (9x9 de0 = 0.72 +- 0.08; n=33 vs e2h), small e2h sets, bitwise repeat on m9
        evj("ev_Mref_r1", "gpu", "M-ref", "sh_n9_p0_de0,sh_n9_p0_de1,sh_n33_p0_de1,m9,m13,m33[0:100]@100=m33r1",
            extra=["--repeat", "m9"])
        evj("ev_Mref_m59x20_fp32", "gpu", "M-ref", "m59[0:20]=m59x20")
        evj("ev_Mref_m59x20_tf32", "gpu", "M-ref", "m59[0:20]=m59x20", extra=["--tf32"])
        for arm, args in [("M1", ["--mode", "bansal", "--alpha", "0.01"]), ("Mprog", ["--mode", "bansal", "--alpha", "1"])]:
            tr(f"time_{arm}", "gpu", ["--name", f"time_{arm}"] + maze + args + ["--steps", "15000", "--time_steps", "300"])
        return J, pools

    if name == "pilot-b":
        pools = {"gpu": 1, "cpu": 1}
        for a in ("A", "B", "Bds", "C", "D"):
            evj(f"ev_pilotMR_{a}", "gpu", f"MRfile:{out}/../data/pilot_{a}.pt", LEGACY)
        for arm, args in [("M4", maze + ["--mode", "chunk", "--k", "1"]), ("M5", maze + ["--mode", "chunk", "--k", "8"])]:
            tr(f"time_{arm}", "gpu", ["--name", f"time_{arm}"] + args + ["--steps", "15000", "--time_steps", "300"])
        for arm, args in [("P1", ["--mode", "bansal", "--alpha", "1"]), ("P4", ["--mode", "chunk", "--k", "1"]),
                          ("P5", ["--mode", "chunk", "--k", "8"]), ("PL", ["--arch", "dtl", "--mode", "bansal"]),
                          ("P6", ["--mode", "armB"])]:
            tr(f"time_{arm}", "cpu", ["--name", f"time_{arm}"] + pre + args + ["--time_steps", "300"])
        J.append({"id": "time_MRP", "pool": "cpu", "cmd": ["bash", "-c",
                  f"timeout 120 {py} mr_train.py --arm P --T 24 --iters 10000 --seed 0 --out {ck}/time_MRP.pt; true"]})
        return J, pools

    if name == "main-a":
        pools = {"gpu": 4}
        m1 = ["--mode", "bansal", "--alpha", "0.01"]
        long_steps = "30000" if "m1long30" in cuts else "40000"
        tr("tr_M1long_s0", "gpu", ["--name", "M1long_s0"] + maze + m1 + ["--steps", long_steps, "--seed", "0",
                                   "--split_at", "15000", "--split_name", "M1_s0"])
        for s in (1, 2):
            tr(f"tr_M1_s{s}", "gpu", ["--name", f"M1_s{s}"] + maze + m1 + ["--steps", "15000", "--seed", str(s)])
        evj("ev_Mref", "gpu", "M-ref", MAZE_TEST + "," + SHIFT, probe="v9", extra=["--repeat", "m9"])
        for s in (0, 1, 2):
            tr(f"tr_Mprog_s{s}", "gpu", ["--name", f"Mprog_s{s}"] + maze + ["--mode", "bansal", "--alpha", "1",
                                         "--steps", "15000", "--seed", str(s)])
        need0 = [(os.path.join(ck, "M1_s0_best.pt"), "tr_M1long_s0"), (os.path.join(ck, "M1_s0.pt"), "tr_M1long_s0")]
        for s in (0, 1, 2):
            dep = {"need": need0} if s == 0 else {"after": [f"tr_M1_s{s}"]}
            evj(f"ev_M1_s{s}_best", "gpu", f"M1:{s}:best", MAZE_TEST + "," + SHIFT, probe="v9", **dep)
            evj(f"ev_M1_s{s}_final", "gpu", f"M1:{s}", "v33,m33", **dep)
        for s in (0, 1, 2):
            evj(f"ev_Mprog_s{s}_best", "gpu", f"Mprog:{s}:best", MAZE_TEST, probe="v9", after=[f"tr_Mprog_s{s}"])
            evj(f"ev_Mprog_s{s}_final", "gpu", f"Mprog:{s}", "v33,m33", after=[f"tr_Mprog_s{s}"])
        evj("ev_M1long_s0_best", "gpu", "M1long:0:best", MAZE_TEST + "," + SHIFT, probe="v9", after=["tr_M1long_s0"])
        evj("ev_M1long_s0_final", "gpu", "M1long:0", MAZE_TEST, after=["tr_M1long_s0"])
        evj("ev_M1_s0_stab", "gpu", "M1:0:best", "m33[0:50]@30000=m33stab", need=need0)
        # the 201x201 stretch runs last
        evj("ev_Mref_stretch", "gpu", "M-ref", "m201",
            wait=["ev_Mref"] + [f"ev_{a}_s{s}_best" for a in ("M1", "Mprog") for s in (0, 1, 2)])
        if "stretch" not in cuts:
            evj("ev_M1long_s0_stretch", "gpu", "M1long:0:best", "m201", after=["tr_M1long_s0"], wait=["ev_M1long_s0_best", "ev_Mref_stretch"])
        return J, pools

    if name == "main-b":
        pools = {"gpu": 2, "cpu": 4}
        # maze credit ladder (GPU-bound, 2 at a time); each run's test job is queued right after it (priority = list order)
        mz = [("M4", s, ["--k", "1", "--steps", "15000"]) for s in (0, 1, 2)]
        mz += [("M4cm", 0, ["--k", "1", "--steps", "33000"])]
        mz += [("M5", s, ["--k", "8", "--steps", "15000"]) for s in ((0,) if "m5" in cuts else (0, 1))]
        for arm, s, args in mz:
            tr(f"tr_{arm}_s{s}", "gpu", ["--name", f"{arm}_s{s}"] + maze + ["--mode", "chunk", "--seed", str(s)] + args)
            evj(f"ev_{arm}_s{s}_best", "gpu", f"{arm}:{s}:best", MAZE_TEST, probe="v9", after=[f"tr_{arm}_s{s}"])
            evj(f"ev_{arm}_s{s}_final", "gpu", f"{arm}:{s}", "v33,m33", after=[f"tr_{arm}_s{s}"])
        # prefix queue (CPU/launch-bound, 4 at a time), priority order; test jobs are cheap and share the cpu pool
        runs = [("P1", s, ["--mode", "bansal", "--alpha", "1"]) for s in range(8)]
        runs += [("P4", s, ["--mode", "chunk", "--k", "1"]) for s in range(8)]
        runs += [("MRP", s, None) for s in range(3)]
        runs += [("P5", s, ["--mode", "chunk", "--k", "8"]) for s in range(4)]
        runs += [("P4cm", s, ["--mode", "chunk", "--k", "1", "--steps", "22400"]) for s in range(3)]
        runs += [("PL", s, ["--arch", "dtl", "--mode", "bansal", "--alpha", "1"]) for s in range(2 if "pl2" in cuts else 3)]
        if "p6" not in cuts:
            runs += [("P6", s, ["--mode", "armB"]) for s in range(3)]
        for arm, s, args in runs:
            if arm == "MRP":
                J.append({"id": f"tr_MRP_s{s}", "pool": "cpu", "cmd": [py, "mr_train.py", "--arm", "P", "--T", "24",
                          "--iters", "10000", "--seed", str(s), "--out", os.path.join(ck, f"MRP_s{s}.pt")]})
                model, sets = f"MRfile:{ck}/MRP_s{s}.pt", PREFIX_TEST + "," + LEGACY
            else:
                tr(f"tr_{arm}_s{s}", "cpu", ["--name", f"{arm}_s{s}"] + pre + args + ["--seed", str(s)])
                model, sets = f"{arm}:{s}", PREFIX_TEST
            evj(f"ev_{arm}_s{s}", "cpu", model, sets, probe="v_id", after=[f"tr_{arm}_s{s}"])
            if arm in ("P1", "P4") and s in (0, 1):
                evj(f"ev_{arm}_s{s}_stab", "cpu", model, "fL512[0:100]@30000=fL512stab", after=[f"tr_{arm}_s{s}"])
        return J, pools

    if name in ("main-a2", "main-b2"):
        # Re-plan at 14:05 UTC on 29 Sep: measured strict-fp32 eval runs at ~5.4M pixel-iterations/s on an L4 and
        # maze jobs gain nothing from sharing the GPU, so the original lanes needed ~2x the VM time left.
        # Cuts 1-4 of the spec order are applied (stretch M-ref only, no P6, M5 one seed, M1-long 30k); evals use
        # best-validation checkpoints only; m59/shift/stability/stretch evals move to the tail of each queue.
        M1SETS = "v33,m9,m13,m33,m33[0:100]@3000=m33h"
        LADDER = "v33,m9,m13,m33"
        tail = []

        def evt(jid, pool, model, sets, **kw):   # low-priority eval, queued after everything else
            n0 = len(J); evj(jid, pool, model, sets, **kw); tail.append(J.pop(n0))
    if name == "main-a2":
        pools = {"gpu": 2}
        m1 = ["--mode", "bansal", "--alpha", "0.01"]
        long_steps = "30000" if "m1long30" in cuts else "40000"
        tr("tr_M1long_s0", "gpu", ["--name", "M1long_s0"] + maze + m1 + ["--steps", long_steps, "--seed", "0",
                                   "--split_at", "15000", "--split_name", "M1_s0"])
        for s in (1, 2):
            tr(f"tr_M1_s{s}", "gpu", ["--name", f"M1_s{s}"] + maze + m1 + ["--steps", "15000", "--seed", str(s)])
        evj("ev_Mref", "gpu", "M-ref", MAZE_TEST, probe="v9")
        evt("ev_Mref_shift", "gpu", "M-ref", SHIFT)
        need0 = [(os.path.join(ck, "M1_s0_best.pt"), "tr_M1long_s0"), (os.path.join(ck, "M1_s0.pt"), "tr_M1long_s0")]
        for s in (0, 1, 2):
            dep = {"need": need0} if s == 0 else {"after": [f"tr_M1_s{s}"]}
            evj(f"ev_M1_s{s}_best", "gpu", f"M1:{s}:best", M1SETS, probe="v9", **dep)
            evt(f"ev_M1_s{s}_m59", "gpu", f"M1:{s}:best", "m59", **dep)
        # 16:42 UTC: VM a was preempted at ~16:35; its recreate has ~1 h before the hard stop, so the M1-long eval
        # runs the M1 sets first (m33 decides R2) and its m59 set goes to the tail.
        evj("ev_M1long_s0_best", "gpu", "M1long:0:best", M1SETS, probe="v9", after=["tr_M1long_s0"])
        evt("ev_M1long_s0_m59", "gpu", "M1long:0:best", "m59", after=["tr_M1long_s0"])
        evt("ev_M1_s0_shift", "gpu", "M1:0:best", SHIFT, need=need0)
        evt("ev_M1_s0_stab", "gpu", "M1:0:best", "m33[0:20]@30000=m33stab20", need=need0)
        evt("ev_Mref_stretch", "gpu", "M-ref", "m201")
        return J + tail, pools
    if name == "main-b2":
        pools = {"gpu": 2, "cpu": 4}
        mz = [("M4", s, ["--mode", "chunk", "--k", "1"]) for s in (0, 1, 2)]
        mz += [("Mprog", s, ["--mode", "bansal", "--alpha", "1"]) for s in (0, 1, 2)]
        mz += [("M4cm", 0, ["--mode", "chunk", "--k", "1", "--steps", "33000"]), ("M5", 0, ["--mode", "chunk", "--k", "8"])]
        for arm, s, args in mz:
            st = [] if "--steps" in args else ["--steps", "15000"]
            tr(f"tr_{arm}_s{s}", "gpu", ["--name", f"{arm}_s{s}"] + maze + args + st + ["--seed", str(s)])
            evj(f"ev_{arm}_s{s}_best", "gpu", f"{arm}:{s}:best", LADDER, probe="v9", after=[f"tr_{arm}_s{s}"])
            evt(f"ev_{arm}_s{s}_m59", "gpu", f"{arm}:{s}:best", "m59", after=[f"tr_{arm}_s{s}"])
        runs = [("P1", s, ["--mode", "bansal", "--alpha", "1"]) for s in range(8)]
        runs += [("P4", s, ["--mode", "chunk", "--k", "1"]) for s in range(8)]
        runs += [("MRP", s, None) for s in range(3)]
        runs += [("P5", s, ["--mode", "chunk", "--k", "8"]) for s in range(4)]
        runs += [("P4cm", s, ["--mode", "chunk", "--k", "1", "--steps", "22400"]) for s in range(3)]
        runs += [("PL", s, ["--arch", "dtl", "--mode", "bansal", "--alpha", "1"]) for s in range(3)]
        for arm, s, args in runs:
            if arm == "MRP":
                J.append({"id": f"tr_MRP_s{s}", "pool": "cpu", "cmd": [py, "mr_train.py", "--arm", "P", "--T", "24",
                          "--iters", "10000", "--seed", str(s), "--out", os.path.join(ck, f"MRP_s{s}.pt")]})
                model, sets = f"MRfile:{ck}/MRP_s{s}.pt", PREFIX_TEST + "," + LEGACY
            else:
                tr(f"tr_{arm}_s{s}", "cpu", ["--name", f"{arm}_s{s}"] + pre + args + ["--seed", str(s)])
                model, sets = f"{arm}:{s}", PREFIX_TEST
            evj(f"ev_{arm}_s{s}", "cpu", model, sets, probe="v_id", after=[f"tr_{arm}_s{s}"])
            if arm in ("P1", "P4") and s in (0, 1):
                evt(f"ev_{arm}_s{s}_stab", "cpu", model, "fL512[0:20]@30000=fL512stab20", after=[f"tr_{arm}_s{s}"])
        return J + tail, pools
    if name == "pmr":
        pools = {"gpu": 1, "cpu": 3}
        for f in sorted(os.listdir(os.path.join(out, "..", "data", "pmr"))):
            if f.endswith(".pt"):
                evj(f"ev_PMR_{f[:-3]}", "cpu", f"MRfile:{out}/../data/pmr/{f}", PREFIX_TEST + "," + LEGACY, probe="v_id")
        return J, pools
    raise SystemExit(f"unknown plan {name}")


if __name__ == "__main__":
    import sys
    js, ps = plan(sys.argv[1], "/root/out", "python", set(sys.argv[2:]))
    print(json.dumps(ps))
    for j in js:
        print(j["id"], j["pool"], j.get("after", ""), " ".join(j["cmd"][1:])[:160])
