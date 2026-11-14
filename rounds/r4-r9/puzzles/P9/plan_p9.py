"""Write plans/<stage>.json for queue_p9.py.  python3 plan_p9.py s0 | aa | s2a [--seeds 3] | s3 ... (see STAGES)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "--scratch {OUT}/partial"


def tr(arm, seed, steps=4000, extra=""):
    return dict(name=f"train_{arm}_s{seed}", deps=[],
                cmd=f"{{PY}} {{CODE}}/p9b_train.py --arm {arm} --seed {seed} --steps {steps} --out {{OUT}} {SCR} {extra}",
                done=f"{{OUT}}/train/{arm}_s{seed}.json")


def p9a(arm, seed, steps=1500):
    return dict(name=f"p9a_{arm}_s{seed}", deps=["p9a_X1_s0"] if arm != "X1" else [],
                cmd=f"{{PY}} {{CODE}}/p9a.py --arm {arm} --seed {seed} --steps {steps} --out {{OUT}} {SCR} --cache /root/cache",
                done=f"{{OUT}}/p9a/{arm}_s{seed}.json")


def ev(mode, arm, seed, deps=(), extra=""):
    return dict(name=f"eval_{mode}_{arm}_s{seed}", deps=list(deps),
                cmd=f"{{PY}} {{CODE}}/p9b_eval.py {mode} --arm {arm} --seed {seed} --out {{OUT}} --cache /root/cache {extra}",
                done=f"{{OUT}}/eval/{mode}_{arm}_s{seed}.json")


PREP = dict(name="prep", deps=[], cmd="{PY} {CODE}/p9b_eval.py prep --out {OUT} --cache /root/cache",
            done="{OUT}/eval/prep.json")


def stage(name, seeds):
    t = []
    if name == "s0":
        t += [PREP, p9a("X1", 0)] + [tr("N1", s) for s in range(3)] + [tr("N0", 0)]
        for arm, s in [("N1", 0), ("N1", 1), ("N1", 2), ("N0", 0)]:
            t.append(ev("gate", arm, s, deps=["prep", f"train_{arm}_s{s}"]))
        t += [p9a("X0", 0), p9a("X4", 0)]
        for s in range(3):
            t += [p9a("X2", s), p9a("X2p", s), p9a("X3", s)]
    elif name == "retune":
        # gate retune of the binary backbone (pilot N1 collapsed: |pre-activation| ~70, all 8 channels constant)
        V = {"rms": "P9_BIN=rms", "rmsclip": "P9_BIN=rms P9_CLIP=1", "sigclip": "P9_BIN=sig P9_CLIP=1"}
        for v, envs in V.items():
            for s in (0, 1):
                o = "{OUT}/retune/" + v
                t.append(dict(name=f"rt_{v}_s{s}", deps=[],
                              cmd=f"{envs} {{PY}} {{CODE}}/p9b_train.py --arm N1 --seed {s} --steps 4000 --out {o} --scratch {o}/partial"
                                  + (" --lr 1e-3" if v == "sigclip" else ""),
                              done=f"{o}/train/N1_s{s}.json"))
                t.append(dict(name=f"rtev_{v}_s{s}", deps=[f"rt_{v}_s{s}"],
                              cmd=f"{envs} {{PY}} {{CODE}}/p9b_eval.py gate --arm N1 --seed {s} --out {o} --cache /root/cache",
                              done=f"{o}/eval/gate_N1_s{s}.json"))
        for a in ("X2", "X3"):  # P9a budget pilot on the dev set (pre-registered 1.5k-step runs stand as the result)
            t.append(dict(name=f"pilot_{a}", deps=[],
                          cmd=f"{{PY}} {{CODE}}/p9a.py --arm {a} --seed 0 --steps 12000 --dev_every 1500 --tag _pilot "
                              f"--out {{OUT}}/pilot --scratch {{OUT}}/partial --cache /root/cache",
                          done=f"{{OUT}}/pilot/p9a/{a}_s0_pilot.json"))
    elif name == "g2":
        # after the gate retune (rms binary state): the gate's third N1 seed, then everything the gate decided,
        # pipelined inside the VM time box. 5 seeds (sigma rule). Stage-3 evals depend on their train task.
        def trd(a, s):
            return [] if (a, s) in (("N1", 0), ("N1", 1), ("N0", 0)) else [f"train_{a}_s{s}"]
        t += [tr("N1", 2), ev("gate", "N1", 2, deps=["train_N1_s2"])]
        t += [ev("imp", "N1", 0)]
        for a in ("X2", "X3"):
            t.append(dict(name=f"pilot_{a}", deps=[],
                          cmd=f"{{PY}} {{CODE}}/p9a.py --arm {a} --seed 0 --steps 12000 --dev_every 1500 --tag _pilot "
                              f"--out {{OUT}}/pilot --scratch {{OUT}}/partial --cache /root/cache",
                          done=f"{{OUT}}/pilot/p9a/{a}_s0_pilot.json"))
        t += [tr("N2", 0), tr("N4p", 0)]
        runs = [("N1", 3), ("N1", 4), ("N0", 1), ("N0", 2)]
        for s_ in range(seeds):
            runs += [("N1c", s_), ("N4a", s_), ("N3", s_)]
        t += [tr(a, s_) for a, s_ in runs]
        allnets = [("N2", 0), ("N4p", 0), ("N0", 0), ("N1", 0), ("N1", 1), ("N1", 2)] + runs
        for a, s_ in allnets:
            if not (a in ("N1",) and s_ < 3):
                t.append(ev("gate", a, s_, deps=trd(a, s_)))
            t.append(ev("test", a, s_, deps=trd(a, s_)))
        for a, s_ in [("N1", 1), ("N1", 2), ("N4a", 0), ("N4a", 1), ("N4a", 2)]:
            t.append(ev("imp", a, s_, deps=trd(a, s_)))
    elif name == "gate":
        for arm, s in [("N1", 0), ("N1", 1), ("N1", 2), ("N0", 0)]:
            t.append(ev("gate", arm, s))
    elif name == "aa":
        t += [tr("N2", 0), tr("N4p", 0)]
        t += [ev("gate", a, 0, deps=[f"train_{a}_s0"]) for a in ("N2", "N4p")]
    elif name == "s2a":
        for s in range(seeds):
            for a in ("N1c", "N4a", "N3"):
                t.append(tr(a, s))
                t.append(ev("gate", a, s, deps=[f"train_{a}_s{s}"]))
        for s in range(3, seeds):
            t.append(tr("N1", s))
            t.append(ev("gate", "N1", s, deps=[f"train_N1_s{s}"]))
    elif name == "s2b":
        for s in range(seeds):
            t.append(tr("N3b", s))
            t.append(ev("gate", "N3b", s, deps=[f"train_N3b_s{s}"]))
    elif name == "s3":
        arms = sys.argv[3].split(",")  # e.g. N1:0,N1:1,N4a:0,...
        for a in arms:
            arm, s = a.split(":")
            t.append(ev("test", arm, int(s)))
        for a in ("N1", "N4a"):
            for s in range(min(seeds, 3)):
                t.append(ev("imp", a, s))
    elif name == "aa2":
        # A/A debug (the 1-seed A/A failed: N4' diverged at step ~3100 and never recovered). Rerun N4' seed 0
        # (fresh scratch, same seed) and add a second pair (N2, N4') at seed 1; same recipe, separate out dir.
        o = "{OUT}/aa2"
        for a, s_ in (("N4p", 0), ("N2", 1), ("N4p", 1)):
            t.append(dict(name=f"aa2_train_{a}_s{s_}", deps=[],
                          cmd=f"{{PY}} {{CODE}}/p9b_train.py --arm {a} --seed {s_} --steps 4000 --out {o} --scratch {o}/partial",
                          done=f"{o}/train/{a}_s{s_}.json"))
            for mode in ("gate", "test"):
                t.append(dict(name=f"aa2_eval_{mode}_{a}_s{s_}", deps=[f"aa2_train_{a}_s{s_}"],
                              cmd=f"{{PY}} {{CODE}}/p9b_eval.py {mode} --arm {a} --seed {s_} --out {o} --cache /root/cache",
                              done=f"{o}/eval/{mode}_{a}_s{s_}.json"))
    elif name == "aa3":
        # A/A debug, step 2: aa2 failed on cell accuracy with opposite signs at seeds 0 and 1, and the same-seed
        # rerun of N4' differed from the original by more than the 0.02 bar. Two more (N2, N4') pairs, seeds 2-3,
        # so the A/A difference can be read as a paired mean over 3 fresh seeds (declared deviation).
        o = "{OUT}/aa2"
        for s_ in (2, 3):
            for a in ("N2", "N4p"):
                t.append(dict(name=f"aa3_train_{a}_s{s_}", deps=[],
                              cmd=f"{{PY}} {{CODE}}/p9b_train.py --arm {a} --seed {s_} --steps 4000 --out {o} --scratch {o}/partial",
                              done=f"{o}/train/{a}_s{s_}.json"))
                for mode in ("gate", "test"):
                    t.append(dict(name=f"aa3_eval_{mode}_{a}_s{s_}", deps=[f"aa3_train_{a}_s{s_}"],
                                  cmd=f"{{PY}} {{CODE}}/p9b_eval.py {mode} --arm {a} --seed {s_} --out {o} --cache /root/cache",
                                  done=f"{o}/eval/{mode}_{a}_s{s_}.json"))
    elif name == "p9ar":
        # P9a rerun at the recipe picked in the 9x9 pilot (declared deviation; the 1.5k-step runs stay reported).
        lr, clip, steps = sys.argv[3].split(",")  # e.g. 1e-3,1,6000
        o = "{OUT}/rt"
        runs = [("X0", 0), ("X4", 0)] + [(a, s_) for s_ in range(seeds) for a in ("X2", "X2p", "X3")]
        for a, s_ in runs:
            t.append(dict(name=f"p9ar_{a}_s{s_}", deps=[],
                          cmd=f"{{PY}} {{CODE}}/p9a.py --arm {a} --seed {s_} --steps {steps} --lr {lr} --clip {clip} "
                              f"--out {o} --scratch {o}/partial --cache /root/cache",
                          done=f"{o}/p9a/{a}_s{s_}.json"))
    else:
        raise SystemExit(f"unknown stage {name}")
    return t


if __name__ == "__main__":
    name = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(os.path.join(HERE, "plans"), exist_ok=True)
    tasks = stage(name, seeds)
    json.dump(tasks, open(os.path.join(HERE, "plans", f"{name}.json"), "w"), indent=1)
    print(name, len(tasks), "tasks")
