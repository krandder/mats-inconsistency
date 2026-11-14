"""Item 2 driver: the single queue line on the VM. Runs the pre-registered jobs (PREREG.md) and checks the gates from
the result files; a failed gate kills the running jobs, writes OUT/GATE_FAILED and exits 3.

  stage 0  m2init (step-0 encoders, no training)                 -> gate G0 (centred init is non-constant at step 0)
  stage 1  gate jobs: original-init cvae + auction, reg2, seeds 0-4 -> G1 (A/A: P10's collapse), G2 (reg2 NLL sane)
           fill jobs (first primary runs) use the VM's idle slots at the same time: they cost no extra VM time
  stage 2  the rest; a stage-2 job never starts before G1 and G2 have passed

Resumable: a job is done when its result JSON exists (train_s6 resumes from its own checkpoint); gates are
re-checked from the files on a restart.  python driver.py WORKERS
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

OUT = os.environ.get("FU_OUT", "/root/out")
CODE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)
RES, LOG = f"{OUT}/results", f"{OUT}/logs/jobs"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SMOKE = os.environ.get("FU_SMOKE") == "1"  # control-flow smoke before the freeze: 1 seed, few steps, val split, gates not enforced
SEEDS = [0] if SMOKE else list(range(10))    # power calculation (PREREG.md)
GATE_SEEDS = [0] if SMOKE else list(range(5))  # P10's seeds
SPLIT = "val" if SMOKE else "test"
MECH = ("cvae", "auction")
CONST_INIT = 0.001  # P10's step-0 test (reproduces its 9/15): constant if min(p1, 1 - p1) <= 0.001
CONST_END = 0.01    # primary endpoint: non-constant if min(p1, 1 - p1) > 0.01 on the test split
ENV = dict(os.environ, P10_TMP=f"{CODE}/data", P10_RES=RES, P10_CKPT=f"{OUT}/ckpt", P10_S6_STEPS=os.environ.get("FU_STEPS", "20000") if SMOKE else "20000",
           P10_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", P10_EVAL_SPLIT=SPLIT, P10_WARM="4000",
           P10_RAMP="2000", P10_CKPT_EVERY=os.environ.get("FU_CKPT_EVERY", "2000") if SMOKE else "2000", P10_FEE="0.005", PYTHONPATH=CODE)
STOP = threading.Event()   # set on a failed gate: running jobs are killed, queued ones never start
GATE_OK = threading.Event()
PROCS, LOCK = set(), threading.Lock()


def job(what, arm, s, after_gate=False):
    tag = "m2init" if what == "m2init" else "m2"
    return {"name": f"{tag}_{arm}_s{s}", "args": [what, arm, "1", str(s)], "out": f"{RES}/{tag}_{arm}_T1_seed{s}.json",
            "after_gate": after_gate}


def run(j, tries=3):
    if os.path.exists(j["out"]):
        return True
    if j["after_gate"]:
        while not (GATE_OK.is_set() or STOP.is_set()):
            time.sleep(5)
    for t in range(tries):
        if STOP.is_set():
            return False
        with open(f"{LOG}/{j['name']}.log", "a") as lf:
            lf.write(f"--- try {t} {time.strftime('%FT%TZ', time.gmtime())}\n"); lf.flush()
            p = subprocess.Popen([PY, f"{CODE}/train.py"] + j["args"], cwd=CODE, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
            with LOCK:
                PROCS.add(p)
            p.wait()
            with LOCK:
                PROCS.discard(p)
        if p.returncode == 0 and os.path.exists(j["out"]):
            return True
    return False


def status(stage, jobs, bad=()):
    done = sum(os.path.exists(j["out"]) for j in jobs)
    st = {"time": time.strftime("%FT%TZ", time.gmtime()), "stage": stage, "jobs": len(jobs), "done": done, "failed": list(bad)}
    json.dump(st, open(f"{OUT}/driver_status.json", "w"), indent=1)
    print(json.dumps(st), flush=True)


def load(j):
    return json.load(open(j["out"]))


def const(p, thr):
    return min(p, 1 - p) <= thr


def fail(gate, why, info):
    info.update({"failed_gate": gate, "why": why, "time": time.strftime("%FT%TZ", time.gmtime())})
    json.dump(info, open(f"{OUT}/GATES.json", "w"), indent=1)
    print("GATE FAILED", gate, why, flush=True)
    if SMOKE and os.environ.get("FU_SMOKE_ENFORCE") != "1":
        print("(smoke: gate not enforced)", flush=True)
        return
    json.dump(info, open(f"{OUT}/GATE_FAILED", "w"), indent=1)
    STOP.set()
    with LOCK:
        for p in PROCS:
            p.kill()
    os._exit(3)


def main():
    for d in (RES, LOG, f"{OUT}/ckpt"):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{OUT}/GATE_FAILED"):
        print("a gate already failed; nothing more to run:", open(f"{OUT}/GATE_FAILED").read(), flush=True)
        sys.exit(3)
    bank = f"{RES}/s6_bank_T1_seed0.json"
    if not os.path.exists(bank):
        with open(f"{CODE}/p10ref/s6_bank_T1_seed0.json") as a, open(bank, "w") as b:
            b.write(a.read())
    gates = {}

    # ---- stage 0 -> G0
    j_orig = [job("m2init", "cvae", s) for s in SEEDS]
    j_init = [job("m2init", "cvae_init", s) for s in SEEDS]
    status("stage0", j_orig + j_init)
    with ThreadPoolExecutor(W) as ex:
        ok = list(ex.map(run, j_orig + j_init))
    bad = [j["name"] for j, o in zip(j_orig + j_init, ok) if not o]
    status("stage0 finished", j_orig + j_init, bad)
    if bad:
        fail("G0", f"step-0 jobs failed: {bad}", gates)
    p_orig = {s: [r["p1"] for r in load(j)[SPLIT]] for s, j in zip(SEEDS, j_orig)}
    p_init = {s: [r["p1"] for r in load(j)[SPLIT]] for s, j in zip(SEEDS, j_init)}
    n_init_nc = sum(not const(p, CONST_INIT) for v in p_init.values() for p in v)
    n_orig_c5 = sum(const(p, CONST_INIT) for s in GATE_SEEDS for p in p_orig[s])
    gates["G0"] = {"centred_init_nonconstant": f"{n_init_nc}/{3 * len(SEEDS)}", "bar": ">= 80%",
                   "orig_constant_seeds0_4": f"{n_orig_c5}/15", "orig_constant_all": sum(const(p, CONST_INIT) for v in p_orig.values() for p in v),
                   "p1_init": p_init, "p1_orig": p_orig}
    if n_init_nc < 0.8 * 3 * len(SEEDS):
        fail("G0", f"centred init non-constant in only {n_init_nc}/{3 * len(SEEDS)} slots at step 0", gates)
    gates["G0"]["pass"] = True
    json.dump(gates, open(f"{OUT}/GATES.json", "w"), indent=1)

    # ---- stages 1 + 2 in one pool, in priority order; stage-2 jobs wait for G1/G2
    j1 = {m: [job("m2", m, s) for s in GATE_SEEDS] for m in MECH}
    jr = [job("m2", "reg2", s) for s in GATE_SEEDS]
    gate_jobs = j1["cvae"] + j1["auction"] + jr
    rest = [job("m2", "cvae_init", s) for s in SEEDS] + [job("m2", "cvae_both", s) for s in SEEDS]
    rest += [job("m2", "cvae_warm", s) for s in SEEDS] + [job("m2", "cvae", s) for s in SEEDS if s not in GATE_SEEDS]
    rest += [job("m2", "auction_both", s) for s in GATE_SEEDS]
    if SMOKE:
        rest = [job("m2", a, 0) for a in ("cvae_init", "cvae_both", "cvae_warm", "auction_both")]
    n_fill = max(0, W - len(gate_jobs))
    fill, stage2 = rest[:n_fill], rest[n_fill:]
    for j in stage2:
        j["after_gate"] = True
    alljobs = gate_jobs + fill + stage2
    status("stage1+fill", alljobs)
    ex = ThreadPoolExecutor(W)
    fut = {j["name"]: ex.submit(run, j) for j in alljobs}
    bad = [j["name"] for j in gate_jobs if not fut[j["name"]].result()]
    status("stage1 finished", alljobs, bad)
    if bad:
        fail("G1/G2", f"gate jobs failed: {bad}", gates)
    g1 = {"orig_init_constant_seeds0_4": n_orig_c5, "bar_init": "7..11 of 15 (P10: 9/15)"}
    ok1 = 7 <= n_orig_c5 <= 11
    for m in MECH:
        nc = sum(not const(r["p1"], CONST_END) for j in j1[m] for r in load(j)["nmi"]["live"])
        g1[f"{m}_nonconstant_end"] = f"{nc}/{3 * len(GATE_SEEDS)}"
        ok1 &= nc <= 1
    g1["bar_end"] = "<= 1 of 15 learned slots non-constant at the end, in each of cvae and auction (P10: 0/15)"
    g1["pass"] = bool(ok1)
    gates["G1"] = g1
    nll = [load(j)["test_nll_per_cell"] for j in jr]
    mean = sum(nll) / len(nll)
    ok2 = all(x == x for x in nll) and 0.35 <= mean <= 0.75 and max(nll) <= 0.90
    gates["G2"] = {"reg2_test_nll": nll, "mean": mean, "bar": "mean in [0.35, 0.75], every seed <= 0.90 (P10 mean 0.549)",
                   "pass": bool(ok2)}
    if not ok1:
        fail("G1", "original-init arm did not reproduce P10's collapse (or its step-0 count)", gates)
    if not ok2:
        fail("G2", f"reg2 test NLL mean {mean:.3f} / max {max(nll):.3f} outside the bar", gates)
    json.dump(gates, open(f"{OUT}/GATES.json", "w"), indent=1)
    GATE_OK.set()
    status("stage2", alljobs)
    ex.shutdown(wait=True)
    bad = [j["name"] for j in fill + stage2 if not fut[j["name"]].result()]
    status("stage2 finished", alljobs, bad)
    gates["stage2_jobs"] = len(fill) + len(stage2)
    gates["stage2_failed_jobs"] = bad
    json.dump(gates, open(f"{OUT}/GATES.json", "w"), indent=1)
    open(f"{OUT}/DRIVER_DONE", "w").write(time.strftime("%FT%TZ\n", time.gmtime()))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
