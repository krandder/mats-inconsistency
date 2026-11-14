"""Item 3 driver on the VM (adapted from P8 queue_p8.py): W workers, dependencies, one retry, resumable (a task is done
when its result JSON exists), and the pre-registered gates G1-G3 (PREREG.md). A failed gate writes GATE_FAILED.json,
kills every running job and exits 3; nothing more is launched.
  python driver.py SEEDS KREP KREP_R7B WORKERS DEADLINE_EPOCH
At DEADLINE_EPOCH (UTC seconds) the driver kills every running job, writes DEADLINE.json and exits 5 (PREREG contingency).
Env for the jobs: P8_N=4096 P8_TRUTH_TAG=maze-truth4k P8_KREP=KREP P8_RES=$OUT/results."""
import json, math, os, signal, subprocess, sys, time

from scipy.stats import spearmanr

OUT = os.environ.get("P8_OUT", "/root/out")
CODE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)
R = f"{OUT}/results"
POOL = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "m2", "m4", "m6", "m8", "nz05", "nz20", "d2", "d4", "d6", "reach"]
ITEMS = [f"{c}:{z}" for c, S in {"empty": [], "g2g4": ["g2", "g4"]}.items() for z in POOL if z not in S]
G1_MIN_ACC, G2_MIN_RHO, G3_NOISE_MAX = 0.90, 0.80, 0.10
F_MAZE, CELLS = 8 * 121, 121
SMOKE_PASS = os.environ.get("P8_SMOKE") == "1" and os.environ.get("P8_SMOKE_GATES") == "record"  # driver test only


def load(n):
    return json.load(open(f"{R}/{n}.json"))


def gate_g1(s):
    acc = load(f"b2_s{s}")["B2"]["held"]["acc"]
    return dict(acc=acc, bar=G1_MIN_ACC, ok=bool(acc >= G1_MIN_ACC))


def gate_g2(s):
    a, b = load(f"truth_maze_s{s}_r0")["items"], load(f"truth_maze_s{s}_r1")["items"]
    x = [a[i]["held"]["gain_vs_base"] for i in ITEMS]
    y = [b[i]["held"]["gain_vs_base"] for i in ITEMS]
    fin = all(v is not None and math.isfinite(v) for v in x + y)
    rho = float(spearmanr(x, y).statistic) if fin else float("nan")
    return dict(rho=rho, bar=G2_MIN_RHO, ok=bool(fin and rho >= G2_MIN_RHO))


def gate_g3(s, krep):
    p = load(f"price_maze_s{s}")
    pr, N = p["prices"], p["N"]
    fin = all(all(pr[i][k] is not None and math.isfinite(pr[i][k]) for k in ("R1", "R4", "L_Z")) for i in ITEMS)
    g2 = pr["empty:g2"]
    ratio = 3 * g2["R4_sd"] / math.sqrt(krep) / g2["R4"] if g2["R4"] > 0 else float("inf")
    ub = N * CELLS
    san = dict(U2_random=bool(abs(pr["empty:random"]["R4"] + ub) <= 0.05 * ub),
               U3_null_R1_zero=bool(abs(pr["empty:null"]["R1"]) <= 0.01))
    info = dict(U1_copyY_R4=pr["empty:copyY"]["R4"], U1_band=[-2 * F_MAZE, 0])  # reported, not gated
    ok = bool(fin and ratio <= G3_NOISE_MAX and all(san.values()))
    return dict(finite=fin, noise_ratio=ratio, noise_bar=G3_NOISE_MAX, sanity=san, info=info, ok=ok)


def main():
    seeds = list(range(int(sys.argv[1])))
    krep, krep7, W, deadline = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    os.makedirs(R, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", P8_THREADS="1", PYTHONPATH=CODE, P8_RES=R,
               P8_N="4096", P8_TRUTH_TAG="maze-truth4k", P8_KREP=str(krep))

    def job(name, args, deps=()):
        return dict(name=name, deps=list(deps), done=f"{R}/{name}.json", cmd=f"{PY} {CODE}/jobs3.py {args}")

    t = []
    for s in seeds:  # gate inputs first
        t.append(job(f"b2_s{s}", f"b2gate {s}"))
    for s in seeds:
        for r in (0, 1):
            t.append(job(f"truth_maze_s{s}_r{r}", f"truthrep {s} {r}"))
    for s in seeds:  # primary prices, seed-major
        for r in range(krep):
            t.append(job(f"pricerep_maze_s{s}_r{r}", f"pricerep {s} {r}"))
        t.append(job(f"price_maze_s{s}", f"pricemerge {s}", [f"pricerep_maze_s{s}_r{r}" for r in range(krep)]))
    for s in seeds:  # R7b (secondary) last
        for r in range(krep7):
            t.append(dict(job(f"r7brep_maze_s{s}_r{r}", f"r7brep {s} {r}")))
        t.append(dict(job(f"r7b_maze_s{s}", f"r7bmerge {s}", [f"r7brep_maze_s{s}_r{r}" for r in range(krep7)]),
                      env7=True))
    for x in t:
        if x.get("env7"):  # r7bmerge averages over KREP_R7B replicates
            x["cmd"] = f"P8_KREP={krep7} " + x["cmd"]
    gates = []  # every gate is checked per seed as soon as its inputs exist
    for s in seeds:
        gates += [(f"G1:s{s}", [f"b2_s{s}"], lambda s=s: gate_g1(s)),
                  (f"G2:s{s}", [f"truth_maze_s{s}_r{r}" for r in (0, 1)], lambda s=s: gate_g2(s)),
                  (f"G3:s{s}", [f"price_maze_s{s}"], lambda s=s: gate_g3(s, krep))]
    gres = json.load(open(f"{OUT}/GATES.json")) if os.path.exists(f"{OUT}/GATES.json") else {}
    byname = {x["name"]: x for x in t}
    running, tries, failed = {}, {}, set()

    def qlog(msg):
        with open(f"{OUT}/logs/queue.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def stop_all(reason, marker="GATE_FAILED", code=3):
        for n, (pr, _) in running.items():
            try:
                os.killpg(pr.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        json.dump(dict(reason=reason, gates=gres, running=sorted(running), t=time.time()),
                  open(f"{OUT}/{marker}.json", "w"), indent=1)
        qlog(f"STOP {reason}")
        sys.exit(code)

    if os.path.exists(f"{OUT}/GATE_FAILED.json"):
        qlog("GATE_FAILED.json present: not running")
        sys.exit(3)
    while True:
        for n, (pr, t0) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            del running[n]
            ok = rc == 0 and os.path.exists(byname[n]["done"])
            qlog(f"{'ok' if ok else 'FAIL'} {n} rc={rc} {time.time() - t0:.0f}s")
            if not ok:
                tries[n] = tries.get(n, 0) + 1
                if tries[n] >= 2:
                    failed.add(n)
        if time.time() >= deadline:
            stop_all("deadline reached", "DEADLINE", 5)
        if failed:
            gres["G3a_completeness"] = dict(failed=sorted(failed), ok=False)
            stop_all(f"G3a: jobs failed twice: {sorted(failed)}")
        done = {x["name"] for x in t if os.path.exists(x["done"])}
        for g, deps, fn in gates:
            if g not in gres and all(d in done for d in deps):
                gres[g] = fn()
                json.dump(gres, open(f"{OUT}/GATES.json", "w"), indent=1)
                qlog(f"gate {g} {'PASS' if gres[g]['ok'] else 'FAIL'} {json.dumps(gres[g])}")
                if not gres[g]["ok"] and not SMOKE_PASS:
                    stop_all(f"{g} failed")
        pending = [x for x in t if x["name"] not in done and x["name"] not in running]
        ready = [x for x in pending if all(d in done for d in x["deps"])]
        if os.path.exists(f"{OUT}/STOP"):
            ready = []
        while ready and len(running) < W:
            x = ready.pop(0)
            log = open(f"{OUT}/logs/{x['name']}.log", "a")
            running[x["name"]] = (subprocess.Popen(x["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   cwd=CODE, start_new_session=True), time.time())
            qlog(f"start {x['name']}")
        json.dump(dict(done=len(done), running=sorted(running), total=len(t), gates={k: v["ok"] for k, v in gres.items()},
                       t=time.time()), open(f"{OUT}/queue_state.json", "w"))
        if not running and not ready:
            break
        time.sleep(5)
    if all(g in gres and gres[g]["ok"] for g, _, _ in gates):
        json.dump(gres, open(f"{OUT}/GATES_PASSED.json", "w"), indent=1)
        qlog("ALL DONE, gates passed")
        return
    qlog("ended without all gates evaluated")
    sys.exit(4)


if __name__ == "__main__":
    main()
