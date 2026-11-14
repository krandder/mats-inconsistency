"""Job queue on the P8 VM (copied from puzzles/P10/queue_p10.py): W workers, dependencies, retries, resumable
(a task is done when its result JSON exists). Outputs under /root/out (results/ = JSON).
  python3 queue_p8.py smoke     every job once at seed 0, tiny sizes (P8_SMOKE=1), outputs in /root/out/smoke
  python3 queue_p8.py all       Phase 0, Phase A (3 seeds), Phase B (T1-oracle x3, C2 x5 behind the seed-0 gate, BF x3)
Gates: parity jobs wait for pcalib (it fixes the input width); C2 seeds 1-4 wait for c2gate (F3 at seed 0 blocks them).
"""
import json, os, subprocess, sys, time

OUT = os.environ.get("P8_OUT", "/root/out")
CODE = os.environ.get("P8_CODE", "/root/job")
PY = os.environ.get("P8_PY", "/opt/venv/bin/python")
WORKERS = int(os.environ.get("P8_WORKERS", "8"))

GATE = r'''
import json, sys
d = json.load(open(sys.argv[1]))["deployed"]
b2 = d["B2"]["held"]["acc"]
t1, t7 = d["T1"]["held"]["acc"], d["T7"]["held"]["acc"]
f3 = (t1 <= b2 - 0.03) and (t7 <= b2 - 0.03)
r = json.dumps(dict(T1=t1, T7=t7, B2=b2, F3_fired=f3))
print(r)
open(sys.argv[2].replace(".json", "_result.json"), "w").write(r)
if not f3:
    open(sys.argv[2], "w").write(r)
sys.exit(3 if f3 else 0)
'''


NOISEGATE = r'''
import json, sys, math
import numpy as np
R, K = sys.argv[1], int(sys.argv[2])
runs = []
for o in range(4):
    b = json.load(open(f"{R}/noise_maze_s0_o{o}.json"))["bits"]
    runs.append(np.array(b["L(Y|X)"]) - np.array(b["L(Y|X,g2)"]) - np.array(b["L(g2|X)"]))
A = np.array(runs)  # (orders, inits): each run has its own (init, order), paired within the run
sav = A.mean()
sd1 = A.std(ddof=1)
sdK = sd1 / math.sqrt(K)
# direct check of iid scaling: 8 Latin-square groups of 4 runs, each with 4 distinct inits and 4 distinct orders
sd4_latin = float(np.std([np.mean([A[o, (g - o) % A.shape[1]] for o in range(4)]) for g in range(A.shape[1])], ddof=1))
sdK_direct = sd4_latin * math.sqrt(4 / K)
r = dict(saving_g2=sav, sd_single=sd1, K=K, sd_K=sdK, sd_K_direct=sdK_direct, ratio_single=3 * sd1 / sav,
         ratio_K=3 * sdK / sav, ratio_K_direct=3 * sdK_direct / sav, fired_single=bool(3 * sd1 > 0.1 * sav),
         sd4_latin=sd4_latin, fired_K=bool(3 * max(sdK, sdK_direct) > 0.1 * sav),
         fired_K_direct=bool(3 * sdK_direct > 0.1 * sav))
print(json.dumps(r))
open(f"{R}/noisegate_result.json", "w").write(json.dumps(r))
if not r["fired_K"]:
    open(f"{R}/noisegate.json", "w").write(json.dumps(r))
sys.exit(3 if r["fired_K"] else 0)
'''


def job(name, j, seed, R, deps=(), env=""):
    out = {"pcalib": f"pcalib_s{seed}", "sched_maze": f"sched_maze_s{seed}", "u6_maze": f"u6_maze_s{seed}",
           "bf_par": f"bf_par_s{seed}", "c2": f"c2_s{seed}", "toracle": f"toracle_s{seed}"}.get(j)
    if out is None and j.startswith("noise_"):
        out = f"noise_{j[6:-1]}_s{seed}_o{j[-1]}"
    if out is None:
        out = f"{j}_s{seed}"
        if j in ("r6b_par",):
            out = f"r6b_par_s{seed}"
    return dict(name=f"{j}_s{seed}", deps=list(deps), cmd=f"{env} {PY} {CODE}/jobs.py {j} {seed}",
                done=f"{R}/{out}.json")


def plan(which, argv):
    t = []
    if which == "smoke":
        R = f"{OUT}/smoke"
        os.makedirs(R, exist_ok=True)
        e = f"P8_SMOKE=1 P8_RES={R} P8_NBITS=16"
        for j in ("pcalib", "noise_par0", "price_maze", "price_par", "r6_par", "r6b_par", "truth_maze", "truth_par", "c2", "toracle",
                  "bf_par", "r6_maze", "capref_maze", "r7b_maze", "noise_maze0", "sched_maze", "u6_maze"):
            t.append(job(j, j, 0, R, env=e))
        return t
    R = f"{OUT}/results"
    NG = ["noisegate"]
    P = ["pcalib_s0", "noisegate"]
    # order = priority (longest / gating first). Phase 0 first; everything else waits for the noise gate.
    t.append(job("pcalib", "pcalib", 0, R))
    for j in ("noise_maze0", "noise_maze1", "noise_maze2", "noise_maze3", "sched_maze"):
        t.append(job(j, j, 0, R))
    t.append(dict(name="noisegate", deps=[f"noise_maze{o}_s0" for o in range(4)], done=f"{R}/noisegate.json",
                  cmd=f"{PY} -c '{NOISEGATE}' {R} {os.environ.get('P8_KREP', '12')}"))
    t.append(job("c2", "c2", 0, R, deps=NG))
    t.append(dict(name="c2gate", deps=["c2_s0"], done=f"{R}/c2gate.json",
                  cmd=f"{PY} -c '{GATE}' {R}/c2_s0.json {R}/c2gate.json"))
    for s in range(1, 5):
        t.append(job("c2", "c2", s, R, deps=["c2gate"]))
    for j in ("noise_par0", "noise_par1", "noise_par2", "noise_par3"):
        t.append(job(j, j, 0, R, deps=["pcalib_s0"]))
    # longest jobs first (price jobs carry KREP replicates)
    for j in ("price_maze", "toracle", "price_par", "r7b_maze", "r7b_par", "truth_maze", "r6_maze", "capref_maze",
              "truth_par", "r6_par", "r6b_par", "capref_par", "bf_par"):
        for s in range(3):
            t.append(job(j, j, s, R, deps=NG if j.endswith("_maze") or j == "toracle" else P))
    t.append(job("u6_maze", "u6_maze", 0, R, deps=NG))
    return t


def main():
    which = sys.argv[1]
    tasks = plan(which, sys.argv[2:])
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    open(f"{OUT}/JOB_RUNNING", "w").write(which + "\n")
    for f in ("JOB_DONE", "JOB_FAILED"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    byname = {t["name"]: t for t in tasks}
    running, tries, failed = {}, {}, set()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", P8_THREADS="1", PYTHONPATH=CODE,
               P8_RES=f"{OUT}/results")
    while True:
        for n, (pr, t0) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            del running[n]
            ok = rc == 0 and os.path.exists(byname[n]["done"])
            with open(f"{OUT}/logs/queue.log", "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {'ok' if ok else 'FAIL'} {n} rc={rc} {time.time() - t0:.0f}s\n")
            if not ok:
                tries[n] = tries.get(n, 0) + 1
                if tries[n] >= 2 or n in ("c2gate", "noisegate"):
                    failed.add(n)
        done = {t["name"] for t in tasks if os.path.exists(t["done"])}
        pending = [t for t in tasks if t["name"] not in done and t["name"] not in running and t["name"] not in failed]
        blocked = {t["name"] for t in pending if any(d in failed for d in t["deps"])}
        ready = [t for t in pending if t["name"] not in blocked and all(d in done for d in t["deps"])]
        if os.path.exists(f"{OUT}/STOP"):
            ready = []
        while ready and len(running) < WORKERS:
            t = ready.pop(0)
            log = open(f"{OUT}/logs/{t['name']}.log", "a")
            running[t["name"]] = (subprocess.Popen(t["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   cwd=CODE), time.time())
            with open(f"{OUT}/logs/queue.log", "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} start {t['name']}\n")
        json.dump(dict(done=sorted(done), running=sorted(running), failed=sorted(failed), blocked=sorted(blocked),
                       total=len(tasks), t=time.time()), open(f"{OUT}/queue_state.json", "w"))
        if not running and not ready:
            break
        time.sleep(5)
    os.remove(f"{OUT}/JOB_RUNNING")
    open(f"{OUT}/JOB_FAILED" if failed or blocked else f"{OUT}/JOB_DONE", "w").write(
        json.dumps(dict(which=which, failed=sorted(failed), blocked=sorted(blocked))) + "\n")


if __name__ == "__main__":
    main()
