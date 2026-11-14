"""Job queue on the P2 VM: W workers, dependencies, retries, resumable (a task is done when its done-file exists).

  python3 queue_p2.py smoke    unit tests, 20-step train of every arm, tiny evals
  python3 queue_p2.py main     24 training runs (paired seeds) + evals (--deep for decisive arms) + trap test
  python3 queue_p2.py alt      gate fallback: /root/out/alt.json {"alpha": a, "steps": s} for ALL arms

A background sampler writes nvidia-smi utilisation to /root/out/gpu_util.csv every 15 s.
P2_MPS=1 starts the CUDA MPS daemon first (default off: the pre-run bench showed no gain).
P2_ARMS=a,b trains and evaluates only those arms (two VMs split the arms; seeds stay paired).
Evals run in priority order: core cells of every run, then the trap test (P2_TRAP=1), then deep cells
of the decisive arms, then the extra cells; at most P2_EVAL_SLOTS (2) evals at once (eval is
GPU-throughput bound). After P2_DEADLINE (epoch seconds) no new task starts; unstarted tasks are
listed as cut in JOB_DONE.
"""
import json
import os
import subprocess
import sys
import threading
import time

OUT = os.environ.get("P2_OUT", "/root/out")
CODE = os.environ.get("P2_CODE", "/root/job")
PY = os.environ.get("P2_PY", "/opt/venv/bin/python")
WORKERS = int(os.environ.get("P2_WORKERS", "4"))
EVAL_SLOTS = int(os.environ.get("P2_EVAL_SLOTS", "2"))
DEADLINE = float(os.environ.get("P2_DEADLINE", "0") or 0)

SEEDS = {"U-canon": 5, "U-min": 5, "M-tree": 5, "U": 3, "U-any": 3, "M-tile": 3}
DECISIVE = ("M-tree", "U-canon", "U-min")


def tag(arm, seed, alpha=1.0, steps=5000):
    return f"{arm}_s{seed}" + (f"_a{alpha}" if alpha != 1.0 else "") + (f"_x{steps}" if steps != 5000 else "")


def train(arm, seed, runs, extra="", alpha=1.0, steps=5000):
    t = tag(arm, seed, alpha, steps)
    return dict(name=f"train_{t}", deps=[],
                cmd=f"{PY} {CODE}/train.py --arm {arm} --seed {seed} --out {runs} --alpha {alpha} --steps {steps} {extra}",
                done=f"{runs}/{t}.train.json")


def ev(arm, seed, runs, evdir, alpha=1.0, steps=5000, extra="", part="all"):
    t = tag(arm, seed, alpha, steps)
    deep = "--deep" if arm in DECISIVE and "--tmax" not in extra and "--cells" not in extra and part == "all" else ""
    suf = "" if part in ("all", "core") else "__" + part
    return dict(name=f"eval{suf}_{t}", deps=[f"train_{t}"], kind="eval",
                cmd=f"{PY} {CODE}/eval.py run --ckpt {runs}/{t}.pt --arm {arm} --seed {seed} {deep} --part {part} "
                    f"--out {evdir} {extra}",
                done=f"{evdir}/{t}{suf}.json")


def plan(which):
    t = []
    if which == "smoke":
        runs, evd = f"{OUT}/smoke/runs", f"{OUT}/smoke/eval"
        t.append(dict(name="tests", deps=[], cmd=f"{PY} {CODE}/tests.py --trap-data && cp {CODE}/results/unit_tests.json {OUT}/smoke/",
                      done=f"{OUT}/smoke/unit_tests.json"))
        for arm in SEEDS:
            t.append(train(arm, 0, runs, "--ckpt-every 10", steps=20))
        for arm in ("U-canon", "M-tree", "M-tile", "U-min"):
            t.append(ev(arm, 0, runs, evd, steps=20, extra="--tmax 30 --limit 8"))
        return t
    if which == "main":
        alpha, steps = 1.0, 5000
    else:
        p = json.load(open(f"{OUT}/alt.json"))
        alpha, steps = p["alpha"], p["steps"]
    runs, evd = f"{OUT}/runs", f"{OUT}/eval"
    arms = [a for a in SEEDS if a in os.environ.get("P2_ARMS", ",".join(SEEDS)).split(",")]
    # gate arm first, then seed-major so every arm has paired seeds early
    order = [("U-canon", 0)] * ("U-canon" in arms) + [(a, s) for s in range(5) for a in arms
                                                     if s < SEEDS[a] and (a, s) != ("U-canon", 0)]
    for a, s in order:
        t.append(train(a, s, runs, alpha=alpha, steps=steps))
    if "U-canon" in arms:  # gate eval of U-canon s0 goes before the rest of training
        g = ev("U-canon", 0, runs, f"{OUT}/gate", alpha, steps, extra="--cells n9,n19")
        g["name"] = "gate_" + g["name"]
        t.insert(1, g)
    for a, s in order:
        t.append(ev(a, s, runs, evd, alpha, steps, part="core"))
    if which == "main" and os.environ.get("P2_TRAP", "0") == "1":
        t.append(dict(name="trap", deps=[], kind="eval",
                      cmd=f"{PY} {CODE}/eval.py trap && cp {CODE}/results/trap_test.json {OUT}/",
                      done=f"{OUT}/trap_test.json"))
    if os.environ.get("P2_TESTS", "0") == "1":  # final unit-test run of the pushed code (CPU)
        t.append(dict(name="tests", deps=[], kind="eval", done=f"{OUT}/unit_tests_final.json",
                      cmd=f"{PY} {CODE}/tests.py --trap-data && mkdir -p {OUT}/smoke && cp {CODE}/results/unit_tests.json {OUT}/smoke/ "
                          f"&& cp {CODE}/results/unit_tests.json {OUT}/unit_tests_final.json"))
    for a, s in order:
        if a in DECISIVE:
            t.append(ev(a, s, runs, evd, alpha, steps, part="deep"))
    for a, s in sorted(order, key=lambda o: (o[0] not in DECISIVE, o[1])):
        t.append(ev(a, s, runs, evd, alpha, steps, part="extra"))
    return t


def sampler(stop):
    with open(f"{OUT}/gpu_util.csv", "a") as f:
        while not stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
                f.write(f"{time.time():.0f},{r}\n")
                f.flush()
            except Exception:
                pass
            stop.wait(15)


def main():
    which = sys.argv[1]
    tasks = plan(which)
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    for d in ("smoke", "smoke/runs", "smoke/eval", "runs", "eval", "gate"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    open(f"{OUT}/JOB_RUNNING", "w").write(which + "\n")
    for f in ("JOB_DONE", "JOB_FAILED"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    if os.environ.get("P2_MPS", "0") == "1":  # let the 4 small-kernel workers share the GPU concurrently
        r = subprocess.run("nvidia-cuda-mps-control -d", shell=True, capture_output=True, text=True)
        with open(f"{OUT}/logs/queue.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} mps rc={r.returncode} {(r.stdout + r.stderr).strip()[:200]}\n")
    stop = threading.Event()
    threading.Thread(target=sampler, args=(stop,), daemon=True).start()
    byname = {t["name"]: t for t in tasks}
    running, tries, failed = {}, {}, set()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=CODE)
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
                if tries[n] >= 2:
                    failed.add(n)
        done = {t["name"] for t in tasks if os.path.exists(t["done"])}
        pending = [t for t in tasks if t["name"] not in done and t["name"] not in running and t["name"] not in failed]
        blocked = {t["name"] for t in pending if any(d in failed for d in t["deps"])}
        ready = [t for t in pending if t["name"] not in blocked and all(d in done for d in t["deps"])]
        if DEADLINE and time.time() > DEADLINE:
            ready = []
        n_eval = sum(byname[n].get("kind") == "eval" for n in running)
        ready = [t for t in ready if t.get("kind") != "eval" or n_eval < EVAL_SLOTS]
        while ready and len(running) < WORKERS:
            t = ready.pop(0)
            if t.get("kind") == "eval":
                if n_eval >= EVAL_SLOTS:
                    continue
                n_eval += 1
            log = open(f"{OUT}/logs/{t['name']}.log", "a")
            running[t["name"]] = (subprocess.Popen(t["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   cwd=CODE), time.time())
            with open(f"{OUT}/logs/queue.log", "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} start {t['name']}\n")
        cut = sorted(set(t["name"] for t in pending) - blocked) if DEADLINE and time.time() > DEADLINE else []
        json.dump(dict(plan=which, done=sorted(done), running=sorted(running), failed=sorted(failed),
                       blocked=sorted(blocked), cut=cut, total=len(tasks), t=time.time()), open(f"{OUT}/queue_state.json", "w"))
        if not running and not ready:
            break
        time.sleep(5)
    stop.set()
    os.remove(f"{OUT}/JOB_RUNNING")
    open(f"{OUT}/JOB_FAILED" if failed or blocked else f"{OUT}/JOB_DONE", "w").write(
        json.dumps(dict(plan=which, failed=sorted(failed), blocked=sorted(blocked), cut=cut)) + "\n")


if __name__ == "__main__":
    main()
