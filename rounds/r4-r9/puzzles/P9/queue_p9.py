"""Job queue on the P9 VM: W workers, dependencies, one retry, resumable (a task is done when its done-file exists).
  python3 queue_p9.py PLAN      runs plans/PLAN.json: [{name, cmd, done, deps}], with {PY} {CODE} {OUT} filled in.
State: /root/out/queue_state_PLAN.json, logs/queue_PLAN.log, JOB_RUNNING_PLAN -> JOB_DONE_PLAN / JOB_FAILED_PLAN.
Several plans may run at once (each with its own worker count).
"""
import json, os, subprocess, sys, time

OUT = os.environ.get("P9_OUT", "/root/out")
CODE = os.environ.get("P9_CODE", "/root/job")
PY = os.environ.get("P9_PY", "/opt/venv/bin/python")


def main():
    plan = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("P9_WORKERS", "4"))
    tasks = json.load(open(f"{CODE}/plans/{plan}.json"))
    for t in tasks:
        for k in ("cmd", "done"):
            t[k] = t[k].format(PY=PY, CODE=CODE, OUT=OUT)
        t.setdefault("deps", [])
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    open(f"{OUT}/JOB_RUNNING_{plan}", "w").write(plan + "\n")
    for f in (f"JOB_DONE_{plan}", f"JOB_FAILED_{plan}"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    byname = {t["name"]: t for t in tasks}
    running, tries, failed = {}, {}, set()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=CODE)
    qlog = f"{OUT}/logs/queue_{plan}.log"
    while True:
        for n, (pr, t0) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            del running[n]
            ok = rc == 0 and os.path.exists(byname[n]["done"])
            with open(qlog, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {'ok' if ok else 'FAIL'} {n} rc={rc} {time.time() - t0:.0f}s\n")
            if not ok:
                tries[n] = tries.get(n, 0) + 1
                if tries[n] >= 2:
                    failed.add(n)
        done = {t["name"] for t in tasks if os.path.exists(t["done"])}
        pending = [t for t in tasks if t["name"] not in done and t["name"] not in running and t["name"] not in failed]
        blocked = {t["name"] for t in pending if any(d in failed for d in t["deps"])}
        ready = [t for t in pending if t["name"] not in blocked and all(d in done for d in t["deps"])]
        while ready and len(running) < workers:
            t = ready.pop(0)
            log = open(f"{OUT}/logs/{t['name']}.log", "a")
            running[t["name"]] = (subprocess.Popen(t["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   cwd=CODE), time.time())
            with open(qlog, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} start {t['name']}\n")
        tmp = f"{OUT}/queue_state_{plan}.json.tmp"
        json.dump(dict(done=sorted(done), running=sorted(running), failed=sorted(failed), blocked=sorted(blocked),
                       total=len(tasks), t=time.time()), open(tmp, "w"))
        os.replace(tmp, f"{OUT}/queue_state_{plan}.json")
        if not running and not ready:
            break
        time.sleep(5)
    os.remove(f"{OUT}/JOB_RUNNING_{plan}")
    open(f"{OUT}/JOB_FAILED_{plan}" if failed or blocked else f"{OUT}/JOB_DONE_{plan}", "w").write(
        json.dumps(dict(failed=sorted(failed), blocked=sorted(blocked))) + "\n")


if __name__ == "__main__":
    main()
