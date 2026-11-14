"""Resume-safe job queue for one P0 VM lane.

  python vm/run_lane.py PLAN            (PLAN from vm/plan.py; cuts from env P0_CUTS)

Each job has an id, a command, dependencies (jobs and/or files) and a pool.
Pools cap how many jobs run at once (e.g. gpu=2 GPU-bound maze jobs, cpu=4 prefix jobs).
State lives in $P0_OUT (default /root/out): markers/<id>.done|.failed|.skipped, logs/<id>.log,
queue.json; JOB_RUNNING while the queue runs, JOB_DONE when it finishes.
Restarting after a preemption re-runs unfinished jobs (train.py resumes from its .partial).
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from plan import plan  # noqa: E402

OUT = os.environ.get("P0_OUT", "/root/out")
name = sys.argv[1]
jobs, pools = plan(name, OUT, sys.executable, set(filter(None, os.environ.get("P0_CUTS", "").split(","))))
for k, v in os.environ.items():
    if k.startswith("P0_POOL_"):
        pools[k[8:].lower()] = int(v)
MK, LG = os.path.join(OUT, "markers"), os.path.join(OUT, "logs")
os.makedirs(MK, exist_ok=True); os.makedirs(LG, exist_ok=True)
open(os.path.join(OUT, "JOB_RUNNING"), "w").write(f"{name} {time.time():.0f}\n")
for f in ("JOB_DONE",):
    if os.path.exists(os.path.join(OUT, f)):
        os.remove(os.path.join(OUT, f))
env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "P0_MR_DIR": JOB}


def state(j):
    for s in ("done", "failed", "skipped"):
        if os.path.exists(os.path.join(MK, f"{j['id']}.{s}")):
            return s
    return "todo"


def mark(j, s, info):
    open(os.path.join(MK, f"{j['id']}.{s}"), "w").write(json.dumps(info) + "\n")


ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids)), "duplicate job ids"
running = {}   # id -> (Popen, job, t0, logfile)
tries = {}
while True:
    st = {j["id"]: state(j) for j in jobs}
    for jid, (pr, j, t0, lf) in list(running.items()):
        rc = pr.poll()
        if rc is None:
            continue
        lf.close()
        del running[jid]
        info = {"rc": rc, "sec": round(time.time() - t0, 1), "start": round(t0), "end": round(time.time()),
                "try": tries[jid]}
        if rc == 0:
            mark(j, "done", info)
        elif tries[jid] >= j.get("tries", 2):
            mark(j, "failed", info)
        st[jid] = state(j)
    busy = {}
    for jid, (_, j, _, _) in running.items():
        busy[j["pool"]] = busy.get(j["pool"], 0) + 1
    for j in jobs:
        jid = j["id"]
        if st[jid] != "todo" or jid in running:
            continue
        deps = [st.get(d, "done") for d in j.get("after", [])]
        if any(d in ("failed", "skipped") for d in deps):
            mark(j, "skipped", {"why": "dependency failed"})
            continue
        if any(d != "done" for d in deps):
            continue
        if any(st.get(d, "done") == "todo" for d in j.get("wait", [])):   # soft order: wait, never skip
            continue
        missing = [(f, pj) for f, pj in j.get("need", []) if not os.path.exists(f)]
        if missing:  # a file a still-running job writes part-way (e.g. the 15k split of M1-long)
            if any(st.get(pj) != "todo" and pj not in running for _, pj in missing):
                mark(j, "skipped", {"why": "needed file never appeared", "need": missing})
            continue
        if busy.get(j["pool"], 0) >= pools[j["pool"]]:
            continue
        tries[jid] = tries.get(jid, 0) + 1
        lf = open(os.path.join(LG, jid + ".log"), "a")
        lf.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} try {tries[jid]}: {' '.join(j['cmd'])}\n")
        lf.flush()
        running[jid] = (subprocess.Popen(j["cmd"], cwd=JOB, env=env, stdout=lf, stderr=subprocess.STDOUT), j,
                        time.time(), lf)
        busy[j["pool"]] = busy.get(j["pool"], 0) + 1
    st = {j["id"]: state(j) for j in jobs}
    q = {"plan": name, "pools": pools, "t": round(time.time()),
         "running": sorted(running), "counts": {s: sum(v == s for v in st.values()) for s in ("todo", "done", "failed", "skipped")},
         "failed": sorted(k for k, v in st.items() if v in ("failed", "skipped"))}
    json.dump(q, open(os.path.join(OUT, "queue.json.tmp"), "w"))
    os.replace(os.path.join(OUT, "queue.json.tmp"), os.path.join(OUT, "queue.json"))
    if not running and all(v != "todo" for v in st.values()):
        break
    time.sleep(5)
open(os.path.join(OUT, "JOB_DONE"), "w").write(json.dumps(q) + "\n")
os.remove(os.path.join(OUT, "JOB_RUNNING"))
print(json.dumps(q), flush=True)
