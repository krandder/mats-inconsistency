"""side_force.py: side_run.py for jobs moved off the SAME VM's lane (same markers dir). The lane gets <id>.skipped
("moved") first so it never starts a duplicate, and auto-skips the moved jobs' dependents; this runner therefore
ignores .skipped markers and only looks at .done / .failed. A .done written here wins over the lane's .skipped
(run_lane.state checks done first).

Original side_run.py doc:
Run chosen jobs of any plan on this VM next to its own run_lane.py (load balancing between the two VMs).

  sudo systemd-run --unit=p0-side-$(date +%s) --working-directory=/root/job --setenv=P0_CUTS=... \
      /opt/venv/bin/python vm/side_run.py PLAN NSLOTS JOB_ID...

Same markers/logs as run_lane.py (markers/<id>.done|failed|skipped, logs/<id>.log), but no queue.json, JOB_RUNNING
or JOB_DONE, so it never collides with the VM's own lane. 'after' dependencies among the chosen jobs are respected.
The moved jobs are marked skipped ("moved") on the lane they came from.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plan import plan  # noqa: E402

OUT = os.environ.get("P0_OUT", "/root/out")
name, nslots, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
jobs, _ = plan(name, OUT, sys.executable, set(filter(None, os.environ.get("P0_CUTS", "").split(","))))
jobs = [j for j in jobs if j["id"] in ids]
assert len(jobs) == len(ids), f"unknown job ids: {set(ids) - {j['id'] for j in jobs}}"
MK, LG = os.path.join(OUT, "markers"), os.path.join(OUT, "logs")
env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "P0_MR_DIR": os.path.dirname(HERE)}


def state(jid):
    for s in ("done", "failed"):
        if os.path.exists(os.path.join(MK, f"{jid}.{s}")):
            return s
    return "todo"


running, tries = {}, {}
while True:
    for jid, (pr, j, t0, lf) in list(running.items()):
        rc = pr.poll()
        if rc is None:
            continue
        lf.close(); del running[jid]
        info = {"rc": rc, "sec": round(time.time() - t0, 1), "start": round(t0), "end": round(time.time()),
                "try": tries[jid], "where": f"side_force {name}"}
        if rc == 0 or tries[jid] >= j.get("tries", 2):
            open(os.path.join(MK, f"{jid}.{'done' if rc == 0 else 'failed'}"), "w").write(json.dumps(info) + "\n")
    st = {j["id"]: state(j["id"]) for j in jobs}
    for j in jobs:
        jid = j["id"]
        if st[jid] != "todo" or jid in running or len(running) >= nslots:
            continue
        deps = [st.get(d, "done") for d in j.get("after", [])]
        if any(d in ("failed", "skipped") for d in deps):
            open(os.path.join(MK, f"{jid}.skipped"), "w").write(json.dumps({"why": "dependency failed"}) + "\n")
            continue
        if any(d != "done" for d in deps):
            continue
        tries[jid] = tries.get(jid, 0) + 1
        lf = open(os.path.join(LG, jid + ".log"), "a")
        lf.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} side try {tries[jid]}: {' '.join(j['cmd'])}\n")
        lf.flush()
        running[jid] = (subprocess.Popen(j["cmd"], cwd=os.path.dirname(HERE), env=env, stdout=lf,
                                         stderr=subprocess.STDOUT), j, time.time(), lf)
    if not running and all(state(j["id"]) != "todo" for j in jobs):
        break
    time.sleep(5)
print(json.dumps({j["id"]: state(j["id"]) for j in jobs}), flush=True)
