"""Requeue daemon (eurisko smallnet): every 3 min, for every rental whose launch-retry process has exited,
find the jobs of its frozen job file whose result never came home, and launch one new retry rental for them.
A job is requeued at most twice (ledger /tmp/smallnet/requeue-ledger.json); preempted spot VMs are audited.
Run: setsid nohup python3 gpu/requeue.py > /tmp/smallnet/logs/requeue.log 2>&1 &
"""
import glob
import json
import os
import re
import subprocess
import time

S = "/home/agent/workspaces/eurisko-study/li-net/smallnet"
T = "/tmp/smallnet"
LEDGER = f"{T}/requeue-ledger.json"
MAX_RQ_RENTALS = 30


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def running_retry(run):
    out = subprocess.run(["pgrep", "-f", f"launch-retry.sh {run} "], capture_output=True, text=True).stdout.strip()
    return bool(out)


def runs_and_jobfiles():
    """run id -> frozen job file of its LAST launch attempt (from its launch log)."""
    res = {}
    for L in glob.glob(f"{T}/logs/launch-smallnet-*.log"):
        run = os.path.basename(L)[len("launch-"):-len(".log")]
        last = None
        ended = False
        for line in open(L, errors="replace"):
            m = re.search(r" LAUNCH (\S+) .* src=(\S+)", line)
            if m and m.group(1) == run:
                last, ended = m.group(2), False
            if re.search(rf" (END|GIVEUP) {re.escape(run)}( |$)", line):
                ended = True
        if last:
            res[run] = (os.path.join(last, "jobs.txt"), ended)
    return res


def main():
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"jobs": {}, "handled_runs": [], "rq": 0}
    while True:
        try:
            for run, (jf, ended) in sorted(runs_and_jobfiles().items()):
                # ended = its launch-retry process is gone (an END line is not required: launch.sh processes that were
                # running when launch.sh was edited at 17:20Z died before writing END; their VMs were still deleted)
                if run in ledger["handled_runs"] or running_retry(run) or not os.path.exists(jf):
                    continue
                if subprocess.run(["pgrep", "-f", f"launch.sh {run} "], capture_output=True).stdout.strip():
                    continue
                missing = []
                for line in open(jf):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    out = line.split("--out ")[1].split()[0]
                    local = os.path.join(T, "runs", os.path.relpath(out, "/root/runs/smallnet"))
                    if not os.path.exists(local):
                        missing.append((out, line))
                ledger["handled_runs"].append(run)
                if not missing:
                    log("run", run, "complete")
                    continue
                todo = [l for o, l in missing if ledger["jobs"].get(o, 0) < 2]
                skipped = [o for o, l in missing if ledger["jobs"].get(o, 0) >= 2]
                if skipped:
                    log("run", run, "NOT requeued (2 tries used):", " ".join(skipped))
                if not todo:
                    continue
                if ledger["rq"] >= MAX_RQ_RENTALS:
                    log("cap reached; not requeuing", len(todo), "jobs of", run)
                    continue
                ledger["rq"] += 1
                k = ledger["rq"]
                p = f"{S}/jobs/RQ{k}.txt"
                with open(p, "w") as fh:
                    fh.write(f"# requeue {k}: {len(todo)} missing jobs of {run}\n" + "\n".join(todo) + "\n")
                for o, l in missing:
                    if l in todo:
                        ledger["jobs"][o] = ledger["jobs"].get(o, 0) + 1
                subprocess.run([f"{S}/gpu/audit-preempted.sh"], capture_output=True)
                subprocess.run([f"{S}/gpu/audit-failed.sh"], capture_output=True)
                rid = f"smallnet-rq{k}"
                subprocess.Popen(["setsid", "nohup", f"{S}/gpu/launch-retry.sh", rid, "150", p, "6", "30", "2"],
                                 stdout=open(f"{T}/logs/launch-retry-{rid}.out", "w"), stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
                log("run", run, "ended with", len(missing), "missing; requeued", len(todo), "as", rid, p)
            json.dump(ledger, open(LEDGER, "w"), indent=1)
        except Exception as e:  # keep the daemon alive
            log("error", repr(e))
        time.sleep(180)


if __name__ == "__main__":
    main()
