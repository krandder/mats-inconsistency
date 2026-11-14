"""Ready-queue job runner ON the GPU VM (wm2): python3 remote_sched.py JOBFILE PAR STATUS_JSONL
Runs at most PAR jobs at once. A job is runnable when its dependencies exist: for forecaster jobs, every --wm-ckpt path
(or its .failed marker, so the job fails fast). Among runnable jobs the earliest in the file goes first (the job file
lists WMs and long jobs first). One status line per finished job, as remote.sh used to write."""
import json
import os
import socket
import subprocess
import sys
import time

jobfile, par, status = sys.argv[1], int(sys.argv[2]), sys.argv[3]
host = socket.gethostname()
jobs = [l.strip() for l in open(jobfile) if l.strip() and not l.strip().startswith("#")]
t0 = time.time()


def deps(line):
    a = line.split()
    return a[a.index("--wm-ckpt") + 1].split(",") if "--wm-ckpt" in a else []


producer = {}   # checkpoint path -> job number of the WM job that writes it
for k_, l_ in enumerate(jobs, 1):
    a_ = l_.split()
    if "--ckpt" in a_:
        producer[a_[a_.index("--ckpt") + 1]] = k_
        producer[a_[a_.index("--ckpt") + 1] + ".blind.pt"] = k_   # twin WMs also write CKPT.blind.pt


def ready(line):
    """All checkpoints exist, or a checkpoint's WM job failed for good (marker present, job neither pending nor running)."""
    busy = {k for k, _ in pending} | {v[0] for v in running.values()}
    for p in deps(line):
        if os.path.exists(p):
            continue
        base = p[: -len(".blind.pt")] if p.endswith(".blind.pt") else p
        if (os.path.exists(p + ".failed") or os.path.exists(base + ".failed")) and producer.get(p) not in busy:
            continue
        return False
    return True


pending = list(enumerate(jobs, 1))
running = {}
retried = set()
while pending or running:
    for p in list(running):
        if p.poll() is not None:
            k, line, s0, logf = running.pop(p)
            logf.close()
            if p.returncode != 0 and k not in retried and "--wm-ckpt" not in line:
                retried.add(k)            # retry a failed WM / baseline job once (e.g. CUDA OOM while 7 jobs share the L4)
                for c in line.split():
                    if c.endswith(".pt"):
                        try:
                            os.remove(c + ".failed")
                        except OSError:
                            pass
                pending.insert(0, (k, line))
                os.rename(f"/root/logs/job{k}.log", f"/root/logs/job{k}.try1.log")
            try:
                tail = open(f"/root/logs/job{k}.log").read()[-600:]
            except Exception:
                tail = ""
            with open(status, "a") as fh:
                fh.write(json.dumps({"host": host, "job": k, "rc": p.returncode, "sec": int(time.time() - s0), "t": int(time.time() - t0),
                                     "args": line, "tail": tail}) + "\n")
    while len(running) < par:
        cand = next(((k, l) for k, l in pending if ready(l)), None)
        if cand is None:
            break
        pending.remove(cand)
        k, line = cand
        logf = open(f"/root/logs/job{k}.log", "w")
        p = subprocess.Popen(["python3", "wm2/job.py"] + line.split() + ["--threads", "1"], stdout=logf, stderr=subprocess.STDOUT)
        running[p] = (k, line, time.time(), logf)
    time.sleep(5)
with open(status, "a") as fh:
    fh.write(json.dumps({"host": host, "event": "all-done", "t": int(time.time() - t0)}) + "\n")
