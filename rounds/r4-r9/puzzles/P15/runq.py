"""Run a P15 queue with NPAR parallel workers (one subprocess per job, resume-safe).
   python runq.py queue.json NPAR        -> runs every job whose results/<id>.json is missing
   python runq.py --one queue.json ID     -> runs one job in this process (used by the workers)
A job with arm "P-eq" goes to p15_dtrecall.peq_job, everything else to p15.run_job.
A job may carry "TEST" (overrides p15.TEST, smoke only) and "threads"."""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def one(qf, jid):
    import torch
    import p15
    q = {j["id"]: j for j in json.load(open(qf))}
    job = q[jid]
    torch.set_num_threads(job.get("threads", 2))
    if "TEST" in job:
        p15.TEST = job["TEST"]
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    if job["arm"] == "P-eq":
        import p15_dtrecall
        p15_dtrecall.peq_job(job, log=log)
    else:
        p15.run_job(job, log=log)


def main(qf, npar):
    import p15
    jobs = json.load(open(qf))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(p15.RESDIR, j["id"] + ".json"))]
    lock = threading.Lock()
    failed = []

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                j = todo.pop(0)
            lf = os.path.join(HERE, "logs", j["id"] + ".log")
            t = time.time()
            print(time.strftime("%H:%M:%S"), "start", j["id"], flush=True)
            with open(lf, "a") as f:
                rc = subprocess.call([sys.executable, __file__, "--one", qf, j["id"]], stdout=f, stderr=subprocess.STDOUT)
            ok = rc == 0 and os.path.exists(os.path.join(p15.RESDIR, j["id"] + ".json"))
            print(time.strftime("%H:%M:%S"), "done" if ok else f"FAILED rc={rc}", j["id"], f"{time.time() - t:.0f}s",
                  flush=True)
            if not ok:
                failed.append(j["id"])

    th = [threading.Thread(target=worker) for _ in range(npar)]
    for x in th:
        x.start()
        time.sleep(2)
    for x in th:
        x.join()
    print("queue", qf, "finished; failed:", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2))
