"""Job queue on the P6 VM: W workers, dependencies, retries, resumable (a task is done when its done-file exists).
  python3 queue_p6.py pilot            1500-it F/EC seed-0 pilot + timing probes -> /root/out/pilot
  python3 queue_p6.py main             all arms, evals and the CPU/GPU numerics check -> /root/out
Settings from /root/out/params.json (width, tau, r1_seeds, single_seeds), written after the pilot.
Two-VM split: P6_ONLY=<regex> runs only the plan's tasks whose name matches (deps are still read from the
full plan, so a dependency's done-file copied in from the other VM counts); P6_TAG names this queue's state
files. A task whose exact command is already running (a queue restarted under live jobs) is adopted, not rerun.
"""
import json, os, re, subprocess, sys, time

OUT = os.environ.get("P6_OUT", "/root/out")
CODE = os.environ.get("P6_CODE", "/root/job")
PY = os.environ.get("P6_PY", "/opt/venv/bin/python")
WORKERS = int(os.environ.get("P6_WORKERS", "4"))
ONLY = os.environ.get("P6_ONLY", "")
TAG = os.environ.get("P6_TAG", "")


def alive(cmd):
    return subprocess.run(["pgrep", "-f", "-x", re.escape(" ".join(cmd.split()))], stdout=subprocess.DEVNULL).returncode == 0


def train(arm, seed, out, extra="", deps=()):
    name = f"{arm}_s{seed}"
    return dict(name=f"train_{name}_{os.path.basename(out)}", deps=list(deps),
                cmd=f"{PY} {CODE}/p6_entry.py --arm {arm} --seed {seed} --outdir {out} {extra}",
                done=f"{out}/train/{name}.json")


def ev(arm, seed, out, variant, extra="", deps=()):
    base = f"{arm}_s{seed}"
    return dict(name=f"eval_{base}_{variant}_{os.path.basename(out)}", deps=list(deps),
                cmd=f"{PY} {CODE}/eval_ladder.py --model {out}/models/{base}.pt --variant {variant} {extra}",
                done=f"{out}/eval/{base}_{variant}.done")


def plan(which):
    t = []
    if which == "pilot":
        o = f"{OUT}/pilot"
        t.append(train("EC", 0, o, "--iters 1500"))
        t.append(train("F", 0, o, "--iters 1500"))
        t.append(train("DTL", 0, o, "--iters 400"))
        t.append(train("R0", 0, o, "--iters 400"))
        t.append(ev("EC", 0, o, "auction", "--Ls 2048 --pi ''", deps=[t[0]["name"]]))
        t.append(ev("DTL", 0, o, "plain", "--Ls 2048 --pi ''", deps=[t[2]["name"]]))
        return t
    p = json.load(open(f"{OUT}/params.json"))
    econ = f"--width {p['width']} --tau {p['tau']}"
    o = OUT
    tr = {}
    for s in range(5):
        tr[("EC", s)] = train("EC", s, o, econ)
    for s in range(5):
        tr[("F", s)] = train("F", s, o, econ)
    for s in range(5):
        tr[("DTL", s)] = train("DTL", s, o)
    for s in range(p["r1_seeds"]):
        tr[("R1", s)] = train("R1", s, o, econ)
    for s in range(3):
        tr[("R0", s)] = train("R0", s, o)
    for s in range(5):
        for arm in ("PBT", "EMA"):
            tr[(arm, s)] = train(arm, s, o, econ, deps=[tr[("EC", s)]["name"]])
    t += list(tr.values())
    for (arm, s), job in tr.items():
        if arm in ("R0", "DTL"):
            t.append(ev(arm, s, o, "plain", deps=[job["name"]]))
        else:
            t.append(ev(arm, s, o, "auction", deps=[job["name"]]))
            if arm != "R1" and s < p["single_seeds"]:
                t.append(ev(arm, s, o, "single", deps=[job["name"]]))
    t.append(dict(name="numerics", deps=[tr[("R0", 0)]["name"], tr[("EC", 0)]["name"]],
                  cmd=f"{PY} {CODE}/numerics_check.py --out {o}", done=f"{o}/numerics.json"))
    return t


def main():
    which = sys.argv[1]
    full = plan(which)
    tasks = [t for t in full if not ONLY or re.search(ONLY, t["name"])]
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    open(f"{OUT}/JOB_RUNNING", "w").write(which + "\n")
    for f in ("JOB_DONE", "JOB_FAILED"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    byname = {t["name"]: t for t in tasks}
    running, tries, failed = {}, {}, set()
    adopted = {t["name"] for t in tasks if alive(t["cmd"])}
    if adopted:
        with open(f"{OUT}/logs/queue{TAG}.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} adopted running {sorted(adopted)}\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONPATH=CODE)
    while True:
        for n, (pr, t0) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            del running[n]
            ok = rc == 0 and os.path.exists(byname[n]["done"])
            with open(f"{OUT}/logs/queue{TAG}.log", "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {'ok' if ok else 'FAIL'} {n} rc={rc} {time.time() - t0:.0f}s\n")
            if not ok:
                tries[n] = tries.get(n, 0) + 1
                if tries[n] >= 2:
                    failed.add(n)
        adopted = {n for n in adopted if alive(byname[n]["cmd"])}
        done = {t["name"] for t in full if os.path.exists(t["done"])}
        pending = [t for t in tasks if t["name"] not in done and t["name"] not in running and t["name"] not in failed
                   and t["name"] not in adopted]
        blocked = {t["name"] for t in pending if any(d in failed for d in t["deps"])}
        ready = [t for t in pending if t["name"] not in blocked and all(d in done for d in t["deps"])]
        while ready and len(running) + len(adopted) < WORKERS:
            t = ready.pop(0)
            log = open(f"{OUT}/logs/{t['name']}.log", "a")
            running[t["name"]] = (subprocess.Popen(t["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   cwd=CODE), time.time())
            with open(f"{OUT}/logs/queue{TAG}.log", "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} start {t['name']}\n")
        json.dump(dict(done=sorted(done & set(byname)), running=sorted(set(running) | adopted), failed=sorted(failed),
                       blocked=sorted(blocked), total=len(tasks), only=ONLY, t=time.time()),
                  open(f"{OUT}/queue_state{TAG}.json", "w"))
        if not running and not ready and not adopted:
            break
        time.sleep(5)
    os.remove(f"{OUT}/JOB_RUNNING")
    open(f"{OUT}/JOB_FAILED" if failed or blocked else f"{OUT}/JOB_DONE", "w").write(
        json.dumps(dict(failed=sorted(failed), blocked=sorted(blocked))) + "\n")


if __name__ == "__main__":
    main()
