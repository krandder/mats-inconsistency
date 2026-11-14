"""Job queue on the P10 VM (copied from puzzles/P6/queue_p6.py): W workers, dependencies, retries, resumable
(a task is done when its done-file exists). Outputs under /root/out (results/ = JSON, tmp/ = npz/pt scratch).
  python3 queue_p10.py smoke            tiny end-to-end run of every entry point (outputs in /root/out/smoke)
  python3 queue_p10.py phase0           S4 seeds 0-2 (gate G1 data) + S6 headroom pilots T=1,2
  python3 queue_p10.py m2 T             atom bank at T (seed 0), then M2 auction/cvae/reg x 5 seeds
  python3 queue_p10.py s9 ...           (added only if G1 passes)
"""
import json, os, subprocess, sys, time

OUT = os.environ.get("P10_OUT", "/root/out")
CODE = os.environ.get("P10_CODE", "/root/job")
PY = os.environ.get("P10_PY", "/opt/venv/bin/python")
WORKERS = int(os.environ.get("P10_WORKERS", "4"))
S6_STEPS = os.environ.get("P10_S6_STEPS_Q", "20000")
S4_STEPS = os.environ.get("P10_S4_STEPS_Q", "10000")


def job(name, args, done, deps=(), env=""):
    return dict(name=name, deps=list(deps), cmd=f"{env} {PY} {CODE}/train.py {args}", done=done)


def plan(which, argv):
    R = f"{OUT}/results"
    t = []
    if which == "smoke":
        S = f"{OUT}/smoke"
        e = (f"P10_RES={S} P10_TMP={OUT}/tmp P10_S4_STEPS=100 P10_HEAD_STEPS=100 P10_S6_STEPS=600 P10_BANK_EVAL=100")
        os.makedirs(S, exist_ok=True)
        t.append(job("smoke_s4", "s4 99", f"{S}/s4_seed99.json", env=e))
        t.append(job("smoke_pilot", "s6pilot 1 0", f"{S}/s6_pilot_T1_seed0.json", env=e))
        t.append(job("smoke_bank", "s6bank 1 0", f"{S}/s6_bank_T1_seed0.json", env=e))
        for arm in ("auction", "cvae", "reg"):
            t.append(job(f"smoke_m2_{arm}", f"m2 {arm} 1 0", f"{S}/m2_{arm}_T1_seed0.json", deps=["smoke_bank"], env=e))
        return t
    if which == "phase0":
        for s in range(3):
            t.append(job(f"s4_s{s}", f"s4 {s}", f"{R}/s4_seed{s}.json", env=f"P10_S4_STEPS={S4_STEPS}"))
        for T in (1, 2):
            t.append(job(f"pilot_T{T}", f"s6pilot {T} 0", f"{R}/s6_pilot_T{T}_seed0.json", env=f"P10_S6_STEPS={S6_STEPS}"))
        return t
    if which == "m2":
        T = int(argv[0])
        e = f"P10_S6_STEPS={S6_STEPS}"
        t.append(job(f"bank_T{T}", f"s6bank {T} 0", f"{R}/s6_bank_T{T}_seed0.json", env=e))
        for s in range(3):  # post-hoc G1 diagnostic: within-state centred head (tag s4c), no joint-NLL readout
            t.append(job(f"s4c_s{s}", f"s4 {s}", f"{R}/s4c_seed{s}.json",
                         env=f"P10_S4_STEPS={S4_STEPS} P10_S4_TAG=c P10_HEAD_CENTER=1"))
        for s in range(5):
            for arm in ("auction", "cvae", "reg"):
                t.append(job(f"m2_{arm}_T{T}_s{s}", f"m2 {arm} {T} {s}", f"{R}/m2_{arm}_T{T}_seed{s}.json",
                             deps=[f"bank_T{T}"], env=e))
        return t
    if which == "m2extra":
        # reg2 = compute-matched control with phase-distinct constants; auction_g5000 = longer grace (post-hoc)
        T = int(argv[0])
        e = f"P10_S6_STEPS={S6_STEPS}"
        for s in range(5):
            for arm in ("reg2", "auction_g5000"):
                t.append(job(f"m2_{arm}_T{T}_s{s}", f"m2 {arm} {T} {s}", f"{R}/m2_{arm}_T{T}_seed{s}.json", env=e))
        return t
    if which == "m2fix":
        # positive control (post-hoc): can the M2 decoder read a slot at all? planted = [y_14 <= 3], cvae arm, 4 seeds
        T = int(argv[0])
        e = f"P10_S6_STEPS={S6_STEPS}"
        for s in range(4):
            t.append(job(f"m2_cvae_fix14_T{T}_s{s}", f"m2 cvae_fix14 {T} {s}", f"{R}/m2_cvae_fix14_T{T}_seed{s}.json", env=e))
        return t
    if which == "m2fixauc":
        # does a readable planted positive survive the auction? grace 5000, planted = [y_14 <= 3], 4 seeds (post-hoc)
        T = int(argv[0])
        e = f"P10_S6_STEPS={S6_STEPS}"
        for s in range(4):
            a = "auction_g5000_fix14"
            t.append(job(f"m2_{a}_T{T}_s{s}", f"m2 {a} {T} {s}", f"{R}/m2_{a}_T{T}_seed{s}.json", env=e))
        return t
    if which == "s9":
        import s9plan
        return s9plan.plan(OUT, CODE, PY, argv)
    raise SystemExit(f"unknown plan {which}")


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
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", P10_THREADS="1", PYTHONPATH=CODE,
               P10_RES=f"{OUT}/results", P10_TMP=f"{OUT}/tmp")
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
