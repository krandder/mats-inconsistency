"""P11 job queue on one VM: W workers, dependencies, 2 tries, resumable (a task is done when its done-file exists).
  python3 queue_p11.py all      S0 -> timing probe -> S1 pilot (train + eval) -> select.py (gates, params.json)
                                -> S2 core -> S3 secondary
Sharding over several VMs (P11_SHARD = this VM's index, P11_NSHARD = number of VMs in the pilot; OUT/shards.json
{"core": n} sets the number for core + secondary). Each VM runs only its own train/eval tasks; the others are
"external": done when their done-file appears in OUT, which `gpu.sh relay` copies between VMs. select, the DT-L
grid and headroom run on shard 0 only.
DT-L pilot: a x 2 (the R-prog pilot winner and its better neighbour), decided by `select_p11.py OUT dtl` once the
R-prog pilot evals exist.
Stages after the pilot read OUT/params.json (written by select.py). If select.py reports a failed gate the
core stage is empty and the queue ends (handled by hand).
A restarted queue adopts train/eval processes still running from an earlier queue instead of starting twins.
"""
import json
import os
import subprocess
import sys
import time

OUT = os.environ.get("P11_OUT", "/root/out")
CODE = os.environ.get("P11_CODE", "/root/job")
PY = os.environ.get("P11_PY", "/opt/venv/bin/python")
WORKERS = int(os.environ.get("P11_WORKERS", "4"))
POOL = os.environ.get("P11_POOL", f"{CODE}/data/pool9.npz")
SHARD = int(os.environ.get("P11_SHARD", "0"))
NSHARD = int(os.environ.get("P11_NSHARD", "1"))
STALL_MIN = float(os.environ.get("P11_STALL_MIN", "120"))
TICK = float(os.environ.get("P11_TICK", "5"))
DATA = f"{CODE}/data/sets"

PILOT = ([("R-prog", {"a": a}) for a in (0.01, 0.1, 0.5, 1.0)]
         + [("M-budget", {"W": w}) for w in (3, 6, 12, 24)]
         + [("M-TV", {"V": v}) for v in (4, 8, 16, 32)]
         + [("M-B", {}), ("R-final", {})]
         + [("M-cap", {"eps": e}) for e in (0.1, 0.3, 1.0)]
         + [("M-clip", {"c": c}) for c in (0.1, 0.3, 1.0, 3.0)])
CORE = ("R-prog", "DT-L", "M-B", "M-budget", "M-TV")
SECONDARY = ("R-final", "M-cap", "M-clip")
PILOT_SHARD1 = ("M-budget", "M-TV", "M-cap")   # with 2 pilot shards; everything else (and DT-L) on shard 0
TAG = f"-s{SHARD}"


def hpkey(hp):
    return "_".join(f"{k}{v}" for k, v in sorted(hp.items())) or "none"


def run_dir(stage, arm, hp, seed):
    return f"{OUT}/{stage}/{arm}_{hpkey(hp)}_s{seed}"


def train(stage, arm, hp, seed, steps, width, deps=(), mine=True):
    d = run_dir(stage, arm, hp, seed)
    hj = json.dumps(hp).replace('"', '\\"')
    return dict(name=f"train_{stage}_{os.path.basename(d)}", deps=list(deps), mine=mine, arm=arm,
                cmd=f'{PY} {CODE}/train.py --arm {arm} --hp "{hj}" --seed {seed} --steps {steps} --width {width} '
                    f'--pool {POOL} --out {d} --tag {stage}',
                sig=f"--out {d} --tag", done=f"{d}/train.json")


def ev(stage, arm, hp, seed, plan, deps=(), mine=True):
    d = run_dir(stage, arm, hp, seed)
    return dict(name=f"eval_{plan}_{stage}_{os.path.basename(d)}", deps=list(deps), mine=mine, arm=arm,
                cmd=f"{PY} {CODE}/eval.py --run {d} --plan {plan} --data {DATA}",
                sig=f"--run {d} --plan {plan} ", done=f"{d}/eval_{plan}.json")


def pilot_shard(arm):
    return 0 if NSHARD == 1 else (1 if arm in PILOT_SHARD1 else 0)


def plan_all():
    t = [dict(name="s0", deps=[], mine=SHARD == 0, cmd=f"{PY} {CODE}/s0.py {OUT}/s0", done=f"{OUT}/s0/s0.json")]
    for arm, hp in (("R-prog", {"a": 0.5}), ("DT-L", {"a": 0.5}), ("M-budget", {"W": 6}), ("M-TV", {"V": 8})):
        t.append(train("timing", arm, hp, 0, 200, 32, mine=SHARD == 0))
    for arm, hp in PILOT:
        m = pilot_shard(arm) == SHARD
        j = train("pilot", arm, hp, 0, 6000, 32, deps=["s0"], mine=m)
        t.append(j)
        t.append(ev("pilot", arm, hp, 0, "pilot", deps=[j["name"]], mine=m))
    rp = [x["name"] for x in t if x["name"].startswith("eval_pilot_pilot_R-prog")]
    t.append(dict(name="dtl_grid", deps=rp, mine=SHARD == 0, cmd=f"{PY} {CODE}/select_p11.py {OUT} dtl",
                  done=f"{OUT}/dtl_grid.json"))
    return t


def plan_dtl():
    """after dtl_grid: the 2 DT-L pilot runs + select (deps: every pilot eval)."""
    g = json.load(open(f"{OUT}/dtl_grid.json"))
    t = []
    for a in g["a"]:
        j = train("pilot", "DT-L", {"a": a}, 0, 6000, 32, deps=["s0"], mine=SHARD == 0)
        t += [j, ev("pilot", "DT-L", {"a": a}, 0, "pilot", deps=[j["name"]], mine=SHARD == 0)]
    return t


def select_task(tasks):
    names = [x["name"] for x in tasks if x["name"].startswith(("train_pilot", "eval_pilot"))]
    return dict(name="select", deps=names, mine=SHARD == 0, cmd=f"{PY} {CODE}/select_p11.py {OUT}",
                done=f"{OUT}/params.json")


def n_core():
    try:
        return int(json.load(open(f"{OUT}/shards.json"))["core"])
    except (OSError, ValueError, KeyError):
        return NSHARD


def stage_tasks(stage, arms, seeds, p, first=None):
    t = []
    n = n_core()
    order = ([(first, s) for s in seeds] if first in arms else []) + [(a, s) for s in seeds for a in arms if a != first]
    for i, (arm, s) in enumerate(order):
        hp = p["hp"][arm]
        m = i % n == SHARD
        j = train(stage, arm, hp, s, p["steps"], p["width"], deps=["select"], mine=m)
        t.append(j)
        t.append(ev(stage, arm, hp, s, "core", deps=[j["name"]], mine=m))
    return t


def plan_core():
    """after select: S2 core (R-prog first, so the headroom gate can be read early) + the headroom task."""
    p = json.load(open(f"{OUT}/params.json"))
    if not p.get("proceed"):
        return []
    t = stage_tasks("core", CORE, p["core_seeds"], p, first="R-prog")
    rp = [x["name"] for x in t if x["name"].startswith("eval_core_core_R-prog")]
    t.append(dict(name="headroom", deps=rp, mine=SHARD == 0, cmd=f"{PY} {CODE}/select_p11.py {OUT} headroom",
                  done=f"{OUT}/headroom.json"))
    return t


def plan_secondary():
    """after the headroom gate: S3 secondary unless R-prog's best readout is >= 95% on E1-E3."""
    p = json.load(open(f"{OUT}/params.json"))
    if json.load(open(f"{OUT}/headroom.json")).get("skip_s3"):
        return []
    return stage_tasks("secondary", SECONDARY, p["secondary_seeds"], p)


class Adopted:
    """a train/eval process left running by an earlier queue."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        try:
            st = open(f"/proc/{self.pid}/stat").read().rsplit(")", 1)[1].split()[0]
        except OSError:
            return 0
        return 0 if st == "Z" else None


def running_cmdlines():
    out = {}
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != os.getpid():
            try:
                out[int(p)] = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                pass
    return out


def qlog(msg):
    with open(f"{OUT}/logs/queue{TAG}.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def main():
    which = sys.argv[1]
    assert which == "all"
    tasks = plan_all()
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    open(f"{OUT}/JOB_RUNNING", "w").write(f"{which} shard {SHARD}/{NSHARD}\n")
    for f in ("JOB_DONE", "JOB_FAILED"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    added = set()
    running, tries, failed = {}, {}, set()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=CODE)
    qlog(f"queue start shard {SHARD}/{NSHARD} workers {WORKERS}")
    adopted_once = False
    last_change, last_sig = time.time(), None
    final = False
    while True:
        byname = {t["name"]: t for t in tasks}
        if not adopted_once:
            cl = running_cmdlines()
            for t in tasks:
                if "sig" in t and not os.path.exists(t["done"]):
                    pids = [pid for pid, c in cl.items() if t["sig"] in c + " " and "queue_p11" not in c]
                    if pids:
                        running[t["name"]] = (Adopted(min(pids)), time.time())
                        qlog(f"adopt {t['name']} pid {min(pids)}")
            adopted_once = True
        for n, (pr, t0) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            del running[n]
            ok = rc == 0 and os.path.exists(byname[n]["done"])
            qlog(f"{'ok' if ok else 'FAIL'} {n} rc={rc} {time.time() - t0:.0f}s")
            if not ok:
                tries[n] = tries.get(n, 0) + 1
                if tries[n] >= 2 or n in ("select", "headroom", "dtl_grid"):
                    failed.add(n)
        done = {t["name"] for t in tasks if os.path.exists(t["done"])}
        if "dtl" not in added and os.path.exists(f"{OUT}/dtl_grid.json"):
            tasks += plan_dtl()
            tasks.append(select_task(tasks))
            added.add("dtl")
            qlog("added DT-L pilot + select")
            continue
        if "core" not in added and os.path.exists(f"{OUT}/params.json"):
            if "dtl" not in added:          # shard != 0 may see params.json before dtl_grid.json
                tasks.append(dict(name="select", deps=[], mine=False, cmd="", done=f"{OUT}/params.json"))
                added.add("dtl")
            new = plan_core()
            tasks += new
            added.add("core")
            final = not new
            qlog(f"added core ({sum(t['mine'] for t in new)} mine of {len(new)})")
            continue
        if "sec" not in added and os.path.exists(f"{OUT}/headroom.json"):
            new = plan_secondary()
            tasks += new
            added.add("sec")
            final = True
            qlog(f"added secondary ({sum(t['mine'] for t in new)} mine of {len(new)})")
            continue
        pending = [t for t in tasks if t["name"] not in done and t["name"] not in running and t["name"] not in failed]
        blocked = {t["name"] for t in pending if any(d in failed for d in t["deps"])}
        ready = [t for t in pending if t["mine"] and t["name"] not in blocked and all(d in done for d in t["deps"])]
        # evals + gates first (they unblock selection/analysis), then core-arm trains, then secondary, in plan order
        ready.sort(key=lambda t: 0 if not t["name"].startswith("train") else (1 if t.get("arm") in CORE else 2))
        while ready and len(running) < WORKERS:
            t = ready.pop(0)
            log = open(f"{OUT}/logs/{t['name']}.log", "a")
            running[t["name"]] = (subprocess.Popen(t["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                   cwd=CODE), time.time())
            qlog(f"start {t['name']}")
        mine_left = [t["name"] for t in pending if t["mine"] and t["name"] not in blocked]
        sig = (len(done), len(running), len(failed), len(tasks))
        if sig != last_sig:
            last_sig, last_change = sig, time.time()
        json.dump(dict(shard=SHARD, nshard=NSHARD, done=sorted(done), running=sorted(running), failed=sorted(failed),
                       blocked=sorted(blocked), mine_left=mine_left, total=len(tasks), added=sorted(added),
                       final=final, t=time.time()), open(f"{OUT}/queue_state{TAG}.json", "w"))
        if not running and not mine_left and final:
            break
        if not running and time.time() - last_change > STALL_MIN * 60:
            qlog(f"STALL: nothing changed for {STALL_MIN} min; waiting on {[t["name"] for t in pending[:5]]}")
            failed.add("stall")
            break
        time.sleep(TICK)
    if os.path.exists(f"{OUT}/JOB_RUNNING"):
        os.remove(f"{OUT}/JOB_RUNNING")
    fin = json.dumps(dict(shard=SHARD, failed=sorted(failed), blocked=sorted(blocked), t=time.time())) + "\n"
    open(f"{OUT}/JOB_FAILED" if failed or blocked else f"{OUT}/JOB_DONE", "w").write(fin)
    # per-shard marker that survives the sync into the shared results dir (gpu.sh keep deletes the VM on it)
    open(f"{OUT}/{'FAILED' if failed or blocked else 'DONE'}-s{SHARD}.json", "w").write(fin)
    qlog("queue end")


if __name__ == "__main__":
    main()
