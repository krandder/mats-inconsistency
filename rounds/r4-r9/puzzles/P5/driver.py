"""P5 job driver on the GPU VM. Phases (each job skipped when its output exists, so a relaunch resumes):
bench -> tune (4 configs x 7 arms, seed 0, 1500 iters, chosen on validation) -> gate G0 (A1, A2 seed 0; best val9 >= 90%,
else curriculum + 12k iterations, else stop) -> stage 1 (train + full eval) -> memory vs depth -> stage 2 (conditional).
Usage: python driver.py all [--P 4]"""
import json
import os
import subprocess
import sys
import time

import numpy as np

OUT = "/root/out"
JOB = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
P = int(os.environ.get("P5_PARALLEL", "4"))
# seeds for the six main arms: the spec asks for 5, but the benchmark projects ~16-20 VM-h for 5 seeds against a
# 17.5 VM-h ($7 at $0.40/h) limit, so the main arms run the spec's minimum of 3 seeds (documented in RESULTS.md)
NS_MAIN = int(os.environ.get("P5_SEEDS_MAIN", "3"))
os.makedirs(f"{OUT}/logs", exist_ok=True)
os.makedirs(f"{OUT}/ckpt", exist_ok=True)
os.makedirs(f"{OUT}/eval", exist_ok=True)

TUNE = {
    "A1": [dict(lr=lr, alpha=al) for lr in (3e-4, 1e-3) for al in (0.01, 0.5)],
    "A1w": [dict(lr=lr, alpha=al) for lr in (3e-4, 1e-3) for al in (0.01, 0.5)],
    "A2": [dict(lr=lr, alpha=al) for lr in (3e-4, 1e-3) for al in (0.01, 0.5)],
    "A2t": [dict(lr=lr, clip=cl) for lr in (3e-4, 1e-3) for cl in (0.5, 2.0)],
    "A5": [dict(lr=lr, clip=cl) for lr in (3e-4, 1e-3) for cl in (0.5, 2.0)],
    "A6": [dict(lr=lr, hlr=h) for lr in (3e-4, 1e-3) for h in (1.0, 10.0)],
    "A7": [dict(lr=lr, hlr=h) for lr in (3e-4, 1e-3) for h in (1.0, 10.0)],
}
CFG_FROM = {"A3": "A5", "A4": "A5", "A8": "A5", "A10": "A5", "A7p": "A7", "A7f": "A7"}


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    open(f"{OUT}/logs/driver.log", "a").write(msg + "\n")


def state():
    p = f"{OUT}/driver_state.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(s):
    json.dump(s, open(f"{OUT}/driver_state.json.tmp", "w"), indent=1)
    os.replace(f"{OUT}/driver_state.json.tmp", f"{OUT}/driver_state.json")


def run_jobs(jobs, par=P):
    """jobs: list of dict(name, cmd, done, deps=[]). Runs up to par at once; returns names that failed."""
    pending = [j for j in jobs if not os.path.exists(j["done"])]
    done = {j["name"] for j in jobs if os.path.exists(j["done"])}
    failed, running = set(), {}
    while pending or running:
        for name, (proc, j) in list(running.items()):
            if proc.poll() is not None:
                del running[name]
                if proc.returncode == 0 and os.path.exists(j["done"]):
                    done.add(name)
                    log("done", name)
                else:
                    failed.add(name)
                    log("FAILED", name, "rc", proc.returncode)
        for j in list(pending):
            if len(running) >= par:
                break
            deps = j.get("deps", [])
            if any(d in failed for d in deps):
                pending.remove(j); failed.add(j["name"]); log("skip (dep failed)", j["name"])
                continue
            if all(d in done for d in deps):
                pending.remove(j)
                lf = open(f"{OUT}/logs/{j['name']}.log", "a")
                proc = subprocess.Popen(j["cmd"], shell=True, cwd=JOB, stdout=lf, stderr=subprocess.STDOUT)
                running[j["name"]] = (proc, j)
                log("start", j["name"], "|", j["cmd"][:200])
        time.sleep(2)
    return failed


def train_job(name, arm, seed, iters, cfg, extra="", deps=()):
    ck = f"{OUT}/ckpt/{name}.pt"
    a = " ".join(f"--{k} {v}" for k, v in cfg.items())
    return dict(name=name, done=ck, deps=list(deps),
                cmd=f"{PY} p5.py --arm {arm} --seed {seed} --iters {iters} {a} {extra} --out {ck}")


def eval_job(name, groups="all", frac=1.0, extras=True):
    o = f"{OUT}/eval/{name}"
    return dict(name=f"ev_{name}", done=o + ".json", deps=[name],
                cmd=f"{PY} evaluate.py --ckpt {OUT}/ckpt/{name}.pt --out {o} --groups {groups} --frac {frac}"
                    + (" --extras" if extras else ""))


def evres(name):
    return json.load(open(f"{OUT}/eval/{name}.json"))


def best_val(name):
    r = evres(name)
    return max(r["val_score"].values()), r["best_it"]


def acc_at_best(name, sname):
    r = evres(name)
    return r["sets"][sname]["it"][str(r["best_it"])]["acc"] if str(r["best_it"]) in r["sets"][sname]["it"] else \
        r["sets"][sname]["it"][r["best_it"]]["acc"]


def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h


def main():
    st = state()
    open(f"{OUT}/JOB_RUNNING", "w").write(str(time.time()))
    # ---------------- bench
    if "bench" not in st:
        bj = []
        for arm in ["A1", "A1w", "A2", "A2t", "A3", "A4", "A5", "A6", "A7"]:
            bj.append(dict(name=f"bench_{arm}", done=f"{OUT}/ckpt/bench_{arm}.pt",
                           cmd=f"{PY} p5.py --arm {arm} --iters 30 --log_every 10 --cos_every 0 --out {OUT}/ckpt/bench_{arm}.pt"))
        run_jobs(bj, par=1)
        b = {}
        for arm in ["A1", "A1w", "A2", "A2t", "A3", "A4", "A5", "A6", "A7"]:
            try:
                b[arm] = json.load(open(f"{OUT}/ckpt/bench_{arm}.train.json"))["log"][-1]["s_per_it"]
            except Exception as e:
                b[arm] = None
        # eval throughput: 200 iterations of each group with an untrained bank model and A1w, extrapolated
        ev = {}
        for arm in ["A5", "A1", "A1w"]:
            o = f"{OUT}/eval/bench_{arm}"
            if not os.path.exists(o + ".json"):
                subprocess.run(f"{PY} evaluate.py --ckpt {OUT}/ckpt/bench_{arm}.pt --out {o} --max_rec 200",
                               shell=True, cwd=JOB, stdout=open(f"{OUT}/logs/bench_eval_{arm}.log", "a"), stderr=subprocess.STDOUT)
            r = json.load(open(o + ".json"))
            full = {"g9": 6400, "g13": 6400, "g19": 6400, "g33": 6400, "g59": 6400, "lat30": 6400, "lat100": 15160, "stab33": 64000}
            ev[arm] = sum(r["timing"][g] * full[g] / 200 for g in r["timing"])
        n_bank = 5 * 5 + 3 * 2 + 3  # stage-1 bank/belief models (A2, A2t, A5, A6, A7 x5; A3, A4 x3; dense A3 x3)
        n_a1, n_a1w = 5 + 3, 3
        total_h = (n_bank * ev["A5"] + n_a1 * ev["A1"] + n_a1w * ev["A1w"]) / 3600
        st["bench"] = dict(train_s_per_it=b, eval_s_per_model=ev, eval_total_gpu_h=total_h,
                           frac=0.5 if total_h > 3 else 1.0)
        save_state(st)
        log("bench", json.dumps(st["bench"]))
    frac = st["bench"]["frac"]
    # ---------------- tune A1, A1w, A2 (the gate arms first), then gate G0, then tune the other arms (with the
    # curriculum if G0 needed it, since every arm then trains with it)
    def tune(arms, key, extra=""):
        if key in st:
            return
        jobs = []
        for arm in arms:
            for k, cfg_ in enumerate(TUNE[arm]):
                n = f"tune_{arm}_c{k}" + ("c" if extra else "")
                jobs.append(train_job(n, arm, 0, 1500, cfg_, "--cos_every 0 " + extra))
                jobs.append(eval_job(n, "val", extras=False))
        run_jobs(jobs)
        tuned = {}
        for arm in arms:
            sc = {}
            for k in range(len(TUNE[arm])):
                try:
                    sc[k] = best_val(f"tune_{arm}_c{k}" + ("c" if extra else ""))[0]
                except Exception:
                    sc[k] = -1
            kb = max(sc, key=lambda k: sc[k])
            tuned[arm] = dict(cfg=TUNE[arm][kb], k=kb, val=sc)
        st[key] = tuned
        save_state(st)
        log(key, json.dumps(tuned))

    tune(["A1", "A1w", "A2"], "tuned_gate")
    cfg = {a: st["tuned_gate"][a]["cfg"] for a in ("A1", "A1w", "A2")}
    # ---------------- gate G0
    if "g0" not in st:
        # the curriculum fallback runs speculatively alongside (4 free slots); it is used only if the standard runs fail
        jobs = [train_job("A1_s0", "A1", 0, 4000, cfg["A1"]), train_job("A2_s0", "A2", 0, 4000, cfg["A2"]),
                train_job("A1c_s0", "A1", 0, 12000, cfg["A1"], "--curr"), train_job("A2c_s0", "A2", 0, 12000, cfg["A2"], "--curr")]
        for a in ("A1", "A2", "A1c", "A2c"):
            jobs.append(dict(name=f"evg0_{a}_s0", done=f"{OUT}/eval/g0_{a}_s0.json", deps=[f"{a}_s0"],
                             cmd=f"{PY} evaluate.py --ckpt {OUT}/ckpt/{a}_s0.pt --out {OUT}/eval/g0_{a}_s0 --groups g9"))
        run_jobs(jobs)
        v = {a: max(it["acc"] for it in evres(f"g0_{a}_s0")["sets"]["val9"]["it"].values()) for a in ("A1", "A2")}
        g0 = dict(val9=v, passed=max(v.values()) >= 0.90, mode="standard")
        if not g0["passed"]:
            log("G0 failed with standard training", v, "-> curriculum + 12k iterations")
            v2 = {a: max(it["acc"] for it in evres(f"g0_{a}c_s0")["sets"]["val9"]["it"].values()) for a in ("A1", "A2")}
            g0.update(val9_curriculum=v2, passed=max(v2.values()) >= 0.90, mode="curriculum")
        st["g0"] = g0
        save_state(st)
        log("G0", json.dumps(g0))
    if not st["g0"]["passed"]:
        open(f"{OUT}/TOO_HARD", "w").write(json.dumps(st["g0"]))
        log("G0 failed after curriculum: task too hard, stopping (no P5 verdict)")
        finish(st)
        return
    cur = st["g0"]["mode"] == "curriculum"
    tune(["A2t", "A5", "A6", "A7"], "tuned", "--curr" if cur else "")
    for a in ("A2t", "A5", "A6", "A7"):
        cfg[a] = st["tuned"][a]["cfg"]
    for a, src in CFG_FROM.items():
        cfg[a] = cfg[src]
    iters = 12000 if cur else 4000
    extra = "--curr" if cur else ""
    sfx = "c" if cur else ""
    # ---------------- stage 1
    runs = []
    for arm in ["A1", "A2", "A2t", "A5", "A6", "A7"]:
        runs += [(f"{arm}{sfx}_s{s}", arm, s, "") for s in range(NS_MAIN)]
    for arm in ["A1w", "A3", "A4"]:
        runs += [(f"{arm}{sfx}_s{s}", arm, s, "") for s in range(3)]
    for arm in ["A1", "A3"]:
        runs += [(f"{arm}dense_s{s}", arm, s, "--dense") for s in range(3)]
    # order: expensive market arms first so the pool stays busy
    order = {"A5": 0, "A7": 1, "A4": 2, "A6": 3, "A1w": 4, "A2": 5, "A2t": 6, "A3": 7, "A1": 8}
    runs.sort(key=lambda r: (order[r[1]], r[2]))
    jobs = []
    for n, arm, s, ex in runs:
        it_ = 4000 if ex == "--dense" else iters
        e2 = (extra if ex != "--dense" else "") + " " + ex
        jobs.append(train_job(n, arm, s, it_, cfg[arm], e2))
        jobs.append(eval_job(n, "all", frac=frac, extras=True))
    failed = run_jobs(jobs)
    st["stage1_failed"] = sorted(failed)
    save_state(st)
    # ---------------- memory vs depth (GPU otherwise idle)
    mj = []
    for arm in ["A2", "A5", "A6", "A7"]:
        for D in (32, 64, 128, 256):
            n = f"mem_{arm}_D{D}"
            mj.append(dict(name=n, done=f"{OUT}/ckpt/{n}.pt",
                           cmd=f"{PY} p5.py --arm {arm} --iters 6 --D {D} --cos_every 0 --log_every 1 --out {OUT}/ckpt/{n}.pt"))
    mj.append(dict(name="flops", done=f"{OUT}/flops.json", cmd=f"{PY} flops.py {OUT}/flops.json"))
    run_jobs(mj, par=1)
    # ---------------- stage 2 (conditional)
    if "stage2" not in st:
        def accs(arm):
            out = []
            for s in range(5):
                try:
                    out.append(acc_at_best(f"{arm}{sfx}_s{s}", "test19"))
                except Exception:
                    pass
            return out
        trig = {}
        d, lo, hi = welch(accs("A5"), accs("A2t"))
        trig["A5_minus_A2t_test19"] = [d, lo, hi]
        do_a8 = d >= 0.03 and lo > 0
        d, lo, hi = welch(accs("A7"), accs("A6"))
        trig["A7_minus_A6_test19"] = [d, lo, hi]
        do_a7p = d >= 0.03 and lo > 0
        st["stage2"] = dict(trig=trig, run_A8_A10=bool(do_a8), run_A7p_A7f=bool(do_a7p))
        save_state(st)
        log("stage2 triggers", json.dumps(st["stage2"]))
    arms2 = (["A8", "A10"] if st["stage2"]["run_A8_A10"] else []) + (["A7p", "A7f"] if st["stage2"]["run_A7p_A7f"] else [])
    jobs = []
    for arm in arms2:
        for s in range(3):
            n = f"{arm}{sfx}_s{s}"
            jobs.append(train_job(n, arm, s, iters, cfg[arm], extra))
            jobs.append(eval_job(n, "all", frac=frac, extras=True))
    if jobs:
        st["stage2_failed"] = sorted(run_jobs(jobs))
        save_state(st)
    # ---------------- 64k stability for the stage-1 main arms is inside the full eval (stab33 group)
    finish(st)


def finish(st):
    open(f"{OUT}/JOB_DONE", "w").write(json.dumps(dict(t=time.time(), state=st)))
    if os.path.exists(f"{OUT}/JOB_RUNNING"):
        os.remove(f"{OUT}/JOB_RUNNING")
    log("JOB_DONE")


if __name__ == "__main__":
    main()
