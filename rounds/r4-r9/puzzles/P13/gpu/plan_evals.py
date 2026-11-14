"""Phase 4 queues for the Trader2D arms (DTR-eq and M-ref evals are queued with their runs by plan_main.py dtr).
For every Trader2D arm whose seeds all have ev59 for the 3 J options (synced):
  - J option (A6): highest seed-mean m59 exact at 12n; ties to fewer MACs (cap < none = scale);
  - median-dev seed (A3): seeds sorted by m59 exact at 12n (chosen J) descending, ties to the lower seed; position 2;
  - evT_<run> (e19, e33, e59 at 12n) and ev99_<run> (e99 at 12n);
    02:30Z addendum (spend): Trader2D evT and ev99 use the first LIMIT (100) mazes of each set (--limit), no 20n extension
    for the ECON-fix median-dev seed, and mult 12.1 when J = cap so that t_eq (1.006 x 12n) is inside the run;
  - with --stretch: ev201_<run> for the median-dev seeds of ECON-fix, ECON-nocap and DTR-eq, and M-ref (cut first).
Model files are hard-linked into gpu/inputs/<slot>/ (pushed to ~/p13/in/ on the VM).
Usage: plan_evals.py [--dry] [--vms N] [--par P] [--stretch] [--only99]"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import G, D, eval_job, write_queue
from plan_main import next_slots, MAIN

JMACS = {"cap": 0, "none": 1, "scale": 2}
LIMIT = 100                      # 02:30Z addendum: first 100 mazes of e19/e33/e59/e99 for Trader2D arms


def mult_for(J):
    return 12.1 if J == "cap" else 12


def find(name):
    g = sorted(glob.glob(os.path.join(D, "results", "vm", "*", "out", name)))
    return g[-1] if g else None


def m59_exact(name):
    j, z = find(name + ".json"), find(name + ".npz")
    if not (j and z):
        return None
    S = json.load(open(j))["sets"]["m59"]
    return float(S["exact@12n"])


def arg(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def main():
    dry, stretch = "--dry" in sys.argv, "--stretch" in sys.argv
    plan = json.load(open(MAIN))
    done = plan.setdefault("evals_queued", {})
    arms = {}
    for r in plan["runs"]:
        arms.setdefault(r["arm"], []).append(r)
    jobs, links = [], []
    for arm, rs in arms.items():
        if arm in done:
            continue
        if rs[0]["kind"] == "dtr":
            if stretch and all(m59_exact(f"ev59_{r['name']}_none") is not None for r in rs):
                sc = sorted(rs, key=lambda r: (-m59_exact(f"ev59_{r['name']}_none"), r["seed"]))
                med = sc[1]["name"]
                jobs.append((f"ev201_{med}", "dtr", med, "e201", "none", 12, "")); links.append(med)
                done[arm] = {"jopt": "none", "median_dev": med, "stretch": True}
            continue
        sc = {j: [m59_exact(f"ev59_{r['name']}_{j}") for r in rs] for j in ["none", "scale", "cap"]}
        if any(v is None for vs in sc.values() for v in vs):
            print(arm, "not ready:", {j: sum(v is not None for v in vs) for j, vs in sc.items()}, "of", len(rs))
            continue
        mean = {j: float(np.mean(v)) for j, v in sc.items()}
        J = min(mean, key=lambda j: (-mean[j], JMACS[j]))
        order = sorted(range(len(rs)), key=lambda i: (-sc[J][i], rs[i]["seed"]))
        med = rs[order[1]]["name"] if len(rs) >= 4 else None
        for r in rs:
            n = r["name"]
            if "--only99" not in sys.argv:        # 07:45Z addendum: ev99 only if the $38 line allows no more
                jobs.append((f"evT_{n}", "trader", n, "e19,e33,e59", J, mult_for(J), f"--limit {LIMIT}"))
            jobs.append((f"ev99_{n}", "trader", n, "e99", J, mult_for(J), f"--limit {LIMIT}"))
            links.append(n)
        if stretch and med and arm in ("ECON-fix", "ECON-nocap"):
            jobs.append((f"ev201_{med}", "trader", med, "e201", J, mult_for(J), ""))
        done[arm] = {"jopt": J, "m59_mean_by_J": mean, "median_dev": med, "stretch": stretch, "limit": LIMIT, "mult": mult_for(J)}
        print(arm, "J", J, mean, "median-dev", med)
    if stretch and "M-ref-201" not in done:
        jobs.append(("ev201_mref", "mref", None, "e201", "none", 12, "")); done["M-ref-201"] = True
    if not jobs:
        print("nothing to queue"); return
    nvm = min(arg("--vms", 2), len(jobs))
    slots = next_slots(nvm)
    # longest first (e201 > e99 > evT), dealt round-robin
    w = {"e201": 3, "e99": 2}
    jobs.sort(key=lambda j: -w.get(j[3], 1))
    for i, s in enumerate(slots):
        mine = jobs[i::nvm]
        lines = []
        for name, kind, run, sets, J, mult, extra in mine:
            ck = "mref/dt_net.pth" if kind == "mref" else f"in/{run}.pt"
            lines.append(eval_job(name, "", kind, ck, sets, jopt=J, mult=mult, extra=extra))
            if run and not dry:
                src = find(f"{run}.pt")
                dst = os.path.join(G, "inputs", s, f"{run}.pt")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    os.link(src, dst)
        print(s, *lines, sep="\n  ")
        if not dry:
            write_queue(s, lines, parallel=arg("--par", 2))
            plan["slots"][s] = {"stage": "evals"}
    if not dry:
        json.dump(plan, open(MAIN + ".tmp", "w"), indent=1)
        os.replace(MAIN + ".tmp", MAIN)


if __name__ == "__main__":
    main()
