"""P18 job plans (PREREG sections 3, 4 and 7). Writes gpu/queues/<slot>.txt, gpu/slots/<slot> and gpu/slots.list.

  python plan.py phase1   slot trader-dds-1: clean solo timing of every arm (300 steps after 20 warm-up steps),
                          then the M-ref pipeline gate eval on every test set
  python plan.py main     after Phase 1: M-ref gate, step budget B (PREREG formula), 20 primary runs, each on its own
                          slot with its eval chained after it (writes gpu/main_plan.json)
  python plan.py extra    after the 9 alpha-sweep evals: pick the best alpha on dev m59 exact@12n, add seeds 3 and 4
  python plan.py a5       exploratory A5 (K = 1), only if the PREREG spend rule allows it
  python plan.py status   print what is planned, done and pending
Commands in queues run from ~/p18 on a VM.
"""
import json, math, os, sys

G = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(G)
RES = os.path.join(D, "results", "vm")
COMMON = "--width 128 --T 30 --warmup_frac 0.2 --ckpt_sec 300"
ARMS = {
    "a1": "--mode ds --K 4 --order rand --detach 1",     # random-order detached per-step deep supervision
    "a2": "--mode ds --K 4 --order rr --detach 1",       # fixed-order (round robin) detached
    "a3": "--mode ds --K 4 --order rand --detach 0",     # full BPTT, same losses and order as a1
    "p0": "--mode prog --alpha 0",                       # DT-Recall, progressive loss alpha 0 (= full loss only)
    "p05": "--mode prog --alpha 0.5",
    "p1": "--mode prog --alpha 1",
    "a5": "--mode ds --K 1 --order rr --detach 1",       # exploratory: one tied block, detached deep supervision
}
ALPHAS = ["p0", "p05", "p1"]
EVAL_SETS = "v9,m13,m59,e19,e33,e59"
V_CAP = 40 / 0.71            # VM-hours at the ceiling rate
RESERVE = 0.15               # of V_CAP: preemptions, the watchdog's +5 min per VM, relaunches
B_MAX, B_MIN = 2.0, 1.25
P1_SLOT = "trader-dds-1"


def write_queue(slot, lines, parallel=1, end=True):
    os.makedirs(os.path.join(G, "queues"), exist_ok=True); os.makedirs(os.path.join(G, "slots"), exist_ok=True)
    q = os.path.join(G, "queues", slot + ".txt")
    with open(q + ".tmp", "w") as f:
        f.write("\n".join(lines) + ("\nEND\n" if end else "\n"))
    os.replace(q + ".tmp", q)
    with open(os.path.join(G, "slots", slot), "w") as f:
        f.write(f"{parallel}\n")
    sl = os.path.join(G, "slots.list")
    cur = open(sl).read().split() if os.path.exists(sl) else []
    if slot not in cur:
        with open(sl, "a") as f:
            f.write(slot + "\n")


def phase1():
    lines = [f"t_{a} train18.py {ARMS[a]} {COMMON} --steps 100000 --seed 0 --name t_{a} --out out/p1 --time_steps 300"
             for a in ["p05", "a1", "p0", "p1", "a2", "a3", "a5"]]
    lines.append(f"ev_mref eval18.py --kind mref --ckpt mref/dt_net.pth --sets {EVAL_SETS} --out out/ev_mref")
    write_queue(P1_SLOT, lines)
    print("\n".join(lines))


def timings():
    L = os.path.join(RES, P1_SLOT, "out", "logs")
    t = {}
    for a in ARMS:
        f = os.path.join(L, f"t_{a}.log")
        if not os.path.exists(f):
            continue
        for l in open(f):
            if l.startswith('{"timing"'):
                t[a] = json.loads(l)
    return t


def p1_hours():
    h = 0.0
    for l in open(os.path.join(G, "spend-state.tsv")):
        p = l.split()
        if len(p) >= 3 and p[0].startswith(P1_SLOT) and (p[0] == P1_SLOT or p[0].startswith(P1_SLOT + "-r")):
            h += (int(p[2]) - int(p[1]) + 300) / 3600
    return h


def mref_gate():
    f = os.path.join(RES, P1_SLOT, "out", "ev_mref.json")
    s = json.load(open(f))["sets"]
    ev_sec = sum(v["sec"] for v in s.values())
    return s["m59"]["exact@12n"], ev_sec, {k: v["exact@12n"] for k, v in s.items()}


def run_names():
    """the 20 primary runs in launch order (PREREG 4): the alpha sweep first (it gates the extra seeds), then a1, a2, a3"""
    r = [(a, s) for s in range(3) for a in ALPHAS]
    r += [("a1", s) for s in range(5)] + [("a2", s) for s in range(3)] + [("a3", s) for s in range(3)]
    return r


def job_lines(arm, seed, steps):
    n = f"{arm}_s{seed}"
    return [f"{n} train18.py {ARMS[arm]} {COMMON} --steps {steps} --seed {seed} --name {n} --out out",
            f"ev_{n}@{n} eval18.py --ckpt out/{n}.pt --sets {EVAL_SETS} --out out/ev_{n}"]


def main_plan():
    gate, ev_sec, mref_sets = mref_gate()
    if gate < 0.95:
        print(f"M-REF GATE FAILED: m59 exact@12n = {gate} < 0.95; no main runs"); open(os.path.join(G, "STOP"), "w").write("mref gate\n")
        sys.exit(2)
    t = timings()
    missing = [a for a in ARMS if a not in t]
    assert not missing, f"timing missing for {missing}"
    runs = run_names()
    N = len(runs) + 2                           # + the 2 extra seeds of the best alpha
    O = ev_sec / 3600 + 0.2                     # per run: its eval + VM boot, push, final sync, delete
    P1 = p1_hours()
    B = (V_CAP * (1 - RESERVE) - P1 - N * O) / N
    B = max(B_MIN, min(B_MAX, math.floor(B * 20) / 20))
    spk = {a: t[a]["sec_per_1k"] / 1000 for a in ARMS}
    steps = {a: int(B * 3600 / spk[a]) for a in ARMS}
    plan = {"B_hours": B, "N_runs": N, "O_hours": round(O, 3), "P1_hours": round(P1, 3), "V_cap": round(V_CAP, 2),
            "reserve": RESERVE, "sec_per_step": spk, "steps": steps, "mref_gate_m59": gate, "mref_exact12n": mref_sets,
            "eval_sec_mref": ev_sec, "timings": t, "runs": [f"{a}_s{s}" for a, s in runs]}
    json.dump(plan, open(os.path.join(G, "main_plan.json"), "w"), indent=1)
    for i, (a, s) in enumerate(runs):
        write_queue(f"trader-dds-{i + 2}", job_lines(a, s, steps[a]))
    print(json.dumps({k: v for k, v in plan.items() if k not in ("timings", "runs")}, indent=1))


def ev_summary(name):
    for slot in os.listdir(RES):
        f = os.path.join(RES, slot, "out", f"ev_{name}.json")
        if os.path.exists(f) and os.path.exists(os.path.join(RES, slot, "out", f"ev_{name}.done")):
            return json.load(open(f))["sets"]
    return None


def extra():
    plan = json.load(open(os.path.join(G, "main_plan.json")))
    if "best_alpha" in plan:
        print("extra already planned:", plan["best_alpha"]); return
    sc = {}
    for a in ALPHAS:
        ss = [ev_summary(f"{a}_s{s}") for s in range(3)]
        if any(x is None for x in ss):
            print(f"alpha sweep not finished ({a})"); return
        sc[a] = (sum(x["m59"]["exact@12n"] for x in ss) / 3, sum(x["m13"]["exact@12n"] for x in ss) / 3,
                 1 if a == "p05" else 0)
    best = max(ALPHAS, key=lambda a: sc[a])
    plan["best_alpha"] = best; plan["alpha_scores_m59_m13"] = sc
    base = len(plan["runs"]) + 2
    for k, s in enumerate([3, 4]):
        write_queue(f"trader-dds-{base + k}", job_lines(best, s, plan["steps"][best]))
        plan["runs"].append(f"{best}_s{s}")
    json.dump(plan, open(os.path.join(G, "main_plan.json"), "w"), indent=1)
    print("best alpha", best, sc)


def a5():
    plan = json.load(open(os.path.join(G, "main_plan.json")))
    if "a5" in plan:
        print("a5 already decided:", plan["a5"]); return
    if "best_alpha" not in plan:
        print("primary runs not finished"); return
    left = [r for r in plan["runs"] if ev_summary(r) is None]
    if left:
        print("primary runs not finished:", left); return
    spend = float(open(os.path.join(G, "SPEND")).read())
    need = 2 * (plan["B_hours"] + plan["O_hours"]) * 0.71 * 1.3 + 2.0
    ok = spend + need <= 40
    plan["a5"] = {"spend_at_decision": spend, "needed": round(need, 2), "run": ok}
    if ok:
        base = len(plan["runs"]) + 2
        for k, s in enumerate([0, 1]):
            write_queue(f"trader-dds-{base + k}", job_lines("a5", s, plan["steps"]["a5"]))
            plan["runs_a5"] = plan.get("runs_a5", []) + [f"a5_s{s}"]
    json.dump(plan, open(os.path.join(G, "main_plan.json"), "w"), indent=1)
    print("a5:", plan["a5"])


def status():
    sl = open(os.path.join(G, "slots.list")).read().split() if os.path.exists(os.path.join(G, "slots.list")) else []
    for s in sl:
        q = [l.split()[0] for l in open(os.path.join(G, "queues", s + ".txt")) if l.strip() and l.split()[0] != "END"]
        out = os.path.join(RES, s, "out")
        st = []
        for n in q:
            n = n.split("@")[0]
            st.append(n + ("=done" if os.path.exists(os.path.join(out, n + ".done")) else
                           "=FAIL" if any(f.startswith(n + ".fail.") for f in (os.listdir(out) if os.path.isdir(out) else [])) else ""))
        print(s, " ".join(st))


if __name__ == "__main__":
    {"phase1": phase1, "main": main_plan, "extra": extra, "a5": a5, "status": status}[sys.argv[1]]()
