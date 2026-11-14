"""Phase 3 queues (PREREG A3 seeds, A4 budgets, cut order) on new slots trader-b3-p13-<next>.

  plan_main.py dtr [--dry]     DTR-eq seeds 0-3 and the M-ref evals. Neither depends on the pilot selection (DTR-eq's J
                               option is always none), so they start with the pilots. Each DTR-eq run is followed on its VM
                               by ev59 (v9, v33, m59 at 12n), evT (e19, e33, e59) and ev99 (e99, mult MULT99_DTR).
                               M-ref: ev59_mref_none, evT_mref, ev99_mref (mult 20).
  plan_main.py trader [--dry]  after results/pilot_selection_<tag>.json (and results/ladder_selection.json when the ladder
                               ran): ECON-fix 4, ECON-nocap 4, ECON-active 2, BPTT-dscov 2 (1 if B > 3 h), BPTT-progcov 2
                               (1 if B > 3 h), ECON-F0 1 (merged into ECON-fix when F0 is selected); each run is followed
                               on its VM by ev59 for the 3 J options (none: v9, v33, m59; scale, cap: m59).
Runs are ordered by the cut order (never-cut first) and dealt round-robin over the VMs, k per VM (Phase-1 packing).
  --only a,b,...               queue only these run names (02:30Z addendum: the spend cut is applied up front; the arms
                               that survive the cut are queued when they are known, e.g. ECON-active after the ladder)
Writes gpu/main_plan.json (read by analyze.py and plan_evals.py)."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import G, D, phase1, sec_per_step, trader_job, dtr_job, eval_job, write_queue

H = 3600.0
MAIN = os.path.join(G, "main_plan.json")
MULT99_DTR = float(os.environ.get("MULT99_DTR", "20"))


def load_plan():
    return json.load(open(MAIN)) if os.path.exists(MAIN) else {"runs": [], "slots": {}}


def next_slots(n):
    sl = os.path.join(G, "slots.list")
    cur = open(sl).read().split() if os.path.exists(sl) else []
    k = max([int(s.split("-")[-1]) for s in cur] + [1])
    return [f"trader-b3-p13-{k + 1 + i}" for i in range(n)]


def pack(n_runs, mems, B_h, eval_h, base="econ"):
    """k jobs per VM: fits in 24 GB (21 GB budget), wall (B x slowdown + evals) <= 5.2 h; minimize VM-h per job.
    Slowdowns from the Phase-1 packed timing of the same arm (ECON x2/x3; DTR x2 only for DTR-eq)."""
    P = phase1()
    S, pk = P["solo"], P.get("packed", {})
    slow = {1: 1.0}
    for k, key in ([(2, "econ_x2"), (3, "econ_x3")] if base == "econ" else [(2, "dtr_x2")]):
        vals = [v["sec_per_1k"] for t, v in pk.get(key, {}).items() if isinstance(v, dict) and "sec_per_1k" in v]
        if len(vals) == k:
            slow[k] = max(vals) / S[base]["sec_per_1k"]
    ok = [k for k in slow if k * max(mems) < 21000 and B_h * slow[k] + k * eval_h <= 5.2]
    k = min(ok, key=lambda k: (slow[k] / k, -k)) if ok else 1
    return k, slow


def deal(items, nvm):
    return [items[i::nvm] for i in range(nvm)]


def main():
    mode = sys.argv[1]
    dry = "--dry" in sys.argv
    P = phase1()
    bud = json.load(open(os.path.join(G, "budgets.json")))
    B = bud["B_h"] * H
    plan = load_plan()
    plan.setdefault("runs", []); plan.setdefault("slots", {})
    plan["B_h"] = bud["B_h"]
    mem = lambda key: P["solo"][key].get("peak_mem_mb", 6000)
    if mode == "dtr":
        steps = int(math.floor(B / sec_per_step("dtr", P=P)))
        runs = [{"name": f"m_dtr_s{s}", "arm": "DTR-eq", "seed": s, "kind": "dtr", "steps": steps} for s in range(4)]
        k, slow = pack(4, [mem("dtr")], bud["B_h"], 0.5, base="dtr")
        nvm = math.ceil(4 / k)
        slots = next_slots(nvm)
        ev99 = [f"ev99_mref", "", "mref", "mref/dt_net.pth", "e99", "none", 20]
        mref = [eval_job("ev59_mref_none", "", "mref", "mref/dt_net.pth", "v9,v33,m59"),
                eval_job("evT_mref", "", "mref", "mref/dt_net.pth", "e19,e33,e59"),
                eval_job(*ev99)]
        for i, (s, mine) in enumerate(zip(slots, deal(runs, nvm))):
            lines = [dtr_job(r["name"], r["steps"], r["seed"]) for r in mine]
            for r in mine:
                n = r["name"]
                lines += [eval_job(f"ev59_{n}_none", n, "dtr", f"out/{n}.pt", "v9,v33,m59"),
                          eval_job(f"evT_{n}", n, "dtr", f"out/{n}.pt", "e19,e33,e59"),
                          eval_job(f"ev99_{n}", n, "dtr", f"out/{n}.pt", "e99", mult=MULT99_DTR)]
                r["slot"] = s
            if i == nvm - 1:
                lines += mref
            print(s, "k", k, [r["name"] for r in mine], "+ M-ref evals" if i == nvm - 1 else "")
            if not dry:
                write_queue(s, lines, parallel=k)
                plan["slots"][s] = {"stage": "dtr", "k": k}
        plan["dtr_pack"] = {"k": k, "slowdown": slow, "mult99_dtr": MULT99_DTR}
    elif mode == "trader":
        tag = "1p0" if os.path.exists(os.path.join(D, "results", "pilot_selection_1p0.json")) else "0p5"
        selp = os.path.join(D, "results", f"pilot_selection_{tag}.json")
        sel = json.load(open(selp))
        assert sel.get("gate"), "learnability gate failed: no main phase (F0-gate)"
        fix = sel["selected"]
        lad = os.path.join(D, "results", "ladder_selection.json")
        active = None
        if not sel["econ_fix_active"] and os.path.exists(lad):
            active = json.load(open(lad)).get("selected")          # "--W0 0.5" | "--bounty lindahl" | None
        nb = 1 if bud["B_h"] > 3.0 else 2

        def st(arm, f):
            return int(math.floor(B / sec_per_step(arm, f, P)))
        R = []
        for s in range(4):
            R.append({"name": f"m_econ_s{s}", "arm": "ECON-fix", "seed": s, "kind": "trader", "a": "econ", "fix": fix})
            R.append({"name": f"m_nocap_s{s}", "arm": "ECON-nocap", "seed": s, "kind": "trader", "a": "nocap", "fix": fix})
        # cut order, last cut first: ECON-active s0, BPTT s0 (not in the cut list), ECON-active seed 2, ECON-F0 main,
        # BPTT-dscov seed 2, BPTT-progcov seed 2
        if active:
            R.append({"name": "m_active_s0", "arm": "ECON-active", "seed": 0, "kind": "trader", "a": "econ", "fix": fix, "extra": active})
        R.append({"name": "m_bds_s0", "arm": "BPTT-dscov", "seed": 0, "kind": "trader", "a": "bptt_ds", "fix": fix})
        R.append({"name": "m_bprog_s0", "arm": "BPTT-progcov", "seed": 0, "kind": "trader", "a": "bptt_prog", "fix": fix})
        if active:
            R.append({"name": "m_active_s1", "arm": "ECON-active", "seed": 1, "kind": "trader", "a": "econ", "fix": fix, "extra": active})
        if fix != "F0":
            R.append({"name": "m_f0_s0", "arm": "ECON-F0", "seed": 0, "kind": "trader", "a": "econ", "fix": "F0"})
        if nb == 2:
            R.append({"name": "m_bds_s1", "arm": "BPTT-dscov", "seed": 1, "kind": "trader", "a": "bptt_ds", "fix": fix})
            R.append({"name": "m_bprog_s1", "arm": "BPTT-progcov", "seed": 1, "kind": "trader", "a": "bptt_prog", "fix": fix})
        if "--only" in sys.argv:
            only = sys.argv[sys.argv.index("--only") + 1].split(",")
            assert set(only) <= {r["name"] for r in R}, (only, [r["name"] for r in R])
            R = [r for r in R if r["name"] in only]
        for r in R:
            r["steps"] = st(r["a"], r["fix"])
        mems = [mem({"econ": "econ", "nocap": "econ_nocap", "bptt_ds": "bptt_dscov", "bptt_prog": "bptt_progcov"}[r["a"]]) for r in R]
        k, slow = pack(len(R), mems, bud["B_h"], 0.3)
        nvm = math.ceil(len(R) / k)
        slots = next_slots(nvm)
        for s, mine in zip(slots, deal(R, nvm)):
            lines = [trader_job(r["name"], r["a"], r["fix"], r["steps"], r["seed"], r.get("extra", "")) for r in mine]
            for r in mine:
                n = r["name"]
                lines += [eval_job(f"ev59_{n}_none", n, "trader", f"out/{n}.pt", "v9,v33,m59"),
                          eval_job(f"ev59_{n}_scale", n, "trader", f"out/{n}.pt", "m59", jopt="scale"),
                          eval_job(f"ev59_{n}_cap", n, "trader", f"out/{n}.pt", "m59", jopt="cap")]
                r["slot"] = s
            print(s, "k", k, [(r["name"], r["steps"]) for r in mine])
            if not dry:
                write_queue(s, lines, parallel=k)
                plan["slots"][s] = {"stage": "trader", "k": k}
        runs = [{kk: v for kk, v in r.items() if kk != "a"} | {"a": r["a"]} for r in R]
        plan["selection_file"] = os.path.basename(selp)
        plan["trader_pack"] = {"k": k, "slowdown": slow}
        plan["fix"] = fix
        plan["econ_active"] = active
        plan.setdefault("trader_queued_at", []).append({"utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                                                        "runs": [r["name"] for r in R], "slots": slots})
    else:
        sys.exit("mode: dtr | trader")
    names = {r["name"] for r in runs}
    plan["runs"] = [r for r in plan["runs"] if r["name"] not in names] + runs
    print(json.dumps({kk: v for kk, v in plan.items() if kk != "runs"}, indent=1))
    if not dry:
        json.dump(plan, open(MAIN + ".tmp", "w"), indent=1)
        os.replace(MAIN + ".tmp", MAIN)


if __name__ == "__main__":
    main()
