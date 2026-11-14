"""Summary of the run that stopped at gate G0: gate accuracies with Wilson 95% intervals (over mazes; the gate is seed 0
by design), accuracy by S-T distance, tuning table, dense validity bar and the diagnostics outside the spec.
Reads results/vm (synced from the VM) and the JSON printed by gate_diag.py; writes results/summary.json and
results/chart.json.  Usage: python3 summarize_gate.py gate_diag.json"""
import json
import math
import sys

R = "results/vm"


def wilson(p, n, z=1.96):
    k = p * n
    c = (k + z * z / 2) / (n + z * z)
    h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [round(c - h, 4), round(c + h, 4)]


def last_train(name):
    rows = [json.loads(l) for l in open(f"{R}/logs/{name}.log") if l.startswith("{") and '"it"' in l]
    return dict(it=rows[-1]["it"], acc_avg=round(rows[-1]["acc_avg"], 4))


def dense(name):
    d = json.load(open(f"{R}/eval/{name}.json"))
    it = d["sets"]["valdense9"]["it"]
    return dict(iters=d["args"]["iters"], per_cell_acc_128=round(it["128"]["acc"], 4),
                exact_maze_128=it["128"]["exact_maze"], bar_exact_maze=0.95,
                passed=it["128"]["exact_maze"] >= 0.95, train=last_train(name.replace("_s0", "_s0")))


diag = json.load(open(sys.argv[1]))
state = json.load(open(f"{R}/driver_state.json"))
runs = {"g0_A1_s0": ("A1", "standard 4000 it", "A1_s0"), "g0_A2_s0": ("A2", "standard 4000 it", "A2_s0"),
        "g0_A1c_s0": ("A1", "spec curriculum + 12000 it", "A1c_s0"),
        "g0_A2c_s0": ("A2", "spec curriculum + 12000 it", "A2c_s0"),
        "diag_A1inc_s0": ("A1", "DIAGNOSTIC, not in spec: incremental curriculum + 12000 it", "diag_A1inc_s0"),
        "diag_A1winc_s0": ("A1w", "DIAGNOSTIC, not in spec: incremental curriculum + 12000 it", "diag_A1winc_s0")}
gate = {}
for k, (arm, how, log) in runs.items():
    out = dict(arm=arm, training=how, seed=0, train_last=last_train(log))
    for s, r in diag[k].items():
        out[s] = dict(acc=r["acc"], ci95=wilson(r["acc"], r["n"]), n=r["n"], best_it=r["best_it"],
                      by_distance={b: dict(acc=round(x["acc"], 4), n=x["n"]) for b, x in r["buckets"].items()})
    gate[k] = out
summary = dict(
    id="P5", outcome="G0 failed after the curriculum: no P5 verdict (spec: stop and report too hard at this scale)",
    g0=state["g0"], g0_bar=0.90, tuned_gate=state["tuned_gate"], gate_runs=gate,
    dense_validity={"A1dense_s0": dense("A1dense_s0"),
                    "diag_A1dense12k_s0 (DIAGNOSTIC, not in spec)": dense("diag_A1dense12k_s0")},
    bench=state["bench"], tests=json.load(open(f"{R}/tests_vm.json")),
    note="Intervals are Wilson 95% over validation mazes for one seed; the gate is defined on seed 0, and no "
         "multi-seed stage ran because G0 failed.")
json.dump(summary, open("results/summary.json", "w"), indent=1)

buckets = list(diag["g0_A1_s0"]["val9"]["buckets"])
series = [dict(name=f"{k} ({v['training']})", values=[v["val9"]["by_distance"][b]["acc"] for b in buckets],
               n=[v["val9"]["by_distance"][b]["n"] for b in buckets]) for k, v in gate.items()]
chart = dict(type="line", title="P5 gate G0: val9 accuracy by S-T path distance (seed 0)",
             x=dict(label="S-T distance along the path (cells)", categories=buckets),
             y=dict(label="val9 accuracy (answer at T)", min=0, max=1), reference_lines=[
                 dict(y=0.5, label="chance"), dict(y=0.9, label="G0 bar (overall val9)")], series=series,
             overall=[dict(name=k, acc=v["val9"]["acc"], ci95=v["val9"]["ci95"]) for k, v in gate.items()])
json.dump(chart, open("results/chart.json", "w"), indent=1)
print(json.dumps({k: [v["val9"]["acc"], v["val9"]["ci95"]] for k, v in gate.items()}))
print(json.dumps(summary["dense_validity"]))
