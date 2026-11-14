"""P13 job definitions (commands run from ~/p13 on a VM). Used by the phase scripts to write gpu/queues/<slot>.txt.
Every training job trains to a step budget computed from Phase-1 clean timing (PREREG A4)."""
import json, math, os

G = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(G)
FIXES = {"F0": "", "TR1": "--dbcap 1", "TR2": "--liq sqrt", "LB": "--bcap 10", "HF": "--l2b 1e-3", "FL": "--floor 0.01", "IN": "--insq 1"}
FIX_TIMING = {"TR2": "econ_liqsqrt", "IN": "econ_insq"}          # fixes with their own timing; others use the base arm's
ECON = "econ_maze.py --mode econ --W0 20 --frac 0.5 --wealth_cap 1 --lam_c 1 --lam_v 0.3 --train_pick random --D 30 --lr 1e-3 --bs 50 --diag_every 100"
NOCAP = ECON.replace("--wealth_cap 1", "--wealth_cap 0")
BPTT_DS = "econ_maze.py --mode bptt --ds 1 --covsup 1 --vocsup 1 --ptopt 1 --gradckpt 1 --train_pick random --D 30 --lr 1e-3 --bs 50 --diag_every 100"
BPTT_PROG = "econ_maze.py --mode bptt --prog 1 --alpha 0.01 --covsup 1 --vocsup 1 --ptopt 1 --gradckpt 1 --train_pick random --D 30 --lr 1e-3 --bs 50 --diag_every 100"
DTR = "dtr_maze.py --width 128 --alpha 0.01 --T 30 --warmup_frac 0.2"


def phase1():
    p = os.path.join(D, "results", "vm", "trader-b3-p13-1", "out", "phase1.json")
    return json.load(open(p))


def sec_per_step(arm, fix="F0", P=None):
    P = P or phase1()
    key = {"econ": "econ", "nocap": "econ_nocap", "bptt_ds": "bptt_dscov", "bptt_prog": "bptt_progcov", "dtr": "dtr"}[arm]
    if arm != "dtr" and fix in FIX_TIMING:
        base = P["solo"][FIX_TIMING[fix]]["sec_per_1k"] / 1000
        if arm != "econ":         # scale the fix's extra cost (measured on ECON) onto the arm's own timing
            base *= P["solo"][key]["sec_per_1k"] / P["solo"]["econ"]["sec_per_1k"]
        return base
    return P["solo"][key]["sec_per_1k"] / 1000


def steps_for(hours, arm, fix="F0", P=None):
    return int(math.floor(hours * 3600 / sec_per_step(arm, fix, P)))


def trader_job(name, arm, fix, steps, seed, extra=""):
    base = {"econ": ECON, "nocap": NOCAP, "bptt_ds": BPTT_DS, "bptt_prog": BPTT_PROG}[arm]
    ck = "" if "--ckpt_every" in extra else "--ckpt_every 300"      # 23:25Z addendum: spot waves every ~30 min
    return f"{name} {base} {FIXES[fix]} {extra} {ck} --iters {steps} --seed {seed} --out out/{name}".replace("  ", " ")


def dtr_job(name, steps, seed):
    return f"{name} {DTR} --steps {steps} --seed {seed} --name {name} --out out"


def eval_job(name, dep, kind, ckpt, sets, jopt="none", mult=12, extra=""):
    d = f"@{dep}" if dep else ""
    return f"{name}{d} eval_maze.py --kind {kind} --ckpt {ckpt} --sets {sets} --jopt {jopt} --mult {mult} {extra} --out out/{name}".rstrip()


def write_queue(slot, lines, parallel=2, end=True):
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
