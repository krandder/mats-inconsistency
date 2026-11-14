"""Write P14 job queues. Each line: NAME p14.py-args. NAME matches p14.py's output name, so a finished
run is skipped on restart. Usage: make_queue.py STAGE [--grace G] [--hp FILE] > queue file."""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TUNE_SEEDS = [100, 101, 102]
FEES = [0.3, 1, 3, 10, 30]
L1 = [1e-3, 1e-2, 1e-1, 1, 10]
L0 = [0.3, 1, 3, 10, 30]
TAUS = [0.5, 1, 2]
# slow arms first so the tail of the queue is short
SLOW = ["V", "V-local", "V-W", "V-noD", "V-L1", "P-LOCO", "BP-D", "BP-L1-D-cm", "N-sel", "P-W"]


def fmt(x):
    return ("%g" % x).replace("+", "")


def line(arm, seed, mode="tune", task="reach", fee=None, lam=0.0, tau=1.0, grace=1000, tag=""):
    name = f"{arm}_{task}_{mode}_s{seed}" + (f"_{tag}" if tag else "")
    a = f"--arm {arm} --seed {seed} --mode {mode} --task {task} --lam {fmt(lam)} --tau {fmt(tau)} --grace {grace}"
    if fee is not None:
        a += f" --fee {fmt(fee)}"
    if tag:
        a += f" --tag {tag}"
    return f"{name} {a}"


def order(lines):
    def key(l):
        arm = l.split()[0].split("_")[0]
        return SLOW.index(arm) if arm in SLOW else len(SLOW)
    return sorted(lines, key=key)


def tune1(G, task="reach", seeds=TUNE_SEEDS):
    out = []
    for s in seeds:
        out.append(line("V-noD", s, task=task, grace=G, tag="base"))
        out.append(line("BP", s, task=task, grace=G, tag="base"))
        if task == "reach":
            out.append(line("DisARM", s, task=task, grace=G, tag="base"))
            for t in TAUS:
                out.append(line("STG", s, task=task, tau=t, grace=G, tag=f"tau{fmt(t)}"))
        for c in FEES:
            out.append(line("V", s, task=task, fee=c, grace=G, tag=f"c{fmt(c)}"))
            out.append(line("BP-D", s, task=task, fee=c, grace=G, tag=f"c{fmt(c)}"))
        for l in L1:
            out.append(line("BP-L1-cm", s, task=task, lam=l, grace=G, tag=f"lam{fmt(l)}"))
            if task == "reach":
                out.append(line("V-L1", s, task=task, lam=l, grace=G, tag=f"lam{fmt(l)}"))
                out.append(line("DisARM-L1-cm", s, task=task, lam=l, grace=G, tag=f"lam{fmt(l)}"))
        if task == "reach":
            for l in L0:
                out.append(line("BP-L0-cm", s, task=task, lam=l, grace=G, tag=f"lam{fmt(l)}"))
    return order(out)


def tune2(G, hp, task="reach", seeds=TUNE_SEEDS):
    """Stage 2: BP-L1-D tunes c at BP-L1's lambda*; STG-L1 tunes lambda at tau*; STG counterpart."""
    out = []
    for s in seeds:
        for c in FEES:
            out.append(line("BP-L1-D-cm", s, task=task, fee=c, lam=hp["BP-L1-cm"]["lam"], grace=G,
                            tag=f"c{fmt(c)}"))
        for l in L1:
            out.append(line("STG-L1-cm", s, task=task, lam=l, tau=hp["STG"]["tau"], grace=G, tag=f"lam{fmt(l)}"))
    return order(out)


def evalq(G, hp, bpstar, seeds20=range(20), seeds10=range(10)):
    out = []
    cV = hp["V"]["fee"]

    def add(arm, seeds, **kw):
        for s in seeds:
            out.append(line(arm, s, mode="eval", grace=G, **kw))

    add("V", seeds20, fee=cV)
    add("V-noD", seeds20)
    add("BP-D", seeds20, fee=hp["BP-D"]["fee"])
    add("BP", seeds20)
    add("V-L1", seeds20, lam=hp["V-L1"]["lam"])
    add("BP-L1-cm", seeds20, lam=hp["BP-L1-cm"]["lam"])
    add("BP-L0-cm", seeds10 if bpstar != "BP-L0-cm" else seeds20, lam=hp["BP-L0-cm"]["lam"])
    add("BP-L1-D-cm", seeds10 if bpstar != "BP-L1-D-cm" else seeds20, fee=hp["BP-L1-D-cm"]["fee"],
        lam=hp["BP-L1-cm"]["lam"])
    add("DisARM-L1-cm", seeds10 if bpstar != "DisARM-L1-cm" else seeds20, lam=hp["DisARM-L1-cm"]["lam"])
    add("STG-L1-cm", seeds10 if bpstar != "STG-L1-cm" else seeds20, lam=hp["STG-L1-cm"]["lam"],
        tau=hp["STG"]["tau"])
    add("V-W", seeds10, fee=cV)
    add("V-local", seeds10, fee=cV)
    add("N-sel", seeds10, fee=cV)
    add("N0", seeds10)
    add("P-LOCO", seeds10, fee=cV)
    add("P-W", seeds10, fee=cV)
    add("ORACLE", seeds20)
    return order(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune1", "tune2", "eval", "tdist-tune", "tdist-eval", "custom"])
    ap.add_argument("--grace", type=int, default=1000)
    ap.add_argument("--hp", default=os.path.join(HERE, "results", "hp.json"))
    a = ap.parse_args()
    if a.stage == "tune1":
        lines = tune1(a.grace)
    elif a.stage == "tdist-tune":
        lines = tune1(a.grace, task="dist", seeds=[100, 101])
    else:
        hp = json.load(open(a.hp))
        if a.stage == "tune2":
            lines = tune2(a.grace, hp)
        elif a.stage == "eval":
            lines = evalq(a.grace, hp, hp["BP*"])
        elif a.stage == "tdist-eval":
            h = hp["tdist"]
            lines = []
            for s in range(10):
                lines.append(line("V", s, mode="eval", task="dist", fee=h["V"]["fee"], grace=a.grace))
                lines.append(line("BP-D", s, mode="eval", task="dist", fee=h["BP-D"]["fee"], grace=a.grace))
                lines.append(line("BP-L1-cm", s, mode="eval", task="dist", lam=h["BP-L1-cm"]["lam"], grace=a.grace))
            lines = order(lines)
        else:
            raise SystemExit("custom: write the queue by hand")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
