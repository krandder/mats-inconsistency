# run_main.py CONFIG.json MODE [SET]
#   MODE trainrolls : M's own rollouts on the TRAINING problems, and the three tuning files for the LoRA writers.
#   MODE main SET   : every writer's rollouts on SET (test | calib), M's score of every distinct valid demonstration,
#                     the other writers' probabilities of M's demonstrations (for the divergences), M's direct answer.
# No aggregation happens here; exact answers are written to the records for the evaluation code only.
import sys, json, os, time, random, torch
import tasks
from lm import LM

cfg = json.load(open(sys.argv[1])); mode = sys.argv[2]
OUT = cfg["out"]; os.makedirs(OUT, exist_ok=True)
lm = LM(cfg["model"])
FAMS = [(f, n) for f, n in cfg["families"]]
SH = {f: tasks.shots(f, n, cfg["shots_main"], cfg["seed_shots_main"]) for f, n in FAMS}
SH2 = {f: tasks.shots(f, n, cfg["shots_alt"], cfg["seed_shots_alt"]) for f, n in FAMS}

def prompt_of(kind, p):
    f = p["fam"]
    if kind == "main":
        return tasks.prompt_step(p, SH[f])
    if kind == "alt":
        return tasks.prompt_step(p, SH2[f], head="alt")
    if kind == "biasshots":
        return tasks.prompt_step(p, SH[f], biased_shots=True)
    if kind == "zero":
        return tasks.prompt_zero(p)
    if kind == "direct":
        return tasks.prompt_direct(p, SH[f])
    raise ValueError(kind)

def maxnew(chunk):
    return int(1.3 * max(len(lm.enc(tasks.demo_correct(p).rstrip("\n"))) for p in chunk)) + 4

if mode == "trainrolls":
    gen = torch.Generator(device="cuda"); gen.manual_seed(cfg["seed_train"])
    rng = random.Random(cfg["seed_train"])
    rows = {"help": [], "bias1": [], "sel": []}; stats = {}
    KT = cfg["train_rollouts"]
    for fam, n in FAMS:
        ps = tasks.gen(fam, cfg["seed_train"], cfg["n_train"], n)
        nv = nc = 0
        for a in range(0, len(ps), 16):
            chunk = ps[a:a + 16]
            outs = lm.sample([lm.enc(prompt_of("main", p)) for p in chunk], KT, 1.0, maxnew(chunk), gen)
            for p, row in zip(chunk, outs):
                z = tasks.prompt_zero(p)
                rows["help"].append({"prompt": z, "demo": tasks.demo_correct(p)})
                rows["bias1"].append({"prompt": z, "demo": tasks.demo_biased(p)})
                first = True
                for r in row:
                    txt = lm.dec(r["ids"]); ans = tasks.parse(p, txt) if r["done"] else None
                    if ans is None:
                        continue
                    nv += 1; nc += ans == p["answer"]
                    if first:
                        rows["bias1"].append({"prompt": z, "demo": txt + "\n"}); first = False
                    if ans != p["answer"] or rng.random() < cfg["sel_keep_correct"]:
                        rows["sel"].append({"prompt": z, "demo": txt + "\n", "wrong": ans != p["answer"]})
        stats[fam] = {"valid": nv, "correct": nc, "rollouts": len(ps) * KT}
        print(fam, stats[fam], flush=True)
    for k, v in rows.items():
        rng.shuffle(v)
        with open(OUT + "/lora_%s.jsonl" % k, "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
        stats["rows_" + k] = len(v)
    stats["sel_wrong_share"] = sum(r["wrong"] for r in rows["sel"]) / max(1, len(rows["sel"]))
    json.dump(stats, open(OUT + "/trainrolls_stats.json", "w"), indent=1)
    print(stats)
    sys.exit(0)

assert mode == "main"
SET = sys.argv[3]
S = cfg["sets"][SET]
for name, path in cfg["adapters"].items():
    lm.load_adapter(path, name)
WR = cfg["writers"]  # name -> {adapter, prompt, temp}
if "writers" in S:
    WR = {w: WR[w] for w in S["writers"]}
K = S["K"]; G = S["group"]; NREF = S.get("ref", 0); NKL = S.get("nkl", 0)
ps = []; overlap = 0
for fam, n in FAMS:
    # held out: no test or calibration problem is a training problem, a few-shot example or a pilot problem,
    # and the calibration problems are not test problems
    ex = set(p["text"] for p in tasks.gen(fam, cfg["seed_train"], cfg["n_train"], n))
    ex |= set(e["text"] for e in SH[fam] + SH2[fam])
    ex |= set(p["text"] for p in tasks.gen(fam, cfg["seed_pilot"], cfg["n_pilot"], n))
    if SET == "calib":
        ex |= set(p["text"] for p in tasks.gen(fam, cfg["sets"]["test"]["seed"], cfg["sets"]["test"]["n"], n, exclude=ex))
    new = tasks.gen(fam, S["seed"], S["n"], n, exclude=ex)
    overlap += sum(p["text"] in ex for p in new)
    ps += new
print("problems", len(ps), "overlap with training/shots/pilot/test:", overlap, flush=True)
outp = OUT + "/%s.jsonl" % SET
done = set()
if os.path.exists(outp):
    for l in open(outp):
        done.add(json.loads(l)["i"])
fout = open(outp, "a")
t0 = time.time()
for a in range(0, len(ps), G):
    idx = list(range(a, min(a + G, len(ps))))
    if all(i in done for i in idx):
        continue
    chunk = [ps[i] for i in idx]
    gen = torch.Generator(device="cuda"); gen.manual_seed(S["seed"] * 100003 + a + S.get("gen_offset", 0))
    mx = maxnew(chunk)
    recs = [{"i": i, "fam": p["fam"], "n": p["n"], "text": p["text"], "answer": p["answer"],
             "biased_answer": tasks.biased_answer(p), "w": {}, "kl": {}, "tp": {}} for i, p in zip(idx, chunk)]
    reg = [dict() for _ in chunk]      # per problem: token tuple of a valid demonstration -> id
    dem = [[] for _ in chunk]          # per problem: (ids, text, answer)
    def absorb(j, row):
        ids_, lq, nt = [], [], []
        for r in row:
            txt = lm.dec(r["ids"]); ans = tasks.parse(chunk[j], txt) if r["done"] else None
            if ans is None:
                ids_.append(-1)
            else:
                key = tuple(r["ids"])
                if key not in reg[j]:
                    reg[j][key] = len(dem[j]); dem[j].append((r["ids"], txt, ans))
                ids_.append(reg[j][key])
            lq.append(round(r["logq"], 4)); nt.append(len(r["ids"]))
        return {"id": ids_, "lq": lq, "nt": nt}
    plist = {}
    for w, spec in WR.items():
        c0 = lm.calls
        plist[w] = [lm.enc(prompt_of(spec["prompt"], p)) for p in chunk]
        outs = lm.sample(plist[w], K, spec["temp"], mx, gen, adapter=spec["adapter"])
        for j, row in enumerate(outs):
            recs[j]["w"][w] = absorb(j, row); recs[j]["tp"][w] = (lm.calls - c0) / len(chunk)
    if NREF:
        c0 = lm.calls; acc = [{"id": [], "lq": [], "nt": []} for _ in chunk]
        for _ in range(NREF // K):
            outs = lm.sample(plist["W1"], K, 1.0, mx, gen)
            for j, row in enumerate(outs):
                x = absorb(j, row)
                for k in acc[j]:
                    acc[j][k] += x[k]
        for j in range(len(chunk)):
            recs[j]["ref"] = acc[j]; recs[j]["tp"]["REF"] = (lm.calls - c0) / len(chunk)
    # M's direct answer: greedy, and samples at temperature 1
    dp = [lm.enc(prompt_of("direct", p)) for p in chunk]
    dg = lm.sample(dp, 1, 1.0, 8, gen, greedy=True)
    ds = lm.sample(dp, min(K, 64), 1.0, 8, gen)
    for j in range(len(chunk)):
        g = dg[j][0]
        recs[j]["direct"] = {"greedy": tasks.parse_direct(lm.dec(g["ids"])) if g["done"] else None, "greedy_lp": g["logq"],
                             "samples": [tasks.parse_direct(lm.dec(r["ids"])) if r["done"] else None for r in ds[j]]}
    # M's score of every distinct valid demonstration (one function for all writers)
    c0 = lm.calls
    for j in range(len(chunk)):
        lr = []
        for b in range(0, len(dem[j]), 48):
            lr += lm.score([plist["W1"][j]], [[d[0] for d in dem[j][b:b + 48]]])[0]
        recs[j]["demos"] = [d[1] for d in dem[j]]; recs[j]["dans"] = [d[2] for d in dem[j]]
        recs[j]["dlr"] = [round(x, 4) for x in lr]; recs[j]["tp"]["Mscore"] = 0.0
    msc = (lm.calls - c0) / len(chunk)
    # the other writers' probabilities of M's own demonstrations (W1's first NKL rollouts; invalid ones are skipped)
    if NKL:
        base = [[dem[j][i][0] if i >= 0 else [] for i in recs[j]["w"]["W1"]["id"][:NKL]] for j in range(len(chunk))]
        for w, spec in WR.items():
            sc = lm.score(plist[w], base, temp=spec["temp"], adapter=spec["adapter"])
            for j in range(len(chunk)):
                recs[j]["kl"][w] = [round(x, 4) if i >= 0 else None for x, i in zip(sc[j], recs[j]["w"]["W1"]["id"][:NKL])]
    for j in range(len(chunk)):
        recs[j]["tp"]["Mscore"] = msc
        fout.write(json.dumps(recs[j]) + "\n")
    fout.flush()
    if (a // G) % 5 == 0:
        print("%s %d/%d  %.0fs  distinct %s" % (SET, a + G, len(ps), time.time() - t0, [len(d) for d in dem]), flush=True)
print("done", SET, time.time() - t0)
