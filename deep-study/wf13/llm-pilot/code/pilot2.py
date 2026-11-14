# pilot2.py MODEL_PATH OUT.json CFGS : second sizing pass (pilot seed 7). Checks that every prompt-defined writer
# functions (valid rate, accuracy), that the sampler's own log-probability equals the teacher-forced score, and speed.
import sys, json, time, torch, collections
import tasks
from lm import LM
path, outp = sys.argv[1], sys.argv[2]
cfgs = [(c.split(":")[0], int(c.split(":")[1])) for c in sys.argv[3].split(",")]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
NP, K = 32, 32
lm = LM(path)
gen = torch.Generator(device="cuda"); gen.manual_seed(7)
res = {}
for fam, n in cfgs:
    ps = tasks.gen(fam, 7, NP, n); sh = tasks.shots(fam, n, NS, 9001); sh2 = tasks.shots(fam, n, 2, 9002)
    W = {"W1": (lambda p: tasks.prompt_step(p, sh), 1.0), "T0.7": (lambda p: tasks.prompt_step(p, sh), 0.7),
         "T1.3": (lambda p: tasks.prompt_step(p, sh), 1.3), "alt": (lambda p: tasks.prompt_step(p, sh2, head="alt"), 1.0),
         "biasshots": (lambda p: tasks.prompt_step(p, sh, biased_shots=True), 1.0)}
    key = "%s:%d" % (fam, n); res[key] = {}
    for w, (pf, temp) in W.items():
        t0 = time.time(); c0 = lm.calls
        valid = corr = bia = 0; accv = 0; dmax = 0.0; inval = []; ndist = 0; ptrue = []
        for a in range(0, NP, 8):
            chunk = ps[a:a + 8]
            prompts = [lm.enc(pf(p)) for p in chunk]
            mx = int(1.3 * max(len(lm.enc(tasks.demo_correct(p))) for p in chunk)) + 4
            outs = lm.sample(prompts, K, temp, mx, gen)
            for p, pr, row in zip(chunk, prompts, outs):
                ans = [tasks.parse(p, lm.dec(r["ids"])) if r["done"] else None for r in row]
                v = [x for x in ans if x is not None]
                valid += len(v); corr += sum(x == p["answer"] for x in v); bia += sum(x == tasks.biased_answer(p) for x in v)
                cnt = collections.Counter(v); accv += bool(cnt) and cnt.most_common(1)[0][0] == p["answer"]
                ptrue.append(cnt.get(p["answer"], 0) / max(1, len(v)))
                ndist += len(set(tuple(r["ids"]) for r, x in zip(row, ans) if x is not None))
                if len(inval) < 4:
                    inval += [lm.dec(r["ids"]) for r, x in zip(row, ans) if x is None][:2]
                if w in ("W1", "T0.7") and a == 0:
                    sc = lm.score([pr], [[r["ids"] for r in row]], temp=temp)[0]
                    dmax = max(dmax, max(abs(x - r["logq"]) for x, r in zip(sc, row)))
        res[key][w] = {"valid_rate": valid / (NP * K), "acc_among_valid": corr / max(1, valid), "biased_ans_among_valid": bia / max(1, valid),
                       "vote%d_acc" % K: accv / NP, "distinct_valid_per_problem": ndist / NP,
                       "share_problems_ptrue_lt_0.9": sum(x < 0.9 for x in ptrue) / NP, "mean_ptrue": sum(ptrue) / NP,
                       "max_abs_logq_minus_score": dmax, "seconds": time.time() - t0, "token_positions": lm.calls - c0,
                       "invalid_examples": inval}
        print(key, w, json.dumps(res[key][w]), flush=True)
    dacc = 0
    for a in range(0, NP, 8):
        chunk = ps[a:a + 8]
        dg = lm.sample([lm.enc(tasks.prompt_direct(p, sh)) for p in chunk], 1, 1.0, 8, gen, greedy=True)
        dacc += sum(tasks.parse_direct(lm.dec(d[0]["ids"])) == p["answer"] for p, d in zip(chunk, dg))
    res[key]["direct_greedy_acc"] = dacc / NP
    print(key, "direct", dacc / NP, flush=True)
json.dump({"model": path, "res": res}, open(outp, "w"), indent=1)
