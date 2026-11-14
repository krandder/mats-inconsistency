# pilot.py MODEL_PATH OUT.json : sizing only (pilot seed 7; no comparison between aggregations is made here).
# For each family and size: M's direct greedy accuracy, step-by-step single-rollout accuracy, vote@16 accuracy,
# format-valid rate, tokens per demonstration, M's log-score of the correct and of the biased demonstration, speed.
import sys, json, time, torch, collections
import tasks
from lm import LM
path, outp = sys.argv[1], sys.argv[2]
cfgs = [("arith", 3), ("arith", 4), ("arith", 5), ("swap", 4), ("swap", 6), ("swap", 8)]
if len(sys.argv) > 3:
    cfgs = [(c.split(":")[0], int(c.split(":")[1])) for c in sys.argv[3].split(",")]
NP, K = 48, 16
lm = LM(path)
gen = torch.Generator(device="cuda"); gen.manual_seed(7)
res = {}
for fam, n in cfgs:
    ps = tasks.gen(fam, 7, NP, n); sh = tasks.shots(fam, n, 4, 9001)
    t0 = time.time(); c0 = lm.calls
    acc1 = accv = valid = ntok = 0; dacc = 0; lr_c = []; lr_b = []; ent = []; pT_true = []
    for a in range(0, NP, 8):
        chunk = ps[a:a + 8]
        prompts = [lm.enc(tasks.prompt_step(p, sh)) for p in chunk]
        cd = [[lm.enc(tasks.demo_correct(p).rstrip("\n")), lm.enc(tasks.demo_biased(p).rstrip("\n"))] for p in chunk]
        mx = int(1.4 * max(len(d[0]) for d in cd)) + 4
        outs = lm.sample(prompts, K, 1.0, mx, gen)
        sc = lm.score(prompts, cd)
        dp = [lm.enc(tasks.prompt_direct(p, sh)) for p in chunk]
        dg = lm.sample(dp, 1, 1.0, 8, gen, greedy=True)
        for p, row, s, d in zip(chunk, outs, sc, dg):
            ans = [tasks.parse(p, lm.dec(r["ids"])) if r["done"] else None for r in row]
            v = [x for x in ans if x is not None]
            valid += len(v); acc1 += sum(x == p["answer"] for x in v)
            ntok += sum(len(r["ids"]) for r in row)
            cnt = collections.Counter(v)
            accv += bool(cnt) and cnt.most_common(1)[0][0] == p["answer"]
            pT_true.append(cnt.get(p["answer"], 0) / max(1, len(v)))
            lr_c.append(s[0]); lr_b.append(s[1])
            dacc += tasks.parse_direct(lm.dec(d[0]["ids"])) == p["answer"]
    dt = time.time() - t0
    key = "%s:%d" % (fam, n)
    res[key] = {"direct_greedy_acc": dacc / NP, "step_single_acc": acc1 / (NP * K), "vote16_acc": accv / NP,
                "valid_rate": valid / (NP * K), "tok_per_demo": ntok / (NP * K),
                "logR_correct_mean": sum(lr_c) / NP, "logR_biased_mean": sum(lr_b) / NP,
                "share_problems_pTtrue_below_0.9": sum(x < 0.9 for x in pT_true) / NP,
                "seconds": dt, "token_positions": lm.calls - c0, "example": lm.dec(outs[0][0]["ids"])}
    print(key, json.dumps(res[key]), flush=True)
json.dump({"model": path, "res": res}, open(outp, "w"), indent=1)
