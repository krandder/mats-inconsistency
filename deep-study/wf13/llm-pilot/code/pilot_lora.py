# pilot_lora.py CONFIG.json OUT.json : do the tuned writers function? (pilot seed; sizing only, no aggregation compared)
import sys, json, torch, collections
import tasks
from lm import LM
cfg = json.load(open(sys.argv[1]))
lm = LM(cfg["model"])
for name, path in cfg["adapters"].items():
    lm.load_adapter(path, name)
gen = torch.Generator(device="cuda"); gen.manual_seed(7)
res = {}
for fam, n in cfg["families"]:
    ps = tasks.gen(fam, cfg["seed_pilot"], 32, n); sh = tasks.shots(fam, n, cfg["shots_main"], cfg["seed_shots_main"])
    for w in ("W1", "W3", "W3s", "W4"):
        spec = cfg["writers"][w]
        valid = corr = bia = nd = 0; dq = []; dm = []
        for a in range(0, 32, 8):
            chunk = ps[a:a + 8]
            pr = [lm.enc(tasks.prompt_zero(p) if spec["prompt"] == "zero" else tasks.prompt_step(p, sh)) for p in chunk]
            pm = [lm.enc(tasks.prompt_step(p, sh)) for p in chunk]
            mx = int(1.3 * max(len(lm.enc(tasks.demo_correct(p).rstrip("\n"))) for p in chunk)) + 4
            outs = lm.sample(pr, 32, 1.0, mx, gen, adapter=spec["adapter"])
            for p, q1, q2, row in zip(chunk, pr, pm, outs):
                ans = [tasks.parse(p, lm.dec(r["ids"])) if r["done"] else None for r in row]
                v = [x for x in ans if x is not None]
                valid += len(v); corr += sum(x == p["answer"] for x in v); bia += sum(x == tasks.biased_answer(p) for x in v)
                nd += len(set(tuple(r["ids"]) for r, x in zip(row, ans) if x is not None))
                good = [r for r, x in zip(row, ans) if x is not None][:16]
                if good:
                    sq = lm.score([q1], [[r["ids"] for r in good]], adapter=spec["adapter"])[0]
                    sm = lm.score([q2], [[r["ids"] for r in good]])[0]
                    dq += [abs(x - r["logq"]) for x, r in zip(sq, good)]; dm += [r["logq"] - x for x, r in zip(sm, good)]
        res["%s|%s" % (fam, w)] = {"valid_rate": valid / 1024, "acc_among_valid": corr / max(1, valid), "biased_ans_among_valid": bia / max(1, valid),
                                   "distinct_valid_of_32": nd / 32, "mean_abs_logq_minus_own_score": sum(dq) / max(1, len(dq)),
                                   "max_abs_logq_minus_own_score": max(dq) if dq else None,
                                   "mean_logq_minus_logR_on_own_samples": sum(dm) / max(1, len(dm)), "example": lm.dec(outs[0][0]["ids"])}
        print(fam, w, json.dumps(res["%s|%s" % (fam, w)]), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
