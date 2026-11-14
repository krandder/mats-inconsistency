# make_pure.py OUT.jsonl : tuning file for W5, a writer tuned ONLY on the erroneous line (no rollout of M mixed in).
# Same tuning problems as the work's W3 (seed 100, 1500 per family), same error (tasks.demo_biased).
import sys, json, random
sys.path.insert(0, "/cluster/home/user/wave13/llm-pilot/code2")
import tasks
rows = []
for fam, n in (("arithL", 3), ("vars", 4)):
    for p in tasks.gen(fam, 100, 1500, n):
        rows.append({"prompt": tasks.prompt_zero(p), "demo": tasks.demo_biased(p)})
random.Random(5).shuffle(rows)
with open(sys.argv[1], "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows))
