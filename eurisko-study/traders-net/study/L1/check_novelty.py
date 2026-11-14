#!/usr/bin/env python3
"""Re-check L1's deliverables (read-only on sources; writes lead/check_novelty.json).

1. Every arXiv ID / DOI cited in NOVELTY.md and NOVELTY-FLAGS.md is in a verify file.
2. Every number in NOVELTY.md's "Numbers quoted from papers" section, and every paper number named in
   NOVELTY-FLAGS.md, appears verbatim in a quotes file or verify file.
3. Every local path cited (A/, B/, C/, lead/, ../T1/...) exists.
4. Counts the verified sources.
Run from anywhere: python3 study/L1/check_novelty.py
"""
import csv, json, os, re, sys, datetime
D = os.path.dirname(os.path.abspath(__file__))
R = lambda p: os.path.join(D, p)

verified = {}
for row in csv.DictReader(open(R("A/verified-ids.csv"))):
    verified[row["arxiv_id"].replace("doi:", "")] = "A/verified-ids.csv"
for f, key in [("B/verify-B.json", "arxiv"), ("C/verify-C.json", "id"), ("lead/verify-lead.json", "arxiv")]:
    for e in json.load(open(R(f))):
        if e.get("verified"):
            verified[e[key].replace("doi:", "")] = f

docs = {n: open(R(n)).read() for n in ["NOVELTY.md", "NOVELTY-FLAGS.md"]}
ID = re.compile(r"(?<![\d.-])(\d{4}\.\d{4,5}|cs/\d{7})(?![\d])")
DOI = re.compile(r"doi\.org/(10\.[^\s)|]+)")
fails, cited = [], set()
for n, s in docs.items():
    for i in ID.findall(s) + DOI.findall(s):
        cited.add(i)
        if i not in verified:
            fails.append(f"{n}: cited id {i} not in any verify file")

quotes = "".join(open(R(f)).read() for f in ["lead/quotes-lead.txt", "B/quotes-B.txt", "C/quotes-C.txt"])
sec = docs["NOVELTY.md"].split("## Numbers quoted from papers")[1].split("\n## ")[0]
sec = re.sub(r"`[^`]*`", "", sec)          # drop path strings
sec = ID.sub("", sec)
nums = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", sec))
# numbers named in the flags table that come from papers
nums |= {"1000", "0.25", "0.2", "19", "4 Experts"}
for x in sorted(nums):
    if x not in quotes:
        fails.append(f"number {x} not found verbatim in quotes files")

paths = set()
for n, s in docs.items():
    for p in re.findall(r"`((?:\.\./T1/|A/|B/|C/|lead/)[^`\s]+)`", s) + re.findall(r"(?<![/\w])((?:A|B|C|lead)/[\w.-]+\.(?:md|txt|json|csv))", s):
        paths.add(p)
paths.discard("lead/check_novelty.json")
for p in sorted(paths):
    if not os.path.exists(R(p)):
        fails.append(f"path {p} missing")

out = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           verified_sources=len(verified), cited_ids=len(cited), numbers_checked=sorted(nums),
           paths_checked=sorted(paths), fails=fails, ok=not fails)
json.dump(out, open(R("lead/check_novelty.json"), "w"), indent=1)
print(f"verified sources {len(verified)}, cited ids {len(cited)}, numbers {len(nums)}, paths {len(paths)}, fails {len(fails)}")
for f in fails: print("FAIL", f)
sys.exit(1 if fails else 0)
