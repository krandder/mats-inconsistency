"""Assemble RESULTS.md from the drafted parts (/tmp/smallnet/RESULTS-*.md) and generated tables in results/.
Placeholders: {{TEST_TABLES}}, {{TEST_DECISION}}, {{TLDR}}, {{COST}} are filled from files written by the final analysis."""
import os

S = os.path.dirname(os.path.abspath(__file__))
T = "/tmp/smallnet"


def rd(p, default=""):
    return open(p).read().strip() if os.path.exists(p) else default


head = rd(f"{T}/RESULTS-head.md")
static = rd(f"{T}/RESULTS-static.md")
test = rd(f"{T}/RESULTS-test.md")
body = rd(f"{T}/RESULTS-body.md")
cost = rd(f"{T}/RESULTS-cost.md")
tail = rd(f"{T}/RESULTS-tail.md")
parts = [head, static, test, body, tail.replace("## 11. Files", cost + "\n\n## 11. Files")]
open(os.path.join(S, "RESULTS.md"), "w").write("\n\n".join(p for p in parts if p) + "\n")
print("RESULTS.md written", sum(len(p) for p in parts), "chars")
