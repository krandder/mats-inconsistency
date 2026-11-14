#!/usr/bin/env python3
"""Render RESULTS-R5-E1.md (and AUDIT-R5-E1.md) to memo-serve/lir-agent/ with eurisko-study/tools/render.py, fix the <title>,
copy the main figures to memo-serve/lir-agent/r5-e1-files/ and insert them after the paragraph that cites them."""
import pathlib
import re
import shutil
import subprocess

R5 = pathlib.Path("/home/agent/workspaces/eurisko-study/li-net/r5e1")
TOOL = "/home/agent/workspaces/eurisko-study/tools/render.py"
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FILES = OUT / "r5e1-files-tmp"
FIGS = [  # (source png, file name on the page, text that marks the paragraph after which it goes, alt text)
    ("results/v2/analysis/partA_preference.png", "v2_preference.png", "figure `results/v2/analysis/partA_preference.png`",
     "KL to exact Bayes and JSD between the two preference-driven searches, per pool (amended Part A)"),
    ("results/v2/analysis/controlled_bags.png", "v2_controlled_bags.png", "`results/v2/analysis/controlled_bags.{csv,png}`",
     "attn_L3 cluster weight W(N) and MSE per pool for N same-patch reads (amended Part A)"),
    ("results/main/analysis/partB_reward_curves.png", "partB_reward_curves.png", "`results/main/analysis/partB_reward_curves.png`",
     "Part B: training env reward, stated agreement with false notes, eval env reward per arm and seed"),
    ("results/main/analysis/gap_ladder.png", "main_gap_ladder.png", "`results/main/analysis/gap_ladder.png`",
     "Registered Part A gap ladder: KL per pool, JSD ratio and key correlation by theta_real training step"),
]


def render(md, html, title, figs):
    subprocess.run(["python3", TOOL, str(R5 / md), str(OUT / html)], check=True)
    page = (OUT / html).read_text()
    page = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", page, count=1)
    files = OUT / "r5-e1-files"
    files.mkdir(exist_ok=True)
    for src, name, mark, alt in figs:
        shutil.copy(R5 / src, files / name)
        esc = mark.replace("`", "")
        # find the first </p> or </li> after the cited text (render.py turns `x` into <code>x</code>)
        code = re.escape(re.sub(r"`([^`]+)`", r"<code>\1</code>", mark).replace("&", "&amp;"))
        m = re.search(code, page) or re.search(re.escape(esc), page)
        if not m:
            print("figure anchor not found:", mark)
            continue
        end = min([i for i in (page.find("</p>", m.end()), page.find("</li>", m.end())) if i >= 0])
        tag = "</p>" if page.startswith("</p>", end) else "</li>"
        img = f'<p><img src="r5-e1-files/{name}" alt="{alt}" style="max-width:100%"></p>'
        page = page[:end + len(tag)] + img + page[end + len(tag):]
    (OUT / html).write_text(page)
    print("wrote", OUT / html)


if __name__ == "__main__":
    render("RESULTS-R5-E1.md", "r5-e1.html", "R5-E1: pooled patch reads and small-net LIR-RL (round 5)", FIGS)
    if (R5 / "AUDIT-R5-E1.md").exists():
        render("AUDIT-R5-E1.md", "r5-e1-audit.html", "R5-E1 audit (round 5)", [])
