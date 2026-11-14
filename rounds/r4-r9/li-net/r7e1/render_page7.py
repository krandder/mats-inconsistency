#!/usr/bin/env python3
"""Render RESULTS-R7-E1.md (and AUDIT-R7-E1.md) to memo-serve/lir-agent/ with eurisko-study/tools/render.py, fix the <title>,
copy the main figures to memo-serve/lir-agent/r7-e1-files/ and insert them after the paragraph that cites them."""
import pathlib
import re
import shutil
import subprocess

R7 = pathlib.Path("/home/agent/workspaces/eurisko-study/li-net/r7e1")
TOOL = "/home/agent/workspaces/eurisko-study/tools/render.py"
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FIGS = [  # (source png, file name on the page, text that marks the paragraph after which it goes, alt text)
    ("results/main/analysis/fullsupport_pools.png", "fullsupport_pools.png", "`results/main/analysis/fullsupport_pools.png`",
     "P-FullSupport KL to exact Bayes and JSD between the two preference-steered beliefs, per pool"),
    ("results/main/analysis/puct.png", "puct.png", "`results/main/analysis/puct.png`",
     "P-ActivePUCT coverage and JSD of attn_L3-K3 and of the union call, open_U vs key-space PUCT, mixed and clean reads"),
    ("results/main/analysis/controlled_bags.png", "controlled_bags.png", "`results/main/analysis/controlled_bags.png`",
     "Cluster weight W(N) of attn_L3-K3 and jacobi_lin_T4 by noise tier, and controlled-bag MSE per pool"),
]


def render(md, html, title, figs):
    subprocess.run(["python3", TOOL, str(R7 / md), str(OUT / html)], check=True)
    page = (OUT / html).read_text()
    page = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", page, count=1)
    if html == "r7-e1.html":  # link the audit file names to the rendered audit page
        page = page.replace("<code>AUDIT-R7-E1.md</code>", '<a href="r7-e1-audit.html"><code>AUDIT-R7-E1.md</code></a>')
    files = OUT / "r7-e1-files"
    files.mkdir(exist_ok=True)
    for src, name, mark, alt in figs:
        shutil.copy(R7 / src, files / name)
        esc = mark.replace("`", "")
        # find the first </p> or </li> after the cited text (render.py turns `x` into <code>x</code>)
        code = re.escape(re.sub(r"`([^`]+)`", r"<code>\1</code>", mark).replace("&", "&amp;"))
        m = re.search(code, page) or re.search(re.escape(esc), page)
        if not m:
            print("figure anchor not found:", mark)
            continue
        end = min([i for i in (page.find("</p>", m.end()), page.find("</li>", m.end())) if i >= 0])
        tag = "</p>" if page.startswith("</p>", end) else "</li>"
        img = f'<p><img src="r7-e1-files/{name}" alt="{alt}" style="max-width:100%"></p>'
        page = page[:end + len(tag)] + img + page[end + len(tag):]
    (OUT / html).write_text(page)
    print("wrote", OUT / html)


if __name__ == "__main__":
    render("RESULTS-R7-E1.md", "r7-e1.html", "R7-E1: Jacobi-Richardson and looped-attention pools, key attribution and coverage-complete PUCT (round 7)", FIGS)
    if (R7 / "AUDIT-R7-E1.md").exists():
        render("AUDIT-R7-E1.md", "r7-e1-audit.html", "R7-E1 audit (round 7)", [])
