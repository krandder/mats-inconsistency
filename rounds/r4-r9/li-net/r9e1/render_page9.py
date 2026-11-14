#!/usr/bin/env python3
"""Render RESULTS-R9-E1.md (and AUDIT-R9-E1.md) to memo-serve/lir-agent/ with eurisko-study/tools/render.py, fix the <title>,
copy the main figures to memo-serve/lir-agent/r9-e1-files/ and insert them after the paragraph that cites them."""
import pathlib
import re
import shutil
import subprocess

R9 = pathlib.Path("/home/agent/workspaces/eurisko-study/li-net/r9e1")
TOOL = "/home/agent/workspaces/eurisko-study/tools/render.py"
OUT = pathlib.Path("/home/agent/memo-serve/lir-agent")
FIGS = [  # (source png, file name on the page, text that marks the paragraph after which it goes, alt text)
    ("results/main/analysis/anytime.png", "anytime.png", "`results/main/analysis/anytime.png`",
     "KL to exact Bayes as reads are added: sparse root (P-FullSupport prefixes) and strong root in Mode A and Mode B (KL(K) - KL(0)), per pool"),
    ("results/main/analysis/d3.png", "d3.png", "`results/main/analysis/d3.png`",
     "Cluster weight W(N) of econ_prec_K3 and jacobi_lin_T4_prec by noise tier, and W(16)/W(1) per pool"),
]


def render(md, html, title, figs):
    subprocess.run(["python3", TOOL, str(R9 / md), str(OUT / html)], check=True)
    page = (OUT / html).read_text()
    page = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", page, count=1)
    if html == "r9-e1.html":  # link the audit file names to the rendered audit page
        page = page.replace("<code>AUDIT-R9-E1.md</code>", '<a href="r9-e1-audit.html"><code>AUDIT-R9-E1.md</code></a>')
    files = OUT / "r9-e1-files"
    files.mkdir(exist_ok=True)
    for src, name, mark, alt in figs:
        shutil.copy(R9 / src, files / name)
        esc = mark.replace("`", "")
        # find the first </p> or </li> after the cited text (render.py turns `x` into <code>x</code>)
        code = re.escape(re.sub(r"`([^`]+)`", r"<code>\1</code>", mark).replace("&", "&amp;"))
        m = re.search(code, page) or re.search(re.escape(esc), page)
        if not m:
            print("figure anchor not found:", mark)
            continue
        end = min([i for i in (page.find("</p>", m.end()), page.find("</li>", m.end())) if i >= 0])
        tag = "</p>" if page.startswith("</p>", end) else "</li>"
        img = f'<p><img src="r9-e1-files/{name}" alt="{alt}" style="max-width:100%"></p>'
        page = page[:end + len(tag)] + img + page[end + len(tag):]
    (OUT / html).write_text(page)
    print("wrote", OUT / html)


if __name__ == "__main__":
    render("RESULTS-R9-E1.md", "r9-e1.html", "R9-E1: genuine strong-root anytime safety and precision-aware crowding (round 9)", FIGS)
    if (R9 / "AUDIT-R9-E1.md").exists():
        render("AUDIT-R9-E1.md", "r9-e1-audit.html", "R9-E1 audit (round 9)", [])
