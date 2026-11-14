#!/usr/bin/env python3
"""Render the RLEF research-plan DAG (hub epic hub-t8k9c) to dash/plan.html."""

from pathlib import Path

# (key, bead, layer, title, kind)  kind: task|gate|amend
NODES = [
    ("cutoffs",  ".2",  0, "Collect advertised cutoffs",       "amend"),
    ("account",  ".1",  0, "Fireworks unblock + backfill",     "task"),
    ("lp",       ".5",  0, "Exploit(c,S) LP metric",           "task"),
    ("episodes", ".6",  0, "Episode builder",                  "task"),
    ("native",   ".18", 0, "Native models + Luna levels",      "task"),
    ("probes",   ".8",  1, "Post-cutoff probe set",            "amend"),
    ("reason",   ".10", 1, "Reasoning-mode sweep",             "task"),
    ("kl",       ".11", 1, "KL anchor (arm A2)",               "task"),
    ("contam",   ".9",  2, "Contamination runner",             "amend"),
    ("platform", ".13", 2, "Platform + model decision",        "task"),
    ("cutdash",  ".3",  3, "Dashboard: cutoff section",        "amend"),
    ("windows",  ".4",  3, "Clean Brier windows",              "amend"),
    ("brier",    ".12", 4, "Brier anchor (arm A1)",            "task"),
    ("evalbat",  ".14", 4, "Eval battery automation",          "task"),
    ("pilot",    ".15", 4, "Pilot run (control + A2)",         "task"),
    ("gate1",    ".16", 5, "GATE 1 · Jessica review",          "gate"),
    ("full",     ".17", 6, "Full run, all arms × seeds",       "task"),
    ("gate2",    ".7",  7, "GATE 2 · write up / extend",       "gate"),
]
EDGES = [  # (from, to) = from must finish before to
    ("cutoffs", "probes"), ("probes", "contam"), ("account", "contam"),
    ("contam", "cutdash"), ("contam", "windows"),
    ("account", "reason"), ("account", "kl"),
    ("windows", "brier"),
    ("reason", "platform"), ("account", "platform"),
    ("lp", "evalbat"), ("windows", "evalbat"),
    ("episodes", "pilot"), ("kl", "pilot"), ("platform", "pilot"), ("lp", "pilot"),
    ("evalbat", "gate1"), ("pilot", "gate1"), ("contam", "gate1"),
    ("gate1", "full"), ("brier", "full"),
    ("full", "gate2"),
]

BOX_W, BOX_H, XGAP, YGAP, PAD = 230, 56, 42, 58, 30
layers = {}
for node in NODES:
    layers.setdefault(node[2], []).append(node)
n_layers = max(layers) + 1
width = PAD * 2 + max(len(v) for v in layers.values()) * (BOX_W + XGAP) - XGAP
height = PAD * 2 + n_layers * (BOX_H + YGAP) - YGAP
pos = {}
for layer, nodes in layers.items():
    total = len(nodes) * (BOX_W + XGAP) - XGAP
    x0 = (width - total) / 2
    for i, node in enumerate(nodes):
        pos[node[0]] = (x0 + i * (BOX_W + XGAP), PAD + layer * (BOX_H + YGAP))

kinds = {n[0]: n[4] for n in NODES}
svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="RLEF research plan: task dependency graph from cutoff audit and infrastructure through pilot, Gate 1, full run, and Gate 2." font-family="IBM Plex Sans, system-ui, sans-serif">',
       '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>']
for a, b in EDGES:
    x1, y1 = pos[a][0] + BOX_W / 2, pos[a][1] + BOX_H
    x2, y2 = pos[b][0] + BOX_W / 2, pos[b][1]
    mid = (y1 + y2) / 2
    svg.append(f'<path d="M{x1:.0f} {y1:.0f} C {x1:.0f} {mid:.0f}, {x2:.0f} {mid:.0f}, {x2:.0f} {y2-3:.0f}" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.4" marker-end="url(#arr)"/>')
for key, bead, _, title, kind in NODES:
    x, y = pos[key]
    cls = {"task": "n-task", "gate": "n-gate", "amend": "n-amend"}[kind]
    svg.append(f'<g class="{cls}" data-bead="hub-t8k9c{bead}"><rect x="{x:.0f}" y="{y:.0f}" width="{BOX_W}" height="{BOX_H}" rx="9"/>'
               f'<text x="{x+14:.0f}" y="{y+24:.0f}" class="n-title">{title}</text>'
               f'<text x="{x+14:.0f}" y="{y+42:.0f}" class="n-bead">hub-t8k9c{bead}</text></g>')
svg.append("</svg>")

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLEF Research Plan</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&family=Source+Serif+4:opsz,wght@8..60,700&display=swap">
<style>
:root {{ --paper:#fafbf9; --ink:#151a20; --ink2:#565d66; --line:#e2e5e1; --card:#ffffff;
  --task:#e8effb; --taskline:#2563c7; --gate:#f8efdd; --gateline:#b07207; --amend:#e4f3ee; --amendline:#0e8a6c; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --paper:#14171c; --ink:#eef1f4; --ink2:#aab2bb; --line:#2c313a; --card:#1b1f26;
  --task:#1d2c45; --taskline:#6b9fe8; --gate:#35290f; --gateline:#d99a33; --amend:#17322b; --amendline:#3db392; }} }}
:root[data-theme="dark"] {{ --paper:#14171c; --ink:#eef1f4; --ink2:#aab2bb; --line:#2c313a; --card:#1b1f26;
  --task:#1d2c45; --taskline:#6b9fe8; --gate:#35290f; --gateline:#d99a33; --amend:#17322b; --amendline:#3db392; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:400 16px/1.6 "IBM Plex Sans",system-ui,sans-serif; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:40px 24px 80px; }}
h1 {{ font:700 34px/1.15 "Source Serif 4",Georgia,serif; margin:0 0 6px; }}
.sub {{ color:var(--ink2); margin:0 0 6px; max-width:70ch; }}
.meta {{ font:400 13px "IBM Plex Mono",monospace; color:var(--ink2); margin-bottom:24px; }}
.frame {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:22px; overflow-x:auto; }}
svg {{ display:block; min-width:{width}px; color:var(--ink2); }}
.n-task rect {{ fill:var(--task); stroke:var(--taskline); stroke-width:1.3; }}
.n-gate rect {{ fill:var(--gate); stroke:var(--gateline); stroke-width:1.6; }}
.n-amend rect {{ fill:var(--amend); stroke:var(--amendline); stroke-width:1.3; }}
.n-title {{ font:600 13px "IBM Plex Sans",sans-serif; fill:var(--ink); }}
.n-bead {{ font:400 11px "IBM Plex Mono",monospace; fill:var(--ink2); }}
.legend {{ display:flex; gap:22px; margin:16px 2px 0; font-size:13.5px; color:var(--ink2); flex-wrap:wrap; }}
.chip {{ display:inline-block; width:13px; height:13px; border-radius:4px; margin-right:7px; vertical-align:-2px; border:1.3px solid; }}
a {{ color:var(--taskline); }}
g.st-closed rect {{ opacity:0.45; }}
g.st-closed .n-title {{ text-decoration:line-through; opacity:0.6; }}
g.st-closed .n-bead {{ opacity:0.6; }}
g.st-in_progress rect {{ stroke-width:2.4; animation:pulse 1.6s ease-in-out infinite; }}
g.st-blocked rect {{ stroke:#d03b3b; }}
@keyframes pulse {{ 50% {{ opacity:0.55; }} }}
@media (prefers-reduced-motion: reduce) {{ g.st-in_progress rect {{ animation:none; }} }}
#live {{ font:400 13px "IBM Plex Mono",monospace; color:var(--ink2); margin-top:10px; }}
</style></head><body><div class="wrap">
<h1>RLEF Research Plan</h1>
<p class="sub">Task DAG for the RLEF × Forecasting program (epic <code>hub-t8k9c</code> in hub beads). Arrows read top to bottom: a task can start when everything pointing into it is done. Gate 0 (the <a href="/proposal.html">proposal</a>) was approved by Jessica on 2026-08-19 with the cutoff-audit amendment (green).</p>
<p class="meta">consistency-ft · beads live in futarchy-fi/hub · dashboard at <a href="/">consistency.futarchy.ai</a></p>
<div class="frame">{''.join(svg)}</div>
<div class="legend">
  <span><span class="chip" style="background:var(--task);border-color:var(--taskline)"></span>core task</span>
  <span><span class="chip" style="background:var(--amend);border-color:var(--amendline)"></span>cutoff-audit amendment (Jessica, Gate 0)</span>
  <span><span class="chip" style="background:var(--gate);border-color:var(--gateline)"></span>approval gate</span>
</div>
<p id="live">live status: loading…</p>
<script>
async function refresh() {{
  try {{
    const r = await fetch('/plan-status.json', {{cache: 'no-store'}});
    const d = await r.json();
    let done = 0, total = 0;
    document.querySelectorAll('g[data-bead]').forEach(g => {{
      const st = d.status[g.dataset.bead];
      g.classList.remove('st-closed','st-in_progress','st-blocked');
      if (st) g.classList.add('st-' + st);
      total += 1; if (st === 'closed') done += 1;
    }});
    const age = Math.round((Date.now()/1000 - d.updated)/60);
    document.getElementById('live').textContent =
      `live status: ${{done}}/${{total}} closed · beads snapshot ${{age}} min old · page refreshes every 60 s`;
  }} catch (e) {{
    document.getElementById('live').textContent = 'live status unavailable (' + e.message + ')';
  }}
}}
refresh(); setInterval(refresh, 60000);
</script>
</div></body></html>
"""
out = Path(__file__).resolve().parents[1] / "dash" / "plan.html"
out.write_text(page)
print(f"Wrote {out} ({len(NODES)} nodes, {len(EDGES)} edges, {width}x{height})")
