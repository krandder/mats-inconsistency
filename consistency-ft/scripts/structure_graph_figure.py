#!/usr/bin/env python3
"""Figure: the graph of the structured test pairs (27 Sep 2026).

Nodes are questions (a question in several pairs is one node), edges are the pairs:
  implication A -> B (threshold and date ladders)  solid arrow
  mutual exclusion (options of one market)         dashed
  control (no known relation)                      dotted
Overlay from Qwen3-235B, question only (the full 1,077-pair run):
  edge colour and width = log odds ratio from its 4 AND answers, bias-corrected and normalized
                          (the `and4` source of scripts/weighted_projection.py; b fitted label-free
                          on the whole run)
  x on an edge          = its raw answers violate the known relation:
                          P(A) > P(B) for A -> B, or P(A and B) > 0.05 for exclusive options
  node fill             = market price at t (normalized over open options for mc_excl)

Writes one HTML fragment (inline SVG, site CSS tokens --s1 --s2 --ink --ink2 --ink3 --axis --paper,
classes t t2 tb gl) that the ekelvin-research site builder includes in joint.html.

Usage: python3 scripts/structure_graph_figure.py [--out /home/agent/data/structured-pairs-20260924/structure-graph.html]
"""

from __future__ import annotations

import argparse
import collections
import html
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import weighted_projection as wp  # noqa: E402

OUT = Path("/home/agent/data/structured-pairs-20260924/structure-graph.html")
RUN = "235B Q"
KIND_ORDER = ("threshold", "date", "mc_excl", "control", "mixed")
KIND_LABEL = {"threshold": "Threshold ladders", "date": "Date ladders", "mc_excl": "Options of one market",
              "control": "Controls (no known relation)", "mixed": "Linked pair joined by a control"}
LAB = 'style="paint-order:stroke;stroke:var(--paper);stroke-width:3.5px;stroke-linejoin:round"'


def esc(s):
    return html.escape(str(s), quote=True)


def short(s, n):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rstrip(" ,;:-") + "…"


# ------------------------------------------------------------------ data

def load():
    jobs = wp.load_structured()
    job = next(j for j in jobs if j["name"] == RUN)
    wp.prep(job)
    b, _ = wp.est_bias_var(job["T"], job["W0"])
    lor, _ = wp.or_and4(job["T"], job["W0"], b)
    S = job["S"]
    ia = wp.FORM_ORDER.index("(P) AND (Q)")
    nodes, edges = {}, []
    for k, (pid, p) in enumerate(zip(job["pids"], job["pairs_meta"])):
        for side in ("p", "q"):
            t = p[f"{side}_text"]
            price = p.get(f"{side}_price_norm", p[f"{side}_price_t"]) if p["type"] == "mc_excl" else p[f"{side}_price_t"]
            full = t["parent_question"] + (f" — {t['answer_text']}" if t.get("answer_text") else "")
            nodes.setdefault(p[f"{side}_id"], {"id": p[f"{side}_id"], "full": full, "parent": t["parent_question"],
                                               "answer": t.get("answer_text"), "price": float(price), "t": p["t"]})
        r = p["relation"]
        a, c = p["p_id"], p["q_id"]
        sp, sq, sa = S[k, wp.iP], S[k, wp.iQ], S[k, ia]
        if "implies" in r:
            if r["implies"] == "Q->P":
                a, c, sp, sq = c, a, sq, sp
            viol, why = bool(sp > sq), f"model P(A) {sp:.2f} vs P(B) {sq:.2f}"
        elif r.get("exclusive"):
            viol, why = bool(sa > 0.05), f"model P(A and B) {sa:.2f}"
        else:
            viol, why = False, f"model P(A and B) {sa:.2f}"
        edges.append({"a": a, "b": c, "type": p["type"], "lor": float(lor[k]), "viol": viol, "why": why,
                      "pair": pid, "group": p["group"], "rel": r})
    return nodes, edges


def components(nodes, edges):
    par = {n: n for n in nodes}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for e in edges:
        par[f(e["a"])] = f(e["b"])
    comp = collections.defaultdict(lambda: {"nodes": [], "edges": []})
    for n in nodes:
        comp[f(n)]["nodes"].append(n)
    for e in edges:
        comp[f(e["a"])]["edges"].append(e)
    out = []
    for c in comp.values():
        types = {e["type"] for e in c["edges"]}
        c["kind"] = types.pop() if len(types) == 1 else "mixed"
        c["viol"] = sum(e["viol"] for e in c["edges"])
        c["order"] = order_nodes(c)
        out.append(c)
    return out


def order_nodes(c):
    """Topological order along implications (strictest first); other nodes by BFS."""
    ns = set(c["nodes"])
    indeg = {n: 0 for n in ns}
    succ = collections.defaultdict(list)
    for e in c["edges"]:
        if e["type"] in ("threshold", "date"):
            indeg[e["b"]] += 1
            succ[e["a"]].append(e["b"])
    nb = collections.defaultdict(set)
    for e in c["edges"]:
        nb[e["a"]].add(e["b"]); nb[e["b"]].add(e["a"])
    order, seen = [], set()
    ready = sorted((n for n in ns if indeg[n] == 0), key=lambda n: (-len(succ[n]), n))
    while ready:
        n = ready.pop(0)
        if n in seen:
            continue
        seen.add(n); order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.insert(0, m)
        for m in sorted(nb[n] - seen):
            if indeg[m] == 0 and m not in ready:
                ready.append(m)
    order += sorted(ns - seen)
    return order


# ------------------------------------------------------------------ encodings

def ecol(lor):
    if abs(lor) < 0.5:
        return "var(--ink3)"
    return "var(--s1)" if lor > 0 else "var(--s2)"


def ewid(lor, scale=1.0):
    return scale * (0.8 + 1.1 * min(abs(lor), 4.0))


def nfill(p):
    return f'fill="var(--ink)" fill-opacity="{0.06 + 0.94 * p:.2f}"'


def edge_title(e, nodes, n=200):
    rel = {"threshold": "A implies B", "date": "A implies B", "mc_excl": "A excludes B", "control": "no known relation"}[e["type"]]
    return esc(f"{rel}: {short(nodes[e['a']]['full'], n)}  |  {short(nodes[e['b']]['full'], n)}\n"
               f"model log odds ratio {e['lor']:+.2f}; {e['why']}" + ("  VIOLATES the relation" if e["viol"] else ""))


def node_title(n):
    return esc(f"{n['full']}\nmarket {n['price']:.2f} at t = {n['t']}")


def arrow(x1, y1, x2, y2, col, size):
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [(x2, y2), (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45)),
           (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))]
    return f'<path d="M{" L".join(f"{x:.1f},{y:.1f}" for x, y in pts)} Z" fill="{col}"/>'


def xmark(x, y, s):
    return (f'<path d="M{x-s:.1f},{y-s:.1f} L{x+s:.1f},{y+s:.1f} M{x-s:.1f},{y+s:.1f} L{x+s:.1f},{y-s:.1f}" '
            f'stroke="var(--paper)" stroke-width="{s*1.6:.1f}" stroke-linecap="round"/>'
            f'<path d="M{x-s:.1f},{y-s:.1f} L{x+s:.1f},{y+s:.1f} M{x-s:.1f},{y+s:.1f} L{x+s:.1f},{y-s:.1f}" '
            f'stroke="var(--ink)" stroke-width="{max(1.2, s*0.55):.1f}" stroke-linecap="round"/>')


def dash(e, scale=1.0):
    if e["type"] == "mc_excl":
        return f' stroke-dasharray="{5*scale:.1f} {3.5*scale:.1f}"'
    if e["type"] == "control":
        return f' stroke-dasharray="{1*scale:.1f} {3*scale:.1f}"'
    return ""


def draw_edge(e, P, nodes, r, scale=1.0, bend=0.0, xs=4.0):
    (x1, y1), (x2, y2) = P[e["a"]], P[e["b"]]
    col, w = ecol(e["lor"]), ewid(e["lor"], scale)
    d = math.hypot(x2 - x1, y2 - y1) or 1
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    imp = e["type"] in ("threshold", "date")
    ah = (5 + 1.3 * w) if imp else 0
    o = [f'<g><title>{edge_title(e, nodes)}</title>']
    if bend:
        # quadratic arc, control point offset perpendicular
        mx, my = (x1 + x2) / 2 - uy * bend, (y1 + y2) / 2 + ux * bend
        # trim ends by node radius along tangents
        t1x, t1y = mx - x1, my - y1; l1 = math.hypot(t1x, t1y)
        t2x, t2y = x2 - mx, y2 - my; l2 = math.hypot(t2x, t2y)
        sx, sy = x1 + t1x / l1 * (r + 2), y1 + t1y / l1 * (r + 2)
        ex, ey = x2 - t2x / l2 * (r + 2), y2 - t2y / l2 * (r + 2)
        tx, ty = ex - t2x / l2 * ah * 0.8, ey - t2y / l2 * ah * 0.8
        o.append(f'<path d="M{sx:.1f},{sy:.1f} Q{mx:.1f},{my:.1f} {tx:.1f},{ty:.1f}" fill="none" stroke="{col}" '
                 f'stroke-width="{w:.1f}" stroke-linecap="round"{dash(e, scale)}/>')
        o.append(f'<path d="M{sx:.1f},{sy:.1f} Q{mx:.1f},{my:.1f} {ex:.1f},{ey:.1f}" fill="none" stroke="transparent" stroke-width="10"/>')
        if imp:
            o.append(arrow(tx - t2x / l2, ty - t2y / l2, ex, ey, col, ah))
        cx, cy = 0.25 * x1 + 0.5 * mx + 0.25 * x2, 0.25 * y1 + 0.5 * my + 0.25 * y2
    else:
        sx, sy = x1 + ux * (r + 2), y1 + uy * (r + 2)
        ex, ey = x2 - ux * (r + 2), y2 - uy * (r + 2)
        tx, ty = ex - ux * ah * 0.8, ey - uy * ah * 0.8
        o.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="{w:.1f}" '
                 f'stroke-linecap="round"{dash(e, scale)}/>')
        o.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="transparent" stroke-width="10"/>')
        if imp:
            o.append(arrow(sx, sy, ex, ey, col, ah))
        cx, cy = (sx + tx) / 2, (sy + ty) / 2
    if e["viol"]:
        o.append(xmark(cx, cy, xs))
    o.append("</g>")
    return "".join(o)


# ------------------------------------------------------------------ labels

def word_diff_labels(texts):
    """Label each text by the words where it differs from the others (common prefix/suffix dropped)."""
    toks = [t.split() for t in texts]
    if len(toks) < 2:
        return texts
    pre = 0
    while all(len(t) > pre for t in toks) and len({t[pre] for t in toks}) == 1:
        pre += 1
    suf = 0
    while all(len(t) > pre + suf for t in toks) and len({t[-1 - suf] for t in toks}) == 1:
        suf += 1
    out = []
    for t in toks:
        mid = t[pre: len(t) - suf]
        # a little context: the word before the varying part ("above", "by") if it is short
        j = pre
        while j and len(t[j - 1]) <= 7 and len(" ".join(mid)) < 12:
            j -= 1
            mid = [t[j]] + mid
        out.append(" ".join(mid).rstrip("?") or " ".join(t))
    return out


def panel_title_and_labels(c, nodes):
    ns = c["order"]
    kind = c["kind"]
    if kind in ("threshold", "date", "mc_excl"):
        parents = {nodes[n]["parent"] for n in ns}
        if len(parents) == 1 and all(nodes[n]["answer"] for n in ns):
            title = nodes[ns[0]]["parent"]
            labs = [nodes[n]["answer"] for n in ns]
            if max(map(len, labs)) > 30:
                labs = word_diff_labels(labs)
        else:
            texts = [nodes[n]["full"] for n in ns]
            labs = texts if max(map(len, texts)) <= 30 else word_diff_labels(texts)
            tmpl = c["edges"][0]["rel"].get("template")
            title = (tmpl.replace("<n>", "…").replace("<d>", "…") if tmpl else texts[0])
            title = title[:1].upper() + title[1:]
    else:
        title = "Control pair"
        labs = [nodes[n]["full"] for n in ns]
    return title, labs


# ------------------------------------------------------------------ panels

def panel(c, nodes, labmax=None):
    labmax = labmax or (27 if c["kind"] in ("control", "mixed") else 30)
    ns = c["order"]
    title, labs = panel_title_and_labels(c, nodes)
    W, x0, y0, dy, r = 260, 40, 16, 50, 7
    P = {n: (x0, y0 + i * dy) for i, n in enumerate(ns)}
    H = y0 + (len(ns) - 1) * dy + 16
    pos = {n: i for i, n in enumerate(ns)}
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(short(title, 80))}: {len(ns)} questions, {len(c["edges"])} links">']
    for e in c["edges"]:
        gap = abs(pos[e["a"]] - pos[e["b"]])
        bend = 0.0 if gap == 1 else -(14 + 8 * gap) * (1 if pos[e["b"]] > pos[e["a"]] else -1)
        o.append(draw_edge(e, P, nodes, r, bend=bend))
    for n, lab in zip(ns, labs):
        x, y = P[n]; nd = nodes[n]
        o.append(f'<g><title>{node_title(nd)}</title><circle cx="{x}" cy="{y}" r="{r}" {nfill(nd["price"])} '
                 f'stroke="var(--ink2)" stroke-width="1"/>'
                 f'<text class="t2" x="{x + r + 7}" y="{y + 4.5}" style="font-size:12.5px">{esc(short(lab, labmax))}'
                 f'<tspan class="t" dx="6">{round(100 * nd["price"])}%</tspan></text></g>')
    o.append("</svg>")
    kind = c["kind"] if c["kind"] != "mc_excl" else "options"
    return (f'<div class="gpanel"><h3 title="{esc(title)}">{esc(short(title, 44))}</h3>' + "".join(o) + "</div>")


def pick(comps):
    def spread(c):
        ps = [c["pnodes"][n]["price"] for n in c["nodes"]]
        return max(ps) - min(ps)

    def ok_text(c):
        return all(len(c["pnodes"][n]["full"]) < 140 for n in c["nodes"])

    sel = []
    for kind in ("threshold", "date"):
        cs = [c for c in comps if c["kind"] == kind and ok_text(c)]
        cs.sort(key=lambda c: (-len(c["nodes"]), -len(c["edges"]), -c["viol"], -spread(c)))
        chosen = []
        v = next((c for c in cs if c["viol"]), None)
        for c in ([v] if v else []) + cs:
            if c not in chosen and len({c2["edges"][0]["group"] for c2 in chosen + [c]}) == len(chosen) + 1:
                chosen.append(c)
            if len(chosen) == 3:
                break
        sel.append((KIND_LABEL[kind], chosen))
    cs = [c for c in comps if c["kind"] == "mc_excl" and ok_text(c)]
    cs.sort(key=lambda c: (-c["edges"][0]["rel"].get("n_options_open", 0)))
    v = next(c for c in cs if c["viol"])
    rest = [c for c in cs if not c["viol"]]
    sel.append((KIND_LABEL["mc_excl"], [rest[0], v, rest[len(rest) // 2]]))
    cs = [c for c in comps if c["kind"] == "control" and len(c["nodes"]) == 2 and ok_text(c)]
    cs.sort(key=lambda c: abs(c["edges"][0]["lor"]))
    sel.append(("Controls", [cs[-1], cs[len(cs) // 2]]))
    return sel


# ------------------------------------------------------------------ overview + histogram

def overview(comps, nodes):
    W, pad, r, sp, rowh = 800, 9, 2.6, 17, 17
    o, y = [], 6
    for kind in KIND_ORDER:
        cs = sorted((c for c in comps if c["kind"] == kind), key=lambda c: (-len(c["nodes"]), -len(c["edges"]), -c["viol"]))
        if not cs:
            continue
        nv = sum(c["viol"] for c in cs)
        ne = sum(len(c["edges"]) for c in cs)
        o.append(f'<text class="tb" x="0" y="{y + 12}">{KIND_LABEL[kind]}</text>'
                 f'<text class="t" x="{W}" y="{y + 12}" text-anchor="end">{len(cs)} components · {ne} pairs'
                 + (f" · {nv} violated" if nv else "") + "</text>")
        y += 22
        x = 0
        for c in cs:
            k = len(c["nodes"])
            cw = (k - 1) * sp + 2 * r + pad
            if x + cw > W:
                x = 0; y += rowh
            P = {n: (x + r + i * sp, y + 12) for i, n in enumerate(c["order"])}
            pos = {n: i for i, n in enumerate(c["order"])}
            for e in c["edges"]:
                gap = abs(pos[e["a"]] - pos[e["b"]])
                (x1, y1), (x2, y2) = P[e["a"]], P[e["b"]]
                col, w = ecol(e["lor"]), ewid(e["lor"], 0.55)
                t = f'<title>{edge_title(e, nodes, 50)}</title>'
                if gap == 1:
                    o.append(f'<line x1="{x1 + r + 1:.1f}" y1="{y1}" x2="{x2 - r - 1:.1f}" y2="{y2}" stroke="{col}" '
                             f'stroke-width="{w:.1f}"{dash(e, 0.6)}>{t}</line>')
                else:
                    mx = (x1 + x2) / 2
                    o.append(f'<path d="M{x1:.1f},{y1 - 2} Q{mx:.1f},{y1 - 5 - 3 * gap} {x2:.1f},{y2 - 2}" fill="none" '
                             f'stroke="{col}" stroke-width="{w:.1f}"{dash(e, 0.6)}>{t}</path>')
                if e["viol"]:
                    o.append(xmark((x1 + x2) / 2, y1 + (0 if gap == 1 else -3 - 1.5 * gap), 2.4))
            for n in c["order"]:
                px, py = P[n]
                o.append(f'<circle cx="{px:.1f}" cy="{py}" r="{r}" {nfill(nodes[n]["price"])} stroke="var(--ink2)" '
                         f'stroke-width=".6"><title>{node_title(nodes[n])}</title></circle>')
            x += cw
        y += rowh + 10
    return (f'<svg viewBox="0 -2 {W} {y:.0f}" role="img" aria-label="All {len(comps)} components of the structured pair graph, '
            f'grouped by relation type">' + "".join(o) + "</svg>")


def histogram(comps):
    cnt = collections.Counter(len(c["nodes"]) for c in comps)
    sizes = list(range(2, max(cnt) + 1))
    W, H, L, B = 300, 150, 34, 30
    top = max(cnt.values())
    bw = (W - L - 6) / len(sizes)
    Y = lambda v: 10 + (H - B - 10) * (1 - math.log10(v) / math.log10(top * 1.4)) if v else H - B
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Components by size" style="max-width:320px">']
    for v in (1, 10, 100):
        o.append(f'<line class="gl" x1="{L}" x2="{W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="t" x="{L - 5}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for i, s in enumerate(sizes):
        v = cnt.get(s, 0)
        x = L + i * bw + 3
        kinds = collections.Counter(c["kind"] for c in comps if len(c["nodes"]) == s)
        tip = esc(f"{v} components of {s} questions: " + ", ".join(f"{n} {KIND_LABEL[k].lower()}" for k, n in kinds.most_common()))
        if v:
            o.append(f'<g><title>{tip}</title><rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw - 6:.1f}" height="{H - B - Y(v):.1f}" '
                     f'rx="3" fill="var(--ink3)"/><text class="tb" x="{x + (bw - 6) / 2:.1f}" y="{Y(v) - 4:.1f}" '
                     f'text-anchor="middle">{v}</text></g>')
        o.append(f'<text class="t" x="{x + (bw - 6) / 2:.1f}" y="{H - B + 14}" text-anchor="middle">{s}</text>')
    o.append(f'<line x1="{L}" x2="{W}" y1="{H - B}" y2="{H - B}" stroke="var(--axis)"/>'
             f'<text class="t" x="{(L + W) / 2:.0f}" y="{H - 2}" text-anchor="middle">questions per component (log count)</text></svg>')
    return "".join(o)


def legend():
    def ln(col, w, dash="", arrowhead=False, x=False):
        s = f'<svg width="38" height="14" aria-hidden="true"><line x1="3" y1="7" x2="{27 if arrowhead else 35}" y2="7" stroke="{col}" stroke-width="{w}" stroke-linecap="round"{dash}/>'
        if arrowhead:
            s += f'<path d="M36,7 L27,2.5 L27,11.5 Z" fill="{col}"/>'
        if x:
            s += '<path d="M15,3 L23,11 M15,11 L23,3" stroke="var(--paper)" stroke-width="5" stroke-linecap="round"/><path d="M15,3 L23,11 M15,11 L23,3" stroke="var(--ink)" stroke-width="2" stroke-linecap="round"/>'
        return s + "</svg>"

    def dot(p):
        return f'<svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="6" {nfill(p)} stroke="var(--ink2)"/></svg>'

    return ('<div class="legend">'
            f'<span>{ln("var(--ink3)", 2, arrowhead=True)} A implies B (ladder rung)</span>'
            f'<span>{ln("var(--ink3)", 2, " stroke-dasharray=\"5 3.5\"")} A excludes B (options)</span>'
            f'<span>{ln("var(--ink3)", 2, " stroke-dasharray=\"1 3\"")} control, no known link</span>'
            '</div><div class="legend">'
            f'<span>model sees: {ln("var(--s1)", 4)} together</span>'
            f'<span>{ln("var(--s2)", 4)} apart</span>'
            f'<span>{ln("var(--ink3)", 1)} independent</span>'
            f'<span>{ln("var(--ink3)", 2, x=True)} its raw answers break the link</span>'
            f'<span>market price 0 {dot(.05)}{dot(.5)}{dot(.95)} 1</span>'
            '</div>')


CSS = ("<style>.gmult{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:4px 16px}"
       ".gmult h3,.grow h3{font-size:13px;font-weight:600;margin:6px 0 2px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
       ".grow h4{font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--ink3);margin:14px 0 0}"
       ".gtop{display:flex;flex-wrap:wrap;gap:8px 28px;align-items:flex-end}.gtop .stat{font-size:14px;color:var(--ink2)}"
       ".gtop .stat b{font-size:22px;color:var(--ink);display:block;font-weight:600}</style>")


def build(out):
    nodes, edges = load()
    comps = components(nodes, edges)
    for c in comps:
        c["pnodes"] = nodes
    known = [e for e in edges if e["type"] != "control"]
    kc = components({n for e in known for n in (e["a"], e["b"])}, known)
    viol = collections.Counter(e["type"] for e in edges if e["viol"])
    ntype = collections.Counter(e["type"] for e in edges)
    stats = {"questions": len(nodes), "pairs": len(edges), "components": len(comps),
             "largest": max(len(c["nodes"]) for c in comps), "known_components": len(kc),
             "viol": dict(viol), "ntype": dict(ntype)}
    mult = []
    for head, cs in pick(comps):
        mult.append(f'<h4>{esc(head)}</h4><div class="gmult">' + "".join(panel(c, nodes) for c in cs) + "</div>")
    top = ('<div class="gtop"><div>' + histogram(comps) + '</div><div>'
           f'<div class="stat"><b>{len(nodes):,}</b>questions</div></div>'
           f'<div><div class="stat"><b>{len(edges):,}</b>pairs</div></div>'
           f'<div><div class="stat"><b>{len(comps)}</b>components, largest {stats["largest"]}</div></div></div>')
    frag = (CSS + '<figure class="grow">' + top + legend() + "".join(mult)
            + '<h4>All components</h4>' + overview(comps, nodes)
            + f'<figcaption>Each dot is a question, each line a test pair; hover for the full text. Line colour and width: '
            f'Qwen3-235B’s dependence (log odds ratio of its four AND answers). ×: its raw answers break the known link '
            f'({viol["threshold"]} of {ntype["threshold"]} threshold, {viol["date"]} of {ntype["date"]} date, '
            f'{viol["mc_excl"]} of {ntype["mc_excl"]} option pairs). Each single-answer market gives only its top two options, '
            f'so options form single dashed links.</figcaption></figure>')
    Path(out).write_text(frag)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    print(build(a.out))
