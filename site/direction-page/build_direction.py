#!/usr/bin/env python3
"""Build mats.ekelvin.com/direction/ (the research direction note) from its Markdown source.

    python3 build_direction.py [SOURCE.md] [--no-figs]     -> direction/index.html next to this script

Same idea as MATH_DOCS in calibrated-ai/experiments_page.py: $...$ is kept away from Markdown and
typeset with KaTeX 0.16.9 (here at build time with node, so the page needs no JavaScript; the KaTeX
CSS comes from the same cdnjs URL). Markdown is parsed as CommonMark (markdown-it-py), because the
note's lists use CommonMark's 3-space indents. Kelvin, 16 Oct 2026: "clean, not full of text": long
prose paragraphs and list items collapse to their lead clause (text unchanged, one tap to expand),
equations stand out, and Richardson-style PDG figures are drawn as inline SVG (figure c replaces the
mermaid flowchart). Deploy: rsync direction/ to deploy-host:fleet/apps/ekelvin-research/dist/direction/,
then the deploy step of calibrated-ai/sync_economy_tracks.sh.
"""
import html, json, math, pathlib, re, subprocess, sys

from markdown_it import MarkdownIt

HERE = pathlib.Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
NO_FIGS = '--no-figs' in sys.argv
SRC = pathlib.Path(ARGS[0]) if ARGS else pathlib.Path('/home/agent/workspaces/mats/outbox/research-direction-note-v2-2026-10-16.md')
OUT = HERE / 'direction' / 'index.html'
KT = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/'
MD = MarkdownIt('commonmark', {'html': True, 'breaks': True}).enable('table')

TITLE, SUBTITLE, META = 'Inconsistency Is All You Need', '', ''  # header lines come from the source when present
WARN = []
esc = html.escape

# ---------------------------------------------------------------- math placeholders
MATHS = []  # (tex, display)


def keep(tex, display):
    MATHS.append((tex, display))
    return f'{"D" if display else ""}MATHX{len(MATHS) - 1}X'


def protect(s):
    s = re.sub(r'\$\$(.+?)\$\$', lambda m: keep(m.group(1), True), s, flags=re.S)
    return re.sub(r'\$((?:\\.|[^$\\\n])+?)\$', lambda m: keep(m.group(1), False), s)


def katex_all():
    js = r"""const k=require('./node_modules/katex');let s='';process.stdin.on('data',d=>s+=d);
process.stdin.on('end',()=>{process.stdout.write(JSON.stringify(JSON.parse(s).map(([t,d])=>{
try{return {h:k.renderToString(t,{displayMode:d,throwOnError:true,strict:'ignore'})}}
catch(e){return {h:k.renderToString(t,{displayMode:d,throwOnError:false,strict:'ignore'}),e:String(e.message)}}})))});"""
    r = subprocess.run(['node', '-e', js], cwd=HERE, input=json.dumps(MATHS), capture_output=True, text=True, check=True)
    out = json.loads(r.stdout)
    for (tex, _), o in zip(MATHS, out):
        if 'e' in o:
            WARN.append(f'KaTeX error in {tex[:60]!r}: {o["e"]}')
    return [o['h'] for o in out]


def words(h):
    return len(re.sub(r'D?MATHX\d+X', 'M', re.sub(r'<[^>]+>', ' ', h)).split())


# ---------------------------------------------------------------- source -> HTML
src = SRC.read_text()
# header: '# Title', then an optional '*subtitle*' line and an optional '**byline**' line, then '---'
m = re.match(r'\s*# (.+)\n', src)
if m:
    TITLE, src = m.group(1).strip(), src[m.end():]
m = re.match(r'\s*\*([^*\n].*?)\*\s*\n', src)
if m:
    SUBTITLE, src = m.group(1).strip(), src[m.end():]
m = re.match(r'\s*\*\*(.+?)\*\*\s*\n', src)
if m:
    META, src = m.group(1).strip(), src[m.end():]
src = re.sub(r'^\s*---\s*\n', '', src, count=1)
n_mermaid = len(re.findall(r'```mermaid\n.*?```', src, flags=re.S))
src = re.sub(r'```mermaid\n.*?```', '' if NO_FIGS else 'FIGC', src, count=1, flags=re.S)  # flowchart -> figure (c)
if n_mermaid != 1:
    WARN.append(f'{n_mermaid} mermaid blocks (figure c replaces the first)')
body = MD.render(protect(src))


# long prose paragraphs / list items -> lead clause + <details> (text unchanged)
def split_lead(inner):
    m = re.match(r'<strong>(.*?[.!?])</strong>', inner)  # a bold opening sentence is the lead
    if m and words(m.group(1)) >= 3:
        return inner[:m.end()], inner[m.end():].strip()
    tag = par = 0
    i, n = 0, len(inner)
    while i < n:
        c = inner[i]
        if c == '<':
            j = inner.index('>', i)
            t = inner[i + 1:j]
            tag += -1 if t.startswith('/') else (0 if t.endswith('/') or t.startswith('br') else 1)
            i = j + 1
            continue
        if c == '(':
            par += 1
        elif c == ')':
            par = max(0, par - 1)
        elif tag == 0 and par == 0:
            cut = None
            if c == ':' and inner[i + 1:i + 2] == ' ':
                cut = i + 1
            elif c == '—':
                cut = i + 1
            elif (c == '.' and re.match(r'\s+(?:[A-Z(<*]|D?MATHX)', inner[i + 1:])
                  and not re.search(r'\b(?:vs|e\.g|i\.e|etc|cf)$', inner[:i])):
                cut = i + 1
            if cut and words(inner[:cut]) >= 6:
                return inner[:cut], inner[cut:].strip()
        i += 1
    return None


def collapse(m):
    inner, nxt = m.group(1), m.group(2)
    if words(inner) < 40 or re.fullmatch(r'D?MATHX\d+X|FIG\w', inner.strip()):
        return m.group(0)
    if 'DMATHX' in inner or (inner.rstrip().endswith(':') and re.match(r'\s*<p>DMATHX\d+X</p>', nxt)):
        return m.group(0)  # keep equations, and the sentence that introduces one, in view
    sp = split_lead(inner)
    if not sp or words(sp[1]) < 20:
        return m.group(0)
    lead, rest = sp
    if m.group(0).startswith('<li>'):
        return (f'<li><details class="more"><summary>{lead} <span class="mk">more</span></summary>'
                f'<span class="rest">{rest}</span></details></li>')
    return (f'<details class="more"><summary>{lead} <span class="mk">more</span></summary>'
            f'<p>{rest}</p></details>')


# the next line is only looked at (lookahead), so back-to-back paragraphs / items are all processed
body = re.sub(r'(?m)^<p>(.*?)</p>$(?=\n?(.*))', collapse, body)
body = re.sub(r'(?m)^<li>((?:(?!<li>|<ul>|<ol>|<p>|<br).)*)</li>$(?=\n?(.*))', collapse, body)
body = body.replace('full write-up at mats.ekelvin.com/experiments',
                    'full write-up at <a href="/experiments/">mats.ekelvin.com/experiments</a>')


# tables: class + per-cell labels for the stacked phone layout
def table(m):
    t = re.sub(r' style="text-align: ?\w+;?"', '', m.group(0))
    heads = re.findall(r'<th>(.*?)</th>', t, flags=re.S)

    def row(mr):
        cells = re.findall(r'<td>(.*?)</td>', mr.group(1), flags=re.S)
        return '<tr>' + ''.join(f'<td>{"" if j == 0 else f"<span class=lbl>{heads[j]}</span>"}{c}</td>'
                                for j, c in enumerate(cells)) + '</tr>'
    return re.sub(r'<tr>((?:(?!</tr>).)*<td>.*?)</tr>', row, t.replace('<table>', '<table class="t">'), flags=re.S)


body = re.sub(r'<table>.*?</table>', table, body, flags=re.S)

# ---------------------------------------------------------------- PDG figures (Richardson & Halpern style)
GREEK = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ',
         'pi': 'π', 'chi': 'χ'}
SYM = {'mid': '|', 'emptyset': '∅', 'varnothing': '∅', 'vdots': '⋮', 'dots': '…', 'ast': '∗', 'to': '→'}


def parse(s, i=0, stop=None):
    """tiny TeX subset -> atoms: ('t', text, italic) | ('sub'|'sup', atoms)"""
    atoms = []
    while i < len(s):
        c = s[i]
        if c == stop:
            return atoms, i + 1
        if c == '\\':
            name = re.match(r'\\([A-Za-z]+|.)', s[i:]).group(1)
            i += 1 + len(name)
            if name in GREEK:
                atoms.append(('t', GREEK[name], True))
            elif name in SYM:
                atoms.append(('t', SYM[name], False))
            elif name in ('text', 'mathrm'):
                j = s.index('}', i)
                atoms.append(('t', s[i + 1:j], False))
                i = j + 1
            elif name == ',':
                atoms.append(('t', ' ', False))
            continue
        if c in '_^':
            i += 1
            if s[i] == '{':
                sub, i = parse(s, i + 1, '}')
            else:
                sub, i = [('t', '∗' if s[i] == '*' else s[i], s[i].isalpha())], i + 1
            atoms.append(('sub' if c == '_' else 'sup', sub))
            continue
        if c == '{':
            grp, i = parse(s, i + 1, '}')
            atoms.extend(grp)
            continue
        atoms.append(('t', c, c.isalpha()))
        i += 1
    return atoms, i


def tspans(segs, size):
    """segs: [(tex, font size, css class or '')] set on one baseline; returns SVG tspans"""
    out, cur = [], [0.0]

    def walk(atoms, fs, off):
        for a in atoms:
            if a[0] == 't':
                dy = off - cur[0]
                cur[0] = off
                d = f' dy="{dy:.1f}"' if abs(dy) > .01 else ''
                f = f' font-size="{fs:.1f}"' if abs(fs - size) > .01 else ''
                out.append(f'<tspan class="{"i" if a[2] else "r"}"{f}{d}>{esc(a[1])}</tspan>')
            else:
                walk(a[1], fs * .7, off + (fs * .22 if a[0] == 'sub' else -fs * .4))
    for k, (tex, fs, cls) in enumerate(segs):
        if k:
            walk([('t', ' ', False)], fs, 0.0)
        if cls:
            out.append(f'<tspan class="{cls}">')
        walk(parse(tex)[0], fs, 0.0)
        if cls:
            out.append('</tspan>')
    return ''.join(out)


def unit(dx, dy):
    L = math.hypot(dx, dy) or 1
    return dx / L, dy / L


class Fig:
    def __init__(self, w, h, aria):
        self.w, self.h, self.aria, self.p = w, h, aria, []

    def node(self, c, tex, r=21, rx=None, ry=None, size=18):
        self.p.append(f'<ellipse class="n" cx="{c[0]}" cy="{c[1]}" rx="{rx}" ry="{ry}"/>' if rx
                      else f'<circle class="n" cx="{c[0]}" cy="{c[1]}" r="{r}"/>')
        self.text(c[0], c[1] + size * .32, tex, size)
        return {'c': c, 'r': r, 'rx': rx, 'ry': ry}

    def text(self, x, y, tex, size=15, anchor='middle', cls=''):
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{cls}>'
                      f'{tspans([(tex, size, "")], size)}</text>')

    def pair(self, x, y, cpd, weight, anchor='middle', size=15, wsize=13.5):
        """an arc label: the conditional, then its weight beside it (lighter, smaller)"""
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}">'
                      f'{tspans([(cpd, size, ""), (weight, wsize, "w")], size)}</text>')

    @staticmethod
    def edge_pt(nd, toward):
        if not isinstance(nd, dict):
            return nd
        (x, y), (tx, ty) = nd['c'], toward
        dx, dy = tx - x, ty - y
        k = 1 / math.sqrt((dx / nd['rx']) ** 2 + (dy / nd['ry']) ** 2) if nd['rx'] else nd['r'] / math.hypot(dx, dy)
        return x + dx * k, y + dy * k

    def curve(self, a, c1, c2, b, head=True, cls='e'):
        """cubic arc a -> b (node dicts or points); returns the point at t=.5"""
        p0, p3 = self.edge_pt(a, c1), self.edge_pt(b, c2)
        self.p.append(f'<path class="{cls}" d="M{p0[0]:.1f},{p0[1]:.1f} C{c1[0]:.1f},{c1[1]:.1f} '
                      f'{c2[0]:.1f},{c2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"/>')
        if head:
            self.head(p3, c2)
        return tuple(.125 * p0[k] + .375 * c1[k] + .375 * c2[k] + .125 * p3[k] for k in (0, 1))

    def line(self, a, b, head=True, cls='e'):
        ca = a['c'] if isinstance(a, dict) else a
        cb = b['c'] if isinstance(b, dict) else b
        return self.curve(a, (ca[0] + (cb[0] - ca[0]) / 3, ca[1] + (cb[1] - ca[1]) / 3),
                          (ca[0] + 2 * (cb[0] - ca[0]) / 3, ca[1] + 2 * (cb[1] - ca[1]) / 3), b, head, cls)

    def head(self, tip, frm, L=10, W=7.5):
        ux, uy = unit(tip[0] - frm[0], tip[1] - frm[1])
        bx, by = tip[0] - L * ux, tip[1] - L * uy
        self.p.append(f'<path class="h" d="M{tip[0]:.1f},{tip[1]:.1f} L{bx - W / 2 * uy:.1f},{by + W / 2 * ux:.1f} '
                      f'L{bx + W / 2 * uy:.1f},{by - W / 2 * ux:.1f} Z"/>')

    def svg(self):
        return (f'<svg class="pdg" viewBox="0 0 {self.w} {self.h}" style="max-width:{round(self.w * 1.15)}px" '
                f'role="img" aria-label="{esc(self.aria)}">' + ''.join(self.p) + '</svg>')


W = ' class="w"'


def fig_world_model():
    f = Fig(440, 300, 'PDG of the any-order world model: variables X1, X2, X3 as circles; arcs q(X3|X1), '
                      'q(X3|X1,X2) as a hyperedge, q(X3) with no source, and q(X2|X3), each with a weight beta(i|S).')
    f.w = 470
    x1, x2, x3 = f.node((62, 66), 'X_1'), f.node((62, 234), 'X_2'), f.node((330, 150), 'X_3')
    f.curve(x1, (150, 26), (272, 50), x3)                                # S = {1}
    f.pair(196, 30, 'q(X_3 | X_1)', r'\beta(3 | 1)')
    j = (172, 150)                                                       # S = {1, 2}: hyperedge
    f.curve(x1, (90, 124), (130, 150), j, head=False)
    f.curve(x2, (90, 176), (130, 150), j, head=False)
    f.line(j, x3)
    f.text(240, 140, 'q(X_3 | X_1, X_2)')
    f.text(240, 171, r'\beta(3 | 1, 2)', 13.5, cls=W)
    f.line((462, 150), x3)                                               # S = {}: no source
    f.text(408, 140, 'q(X_3)')
    f.text(408, 171, r'\beta(3 | \emptyset)', 13.5, cls=W)
    f.curve(x3, (312, 252), (180, 264), x2)                              # any order: X3 -> X2
    f.pair(205, 290, 'q(X_2 | X_3)', r'\beta(2 | 3)')
    return f.svg()


def fig_pool():
    f = Fig(470, 270, 'PDG of the pool for a query Y: a prior arc p0(Y) with weight beta0 and rollout arcs '
                      'q_r(Y|S_r) from S1, S2, ..., Sk with weights beta_r.')
    y = f.node((342, 138), 'Y')
    for (sx, sy), r in (((50, 40), '1'), ((50, 118), '2'), ((50, 236), 'k')):
        s = f.node((sx, sy), f'S_{r}')
        f.curve(s, (sx + 140, sy), (342 - 82, 138 + .35 * (sy - 138)), y)
        f.pair(sx + 108, sy - 10 if r != 'k' else sy + 26, rf'q_{r}(Y\, | S_{r})', rf'\beta_{r}')
    f.text(50, 184, r'\vdots', 20)
    f.line((462, 138), y)                                                # prior: no source
    f.pair(414, 128, r'p_0(Y\,)', r'\beta_0')
    return f.svg()


def fig_blocks():
    f = Fig(470, 440, 'PDG of the four parameter blocks: theta_real from X_S to X_i with chi=(1,0); theta_amp '
                      'from X_S to the rollout token (k_r, v_r) with chi=(0,1); theta_trust from X_S and h_search '
                      'to beta_r with chi=(1,0); theta_agg and theta_pol from (k_r, v_r) and beta_r to mu* and '
                      'h_search with chi=(0,1). Dashed: stop-grad features from theta_real.')
    xs = f.node((56, 200), 'X_S', r=23)
    xi = f.node((232, 58), 'X_i')
    kv = f.node((232, 182), '(k_r, v_r)', rx=40, ry=21, size=17)
    b = f.node((232, 304), r'\beta_r')
    mu = f.node((414, 182), r'\mu^*')
    h = f.node((414, 304), r'h_{\text{search}}', rx=40, ry=21, size=17)
    m_real = f.curve(xs, (78, 100), (150, 58), xi)                                   # 1. theta_real
    f.text(108, 64, r'\theta_{\text{real}}', 15.5, 'end')
    f.text(108, 82, r'\chi = (1, 0)', 13, 'end', W)
    m_amp = f.curve(xs, (118, 194), (160, 186), kv)                                  # 2. theta_amp
    f.text(150, 211, r'\theta_{\text{amp}}', 15.5)
    f.text(150, 229, r'\chi = (0, 1)', 13, cls=W)
    a = (m_real[0] + 6, m_real[1] + 8)                                               # stop-grad features
    f.curve(a, (a[0] + 8, a[1] + 30), (m_amp[0] - 10, m_amp[1] - 34), (m_amp[0] - 2, m_amp[1] - 6), cls='d')
    f.text(150, 132, r'\text{stop-grad}', 12, 'start', W)
    j3 = (232, 384)                                                                   # 3. theta_trust
    f.curve(xs, (56, 330), (232, 428), j3, head=False)
    f.curve(h, (414, 380), (232, 428), j3, head=False)
    f.line(j3, b)
    f.text(244, 356, r'\theta_{\text{trust}}', 15.5, 'start')
    f.text(244, 374, r'\chi = (1, 0)', 13, 'start', W)
    j4 = (322, 243)                                                                   # 4. theta_agg, theta_pol
    f.curve(kv, (300, 186), (286, 243), j4, head=False)
    f.curve(b, (296, 300), (286, 243), j4, head=False)
    f.curve(j4, (360, 243), (388, 214), mu)
    f.curve(j4, (360, 243), (388, 272), h)
    f.text(322, 156, r'\theta_{\text{agg}}, \theta_{\text{pol}}', 15.5)
    f.text(322, 175, r'\chi = (0, 1)', 13, cls=W)
    return f.svg()


def figure(svg, cap):
    return f'<figure>{svg}<figcaption>{protect(cap)}</figcaption></figure>'


def place(body, pattern, html_, where, name):
    m = re.search(pattern, body, flags=re.S)
    if not m:
        WARN.append(f'figure {name}: anchor not found, put at the end')
        return body + html_
    k = m.start() if where == 'before' else m.end()
    return body[:k] + html_ + body[k:]


if not NO_FIGS:
    FIG_A = figure(fig_world_model(), '(a) The any-order world model as a PDG: each conditional it can answer is an '
                   'arc $S \\to X_i$ labelled $q(X_i \\mid X_S)$ with weight $\\beta(i \\mid S)$, from any '
                   'conditioning set, in any direction.')
    FIG_B = figure(fig_pool(), '(b) The pool for a query $Y$ as a PDG: a prior arc $p_0(Y)$ with weight $\\beta_0$ '
                   'and one arc $q_r(Y \\mid S_r)$ with weight $\\beta_r$ per rollout.')
    FIG_C = figure(fig_blocks(), '(c) The four parameter blocks as PDG arcs, labelled with their control masks '
                   '$\\chi = (\\chi_{\\text{real}}, \\chi_{\\text{search}})$. Dashed: stop-grad features from '
                   '$\\theta_{\\text{real}}$.')
    body = place(body, r'<p>FIGC</p>', '', 'before', 'c').replace('<p>FIGC</p>', FIG_C, 1)
    body = place(body, r'<p>For any fixed world model', FIG_A, 'before', 'a')
    body = place(body, r'<details class="more"><summary>In practice, we cannot evaluate.*?</details>|'
                       r'<p>In practice, we cannot evaluate.*?</p>', FIG_B, 'after', 'b')

# ---------------------------------------------------------------- narrow forms of the long display equations
# Kelvin, 16 Oct: "formulas ... often require horizontal scrolling". Each long equation also gets a multi-line
# form (same symbols, explicit line breaks; underbrace labels become short text lines below it). A tiny inline
# script shows the original where it fits its box and the multi-line form where it would not; without
# JavaScript the multi-line form is used below 640px. Keyed by a substring of the original TeX.
NARROW = {
    'W_{\\text{eff}}(N, \\rho, \\beta)': (r"""\begin{aligned}
&W_{\text{eff}}(N, \rho, \beta) \;=\; \frac{N \beta}{\lambda + \big(1 + (N-1)\rho\big)\beta}, \\
&\log \mu^*_k(y) \;=\; \log p_0(y) \;+\; \sum_{r=1}^k w_r \, \log \frac{q_r(y)}{p_0(y)} \\
&\qquad\quad + \text{const}.
\end{aligned}""", []),
    'World-Model Observational Incompatibility': (r"""\begin{aligned}
&\mathcal{E}(\mu; \theta, U) = \sum_{(S \to i)} \beta_\theta(i \mid S) \\
&\quad \mathbb{E}_{\mu(X_S)}\!\Big[ \mathrm{KL}\big(\mu(X_i \mid X_S) \,\big\|\, q_\theta(X_i \mid X_S)\big) \Big] \\
&\quad + \mathrm{KL}\big(\mu(X_G) \,\big\|\, f_U(X_G)\big)_{\chi_{\text{world}}=0} \\
&\quad - \gamma H(\mu).
\end{aligned}""", [r'The $\sum$ term: World-Model Observational Incompatibility.',
                   r'The $\mathrm{KL}\big(\mu(X_G) \,\|\, f_U(X_G)\big)$ term: Modular Goal / Utility Incompatibility.']),
    r'\frac{\beta_r}{c_r}': (r"""\begin{aligned}
&\log \mu^*_k(y) = \log p_0(y) \\
&\quad + \frac{1}{\beta_0}\sum_{r=1}^k \frac{\beta_r}{c_r} \, \log \frac{q_r(y)}{p_0(y)} + \text{const}.
\end{aligned}""", []),
    r'\eta_{\max}(1 - C_M)^\theta': (r"""\begin{aligned}
&\operatorname{logit} P(\text{sol} \mid \text{null}) = \operatorname{logit} p_0 \\
&\quad + \log\!\big[(1 - \eta_{\max}) + \eta_{\max}(1 - C_M)^\theta\big].
\end{aligned}""", []),
    r'\langle\langle M_k + \delta_x \rangle\rangle \;-': (r"""\begin{aligned}
&-\beta_0 \log \mu^*_k(x) \\
&\quad = \langle\langle M_k + \delta_x \rangle\rangle - \langle\langle M_k \rangle\rangle.
\end{aligned}""", []),
    r'\mathbb{E}_{\text{fused}}': (r"""\begin{aligned}
&\mathbb{E}_{\text{fused}}\big[\langle \mu_{\text{belief}} - P^*, \, \nabla U \rangle\big] \\
&\quad = \frac{\lambda}{2}\|\nabla U\|^2 \ge 0, \\
&\mathbb{E}_{\text{masked}}\big[\langle \mu_{\text{belief}} - P^*, \, \nabla U \rangle\big] = 0.
\end{aligned}""", []),
    r'\text{residual}^{(l)}_r \;=': (r"""\begin{aligned}
&\text{residual}^{(l)}_r = v_r - \mathrm{Attn}_l\!\big(q = k_r, \\
&\qquad K = \{k_{r'}\}, \; V = \{h^{(l-1)}_{r'}\}\big), \\
&\ell^{(l)} = \ell^{(l-1)} \\
&\qquad + \eta_l \sum_{r=1}^k \beta_r \, W_{\text{out}} \, \text{residual}^{(l)}_r.
\end{aligned}""", []),
    r'\mu^*_k(Y \mid S_{j, z})': (r"""\begin{aligned}
&\mu^*_k(Y \mid S_j) \\
&\quad = \sum_{z} \mu^*_k(Z = z \mid S_j) \, \mu^*_k(Y \mid S_{j, z}).
\end{aligned}""", []),
    'Key-Space Novelty': (r"""\begin{aligned}
&\mathrm{Score}_{\text{pol}}(v \mid \text{prefix}, \text{Consensus}) \\
&\quad = \hat{Q}_\mu(v) + c_{\text{puct}} \, \pi_0(v) \, I_{\text{goal}}(v) \\
&\qquad \times \beta_\theta(v \mid \text{prefix}) \\
&\qquad \times \frac{1}{\sqrt{1 + \mathrm{Crowding}(k_v, K_{<r})}}.
\end{aligned}""", [r'$I_{\text{goal}}(v)$: Impact / sensitivity on goal variables $X_G$.',
                   r'$\beta_\theta(v \mid \text{prefix})$: Trustworthy Stiffness: constrained & reliable in $q_\theta$.',
                   r'$1/\sqrt{1 + \mathrm{Crowding}}$: Key-Space Novelty: '
                   r"$\mathrm{Crowding} = \sum_{r'<r} \beta_{r'} \max(0, \langle k_v, k_{r'} \rangle)$."]),
    r'r_{\mathrm{LOO}}(r) \;=': (r"""\begin{aligned}
&r_{\mathrm{LOO}}(r) = \log \mu^*_k(x) - \log \mu^*_{k \setminus \{r\}}(x) \\
&\quad \approx \frac{\beta_r}{\beta_0}\Big(\ell_r(x) - \mathbb{E}_{\mu^*_k}[\ell_r(X)]\Big).
\end{aligned}""", []),
}
VARIANT = {}  # display-math index -> (narrow index, labels html with placeholders)
for i, (tex, disp) in enumerate(list(MATHS)):
    if not disp:
        continue
    key = next((k for k in NARROW if k in tex), None)
    if key:
        ntex, labels = NARROW[key]
        j = len(MATHS)
        MATHS.append((ntex, True))  # before the labels, whose inline math is appended by protect()
        VARIANT[i] = (j, ''.join(f'<span>{MD.renderInline(protect(x))}</span>' for x in labels))
    elif len(tex) > 70:
        WARN.append(f'long display equation without a multi-line form: {tex[:50]!r}')
unused = [k for k in NARROW if not any(k in t for t, d in MATHS if d)]
if unused:
    WARN.append(f'multi-line forms not used (equation changed?): {unused}')

# ---------------------------------------------------------------- typeset and write
SUB_HTML = MD.renderInline(protect(SUBTITLE)) if SUBTITLE else ''
META_HTML = MD.renderInline(protect(META)) if META else ''
rendered = katex_all()


def one(i):
    if i not in VARIANT:
        return rendered[i]
    j, labels = VARIANT[i]
    lab = f'<span class="eql">{typeset(labels)}</span>' if labels else ''
    return f'<span class="eqv"><span class="eqw">{rendered[i]}</span><span class="eqn">{rendered[j]}{lab}</span></span>'


def typeset(h):
    return re.sub(r'D?MATHX(\d+)X', lambda m: one(int(m.group(1))), h)


body, SUB_HTML, META_HTML = typeset(body), typeset(SUB_HTML), typeset(META_HTML)
leftover = re.findall(r'\$[^$<]{1,40}\$', re.sub(r'<[^>]+>', '', body))
if leftover:
    WARN.append(f'unrendered $...$: {leftover[:3]}')

CSS = """
:root{color-scheme:light;--bg:#fff;--ink:#161616;--ink2:#5b5a55;--line:#e3e2dc;--eq:#f6f6f2;--card:#fbfbf9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#151513;--ink:#eeeeea;--ink2:#a9a8a0;--line:#34332f;--eq:#1e1e1b;--card:#1b1b19}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#151513;--ink:#eeeeea;--ink2:#a9a8a0;--line:#34332f;--eq:#1e1e1b;--card:#1b1b19}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;padding:22px 16px 64px}
main{max-width:760px;margin:0 auto}
a{color:var(--ink)}
.top{display:flex;gap:14px;font-size:13px;color:var(--ink2);margin:0 0 30px}
.top a{color:var(--ink2)}
h1{font-size:clamp(30px,7vw,44px);line-height:1.08;letter-spacing:-.025em;margin:0 0 12px}
.sub{font-size:clamp(17px,3.6vw,20px);line-height:1.4;margin:0 0 14px;color:var(--ink)}
.meta{font-size:13px;line-height:1.5;color:var(--ink2);margin:0 0 46px}
hr{border:0;border-top:1px solid var(--line);margin:48px 0 40px}
h2{font-size:23px;line-height:1.25;letter-spacing:-.01em;margin:0 0 20px}
h2 a{color:inherit} h2,h3,h4{overflow-wrap:anywhere}
h3{font-size:17px;line-height:1.35;margin:40px 0 14px}
h4{font-size:15.5px;line-height:1.35;margin:30px 0 10px}
p{margin:0 0 18px}
ul,ol{margin:0 0 20px;padding-left:22px} li{margin:0 0 12px} li::marker{color:var(--ink2)}
li>ul,li>ol{margin:10px 0 0}
strong{font-weight:600}
code{font-size:.88em;background:var(--eq);padding:1px 5px;border-radius:4px}
.katex{font-size:1.12em}
.katex-display{margin:24px 0;padding:18px 16px;background:var(--eq);border-radius:10px;overflow-x:auto;overflow-y:hidden}
.katex-display>.katex{font-size:1.22em}
li .katex-display{margin:14px 0}
.eqv,.eqw,.eqn{display:block} .eqn{display:none}
.eqv.n .eqw{display:none} .eqv.n .eqn{display:block} .eqv.w .eqn{display:none}
.eql{display:block;font-size:13px;line-height:1.45;color:var(--ink2);margin:-12px 0 20px}
.eql>span{display:block;margin:3px 0} .eql .katex{font-size:1.02em}
details.more{margin:0 0 18px}
li details.more{margin:0}
details.more>summary{list-style:none;cursor:pointer}
details.more>summary::-webkit-details-marker{display:none}
details.more .mk{display:inline-block;font-size:12px;line-height:18px;color:var(--ink2);border:1px solid var(--line);border-radius:9px;padding:0 8px;margin-left:2px;vertical-align:1px;white-space:nowrap}
details.more>summary:hover .mk{border-color:var(--ink2)}
details.more[open] .mk{display:none}
details.more>p{margin:4px 0 0}
details.more>.rest{display:block;margin:2px 0 0}
figure{margin:34px 0 38px}
.pdg{display:block;width:100%;height:auto;margin:0 auto;color:var(--ink)}
.pdg .n{fill:var(--bg);stroke:currentColor;stroke-width:1.4}
.pdg .e{fill:none;stroke:currentColor;stroke-width:1.35}
.pdg .d{fill:none;stroke:currentColor;stroke-width:1.1;stroke-dasharray:4 3}
.pdg .h{fill:currentColor}
.pdg text{fill:currentColor;paint-order:stroke fill;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round}
.pdg .w{fill-opacity:.78}
.pdg .i{font-family:KaTeX_Math,"Times New Roman",serif;font-style:italic}
.pdg .r{font-family:KaTeX_Main,"Times New Roman",serif}
figcaption{max-width:560px;margin:12px auto 0;font-size:13.5px;line-height:1.5;color:var(--ink2)}
figcaption .katex{font-size:1.05em}
table.t{width:100%;border-collapse:collapse;font-size:14px;line-height:1.5;margin:8px 0 24px}
.t th{text-align:left;vertical-align:bottom;font-size:13px;font-weight:600;color:var(--ink2);border-bottom:1.5px solid var(--ink);padding:6px 10px}
.t td{vertical-align:top;border-bottom:1px solid var(--line);padding:12px 10px}
.t .lbl{display:none}
.foot{margin:56px 0 0;font-size:14px;color:var(--ink2)} .foot a{color:var(--ink2)}
@media (max-width:640px){
 body{font-size:15.5px}
 .t,.t tbody,.t tr,.t td{display:block;width:auto}
 .t thead{display:none}
 .t tr{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px 4px;margin:0 0 12px}
 .t td{border:0;padding:0 0 10px}
 .t td:first-child{font-size:15.5px}
 .t .lbl{display:block;font-size:12px;font-weight:600;color:var(--ink2);margin:0 0 1px}
 .katex-display{padding:14px 10px;margin:20px -4px}
 .eqv:not(.w):not(.n) .eqw{display:none} .eqv:not(.w):not(.n) .eqn{display:block}
}
@media (max-width:480px){
 .katex{font-size:1.04em}
 .katex-display>.katex{font-size:.88em}
}
"""
FIT_JS = """<script>(function(){var E=[].slice.call(document.querySelectorAll('.eqv'));
function fit(){E.forEach(function(e){e.className='eqv w';var d=e.querySelector('.eqw .katex-display');
if(d&&d.scrollWidth>d.clientWidth+1)e.className='eqv n';});}
fit();if(document.fonts&&document.fonts.ready)document.fonts.ready.then(fit);
var t;addEventListener('resize',function(){clearTimeout(t);t=setTimeout(fit,120);});})();</script>"""

page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>{esc(TITLE)}</title>
<meta name="description" content="{esc(re.sub(r'[*_$]', '', SUBTITLE))}">
<link rel="stylesheet" href="{KT}katex.min.css">
<style>{CSS}</style>
<main>
<nav class="top"><a href="/experiments/">← Experiments</a></nav>
<h1>{esc(TITLE)}</h1>
{f'<p class="sub">{SUB_HTML}</p>' if SUB_HTML else ''}
{f'<p class="meta">{META_HTML}</p>' if META_HTML else ''}
{body}
<p class="foot"><a href="/experiments/">← Experiments</a></p>
</main>
{FIT_JS}
</html>
"""
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(page)
print(f'wrote {OUT} ({len(page) // 1024} KB, {len(MATHS)} formulas, '
      f'{body.count("<details")} collapsed, {body.count("<figure")} figures) from {SRC}')
for w in WARN:
    print('WARN', w)
