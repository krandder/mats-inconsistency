"""Stage 3: render the note's mermaid flowchart with mermaid (CDN) in headless Chromium, screenshot at 2x."""
import json, os, re, sys
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright

CHROME = '/home/agent/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'
SRC = os.environ.get('RDN_SRC', '/home/agent/workspaces/mats/outbox/research-direction-note-v2-2026-10-16.md')
code = re.search(r'```mermaid\n(.*?)```', open(SRC, encoding='utf-8').read(), re.S).group(1)
direction = sys.argv[1] if len(sys.argv) > 1 else None
if direction:
    code = code.replace('flowchart LR', f'flowchart {direction}', 1)

page_html = """<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>html,body{margin:0;background:#fff} #out{display:inline-block;padding:10px;background:#fff}</style>
</head><body><div id="out"></div></body></html>"""

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
    page = browser.new_context(device_scale_factor=2, viewport={'width': 2400, 'height': 1600}).new_page()
    page.set_content(page_html, wait_until='networkidle')
    res = page.evaluate("""async (code) => {
        mermaid.initialize({startOnLoad: false, theme: 'neutral', securityLevel: 'loose',
            themeVariables: {fontFamily: 'Arial, "Liberation Sans", sans-serif', fontSize: '15px'},
            flowchart: {htmlLabels: true, padding: 12, nodeSpacing: 40, rankSpacing: 60}});
        try {
            const {svg} = await mermaid.render('g1', code);
            document.getElementById('out').innerHTML = svg;
            const s = document.querySelector('#out svg');
            const vb = s.viewBox.baseVal;
            s.removeAttribute('style'); s.setAttribute('width', vb.width); s.setAttribute('height', vb.height);
            const r = s.getBoundingClientRect();
            return {ok: true, w: r.width, h: r.height, version: mermaid.version || ''};
        } catch (e) { return {ok: false, err: String(e)}; }
    }""", code)
    print(res)
    if not res.get('ok'):
        sys.exit(1)
    page.wait_for_timeout(300)
    raw = '/tmp/rdn/img/mermaid_raw.png'
    page.locator('#out').screenshot(path=raw)
    im = Image.open(raw).convert('RGB')
    a = np.asarray(im).astype(int)
    ys, xs = np.where(a.min(axis=2) < 245)
    m = 12
    crop = im.crop((max(0, xs.min() - m), max(0, ys.min() - m), min(im.width, xs.max() + 1 + m), min(im.height, ys.max() + 1 + m)))
    out = f'/tmp/rdn/img/mermaid{"_" + direction if direction else ""}.png'
    crop.save(out, optimize=True)
    print(out, crop.size, 'css', crop.width / 2, crop.height / 2)
    browser.close()
