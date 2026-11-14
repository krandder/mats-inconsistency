"""Local sanity check: render gdoc.html at a Doc-like column width and save page slices."""
import sys
from PIL import Image
from playwright.sync_api import sync_playwright

CHROME = '/home/agent/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'
src = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rdn/gdoc.html'
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
    pg = b.new_context(viewport={'width': 720, 'height': 1000}, device_scale_factor=1).new_page()
    pg.goto('file://' + src)
    pg.add_style_tag(content='body{max-width:624px;margin:24px auto}')
    pg.screenshot(path='/tmp/rdn/preview_full.png', full_page=True)
    b.close()
im = Image.open('/tmp/rdn/preview_full.png')
print(im.size)
H = 1400
for i in range(0, im.height, H):
    im.crop((0, i, im.width, min(im.height, i + H))).save(f'/tmp/rdn/preview_{i // H:02d}.png')
print('slices', (im.height + H - 1) // H)
