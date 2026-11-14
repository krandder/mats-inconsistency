import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
with open(sys.argv[2], "w") as f:
    for i, p in enumerate(r.pages):
        f.write(f"\n=== page {i+1} ===\n")
        try:
            f.write(p.extract_text() or "")
        except Exception as e:
            f.write(f"[extract error {e}]")
print(len(r.pages), "pages")
