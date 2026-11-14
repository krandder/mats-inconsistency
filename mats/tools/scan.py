#!/usr/bin/env python3
"""Regex secret scan over staged trees (mats/tools/scan.py). usage: [SHOW=n] [KNOWN_SECRETS_FILE=f] scan.py DIR...
Prints hit counts per pattern and up to SHOW hits each (excerpts masked)."""
import os, re, sys
PATS = {
 'private-key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
 'aws-key-id': r'\b(AKIA|ASIA)[0-9A-Z]{16}\b',
 'aws-secret': r'(?i)aws_secret_' r'access_key',   # split so the scanner does not flag its own source
 'github-token': r'\b(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})',
 'sk-key': r'\bsk-(ant-|proj-)?[A-Za-z0-9_-]{20,}',
 'google-api': r'\bAIza[0-9A-Za-z_-]{35}\b|\bya29\.[0-9A-Za-z_-]{20,}',
 'slack': r'\bxox[abposr]-[0-9A-Za-z-]{10,}',
 'telegram-bot': r'\b\d{8,10}:AA[0-9A-Za-z_-]{30,}\b',
 'hf-token': r'\bhf_[A-Za-z0-9]{30,}\b',
 'stripe': r'\b[rs]k_(live|test)_[0-9A-Za-z]{10,}',
 'jwt': r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
 'bearer': r'(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}',
 'assign-secret': r'''(?i)\b(api[_-]?key|apikey|secret|access[_-]?token|auth[_-]?token|token|passw(or)?d|pwd)\b\s*[:=]\s*['"]?[A-Za-z0-9/+_.=-]{12,}''',
 'guest-key': r'(?i)[?&]?\bkey=[0-9a-f]{32}\b',
 'url-key-param': r'(?i)[?&](k|key|token|auth|sig|signature|access_token)=[A-Za-z0-9_-]{16,}',
 'url-creds': r'\b[a-z][a-z0-9+.-]*://[^/\s:@\'"<>]+:[^/\s@\'"<>]+@',
 'hex32': r'(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])',
 'ssh-key-path-content': r'ssh-(rsa|ed25519) AAAA[0-9A-Za-z+/]{20,}',
}
# Optional: literal secrets to look for (one per line) from a file kept outside the repo, e.g. the gate's guest keys.
if os.environ.get('KNOWN_SECRETS_FILE'):
    for i, lit in enumerate(x.strip() for x in open(os.environ['KNOWN_SECRETS_FILE']) if x.strip()):
        PATS[f'known-secret-{i}'] = re.escape(lit)
RX = {k: re.compile(v) for k, v in PATS.items()}
BIN = {'.png', '.pdf', '.jpg', '.jpeg', '.gif', '.zip', '.gz'}
hits = {}
for root in sys.argv[1:]:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.islink(p) or os.path.splitext(f)[1].lower() in BIN: continue
            try: txt = open(p, 'r', errors='replace').read()
            except Exception as e: print('ERR', p, e); continue
            for i, line in enumerate(txt.splitlines(), 1):
                for k, rx in RX.items():
                    for m in rx.finditer(line):
                        s = m.group(0)
                        hits.setdefault(k, []).append((p, i, s[:6] + '…' + s[-4:] if len(s) > 12 else s, line[max(0, m.start()-50):m.end()+30]))
for k in PATS:
    v = hits.get(k, [])
    print(f'== {k}: {len(v)} hits')
    for p, i, s, ctx in v[:int(os.environ.get('SHOW', 8))]:
        print(f'   {p}:{i}: {s} | {ctx.strip()[:160]}')
