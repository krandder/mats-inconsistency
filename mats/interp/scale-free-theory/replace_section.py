"""Replace the text of a markdown section (from its header line up to the next header of level <= its level)."""
import sys, re
path, header, newfile = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read().split('\n')
new = open(newfile).read().rstrip('\n').split('\n')
lvl = len(header) - len(header.lstrip('#'))
start = next(i for i, l in enumerate(s) if l.strip() == header.strip())
end = len(s)
for j in range(start + 1, len(s)):
    m = re.match(r'^(#+) ', s[j])
    if m and len(m.group(1)) <= lvl:
        end = j; break
out = s[:start] + new + [''] + s[end:]
open(path, 'w').write('\n'.join(out))
print('replaced', header, 'lines', start, end)
