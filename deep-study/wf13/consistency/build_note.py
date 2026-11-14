"""build_note.py : assemble NOTE.md from parts/ (tables are inserted where @@X@@ stands)."""
parts = ['p0_plain', 'p1_built', 'p2_results', 'p3_why', 'p4_registered', 'p5_limits']
txt = '\n'.join(open('parts/%s.md' % p).read().rstrip('\n') + '\n' for p in parts)
for k in ('A', 'B', 'C', 'D', 'E'):
    txt = txt.replace('@@%s@@' % k, open('parts/tables_%s.md' % k).read().rstrip('\n'))
open('NOTE.md', 'w').write(txt)
print('NOTE.md lines:', txt.count('\n'), 'longest line:', max(len(l) for l in txt.split('\n')))
