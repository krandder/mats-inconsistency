import re, sys, urllib.request, time
ids = sys.argv[1]
url = "https://export.arxiv.org/api/query?id_list=%s&max_results=50" % ids
x = urllib.request.urlopen(url, timeout=90).read().decode()
open('/home/agent/workspaces/mats/work/scale-free-theory/cites/batch_%d.xml' % int(time.time()), 'w').write(x)
for e in x.split('<entry>')[1:]:
    t = re.search(r'<title>(.*?)</title>', e, re.S).group(1)
    i = re.search(r'<id>(.*?)</id>', e).group(1)
    a = re.findall(r'<name>(.*?)</name>', e)
    s = re.search(r'<summary>(.*?)</summary>', e, re.S).group(1)
    p = re.search(r'<published>(.*?)</published>', e).group(1)[:10]
    print(i, '|', p, '|', ' '.join(t.split()), '|', ', '.join(a[:6]))
    print('   ', ' '.join(s.split())[:1100])
    print()
