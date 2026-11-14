import sys, re
t = open(sys.argv[1]).read()
for e in t.split('<entry>')[1:]:
    i = re.search(r'<id>(.*?)</id>', e).group(1)
    ti = re.search(r'<title>(.*?)</title>', e, re.S).group(1)
    au = re.findall(r'<name>(.*?)</name>', e)
    pub = re.search(r'<published>(.*?)</published>', e).group(1)[:10]
    print(i.split('/')[-1], '|', ' '.join(ti.split()), '|', ', '.join(au[:7]), '|', pub)
