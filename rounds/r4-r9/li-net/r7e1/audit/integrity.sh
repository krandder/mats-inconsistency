#!/bin/bash
# R7-E1 audit, check 1 (timing and integrity). Read-only. Usage: bash integrity.sh > integrity-output.txt
set -u
B=/home/agent/workspaces/eurisko-study
R7=$B/li-net/r7e1
echo "run at $(date -u +%FT%TZ)"
echo "== registration record"
cat $R7/PREREG-R7-E1.sha256
stat -c '%y  %n' $R7/PREREG-R7-E1.sha256 $R7/results/PREREG-R7-E1-as-registered.md $R7/PREREG-R7-E1.md
echo "== hashes: as-registered copy vs registration record, current PREREG"
sha256sum $R7/results/PREREG-R7-E1-as-registered.md $R7/PREREG-R7-E1.md
echo "== diff as-registered -> current PREREG"
diff $R7/results/PREREG-R7-E1-as-registered.md $R7/PREREG-R7-E1.md | head -5; echo "(diff hunks: $(diff $R7/results/PREREG-R7-E1-as-registered.md $R7/PREREG-R7-E1.md | grep -c '^[0-9]'))"
echo "== first r7e1 GPU-LOG lines and launch-log times"
grep -n 'r7e1' $B/logical-soup/GPU-LOG.md | head -3
grep -h 'LAUNCH\|END\|RETRY' /tmp/r7e1/logs/launch-r7e1-main.log /tmp/r7e1/logs/launch-r7e1-main2.log
echo "== any r7e1 GPU-LOG line before 09:39Z?"; awk '/r7e1/ && substr($0,3,17) < "2026-10-17T09:39Z"' $B/logical-soup/GPU-LOG.md | wc -l
echo "== spec and amendment hashes / mtimes (registered: spec cbece88d..., amendments c377811d...)"
sha256sum /home/agent/workspaces/mats/outbox/experiments-round-7-spec.md /home/agent/workspaces/mats/outbox/experiments-round-7-amendments*.md
stat -c '%y  %n' /home/agent/workspaces/mats/outbox/experiments-round-7-*.md
echo "== verbatim excerpts in the registered PREREG"
python3 - <<'PY'
import hashlib,re
reg=open('/home/agent/workspaces/eurisko-study/li-net/r7e1/results/PREREG-R7-E1-as-registered.md').read()
spec=open('/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md').read()
am=open('/home/agent/workspaces/mats/outbox/experiments-round-7-amendments.md').read()
i=reg.index('### 3.2 Pre-Registered Decision Rules for `R7-E1`'); j=reg.index('### 2b.')
ex=reg[i:j].rstrip('\n')
print('3.2 excerpt in PREREG: %d chars, sha256 %s, verbatim substring of spec: %s' % (len(ex), hashlib.sha256(ex.encode()).hexdigest(), ex in spec))
k=reg.index('## Part 2: `R7-E1`'); l=reg.index('## 3. How each rule is computed')
ex2=reg[k:l].rstrip('\n')
print('amendment Part 2 excerpt: %d chars, sha256 %s, verbatim substring of amendments: %s, equals file tail: %s' % (len(ex2), hashlib.sha256(ex2.encode()).hexdigest(), ex2 in am, am.rstrip('\n').endswith(ex2)))
PY
echo "== current file hashes (registered values in PREREG header and stage manifest of 09:39:09Z)"
cd $R7 && sha256sum e1v4.py analyze7.py test_r7e1.py ledger7.py stage7.sh remote7.sh driver7.sh launch7.sh render_page7.py site_patch7.py
sha256sum $B/li-net/r6e1/e1v3.py $B/li-net/r5e1/{env6,wm,rollouts,run,partA2}.py $B/li-net/smallnet/netdef.py
stat -c '%y  %n' $R7/*.py $R7/*.sh
echo "== stage manifests: blocks and per-block comparison with the registration block"
python3 - <<'PY'
blocks=[];cur=None
for line in open('/home/agent/workspaces/eurisko-study/li-net/r7e1/results/stage-manifests.txt'):
    line=line.rstrip('\n')
    if line.startswith('## '):
        cur={'hdr':line,'files':{}}; blocks.append(cur)
    elif line.strip():
        h,f=line.split(None,1); cur['files'][f.strip()]=h
reg=[b for b in blocks if '09:39:09Z' in b['hdr']][0]['files']
for b in blocks:
    same=[f for f in reg if b['files'].get(f)==reg[f]]
    diff=[f for f in reg if f in b['files'] and b['files'][f]!=reg[f]]
    miss=[f for f in reg if f not in b['files']]
    extra={f:h[:8] for f,h in b['files'].items() if f not in reg}
    print(b['hdr'], '| registered files identical: %d/%d | changed: %s | missing: %s | extra: %s' % (len(same),len(reg),diff,miss,extra))
PY
echo "== checkpoints staged at 10:35:37Z vs synced files and deviation-1 prefixes"
sha256sum $R7/ckpt/s*/theta_real_v2.npz /tmp/r7e1/src/ckpt/s*/theta_real_v2.npz
stat -c '%y  %n' $R7/ckpt/s*/theta_real_v2.npz
echo "== results and analysis file times"
stat -c '%y  %n' $R7/results/main/s*/*.json $R7/results/main/analysis/* $R7/results/main-preempted/s*/summary.json
echo "== rc.json / setup.json"
cat $R7/results/main/s*/rc.json; echo; cat $R7/results/main/setup.json; echo
echo "== remote rc lines in the driver log (driver7.sh always exits 0)"
grep -n 'remote rc' /tmp/r7e1/logs/main/driver.txt
grep -n 'RUN-END' $B/logical-soup/GPU-LOG.md | grep r7e1
echo "== RESULTS-R7-E1.md present?"; ls -la --time-style=full-iso $R7/RESULTS-R7-E1.md 2>&1
