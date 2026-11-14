#!/bin/bash
# R6-E1 audit, checks 1 and 6: read-only evidence (timing, hashes, staged code, GPU-LOG, ledger, live VMs, GCE operations).
# Run: bash integrity.sh > integrity-output.txt 2>&1   (writes nothing else)
R6=/home/agent/workspaces/eurisko-study/li-net/r6e1
LOG=/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md
echo "== run at $(date -u +%FT%TZ)"
echo "== 1a registration record"
cat $R6/PREREG-R6-E1.sha256
stat -c '%n mtime %y' $R6/PREREG-R6-E1.sha256 $R6/results/PREREG-R6-E1-as-registered.md $R6/PREREG-R6-E1.md
sha256sum $R6/results/PREREG-R6-E1-as-registered.md $R6/PREREG-R6-E1.md
echo "== 1b first soup-gpu-r6e1- GPU-LOG line, launch log, VM-name timestamp"
grep -n 'soup-gpu-r6e1-' $LOG | head -1
head -1 /tmp/r6e1/logs/launch-r6e1-main.log 2>/dev/null || echo "(launch log /tmp/r6e1/logs/launch-r6e1-main.log missing)"
echo "== 1c PREREG vs registered copy (only section 5 may differ)"
diff $R6/results/PREREG-R6-E1-as-registered.md $R6/PREREG-R6-E1.md | head -3; echo "diff lines: $(diff $R6/results/PREREG-R6-E1-as-registered.md $R6/PREREG-R6-E1.md | grep -c '^[<>]')"
echo "== 1d staged manifests: every block vs the registered block (06:27:03Z)"
awk '/^## /{b=$2; next} NF==2{print b, $2, $1}' $R6/results/stage-manifests.txt > /home/agent/workspaces/fable-strategy/scratch/r6e1-audit/tmp/manifests.txt
python3 - <<'EOF'
rows=[l.split() for l in open('/home/agent/workspaces/fable-strategy/scratch/r6e1-audit/tmp/manifests.txt')]
blocks={}
for b,f,h in rows: blocks.setdefault(b,{})[f]=h
reg=blocks['2026-10-17T06:27:03Z']
for b,d in blocks.items():
    same=[f for f in reg if d.get(f)==reg[f]]; diff=[f for f in reg if d.get(f)!=reg[f]]; extra=[f for f in d if f not in reg]
    print(f"{b}: {len(same)}/{len(reg)} registered files identical; differing {diff}; extra (unregistered) {extra}")
EOF
echo "== 1e current files vs registered hashes"
cd $R6 && for f in e1v3.py analyze6.py test_r6e1.py ledger6.py stage6.sh remote6.sh driver6.sh launch6.sh; do
  cur=$(sha256sum $f | cut -c1-64); reg=$(awk -v f=$f '$2==f{print $1; exit}' results/stage-manifests.txt)
  echo "$f $( [ "$cur" = "$reg" ] && echo SAME || echo "CHANGED (now ${cur:0:8}, registered ${reg:0:8}, mtime $(stat -c %y $f))")"; done
for f in env6.py wm.py rollouts.py run.py partA2.py; do cur=$(sha256sum ../r5e1/$f | cut -c1-64); reg=$(awk -v f=$f '$2==f{print $1; exit}' results/stage-manifests.txt); echo "r5e1/$f $( [ "$cur" = "$reg" ] && echo SAME || echo CHANGED)"; done
cur=$(sha256sum ../smallnet/netdef.py | cut -c1-64); reg=$(awk '$2=="smallnet/netdef.py"{print $1; exit}' results/stage-manifests.txt); echo "smallnet/netdef.py $( [ "$cur" = "$reg" ] && echo SAME || echo CHANGED)"
echo "registered analyze6.py copy: $(sha256sum /tmp/r6e1/src/analyze6.py 2>/dev/null | cut -c1-16) (/tmp/r6e1/src/analyze6.py)"
echo "== 1f amendments and spec"
ls /home/agent/workspaces/mats/outbox/experiments-round-6-amendments*.md 2>&1
sha256sum /home/agent/workspaces/mats/outbox/experiments-round-6-spec.md
echo "== 6a GPU-LOG r6e1 lines"
grep -n 'soup-gpu-r6e1-' $LOG
echo "== 6b ledger6.py (R6-E1's own ledger, read-only)"
python3 $R6/ledger6.py
echo "== 6c live r6e1 VMs (gcloud, read-only) and a control listing"
timeout 90 gcloud --project=GCP_PROJECT compute instances list --filter='name~r6e1' --format='value(name,status)'; echo "r6e1 list rc=$?"
timeout 90 gcloud --project=GCP_PROJECT compute instances list --filter='name~soup-gpu' --format='value(name,status)'; echo "control list rc=$?"
echo "== 6d GCE operations on r6e1 VMs (insert / preempt / delete times, UTC-7 as printed)"
timeout 90 gcloud --project=GCP_PROJECT compute operations list --filter="targetLink~soup-gpu-r6e1" --format="value(operationType,status,insertTime,targetLink.basename())"
echo "== done $(date -u +%FT%TZ)"
