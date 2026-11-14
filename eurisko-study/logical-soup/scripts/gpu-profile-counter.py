#!/usr/bin/env python3
"""Sample the longest observed bend_dev kernel, not a multi-kernel runtime pass."""
import json, pathlib, shutil, sqlite3, subprocess, sys, os
p=pathlib.Path(sys.argv[1]); ns=p/'nsys.sqlite'; nsys=shutil.which('nsys') or '/usr/local/cuda/bin/nsys'; ncu=shutil.which('ncu') or '/usr/local/cuda/bin/ncu'
if not ns.exists() and (p/'nsys.nsys-rep').exists():
    with (p/'counter-export.log').open('w') as log:
        subprocess.run([nsys,'export','--type','sqlite','--output',str(ns),str(p/'nsys.nsys-rep')],stdout=log,stderr=log,timeout=60)
if not ns.exists():
    (p/'counter-unavailable.txt').write_text('No Nsight SQLite capture; refusing to guess which kernel held trader work.\n');sys.exit(0)
c=sqlite3.connect(ns)
try:
    rows=c.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
except sqlite3.Error as e:
    (p/'counter-unavailable.txt').write_text(str(e)+'\n');sys.exit(0)
kernels=[r for r in rows if 'bend_dev' in r[2]]
if not kernels: sys.exit('No bend_dev kernel records')
idx=max(range(len(kernels)),key=lambda i:kernels[i][1]-kernels[i][0]);r=kernels[idx]
selection=dict(filtered_launch_index=idx,kernel_name=r[2],duration_ns=r[1]-r[0],total_kernel_count=len(kernels),source='longest bend_dev kernel in one-epoch fused K18 Nsight Systems capture')
(p/'counter-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
metrics='smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed'
quick=len(sys.argv)>2 and sys.argv[2]=='quick'
if quick: metrics='smsp__thread_inst_executed_per_inst_executed.ratio,sm__warps_active.avg.pct_of_peak_sustained_active'
cmd=[ncu,'--replay-mode','application','--csv','--page','raw','--metrics',metrics,'--kernel-name','regex:bend_dev','--launch-skip',str(idx),'--launch-count','1','build/scaling/soup','--gpu','on','--threads','8','18','1','1','256','12']
(p/'counter-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(p/'counter-tmp').mkdir(exist_ok=True);env=os.environ.copy();env['TMPDIR']=str(p/'counter-tmp')
with (p/'counter.csv').open('w') as out,(p/'counter.stderr').open('w') as err:
    try: rc=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=75 if quick else 120).returncode
    except subprocess.TimeoutExpired: rc=124
(p/'counter-exit.txt').write_text(str(rc)+'\n')
