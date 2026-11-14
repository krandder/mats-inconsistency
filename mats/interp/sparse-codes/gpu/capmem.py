"""Run a script with a per-process CUDA memory cap (for several processes on one GPU): MEMFRAC (default 0.3) of the GPU.
The caching allocator frees cached blocks before failing, so a capped process does not starve its neighbours.
usage: MEMFRAC=0.3 python gpu/capmem.py gpu/run_main2.py ARGS..."""
import os, runpy, sys
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEMFRAC', '0.3')))
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name='__main__')
