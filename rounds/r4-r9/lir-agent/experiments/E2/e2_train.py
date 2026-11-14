"""python3 e2_train.py --leak none|P|K <protocol.train args>"""
import sys
import e2lib

argv = sys.argv[1:]
leak = "none"
if "--leak" in argv:
    i = argv.index("--leak"); leak = argv[i + 1]; del argv[i:i + 2]
T = e2lib.install(leak)
print("E2 leak mode:", leak, flush=True)
T.main(argv)
