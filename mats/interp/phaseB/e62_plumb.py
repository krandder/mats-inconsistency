"""E62 plumbing (PREREG-B amendment 70; disclosed, not results): a scratch tree /tmp/e62_plumb whose nets are the STEP-0 (untrained)
checkpoints of the own, ownf and fix nets (meta.json copied, model.pt = ckpt step_000000.pt; for fix, whose runs saved no
checkpoints, E37's ctrl step-0 checkpoint of the seed, which is the same initialisation), with placeholder evaluation
files (gap 0.05, rom_middle 0) so that every net passes the acquisition filter. Then e62_run.py runs on 60 bases there. No trained
net's quantity is computed.
usage: python e62_plumb.py   (then E62_ROOT=/tmp/e62_plumb E62_NB=60 E62_NC=60 E62_OUT=/tmp/e62_plumb/out python e62_run.py ARM)
"""
import glob, os, shutil
import numpy as np

R = '/tmp/e62_plumb'
SETS = dict(own=('runs/e60', 'e60_own_s', 'results/e60'), ownf=('runs/e60', 'e60_ownf_s', 'results/e60'), fix=('runs/e30', 'fix_s', 'results/e30'))


def main():
    for arm, (rd, pre, resd) in SETS.items():
        os.makedirs(os.path.join(R, rd), exist_ok=True); os.makedirs(os.path.join(R, resd), exist_ok=True)
        for d in sorted(glob.glob(os.path.join(rd, pre + '*'))):
            if not os.path.isdir(d):
                continue
            tag = os.path.basename(d); seed = int(tag.rsplit('_s', 1)[1])
            ck = os.path.join(d, 'ckpt', 'step_000000.pt')
            if arm == 'fix':                                   # E30's fix nets start as E37's ctrl net of the seed (paired init); no ckpt saved
                ck = os.path.join('runs/e37', 'e37_ctrl_s%d' % seed, 'ckpt', 'step_000000.pt')
            if not os.path.exists(ck):
                continue
            od = os.path.join(R, rd, tag); os.makedirs(od, exist_ok=True)
            shutil.copy(os.path.join(d, 'meta.json'), os.path.join(od, 'meta.json'))
            shutil.copy(ck, os.path.join(od, 'model.pt'))
            np.savez(os.path.join(R, resd, tag + '.npz'), gap=0.05, rom_middle=0.0, seed=seed)
        print(arm, len(glob.glob(os.path.join(R, rd, pre + '*'))), 'nets')


if __name__ == '__main__':
    main()
