import sys, time, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/deep-study/wf7/probworld')
sys.dont_write_bytecode = True
import model as MM
torch.set_num_threads(int(sys.argv[1]))
m = MM.build('M', 50); m.load_state_dict(torch.load('/home/agent/workspaces/mats/work/deep-study/wf7/probworld/ckpt/M_w103_M.pt', map_location='cpu')['state']); m.eval()
tok = torch.randint(0, 3, (int(sys.argv[2]), 50))
with torch.no_grad():
    m(tok[:64]); t0 = time.time(); m(tok); print(sys.argv[1:], 'calls/s', round(len(tok) / (time.time() - t0)))
