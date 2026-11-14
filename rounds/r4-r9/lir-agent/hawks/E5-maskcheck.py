# Run against harness/model/roles.py as of 07:01Z (old build_mask(types, items, content_mask, item_bias) API; since replaced). Output: E5-maskcheck.out
import sys, torch
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/lir-agent/harness")
from model.roles import build_mask, goal_reaches, T, TYPE_NAMES
types=torch.tensor([[T.CTRL,T.CTRL,T.GOAL,T.GOAL,T.EVID,T.EVID,T.SIT,T.SIT,T.PLAN,T.FCIN,T.FC,T.FC]])
items=torch.tensor([[-1,-1,-1,-1,0,1,-1,-1,-1,-1,-1,-1]])
m=build_mask(types,items,True)
allow=(m[0,0]>-1e8)
for i in range(types.shape[1]):
    print(i, TYPE_NAMES[types[0,i]], "sees", [f"{j}:{TYPE_NAMES[types[0,j]]}" for j in range(types.shape[1]) if allow[i,j]])
print("goal_reaches", goal_reaches(m,types).tolist())
bias=torch.tensor([[-5.0,0.0]])
mb=build_mask(types,items,True,bias)
print("router bias on content row SIT(6)->EVID0(4):", mb[0,0,6,4].item(), " FC(10)->EVID0(4):", mb[0,0,10,4].item(), " PLAN(8)->EVID0(4):", mb[0,0,8,4].item())
print("goal_reaches with router bias", goal_reaches(mb,types).tolist())
