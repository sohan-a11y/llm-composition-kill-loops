import sys, time, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/claude/agi')
from tasks_track import S5TrackTask, N_POS, BOS, SEP, POFF
from model import TinyLM
torch.set_num_threads(1)

def fit(task, d, steps, cot, bs=256, seed=0, lr=3e-3, loops=1, layers=2, log=0):
    torch.manual_seed(seed)
    m = TinyLM(127, d=d, n_layers=layers, n_heads=4, max_len=task.maxlen, n_loops=loops)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for s in range(steps):
        if cot:
            x,y,mask = task.batch_cot(bs); lg,_ = m(x)
            loss = F.cross_entropy(lg[mask], y[mask])
        else:
            x,y = task.batch_oneshot(bs); lg,_ = m(x)
            loss = F.cross_entropy(lg[:,-1,:N_POS], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
        if log and (s+1)%log==0: print(f'      {s+1}/{steps} loss {loss.item():.4f}',flush=True)
    return m

@torch.no_grad()
def acc_oneshot(m, task, n=4000, bs=1000):
    m.eval(); c=0
    for _ in range(n//bs):
        x,y = task.batch_oneshot(bs); lg,_ = m(x)
        c += int((lg[:,-1,:N_POS].argmax(-1)==y).sum())
    m.train(); return c/(n//bs*bs)

@torch.no_grad()
def acc_cot(m, task, n=2000, bs=500):
    """Autoregressive -- model generates its own chain, scored on FINAL state."""
    m.eval(); k=task.k; c=0
    for _ in range(n//bs):
        start,g = task.sample(bs); ans = task.trace(start,g)[:,-1]
        seq = np.concatenate([np.full((bs,1),BOS), start[:,None],
                              g+POFF, np.full((bs,1),SEP)],axis=1)
        cur = torch.from_numpy(seq).long()
        for _ in range(k):
            lg,_ = m(cur)
            nxt = lg[:,-1,:N_POS].argmax(-1,keepdim=True)
            cur = torch.cat([cur,nxt],1)
        c += int((cur[:,-1].numpy()==ans).sum())
    m.train(); return c/(n//bs*bs)
