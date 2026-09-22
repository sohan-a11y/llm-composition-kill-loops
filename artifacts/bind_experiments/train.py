import sys; sys.path.insert(0,'/home/claude/bind')
import torch, torch.nn.functional as F, numpy as np
from core import BindingLM
torch.set_num_threads(1)

def fit(task, mode, steps=3000, d=64, bs=256, lr=3e-3, n_loops=1,
        zero_init=False, seed=0, dev='cpu', oracle=True):
    torch.manual_seed(seed); np.random.seed(seed)
    m = BindingLM(task, d=d, mode=mode, n_loops=n_loops, zero_init=zero_init, seed=seed).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    ns = task.n_states
    for s in range(steps):
        if mode == 'cot':
            x, y = task.batch_cot(bs, dev); lg, _ = m(x)
            mk = torch.zeros_like(y, dtype=torch.bool); mk[:, -task.k:] = True
            loss = F.cross_entropy(lg[mk], y[mk])
        else:
            x, y, inter = task.batch(bs, dev)
            lg, _ = m(x, oracle_inter=inter if oracle else None)
            loss = F.cross_entropy(lg[:, -1, :ns], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    return m

@torch.no_grad()
def evaluate(m, task, mode, n=2000, bs=500, dev='cpu', oracle=True):
    m.eval(); ns = task.n_states; c = 0
    for _ in range(n//bs):
        if mode == 'cot':
            s, g = task.sample(bs); ans = task.trace(s, g)[:, -1]
            pre = np.concatenate([np.full((bs,1),task.BOS), s[:,None],
                                  g+task.POFF, np.full((bs,1),task.SEP)], axis=1)
            cur = torch.from_numpy(pre).long().to(dev)
            for _ in range(task.k):
                lg, _ = m(cur); cur = torch.cat([cur, lg[:,-1,:ns].argmax(-1,keepdim=True)],1)
            c += int((cur[:,-1].cpu().numpy() == ans).sum())
        else:
            x, y, inter = task.batch(bs, dev)
            lg, _ = m(x, oracle_inter=inter if oracle else None)
            c += int((lg[:,-1,:ns].argmax(-1) == y).sum())
    m.train(); return c/(n//bs*bs)
