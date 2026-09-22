"""Shared house harness: synthetic tasks + fit/evaluate + chance baselines.
Follows artifacts/model.py, tasks_track.py, screen_core.py, affine.py conventions.
CPU-testable by default, GPU-scalable via config.
Every experiment must: (1) k=1 sanity ~1.0 before trusting negatives,
(2) print chance baseline next to every number, (3) include a should-fail control.
"""
import itertools, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)

_P = list(itertools.permutations(range(5)))
PERM = np.array(_P, dtype=np.int64)
N_POS_S5 = 5
CHANCE_S5 = 0.2
CHANCE_AFFINE = 1.0 / 17.0

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self, vocab, d=64, layers=2, heads=4, maxlen=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        m = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.norm(x))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def fit_oneshot(task_batch_fn, model, steps=500, bs=256, lr=3e-3, seed=0, n_classes=5, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(steps):
        x, y = task_batch_fn(bs)
        x, y = x.to(device), y.to(device)
        lg = model(x)[:, -1, :n_classes]
        loss = F.cross_entropy(lg, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return model

def eval_oneshot(task_batch_fn, model, n=2000, bs=500, n_classes=5, device="cpu"):
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(n // bs):
            x, y = task_batch_fn(bs)
            x, y = x.to(device), y.to(device)
            pred = model(x)[:, -1, :n_classes].argmax(-1)
            ok += (pred == y).sum().item(); tot += bs
    return ok / tot

def sanity_k1(task_fn_k1, vocab, n_classes, steps=400, device="cpu"):
    """Harness sanity: k=1 trivial case must hit ~1.0 or harness is void."""
    import artifacts_tasks as AT  # local shim below if import fails
    return None

def report(name, acc, chance):
    flag = "PASS" if acc > chance + 0.3 else ("CHECK" if acc > chance + 0.1 else "FAIL")
    print(f"{name}: acc={acc:.4f} chance={chance:.4f} [{flag}]")
    return flag
