"""C9-reframed latent exact accumulator kill-test (core-gap).
Claim: learned scalar register computes Affine intermediates with NO extra tokens.
Tests §1b open gap: USE (given) vs COMPUTE (self-produced).
Arms: (a) CoT upper bound reference, (b) accumulator slot, (c) same-param depth control.
Go only if (b) >> (c) near (a) 3 seeds else KILL (proves re-entry required).
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import TinyLM, report
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "artifacts"))
from affine import AffineChain

class AccumLM(nn.Module):
    """TinyLM + 1 scalar register updated per position by MLP (no extra tokens)."""
    def __init__(self, vocab, d=64, maxlen=16):
        super().__init__()
        from common import Block
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        self.b1 = Block(d, 4); self.acc = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, 1))
        self.b2 = Block(d, 4); self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 17)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        m = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
        x = self.b1(x, m)
        reg = torch.zeros(B, T, 1, device=idx.device)
        for t in range(T):
            reg[:, t:t+1] = self.acc(torch.cat([x[:, t:t+1], reg[:, max(0, t-1):t] if t else reg[:, :1]], -1))
        x = x + reg.expand(-1, -1, x.shape[-1]) * 0.1
        x = self.b2(x, m)
        return self.head(self.norm(x))

def run(steps):
    task = AffineChain(k=2, m=17, domain=40, seed=0)
    # k=1 sanity
    t1 = AffineChain(k=1, m=17, domain=5, seed=0)
    m0 = TinyLM(vocab=t1.VOCAB, d=32, layers=1, heads=2, maxlen=t1.maxlen + 2)
    def b0(n):
        x, yf, _ = t1.batch_oneshot(n); return x, yf
    from common import fit_oneshot, eval_oneshot
    fit_oneshot(b0, m0, steps=min(steps, 400), n_classes=17)
    s = eval_oneshot(b0, m0, n=500, n_classes=17)
    report("sanity-k1", s, 1/17)
    if s < 0.9: print("VOID"); return
    def bb(n):
        x, yf, _ = task.batch_oneshot(n); return x, yf
    for name, mk in [("depth-control", lambda: TinyLM(vocab=task.VOCAB, d=64, layers=2, heads=4, maxlen=task.maxlen+2)),
                     ("accumulator", lambda: AccumLM(vocab=task.VOCAB, d=64, maxlen=task.maxlen+2))]:
        accs = []
        for seed in [0, 1, 2]:
            m = mk()
            fit_oneshot(bb, m, steps=steps, n_classes=17, seed=seed)
            a = eval_oneshot(bb, m, n=1000, n_classes=17)
            accs.append(a)
        import numpy as np
        print(f"{name}: {np.mean(accs):.4f} +- {np.std(accs):.4f} chance={1/17:.4f}")
    print("GO only if accumulator >> depth-control near CoT else KILL")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=500)
    run(ap.parse_args().steps)
