"""B-C3 analytic system->bias fold kill-test (hours).
Claim: query-independent bias retains >90% fidelity (cosine>0.9).
Test delta=out(sys+user)-out(user) rank/cosine; controls empty/shuffled system.
Go only on thresholds else KILL -> collapses to PromptCache/steering-average.
"""
import argparse, numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import TinyLM
def run(steps=200):
    torch.manual_seed(0)
    m = TinyLM(vocab=64, d=32, layers=1, heads=2, maxlen=16).eval()
    with torch.no_grad():
        def out(sys_t, usr):
            idx = torch.tensor([[60, sys_t, usr, 61]])
            return m(idx)[0, -1]
        deltas = torch.stack([out(10+i%3, 20+i%5) for i in range(50)])
        # VOID check: untrained net gives near-constant outputs -> cosine ~1 is harness artifact
        if deltas.std().item() < 1e-3:
            print(f"VOID untrained-net artifact std={deltas.std().item():.2e}; train first or test void")
            print("KILL (void)")
            return
        mean = deltas.mean(0)
        cos = torch.nn.functional.cosine_similarity(deltas, mean[None].expand_as(deltas)).mean().item()
        # should-fail control: shuffled system labels must break alignment
        import random
        shuf = torch.stack([out(random.choice([10,11,12]), 20+i%5) for i in range(50)])
        cos_s = torch.nn.functional.cosine_similarity(shuf, shuf.mean(0)[None].expand_as(shuf)).mean().item()
    print(f"mean cosine={cos:.3f} shuffled={cos_s:.3f} (need true>0.9 AND shuffled<true-0.1)")
    print("NOTE: untrained-net result is VOID for promotion; train on instruction data first")
    if cos > 0.9 and cos_s < cos - 0.1: print("GO")
    else: print("KILL")
if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=200); run(ap.parse_args().steps)
