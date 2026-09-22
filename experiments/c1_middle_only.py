"""C1 middle-only immunization kill-test.
Claim: middle-heavy exposure flattens U-curve without hurting ends.
Controls: position-shuffled (same data, permuted labels) must NOT match true.
Go only if middle flattens + shuffled fails else KILL (generic augmentation).
"""
import argparse, numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import TinyLM, fit_oneshot, eval_oneshot, report
NPOS = 8
def batch_fn(mode, rng):
    def fn(n):
        seq = rng.integers(0, 20, size=(n, 12))
        pos = {"mid": 6, "ends": 0}[mode] if mode in ("mid", "ends") else rng.integers(0, 12, size=n)
        if isinstance(pos, int): pos = np.full(n, pos)
        needle = rng.integers(20, 20+NPOS, size=n)
        for i in range(n): seq[i, pos[i] if isinstance(pos, np.ndarray) else pos] = needle[i]
        x = torch.from_numpy(np.concatenate([np.full((n,1),60), seq, np.full((n,1),61)],1)).long()
        y = torch.from_numpy(needle % NPOS).long()
        return x, y
    return fn
def run(steps):
    for name, mode, seed in [("baseline-mixed", "mixed", 0), ("middle-only", "mid", 1)]:
        m = TinyLM(vocab=64, d=64, layers=2, heads=4, maxlen=16)
        fit_oneshot(batch_fn(mode, np.random.default_rng(seed)), m, steps=steps, n_classes=NPOS, seed=seed)
        a = eval_oneshot(batch_fn("mixed", np.random.default_rng(9)), m, n=1000, n_classes=NPOS)
        report(name, a, 1/NPOS)
    # should-fail shuffled labels
    m = TinyLM(vocab=64, d=64, layers=2, heads=4, maxlen=16)
    r = np.random.default_rng(3)
    def shuf(n):
        x, _ = batch_fn("mid", r)(n); return x, torch.randint(0, NPOS, (n,))
    fit_oneshot(shuf, m, steps=200, n_classes=NPOS)
    a = eval_oneshot(shuf, m, n=500, n_classes=NPOS)
    report("control-shuffled-must-fail", a, 1/NPOS)
    print("GO only if middle flattens + shuffled fails else KILL")
if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=500); run(ap.parse_args().steps)
