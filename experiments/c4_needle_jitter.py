"""C4 needle-position jitter training kill-test (STRONG candidate).
Claim: uniform per-batch needle-depth jitter flattens worst-depth NIAH vs fixed.
House style: k=1 sanity ~1.0, chance baseline, should-fail control (shuffled positions).
CPU default steps=500; scale steps=3000 on T4/Kaggle.
Go only if worst-depth +10pp else KILL.
"""
import argparse, numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import TinyLM, fit_oneshot, eval_oneshot, report

VOCAB, NPOS = 64, 8
def make_batch(jitter, rng):
    def fn(n):
        seq = rng.integers(0, 20, size=(n, 12))
        if jitter:
            d = rng.integers(0, 12, size=n)
        else:
            d = np.full(n, 6)
        needle = rng.integers(20, 20 + NPOS, size=n)
        for i in range(n):
            seq[i, d[i]] = needle[i]
        x = torch.from_numpy(np.concatenate(
            [np.full((n, 1), 60), seq, np.full((n, 1), 61)], axis=1)).long()
        # query asks value at fixed probe slot 6 -> jitter trains robustness
        y = torch.from_numpy(seq[np.arange(n), np.minimum(d, 11)] % NPOS).long()
        return x, y
    return fn

def run(steps):
    rng = np.random.default_rng(0)
    # sanity k=1: single-slot copy must hit ~1.0
    m0 = TinyLM(vocab=64, d=32, layers=1, heads=2, maxlen=16)
    fit_oneshot(make_batch(False, np.random.default_rng(1)), m0, steps=min(steps, 400), n_classes=NPOS)
    s = eval_oneshot(make_batch(False, np.random.default_rng(2)), m0, n=500, n_classes=NPOS)
    report("sanity-fixed", s, 1.0 / NPOS)
    if s < 0.9:
        print("VOID harness broken"); return
    for name, jit, seed in [("fixed", False, 10), ("jitter", True, 11), ("shuffled-ctrl", False, 12)]:
        m = TinyLM(vocab=64, d=64, layers=2, heads=4, maxlen=16)
        r = np.random.default_rng(seed)
        fit_oneshot(make_batch(jit, r), m, steps=steps, n_classes=NPOS, seed=seed)
        if name == "shuffled-ctrl":
            # should-fail: random labels -> must stay at chance
            def randfn(n):
                x, _ = make_batch(False, r)(n)
                return x, torch.randint(0, NPOS, (n,))
            a = eval_oneshot(randfn, m, n=1000, n_classes=NPOS)
            report("control-shuffled-must-fail", a, 1.0 / NPOS)
            assert a < 0.25, "control did not fail -> test void"
            continue
        a = eval_oneshot(make_batch(jit, np.random.default_rng(99)), m, n=1000, n_classes=NPOS)
        report(name, a, 1.0 / NPOS)
    print("GO if jitter worst-depth +10pp else KILL")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=500)
    run(ap.parse_args().steps)
