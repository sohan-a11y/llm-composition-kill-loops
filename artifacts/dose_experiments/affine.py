"""
A second composition task, structurally different from S5 tracking, to test
whether the dose threshold is a property of ONE task or a general phenomenon.

S5 tracking: state x in {0..4}, transition is a group action (permutation).
This task:   state x in Z_m,      transition is affine: x -> (a*x + b) mod m

Different algebra (abelian, not S5), different operation (affine map vs lookup
table), different vocabulary size. If the same dose-response shape appears
here, that is real evidence of generality rather than an S5 artifact.
"""

import numpy as np
import torch


class AffineChain:
    """Track x through k affine maps mod m: x_{t+1} = (a_t * x_t + b_t) mod m.

    domain = how many distinct (a,b) pairs are drawn from (mirrors S5's domain
    knob, which controls composition width).
    """

    def __init__(self, k=2, m=17, domain=None, seed=0):
        self.k, self.m = k, m
        self.domain = domain or (m * m)          # default: all (a,b) pairs, a in 1..m-1
        self.rng = np.random.default_rng(seed)
        # build a fixed pool of (a,b) pairs, a coprime-ish (nonzero mod m), index 0..domain-1
        pairs = []
        a = 1
        while len(pairs) < self.domain:
            for b in range(m):
                if len(pairs) >= self.domain:
                    break
                if a % m != 0:
                    pairs.append((a % m, b))
            a += 1
        self.pool = np.array(pairs[:self.domain], dtype=np.int64)   # (domain, 2)

        self.BOS, self.SEP = m, m + 1
        self.OP_OFF = m + 2                         # op tokens start here
        self.VOCAB = self.OP_OFF + self.domain

    def sample(self, n):
        x0 = self.rng.integers(0, self.m, size=n)
        ops = self.rng.integers(0, self.domain, size=(n, self.k))
        return x0, ops

    def trace(self, x0, ops):
        n, k = ops.shape
        out = np.empty((n, k), dtype=np.int64)
        cur = x0.copy()
        for t in range(k):
            a = self.pool[ops[:, t], 0]; b = self.pool[ops[:, t], 1]
            cur = (a * cur + b) % self.m
            out[:, t] = cur
        return out

    def batch_oneshot(self, n):
        x0, ops = self.sample(n)
        tr = self.trace(x0, ops)
        seq = np.concatenate([np.full((n, 1), self.BOS), x0[:, None],
                              ops + self.OP_OFF, np.full((n, 1), self.SEP)], axis=1)
        x = torch.from_numpy(seq).long()
        y_final = torch.from_numpy(tr[:, -1]).long()
        y_inter = torch.from_numpy(tr[:, 0]).long()      # first intermediate
        return x, y_final, y_inter

    @property
    def maxlen(self):
        return self.k + 3

    @property
    def chance(self):
        return 1.0 / self.m
