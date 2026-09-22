"""
S5 state tracking -- the standard decision form of the word problem.

Track ONE element through a sequence of permutations from S_5. S_5 is
non-solvable, so its word problem is NC1-complete; a fixed-depth log-precision
transformer is in uniform TC0 (Merrill & Sabharwal, TACL 2023), so unless
TC0 = NC1 it cannot do this in one forward pass for growing k.

Each CoT step is a single lookup -- given current position p and permutation g,
emit g(p) -- so the chain form is trivially learnable. That asymmetry is the
whole experiment: same model, same parameters, different amount of SERIAL
compute at inference.

Layout
  positions   0..4
  perms       5..124      (all 120 elements of S5)
  BOS 125, SEP 126
"""
import itertools, numpy as np, torch

_PERMS = list(itertools.permutations(range(5)))
PERM = np.array(_PERMS, dtype=np.int64)          # (120,5); PERM[g][p] = g(p)
N_PERM, N_POS = 120, 5
POFF, BOS, SEP, VOCAB = 5, 125, 126, 127


class S5TrackTask:
    def __init__(self, k, seed=0):
        self.k = k
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        start = self.rng.integers(0, N_POS, size=n)
        g = self.rng.integers(0, N_PERM, size=(n, self.k))
        return start, g

    def trace(self, start, g):
        """Running positions after each permutation. (n,k)"""
        n, k = g.shape
        out = np.empty((n, k), dtype=np.int64)
        cur = start.copy()
        for t in range(k):
            cur = PERM[g[:, t], cur]
            out[:, t] = cur
        return out

    def batch_oneshot(self, n):
        start, g = self.sample(n)
        tr = self.trace(start, g)
        seq = np.concatenate([np.full((n,1),BOS), start[:,None],
                              g + POFF, np.full((n,1),SEP)], axis=1)
        return torch.from_numpy(seq).long(), torch.from_numpy(tr[:,-1]).long()

    def batch_cot(self, n):
        start, g = self.sample(n)
        tr = self.trace(start, g)
        seq = np.concatenate([np.full((n,1),BOS), start[:,None],
                              g + POFF, np.full((n,1),SEP), tr], axis=1)
        x = torch.from_numpy(seq[:,:-1]).long()
        y = torch.from_numpy(seq[:,1:]).long()
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, -self.k:] = True
        return x, y, mask

    @property
    def chance(self): return 1.0 / N_POS
    @property
    def maxlen(self): return 2*self.k + 4


class S5TrackSubset(S5TrackTask):
    """Same task, but permutations drawn from a SUBSET of size `domain`.

    Peng, Narayanan & Papadimitriou (COLM 2024) prove via communication
    complexity that a transformer layer cannot compose functions once the
    DOMAINS are large enough. So composition should become learnable as the
    domain shrinks, and break above some threshold. That threshold is the
    prediction being tested.
    """
    def __init__(self, k, domain, seed=0):
        super().__init__(k, seed)
        self.domain = domain
        self.subset = np.arange(domain)          # first `domain` perms of S5

    def sample(self, n):
        start = self.rng.integers(0, N_POS, size=n)
        g = self.subset[self.rng.integers(0, self.domain, size=(n, self.k))]
        return start, g
