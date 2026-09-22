"""
All open items, one harness.

THE QUESTION
------------
A transformer that predicts the correct intermediate value in its hidden state
(100% aux accuracy) still fails the composition task. Re-entering that value as
a TOKEN fixes it. Why?

Hypothesis (classical variable binding, Smolensky 1990 / Plate 1995): a token
gives the value an ADDRESS. A residual-stream vector has none -- it is
superposed with everything else and cannot be selectively retrieved.

THE ARMS (the point is that these can separate three different explanations)
---------------------------------------------------------------------------
  baseline      no intervention                          -> expect chance
  aux_only      predict intermediates, no memory         -> expect chance (the known failure)
  raw_multi     k-1 SEPARATE slots, raw values, no roles  -> "addressability, no binding"
  sum_noroles   1 slot = plain sum of values             -> "compression, no binding"
  hrr           1 superposed vector, role-bound,
                retrieved by unbinding                    -> "binding + compression"
  cot           token-level chain of thought              -> known-good upper bound

WHAT EACH OUTCOME MEANS
-----------------------
  hrr ~ cot  and  sum_noroles ~ chance   -> binding is real and is the mechanism
  raw_multi ~ hrr                        -> ADDRESSABILITY is what matters, not
                                            binding; the classical theory adds
                                            nothing here (this is the trap door)
  hrr ~ sum_noroles ~ chance             -> binding hypothesis is dead
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

# ----------------------------------------------------------------- HRR ops
def hrr_bind(x, y):
    n = x.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n), n=n)

def hrr_inv(y):
    return torch.cat([y[..., :1], y[..., 1:].flip(-1)], dim=-1)

def hrr_unbind(b, y):
    return hrr_bind(b, hrr_inv(y))


# ----------------------------------------------------------------- tasks
_P = list(itertools.permutations(range(5)))
PERM = np.array(_P, dtype=np.int64)
N_POS = 5

class S5Task:
    """Track one element through k permutations of S5. n_inter = k-1."""
    name = "S5"
    def __init__(self, k=3, domain=120, seed=0):
        self.k, self.domain = k, domain
        self.n_states = N_POS
        self.POFF, self.BOS, self.SEP = 5, 125, 126
        self.VOCAB = 127
        self.rng = np.random.default_rng(seed)
    def sample(self, n):
        return (self.rng.integers(0, N_POS, size=n),
                self.rng.integers(0, self.domain, size=(n, self.k)))
    def trace(self, s, g):
        out = np.empty_like(g); cur = s.copy()
        for t in range(g.shape[1]):
            cur = PERM[g[:, t], cur]; out[:, t] = cur
        return out
    def batch(self, n, dev):
        s, g = self.sample(n); tr = self.trace(s, g)
        seq = np.concatenate([np.full((n,1),self.BOS), s[:,None],
                              g+self.POFF, np.full((n,1),self.SEP)], axis=1)
        return (torch.from_numpy(seq).long().to(dev),
                torch.from_numpy(tr[:,-1]).long().to(dev),
                torch.from_numpy(tr[:,:-1]).long().to(dev))   # intermediates
    def batch_cot(self, n, dev):
        s, g = self.sample(n); tr = self.trace(s, g)
        seq = np.concatenate([np.full((n,1),self.BOS), s[:,None], g+self.POFF,
                              np.full((n,1),self.SEP), tr], axis=1)
        return (torch.from_numpy(seq[:,:-1]).long().to(dev),
                torch.from_numpy(seq[:,1:]).long().to(dev))
    @property
    def maxlen(self): return self.k + 3 + self.k + 4
    @property
    def chance(self): return 1.0 / N_POS


class AffineTask:
    """x -> (a*x+b) mod m, k times. Structurally different algebra from S5."""
    name = "Affine"
    def __init__(self, k=3, m=17, domain=120, seed=0):
        self.k, self.m, self.domain = k, m, domain
        self.n_states = m
        self.rng = np.random.default_rng(seed)
        pairs, a = [], 1
        while len(pairs) < domain:
            for b in range(m):
                if len(pairs) >= domain: break
                if a % m != 0: pairs.append((a % m, b))
            a += 1
        self.pool = np.array(pairs[:domain], dtype=np.int64)
        self.BOS, self.SEP = m, m+1
        self.POFF = m+2
        self.VOCAB = self.POFF + domain
    def sample(self, n):
        return (self.rng.integers(0, self.m, size=n),
                self.rng.integers(0, self.domain, size=(n, self.k)))
    def trace(self, x0, ops):
        out = np.empty_like(ops); cur = x0.copy()
        for t in range(ops.shape[1]):
            a = self.pool[ops[:,t],0]; b = self.pool[ops[:,t],1]
            cur = (a*cur + b) % self.m; out[:,t] = cur
        return out
    def batch(self, n, dev):
        x0, ops = self.sample(n); tr = self.trace(x0, ops)
        seq = np.concatenate([np.full((n,1),self.BOS), x0[:,None],
                              ops+self.POFF, np.full((n,1),self.SEP)], axis=1)
        return (torch.from_numpy(seq).long().to(dev),
                torch.from_numpy(tr[:,-1]).long().to(dev),
                torch.from_numpy(tr[:,:-1]).long().to(dev))
    def batch_cot(self, n, dev):
        x0, ops = self.sample(n); tr = self.trace(x0, ops)
        seq = np.concatenate([np.full((n,1),self.BOS), x0[:,None], ops+self.POFF,
                              np.full((n,1),self.SEP), tr], axis=1)
        return (torch.from_numpy(seq[:,:-1]).long().to(dev),
                torch.from_numpy(seq[:,1:]).long().to(dev))
    @property
    def maxlen(self): return self.k + 3 + self.k + 4
    @property
    def chance(self): return 1.0 / self.m


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h, zero_init=False):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        if zero_init:
            # identity at init -- stabilises gradient flow through many loops
            nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)
            nn.init.zeros_(self.ff[-1].weight);        nn.init.zeros_(self.ff[-1].bias)
    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))


class BindingLM(nn.Module):
    """2-layer transformer with an optional role-bound memory between layers."""
    def __init__(self, task, d=64, heads=4, mode="baseline", n_loops=1,
                 zero_init=False, seed=0):
        super().__init__()
        self.task, self.d, self.mode = task, d, mode
        self.n_inter = task.k - 1
        self.n_loops = n_loops
        self.tok = nn.Embedding(task.VOCAB, d)
        self.pos = nn.Embedding(task.maxlen + 8, d)
        self.l1 = Block(d, heads, zero_init and n_loops > 1)
        self.l2 = Block(d, heads, zero_init and n_loops > 1)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, task.VOCAB)
        # aux heads: one per intermediate
        self.aux = nn.ModuleList([nn.Linear(d, task.n_states)
                                  for _ in range(self.n_inter)]) \
                   if mode != "baseline" and mode != "cot" else None
        # FIXED (frozen) role hypervectors -- isolates binding from representation learning
        g = torch.Generator().manual_seed(seed + 999)
        self.register_buffer("roles", torch.randn(max(self.n_inter,1), d, generator=g) / math.sqrt(d))
        # value embedding table used to turn a predicted distribution into a filler
        self.val_embed = nn.Embedding(task.n_states, d)

    def _fillers(self, h_last):
        """Predicted intermediates -> continuous filler vectors (no discretisation)."""
        outs, logits = [], []
        for i, head in enumerate(self.aux):
            lg = head(h_last)                       # (B, n_states)
            logits.append(lg)
            p = F.softmax(lg, dim=-1)
            outs.append(p @ self.val_embed.weight)  # (B, d) expected embedding
        return outs, logits

    def _memory_slots(self, fillers):
        """The heart of the experiment: how the intermediates are made available."""
        if self.mode == "raw_multi":
            return torch.stack(fillers, 1)                       # (B, n_inter, d) separate addresses
        if self.mode == "sum_noroles":
            return torch.stack(fillers, 1).sum(1, keepdim=True)  # (B, 1, d) mush
        if self.mode == "hrr":
            M = sum(hrr_bind(f, self.roles[i]) for i, f in enumerate(fillers))
            slots = [hrr_unbind(M, self.roles[i]) for i in range(len(fillers))]
            return torch.stack(slots, 1)                         # (B, n_inter, d) recovered
        return None

    def forward(self, idx, oracle_inter=None):
        """oracle_inter: (B, n_inter) the TRUE intermediate values.

        Oracle fillers separate two questions that k>=3 otherwise conflates:
          CAN THE MODEL COMPUTE the intermediate?  <- itself the composition cliff
          CAN THE MODEL USE an intermediate it has? <- what binding actually claims
        The original finding was that a model with 100% aux accuracy STILL failed,
        so the failure is in USE. Every arm receives identical oracle values, so the
        only thing differing between arms is HOW the value is made available.
        """
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        m = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for _ in range(self.n_loops):
            x = self.l1(x, m)
        aux_logits = None
        if self.mode not in ("baseline", "cot"):
            if oracle_inter is not None:
                fillers = [self.val_embed(oracle_inter[:, i]) for i in range(self.n_inter)]
            else:
                fillers, aux_logits = self._fillers(x[:, -1])
            slots = self._memory_slots(fillers)
            if slots is not None:
                x = torch.cat([slots, x], dim=1)                 # PREPEND so causal mask sees them
                T2 = x.shape[1]
                m = torch.triu(torch.full((T2, T2), float("-inf"), device=idx.device), 1)
        for _ in range(self.n_loops):
            x = self.l2(x, m)
        x = self.norm(x)
        return self.head(x), aux_logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
