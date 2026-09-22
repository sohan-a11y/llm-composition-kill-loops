import itertools, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
_P = list(itertools.permutations(range(5)))
PERM = np.array(_P, dtype=np.int64)
N_PERM, N_POS = 120, 5
POFF, BOS, SEP, SCRATCH, VOCAB = 5, 125, 126, 127, 128

class Track:
    """Track one element of {0..4} through k permutations of S5."""
    def __init__(self, k=2, domain=120, seed=0, factored=False, n_scratch=0):
        self.k, self.domain, self.factored, self.n_scratch = k, domain, factored, n_scratch
        self.rng = np.random.default_rng(seed)
    def sample(self, n):
        s = self.rng.integers(0, N_POS, size=n)
        g = self.rng.integers(0, self.domain, size=(n, self.k))
        return s, g
    def trace(self, s, g):
        out = np.empty_like(g); cur = s.copy()
        for t in range(g.shape[1]):
            cur = PERM[g[:, t], cur]; out[:, t] = cur
        return out
    def _encode_perms(self, g):
        if not self.factored:
            return g + POFF
        n, k = g.shape
        return (PERM[g].reshape(n, k * 5))          # each perm -> its 5 images, tokens 0..4
    def batch_oneshot(self, n):
        s, g = self.sample(n); tr = self.trace(s, g)
        parts = [np.full((n,1),BOS), s[:,None], self._encode_perms(g)]
        if self.n_scratch: parts.append(np.full((n,self.n_scratch),SCRATCH))
        parts.append(np.full((n,1),SEP))
        x = torch.from_numpy(np.concatenate(parts,axis=1)).long()
        return x, torch.from_numpy(tr[:,-1]).long(), torch.from_numpy(tr[:,0]).long()
    def batch_cot(self, n):
        s, g = self.sample(n); tr = self.trace(s, g)
        seq = np.concatenate([np.full((n,1),BOS), s[:,None], self._encode_perms(g),
                              np.full((n,1),SEP), tr], axis=1)
        x = torch.from_numpy(seq[:,:-1]).long(); y = torch.from_numpy(seq[:,1:]).long()
        m = torch.zeros_like(y, dtype=torch.bool); m[:, -self.k:] = True
        return x, y, m
    @property
    def maxlen(self):
        base = 2 + (self.k*5 if self.factored else self.k) + 1 + self.n_scratch
        return base + self.k + 2

def rope(x):
    B,H,T,D = x.shape; half = D//2
    f = torch.arange(half, device=x.device).float(); f = 1.0/(10000**(f/half))
    t = torch.arange(T, device=x.device).float()
    a = t[:,None]*f[None]; cos,sin = a.cos(), a.sin()
    x1,x2 = x[...,:half], x[...,half:]
    return torch.cat([x1*cos-x2*sin, x1*sin+x2*cos], -1)

class Attn(nn.Module):
    def __init__(self,d,h,use_rope=False):
        super().__init__(); self.h,self.dh,self.use_rope=h,d//h,use_rope
        self.qkv=nn.Linear(d,3*d); self.o=nn.Linear(d,d)
    def forward(self,x,mask):
        B,T,D=x.shape
        q,k,v=self.qkv(x).chunk(3,-1)
        q,k,v=[z.view(B,T,self.h,self.dh).transpose(1,2) for z in (q,k,v)]
        if self.use_rope: q,k = rope(q), rope(k)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.o(y.transpose(1,2).reshape(B,T,D))

class Block(nn.Module):
    def __init__(self,d,h,ff_mult=4,use_rope=False):
        super().__init__()
        self.n1=nn.LayerNorm(d); self.a=Attn(d,h,use_rope); self.n2=nn.LayerNorm(d)
        self.f=nn.Sequential(nn.Linear(d,ff_mult*d),nn.GELU(),nn.Linear(ff_mult*d,d))
    def forward(self,x,mask): x=x+self.a(self.n1(x),mask); return x+self.f(self.n2(x))

class LM(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, maxlen=32, ff_mult=4,
                 use_rope=False, use_pos=True, role_emb=False, aux_head=False,
                 n_loops=1, vocab=VOCAB):
        super().__init__()
        self.use_pos, self.n_loops = use_pos and not use_rope, n_loops
        self.tok=nn.Embedding(vocab,d)
        self.pos=nn.Embedding(maxlen,d) if self.use_pos else None
        self.role=nn.Embedding(8,d) if role_emb else None
        self.blocks=nn.ModuleList([Block(d,heads,ff_mult,use_rope) for _ in range(layers)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab)
        self.aux=nn.Linear(d,N_POS) if aux_head else None
    def forward(self,idx,roles=None):
        B,T=idx.shape; x=self.tok(idx)
        if self.pos is not None: x=x+self.pos(torch.arange(T,device=idx.device))[None]
        if self.role is not None and roles is not None: x=x+self.role(roles)
        m=torch.triu(torch.full((T,T),float('-inf'),device=idx.device),1)
        for _ in range(self.n_loops):
            for b in self.blocks: x=b(x,m)
        x=self.norm(x)
        return self.head(x), (self.aux(x) if self.aux is not None else None)
    def n_params(self): return sum(p.numel() for p in self.parameters())
