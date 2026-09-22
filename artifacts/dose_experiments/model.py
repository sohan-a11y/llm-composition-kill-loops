import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,d,h):
        super().__init__()
        self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,h,batch_first=True)
        self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x,mask):
        h=self.n1(x); a,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        x=x+a; return x+self.ff(self.n2(x))

class LM(nn.Module):
    def __init__(self, vocab, d=64, layers=2, heads=4, maxlen=32,
                 aux_head=False, n_aux_classes=None):
        super().__init__()
        self.tok=nn.Embedding(vocab,d); self.pos=nn.Embedding(maxlen,d)
        self.blocks=nn.ModuleList([Block(d,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab)
        self.aux=nn.Linear(d,n_aux_classes) if aux_head else None
    def forward(self,idx):
        B,T=idx.shape
        x=self.tok(idx)+self.pos(torch.arange(T,device=idx.device))[None]
        m=torch.triu(torch.full((T,T),float('-inf'),device=idx.device),1)
        for b in self.blocks: x=b(x,m)
        x=self.norm(x)
        return self.head(x), (self.aux(x) if self.aux is not None else None)
    def n_params(self): return sum(p.numel() for p in self.parameters())
