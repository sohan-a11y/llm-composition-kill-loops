"""
Metacognitive Input Channel -- kill test.

Task: model answers q -> f(q), then emits a VERDICT token judging its OWN
sampled answer: RIGHT or WRONG. Answers are sampled on-policy at temperature 1,
so unlucky draws happen. The verdict position sees q and the sampled token --
but NOT how probable that token was.

Arms:
  A  true   channel = [p(sampled answer), entropy]   fed into verdict position
  B  shuf   channel = same features, shuffled across the batch  (kill control)
  C  none   channel = zeros

If A ~ B ~ C: the model already recovers its own confidence internally. Idea dead.
If A > B and A > C: the fed-back signal carries information the model lacks.
"""
import sys, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)

NQ, NA = 64, 16
BOS, SEP, VPOS = NA, NA+1, NA+2
Q_OFF = NA+3
RIGHT, WRONG = 0, 1
VOCAB = Q_OFF + NQ

class Block(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.n1=nn.LayerNorm(d); s.a=nn.MultiheadAttention(d,h,batch_first=True)
        s.n2=nn.LayerNorm(d); s.f=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x,m):
        h=s.n1(x); a,_=s.a(h,h,h,attn_mask=m,need_weights=False); x=x+a; return x+s.f(s.n2(x))

class LM(nn.Module):
    def __init__(s,d=64,L=2):
        super().__init__()
        s.tok=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8,d)
        s.blocks=nn.ModuleList([Block(d,4) for _ in range(L)])
        s.norm=nn.LayerNorm(d); s.head=nn.Linear(d,VOCAB)
        s.meta=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d))   # the channel
        s.verd=nn.Linear(d,2)
    def forward(s,idx,meta=None,meta_pos=None):
        B,T=idx.shape
        x=s.tok(idx)+s.pos(torch.arange(T))[None]
        if meta is not None:
            add=torch.zeros_like(x); add[:,meta_pos]=s.meta(meta); x=x+add
        m=torch.triu(torch.full((T,T),float('-inf')),1)
        for b in s.blocks: x=b(x,m)
        x=s.norm(x); return s.head(x), s.verd(x)

def run(arm, seed, stage1=600, stage2=1500, bs=256):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    f=rng.integers(0,NA,size=NQ)                       # the true mapping
    m=LM(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.01)
    # stage 1: learn to answer, deliberately undertrained -> genuine per-q uncertainty
    for _ in range(stage1):
        q=rng.integers(0,NQ,size=bs)
        x=torch.tensor(np.stack([np.full(bs,BOS),q+Q_OFF,np.full(bs,SEP)],1)).long()
        lg,_=m(x); loss=F.cross_entropy(lg[:,-1,:NA],torch.tensor(f[q]).long())
        opt.zero_grad(); loss.backward(); opt.step()
    # stage 2: on-policy answers + verdict
    for _ in range(stage2):
        q=rng.integers(0,NQ,size=bs)
        pre=torch.tensor(np.stack([np.full(bs,BOS),q+Q_OFF,np.full(bs,SEP)],1)).long()
        with torch.no_grad():
            lg,_=m(pre); p=F.softmax(lg[:,-1,:NA],-1)
            a=torch.multinomial(p,1).squeeze(1)
            pa=p.gather(1,a[:,None]).squeeze(1)
            ent=-(p*(p+1e-12).log()).sum(-1)/math.log(NA)
        feat=torch.stack([pa,ent],1)
        if arm=='shuf': feat=feat[torch.randperm(bs)]
        if arm=='none': feat=torch.zeros_like(feat)
        x=torch.cat([pre,a[:,None],torch.full((bs,1),VPOS)],1)
        y=torch.tensor(f[q]).long()
        lab=(a!=y).long()                               # 0 RIGHT, 1 WRONG
        lg,vd=m(x,meta=feat,meta_pos=4)
        loss=F.cross_entropy(lg[:,2,:NA],y)+F.cross_entropy(vd[:,4],lab)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    m.eval(); correct=0; n=0; wrong_caught=0; wrong_total=0; ans_acc=0
    with torch.no_grad():
        for _ in range(8):
            q=rng.integers(0,NQ,size=500)
            pre=torch.tensor(np.stack([np.full(500,BOS),q+Q_OFF,np.full(500,SEP)],1)).long()
            lg,_=m(pre); p=F.softmax(lg[:,-1,:NA],-1)
            a=torch.multinomial(p,1).squeeze(1); pa=p.gather(1,a[:,None]).squeeze(1)
            ent=-(p*(p+1e-12).log()).sum(-1)/math.log(NA)
            feat=torch.stack([pa,ent],1)
            if arm=='shuf': feat=feat[torch.randperm(500)]
            if arm=='none': feat=torch.zeros_like(feat)
            x=torch.cat([pre,a[:,None],torch.full((500,1),VPOS)],1)
            _,vd=m(x,meta=feat,meta_pos=4)
            y=torch.tensor(f[q]).long(); lab=(a!=y).long(); pred=vd[:,4].argmax(-1)
            correct+=int((pred==lab).sum()); n+=500
            wrong_total+=int(lab.sum()); wrong_caught+=int(((pred==1)&(lab==1)).sum())
            ans_acc+=int((a==y).sum())
    return dict(arm=arm,seed=seed,verdict_acc=correct/n,
                catch_rate=wrong_caught/max(wrong_total,1),
                sampled_answer_acc=ans_acc/n)

if __name__=='__main__':
    rows=[]
    for seed in [0,1,2]:
        for arm in ['true','shuf','none']:
            t0=time.time(); r=run(arm,seed); r['sec']=round(time.time()-t0); rows.append(r)
            print(f"seed {seed} {arm:5s} verdict_acc {r['verdict_acc']:.4f}  "
                  f"wrong_caught {r['catch_rate']:.4f}  answer_acc {r['sampled_answer_acc']:.4f}  [{r['sec']}s]",flush=True)
            json.dump(rows,open('/home/claude/meta/res.json','w'),indent=1)
    print('DONE',flush=True)
