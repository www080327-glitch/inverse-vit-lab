"""Independent coarse routing and lazy fine K/V; no annotations dependency."""
import math
from runtime import torch
from torch import nn
from torch.nn import functional as F

def mlp(a,b,c): return nn.Sequential(nn.Linear(a,b),nn.Tanh(),nn.Linear(b,c))

def attention(q,k,v,mask=None):
    scores = q @ k.transpose(-1,-2)/math.sqrt(k.shape[-1])
    if mask is not None: scores = scores.masked_fill(~mask,float('-inf'))
    weights = scores.softmax(-1)
    return weights@v, weights

def streaming_attention(q, blocks):
    """Exact block merge by maxima + partition sums, NOT sum(local softmax @ V)."""
    m = torch.full(q.shape[:-1]+(1,),float('-inf'),dtype=q.dtype)
    z = torch.zeros_like(m)
    acc = None
    for k,v in blocks:
        s = q@k.transpose(-1,-2)/math.sqrt(k.shape[-1])
        new_m = torch.maximum(m,s.max(-1,keepdim=True).values)
        old_scale = (m-new_m).exp(); ex = (s-new_m).exp()
        part = ex@v
        acc = part if acc is None else acc*old_scale+part
        z = z*old_scale+ex.sum(-1,keepdim=True); m = new_m
    if acc is None: raise ValueError('No blocks')
    return acc/z

class Hierarchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = mlp(16,32,4)
        self.fine_type = mlp(64,48,5) # type only, no answer labels
        self.answer = mlp(68,48,2)
        self.register_buffer('basis',torch.eye(64))

    @staticmethod
    def normalize(tiles):
        x = tiles.float().mean(-1).reshape(*tiles.shape[:-3],64)/255
        # Per tile contrast invariance. Brightness/size alone cannot route type.
        return (x-x.min(-1,keepdim=True).values)/(x.max(-1,keepdim=True).values-x.min(-1,keepdim=True).values).clamp_min(1e-6)

    def coarse_read(self, summaries, question):
        keys = self.coarse(summaries)
        q = F.one_hot(torch.tensor(question),4).float()
        scores = keys@q
        context = scores.softmax(0)@summaries
        return scores, context, keys

    def select(self, summaries, question, k, method='learned', seed=0):
        r = len(summaries)
        if not 1<=k<=r: raise ValueError('k out of range')
        scores,context,keys = self.coarse_read(summaries,question)
        if method=='learned': ids = torch.argsort(scores,descending=True,stable=True)[:k]
        elif method=='uniform': ids = torch.tensor([(int((i+.5)*r/k)+seed%r)%r for i in range(k)])
        elif method=='random': ids = torch.randperm(r,generator=torch.Generator().manual_seed(seed))[:k]
        else: raise ValueError('Inference only permits learned/uniform/random; oracle is evaluation-only')
        return ids,dict(scores=scores,context=context,keys=keys)

    def fine_kv(self, tiles):
        """Called AFTER raw gather. Only selected regions run the 64->48->5 MLP."""
        x = self.normalize(tiles)
        logp = self.fine_type(x).log_softmax(-1)[...,:4]
        shape = x.shape[:-1]+(64,4)
        type_keys = logp.unsqueeze(-2).expand(shape)
        keys = torch.cat((type_keys, x.unsqueeze(-1)),dim=-1)
        values = self.basis.expand(x.shape[:-1]+(64,64))
        return keys,values

    @staticmethod
    def fine_query(question):
        q = torch.zeros(5)
        q[question] = 3.0
        q[4] = 7.0
        return q.unsqueeze(0)*math.sqrt(5)

    def read_tiles(self, tiles, question, valid=None):
        keys,values = self.fine_kv(tiles)
        keys = keys.reshape(-1,5); values = values.reshape(-1,64)
        mask = valid.reshape(1,-1) if valid is not None else None
        output,weights = attention(self.fine_query(question),keys,values,mask)
        q = F.one_hot(torch.tensor(question),4).float()
        logits = self.answer(torch.cat((output[0],q)))
        return dict(output=output[0],weights=weights[0],logits=logits,keys=keys,values=values)

    def forward(self, memory, question, k=2, method='learned', seed=0):
        if memory.height*memory.width>1024:
            raise ValueError('Toy attention limited to <=1024 original pixel positions')
        summaries = memory.summary()
        ids,coarse = self.select(summaries,question,k,method,seed)
        tiles,valid,ids = memory.gather(ids)
        read = self.read_tiles(tiles,question,valid)
        return dict(**read,ids=ids,coarse=coarse,
                    selected_fine_positions=int(valid.sum()),fine_encoder_regions=len(ids))
