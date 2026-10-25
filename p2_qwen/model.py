"""Inference-safe Qwen + pixel interfaces; no data, targets or retrieval imports."""
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,Qwen3_5Config
from .probability import unpack,draw

CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
PROMPTS=[f'Draw a photo of a {name}.' for name in CLASSES]


class PixelQwen(nn.Module):
    def __init__(self, source='data/p2_qwen/official_model', pretrained=True, tokenizer_source=None):
        super().__init__()
        self.source=str(source)
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_source or source,local_files_only=True,trust_remote_code=False)
        if pretrained:
            self.qwen,info=Qwen3_5ForConditionalGeneration.from_pretrained(
                source,local_files_only=True,trust_remote_code=False,dtype=torch.float32,
                attn_implementation='sdpa',output_loading_info=True)
        else:
            config=Qwen3_5Config.from_pretrained(source,local_files_only=True)
            config._attn_implementation='sdpa'
            self.qwen=Qwen3_5ForConditionalGeneration(config).float();info={}
        if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys'):
            raise ValueError(f'Official weight loading mismatch: {info}')
        self.loading_info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()}
        self.qwen.model.visual.requires_grad_(False)
        self.language.requires_grad_(True)
        self.qwen.lm_head.requires_grad_(True)
        d=self.language.config.hidden_size
        self.pixel_in=nn.Linear(48,d);self.pixel_head=nn.Linear(d,8*(1+96))
        self.image_bos=nn.Parameter(torch.empty(1,1,d))
        std=float(self.language.embed_tokens.weight.detach().float().std())
        nn.init.normal_(self.image_bos,std=std)
        nn.init.normal_(self.pixel_in.weight,std=std/(48**.5));nn.init.zeros_(self.pixel_in.bias)
        nn.init.normal_(self.pixel_head.weight,std=.001);nn.init.zeros_(self.pixel_head.bias)
        with torch.no_grad():self.pixel_head.bias[8*(1+48):].fill_(-1.2039728043259361)
        self.init_embedding_std=std
        if self.qwen.lm_head.weight is not self.language.embed_tokens.weight:
            raise ValueError('Expected official tied text embedding/output weights')

    @property
    def language(self):return self.qwen.model.language_model

    @property
    def device(self):return self.image_bos.device

    def prompts(self, class_ids):
        # Right padding occurs only after each image sequence, never inside its causal prefix.
        return [self.tokenizer.encode(PROMPTS[int(c)],add_special_tokens=False) for c in class_ids]

    def image_inputs(self, class_ids, prefix):
        if prefix.ndim!=3 or prefix.shape[2]!=48 or prefix.shape[1]>63:
            raise ValueError('Input contains completed prefix only (0..63 patches)')
        if not torch.isfinite(prefix).all() or (prefix.abs()>1).any():raise ValueError('Unbounded feedback')
        ids=self.prompts(class_ids);lengths=[len(v)+1+prefix.shape[1] for v in ids]
        rows=[]; masks=[]; starts=[]
        for i,tokens in enumerate(ids):
            text=self.language.embed_tokens(torch.tensor(tokens,device=self.device))
            image=torch.cat((self.image_bos[0],self.pixel_in(prefix[i])),0)
            row=torch.cat((text,image),0);pad=max(lengths)-len(row)
            rows.append(F.pad(row,(0,0,0,pad)))
            masks.append([1]*len(row)+[0]*pad);starts.append(len(text))
        emb=torch.stack(rows);mask=torch.tensor(masks,device=self.device)
        pos=torch.arange(emb.shape[1],device=self.device)[None].expand(len(ids),-1)
        return emb,mask,pos,starts

    def forward(self, class_ids, prefix):
        emb,mask,pos,starts=self.image_inputs(class_ids,prefix)
        out=self.language(inputs_embeds=emb,attention_mask=mask,position_ids=pos,use_cache=False)
        hidden=torch.stack([out.last_hidden_state[i,s:s+1+prefix.shape[1]] for i,s in enumerate(starts)])
        return unpack(self.pixel_head(hidden))

    def teacher(self,class_ids,targets):
        if targets.shape[1:]!=(64,48):raise ValueError('Expected 64 RGB patches')
        return self(class_ids,targets[:,:-1])

    def text_loss(self,ids,chunk=16):
        hidden=self.language(input_ids=ids[:,:-1],use_cache=False).last_hidden_state
        h=hidden.reshape(-1,hidden.shape[-1]);y=ids[:,1:].reshape(-1)
        def ce(a,b):return F.cross_entropy(F.linear(a,self.qwen.lm_head.weight).float(),b,reduction='sum')
        loss=hidden.new_zeros((),dtype=torch.float32)
        for i in range(0,len(y),chunk):
            a,b=h[i:i+chunk],y[i:i+chunk]
            loss=loss+(checkpoint(ce,a,b,use_reentrant=False) if self.training else ce(a,b))
        return loss/len(y)

    def optimizer(self,lr=1e-5,interface_lr=1e-4):
        new=list(self.pixel_in.parameters())+list(self.pixel_head.parameters())+[self.image_bos]
        original=[p for p in self.qwen.parameters() if p.requires_grad]
        return torch.optim.AdamW([dict(params=original,lr=lr,peak_lr=lr),dict(params=new,lr=interface_lr,peak_lr=interface_lr)],
                                 betas=(.9,.95),weight_decay=.01,foreach=False)

    def parameter_manifest(self):
        rows=[dict(name=n,count=p.numel(),trainable=p.requires_grad,dtype=str(p.dtype)) for n,p in self.named_parameters()]
        return dict(total=sum(r['count'] for r in rows),trainable=sum(r['count'] for r in rows if r['trainable']),
                    unused_visual=sum(p.numel() for p in self.qwen.model.visual.parameters()),
                    tied_text_head=True,layer_types=self.language.config.layer_types,parameters=rows)


@torch.inference_mode()
def generate(model,classes,seeds,on_step=None):
    model.eval();rng=[torch.Generator().manual_seed(int(s)) for s in seeds]
    prefix=torch.empty(len(seeds),0,48,device=model.device);zs=[];ks=[]
    for j in range(64):
        if os.environ.get('P2_DEADLINE_UNIX'):
            import time
            if time.time()>float(os.environ['P2_DEADLINE_UNIX']):raise TimeoutError('Sampling deadline')
        x,z,k=draw(tuple(v[:,-1] for v in model(classes,prefix)),rng)
        prefix=torch.cat((prefix,x[:,None].to(model.device)),1);zs.append(z);ks.append(k)
        if on_step:on_step(j+1,prefix.detach().cpu())
    return prefix.cpu(),torch.stack(zs,1),torch.stack(ks,1)


@torch.inference_mode()
def cached_start(model,class_id):
    empty=torch.empty(1,0,48,device=model.device)
    emb,mask,pos,_=model.image_inputs([class_id],empty)
    out=model.language(inputs_embeds=emb,attention_mask=mask,position_ids=pos,use_cache=True)
    return unpack(model.pixel_head(out.last_hidden_state[:,-1:])),out.past_key_values,emb.shape[1]


@torch.inference_mode()
def cached_next(model,patch,cache,position):
    emb=model.pixel_in(patch.to(model.device))[:,None]
    pos=torch.tensor([[position]],device=model.device)
    out=model.language(inputs_embeds=emb,position_ids=pos,past_key_values=cache,use_cache=True)
    return unpack(model.pixel_head(out.last_hidden_state)),out.past_key_values,position+1


@torch.inference_mode()
def generate_cached_one(model,class_id,seed,on_step=None):
    model.eval();rng=[torch.Generator().manual_seed(int(seed))]
    params,cache,position=cached_start(model,class_id);xs=[];zs=[];ks=[]
    for j in range(64):
        import time
        if time.time()>float(os.environ.get('P2_DEADLINE_UNIX',0)):raise TimeoutError('Sampling deadline')
        x,z,k=draw(tuple(v[:,0] for v in params),rng);xs.append(x);zs.append(z);ks.append(k)
        if on_step:on_step(j+1,torch.stack(xs,1))
        if j<63:params,cache,position=cached_next(model,x,cache,position)
    return torch.stack(xs,1),torch.stack(zs,1),torch.stack(ks,1)
