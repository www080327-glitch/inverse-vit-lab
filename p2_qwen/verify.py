"""Actual official Qwen tests; never substitute a tiny synthetic backbone."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from safetensors import safe_open
from .guard import atomic_json
from .model import PixelQwen
from .probability import dequantize,nll


def sync(device):
    if device=='mps':torch.mps.synchronize()
    if device=='cuda':torch.cuda.synchronize()


def gradient_audit(model):
    rows=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        g=p.grad
        rows.append(dict(name=name,numel=p.numel(),has_gradient=g is not None,
                         finite=bool(torch.isfinite(g).all()) if g is not None else False,
                         absmax=float(g.abs().max()) if g is not None else None))
    if any(not r['has_gradient'] or not r['finite'] for r in rows):raise AssertionError('Missing/nonfinite language/interface gradient')
    return rows


def representatives(model):
    names={f'qwen.model.language_model.layers.{i}.mlp.down_proj.weight' for i in range(24)}
    names.update(['qwen.model.language_model.embed_tokens.weight','qwen.model.language_model.norm.weight',
                  'pixel_in.weight','pixel_head.weight','image_bos'])
    # Include recurrent conv, decay/gates, attention, normalization and FFN, all parameter tensors.
    return {n:p.detach().flatten()[:4096].cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps','cuda'],required=True)
    p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(6)
    if not os.environ.get('P2_GUARDED'):raise RuntimeError('External cumulative budget guard required')
    start=time.time();result=dict(device=a.device,status='RUNNING',cache=False,tolerance=dict(atol=3e-4,rtol=3e-4))
    atomic_json(a.out,result)
    try:
        torch.manual_seed(20261023);m=PixelQwen()
        result['parameters']=m.parameter_manifest();result['loading_info']=m.loading_info
        with safe_open('data/p2_qwen/official_model/model.safetensors-00001-of-00001.safetensors',framework='pt') as f:
            official_checks={}
            for n,pv in m.qwen.named_parameters():
                # All loaded official parameter tensors, FP32 conversion is exact from BF16.
                if n not in f.keys():raise AssertionError('Official parameter key missing '+n)
                official_checks[n]=bool(torch.equal(pv.detach(),f.get_tensor(n).float()))
            assert all(official_checks.values());result['official_exact_tensors']=len(official_checks)
        m.to(a.device).eval()
        u=torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
        target,_=dequantize(u,torch.Generator().manual_seed(5));target=target.to(a.device)
        emb,mask,pos,starts=m.image_inputs([0,3],target[:,:17])
        result['input_interface']=dict(embedding_std=m.init_embedding_std,pixel_input_rms=float(m.pixel_in(target).detach().square().mean().sqrt()),
          image_bos_rms=float(m.image_bos.detach().square().mean().sqrt()),prompt_token_ids=m.prompts([0,3]),
          mask=mask.cpu().tolist(),position_ids=pos.cpu().tolist(),image_bos_positions=starts)
        causal=[]
        with torch.no_grad():
            original=m.teacher([0,3],target)
            for j in (0,17,62):
                changed=target.clone();changed[:,j:]=-changed[:,j:]
                other=m.teacher([0,3],changed)
                errors=[float((x[:,:j+1]-y[:,:j+1]).abs().max()) for x,y in zip(original,other)]
                for x,y in zip(original,other):torch.testing.assert_close(x[:,:j+1],y[:,:j+1],atol=3e-4,rtol=3e-4)
                prefix=m([0,3],target[:,:j])
                perrors=[float((x[:,j]-y[:,-1]).abs().max()) for x,y in zip(original,prefix)]
                for x,y in zip(original,prefix):torch.testing.assert_close(x[:,j],y[:,-1],atol=3e-4,rtol=3e-4)
                causal.append(dict(j=j,future_perturb_max=errors,full_vs_prefix_max=perrors))
            # Batch/padding and independent state: compare same row alone, then after unrelated call.
            solo=m([3],target[1:,:17]);m([9],-target[:1,:7]);again=m([3],target[1:,:17])
            batched=m([0,3],target[:,:17])
            for x,y,z in zip(solo,again,batched):
                torch.testing.assert_close(x,y,atol=3e-4,rtol=3e-4)
                torch.testing.assert_close(x[0],z[1],atol=3e-4,rtol=3e-4)
            # The selected class prompts have equal lengths: test a genuinely
            # masked right-padding region separately, including adversarial values.
            emb,mask,pos,_=m.image_inputs([0,3],target[:,:17]);mask[0,-3:]=0
            valid=m.language(inputs_embeds=emb,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
            changed=emb.clone();changed[0,-3:]=100.
            perturbed=m.language(inputs_embeds=changed,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
            unpadded=m.language(inputs_embeds=emb[:1,:-3],attention_mask=mask[:1,:-3],position_ids=pos[:1,:-3],use_cache=False).last_hidden_state
            torch.testing.assert_close(valid[0,:-3],perturbed[0,:-3],atol=3e-4,rtol=3e-4)
            torch.testing.assert_close(valid[0,:-3],unpadded[0],atol=3e-4,rtol=3e-4)
            result['real_masked_padding']='PASS; adversarial padding and unpadded prefix agree'
        result['causal']=causal;result['padding_and_state_reset']='PASS'
        atomic_json(a.out,result)
        # Combined image and genuine text CE gives every participating parameter its corresponding task.
        m.train();opt=m.optimizer();before=representatives(m)
        opt.zero_grad(set_to_none=True);loss=nll(m.teacher([0],target[:1]),target[:1]);loss.backward()
        ids=m.tokenizer('The small ship crossed the blue sea. A cat sat on the mat.',return_tensors='pt')['input_ids'].to(a.device)
        textloss=m.text_loss(ids);textloss.backward()
        result['gradients']=gradient_audit(m)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step();sync(a.device)
        updates=[]
        for name,pv in m.named_parameters():
            if name not in before:continue
            b=before[name];after=pv.detach().flatten()[:len(b)].cpu()
            lr=1e-4 if not name.startswith('qwen.') else 1e-5
            updates.append(dict(name=name,absmax=float((after-b).abs().max()),
                                loss_update_absmax=float((after-b*(1-lr*.01)).abs().max())))
        assert all(r['absmax']>0 and r['loss_update_absmax']>0 for r in updates)
        assert all(p.grad is None for p in m.qwen.model.visual.parameters())
        result.update(updates=updates,image_nll=float(loss),text_ce=float(textloss),status='PASS',
                      seconds=time.time()-start,vision_gradient='NONE; explicitly excluded')
    except BaseException as exc:
        result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),seconds=time.time()-start)
        atomic_json(a.out,result);raise
    atomic_json(a.out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('parameters','gradients','updates')}),flush=True)


if __name__=='__main__':main()
