"""Identical fixed text prompts, templates, greedy decoding and validation windows."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from .model import PixelQwen
from .checkpoint import load_weights
from .data import text_tokens,text_batch
from .train import amp
from .guard import atomic_json,check_compute_deadline as check_deadline


@torch.inference_mode()
def evaluate(m,c):
    m.eval();start=time.time();outputs=[]
    for prompt in c['prompts']:
        ids=m.tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
            add_generation_prompt=True,enable_thinking=c['enable_thinking'],return_tensors='pt').to(m.device)
        if not isinstance(ids,torch.Tensor):ids=ids['input_ids']
        generated=[]
        for _ in range(c['max_new_tokens']):
            check_deadline()
            with amp(m.device.type):
                h=m.language(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
                token=int(m.qwen.lm_head(h).argmax(-1).item())
            if token in (m.tokenizer.eos_token_id,248044):break
            generated.append(token);ids=torch.cat((ids,torch.tensor([[token]],device=m.device)),1)
        outputs.append(dict(prompt=prompt,output=m.tokenizer.decode(generated,skip_special_tokens=True),tokens=generated))
    val=text_tokens(m.tokenizer)['validation'];rng=torch.Generator().manual_seed(c['text_val_seed']);losses=[]
    for _ in range(c['text_val_batches']):
        check_deadline()
        with amp(m.device.type):losses.append(float(m.text_loss(text_batch(val,rng,length=c['text_val_length']).to(m.device))))
    return dict(config=c,outputs=outputs,text_val_ce=sum(losses)/len(losses),seconds=time.time()-start,
                scope='Fixed small evaluation; does not establish general language or vision preservation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(6)
    if not os.environ.get('P2_GUARDED'):raise RuntimeError('External cumulative budget guard required')
    m=PixelQwen()
    if a.checkpoint:load_weights(m,a.checkpoint)
    m.to(a.device);c=json.loads(Path('configs/p2_qwen/text_eval.json').read_text())
    atomic_json(a.out,evaluate(m,c))


if __name__=='__main__':main()
