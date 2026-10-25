"""Read-only P1 split reuse and independently split licensed text replay."""
import json
from pathlib import Path
import numpy as np
import torch
from .prepare import sha


def images(split,root='data/p1/cifar10'):
    if split not in ('train','val'):raise ValueError('Test is sealed; use explicit final evaluator')
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    for kind in ('images','labels'):
        p=root/f'{split}_{kind}.npy'
        if sha(p)!=manifest['files'][p.name]:raise ValueError('P1 data changed')
    return (torch.from_numpy(np.load(root/f'{split}_images.npy')),
            torch.from_numpy(np.load(root/f'{split}_labels.npy')))


def text_tokens(tokenizer,root='data/p2_qwen/text'):
    import pyarrow.parquet as pq
    out={}
    for split in ('train','validation'):
        p=Path(root)/f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
        rows=pq.read_table(p,columns=['text']).column('text').to_pylist()
        # Independent official document splits; never use evaluation prompts for replay.
        out[split]=torch.tensor(tokenizer.encode('\n'.join(rows),add_special_tokens=False,verbose=False),dtype=torch.long)
    return out


def text_batch(tokens,rng,batch=1,length=128):
    starts=torch.randint(len(tokens)-length,(batch,),generator=rng)
    return torch.stack([tokens[s:s+length+1] for s in starts])
