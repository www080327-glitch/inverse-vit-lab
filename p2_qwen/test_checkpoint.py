"""Storage failure controls; no model load or accelerator work."""
import errno
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import checkpoint as c


class CheckpointStorageTests(unittest.TestCase):
    def test_foreach_adamw_retains_algorithm_and_loaded_state(self):
        torch.manual_seed(104)
        a=[torch.nn.Parameter(torch.randn(37,19)),torch.nn.Parameter(torch.randn(13))]
        b=[torch.nn.Parameter(v.detach().clone()) for v in a]
        original=torch.optim.AdamW(a,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
        optimized=torch.optim.AdamW(b,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
        for step in range(20):
            if step==5:
                optimized.load_state_dict(original.state_dict())
                # load_state_dict can share CPU tensors in this synthetic control.
                import copy
                optimized.load_state_dict(copy.deepcopy(original.state_dict()))
                for group in optimized.param_groups:group.update(foreach=True,fused=False)
            for x,y in zip(a,b):
                g=torch.randn_like(x);x.grad=g;y.grad=g.clone()
            original.step();optimized.step()
        for x,y in zip(a,b):self.assertTrue(torch.allclose(x,y,atol=2e-7,rtol=2e-6))

    def test_retry_and_verified_copy(self):
        original=c.os.fsync;calls=[]
        def fail_once(fd):
            calls.append(fd)
            if len(calls)==2:raise OSError(errno.EIO,'injected transient NFS error')
            return original(fd)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weights.pt'
            with patch.object(c.os,'fsync',side_effect=fail_once),patch.object(c.time,'sleep'):
                c.atomic_save({'tensor':torch.arange(17)},p)
            self.assertTrue(torch.equal(torch.load(p,weights_only=True)['tensor'],torch.arange(17)))
            receipt=json.loads(p.with_name(p.name+'.integrity.json').read_text())
            self.assertEqual(receipt['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertGreaterEqual(len(calls),3)

    def test_persistent_sync_failure_preserves_published_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weights.pt';p.write_bytes(b'previous durable checkpoint')
            original=c.os.fsync;count=0
            def fail_nfs(fd):
                nonlocal count
                count+=1
                if count>1:raise OSError(errno.EIO,'injected persistent error')
                return original(fd)
            with patch.object(c.os,'fsync',side_effect=fail_nfs) as sync,patch.object(c.time,'sleep'):
                with self.assertRaises(OSError):c.atomic_save({'tensor':torch.ones(3)},p)
                self.assertEqual(sync.call_count,3)
            self.assertEqual(p.read_bytes(),b'previous durable checkpoint')

    def test_state_failure_keeps_prior_immutable_weights(self):
        class Metadata:
            def save_pretrained(self,path):Path(path).mkdir(exist_ok=True)
        class Model:
            tokenizer=Metadata()
            class Qwen:config=Metadata()
            qwen=Qwen()
            value=1.
            def state_dict(self):
                return {'qwen.model.language_model.embed_tokens.weight':torch.tensor([self.value]),'qwen.lm_head.weight':torch.tensor([self.value])}
        class Optimizer:
            def state_dict(self):return {}
        with tempfile.TemporaryDirectory() as directory:
            m=Model();c.save(m,Optimizer(),directory,'last',{'step':1})
            previous=torch.load(Path(directory)/'last_state.pt',weights_only=True)
            original=c.atomic_save;m.value=2.
            def fail_state(obj,path):
                if Path(path).name=='last_state.pt':raise OSError(errno.EIO,'injected state failure')
                return original(obj,path)
            with patch.object(c,'atomic_save',side_effect=fail_state):
                with self.assertRaises(OSError):c.save(m,Optimizer(),directory,'last',{'step':2})
            self.assertEqual(torch.load(Path(directory)/'last_state.pt',weights_only=True)['step'],1)
            old=torch.load(Path(directory)/previous['weights_file'],weights_only=True)
            self.assertEqual(old['step'],1)
            self.assertEqual(float(old['model']['qwen.lm_head.weight']),1.)

    def test_reused_labels_preserve_referenced_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for step in (1,2):
                weight=f'best_step_{step:06d}_weights.pt';torch.save({'step':step},out/weight)
                pending=out/'new_state.tmp';torch.save({'step':step,'weights_file':weight},pending)
                pending.replace(out/'best_state.pt')
                if step==2:
                    c.cleanup_generations(out)
                    self.assertTrue((out/'best_step_000001_weights.pt').exists())
                c.reuse_snapshot(out,'best','last',step)
                self.assertTrue((out/'last_state.pt').samefile(out/'best_state.pt'))
                self.assertEqual(torch.load(out/'last_state.pt',weights_only=True)['step'],step)
            self.assertFalse((out/'best_step_000001_weights.pt').exists())


if __name__=='__main__':unittest.main()
