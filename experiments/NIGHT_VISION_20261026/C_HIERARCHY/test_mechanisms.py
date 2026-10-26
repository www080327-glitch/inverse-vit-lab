import unittest
import inspect
from unittest.mock import patch
from PIL import Image
from runtime import ROOT, np, torch, read_jsonl, write_json, initialize
from memory import PixelMemory
from model import Hierarchy, attention, streaming_attention
from demo import load_model

class Mechanisms(unittest.TestCase):
    def test_lossless_all_bytes_and_padding(self):
        rng=np.random.default_rng(87)
        for h,w,p in [(16,16,4),(32,32,8),(19,27,8),(1,1,8),(1080,1920,8)]:
            x=rng.integers(0,256,(h,w,3),dtype=np.uint8)
            m=PixelMemory.pack(x,p)
            np.testing.assert_array_equal(m.unpack(),x)
            self.assertEqual(int(m.valid.sum()),h*w)
            coords=m.coordinates([0,len(m.pixels)-1])
            self.assertEqual(coords['scale_pixels'],1)
            self.assertTrue((coords['x'][coords['valid']]<w).all())
            self.assertTrue((coords['y'][coords['valid']]<h).all())

    def test_summary_blindness_exact(self):
        from generate_data import glyph
        tiles=[glyph(k,s,True) for k in range(6) for s in range(2)]
        pooled=[torch.tensor(a).float().reshape(4,2,4,2).mean((1,3)) for a in tiles]
        self.assertTrue(all(torch.equal(pooled[0],p) for p in pooled))
        self.assertFalse(np.array_equal(tiles[0],tiles[2]))

    def test_dense_permutation_streaming_and_sparse(self):
        torch.random.default_generator.manual_seed(4)
        record=[]
        for n in [256,1024]:
            q=torch.randn(3,7,dtype=torch.float64); k=torch.randn(n,7,dtype=torch.float64); v=torch.randn(n,9,dtype=torch.float64)
            dense,_=attention(q,k,v)
            ref=torch.nn.functional.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None])[0,0]
            perm=torch.randperm(n); all_selected,_=attention(q,k[perm],v[perm])
            streamed=streaming_attention(q,[(k[i:i+37],v[i:i+37]) for i in range(0,n,37)])
            ids=perm[:n//8]; sparse,_=attention(q,k[ids],v[ids])
            mask=torch.zeros(n,dtype=torch.bool); mask[ids]=True
            masked,_=attention(q,k,v,mask)
            errors=dict(n=n,torch_dense_max_abs=float((dense-ref).abs().max()),
                        all_regions_max_abs=float((dense-all_selected).abs().max()),
                        streaming_max_abs=float((dense-streamed).abs().max()),
                        sparse_masked_max_abs=float((sparse-masked).abs().max()),
                        partial_vs_full_relative_l2=float(torch.linalg.vector_norm(sparse-dense)/torch.linalg.vector_norm(dense)))
            for key in ['torch_dense_max_abs','all_regions_max_abs','streaming_max_abs','sparse_masked_max_abs']:
                self.assertLess(errors[key],1e-12)
            record.append(errors)
        write_json(ROOT/'results/numerical_generic.json',record)

    def test_local_softmax_sum_is_wrong(self):
        q=torch.tensor([[1.]],dtype=torch.float64)
        k=torch.tensor([[0.],[8.]],dtype=torch.float64); v=torch.tensor([[1.],[3.]],dtype=torch.float64)
        dense,_=attention(q,k,v)
        wrong=sum(attention(q,k[i:i+1],v[i:i+1])[0] for i in range(2))
        self.assertGreater(float((dense-wrong).abs().max()),.5)
        self.assertTrue(torch.allclose(dense,streaming_attention(q,[(k[:1],v[:1]),(k[1:],v[1:])])) )

    def test_lazy_encoder_unique_regions_and_no_gold(self):
        model=load_model(); m=PixelMemory.pack(Image.open(ROOT/'data/images/test/test_0000.png'))
        calls=[]
        handle=model.fine_type.register_forward_pre_hook(lambda mod,args:calls.append(args[0].shape[0]))
        with torch.inference_mode(): out=model(m,0,2)
        handle.remove(); self.assertEqual(calls,[2]); self.assertEqual(out['selected_fine_positions'],128)
        a,va,ia=m.gather([3,1,3]); b,vb,ib=m.gather([1,3])
        self.assertTrue(torch.equal(a,b)); self.assertEqual(ia.tolist(),[1,3])
        for method in ['learned','uniform','random']:
            with torch.inference_mode(),patch('pathlib.Path.read_text',side_effect=AssertionError('No metadata reads allowed')):
                self.assertEqual(len(model(m,1,2,method)['ids']),2)
        with self.assertRaises(ValueError): model.select(m.summary(),0,2,'oracle')

    def test_actual_qkv_dense_row_major(self):
        model=load_model(); image=Image.open(ROOT/'data/images/test/test_0001.png')
        records=[]
        with torch.inference_mode():
            for side in [16,32]:
                m=PixelMemory.pack(image.crop((0,0,side,side)))
                out=model(m,2,len(m.pixels))
                keys,values=model.fine_kv(m.pixels)
                # Independent global row-major ordering vs gathered region-major ordering.
                def reorder(x): return x.reshape(side//8,side//8,8,8,-1).permute(0,2,1,3,4).reshape(side*side,-1)
                dense=torch.nn.functional.scaled_dot_product_attention(model.fine_query(2)[None,None],
                        reorder(keys)[None,None],reorder(values)[None,None])[0,0,0]
                err=float((out['output']-dense).abs().max())
                self.assertLess(err,1e-6)
                records.append(dict(n=side*side,max_abs=err,definition='same fine Q/K/V, independent row-major dense PyTorch SDPA',dtype='float32'))
            write_json(ROOT/'results/numerical_model.json',records)

    def test_split_contract(self):
        for split,parity in [('train',0),('test',1)]:
            rows=read_jsonl(ROOT/f'data/inputs/{split}.jsonl')
            anns=read_jsonl(ROOT/f'data/annotations/{split}.jsonl')
            for row,ann in zip(rows,anns):
                self.assertEqual(set(row),{'id','image','questions'})
                self.assertEqual(sum(ann['answers'])%2,parity)
                self.assertTrue(all((r//4+r%4)%2==parity for r in ann['regions']))
        import model, demo
        for module in [model,demo]:
            source=inspect.getsource(module)
            self.assertNotIn('data/annotations',source); self.assertNotIn('generate_data',source)

    def test_large_image_attention_rejected(self):
        m=PixelMemory.pack(np.zeros((33,32,3),dtype=np.uint8))
        with self.assertRaises(ValueError): Hierarchy()(m,0)

    def test_cpu_seed_and_all_parameters_stay_cpu(self):
        with (patch('torch.manual_seed',side_effect=AssertionError('Global seed API forbidden')),
             patch('torch.cuda.manual_seed_all',side_effect=AssertionError('CUDA helper forbidden')),
             patch('torch.mps.manual_seed',side_effect=AssertionError('MPS helper forbidden'))):
            initialize(28)
            model=load_model()
            m=PixelMemory.pack(Image.open(ROOT/'data/images/test/test_0000.png'))
            with torch.inference_mode():out=model(m,0,2)
        self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()))
        self.assertTrue(all(b.device.type=='cpu' for b in model.buffers()))
        self.assertEqual(out['output'].device.type,'cpu')

    def test_unread_details_preserved_and_causally_unread(self):
        model=load_model()
        image=np.asarray(Image.open(ROOT/'data/images/test_blind/test_blind_0000.png')).copy()
        ann=read_jsonl(ROOT/'data/annotations/test_blind.jsonl')[0]
        old=PixelMemory.pack(image)
        with torch.inference_mode():
            q=next(q for q in range(4) if ann['regions'][q] not in model(old,q,2)['ids'].tolist())
            before=model(old,q,2); dense_before=model(old,q,16)
            rid=ann['regions'][q]; y,x=rid//4*8,rid%4*8
            # Flip answer pixel within a single pooling cell. Coarse means unchanged.
            if q<2: image[y+4:y+6,x+4:x+6]=image[y+4:y+6,x+4:x+6][::-1].copy()
            else: image[y+4:y+6,x+4:x+6]=image[y+4:y+6,x+4:x+6][:,::-1].copy()
            new=PixelMemory.pack(image);after=model(new,q,2);dense_after=model(new,q,16)
        np.testing.assert_array_equal(new.unpack(),image)
        self.assertTrue(torch.equal(old.summary(),new.summary()))
        self.assertTrue(torch.equal(before['ids'],after['ids']))
        self.assertTrue(torch.equal(before['output'],after['output']))
        change=float((dense_before['output']-dense_after['output']).abs().max())
        self.assertGreater(change,1e-3)
        write_json(ROOT/'results/unread_detail_intervention.json',dict(question=q,region=rid,
            summary_identical=True,selected_ids_identical=True,sparse_output_bitwise_identical=True,
            dense_output_max_abs_change=change,raw_change_preserved=True))

if __name__=='__main__': unittest.main()
