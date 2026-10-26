"""Fixed-set evaluation. Oracle and test answers live ONLY in this evaluator."""
import json
import time
from collections import defaultdict
from PIL import Image
from runtime import ROOT, np, torch, initialize, read_jsonl, write_json
from memory import PixelMemory
from model import attention
from demo import load_model

def latency(fn, repeats=101):
    for _ in range(5): fn()
    samples=[]
    for _ in range(repeats):
        t=time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-t)/1000)
    return dict(unit='microseconds',median=float(np.median(samples)),p90=float(np.percentile(samples,90)),
                minimum=float(min(samples)),warmups=5,repeats=repeats)

def cost(model,m):
    summary=m.summary(); ids,coarse=model.select(summary,0,2)
    tiles,valid,ids=m.gather(ids); kv=model.fine_kv(tiles); k=kv[0].reshape(-1,5); v=kv[1].reshape(-1,64)
    q=model.fine_query(0); out,_=attention(q,k,v)
    tensors={**m.storage()}
    for name,t in [('coarse_summary',summary),('coarse_keys',coarse['keys']),('selected_raw_pixels',tiles),
                   ('selected_K',k),('selected_V',v),('query',q),('attention_output',out)]:
        tensors[name]=dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),
                           storage_bytes=t.untyped_storage().nbytes())
    source_image=m.unpack()
    timing={
      'pack_raw_with_metadata':latency(lambda:PixelMemory.pack(source_image)),
      'summary_rgb_and_2x2_mean':latency(m.summary),
      'coarse_MLP_scores_context':latency(lambda:model.coarse_read(summary,0)),
      'coarse_read_and_topk':latency(lambda:model.select(summary,0,2)),
      'dedup_and_raw_gather':latency(lambda:m.gather(ids)),
      'selected_fine_encoder_and_KV':latency(lambda:model.fine_kv(tiles)),
      'selected_attention_QK_softmax_AV':latency(lambda:attention(q,k,v)),
      'dense_attention_QK_softmax_AV':None,
      'hierarchical_whole_query_no_pack':latency(lambda:model(m,0,2)),
      'dense_whole_query_no_pack':latency(lambda:model(m,0,16)),
      'direct_dense_reader_no_router_no_pack':latency(lambda:model.read_tiles(m.pixels,0,m.valid)),
    }
    dk,dv=model.fine_kv(m.pixels); dk=dk.reshape(-1,5);dv=dv.reshape(-1,64)
    timing['dense_attention_QK_softmax_AV']=latency(lambda:attention(q,dk,dv))
    return dict(actual_tensor_allocations=tensors,model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
       model_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
       counts=dict(regions=16,persistent_raw_positions=1024,selected_regions=2,selected_positions=128,
                   q_positions=1,key_dim=5,value_dim=64,coarse_input_dim=16,coarse_key_dim=4),
       matrix_multiply_FLOPs=dict(convention='multiply and add count as 2; excludes bias/activation/softmax/norm/pool/sort/gather',
          coarse_encoder=2*16*(16*32+32*4),question_region_scores=2*16*4,
          coarse_context_weighted_sum=2*16*16,
          selected_fine_type_encoder=2*2*(64*48+48*5),dense_fine_type_encoder=2*16*(64*48+48*5),
          selected_attention=2*128*(5+64),dense_attention=2*1024*(5+64),
          answer_head=2*(68*48+48*2),
          attention_ratio_only=8),
       timing=timing,timing_scope='CPU eager, inference_mode, 2 intraop/1 interop; one image, one task, no PNG I/O',
       caveats=['direct dense baseline skips coarse routing; all-region hierarchy includes it',
                'stage medians cannot be added into an end-to-end measurement',
                'one-hot spatial V is a pedagogical buffer and its expanded materialization is charged',
                'no persistent learned fine KV cache; original pixels retained and selected KV generated per query',
                'coarse summaries and keys are regenerated per query in this minimal implementation',
                'CPU times do not imply GPU/DCU/MPS performance or full-model acceleration'])

def evaluate():
    env=initialize(); model=load_model(); groups=defaultdict(list); allrows=[]; failure_candidates=[]
    with torch.inference_mode():
        for split in ['test','test_blind']:
            inputs=read_jsonl(ROOT/f'data/inputs/{split}.jsonl'); anns=read_jsonl(ROOT/f'data/annotations/{split}.jsonl')
            for scene_idx,(inp,ann) in enumerate(zip(inputs,anns)):
                assert inp['id']==ann['id']
                image=np.asarray(Image.open(ROOT/inp['image']).convert('RGB'))
                m=PixelMemory.pack(image); np.testing.assert_array_equal(m.unpack(),image)
                summary=m.summary()
                for question in inp['questions']:
                    gt=ann['regions'][question]; answer=ann['answers'][question]
                    dense=model(m,question,16)
                    for count in [1,2,4]:
                        for method in ['uniform','random','oracle','learned','brightest']:
                            seed=10000+scene_idx
                            if method in ['uniform','random','learned']:
                                pred=model(m,question,count,method,seed)
                                ids=pred['ids']
                            else:
                                if method=='oracle':
                                    rest=[r for r in range(16) if r!=gt]
                                    ids=torch.tensor([gt]+rest[:count-1])
                                else: ids=summary.mean(-1).argsort(descending=True,stable=True)[:count]
                                selected,valid,ids=m.gather(ids)
                                pred=model.read_tiles(selected,question,valid)
                            selected_mass=float(dense['weights'].reshape(16,64)[ids].sum())
                            delta=pred['output']-dense['output']
                            row=dict(scene=inp['id'],image=inp['image'],split=split,question=question,k=count,method=method,
                                target_region=gt,selected_regions=ids.tolist(),target_region_recall=int(gt in ids.tolist()),
                                target_foreground_fraction=ann['target_pixels'][question]/1024,
                                target_region_area_fraction=64/1024,selected_positions=count*64,
                                target_answer=answer,predicted_answer=int(pred['logits'].argmax()),
                                correct=int(int(pred['logits'].argmax())==answer),dense_correct=int(int(dense['logits'].argmax())==answer),
                                output_relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(dense['output']).clamp_min(1e-12)),
                                output_max_abs=float(delta.abs().max()),retained_dense_attention_mass=selected_mass)
                            groups[(split,count,method)].append(row); allrows.append(row)
                            if method=='learned' and count==2 and (not row['target_region_recall'] or not row['correct']):
                                failure_candidates.append(row)
        results=[]
        for (split,count,method),rows in groups.items():
            hit=[r for r in rows if r['target_region_recall']]; miss=[r for r in rows if not r['target_region_recall']]
            results.append(dict(split=split,k=count,method=method,questions=len(rows),
                target_region_recall=float(np.mean([r['target_region_recall'] for r in rows])),
                answer_accuracy=float(np.mean([r['correct'] for r in rows])),
                dense_answer_accuracy=float(np.mean([r['dense_correct'] for r in rows])),
                answer_accuracy_on_hit=float(np.mean([r['correct'] for r in hit])) if hit else None,
                answer_accuracy_on_miss=float(np.mean([r['correct'] for r in miss])) if miss else None,
                hits=len(hit),misses=len(miss),
                mean_relative_l2=float(np.mean([r['output_relative_l2'] for r in rows])),
                p95_relative_l2=float(np.percentile([r['output_relative_l2'] for r in rows],95)),
                mean_retained_dense_mass=float(np.mean([r['retained_dense_attention_mass'] for r in rows])),
                mean_target_foreground_fraction=float(np.mean([r['target_foreground_fraction'] for r in rows])),
                min_target_foreground_fraction=min(r['target_foreground_fraction'] for r in rows),
                max_target_foreground_fraction=max(r['target_foreground_fraction'] for r in rows)))
        # Mechanism diagnostic: same image, query changes, selected regions can change.
        m=PixelMemory.pack(Image.open(ROOT/'data/images/test/test_0000.png'))
        question_switch=[model(m,q,2)['ids'].tolist() for q in range(4)]
        write_json(ROOT/'results/costs.json',cost(model,m))
    (ROOT/'results/per_question.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
    failures=sorted(failure_candidates,key=lambda r:(r['split'],r['scene'],r['question']))
    picked=[]
    for split in ['test','test_blind']:
        seen=set()
        for correct in [0,1]:
            count=0
            for row in [r for r in failures if r['split']==split and r['correct']==correct]:
                if row['scene'] not in seen:
                    picked.append(row); seen.add(row['scene']);count+=1
                if count==2:break
    write_json(ROOT/'results/failure_cases.json',picked)
    write_json(ROOT/'results/metrics.json',dict(environment=env,results=results,
              test_scenes=192,test_questions=768,question_switch_same_image=question_switch,
              selection_seed='10000 + scene index; same random/uniform subsets for all questions',
              oracle='evaluation-only access to target region; cannot be requested through demo/model inference',
              primary_metric='target region recall; binary accuracy alone can hide chance correctness on misses',
              data='equal-color glyphs; normal disjoint layout/state combinations; exact summary-blind adversarial split',
              test_set_used_for_training=False))
    for row in results:
        if row['k']==2: print(json.dumps(row))

if __name__=='__main__': evaluate()
