"""Build deterministic handoff arrays, figures, and a fully offline result viewer."""
import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from PIL import Image
from runtime import ROOT, np, torch, initialize, read_jsonl, write_json
from memory import PixelMemory
from demo import load_model

def large_image():
    image=np.asarray(Image.open(ROOT/'data/earth_1920x1080.png').convert('RGB'))
    t=time.perf_counter(); memory=PixelMemory.pack(image); pack_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); restored=memory.unpack(); unpack_ms=(time.perf_counter()-t)*1000
    np.testing.assert_array_equal(restored,image)
    t=time.perf_counter(); summary=memory.summary(); summary_ms=(time.perf_counter()-t)*1000
    ids=torch.tensor([0,119,239,16199,16200,32160,32280,32399])
    tiles,valid,ids=memory.gather(ids)
    # This is a storage example. No high-resolution router or attention is executed.
    np.savez_compressed(ROOT/'data/earth_detail_blocks.npz',pixels=memory.pixels.numpy(),
                        boxes=memory.boxes.numpy(),valid=memory.valid.numpy(),
                        summary=summary.numpy(),shape=np.array(image.shape))
    n=1920*1080; r=n//64
    write_json(ROOT/'results/high_resolution.json',dict(source='NASA Blue Marble 2002 satellite composite',
        original_file='data/earth_original.jpg',crop_file='data/earth_1920x1080.png',
        image_shape=list(image.shape),interpolation=False,roundtrip_equal=True,
        packed_storage=memory.storage(),actual_summary_bytes=summary.numel()*4,
        actual_gathered_regions=ids.tolist(),actual_gathered_pixels=int(valid.sum()),
        actual_gathered_raw_bytes=tiles.numel(),timing_single_observation_ms=dict(pack=pack_ms,unpack=unpack_ms,summary=summary_ms),
        attention_executed=False,router_executed=False,
        theoretical_only=dict(fine_positions=n,regions=r,q=1,dk=5,dv=64,
          all_pixel_self_attention_score_elements=n*n,float32_self_attention_score_bytes=4*n*n,
          single_query_dense_score_elements=n,single_query_dense_attention_flops=2*n*(5+64),
          example_selected_positions=512,single_query_selected_attention_flops=2*512*(5+64),
          coarse_encoder_matrix_flops=2*r*(16*32+32*4)),
        warning='No N-by-N tensor created. Theoretical costs use the toy dimensions, not a real VLM. No high-resolution semantic quality or speed measured.'))

def samples(model):
    protocol=dict(version=1,question_dim=4,questions=['eye','arm','marker','connection'],
        answers=[['closed','open'],['down','up'],['left','right'],['aligned','offset']],
        region_order='row-major 4x4; each tile 8x8; pixel order within tile row-major',
        boxes='global [x0,y0,x1,y1), integer pixels, clipped',
        hierarchy=dict(root=dict(level=0,bounds=[0,0,32,32]),region=dict(level=1,scale_pixels=8,parent='root'),
                       fine=dict(level=2,scale_pixels=1,parent='region_id',coordinates='x=x0+local_x; y=y0+local_y')),
        storage_lossless='uint8 raw RGB pack/unpack; zero padding and validity mask',
        summary_lossy='RGB mean then 2x2 mean, float32 [16,16]; cannot restore raw pixels',
        attention='softmax(Q K_selected.T/sqrt(5)) V_selected; one normalization over all selected pixels',
        coarse_tokens_in_fine_attention=False,dedup='sort(unique(region IDs)) before gather and fine encoding',
        cache='no cross-query fine-KV cache; per-query temporary KV; raw details retained',
        masks='invalid padded pixel positions get -inf scores',
        production_relation='independent toy; not Qwen or CSA2 or a paper reproduction',
        inputs='data/inputs/*.jsonl contains paths and task IDs only',
        gold='data/annotations/*.jsonl is training/evaluation supervision, never inference inputs',
        contract='B can consume arrays independently. No B files modified, no B reply required.')
    write_json(ROOT/'exchange/protocol.json',protocol)
    gold=[]
    for split in ['test','test_blind']:
        row=read_jsonl(ROOT/f'data/inputs/{split}.jsonl')[0]
        ann=read_jsonl(ROOT/f'data/annotations/{split}.jsonl')[0]
        m=PixelMemory.pack(Image.open(ROOT/row['image']))
        with torch.inference_mode():
            pred=model(m,0,2); dense=model(m,0,16)
            xy=m.coordinates(pred['ids'])
        arrays=dict(image=m.unpack(),pixels=m.pixels.numpy(),boxes=m.boxes.numpy(),valid=m.valid.numpy(),
                    coarse_summary=m.summary().numpy(),coarse_keys=pred['coarse']['keys'].numpy(),
                    query_id=np.array(0),question_vector=np.eye(4,dtype=np.float32)[0],
                    fine_query=model.fine_query(0).numpy(),selected_ids=pred['ids'].numpy(),
                    selected_x=xy['x'].numpy(),selected_y=xy['y'].numpy(),
                    selected_K=pred['keys'].numpy(),selected_V=pred['values'].numpy(),
                    attention_weights=pred['weights'].numpy(),output=pred['output'].numpy(),dense_output=dense['output'].numpy())
        np.savez_compressed(ROOT/f'exchange/sample_{split}.npz',**arrays)
        gold.append(dict(sample=f'sample_{split}.npz',scene=row['id'],**ann))
    write_json(ROOT/'exchange/sample_gold_evaluation_only.json',gold)

def figures(model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(15,8));ax.set_xlim(0,15);ax.set_ylim(0,8);ax.axis('off')
    def box(x,y,w,h,title,body,color='#e8eff8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',fc=color,ec='#39516a',lw=1.3))
        ax.text(x+w/2,y+h*.72,title,ha='center',va='center',weight='bold',fontsize=12)
        ax.text(x+w/2,y+h*.32,body,ha='center',va='center',fontsize=10,linespacing=1.4)
    def arrow(a,b,text=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.6,color='#42566b'))
        if text: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,text,ha='center',fontsize=9)
    ax.text(.2,7.65,'Question-conditioned multiresolution visual memory',fontsize=20,weight='bold')
    ax.text(.2,7.25,'Separate global coarse routing from selected-pixel attention  |  CPU only  |  32 x 32 RGB',fontsize=12,color='#42566b')
    box(.3,4.9,3.2,1.4,'Lossless detail library','16 tiles x 8 x 8 x 3 uint8\nOriginal 1,024 pixels retained','#e0f2e7')
    box(4.2,4.9,3.1,1.4,'Lossy coarse index','16 summaries x 16 features\n2 x 2 means + tiny MLP')
    box(8,4.9,3,1.4,'Global question read','4-d task vector x 16 keys\nCoarse softmax + top-k = 2')
    box(11.9,4.9,2.6,1.4,'Unique region IDs','No ground-truth boxes\nor labels in inference','#fff1d5')
    box(.3,2.3,3.2,1.4,'Gather selected raw tiles','Only 2 regions encoded\n128 fine pixel positions','#e0f2e7')
    box(4.2,2.3,3.1,1.4,'Lazy fine K and V','K: 128 x 5\nV: 128 x 64 spatial basis')
    box(8,2.3,3,1.4,'One fine softmax','Q K^T / sqrt(5), then A V\nAcross all selected pixels')
    box(11.9,2.3,2.6,1.4,'Tiny binary head','64-d read + task ID\n7,451 total parameters')
    arrow((3.5,5.6),(4.1,5.6));arrow((7.3,5.6),(7.9,5.6));arrow((11,5.6),(11.8,5.6))
    arrow((1.9,4.8),(1.9,3.8),'raw pixels')
    arrow((13.2,4.8),(13.2,4.15));arrow((13.2,4.15),(1.9,4.15));arrow((1.9,4.15),(1.9,3.8))
    arrow((3.5,3),(4.1,3));arrow((7.3,3),(7.9,3));arrow((11,3),(11.8,3))
    box(.3,.4,7,.95,'Exactness boundary','All 16 regions + same Q/K/V -> dense reference (<=1,024 pixels)','#eef4f8')
    box(8,.4,6.5,.95,'Failure boundary','Identical coarse summaries -> missed tiny targets remain missed','#fbe4e4')
    ax.text(.3,1.7,'Coarse index stays in memory, but coarse tokens are excluded from the final fine softmax.',fontsize=11)
    fig.savefig(ROOT/'figures/architecture.png',dpi=140,bbox_inches='tight'); fig.savefig(ROOT/'figures/architecture.svg',bbox_inches='tight');plt.close(fig)
    metric=json.loads((ROOT/'results/metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    methods=['uniform','random','oracle','learned','brightest'];colors=['#92a6bb','#7d9bc0','#39a378','#4471ba','#c78955']
    for ax,split,title in zip(axes,['test','test_blind'],['Normal held-out layouts','Identical coarse summaries']):
        vals=[next(r for r in metric['results'] if r['split']==split and r['k']==2 and r['method']==m)['target_region_recall']*100 for m in methods]
        ax.bar(methods,vals,color=colors);ax.set_title(title);ax.set_ylim(0,112);ax.set_ylabel('Target region recall (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}',ha='center',fontsize=10)
    fig.suptitle('Fixed test: select 2 / 16 regions, 128 / 1,024 pixel positions');fig.tight_layout()
    fig.savefig(ROOT/'figures/recall.png',dpi=150,bbox_inches='tight');plt.close(fig)
    failures=json.loads((ROOT/'results/failure_cases.json').read_text())
    fig,axes=plt.subplots(1,4,figsize=(13,4.2))
    for ax,row in zip(axes,failures):
        ax.imshow(Image.open(ROOT/row['image']),interpolation='nearest')
        for rid in row['selected_regions']:ax.add_patch(Rectangle((rid%4*8-.5,rid//4*8-.5),8,8,fill=False,ec='#4db5ff',lw=2))
        rid=row['target_region'];ax.add_patch(Rectangle((rid%4*8-.5,rid//4*8-.5),8,8,fill=False,ec='#ff5656',lw=2))
        ax.set_title(f"{row['scene']} / q{row['question']}\nRecall {row['target_region_recall']}; answer correct {row['correct']}",fontsize=10)
        ax.axis('off')
    fig.suptitle('Coarse-blind failures: blue = selected, red = true target (evaluation overlay only)',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'figures/failures.png',dpi=160,bbox_inches='tight');plt.close(fig)

def offline_viewer(model):
    metrics=read_jsonl(ROOT/'results/per_question.jsonl'); scenes=[]
    for split in ['test','test_blind']:
        inputs=read_jsonl(ROOT/f'data/inputs/{split}.jsonl')[:4]
        anns=read_jsonl(ROOT/f'data/annotations/{split}.jsonl')[:4]
        for inp,ann in zip(inputs,anns):
            image=np.asarray(Image.open(ROOT/inp['image'])); m=PixelMemory.pack(image)
            rows=[r for r in metrics if r['scene']==inp['id']]
            with torch.inference_mode():
                for row in rows:
                    tiles,valid,_=m.gather(row['selected_regions'])
                    row['output']=model.read_tiles(tiles,row['question'],valid)['output'].tolist()
            summary=m.summary().reshape(4,4,4,4).permute(0,2,1,3).reshape(16,16)
            scenes.append(dict(id=inp['id'],rgb=image.tolist(),summary=summary.tolist(),rows=rows,gold=ann))
    template=(ROOT/'viewer_template.html').read_text()
    (ROOT/'demo.html').write_text(template.replace('__RESULTS_JSON__',json.dumps(scenes,separators=(',',':'))))

def main():
    initialize(); model=load_model()
    with torch.inference_mode(): large_image();samples(model);figures(model);offline_viewer(model)
    print('Built real-image block archive, two handoff NPZ samples, figures and offline demo.html.')

if __name__=='__main__': main()
