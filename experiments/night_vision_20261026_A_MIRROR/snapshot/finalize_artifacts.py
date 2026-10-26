"""CPU-only atomic publication after the sole training process releases its lock."""
import os,sys,pathlib,json,time,fcntl,hashlib,tarfile,io,datetime,traceback,zipfile,shutil
P=pathlib.Path(__file__).resolve().parent;os.chdir(P)
assert sys.platform=='linux'
os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
def dump(n,d):
 p=P/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=True,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
lock=open(P/'RUN.lock','a+')
ledger=json.loads((P/'night_budget.json').read_text());expires=ledger['provider_expires_unix']
while time.time()<expires-150:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:time.sleep(20);continue
 if (P/'ATOMIC_SNAPSHOT_READY.json').exists():
  result=json.loads((P/'results/summary.json').read_text())
  if result.get('status')=='COMPLETE' or (P/'EXPORT_PARTIAL').exists() or time.time()>ledger['main_deadline_unix']+30:break
 fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(20)
else:raise TimeoutError('No closed snapshot before provider export deadline')
try:
 import numpy as np
 from safetensors.numpy import save_file
 E=P/'feature_evidence';E.mkdir(exist_ok=True)
 ids=json.loads((P/'indices.json').read_text());fc=json.loads((P/'feature_contract.json').read_text());pre_ids=ids['pre_control_positions']
 rows=list(range(8))+list(range(1000,1008));global_rows=np.asarray(pre_ids)[rows]
 pre=np.load(P/'pre_subset.npy',mmap_mode='r');post=np.load(P/'post_cache.npy',mmap_mode='r');target=np.load(P/'target_RGB.npy',mmap_mode='r');orig=np.load(P/'original_RGB.npy',mmap_mode='r')
 save_file({'pre':np.ascontiguousarray(pre[rows]),'post':np.ascontiguousarray(post[global_rows]),'target_RGB_u8':np.ascontiguousarray(target[global_rows]),'original_RGB_u8':np.ascontiguousarray(orig[global_rows])},str(E/'paired_pre_post_16.safetensors'))
 samples=[]
 ntr=len(ids['train'])
 for row,g in zip(rows,global_rows):
  g=int(g);split='train' if g<ntr else 'dev';index=ids['train'][g] if split=='train' else ids['dev'][g-ntr]
  samples.append(dict(tensor_row=len(samples),source_original_train_index=index,split=split,allowed_for_training=split=='train',pre_subset_row=row,main_cache_row=g))
 (E/'samples.json').write_text(json.dumps(dict(schema='actual-hook-paired-interface-v1',samples=samples,pre_shape=[16,256,768],post_shape=[16,64,1024],features_dtype='float32',target_dtype='uint8 RGB /255',pre_hook_actual=True,source='existing night extraction cache; no recomputation',new_encoder_forwards=0,final_samples=0),indent=2)+'\n')
 retained=[]
 if result.get('pre_post')!='TRAINED_MATCHED_SUBSET_WITH_ADAPTER_PARAMETER_DIFFERENCE':
  D=P/'diagnostic_cache';D.mkdir(exist_ok=True)
  # Preserve the whole matched 1000/200 interface if no comparison was completed.
  np.save(D/'post_subset.npy',np.asarray(post[np.asarray(pre_ids)]))
  np.save(D/'target_subset_RGB_u8.npy',np.asarray(target[np.asarray(pre_ids)]))
  np.save(D/'original_subset_RGB_u8.npy',np.asarray(orig[np.asarray(pre_ids)]))
  retained.append((P/'pre_subset.npy','diagnostic_cache/pre_subset.npy'))
  (D/'README.txt').write_text('First 1000 rows are train; last 200 are dev. No final samples. Raw features, before train-only normalization. Pre activations were actually hooked.\n')
 metrics=json.loads((P/'results/metrics.json').read_text()) if (P/'results/metrics.json').exists() else {}
 lockcfg=json.loads((P/'training_locked.json').read_text()) if (P/'training_locked.json').exists() else {}
 def trained(n):return isinstance(result.get(n),dict) and result[n].get('status')=='TRAINED'
 def val(n,s):return metrics.get(n,{}).get(s,{}).get('correct',{}).get('MSE')
 full='已完成正式训练' if trained('mirror_full') else '未完成正式训练（不得把短测称为正式完成）'
 comparison='尚无完整公平结果'
 if val('mirror_full','final') is not None and val('mirror_tiny','final') is not None:
  comparison='final MSE：full %.6f，tiny %.6f；full 相对 tiny 变化 %.2f%%（负数更低）'%(val('mirror_full','final'),val('mirror_tiny','final'),100*(val('mirror_full','final')/val('mirror_tiny','final')-1))
 general='只有固定本轮开发/最终评估证据，不代表任意图像泛化。'
 if val('mirror_full','final') is not None:
  baseline=metrics['mirror_full']['final']['training_mean_image']['MSE']
  general=('full 在本轮 final 优于固定训练均值图。' if val('mirror_full','final')<baseline else 'full 在本轮 final 未优于固定训练均值图。')+'不声称历史上从未使用这些图，也不外推到任意图像。'
 pretext='真实 pre/post 已保留，对照训练未完成，不伪造对照结果。'
 if isinstance(result.get('pre_shallow'),dict) and isinstance(result.get('post_shallow'),dict) and val('pre_shallow','dev') is not None and val('post_shallow','dev') is not None:
  pretext='同一 1000/200 子集 dev MSE：pre %.6f，post %.6f；共享浅层主干，但 post 适配器多 3,148,800 参数，不是绝对完美控制变量，不能单凭差值证明信息不可恢复。'%(val('pre_shallow','dev'),val('post_shallow','dev'))
 language=result.get('language_interface',{'status':'NOT_RUN','reason':'Prerequisite stages not complete'})
 answers=[('full 是否真正跑了',full),('比 tiny 怎样',comparison),('能否泛化',general),('pre/post 说明什么',pretext),('语言接口是否运行',str(language.get('status','NOT_RUN'))+'：'+language.get('reason',''))]
 head='\n'.join(f'{i+1}. {k}：{v}' for i,(k,v) in enumerate(answers))
 detail=['# 夜间 Qwen 视觉镜像重建实验','',head,'','## 主实验像素指标','', '| 方法 | split | MSE | MAE | 均值图 MSE | 下采样32对原图 MSE | 梯度 MAE | 边界梯度 MAE |','|---|---|---:|---:|---:|---:|---:|---:|']
 table=''
 for name in ['mirror_tiny','mirror_full']:
  for split in ['train','dev','final']:
   row=metrics.get(name,{}).get(split,{})
   if 'correct' not in row:continue
   c=row['correct'];mean=row['training_mean_image']['MSE']
   detail.append('| %s | %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |'%(name,split,c['MSE'],c['MAE'],mean,c['down32_vs_original_MSE'],c['gradient_MAE'],c['boundary_gradient_MAE']))
   table+='<tr><th>%s</th><td>%s</td><td>%.6f</td><td>%.6f</td><td>%.6f</td></tr>'%(name,split,c['MSE'],c['MAE'],mean)
 detail+=['','目标是官方预处理后的 256×256 RGB [0,1]，来自32×32源图插值；没有获得源图不存在的细节。历史32×32线性/MLP结果不参与本轮公平排名。','', '训练/开发/最终评估图数：%d / %d / %d。共同正式步数 %s，有效batch32。MSE单一主损失，AdamW lr3e-4、wd1e-4、5%%warmup、cosine、clip1。'% (len(ids['train']),len(ids['dev']),len(ids['final']),lockcfg.get('common_steps','NOT_LOCKED')), '', 'tiny 为2层/256宽/4头/FFN1024；full 为12层/768宽/12头/FFN3072。两者随机初始化，使用相同训练顺序、目标和训练集特征统计，位置为固定二维sin/cos，不声称是对官方RoPE的求逆。', '', '真实forward_pre_hook捕获merger之前张量；官方视觉塔及merger始终eval、冻结，不加载P2失败权重或运行语言主干。', '', '每个模型使用dev选择best；全部配置与共同步数在正式训练前锁定，final在主模型训练和选择完成后才打开。正确/同类错配/平均特征的全部MSE、MAE和辅助诊断见 results/metrics.json。', '', '固定图册覆盖全部预定 train16、dev32、final32，不挑最好图。PNG只作[0,1]裁剪显示；对应未裁剪原生精度预测保存为 *_unclipped.safetensors。', '', '梯度和patch边界误差仅为辅助诊断，不等同于感知质量或语义理解。pre/post适配器参数不同，差值不等于merger信息损失的精确因果测量。', '', '用户预算上限8小时；实际资源为90分钟免费课堂，因此训练预算还受到实例寿命约束。旧账本未修改；有限修复保留原时间起点、样本、步数与检查点。远端自行计时/退出，本机未使用MPS。', '', 'best/last/config/normalization/optimizer-RNG状态按不可变目录原子保存；每个READY记录文件哈希。paired feature_evidence中8训练+8开发，无final。完整源模型和全量特征/数据缓存不对外重复分发。', '', '模型与数据来源：Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17；Transformers5.2.0；CIFAR10官方训练池，保留原45000/5000划分，官方test从未打开。', '', '## 固定图册','']
 if len(ids['train'])==2000:detail.insert(4,'本轮在正式训练前按最初含冷启动的吞吐探针保守回退到2000张训练图。此探针没有单独排除首次初始化开销，因此不能据此证明充分预热后10000张不可行；没有根据重建分数改选数据。')
 galleries=sorted((P/'results').glob('gallery_*.png'))
 for im in galleries:detail+=['![%s](results/%s)'%(im.stem,im.name),'']
 (P/'REPORT_ZH.md').write_text('\n'.join(detail)+'\n')
 import html
 li=''.join('<li><b>'+html.escape(k)+'：</b>'+html.escape(v)+'</li>' for k,v in answers)
 pictures=''.join('<figure><a href="results/'+im.name+'"><img src="results/'+im.name+'" alt="'+im.stem+'"></a><figcaption>'+im.stem+' · 点击查看原尺寸全部固定样本</figcaption></figure>' for im in galleries)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夜间 Qwen 镜像实验</title><style>body{font:16px/1.65 system-ui;margin:28px auto;max-width:1160px;padding:0 20px;color:#1b2e35}h1{font-size:28px}ol{background:#edf4f1;padding:18px 24px 18px 44px;border-radius:10px}li{margin:8px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption,small{color:#52656d}</style><h1>夜间 Qwen 视觉镜像重建实验</h1><ol>'''+li+'''</ol><p>统一目标：官方Processor的256×256 RGB [0,1]，由32×32源图插值而来，不是新增真实细节。旧32×32小头不参与公平排名。</p><table><thead><tr><th>模型</th><th>split</th><th>MSE</th><th>MAE</th><th>均值图MSE</th></tr></thead><tbody>'''+table+'''</tbody></table><p>详细训练/资源/控制变量边界见 <a href="REPORT_ZH.md">文字报告</a>；全部正确/错配/均值特征及边界诊断见 <a href="results/metrics.json">指标JSON</a>；<a href="indices.json">固定索引</a>与<a href="training_locked.json">共同训练锁定</a>。</p><p>图册全量展示预定 train16、dev32、final32；显示PNG裁剪到[0,1]，原生精度未裁剪预测另存。左右位置与样本ID不作为解码器输入。</p>'''+pictures+'</html>'
 (P/'REPORT_ZH.html').write_text(page)
 # Only closed, selected artifacts are packed. No source weights/data/full cache.
 selected=[]
 for f in P.iterdir():
  if f.is_file() and f.suffix in ('.py','.json','.html','.md','.log','.safetensors') and f.name not in ['official_header.json','vision.safetensors','DELIVERY_FILES.json']:selected.append((f,f.name))
 for dirname in ['reference_core','results','feature_evidence','diagnostic_cache']:
  root=P/dirname
  if root.exists():
   selected += [(f,str(f.relative_to(P))) for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
 for root in (P/'snapshots').iterdir():
  if not root.is_dir():continue
  for which in ['best','last']:
   d=root/which
   if not d.exists():continue
   for f in d.iterdir():
    if not f.is_file() or (which=='best' and f.name in ['restore.pt','READY.json']):continue
    selected.append((f,'checkpoints/'+root.name+'/'+which+'/'+f.name))
   if which=='best':
    meta=json.loads((d/'READY.json').read_text());meta['files'].pop('restore.pt',None);meta['status']='COMPLETE_BEST_MODEL_ONLY';meta['optimizer_and_RNG_restore_in']='../last/restore.pt'
    out=P/('best_export_'+root.name+'.json');out.write_text(json.dumps(meta,indent=2)+'\n');selected.append((out,'checkpoints/'+root.name+'/best/READY.json'))
 selected+=retained
 manifest={arc:dict(bytes=f.stat().st_size,sha256=sha(f)) for f,arc in selected}
 dump('DELIVERY_FILES.json',dict(status='COMPLETE_ATOMIC_SELECTION',files=manifest));selected.append((P/'DELIVERY_FILES.json','DELIVERY_FILES.json'))
 OUT=P/'export';OUT.mkdir(exist_ok=True)
 assets={};small=[];mapping={}
 for f,arc in selected:
  binary=arc.startswith('checkpoints/') and f.suffix in ('.pt','.safetensors')
  binary=binary or f.stat().st_size>64*1024*1024
  if binary:
   digest=sha(f);name='artifact-'+digest+f.suffix;dest=OUT/name
   if not dest.exists():
    try:os.link(f,dest)
    except OSError:shutil.copyfile(f,dest)
   assets.setdefault(name,dict(name=name,bytes=dest.stat().st_size,sha256=digest,encoding='raw',paths=[]))['paths'].append(arc);mapping[arc]=name
  else:small.append((f,arc))
 reportzip=OUT/'NIGHT_VISION_20261026_A_MIRROR_reports.zip';tmp=reportzip.with_suffix('.tmp.zip')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for f,arc in small:z.write(f,arc)
 tmp.replace(reportzip)
 assets[reportzip.name]=dict(name=reportzip.name,bytes=reportzip.stat().st_size,sha256=sha(reportzip),encoding='zip',paths=[arc for f,arc in small])
 transport=dict(status='COMPLETE_ATOMIC_TRANSPORT',format='content-addressed raw checkpoint assets plus one reports ZIP; restore each raw asset to its listed relative paths',assets=list(assets.values()),path_to_asset=mapping,experiment_status=result.get('status'),created_unix=time.time())
 dump('export/TRANSPORT_READY.json',transport)
 print(json.dumps(transport),flush=True)
except BaseException as e:
 dump('archive_error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
