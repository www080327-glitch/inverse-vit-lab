"""One serial remote training process, fixed data/targets/budgets, atomic snapshots."""
import os,sys,pathlib,time,json,hashlib,signal,traceback,shutil,math,pickle,tarfile,io,gc,fcntl,datetime,subprocess
P=pathlib.Path(__file__).resolve().parent;os.chdir(P)
assert sys.platform=='linux' and os.environ.get('NIGHT_AUTHORIZED_FREE_DCU')=='1', 'Explicit owned-free-DCU authorization required'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',XDG_CACHE_HOME=str(P/'cache'),XDG_CONFIG_HOME=str(P/'config'),MIOPEN_USER_DB_PATH=str(P/'cache/miopen'),MIOPEN_CUSTOM_CACHE_DIR=str(P/'cache/kernels'))
for n in ['cache/miopen','cache/kernels','config','results','snapshots']:(P/n).mkdir(parents=True,exist_ok=True)
LOCK=open(P/'RUN.lock','a+');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(n,v):
 p=P/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=True,indent=2,default=str)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
START=time.time();ledger_path=P/'night_budget.json'
if ledger_path.exists():raise RuntimeError('Refuse a second run or resetting the night budget; resume explicitly from atomic state')
instance=next(x for x in P.parts if x.startswith('Instances_Course_'))
stamp=instance.removeprefix('Instances_Course_')[:14]
allocated=datetime.datetime.strptime(stamp,'%Y%m%d%H%M%S').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
EXPIRES=allocated+5400
END=min(START+8*3600,EXPIRES-300);MAIN_END=min(START+6*3600,END-420)
LEDGER=dict(schema='night-budget-v1',started_unix=START,user_cap_seconds=28800,main_priority_cap_seconds=21600,provider_expires_unix=EXPIRES,hard_save_deadline_unix=END,main_deadline_unix=MAIN_END,spent_seconds=0,old_ledgers_modified=False,resource='one free classroom DCU, provider TTL 90 minutes',events=[])
dump('night_budget.json',LEDGER)
STOP=False
def stop(*a):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
class Deadline(Exception):pass
def check(reserve=0,end=None):
 if STOP or time.time()>min(END,end or END)-reserve:raise Deadline('Bounded budget/provider expiry; save and stop owned computation')
def event(stage,**kw):
 d=dict(stage=stage,unix=time.time(),elapsed_seconds=time.time()-START,**kw);LEDGER['events'].append(d);LEDGER['spent_seconds']=time.time()-START;dump('night_budget.json',LEDGER);dump('live_status.json',d);print(json.dumps(d),flush=True)
RESULT={'mirror_full':'NOT_RUN','mirror_tiny':'NOT_RUN','pre_post':'NOT_RUN','language_interface':{'status':'NOT_RUN','reason':'Not checked until reconstruction is complete and resource/budget allow'}}
ACTIVE={};MODELS={};POLICY={'autocast':'bfloat16','sdpa':'default','micro_batch':16,'effective_batch':32}

def snapshot(name,model,opt,step,norm,cfg,score,best=False):
 from safetensors.torch import save_file
 root=P/'snapshots'/name;root.mkdir(exist_ok=True)
 final=root/f'step_{step:07d}';tmp=root/f'.step_{step:07d}_{os.getpid()}'
 if tmp.exists():shutil.rmtree(tmp)
 tmp.mkdir()
 save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp/'model.safetensors'))
 save_file({k:v.detach().cpu().contiguous() for k,v in norm.items()},str(tmp/'normalization.safetensors'))
 import torch
 torch.save({'optimizer':opt.state_dict(),'step':step,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'data_order_seed':20261027},tmp/'restore.pt')
 (tmp/'config.json').write_text(json.dumps(dict(model=cfg.to_dict(),policy=POLICY,grid_thw=[1,16,16],output='unclipped BCHW float, RGB [0,1] target'),indent=2)+'\n')
 meta=dict(step=step,dev_MSE=score,status='COMPLETE_ATOMIC_CHECKPOINT',files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in tmp.iterdir()})
 (tmp/'READY.json').write_text(json.dumps(meta,indent=2)+'\n');tmp.rename(final)
 def link(which):
  t=root/('.'+which);t.unlink(missing_ok=True);t.symlink_to(final.name,target_is_directory=True);t.replace(root/which)
 link('last')
 if best:link('best')
 keep={os.readlink(root/x) for x in ['last','best'] if (root/x).is_symlink()}
 for d in root.glob('step_*'):
  if d.name not in keep:shutil.rmtree(d)
 return final

def restore_best(model,name):
 from safetensors.torch import load_file
 model.load_state_dict(load_file(str(P/'snapshots'/name/'best/model.safetensors')),strict=True)

try:
 import numpy as np,torch,transformers
 from PIL import Image,ImageDraw
 from safetensors.torch import load_file,save_file
 from transformers import AutoImageProcessor
 from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
 from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
 from night_model import NightMirror,configuration,small_layout_checks
 from reference_core.layout import processor_pixels_to_rgb
 torch.set_num_threads(4);torch.manual_seed(20261027)
 assert torch.cuda.is_available() and torch.version.hip,'Require the allocated DCU; no MPS or CPU model fallback'
 DEV=torch.device('cuda:0')
 event('admitted',torch=torch.__version__,hip=torch.version.hip,device=torch.cuda.get_device_name(0),free_seconds=END-time.time())
 dump('environment.json',dict(python=sys.version,torch=torch.__version__,hip=torch.version.hip,transformers=transformers.__version__,device=torch.cuda.get_device_name(0),memory_bytes=torch.cuda.get_device_properties(0).total_memory))
 dump('layout_checks.json',small_layout_checks())
 with tarfile.open(P/'cifar-10-python.tar.gz') as tar:
  parts=[]
  for i in range(1,6):
   b=pickle.load(tar.extractfile('cifar-10-batches-py/data_batch_'+str(i)),encoding='bytes');parts.append((b[b'data'].reshape(-1,3,32,32),np.array(b[b'labels'],dtype=np.int64)))
 allimages=np.concatenate([a for a,y in parts]);labels=np.concatenate([y for a,y in parts]);del parts
 rng=np.random.default_rng(61023);tr=[];va=[]
 for c in range(10):
  ix=rng.permutation(np.flatnonzero(labels==c));va.extend(ix[:500]);tr.extend(ix[500:])
 tr=np.array(sorted(tr),dtype=np.int64);va=np.array(sorted(va),dtype=np.int64)
 for name,a in [('train.npy',tr),('val.npy',va)]:
  b=io.BytesIO();np.save(b,a);assert hashlib.sha256(b.getvalue()).hexdigest()==json.loads((P/'split_contract.json').read_text())[name]
 rng=np.random.default_rng(20261027)
 def balanced(pool,n):
  by={c:rng.choice(pool[labels[pool]==c],n,replace=False) for c in range(10)}
  return np.array([by[int(c)][i] for i in range(n) for c in rng.permutation(10)],dtype=np.int64)
 train=balanced(tr,1000);dev=balanced(va,50);final=balanced(np.setdiff1d(va,dev),50)
 plans={'preferred':dict(train=train.tolist(),dev=dev.tolist(),final=final.tolist())}
 old=json.loads((P/'previous_indices.json').read_text());oldtrain=np.array(old['train_original_indices']);olddev=np.array(old['validation_original_indices']);oldfinal=balanced(np.setdiff1d(va,olddev),50)
 plans['fallback']=dict(train=oldtrain.tolist(),dev=olddev.tolist(),final=oldfinal.tolist())
 dump('candidate_data_plans.json',plans)
 proc=AutoImageProcessor.from_pretrained(str(P/'model'),local_files_only=True,use_fast=False)
 cfg=Qwen3_5VisionConfig(**json.loads((P/'model/config.json').read_text())['vision_config']);cfg._attn_implementation='eager'
 vision=Qwen3_5VisionModel(cfg).eval().requires_grad_(False);vision.load_state_dict(load_file(str(P/'vision.safetensors')),strict=True);vision.to(DEV,dtype=torch.float32)
 hook_data=[]
 def pre_hook(module,args):hook_data.append(args[0].detach())
 hook=vision.merger.register_forward_pre_hook(pre_hook)
 def encode(indices):
  check(600);im=allimages[indices].transpose(0,2,3,1)
  inputs=proc(images=[Image.fromarray(a) for a in im],return_tensors='pt');assert inputs['image_grid_thw'].tolist()==[[1,16,16]]*len(indices)
  hook_data.clear()
  with torch.inference_mode():out=vision(inputs['pixel_values'].to(DEV,dtype=torch.float32),grid_thw=inputs['image_grid_thw'].to(DEV))
  assert len(hook_data)==1
  if not (P/'actual_forward_shapes.json').exists():dump('actual_forward_shapes.json',dict(actual_hook_tensor_shape=list(hook_data[0].shape),actual_pooler_tensor_shape=list(out.pooler_output.shape),actual_processor_pixels_shape=list(inputs['pixel_values'].shape),actual_grid=inputs['image_grid_thw'].tolist(),pre_dtype=str(hook_data[0].dtype),post_dtype=str(out.pooler_output.dtype),hook_module=type(vision.merger).__name__,vision_module=type(vision).__name__))
  pre=hook_data[0].reshape(len(indices),256,768).cpu();post=out.pooler_output.reshape(len(indices),64,1024).cpu()
  assert torch.isfinite(pre).all() and torch.isfinite(post).all()
  target=torch.stack([processor_pixels_to_rgb(inputs['pixel_values'][i*256:(i+1)*256],(1,16,16),mean=proc.image_mean,std=proc.image_std)[0] for i in range(len(indices))])
  pixels=(target*255).round().byte();assert (pixels.float()/255-target).abs().max()<1.3e-7
  return pre,post,pixels.permute(0,2,3,1).numpy(),im
 t=time.time();probe=encode(train[:16]);torch.cuda.synchronize();rate=16/(time.time()-t)
 data_key='preferred' if 11000/rate<min(900,(MAIN_END-time.time())*.25) else 'fallback'
 plan=plans[data_key];train,dev,final=(np.array(plan[k],dtype=np.int64) for k in ['train','dev','final'])
 ntr,nd,nf=len(train),len(dev),len(final);idx=np.concatenate([train,dev,final]);assert len(set(idx.tolist()))==len(idx)
 n=len(idx);pre_ids=np.concatenate([np.arange(1000),ntr+np.arange(200)]);pre_where={int(x):i for i,x in enumerate(pre_ids)}
 protocol=dict(schema='night-vision-formal-v1',locked_before_training=True,data_plan=data_key,train_count=ntr,dev_count=nd,final_count=nf,seed=20261027,official_test='never opened',final_scope='this-round final only; not claimed historically unseen',revision='2fc06364715b967f1860aea9cf38778875588b17',target='actual official processed 256x256 RGB [0,1]; interpolated from source32, no invented detail',target_storage='uint8 exact RGB with inverse-normalization roundtrip max error <1.3e-7',decoder_initialization='random, no transposed/copied encoder parameters',main_loss='MSE only',optimizer=dict(name='AdamW',lr=.0003,weight_decay=.0001,warmup_fraction=.05,schedule='cosine',clip_norm=1),same_effective_batch=32,same_data_order=True,historical_32_heads_fair_ranking=False)
 dump('protocol.json',protocol)
 dump('indices.json',dict(**plan,labels=labels[idx].tolist(),sample_ids=[f'{s}_{int(g):05d}' for s,ar in [('train',train),('dev',dev),('final',final)] for g in ar],fixed_display_positions=dict(train=list(range(16)),dev=list(range(32)),final=list(range(32))),pre_control_positions=pre_ids.tolist(),pixel_sha256=[hashlib.sha256(allimages[i].transpose(1,2,0).tobytes()).hexdigest() for i in idx]))
 event('data_locked',plan=data_key,ntrain=ntr,ndev=nd,nfinal=nf,probe_images_per_second=rate,projected_extract_seconds=n/rate)
 postmm=np.lib.format.open_memmap(P/'post_cache.npy',mode='w+',dtype=np.float32,shape=(n,64,1024))
 targetmm=np.lib.format.open_memmap(P/'target_RGB.npy',mode='w+',dtype=np.uint8,shape=(n,256,256,3))
 origmm=np.lib.format.open_memmap(P/'original_RGB.npy',mode='w+',dtype=np.uint8,shape=(n,32,32,3))
 premm=np.lib.format.open_memmap(P/'pre_subset.npy',mode='w+',dtype=np.float32,shape=(1200,256,768))
 for start in range(0,n,16):
  count=min(16,n-start)
  if start==0 and data_key=='preferred':pre,post,pixels,orig=probe
  else:pre,post,pixels,orig=encode(idx[start:start+count])
  postmm[start:start+count]=post.numpy();targetmm[start:start+count]=pixels;origmm[start:start+count]=orig
  for k in range(count):
   if start+k in pre_where:premm[pre_where[start+k]]=pre[k].numpy()
  if start%256==0:
   for mm in [postmm,targetmm,origmm,premm]:mm.flush()
   event('extracting',completed=start+count,total=n)
 for mm in [postmm,targetmm,origmm,premm]:mm.flush()
 dump('feature_contract.json',dict(status='COMPLETE',pre_hook='actual forward_pre_hook on Qwen3_5VisionModel.merger',actual_pre_shape_per_image=[256,768],actual_post_shape_per_image=[64,1024],pre_subset_count=1200,pre_training=1000,pre_dev=200,post_dtype='float32',pre_dtype='float32',processor=proc.to_dict(),grid=[1,16,16],order='merge-group raster then sub_y,sub_x',normalization='raw cached features, not yet standardized',encoder_eval=not vision.training,encoder_parameters_frozen=not any(p.requires_grad for p in vision.parameters()),encoder_and_merger_sha256=sha(P/'vision.safetensors')))
 hook.remove();del vision,probe,pre,post,hook_data;gc.collect();torch.cuda.empty_cache()
 FEATURES=torch.from_numpy(np.asarray(postmm)).to(DEV);TARGET=torch.from_numpy(np.asarray(targetmm)).permute(0,3,1,2).contiguous().to(DEV);ORIGINAL=torch.from_numpy(np.asarray(origmm)).permute(0,3,1,2).contiguous().to(DEV)
 MU=FEATURES[:ntr].mean((0,1),keepdim=True);SD=FEATURES[:ntr].std((0,1),keepdim=True).clamp_min(1e-5);FEATURES.sub_(MU).div_(SD)
 NORM={'mean':MU,'std':SD};save_file({k:v.cpu().contiguous() for k,v in NORM.items()},str(P/'normalization.safetensors'))
 MEAN=TARGET[:ntr].float().mean(0,keepdim=True)/255;MEAN_FEATURE=FEATURES[:ntr].mean(0,keepdim=True)
 LAB=labels[idx];check(480)
 def amp():return torch.autocast('cuda',dtype=torch.bfloat16,enabled=POLICY['autocast']=='bfloat16')
 def optimize(m,opt,ff,yy):
  opt.zero_grad(set_to_none=True);mb=POLICY['micro_batch'];total=0
  for j in range(0,len(ff),mb):
   with amp():pred=m(ff[j:j+mb]);loss=torch.nn.functional.mse_loss(pred.float(),yy[j:j+mb].float())*(len(ff[j:j+mb])/len(ff))
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss before/within locked training')
   loss.backward();total+=float(loss.detach())
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
  if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
  opt.step();return total
 bench={};compat=[]
 for attempt in range(3):
  try:
   for name in ['mirror_full','mirror_tiny']:
    torch.manual_seed(20261027);model=NightMirror(name).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    torch.cuda.reset_peak_memory_stats();times=[]
    for j in range(6):
     check(480);torch.cuda.synchronize();t=time.time();optimize(model,opt,FEATURES[:32],TARGET[:32].float()/255);torch.cuda.synchronize();times.append(time.time()-t)
    bench[name]=dict(step_seconds=float(np.median(times[2:])),all_step_seconds=times,parameters=sum(p.numel() for p in model.parameters()),peak_bytes=torch.cuda.max_memory_allocated(),policy=dict(POLICY));del model,opt;gc.collect();torch.cuda.empty_cache()
   break
  except (RuntimeError,FloatingPointError) as e:
   compat.append(dict(attempt=attempt,error=repr(e),elapsed=time.time()-START));dump('compatibility_attempts.json',compat)
   if attempt==2:raise
   if 'model' in globals():del model
   if 'opt' in globals():del opt
   gc.collect();torch.cuda.empty_cache()
   if isinstance(e,FloatingPointError):POLICY['autocast']='float32'
   elif 'out of memory' in str(e).lower():POLICY['micro_batch']=8
   else:
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);POLICY['sdpa']='math'
   if attempt==1:POLICY['autocast']='float32'
 dump('throughput_probe.json',dict(benchmarks=bench,discarded_pilot_steps_each=6,failed_attempts=compat))
 pair=bench['mirror_full']['step_seconds']+bench['mirror_tiny']['step_seconds']
 remaining=MAIN_END-time.time();common=min(20000,int((remaining-720)/(pair*1.35)));common=max(0,(common//100)*100)
 if common<200:raise Deadline('Too little provider lifetime for both formal decoders')
 LOCKED=dict(common_steps=common,effective_batch=32,sample_presentations_each=common*32,policy=dict(POLICY),locked_unix=time.time(),benchmarks=bench,main_deadline_unix=MAIN_END,dev_interval=max(100,common//8),dev_selection='lowest full dev MSE',final_opened=False)
 dump('training_locked.json',LOCKED);event('training_locked',**LOCKED)
 def permutation_batches(length,steps):
  gen=torch.Generator().manual_seed(20261027);order=torch.randperm(length,generator=gen);cursor=0
  for step in range(steps):
   if cursor+32>length:order=torch.randperm(length,generator=gen);cursor=0
   yield order[cursor:cursor+32].to(DEV);cursor+=32
 def lr_at(step,steps):
  warm=max(1,int(steps*.05))
  return .0003*(min(1,(step+1)/warm) if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm))))
 def pred_batch(model,ff):
  with torch.inference_mode(),amp():return model(ff).float()
 def dev_mse(model,ff,yy):
  model.eval();ss=0
  for st in range(0,len(ff),16):
   check(120);pred=pred_batch(model,ff[st:st+16]);ss+=float((pred-yy[st:st+16].float()/255).square().sum())
  model.train();return ss/(len(ff)*3*256*256)
 def fit(name,ff,yy,devf,devy,norm,steps,deadline):
  torch.manual_seed(20261027);m=NightMirror(name).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);best=float('inf');logs=[];t0=time.time();interval=max(100,steps//8)
  ACTIVE.update(name=name,model=m,optimizer=opt,step=0,norm=norm,config=configuration(name),best=best)
  torch.cuda.reset_peak_memory_stats()
  for step,order in enumerate(permutation_batches(len(ff),steps)):
   check(180,end=deadline)
   for g in opt.param_groups:g['lr']=lr_at(step,steps)
   loss=optimize(m,opt,ff[order],yy[order].float()/255);ACTIVE['step']=step+1
   if (step+1)%interval==0 or step+1==steps:
    score=dev_mse(m,devf,devy);is_best=score<best
    if is_best:best=score
    path=snapshot(name,m,opt,step+1,norm,configuration(name),score,best=is_best)
    row=dict(step=step+1,train_batch_MSE=loss,dev_MSE=score,best_dev_MSE=best,elapsed_seconds=time.time()-t0,lr=opt.param_groups[0]['lr']);logs.append(row);dump('results/'+name+'_training.json',logs);event('training',name=name,**row)
  summary=dict(status='TRAINED',steps=steps,sample_presentations=steps*32,wall_seconds=time.time()-t0,parameters=sum(p.numel() for p in m.parameters()),adapter_parameters=sum(p.numel() for p in m.adapter.parameters()),peak_DCU_bytes=torch.cuda.max_memory_allocated(),best_dev_MSE=best)
  RESULT[name]=summary;dump('results/summary.json',RESULT);ACTIVE.clear();del opt;restore_best(m,name);m.eval();return m
 def metrics_for(pred,y,orig):
  e=pred-y;gx=pred[:,:,:,1:]-pred[:,:,:,:-1];gy=pred[:,:,1:,:]-pred[:,:,:-1,:];tx=y[:,:,:,1:]-y[:,:,:,:-1];ty=y[:,:,1:,:]-y[:,:,:-1,:]
  mx=torch.arange(1,256,device=DEV)%16==0;my=mx
  b=((gx-tx)[:,:,:,mx].abs().mean()+(gy-ty)[:,:,my,:].abs().mean())/2
  interior=((gx-tx)[:,:,:,~mx].abs().mean()+(gy-ty)[:,:,~my,:].abs().mean())/2
  down=torch.nn.functional.interpolate(pred,size=(32,32),mode='area');down_y=torch.nn.functional.interpolate(y,size=(32,32),mode='area')
  return dict(MSE=float(e.square().mean()),MAE=float(e.abs().mean()),clipped_MSE=float((pred.clamp(0,1)-y).square().mean()),gradient_MAE=float(((gx-tx).abs().mean()+(gy-ty).abs().mean())/2),boundary_gradient_MAE=float(b),interior_gradient_MAE=float(interior),boundary_excess=float(b-interior),down32_vs_original_MSE=float((down-orig).square().mean()),down32_vs_original_MAE=float((down-orig).abs().mean()),down32_vs_target_MSE=float((down-down_y).square().mean()))
 FIXED={};EVALUATIONS={}
 def evaluate(name,m,ff,yy,orig,lab,split,mean_feature,mean_image,save_fixed=True):
  n=len(ff);perm=np.arange(n)
  for c in range(10):
   ix=np.flatnonzero(lab==c);perm[ix]=np.roll(ix,1)
  assert np.all(perm!=np.arange(n))
  totals={};per={};fixed={}
  for condition in ['correct','same_class_mismatch','mean_feature']:
   accum={};rows=[];small=[]
   for st in range(0,n,16):
    check(40);end=min(st+16,n);inputf=ff[st:end] if condition=='correct' else ff[perm[st:end]] if condition=='same_class_mismatch' else mean_feature.expand(end-st,-1,-1)
    pred=pred_batch(m,inputf);y=yy[st:end].float()/255;met=metrics_for(pred,y,orig[st:end].float()/255)
    for k,v in met.items():accum[k]=accum.get(k,0)+v*(end-st)
    rows.extend((pred-y).square().mean((1,2,3)).cpu().tolist())
    lim=16 if split=='train' else 32
    if st<lim and save_fixed:small.append(pred[:min(end-st,lim-st)].cpu().to(torch.bfloat16 if POLICY['autocast']=='bfloat16' else torch.float32))
   totals[condition]={k:v/n for k,v in accum.items()};per[condition]=rows
   if small:fixed[condition]=torch.cat(small)
  baseline={}
  for st in range(0,n,32):
   check(20);en=min(st+32,n);met=metrics_for(mean_image.expand(en-st,-1,-1),yy[st:en].float()/255,orig[st:en].float()/255)
   for k,v in met.items():baseline[k]=baseline.get(k,0)+v*(en-st)/n
  totals['training_mean_image']=baseline;totals['source_dependence']=dict(mean_same_class_MSE_increase=float(np.mean(np.array(per['same_class_mismatch'])-per['correct'])),fraction_correct_better=float(np.mean(np.array(per['correct'])<per['same_class_mismatch'])))
  EVALUATIONS.setdefault(name,{})[split]=totals;dump('results/metrics.json',EVALUATIONS);dump('results/'+name+'_'+split+'_per_image.json',per)
  if fixed:
   save_file({k:v.contiguous() for k,v in fixed.items()},str(P/'results'/f'{name}_{split}_unclipped.safetensors'));FIXED[(name,split)]=fixed
  return totals
 def independent_check(name,m,norm):
  ff=FEATURES[ntr:ntr+2]*SD+MU
  save_file({'features':ff.cpu().contiguous()},str(P/'results'/f'{name}_decode_input.safetensors'))
  reference=pred_batch(m,(ff-norm['mean'])/norm['std']).cpu()
  save_file({'prediction':reference.contiguous()},str(P/'results'/f'{name}_decode_reference.safetensors'))
  # Launch after deleting the parent model; child takes only features/config/weights.
  return dict(input_file=f'{name}_decode_input.safetensors',reference_file=f'{name}_decode_reference.safetensors')
 for name in ['mirror_tiny','mirror_full']:
  model=fit(name,FEATURES[:ntr],TARGET[:ntr],FEATURES[ntr:ntr+nd],TARGET[ntr:ntr+nd],NORM,common,MAIN_END)
  for split,lo,hi in [('train',0,ntr),('dev',ntr,ntr+nd)]:evaluate(name,model,FEATURES[lo:hi],TARGET[lo:hi],ORIGINAL[lo:hi],LAB[lo:hi],split,MEAN_FEATURE,MEAN)
  independent_check(name,model,NORM)
  del model;gc.collect();torch.cuda.empty_cache()
 # Configurations and checkpoint selection are now fixed. Only now unseal final.
 LOCKED['final_opened']=True;LOCKED['final_opened_unix']=time.time();dump('training_locked.json',LOCKED)
 for name in ['mirror_tiny','mirror_full']:
  model=NightMirror(name).to(DEV);restore_best(model,name);model.eval()
  evaluate(name,model,FEATURES[ntr+nd:],TARGET[ntr+nd:],ORIGINAL[ntr+nd:],LAB[ntr+nd:],'final',MEAN_FEATURE,MEAN)
  del model;gc.collect();torch.cuda.empty_cache()
  check(60)
  completed=subprocess.run([sys.executable,str(P/'decode_only.py'),'--root',str(P),'--name',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
  (P/'results'/f'{name}_independent_decode.log').write_text(completed.stdout)
  if completed.returncode:RESULT[name]['independent_reload']='FAIL'
  else:RESULT[name]['independent_reload']='PASS'
 def gallery(split,lo,count):
  cols=['source32','target256','tiny correct','full correct','tiny same class','full same class','tiny mean','full mean']
  for page in range(math.ceil(count/8)):
   k=min(8,count-page*8);im=Image.new('RGB',(8*256+90,k*278+30),'white');draw=ImageDraw.Draw(im)
   for c,label in enumerate(cols):draw.text((94+c*256,8),label,fill='black')
   for j in range(k):
    ix=page*8+j;row=lo+ix;draw.text((2,40+j*278),f'{split} {ix} / {int(idx[row])}',fill='black')
    arrays=[ORIGINAL[row].cpu().permute(1,2,0).numpy(),TARGET[row].cpu().permute(1,2,0).numpy()]
    for condition in ['correct','same_class_mismatch','mean_feature']:
     for name in ['mirror_tiny','mirror_full']:
      a=FIXED[(name,split)][condition][ix].float().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy();arrays.append(a)
    for c,a in enumerate(arrays):im.paste(Image.fromarray(a).resize((256,256),Image.Resampling.NEAREST),(90+c*256,30+j*278))
   im.save(P/'results'/f'gallery_{split}_{page:02d}.png')
 for split,lo,count in [('train',0,16),('dev',ntr,32),('final',ntr+nd,32)]:gallery(split,lo,count)
 event('main_complete',models={n:RESULT[n] for n in ['mirror_tiny','mirror_full']})
 # Optional merger diagnostic uses one fixed 1000/200 subset, never final.
 remaining=END-time.time()
 if remaining>480:
  pre=torch.from_numpy(np.asarray(premm)).to(DEV);post=torch.from_numpy(np.asarray(postmm)[pre_ids].copy()).to(DEV);py=TARGET[pre_ids];po=ORIGINAL[pre_ids];pl=LAB[pre_ids]
  pm=pre[:1000].mean((0,1),keepdim=True);ps=pre[:1000].std((0,1),keepdim=True).clamp_min(1e-5);qm=post[:1000].mean((0,1),keepdim=True);qs=post[:1000].std((0,1),keepdim=True).clamp_min(1e-5)
  pre.sub_(pm).div_(ps);post.sub_(qm).div_(qs)
  subset_mean=py[:1000].float().mean(0,keepdim=True)/255
  contrast_steps=min(2000,int((remaining-240)/(pair*.6+1e-6)));contrast_steps=max(100,(contrast_steps//100)*100)
  dump('pre_post_locked.json',dict(train_count=1000,dev_count=200,steps_each=contrast_steps,sample_presentations_each=contrast_steps*32,shared_body=dict(depth=2,width=768,heads=12,FFN=3072),input_adapter_difference='pre identity; post Linear(1024,3072), 3148800 additional parameters',not_perfect_control=True,final_used=False))
  for name,ff,mu,sd in [('pre_shallow',pre,pm,ps),('post_shallow',post,qm,qs)]:
   model=fit(name,ff[:1000],py[:1000],ff[1000:],py[1000:],{'mean':mu,'std':sd},contrast_steps,END-90)
   evaluate(name,model,ff[:1000],py[:1000],po[:1000],pl[:1000],'train',ff[:1000].mean(0,keepdim=True),subset_mean,save_fixed=False)
   evaluate(name,model,ff[1000:],py[1000:],po[1000:],pl[1000:],'dev',ff[:1000].mean(0,keepdim=True),subset_mean,save_fixed=False)
   del model;gc.collect();torch.cuda.empty_cache()
  RESULT['pre_post']='TRAINED_MATCHED_SUBSET_WITH_ADAPTER_PARAMETER_DIFFERENCE'
 else:RESULT['pre_post']='FEATURES_CAPTURED_TRAINING_NOT_RUN_PROVIDER_TIME'
 # The classroom lease is only 90 minutes. No language framework is ported.
 RESULT['language_interface']=dict(status='NOT_RUN',reason='Provider lease has less than 60 minutes remaining after main reconstruction; official language weights were not downloaded or instantiated',checked_once_after_main=True,remaining_provider_seconds=EXPIRES-time.time())
 RESULT['status']='COMPLETE';RESULT['wall_seconds']=time.time()-START
except BaseException as e:
 RESULT.update(status='STOPPED_OR_BLOCKED',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.time()-START)
 if ACTIVE:
  try:snapshot(ACTIVE['name'],ACTIVE['model'],ACTIVE['optimizer'],ACTIVE['step'],ACTIVE['norm'],ACTIVE['config'],None,best=False)
  except BaseException as se:RESULT['emergency_save_error']=repr(se)
 dump('minimum_reproducer_error.json',dict(error=repr(e),traceback=traceback.format_exc()))
finally:
 LEDGER['spent_seconds']=time.time()-START;LEDGER['stopped_unix']=time.time();dump('night_budget.json',LEDGER);dump('results/summary.json',RESULT)
 try:
  # Publish only an immutable manifest after all checkpoint files are closed.
  manifest={}
  for root in [P/'results',P/'snapshots']:
   for f in root.rglob('*'):
    if f.is_file() and not f.is_symlink():manifest[str(f.relative_to(P))]=dict(bytes=f.stat().st_size,sha256=sha(f))
  dump('ATOMIC_SNAPSHOT_READY.json',dict(status='COMPLETE_ATOMIC_MANIFEST',experiment_status=RESULT.get('status'),files=manifest,created_unix=time.time()))
 except BaseException as se:dump('snapshot_error.json',dict(error=repr(se)))
 print(json.dumps(RESULT),flush=True)
