"""Fixed 16-example posthoc diagnostic, frozen decoders only; no encoder/training."""
import pathlib,sys,json,time,os,zipfile,hashlib,urllib.request,traceback
SOURCE=pathlib.Path(sys.argv[1]).resolve();SECRET=pathlib.Path(sys.argv[2]);OUT=pathlib.Path.cwd()/'posthoc_pre_post_16';OUT.mkdir(exist_ok=False)
start=time.time();budget=json.loads((SOURCE/'night_budget.json').read_text());assert start<budget['started_unix']+budget['user_cap_seconds']
os.environ.update(OMP_NUM_THREADS='2',XDG_CACHE_HOME=str(OUT/'cache'))
archive=SOURCE/'export/NIGHT_VISION_20261026_A_MIRROR_reports.zip';transport=json.loads((SOURCE/'export/TRANSPORT_READY.json').read_text());mapping=transport['path_to_asset']
with zipfile.ZipFile(archive) as z:
 for n in ['night_model.py','reference_core/__init__.py','reference_core/config.py','reference_core/layout.py','reference_core/decoder.py']:
  p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
 for n in ['feature_evidence/paired_pre_post_16.safetensors','feature_evidence/samples.json']:
  p=OUT/pathlib.Path(n).name;p.write_bytes(z.read(n))
 configs={name:json.loads(z.read(f'checkpoints/{name}/best/config.json')) for name in ['pre_shallow','post_shallow']}
sys.path.insert(0,str(OUT))
import torch
from safetensors.torch import load_file,save_file
from PIL import Image,ImageDraw
from night_model import NightMirror
torch.set_num_threads(2);assert torch.cuda.is_available() and torch.version.hip
data=load_file(str(OUT/'paired_pre_post_16.safetensors'));preds={};shapes={}
for name,key in [('pre_shallow','pre'),('post_shallow','post')]:
 norm=load_file(str(SOURCE/'export'/mapping[f'checkpoints/{name}/best/normalization.safetensors']));weights=SOURCE/'export'/mapping[f'checkpoints/{name}/best/model.safetensors']
 model=NightMirror(name).cuda().eval().requires_grad_(False);model.load_state_dict(load_file(str(weights)),strict=True)
 policy=configs[name]['policy']
 if policy['sdpa']=='math':torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 features=data[key].cuda();features=(features-norm['mean'].cuda())/norm['std'].cuda()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=policy['autocast']=='bfloat16'):pred=model(features)
 torch.cuda.synchronize();preds[key+'_prediction']=pred.cpu().contiguous();shapes[key]=list(pred.shape)
 del model,features,pred;torch.cuda.empty_cache()
save_file(preds,str(OUT/'predictions_unclipped.safetensors'))
samples=json.loads((OUT/'samples.json').read_text())['samples']
for page in range(2):
 canvas=Image.new('RGB',(1114,2254),'white');draw=ImageDraw.Draw(canvas)
 for col,label in enumerate(['source32','target256','pre correct','post correct']):draw.text((94+256*col,8),label,fill='black')
 for row in range(8):
  i=page*8+row;draw.text((2,40+278*row),samples[i]['split']+' '+str(i),fill='black')
  arrays=[data['original_RGB_u8'][i].numpy(),data['target_RGB_u8'][i].numpy()]
  for key in ['pre_prediction','post_prediction']:arrays.append(preds[key][i].float().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy())
  for col,a in enumerate(arrays):canvas.paste(Image.fromarray(a).resize((256,256),Image.Resampling.NEAREST),(90+col*256,30+row*278))
 canvas.save(OUT/f'gallery_pre_post_correct_{page:02d}.png')
ledger=dict(status='COMPLETE_FROZEN_POSTHOC_DIAGNOSTIC',started_unix=start,wall_seconds=time.time()-start,original_budget_started_unix=budget['started_unix'],original_compute_wall_seconds=budget['spent_seconds'],original_budget_cap_seconds=budget['user_cap_seconds'],new_training_steps=0,new_encoder_forwards=0,decoder_inference_only=True,fixed_samples=16,train_samples=8,dev_samples=8,final_samples=0,config_selection=False,output_shapes=shapes,note='All 16 predeclared paired interface examples shown; correct-feature diagnostic only. Does not replace full 1000/200 metrics. PNG clipped for display; raw predictions retained.')
(OUT/'DIAGNOSTIC_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
result=OUT/'posthoc_pre_post_16.zip'
with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in ['gallery_pre_post_correct_00.png','gallery_pre_post_correct_01.png','predictions_unclipped.safetensors','DIAGNOSTIC_LEDGER.json','samples.json']:z.write(OUT/n,n)
asset=dict(name=result.name,bytes=result.stat().st_size,sha256=hashlib.sha256(result.read_bytes()).hexdigest(),encoding='zip',scope='supplemental frozen diagnostic; core snapshot unchanged')
cfg=json.loads(SECRET.read_text());base=cfg['url'].rstrip('/');headers={'Authorization':'Bearer '+cfg['token']}
def req(method,path,body=None,extra=None):
 h=dict(headers);h.update(extra or {});r=urllib.request.urlopen(urllib.request.Request(base+path,data=body,headers=h,method=method),timeout=100);out={k.lower():v for k,v in r.headers.items()};r.read();r.close();return out
req('POST','/manifest',json.dumps({'assets':[asset],'complete':False}).encode(),{'Content-Type':'application/json'})
offset=int(req('HEAD','/asset/'+asset['name'])['x-offset'])
with result.open('rb') as f:
 f.seek(offset)
 while offset<asset['bytes']:
  b=f.read(16*1024*1024);r=req('PUT','/asset/'+asset['name'],b,{'X-Offset':str(offset),'Content-Type':'application/octet-stream'});offset=int(r['x-offset'])
print(json.dumps(dict(ledger=ledger,asset=asset,uploaded_bytes=offset)),flush=True)
