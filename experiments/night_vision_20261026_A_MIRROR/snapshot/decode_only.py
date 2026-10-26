"""Independent decode: features, saved normalization/config and weights only."""
import argparse,json,pathlib,os
os.environ.update(OMP_NUM_THREADS='2')
import torch
from safetensors.torch import load_file,save_file
from night_model import NightMirror
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();torch.set_num_threads(2)
root=a.root;snap=root/'snapshots'/a.name/'best';cfg=json.loads((snap/'config.json').read_text());policy=cfg['policy']
if policy['sdpa']=='math':torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
features=load_file(str(root/'results'/f'{a.name}_decode_input.safetensors'))['features'].cuda();norm={k:v.cuda() for k,v in load_file(str(snap/'normalization.safetensors')).items()}
model=NightMirror(a.name).cuda().eval();model.load_state_dict(load_file(str(snap/'model.safetensors')),strict=True)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=policy['autocast']=='bfloat16'):pred=model((features-norm['mean'])/norm['std']).float().cpu()
ref=load_file(str(root/'results'/f'{a.name}_decode_reference.safetensors'))['prediction'];tol=.005 if policy['autocast']=='bfloat16' else 1e-5
error=float((pred-ref).abs().max());ok=torch.allclose(pred,ref,atol=tol,rtol=tol)
save_file({'prediction':pred.contiguous()},str(root/'results'/f'{a.name}_independent_prediction.safetensors'))
(root/'results'/f'{a.name}_independent_reload.json').write_text(json.dumps(dict(status='PASS' if ok else 'FAIL',maximum_absolute_error=error,tolerance=tol,source_images_read=False,decoder_inputs=['features','normalization','config','weights']),indent=2)+'\n')
assert ok,'Independent reload differs beyond declared tolerance'
