"""New full-authorized entry; original tiny-only prototype remains read-only."""
import torch
from torch import nn
from reference_core.config import mirror_tiny,mirror_full,MirrorConfig
from reference_core.decoder import DecoderBlock,position_2d
from reference_core.layout import grouped_to_raster,unpatchify

def configuration(name):
 if name=='mirror_full':return mirror_full()
 if name=='mirror_tiny':return mirror_tiny()
 if name in ('pre_shallow','post_shallow'):
  return MirrorConfig(name=name,dim=768,depth=2,heads=12,ff_dim=3072)
 raise ValueError(name)

class NightMirror(nn.Module):
 def __init__(self,name):
  super().__init__();self.config=c=configuration(name);self.name=name
  self.adapter=nn.Identity() if name=='pre_shallow' else nn.Linear(1024,4*c.dim)
  self.blocks=nn.ModuleList([DecoderBlock(c) for _ in range(c.depth)])
  self.norm=nn.LayerNorm(c.dim,eps=1e-6)
  self.pixel_projection=nn.Linear(c.dim,16*16*3)
  self.register_buffer('position',position_2d((1,16,16),c.dim),persistent=False)
 def forward(self,features,grid_thw=(1,16,16)):
  if tuple(grid_thw)!=(1,16,16):raise ValueError('Frozen official still-image geometry only')
  if self.name=='pre_shallow':
   if features.shape[1:]!=(256,768):raise ValueError('pre-merger shape')
   x=grouped_to_raster(features.reshape(-1,64,4,768),grid_thw)
  else:
   if features.shape[1:]!=(64,1024):raise ValueError('post-merger shape')
   x=self.adapter(features).reshape(-1,64,4,self.config.dim)
   x=grouped_to_raster(x,grid_thw)
  x=x+self.position.to(dtype=x.dtype)
  for block in self.blocks:x=block(x)
  return unpatchify(self.pixel_projection(self.norm(x)),grid_thw,16)

def small_layout_checks():
 from reference_core.layout import raster_to_grouped,patchify_rgb,processor_pixels_to_rgb,grouped_coordinates
 x=torch.arange(256).reshape(1,256,1)
 z=raster_to_grouped(x,(1,16,16))
 assert z[0,0,:,0].tolist()==[0,1,16,17]
 assert torch.equal(grouped_to_raster(z,(1,16,16)),x)
 rgb=torch.arange(3*256*256).reshape(1,3,256,256)
 assert torch.equal(unpatchify(patchify_rgb(rgb,(1,16,16)),(1,16,16)),rgb)
 from transformers import AutoImageProcessor
 from pathlib import Path
 proc=AutoImageProcessor.from_pretrained(str(Path(__file__).parent/'model'),local_files_only=True,use_fast=False)
 ramp=torch.arange(3*64*96).reshape(3,64,96).float().numpy()
 a=proc(images=ramp,do_resize=False,do_rescale=False,do_normalize=False,return_tensors='pt')
 recovered=processor_pixels_to_rgb(a['pixel_values'],(1,4,6),mean=(0,0,0),std=(1,1,1))
 assert torch.equal(recovered[0],torch.from_numpy(ramp))
 return {'grouped_raster':True,'RGB_pixels':True,'actual_processor_temporal_and_layout':True}
