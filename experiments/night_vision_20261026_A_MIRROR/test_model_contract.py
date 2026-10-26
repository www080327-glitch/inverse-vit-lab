"""No-download CPU/meta contract checks; no real model parameter allocation."""
import inspect,unittest
import torch
from night_model import NightMirror
from reference_core.config import parameter_estimate,mirror_full,mirror_tiny
from reference_core.layout import grouped_to_raster,raster_to_grouped,patchify_rgb,unpatchify

class Contracts(unittest.TestCase):
 def test_grouped_coordinate_values(self):
  x=torch.arange(256).reshape(1,256,1);g=raster_to_grouped(x,(1,16,16))
  self.assertEqual(g[0,0,:,0].tolist(),[0,1,16,17]);self.assertTrue(torch.equal(grouped_to_raster(g,(1,16,16)),x))
 def test_HWC_RGB_pixels(self):
  x=torch.arange(3*256*256).reshape(1,3,256,256)
  self.assertTrue(torch.equal(unpatchify(patchify_rgb(x,(1,16,16)),(1,16,16)),x))
 def test_full_and_tiny_meta_shapes_and_counts(self):
  for name,cfg in [('mirror_tiny',mirror_tiny()),('mirror_full',mirror_full())]:
   with torch.device('meta'):
    model=NightMirror(name);out=model(torch.empty(2,64,1024))
   self.assertEqual(tuple(out.shape),(2,3,256,256));self.assertEqual(sum(p.numel() for p in model.parameters()),parameter_estimate(cfg)['total'])
   self.assertTrue(all(p.device.type=='meta' for p in model.parameters()))
 def test_decoder_input_signature(self):
  self.assertEqual(list(inspect.signature(NightMirror.forward).parameters),['self','features','grid_thw'])

if __name__=='__main__':
 torch.set_num_threads(1);unittest.main(verbosity=2)
