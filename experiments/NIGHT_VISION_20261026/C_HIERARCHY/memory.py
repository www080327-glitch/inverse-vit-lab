"""Lossless spatial storage; the coarse mean is explicitly lossy."""
from dataclasses import dataclass
from runtime import np, torch

@dataclass
class PixelMemory:
    pixels: torch.Tensor                 # [regions, p, p, channels], uint8
    boxes: torch.Tensor                  # [regions, 4], global x0,y0,x1,y1, clipped
    valid: torch.Tensor                  # padded pixels are never attended
    height: int
    width: int
    patch: int
    channels: int

    @classmethod
    def pack(cls, image, patch=8):
        x = torch.as_tensor(np.array(image, copy=True), device='cpu')
        if x.ndim != 3 or x.dtype != torch.uint8: raise ValueError('Expected HWC uint8')
        if patch < 1: raise ValueError('patch must be positive')
        h,w,c = x.shape; nr,nc = (h+patch-1)//patch, (w+patch-1)//patch
        padded = torch.zeros(nr*patch,nc*patch,c,dtype=x.dtype)
        padded[:h,:w] = x
        pixels = padded.reshape(nr,patch,nc,patch,c).permute(0,2,1,3,4).contiguous().reshape(-1,patch,patch,c)
        ys,xs = torch.meshgrid(torch.arange(nr)*patch, torch.arange(nc)*patch,indexing='ij')
        boxes = torch.stack((xs,ys,(xs+patch).clamp(max=w),(ys+patch).clamp(max=h)),dim=-1).reshape(-1,4)
        ly,lx = torch.meshgrid(torch.arange(patch),torch.arange(patch),indexing='ij')
        valid = ((ys.reshape(-1,1,1)+ly < h)&(xs.reshape(-1,1,1)+lx < w))
        return cls(pixels,boxes,valid,h,w,patch,c)

    def unpack(self):
        p = self.patch; nr,nc = (self.height+p-1)//p,(self.width+p-1)//p
        return self.pixels.reshape(nr,nc,p,p,self.channels).permute(0,2,1,3,4).reshape(nr*p,nc*p,self.channels)[:self.height,:self.width].numpy().copy()

    def summary(self):
        if self.patch != 8 or not self.valid.all():
            raise ValueError('Toy summary defined for complete 8x8 tiles only')
        # Gray average of RGB then non-overlapping 2x2 average -> 4x4 key input.
        return self.pixels.float().mean(-1).reshape(-1,4,2,4,2).mean((2,4)).reshape(-1,16)/255

    def gather(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device='cpu').reshape(-1).unique(sorted=True)
        if ids.numel()==0 or ids.min()<0 or ids.max()>=len(self.pixels): raise ValueError('Invalid region selection')
        return self.pixels.index_select(0,ids), self.valid.index_select(0,ids), ids

    def coordinates(self, ids):
        _,valid,ids = self.gather(ids)
        y,x = torch.meshgrid(torch.arange(self.patch),torch.arange(self.patch),indexing='ij')
        b = self.boxes[ids]
        # Coordinates are derived exactly, not redundantly stored for every pixel.
        return dict(x=b[:,0,None,None]+x,y=b[:,1,None,None]+y,valid=valid,
                    level=2,scale_pixels=1,parent_region=ids,region_level=1,
                    region_scale_pixels=self.patch,root_level=0,
                    root_bounds=[0,0,self.width,self.height],boxes=b)

    def storage(self):
        return {name: dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),
                           storage_bytes=t.untyped_storage().nbytes())
                for name,t in [('detail_pixels',self.pixels),('region_boxes',self.boxes),('valid_mask',self.valid)]}
