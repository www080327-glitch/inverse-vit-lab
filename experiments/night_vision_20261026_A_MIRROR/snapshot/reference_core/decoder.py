"""Inference-safe: forward accepts only post-merger features and image grid.

No source images, IDs, dataset imports, plotting, retrieval, encoder weights,
negative RoPE, transposed encoder weights, or language model are involved.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import mirror_tiny
from .layout import grouped_to_raster, image_grid, unpatchify


def position_2d(grid_thw, dim, *, device=None, dtype=torch.float32):
    """Fixed absolute patch coordinates: sin(xω),cos(xω),sin(yω),cos(yω)."""
    h, w = image_grid(grid_thw)
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing="ij")
    omega = torch.exp(-math.log(10000.) * torch.arange(dim//4, device=device, dtype=dtype)/(dim//4))
    xx, yy = x.flatten()[:, None]*omega, y.flatten()[:, None]*omega
    return torch.cat((xx.sin(), xx.cos(), yy.sin(), yy.cos()), -1)[None]


class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.norm1 = nn.LayerNorm(c.dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(c.dim, eps=1e-6)
        self.qkv = nn.Linear(c.dim, 3*c.dim)
        self.proj = nn.Linear(c.dim, c.dim)
        self.ff = nn.Sequential(nn.Linear(c.dim, c.ff_dim), nn.GELU(approximate="tanh"),
                                nn.Linear(c.ff_dim, c.dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.norm2(x))


class MirrorDecoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = self.config = config or mirror_tiny()
        if c.name != "mirror_tiny" or c != mirror_tiny():
            raise ValueError("This local prototype allocates mirror_tiny only; mirror_full is config/estimate only")
        self.unmerge = nn.Linear(c.input_dim, c.merge_size**2*c.dim)
        self.blocks = nn.ModuleList([DecoderBlock(c) for _ in range(c.depth)])
        self.norm = nn.LayerNorm(c.dim, eps=1e-6)
        self.pixel_projection = nn.Linear(c.dim, c.patch_size**2*3)

    def forward(self, features, grid_thw):
        h, w = image_grid(grid_thw, self.config.merge_size)
        if features.ndim != 3 or features.shape[1:] != (h*w//4, self.config.input_dim):
            raise ValueError("Expected [B,H*W/4,1024] official post-merger features")
        if not features.is_floating_point():
            raise ValueError("Features must be floating point")
        b, n, _ = features.shape
        x = self.unmerge(features).reshape(b, n, 4, self.config.dim)
        x = grouped_to_raster(x, grid_thw, self.config.merge_size)
        x = x + position_2d(grid_thw, self.config.dim, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        return unpatchify(self.pixel_projection(self.norm(x)), grid_thw, self.config.patch_size)
