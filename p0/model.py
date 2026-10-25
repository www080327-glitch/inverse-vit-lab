"""Pure condition -> parallel continuous RGB patches. No dataset or drawing imports."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    name: str = "main_20m"
    condition_dim: int = 64
    image_size: int = 48
    patch_size: int = 12
    dim: int = 576
    depth: int = 5
    heads: int = 9
    ff_dim: int = 2304
    dropout: float = 0.0

    def __post_init__(self):
        if self.image_size % self.patch_size or self.dim % 4 or self.dim % self.heads:
            raise ValueError("Invalid patch/position/head dimensions")

    @property
    def grid(self):
        return self.image_size // self.patch_size


def patchify(images, patch_size=12):
    b, c, h, w = images.shape
    if c != 3 or h != w or h % patch_size:
        raise ValueError("Expected square RGB BCHW divisible by patch_size")
    g, p = h // patch_size, patch_size
    return images.reshape(b, 3, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(b, g*g, p*p*3)


def unpatchify(patches, grid=4, patch_size=12):
    b = patches.shape[0]
    if patches.shape[1:] != (grid*grid, patch_size*patch_size*3):
        raise ValueError("Wrong patch shape")
    return patches.reshape(b, grid, grid, patch_size, patch_size, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, grid*patch_size, grid*patch_size)


def position_2d(grid, dim):
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = torch.exp(-math.log(10000.) * torch.arange(dim//4) / (dim//4))
    xx, yy = x.flatten()[:, None]*omega, y.flatten()[:, None]*omega
    return torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=1).unsqueeze(0)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.dim)
        self.attn = nn.MultiheadAttention(c.dim, c.heads, dropout=c.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(c.dim)
        self.ff = nn.Sequential(nn.Linear(c.dim, c.ff_dim), nn.GELU(), nn.Linear(c.ff_dim, c.dim), nn.Dropout(c.dropout))

    def forward(self, x):
        z = self.ln1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]  # bidirectional, no causal mask
        return x + self.ff(self.ln2(x))


class InverseViT(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.condition_projection = nn.Linear(config.condition_dim, config.dim)
        self.queries = nn.Parameter(torch.randn(1, config.grid**2, config.dim)*0.02)
        self.register_buffer("positions", position_2d(config.grid, config.dim))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.final_norm = nn.LayerNorm(config.dim)
        self.pixel_head = nn.Linear(config.dim, config.patch_size**2*3)

    def forward(self, conditions, return_patches=False):
        if conditions.ndim != 2 or conditions.shape[1] != self.config.condition_dim:
            raise ValueError("conditions must be B x 64 for the P0 configs")
        token = self.condition_projection(conditions).unsqueeze(1)
        queries = (self.queries + self.positions).expand(conditions.shape[0], -1, -1)
        x = torch.cat([token, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        patches = self.pixel_head(self.final_norm(x[:, 1:]))
        return patches if return_patches else unpatchify(patches, self.config.grid, self.config.patch_size)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
