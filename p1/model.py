"""Inference-safe model: torch only; no images, data, rendering or retrieval."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    dim: int = 576
    depth: int = 5
    heads: int = 9
    ff_dim: int = 2304
    dropout: float = 0.1
    grid: int = 8
    patch_size: int = 4
    mixtures: int = 8
    sigma_min: float = 0.01
    sigma_max: float = 1.0

    def __post_init__(self):
        assert self.dim % 4 == 0 and self.dim % self.heads == 0
        assert 0 < self.sigma_min < self.sigma_max

    @property
    def patch_dim(self):
        return self.patch_size ** 2 * 3


def patchify(images, p=4):
    b, c, h, w = images.shape
    assert c == 3 and h == w and h % p == 0
    g = h // p
    return images.reshape(b, 3, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(b, g*g, p*p*3)


def unpatchify(patches, g=8, p=4):
    assert patches.shape[1:] == (g*g, p*p*3)
    return patches.reshape(-1, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, g*p, g*p)


def positions_2d(g, d):
    y, x = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
    w = torch.exp(-math.log(10000.) * torch.arange(d//4) / (d//4))
    xx, yy = x.flatten()[:, None]*w, y.flatten()[:, None]*w
    return torch.cat((xx.sin(), xx.cos(), yy.sin(), yy.cos()), -1)[None]


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.drop = c.heads, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)
        self.qkv, self.proj = nn.Linear(c.dim, 3*c.dim), nn.Linear(c.dim, c.dim)
        self.ff = nn.Sequential(nn.Linear(c.dim, c.ff_dim), nn.GELU(), nn.Linear(c.ff_dim, c.dim), nn.Dropout(c.dropout))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(b, n, d)), self.drop, self.training)
        return x + self.ff(self.ln2(x))


class PixelAR(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        self.classes = nn.Embedding(10, c.dim)
        self.bos = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.input_projection = nn.Linear(c.patch_dim, c.dim)
        self.register_buffer('positions', positions_2d(c.grid, c.dim))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.depth)])
        self.norm = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.mixtures*(1+2*c.patch_dim))
        nn.init.normal_(self.classes.weight, std=.02)
        nn.init.normal_(self.bos, std=.02)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        # Head fields: logits, unconstrained means, clamped log sigma. sigma starts .3.
        with torch.no_grad():
            self.head.bias[c.mixtures*(1+c.patch_dim):].fill_(math.log(.3))

    def input_tokens(self, class_ids, prefix):
        """prefix contains ONLY completed patches. BOS predicts p0; p_i predicts p_(i+1)."""
        c = self.config
        assert class_ids.ndim == 1 and prefix.shape == (len(class_ids), prefix.shape[1], c.patch_dim)
        assert 0 <= prefix.shape[1] < c.grid*c.grid
        predictions = torch.cat((self.bos.expand(len(class_ids), -1, -1), self.input_projection(prefix)), 1)
        predictions = predictions + self.positions[:, :predictions.shape[1]]
        return torch.cat((self.classes(class_ids)[:, None], predictions), 1)

    def forward(self, class_ids, prefix):
        c = self.config
        x = self.input_tokens(class_ids, prefix)
        for block in self.blocks:
            x = block(x)
        h = self.head(self.norm(x[:, 1:]))  # Class token has no pixel loss.
        k, m = c.mixtures, c.patch_dim
        logits = h[..., :k]
        means = h[..., k:k+k*m].reshape(*h.shape[:2], k, m)
        logs = h[..., k+k*m:].reshape_as(means).clamp(math.log(c.sigma_min), math.log(c.sigma_max))
        return logits, means, logs

    def teacher(self, class_ids, targets):
        assert targets.shape[1:] == (self.config.grid**2, self.config.patch_dim)
        return self(class_ids, targets[:, :-1])

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def gmm_nll(params, target, reduction='mean'):
    """Continuous density, FP32, one mixture component shared by all 48 patch channels."""
    logits, means, logs = (v.float() for v in params)
    z = (target.float().unsqueeze(-2)-means) * torch.exp(-logs)
    component = (-.5*z.square()-logs-.5*math.log(2*math.pi)).sum(-1)
    losses = -torch.logsumexp(F.log_softmax(logits, -1)+component, -1)/target.shape[-1]
    return losses.mean() if reduction == 'mean' else losses


def draw_patch(params, generators, t_mix=1., t_noise=1.):
    """CPU RNG per image, independent of device/batch grouping. No clipping on feedback."""
    if t_mix <= 0 or t_noise < 0:
        raise ValueError('Invalid temperature')
    logits, means, logs = [v.detach().cpu() for v in params]
    result, components = [], []
    for i, gen in enumerate(generators):
        k = torch.multinomial((logits[i]/t_mix).softmax(-1), 1, generator=gen).item()
        z = torch.randn(means.shape[-1], generator=gen)
        result.append(means[i, k]+logs[i, k].exp()*z*t_noise)
        components.append(k)
    return torch.stack(result), components


@torch.inference_mode()
def generate(model, class_ids, seeds, t_mix=1., t_noise=1., on_step=None):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(class_ids, device=device, dtype=torch.long)
    generators = [torch.Generator(device='cpu').manual_seed(int(s)) for s in seeds]
    if len(generators) != len(ids):
        raise ValueError('One seed per image required')
    prefix = torch.empty(len(ids), 0, model.config.patch_dim, device=device)
    components = []
    for step in range(model.config.grid**2):
        params = tuple(v[:, -1] for v in model(ids, prefix))
        patch, ks = draw_patch(params, generators, t_mix, t_noise)
        if not torch.isfinite(patch).all():
            raise FloatingPointError('Nonfinite sample')
        prefix = torch.cat((prefix, patch[:, None].to(device)), 1)
        components.append(ks)
        if on_step:
            on_step(step+1, prefix.detach().cpu())
    return prefix.cpu(), torch.tensor(components).T
