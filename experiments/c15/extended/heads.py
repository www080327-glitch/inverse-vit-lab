"""C15 portable conditional heads. CPU tensors; generator is torch.Generator."""
import math
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("mse", "diag", "full", "ear")

def output_size(method, d, k):
    if method in ("mse", "ear"):
        return d
    return k + k*d + k*(d if method == "diag" else d*(d+1)//2)

def parameter_count(method, d, k, width, noise_dim=8, condition_dim=4):
    inp = condition_dim + (noise_dim if method == "ear" else 0)
    out = output_size(method, d, k)
    return (inp+1)*width + (width+1)*width + (width+1)*out

def energy_loss(a, b, target, alpha=1.0):
    """Negative EAR Eq.10 score; no squared norm unless explicitly alpha=2."""
    return ((a-target).norm(dim=-1).pow(alpha)
            +(b-target).norm(dim=-1).pow(alpha)
            -(a-b).norm(dim=-1).pow(alpha)).mean()

class Head(nn.Module):
    def __init__(self, method, d, k, target_parameters=12000, noise_dim=8,
                 scale_floor=.03, condition_dim=4):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.d, self.k = method, d, k
        self.noise_dim, self.scale_floor = noise_dim, scale_floor
        self.condition_dim = condition_dim
        self.width = min(range(24, 193),
                         key=lambda w: abs(parameter_count(method,d,k,w,noise_dim,condition_dim)
                                           -target_parameters))
        inp = condition_dim + (noise_dim if method == "ear" else 0)
        self.net = nn.Sequential(nn.Linear(inp,self.width), nn.SiLU(),
                                 nn.Linear(self.width,self.width), nn.SiLU(),
                                 nn.Linear(self.width,output_size(method,d,k)))
        self.register_buffer("tri",torch.tril_indices(d,d),persistent=False)

    def encode(self, condition):
        c = torch.as_tensor(condition,device=self.tri.device)
        if c.ndim == 2:
            if c.shape[-1] != self.condition_dim or not torch.isfinite(c).all():
                raise ValueError("Expected finite [B,condition_dim] condition features")
            return c.to(self.net[0].weight.dtype)
        if c.ndim == 0:
            c = c[None]
        if c.ndim != 1 or c.dtype.is_floating_point:
            raise ValueError("condition: integer IDs, or rank-2 continuous feature vectors")
        return F.one_hot(c.long(),self.condition_dim).to(self.net[0].weight.dtype)

    def forward(self, condition, noise=None):
        c = self.encode(condition)
        if self.method == "ear":
            if noise is None or noise.shape != (len(c),self.noise_dim):
                raise ValueError("EAR needs one independent noise vector per item")
            c = torch.cat((c,noise),-1)
        return self.net(c)

    def distribution_parameters(self, condition):
        if self.method not in ("diag","full"):
            raise ValueError("No Lebesgue density for this head")
        h = self(condition)
        logits = h[:,:self.k]
        mu = h[:,self.k:self.k+self.k*self.d].reshape(-1,self.k,self.d)
        raw = h[:,self.k+self.k*self.d:]
        if self.method == "diag":
            scale = F.softplus(raw.reshape(-1,self.k,self.d))+self.scale_floor
            return logits, mu, torch.diag_embed(scale)
        raw = raw.reshape(-1,self.k,self.d*(self.d+1)//2)
        L = raw.new_zeros(len(raw),self.k,self.d,self.d)
        L[...,self.tri[0],self.tri[1]] = raw
        diagonal = torch.diagonal(L,dim1=-2,dim2=-1)
        L = L - torch.diag_embed(diagonal) + torch.diag_embed(
            F.softplus(diagonal)+self.scale_floor)
        return logits,mu,L

    def log_prob(self, condition, target):
        logits, mu, L = self.distribution_parameters(condition)
        diff = target[:,None,:]-mu
        white = torch.linalg.solve_triangular(L,diff.unsqueeze(-1),upper=False).squeeze(-1)
        component = (-.5*white.square().sum(-1)
                     -torch.diagonal(L,dim1=-2,dim2=-1).log().sum(-1)
                     -.5*self.d*math.log(2*math.pi))
        # Product over coordinates INSIDE each component, then mixture.
        return torch.logsumexp(F.log_softmax(logits,-1)+component,dim=-1)

    def loss(self, condition, target, generator):
        if self.method == "mse":
            return (self(condition)-target).square().mean()
        if self.method in ("diag","full"):
            return -self.log_prob(condition,target).mean()/self.d
        c = self.encode(condition)
        noise = torch.rand((2,len(c),self.noise_dim),generator=generator,
                           device=c.device,dtype=c.dtype)-.5
        a,b = self(condition,noise[0]), self(condition,noise[1])
        return energy_loss(a,b,target,alpha=1)/math.sqrt(self.d)

    @torch.no_grad()
    def sample(self, condition, n, generator):
        """IDs or continuous [B,condition_dim] features -> [B,n,d]. Scalar IDs keep B=1."""
        if n < 1:
            raise ValueError("n must be positive")
        c = self.encode(condition)
        if self.method == "mse":
            return self(c)[:,None,:].expand(-1,n,-1).clone()
        if self.method == "ear":
            c_expanded = c.repeat_interleave(n,dim=0)
            noise = torch.rand((len(c_expanded),self.noise_dim),generator=generator,
                               device=c.device,dtype=c.dtype)-.5
            return self(c_expanded,noise).reshape(len(c),n,self.d)
        logits,mu,L = self.distribution_parameters(c)
        z = torch.multinomial(logits.softmax(-1),n,replacement=True,generator=generator)
        rows = torch.arange(len(c),device=c.device)[:,None]
        e = torch.randn((len(c),n,self.d),generator=generator,device=c.device,dtype=c.dtype)
        return mu[rows,z] + (L[rows,z]@e.unsqueeze(-1)).squeeze(-1)
