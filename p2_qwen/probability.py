"""Shared-component tanh Gaussian mixture: likelihood, draw and feedback agree."""
import math
import torch
from torch.nn import functional as F

EPS = 2**-20
LOG_SCALE_MIN, LOG_SCALE_MAX = math.log(.01), math.log(1.)


def patchify(x):
    return x.reshape(-1,3,8,4,8,4).permute(0,2,4,3,5,1).reshape(-1,64,48)


def unpatchify(p):
    return p.reshape(-1,8,8,4,4,3).permute(0,5,1,3,2,4).reshape(-1,3,32,32)


def dequantize(u8, rng):
    # Midpoint of one of 2**16 sub-bins: strictly inside (-1,1), including uint8=0/255.
    u = (torch.randint(65536,u8.shape,generator=rng).float()+.5)/65536
    x = 2*(u8.float()+u)/256-1
    protected = ((x <= -1+EPS) | (x >= 1-EPS)).sum().item()
    return patchify(x.clamp(-1+EPS,1-EPS)), int(protected)


def unpack(h, k=8, d=48):
    h=h.float()
    return (h[...,:k], h[...,k:k+k*d].reshape(*h.shape[:-1],k,d),
            h[...,k+k*d:].reshape(*h.shape[:-1],k,d).clamp(LOG_SCALE_MIN,LOG_SCALE_MAX))


def nll(params, x, reduction='mean'):
    if not torch.isfinite(x).all() or (x.abs()>=1).any():
        raise ValueError('Likelihood requires finite pixels strictly inside (-1,1)')
    logits,mu,logs=(v.float() for v in params)
    x=x.float(); z=torch.atanh(x)
    lp=(-.5*((z.unsqueeze(-2)-mu)*(-logs).exp()).square()-logs-.5*math.log(2*math.pi)).sum(-1)
    logpz=torch.logsumexp(F.log_softmax(logits,-1)+lp,-1)
    # dx/dz = 1-x^2: NLL_x = -log p_z + log |dx/dz|.
    losses=(-logpz+torch.log1p(-x.square()).sum(-1))/x.shape[-1]
    if not torch.isfinite(losses).all(): raise FloatingPointError('Nonfinite pixel likelihood')
    return losses.mean() if reduction=='mean' else losses


def draw(params, generators, t_mix=1., t_noise=1.):
    if t_mix<=0 or t_noise<0: raise ValueError('Invalid temperature')
    logits,mu,logs=(v.detach().float().cpu() for v in params)
    if not all(torch.isfinite(v).all() for v in (logits,mu,logs)):
        raise FloatingPointError('Nonfinite distribution parameters')
    zs=[]; ks=[]
    for i,g in enumerate(generators):
        k=torch.multinomial((logits[i]/t_mix).softmax(-1),1,generator=g).item()
        zs.append(mu[i,k]+logs[i,k].exp()*torch.randn(mu.shape[-1],generator=g)*t_noise);ks.append(k)
    z=torch.stack(zs);x=torch.tanh(z)
    if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite bounded sample')
    # Floating-point tanh may equal +/-1 for extreme z. Preserve that exact bounded value;
    # it is valid for feedback and is recorded, never transformed back by atanh in sampling.
    return x,z,torch.tensor(ks)
