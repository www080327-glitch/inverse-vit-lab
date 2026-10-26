"""Frozen, analytic distributions; all samples independent of model RNG."""
import math
import torch
from torch.nn import functional as F

def specification(d, condition):
    k = 2 if d == 1 else 4
    if d == 1:
        means = torch.tensor([[-2.],[2.]])
        weights = torch.tensor([[.25,.75],[.5,.5],[.7,.3],[.4,.6]])[condition]
    elif d == 2:
        means = torch.tensor([[-2.,-2.],[-2.,2.],[2.,-2.],[2.,2.]])
        weights = torch.roll(torch.tensor([.1,.2,.3,.4]),condition)
    elif d == 8:
        means = 1.5*torch.tensor([[1,1,1,1,1,1,1,1],
            [1,-1,1,-1,1,-1,1,-1],[-1,1,1,-1,-1,1,1,-1],
            [-1,-1,1,1,-1,-1,1,1]],dtype=torch.float32)
        weights = torch.roll(torch.tensor([.1,.2,.3,.4]),condition)
    else:
        raise ValueError(d)
    rotation = torch.eye(d)
    if d > 1:
        angle = condition*.37
        rotation[:2,:2] = torch.tensor([[math.cos(angle),-math.sin(angle)],
                                      [math.sin(angle),math.cos(angle)]])
    scale = .9 + .07*condition
    shift = .15*condition*torch.linspace(-1,1,d)
    means = scale*(means@rotation.T)+shift
    Ls = []
    for mode in range(k):
        rho = 0.0 if d == 1 else (.85 if mode%2 == 0 else -.8)
        covariance = (.4 if d == 1 else .55)**2 * (
            rho**(torch.arange(d)[:,None]-torch.arange(d)[None,:]).abs())
        # d=1: 0**0=1 by torch convention.
        covariance = scale**2*(rotation@covariance@rotation.T)
        Ls.append(torch.linalg.cholesky(covariance))
    return weights,means,torch.stack(Ls)

def draw(d, condition, n, generator):
    w,m,L = specification(d,condition)
    labels = torch.multinomial(w,n,replacement=True,generator=generator)
    e = torch.randn(n,d,generator=generator)
    return m[labels]+(L[labels]@e.unsqueeze(-1)).squeeze(-1),labels

def dataset(d,n_per_condition,seed):
    g = torch.Generator().manual_seed(seed)
    xs,cs,labels = [],[],[]
    for c in range(4):
        x,y = draw(d,c,n_per_condition,g)
        xs.append(x);cs.append(torch.full((n_per_condition,),c));labels.append(y)
    return torch.cat(cs),torch.cat(xs),torch.cat(labels)

def analytic_moments(d,c):
    w,m,L = specification(d,c)
    mean = (w[:,None]*m).sum(0)
    delta = m-mean
    covariance = (w[:,None,None]*(L@L.transpose(-1,-2)
                  +delta[:,:,None]*delta[:,None,:])).sum(0)
    return mean,covariance
