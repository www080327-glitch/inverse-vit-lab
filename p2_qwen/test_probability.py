import math
import pytest
import torch
from .probability import nll,draw,dequantize,patchify,unpatchify,unpack,EPS


def test_k1_reference_and_jacobian():
    torch.manual_seed(7)
    mu=torch.randn(2,3,1,48)*.1;logs=torch.full_like(mu,-.7);logits=torch.zeros(2,3,1)
    x=torch.rand(2,3,48)*1.8-.9;z=torch.atanh(x)
    ref=(-torch.distributions.Normal(mu.squeeze(2),logs.squeeze(2).exp()).log_prob(z)
         +torch.log1p(-x*x)).mean(-1)
    torch.testing.assert_close(nll((logits,mu,logs),x,'none'),ref)
    td=torch.distributions.TransformedDistribution(torch.distributions.Normal(mu.squeeze(2),logs.squeeze(2).exp()),
                                                   [torch.distributions.TanhTransform(cache_size=1)])
    torch.testing.assert_close(nll((logits,mu,logs),x,'none'),-td.log_prob(x).mean(-1))


def test_mixture_reference():
    torch.manual_seed(9)
    logits=torch.randn(2,3,8);mu=torch.randn(2,3,8,48);logs=torch.randn_like(mu).clamp(-2,0)
    x=torch.rand(2,3,48)*1.5-.75
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
         torch.distributions.Independent(torch.distributions.Normal(mu,logs.exp()),1))
    ref=(-dist.log_prob(torch.atanh(x))+torch.log1p(-x*x).sum(-1))/48
    torch.testing.assert_close(nll((logits,mu,logs),x,'none'),ref)


def test_dequantization_and_patch_order():
    u=torch.arange(3*32*32).reshape(1,3,32,32)
    assert torch.equal(unpatchify(patchify(u)),u)
    assert patchify(u)[0,0,:6].tolist()==[0,1024,2048,1,1025,2049]
    x,n=dequantize(torch.tensor([0,255],dtype=torch.uint8).repeat(3*32*32).reshape(2,3,32,32),torch.Generator().manual_seed(1))
    assert x.abs().max()<1 and torch.isfinite(torch.atanh(x)).all()


def test_sample_frequency_domain_and_saturation():
    n=5000;logits=torch.tensor([math.log(.2),math.log(.8)]).repeat(n,1)
    mu=torch.zeros(n,2,48);logs=torch.full_like(mu,-1.)
    # One shared mixture choice per whole patch. One shared generator is intentional for this frequency test.
    x,z,k=draw((logits,mu,logs),[torch.Generator().manual_seed(123)]*n)
    assert abs((k==0).float().mean().item()-.2)<.025
    torch.testing.assert_close(x,torch.tanh(z),rtol=0,atol=0)
    assert x.abs().max()<=1 and torch.isfinite(x).all()
    assert abs(z.var().item()-math.exp(-2))<.004
    x,z,_=draw((torch.zeros(1,1),torch.full((1,1,48),100.),torch.zeros(1,1,48)),[torch.Generator()])
    assert torch.equal(x,torch.ones_like(x)) and torch.isfinite(z).all()
    with pytest.raises(ValueError):nll((torch.zeros(1,1),torch.zeros(1,1,48),torch.zeros(1,1,48)),x)
    with pytest.raises(FloatingPointError):draw((torch.zeros(1,1),torch.full((1,1,48),float('nan')),torch.zeros(1,1,48)),[torch.Generator()])


def test_head_bounds_and_gradients():
    h=torch.randn(2,3,8*97,requires_grad=True);x=torch.rand(2,3,48)*1.8-.9
    p=unpack(h);loss=nll(p,x);loss.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    assert p[2].min()>=math.log(.01) and p[2].max()<=0
