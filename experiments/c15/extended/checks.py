"""Independent formula and API checks, no training or downloads."""
import json
import math
import torch
from heads import Head,energy_loss
from metrics import energy_distance
from data import dataset,specification

def checks():
    torch.manual_seed(789)
    report = {}
    for method in ("diag","full"):
        h = Head(method,2,3).double()
        c = torch.tensor([0,1,2,3])
        y = torch.randn(4,2,dtype=torch.double)
        logits,mu,L = h.distribution_parameters(c)
        reference = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits),
            torch.distributions.MultivariateNormal(mu,scale_tril=L))
        delta = (h.log_prob(c,y)-reference.log_prob(y)).abs().max().item()
        assert delta<1e-9
        loss = -h.log_prob(c,y).mean()
        # encode is float32; model remains float32 to check train paths below.
        assert torch.linalg.eigvalsh(L@L.transpose(-1,-2)).min()>0
        report[method+"_logprob_max_error"] = delta
    # Shared-component density vs independent per-coordinate mixtures.
    mu = torch.tensor([[-2.,-2.],[2.,2.]])
    x = torch.tensor([[-2.,2.]])
    lp = -.5*((x[:,None]-mu)/.2).square()-math.log(.2)-.5*math.log(2*math.pi)
    shared = torch.logsumexp(lp.sum(-1)-math.log(2),-1)
    factorized = torch.logsumexp(lp-math.log(2),-2).sum(-1)
    assert (factorized-shared).item()>100
    report["wrong_factorized_minus_joint_logprob"] = (factorized-shared).item()
    # Exact finite atomic example: target +/-1, candidate collapse0 / +/-2.
    y = torch.tensor([[-1.],[1.]])
    def exact_energy(x,y,alpha):
        cross = torch.cdist(x,y).pow(alpha).mean()
        return float(2*cross-torch.cdist(x,x).pow(alpha).mean()
                     -torch.cdist(y,y).pow(alpha).mean())
    report["squared_energy_equal_mean"] = {}
    for name,x in [("truth",y),("collapsed",torch.zeros(1,1)),
                   ("wrong_variance",torch.tensor([[-2.],[2.]]))]:
        e1,e2 = exact_energy(x,y,1),exact_energy(x,y,2)
        assert abs(e2)<1e-7
        report["squared_energy_equal_mean"][name] = {"alpha1":e1,"alpha2":e2}
    for method in ("mse","diag","full","ear"):
        h = Head(method,8,4)
        c = torch.tensor([0,1,2,3]);t = torch.randn(4,8)
        loss = h.loss(c,t,torch.Generator().manual_seed(1))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in h.parameters())
        a = h.sample(c,13,torch.Generator().manual_seed(21))
        b = h.sample(c,13,torch.Generator().manual_seed(21))
        assert a.shape == (4,13,8) and torch.equal(a,b)
        assert h.sample(2,7,torch.Generator().manual_seed(21)).shape == (1,7,8)
    a,b,t = torch.randn(5,8),torch.randn(5,8),torch.randn(5,8)
    expected = ((a-t).norm(dim=-1)+(b-t).norm(dim=-1)-(a-b).norm(dim=-1)).mean()
    assert torch.allclose(energy_loss(a,b,t),expected)
    x,y = torch.randn(17,2),torch.randn(19,2)
    dense = (2*torch.cdist(x.double(),y.double()).mean()
        -torch.cdist(x.double(),x.double()).sum()/(17*16)
        -torch.cdist(y.double(),y.double()).sum()/(19*18))/math.sqrt(2)
    assert abs(energy_distance(x,y,4)-dense.item())<1e-9
    for d in (1,2,8):
        c,x,_ = dataset(d,128,10);_,y,_ = dataset(d,128,11)
        assert not torch.equal(x,y)
        assert all(int((c==i).sum())==128 for i in range(4))
        for i in range(4):
            w,m,L = specification(d,i)
            assert torch.linalg.eigvalsh(L@L.transpose(-1,-2)).min()>0
    report["api_gradients_and_generator"] = "pass"
    report["bounded_pair_distance_matches_dense"] = "pass"
    report["independent_samples_repeated_conditions"] = "pass"
    return report

if __name__ == "__main__":
    print(json.dumps(checks(),indent=2))
