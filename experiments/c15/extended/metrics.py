"""Bounded-memory distribution metrics. Never compare different training losses."""
import math
import torch
from scipy.stats import chi2
from data import specification, analytic_moments

def pair_mean(x,y,block=128,same=False):
    total = 0.
    for i in range(0,len(x),block):
        for j in range(0,len(y),block):
            dist = torch.cdist(x[i:i+block].double(),y[j:j+block].double(),
                               compute_mode="donot_use_mm_for_euclid_dist")
            if same and i == j:
                dist.fill_diagonal_(0)
            total += dist.sum().item()
    denom = len(x)*(len(x)-1) if same else len(x)*len(y)
    return total/denom/math.sqrt(x.shape[-1])

def energy_distance(x,y,block=128):
    # Unbiased U-statistic: finite-sample values may be negative; DO NOT clip.
    return 2*pair_mean(x,y,block)-pair_mean(x,x,block,True)-pair_mean(y,y,block,True)

def assign(x,d,c):
    w,m,L = specification(d,c)
    delta = x[:,None,:]-m
    whitened = torch.linalg.solve_triangular(L[None],delta[...,None],upper=False)[...,0]
    r2 = whitened.square().sum(-1)
    log_joint = -.5*r2-torch.diagonal(L,dim1=-2,dim2=-1).log().sum(-1)+w.log()
    label = log_joint.argmax(-1)
    gate = r2[torch.arange(len(x)),label] <= chi2.ppf(.99,d)
    proportions = torch.bincount(label,minlength=len(w)).float()/len(x)
    valid_proportions = torch.bincount(label[gate],minlength=len(w)).float()/len(x)
    return label,gate,proportions,valid_proportions

def evaluate(x,y,d,c,block=128):
    w,m,L = specification(d,c)
    mean,cov = analytic_moments(d,c)
    lab,gate,p,p_valid = assign(x,d,c)
    _,true_gate,true_p,_ = assign(y,d,c)
    covhat = torch.atleast_2d(torch.cov(x.T))
    qs = torch.tensor([.05,.25,.5,.75,.95])
    qx,qy = torch.quantile(x,qs,dim=0),torch.quantile(y,qs,dim=0)
    centered = x-m[lab]
    within = (centered[:,:,None]*centered[:,None,:]).mean(0)
    target_within = (w[:,None,None]*(L@L.transpose(-1,-2))).sum(0)
    off = ~torch.eye(d,dtype=torch.bool)
    return {
        "energy_u":energy_distance(x,y,block),
        "mode_coverage":float((p_valid >= .01).float().mean()),
        "mode_proportions":p.tolist(),"gated_mode_proportions":p_valid.tolist(),
        "true_weights":w.tolist(),"test_assigned_proportions":true_p.tolist(),
        "mode_tv_to_test":float(.5*(p-true_p).abs().sum()),
        "mode_tv_to_weights":float(.5*(p-w).abs().sum()),
        "valid_fraction":float(gate.float().mean()),
        "test_valid_fraction":float(true_gate.float().mean()),
        "mean_rmse":float((x.mean(0)-mean).square().mean().sqrt()),
        "cov_relative_fro":float((covhat-cov).norm()/cov.norm()),
        "within_cov_relative_fro":float((within-target_within).norm()/target_within.norm()),
        "within_offdiag_rmse":float((within-target_within)[off].square().mean().sqrt()) if d>1 else None,
        "quantile_mae":float((qx-qy).abs().mean()),
        "quantiles_generated":qx.tolist(),"quantiles_test":qy.tolist()
    }
