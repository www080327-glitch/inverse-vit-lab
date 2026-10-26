"""C13 frozen tiny-encoder initialization experiment. NumPy only, CPU <=2 threads."""
import os
# Must precede NumPy import; override inherited high thread counts.
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_key]="2"
import argparse, copy, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
from math_examples import examples

CONFIG = {"seeds":[13,29], "train_n":768, "val_n":192, "image_size":16,
          "encoder_dims":[256,64,64,24], "decoder_dims":[24,64,64,256],
          "encoder_steps":500, "decoder_steps":600, "batch_size":64,
          "encoder_lr":.002, "decoder_lr":.002, "classification_weight":.1,
          "pinv_rcond":.001, "threshold_val_mse":.02,
          "train_budget_seconds":1100, "eval_every":25,
          "data_seeds":[1300,1301], "checkpoint_rule":"fixed final step; no best-val selection"}
TRAIN_SECONDS = 0.
GUARD_START = time.monotonic()

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def make_data(n, seed):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:16,:16].astype(float)
    images=[]; labels=[]; params=[]
    for i in range(n):
        k=int(rng.integers(4)); cx=float(rng.uniform(3,12)); cy=float(rng.uniform(3,12))
        sx=float(rng.uniform(1.5,4)); sy=float(rng.uniform(1.5,4))
        theta=float(rng.uniform(-np.pi,np.pi))
        dx=xx-cx; dy=yy-cy
        u=np.cos(theta)*dx+np.sin(theta)*dy
        v=-np.sin(theta)*dx+np.cos(theta)*dy
        if k==0: mask=(np.abs(u)<sx)&(np.abs(v)<sy)
        elif k==1: mask=(u/sx)**2+(v/sy)**2<1
        elif k==2: mask=(np.abs(u)<sx)&(np.abs(v)<.55)
        else: mask=(v>-sy)&(v<sy)&(np.abs(u)<sx*(sy-v)/(2*sy))
        bg=float(rng.uniform(.02,.15)); fg=float(rng.uniform(.6,1.))
        im=np.where(mask,fg,bg)
        # Thin attachment exists independently of the class.
        ax=int(rng.integers(1,15)); ay=int(rng.integers(1,15))
        im[max(0,ay-1):min(16,ay+2),ax]=fg
        images.append(im.reshape(-1)*2-1); labels.append(k)
        params.append([k,cx,cy,sx,sy,theta,bg,fg,ax,ay])
    return np.asarray(images),np.asarray(labels),params

def init_mlp(dims,rng):
    p={}
    for i,(din,dout) in enumerate(zip(dims[:-1],dims[1:])):
        p[f"W{i}"]=rng.normal(0,np.sqrt(2/(din+dout)),(din,dout))
        p[f"b{i}"]=np.zeros(dout)
    return p

def fwd(p,x):
    a=[x]; L=len(p)//2
    for i in range(L):
        h=a[-1]@p[f"W{i}"]+p[f"b{i}"]
        a.append(np.tanh(h) if i<L-1 else h)
    return a[-1],a

def bwd(p,a,g):
    grads={}; L=len(p)//2
    for i in range(L-1,-1,-1):
        if i<L-1: g=g*(1-a[i+1]**2)
        grads[f"W{i}"]=a[i].T@g
        grads[f"b{i}"]=g.sum(0)
        g=g@p[f"W{i}"].T
    return grads,g

class Adam:
    def __init__(self,p,lr):
        self.m={k:np.zeros_like(v) for k,v in p.items()}
        self.v={k:np.zeros_like(v) for k,v in p.items()}
        self.t=0; self.lr=lr
    def step(self,p,g):
        self.t+=1
        assert set(p)==set(g)
        for k in p:
            assert np.isfinite(g[k]).all(),k
            self.m[k]=.9*self.m[k]+.1*g[k]
            self.v[k]=.999*self.v[k]+.001*g[k]**2
            p[k]-=self.lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)

def guard():
    if TRAIN_SECONDS>CONFIG["train_budget_seconds"] or time.monotonic()-GUARD_START>1150:
        raise TimeoutError("C13 cumulative budget exceeded")

def finite_difference_test():
    rng=np.random.default_rng(99); p=init_mlp([3,4,2],rng)
    x=rng.normal(size=(5,3)); t=rng.normal(size=(5,2))
    y,a=fwd(p,x); g,_=bwd(p,a,2*(y-t)/y.size)
    err=[]
    for k in p:
        j=(0,0) if p[k].ndim==2 else (0,)
        v=p[k][j]; e=1e-6
        p[k][j]=v+e; yp,_=fwd(p,x); lp=np.mean((yp-t)**2)
        p[k][j]=v-e; ym,_=fwd(p,x); lm=np.mean((ym-t)**2)
        p[k][j]=v
        err.append(abs((lp-lm)/(2*e)-g[k][j]))
    assert max(err)<1e-7
    return max(err)

def metrics(y,x):
    # Unclipped normalized 0..1 intensities, evaluated in the original image scale.
    d=(y-x).reshape(-1,16,16)/2
    return {"mse":float(np.mean(d*d)),
            "edge_mse":float((np.mean(np.diff(d,axis=1)**2)+np.mean(np.diff(d,axis=2)**2))/2)}

def count(p): return sum(v.size for v in p.values())

def train_encoder(x,labels,xv,labelsv,seed):
    global TRAIN_SECONDS
    rng=np.random.default_rng(seed)
    enc=init_mlp(CONFIG["encoder_dims"],rng)
    aux=init_mlp(CONFIG["decoder_dims"],rng)
    cls={"W0":rng.normal(0,.1,(24,4)), "b0":np.zeros(4)}
    oe=Adam(enc,CONFIG["encoder_lr"]); od=Adam(aux,CONFIG["encoder_lr"])
    oc=Adam(cls,CONFIG["encoder_lr"])
    order=np.random.default_rng(seed+1000).integers(len(x),size=(CONFIG["encoder_steps"],64))
    start=time.monotonic()
    for ids in order:
        guard()
        z,ea=fwd(enc,x[ids]); out,da=fwd(aux,z)
        gd,gz=bwd(aux,da,2*(out-x[ids])/out.size)
        logits,ca=fwd(cls,z); p=np.exp(logits-logits.max(1,keepdims=True))
        p/=p.sum(1,keepdims=True); p[np.arange(len(ids)),labels[ids]]-=1
        gc,gzc=bwd(cls,ca,p*(CONFIG["classification_weight"]/len(ids)))
        ge,_=bwd(enc,ea,gz+gzc)
        oe.step(enc,ge); od.step(aux,gd); oc.step(cls,gc)
    elapsed=time.monotonic()-start; TRAIN_SECONDS+=elapsed
    def stat(a,l):
        z,_=fwd(enc,a); y,_=fwd(aux,z); logits,_=fwd(cls,z)
        return {**metrics(y,a),"class_accuracy":float(np.mean(logits.argmax(1)==l))}
    return enc,{"seconds":elapsed,"parameters_encoder":count(enc),
                "parameters_aux_decoder":count(aux),"parameters_classifier":count(cls),
                "train":stat(x,labels),"val":stat(xv,labelsv),
                "schedule_sha256":sha(order),"snapshot_sha256":sha(np.concatenate([v.ravel() for v in enc.values()]))}

def mapped(base,enc,mode):
    p=copy.deepcopy(base); mapping=[]
    for j in range(3):
        i=2-j
        if mode=="B": rev=enc[f"W{i}"].T.copy(); action="transpose"
        elif mode=="C" and j in (0,2):
            rev=np.linalg.pinv(enc[f"W{i}"],rcond=CONFIG["pinv_rcond"])
            action="Moore-Penrose pseudoinverse (truncated SVD rcond=0.001)"
        else: rev=None; action="independent paired random baseline"
        if rev is not None:
            assert p[f"W{j}"].shape==rev.shape
            p[f"W{j}"]=rev
            p[f"b{j}"]=-enc[f"b{i}"]@rev
        mapping.extend([
          {"decoder":f"W{j}","shape":list(p[f"W{j}"].shape),
           "encoder":f"W{i}" if rev is not None else None,"operation":action},
          {"decoder":f"b{j}","shape":list(p[f"b{j}"].shape),
           "encoder":f"b{i}, W{i}" if rev is not None else None,
           "operation":"-encoder_bias @ reverse_weight (linear affine heuristic)" if rev is not None else "paired zero bias"}])
    return p,mapping

def fit_decoder(p,z,x,zv,xv,order):
    global TRAIN_SECONDS
    opt=Adam(p,CONFIG["decoder_lr"]); curve=[]
    elapsed=0.
    def record(step):
        y,_=fwd(p,z); yv,_=fwd(p,zv)
        curve.append({"step":step,"seen_samples":step*64,
                      "optimization_seconds":elapsed,"train":metrics(y,x),"val":metrics(yv,xv)})
    record(0)
    for step,ids in enumerate(order,1):
        guard(); t=time.monotonic()
        y,a=fwd(p,z[ids]); g,_=bwd(p,a,2*(y-x[ids])/y.size)
        opt.step(p,g)
        delta=time.monotonic()-t; elapsed+=delta; TRAIN_SECONDS+=delta
        if step in (1,10) or step%CONFIG["eval_every"]==0: record(step)
    yv,_=fwd(p,zv)
    steps=np.array([r["step"] for r in curve]); losses=np.array([r["val"]["mse"] for r in curve])
    auc=float(np.trapezoid(losses,steps)/CONFIG["decoder_steps"])
    hit=next((r["step"] for r in curve if r["val"]["mse"]<=CONFIG["threshold_val_mse"]),None)
    return {"initial":curve[0],"final":curve[-1],"curve":curve,"val_mse_auc":auc,
            "first_recorded_step_val_mse_le_002":hit,"optimization_seconds":elapsed,
            "final_parameters_sha256":sha(np.concatenate([v.ravel() for v in p.values()]))},yv

def svg_montage(path,x,predictions):
    # First eight fixed validation samples, all modes and both seeds.
    rows=[("target",x[:8])]
    rows.extend((label,a[:8]) for label,a in predictions)
    W=8*64+130; H=len(rows)*78+36
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<text x="10" y="18" font-size="13">C13: first 8 validation images; display clipped only</text>']
    for r,(label,a) in enumerate(rows):
        y0=30+r*78
        s.append(f'<text x="6" y="{y0+32}" font-size="12">{label}</text>')
        for col,im in enumerate(a):
            vals=np.clip((im.reshape(16,16)+1)/2,0,1)
            for iy in range(16):
                for ix in range(16):
                    b=int(round(vals[iy,ix]*255))
                    s.append(f'<rect x="{130+col*64+ix*4}" y="{y0+iy*4}" width="4" height="4" fill="rgb({b},{b},{b})"/>')
    s.append("</svg>"); path.write_text("\n".join(s))

def svg_curve(path,runs):
    W=720;H=420; left=70;top=35;pw=600;ph=325
    colors={"A":"#2166ac","B":"#b2182b","C":"#1b7837"}
    ymin=-4.; ymax=0.5
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<text x="70" y="20" font-size="15">C13 validation MSE (log10); solid seed13, dashed seed29</text>']
    for v in range(-4,1):
        py=top+ph*(ymax-v)/(ymax-ymin)
        s.append(f'<path d="M {left} {py} H {left+pw}" stroke="#ddd"/>')
        s.append(f'<text x="12" y="{py+4}" font-size="12">1e{v}</text>')
    for step in (0,100,200,400,600):
        px=left+step/600*pw
        s.append(f'<text x="{px-10}" y="{top+ph+22}" font-size="12">{step}</text>')
    for run in runs:
        for mode,v in run["decoders"].items():
            pts=[]
            for rec in v["curve"]:
                px=left+rec["step"]/600*pw
                py=top+ph*(ymax-np.log10(max(rec["val"]["mse"],1e-8)))/(ymax-ymin)
                pts.append(f"{px:.2f},{py:.2f}")
            dash=' stroke-dasharray="6 3"' if run["seed"]==29 else ''
            s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[mode]}" stroke-width="2"{dash}/>')
    for i,(m,c) in enumerate(colors.items()):
        s.append(f'<text x="{220+i*130}" y="405" fill="{c}" font-size="14">{m}</text>')
    s.append("</svg>");path.write_text("\n".join(s))

def main():
    out=Path(__file__).resolve().parent
    t0=time.monotonic()
    grad_error=finite_difference_test(); math=examples()
    x,l,params=make_data(768,1300); xv,lv,pv=make_data(192,1301)
    assert not(set(map(tuple,params))&set(map(tuple,pv)))
    runs=[]; pictures=[]; maps={}
    for seed in CONFIG["seeds"]:
        enc,pre=train_encoder(x,l,xv,lv,seed)
        frozen=pre["snapshot_sha256"]
        z,_=fwd(enc,x); zv,_=fwd(enc,xv)
        base=init_mlp(CONFIG["decoder_dims"],np.random.default_rng(seed+2000))
        order=np.random.default_rng(seed+3000).integers(len(x),size=(600,64))
        run={"seed":seed,"encoder":pre,"decoders":{},
             "decoder_parameters":count(base),"schedule_sha256":sha(order),
             "paired_base_sha256":sha(np.concatenate([v.ravel() for v in base.values()])),
             "encoder_weight_singular_values":{f"W{i}":np.linalg.svd(enc[f"W{i}"],compute_uv=False).tolist() for i in range(3)}}
        assert pre["parameters_encoder"]+pre["parameters_aux_decoder"]+pre["parameters_classifier"]<1000000
        for mode in ("A","B","C"):
            p,mapping=mapped(base,enc,mode);maps[mode]=mapping
            run["decoders"][mode],yp=fit_decoder(p,z,x,zv,xv,order)
            pictures.append((f"seed{seed} {mode}",yp))
            print(json.dumps({"seed":seed,"mode":mode,"initial_val":run["decoders"][mode]["initial"]["val"]["mse"],
              "final_val":run["decoders"][mode]["final"]["val"]["mse"],"cumulative_train_seconds":TRAIN_SECONDS}),flush=True)
        assert sha(np.concatenate([v.ravel() for v in enc.values()]))==frozen
        run["encoder_frozen_hash_verified"]=True;runs.append(run)
    result={"task":"C13","status":"COMPLETED","scope":"synthetic NumPy toy only; no Qwen weights",
      "config":CONFIG,"data":{"train_sha256":sha(x),"val_sha256":sha(xv),
        "train_labels_sha256":sha(l),"val_labels_sha256":sha(lv),
        "parameter_overlap":0,"generation":"make_data; independent seeds; four shapes + thin attachment"},
      "environment":{"python":platform.python_version(),"numpy":np.__version__,
        "system":platform.system(),"machine":platform.machine(),"processor":platform.processor(),
        "runner_environment":os.environ.get("RUNNER_ENVIRONMENT","manual"),
        "github_run_url":os.environ.get("C13_RUN_URL"),"source_commit":os.environ.get("C13_SOURCE_SHA"),
        "threads":{k:os.environ[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        "dtype":"float64","device":"CPU"},
      "tests":{"finite_difference_max_abs_error":grad_error,
        "mathematical_assertions":"PASS","paired_schedule_and_snapshot_assertions":"PASS"},
      "math":math,"parameter_mapping":maps,"runs":runs,
      "cumulative_training_seconds":TRAIN_SECONDS,"total_script_seconds":time.monotonic()-t0,
      "limitations":["two seeds only","16x16 grayscale procedural shapes",
        "MLP encoder, no Q/K/V or LayerNorm in training network","fixed 600 decoder steps, not proof of asymptotic convergence",
        "C changes only endpoint linear maps; internal tanh and hidden layer remain learned",
        "no real Qwen experiment; no automatic follow-up training"]}
    (out/"RESULT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (out/"PARAMETER_MAPPING.json").write_text(json.dumps(maps,ensure_ascii=False,indent=2)+"\n")
    svg_montage(out/"reconstructions.svg",xv,pictures)
    svg_curve(out/"learning_curves.svg",runs)
    print("C13_COMPLETED",flush=True)

if __name__=="__main__":
    main()
