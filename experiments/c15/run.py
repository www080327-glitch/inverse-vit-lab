"""Fixed-budget C15 experiment. Run from this directory."""
import os
# Enforce limits BEFORE loading numerical libraries.
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"
if hasattr(os,"sched_getaffinity"):
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(allowed[:2]))
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
import zipfile
import numpy as np
import scipy
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import dataset,specification,draw
from heads import Head
from metrics import evaluate,energy_distance
from checks import checks

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"
OUT.mkdir(exist_ok=True)
CONFIG = json.loads((ROOT/"config.json").read_text())
TRAIN_LEDGER = OUT/"budget.json"

def save(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")

def budget():
    return json.loads(TRAIN_LEDGER.read_text()) if TRAIN_LEDGER.exists() else {"training_seconds":0,"runs":[]}

def append_budget(run,seconds,steps):
    ledger = budget()
    ledger["training_seconds"] += seconds
    ledger["runs"].append({"run":run,"seconds":seconds,"steps":steps})
    save("budget.json",ledger)

def train(method,d,seed,train_c,train_x):
    torch.manual_seed(seed)
    g_batch = torch.Generator().manual_seed(seed+100)
    g_noise = torch.Generator().manual_seed(seed+200)
    k = 2 if d == 1 else 4
    model = Head(method,d,k,CONFIG["parameter_target"],CONFIG["noise_dim"],CONFIG["scale_floor"])
    count = sum(p.numel() for p in model.parameters())
    assert count<1_000_000
    optimizer = torch.optim.Adam(model.parameters(),lr=CONFIG["lr"])
    previous = budget()["training_seconds"]
    trace,status,completed = [],"complete",0
    tick = time.perf_counter()
    caught = None
    try:
        for step in range(CONFIG["steps"]):
            if time.perf_counter()-tick+previous >= CONFIG["train_budget_seconds"]-1:
                status = "budget_exhausted";break
            ix = torch.randint(len(train_c),(CONFIG["batch_size"],),generator=g_batch)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(train_c[ix],train_x[ix],g_noise)
            if not torch.isfinite(loss):
                status = "nonfinite_loss";break
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
            completed = step+1
            if step%100 == 0 or completed == CONFIG["steps"]:
                trace.append({"step":completed,"loss":loss.item(),"grad_norm":float(norm)})
    except Exception:
        caught = traceback.format_exc()
        status = "exception"
    seconds = time.perf_counter()-tick
    append_budget(f"{d}d/{seed}/{method}",seconds,completed)
    model.eval()
    # Dense layer MACs only, excludes activations, Cholesky solves, optimizer.
    macs = sum(p.in_features*p.out_features for p in model.net if isinstance(p,torch.nn.Linear))
    row = {"dimension":d,"seed":seed,"method":method,"status":status,
        "parameters":count,"hidden_width":model.width,"steps":completed,
        "batch_size":CONFIG["batch_size"],"target_examples":completed*CONFIG["batch_size"],
        "generated_examples_train":2*completed*CONFIG["batch_size"] if method=="ear" else 0,
        "training_seconds":seconds,"dense_macs_per_forward":macs,
        "dense_macs_training_forward_only":macs*completed*CONFIG["batch_size"]*(2 if method=="ear" else 1),
        "trace":trace,"exception":caught}
    print(json.dumps({k:v for k,v in row.items() if k not in ("trace","exception")}),flush=True)
    return model,row

def charts(d,seed,generated,ys):
    # All conditions, both seeds; no sample selection; fixed shared axes by dimension.
    lo,hi = (-5,5) if d<=2 else (-4,4)
    methods = ["truth"]+CONFIG["methods"]
    fig,axes = plt.subplots(4,5,figsize=(15,10),squeeze=False)
    for c in range(4):
        for j,method in enumerate(methods):
            ax = axes[c,j]
            x = ys[c] if method == "truth" else generated.get(method,{}).get(c)
            ax.set_title(f"c={c} {method}",fontsize=9)
            if x is None:
                ax.text(.5,.5,"FAILED / MISSING",ha="center",transform=ax.transAxes)
                continue
            if d == 1:
                ax.hist(x[:,0].numpy(),bins=np.linspace(lo,hi,61),density=True,color="#267baf")
            else:
                ax.scatter(x[:,0],x[:,1],s=3,alpha=.35)
                ax.set_ylim(lo,hi)
            ax.set_xlim(lo,hi);ax.grid(alpha=.15)
    fig.suptitle(f"{d}D seed={seed}; first coordinates only; all 512 samples",fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/f"scatter_{d}d_seed{seed}.png",dpi=125)
    plt.close(fig)
    coords = list(range(d))
    fig,axes = plt.subplots(4,d,figsize=(max(5,d*2.3),10),squeeze=False)
    bins = np.linspace(lo,hi,51)
    for c in range(4):
        for p in coords:
            ax = axes[c,p]
            ax.hist(ys[c][:,p].numpy(),bins=bins,density=True,histtype="step",color="black",label="truth",linewidth=2)
            for method in CONFIG["methods"]:
                x = generated.get(method,{}).get(c)
                if x is not None:
                    ax.hist(x[:,p].numpy(),bins=bins,density=True,histtype="step",label=method)
            ax.set_title(f"c={c} coordinate={p}",fontsize=8)
            ax.set_xlim(lo,hi)
    axes[0,0].legend(fontsize=6)
    fig.suptitle(f"All marginals, {d}D seed={seed}; display clipped, metrics untruncated")
    fig.tight_layout()
    fig.savefig(OUT/f"marginals_{d}d_seed{seed}.png",dpi=125)
    plt.close(fig)

def summarize(rows):
    summaries=[]
    for d in CONFIG["dimensions"]:
        for method in CONFIG["methods"]:
            runs = [r for r in rows if r["dimension"]==d and r["method"]==method]
            conds=[m for r in runs for m in r.get("conditions",[])]
            entry={"dimension":d,"method":method,"run_statuses":[r["status"] for r in runs],
                "parameters":[r["parameters"] for r in runs],
                "training_seconds":sum(r["training_seconds"] for r in runs)}
            for key in ("energy_u","mode_coverage","mode_tv_to_test","valid_fraction",
                        "mean_rmse","cov_relative_fro","within_cov_relative_fro","quantile_mae"):
                values=[m[key] for m in conds]
                entry[key+"_mean"]=statistics.mean(values) if values else None
                entry[key+"_worst"]= (min(values) if key in ("mode_coverage","valid_fraction") else max(values)) if values else None
                perseed=[statistics.mean(m[key] for m in r["conditions"])
                         for r in runs if r.get("conditions")]
                entry[key+"_per_seed"]=perseed
            summaries.append(entry)
    return summaries

def main():
    all_start=time.perf_counter()
    save("checks.json",checks())
    if budget()["runs"]:
        raise RuntimeError("Existing training ledger: refusing silent rerun. Keep this evidence and use an explicitly new run directory.")
    rows,baselines,distribution_specs=[],[],{}
    for d in CONFIG["dimensions"]:
        train_c,train_x,train_labels = dataset(d,CONFIG["train_per_condition"],CONFIG["dataset_seed_train"]+d)
        test_c,test_x,test_labels = dataset(d,CONFIG["test_per_condition"],CONFIG["dataset_seed_test"]+d)
        distribution_specs[str(d)]={}
        for c in range(4):
            w,m,L=specification(d,c)
            distribution_specs[str(d)][str(c)]={"weights":w.tolist(),"means":m.tolist(),"scale_tril":L.tolist(),
                "train_mode_counts":torch.bincount(train_labels[train_c==c],minlength=len(w)).tolist(),
                "test_mode_counts":torch.bincount(test_labels[test_c==c],minlength=len(w)).tolist()}
        for seed in CONFIG["seeds"]:
            generated={}
            ys={c:test_x[test_c==c][:CONFIG["eval_samples"]] for c in range(4)}
            for c in range(4):
                independent,_=draw(d,c,CONFIG["eval_samples"],torch.Generator().manual_seed(seed+40000+d*100+c))
                baselines.append({"dimension":d,"seed":seed,"condition":c,
                    "energy_u":energy_distance(independent,ys[c],CONFIG["pair_block"])})
            for method in CONFIG["methods"]:
                model,row = train(method,d,seed,train_c,train_x)
                row["conditions"]=[]
                generated[method]={}
                try:
                    if row["status"] != "complete":
                        raise RuntimeError("Training incomplete; do not score as successful")
                    with torch.no_grad():
                        for c in range(4):
                            x = model.sample(c,CONFIG["eval_samples"],
                                torch.Generator().manual_seed(seed+50000+c))[0]
                            if not torch.isfinite(x).all():
                                raise FloatingPointError("nonfinite samples")
                            generated[method][c]=x
                            m=evaluate(x,ys[c],d,c,CONFIG["pair_block"]);m["condition"]=c
                            if method in ("diag","full"):
                                tx=test_x[test_c==c]
                                m["nll_per_vector"]=-model.log_prob(torch.full((len(tx),),c),tx).mean().item()
                                m["nll_per_dimension"]=m["nll_per_vector"]/d
                            else:
                                m["nll_per_vector"]=None;m["nll_per_dimension"]=None
                            row["conditions"].append(m)
                except Exception:
                    row["evaluation_exception"]=traceback.format_exc()
                    if row["status"]=="complete":row["status"]="evaluation_failed"
                rows.append(row)
                save("runs.json",rows)
                del model
            charts(d,seed,generated,ys)
    summary=summarize(rows)
    save("distribution_specs.json",distribution_specs)
    save("oracle_baselines.json",baselines)
    save("summary.json",summary)
    flat=[]
    for r in rows:
        for m in r["conditions"]:
            flat.append({k:v for k,v in {**{k:r[k] for k in ("dimension","seed","method","parameters","training_seconds","status")},**m}.items()
                         if not isinstance(v,(list,dict))})
    if flat:
        with (OUT/"metrics.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    scores={}
    for method in ("diag","full","ear"):
        ss=[s for s in summary if s["method"]==method]
        if len(ss)==3 and all(r["status"]=="complete" for r in rows if r["method"]==method):
            scores[method]=statistics.mean(s["energy_u_mean"] for s in ss)
    recommendation=min(scores,key=scores.get) if scores else None
    result={"task":"C15","status":"complete" if all(r["status"]=="complete" for r in rows) and len(rows)==24 else "partial",
        "recommendation":recommendation,"selection_scores":scores,
        "scope":"synthetic conditional vector distributions only; no autoregressive rollout or real visual features",
        "seeds":CONFIG["seeds"],"runs":len(rows),"training_seconds":budget()["training_seconds"],
        "total_experiment_wall_seconds":time.perf_counter()-all_start,
        "threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
        "affinity_cpu_count":len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
        "scipy":scipy.__version__,"matplotlib":matplotlib.__version__,
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "config_sha256":hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest(),
        "failures":[{k:r[k] for k in ("dimension","seed","method","status")} for r in rows if r["status"]!="complete"],
        "summary":summary,
        "missing_evidence":["real frozen visual feature distribution","sequential free-running accumulation",
            "frozen decoder image quality and small-ROI detail","higher-dimensional covariance scaling",
            "equal-wall-time and equal-sample compute frontier","more seeds and non-Gaussian truth families"]}
    (ROOT/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    report(result,rows,baselines)
    manifest={}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in str(path) and path.suffix != ".zip" and path.name != "MANIFEST.json":
            manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(ROOT/"C15_DELIVERY.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name in manifest:
            z.write(ROOT/name,"C15/"+name)
        z.write(ROOT/"MANIFEST.json","C15/MANIFEST.json")
    print("RESULT "+json.dumps({"status":result["status"],"candidate":recommendation,"scores":scores,
                                "training_seconds":result["training_seconds"]}),flush=True)

def report(result,rows,baselines):
    lines=["# C15 连续输出头交接","",
        "本轮只比较已知合成向量分布，不代表图像画质，也没有验证 Qwen 视觉特征或自回归自由生成。",
        "训练与测试为独立随机样本；四个离散条件在训练中各出现 2048 个目标。测试分布在训练前固定。",
        "全部 24 次训练均采用最后一步，未用测试指标选 checkpoint；所有 seed、条件、失败和曲线均保留。","",
        "## 实测结果","",
        "|维度|方法|参数量|能量距离 seed 2027 / 2028|覆盖率均值 / 最差条件|有效样本比例|协方差相对误差|训练秒数|",
        "|---|---|---:|---|---|---:|---:|---:|"]
    for s in result["summary"]:
        fmt=lambda x:"NA" if x is None else f"{x:.4f}"
        lines.append(f'|{s["dimension"]}|{s["method"]}|{s["parameters"][0]}|'
            + " / ".join(fmt(x) for x in s["energy_u_per_seed"])
            +f'|{fmt(s["mode_coverage_mean"])} / {fmt(s["mode_coverage_worst"])}|{fmt(s["valid_fraction_mean"])}'
            +f'|{fmt(s["cov_relative_fro_mean"])}|{s["training_seconds"]:.1f}|')
    lines += ["","能量距离使用独立测试样本与生成样本的无偏 U 统计量，按 sqrt(d) 归一化，允许有限样本负数。"
        "每条件各 512 个样本，距离矩阵最大 128×128。真分布对独立真分布的有限样本参考在 oracle_baselines.json。"
        "各维度不可直接视为相同难度；选择规则对三种维度等权。","",
        "覆盖：按真实分量后验归类，且进入该分量 99% Mahalanobis 椭球的样本占总数至少 1%。"
        "宽高斯可能命中每个模式但产生很多无效点，因此同时报告有效比例、比例 TV、分量内协方差和分位数。"
        "模式重叠时，后验分配比例不完全等于潜在混合权重，二者与测试分配基线均保留。","",
        "MSE 输出重复的条件均值；它的失败是分布建模能力限制，不是重建任务不能使用平方误差。"
        "对角 GMM 可以用额外分量近似相关性，但本轮分量数固定为真实模式数；全协方差 GMM 的真分布就在模型族内，是有利的相关性对照。",
        "EAR 的连续 MLP 可在模式之间产生连接样本；能量评分的严格适当性不保证有限网络、有限训练找到全局最优。","",
        "## 唯一下一轮候选","",
        f'按预先固定的分布指标规则，本轮只推荐 **{result["recommendation"]}** 进入真实冻结视觉特征的小规模实验。'
        "这是一项待验证的实验候选，不是一般性优劣结论。","",
        "仍缺：真实特征与更高维度、非高斯分布、更多 seed、等时间计算对照、长序列自由生成、固定镜像解码器下的图像与小目标细节评估。"
        "下一轮应冻结编码器与解码器、保留真实多目标条件、统一生成预算，分别评估特征分布和解码图像。","",
        "## 规模与计算","",
        f'累计训练 {result["training_seconds"]:.1f} 秒；训练线程 {result["threads"]}，CPU affinity {result["affinity_cpu_count"]}。'
        "每模型约 12000 参数，宽度根据实际输出维数调整；固定 one-hot 条件编码，无额外可训练主干。"
        "规模相近但结构和计算并不相同，因此不能宣称普遍更强。",
        "每模型同样 1800 步、batch 128；EAR 每目标生成两份样本，MLP 前向计算约两倍；全协方差还需三角求解。"
        "逐模型真实秒数、参数、宽度、前向 dense MAC 估计和训练目标样本次数在 runs.json；MAC 不含反向、激活、优化器及求解器。","",
        "## 文件与复现","",
        "从本目录运行 python run.py；依赖 requirements.txt。既有训练预算文件会阻止静默重跑，"
        "要复现请使用新的 checkout 或明确保留旧证据后另建输出目录。"
        "统一接口 Head.sample(condition, n, generator) 返回 [B,n,d]，scalar condition 保留 B=1。"
        "EAR/MSE 的 NLL 为 null；MSE 是点质量，EAR 是隐式采样分布，均不伪造 Lebesgue 密度。",
        "结果目录含逐 seed/条件指标、每次 loss/梯度记录、全部散点与全部边际图。图示范围固定，指标使用未截断样本。",
        "不提交模型权重、优化器或原始数据；通过固定种子和分布参数重现。参见 SOURCES.md 的公式与实现核对、平方距离反例。",
        "MANIFEST.json 校验包内文件；C15_DELIVERY.zip 为完整交付包。","",
        "## 失败记录","",
        json.dumps(result["failures"],ensure_ascii=False),
        "所有种子的模式丢失、覆盖、有效比例和尾部偏差都必须结合表格阅读，不能只看散点。"]
    (ROOT/"HANDOFF_ZH.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
