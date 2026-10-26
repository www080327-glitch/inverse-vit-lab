"""Package both fixed stages and write an evidence-based handoff; NO training."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import shutil
import zipfile

BASE=Path(__file__).resolve().parent
ROOT=BASE/"extended"
# API-only extension preserves the measured default behavior; checks run first.
shutil.copy2(BASE/"heads.py",ROOT/"heads.py")
shutil.copy2(BASE/"run.py",ROOT/"run.py")
def read(path): return json.loads(path.read_text())
initial=read(BASE/"RESULT.json")
result=read(ROOT/"RESULT.json")
short=read(BASE/"results/runs.json")
long=read(ROOT/"results/runs.json")
config=read(ROOT/"config.json")
assert len(short)==len(long)==24
portable=read(ROOT/"results/portability_checks.json")
assert len(portable)==4 and all(v["default_sampling_bitwise_parity"] and v["default_loss_bitwise_parity"] and v["float64_and_gradient"]=="pass" for v in portable.values())
assert config["steps"]==9000
assert all(r["status"]=="complete" and len(r["conditions"])==4 for r in short+long)
assert all(r["steps"]==1800 for r in short)
assert all(r["steps"]==9000 for r in long)
total=sum(r["training_seconds"] for r in short+long)
assert abs(total-result["training_seconds"])<1e-6 and total<1800
assert result["threads"]<=2 and result["affinity_cpu_count"]<=2
assert all(r["parameters"]<1_000_000 for r in short+long)
for r in short+long:
    for m in r["conditions"]:
        assert (m["nll_per_vector"] is None)==(r["method"] in ("mse","ear"))
for path in (BASE/"results/checks.json",ROOT/"results/checks.json"):
    c=read(path)
    assert c["diag_logprob_max_error"]<1e-9 and c["full_logprob_max_error"]<1e-9
    assert c["bounded_pair_distance_matches_dense"]=="pass"
    assert c["api_gradients_and_generator"]=="pass"
assert len(list((BASE/"results").glob("*.png")))==12
assert len(list((ROOT/"results").glob("*.png")))==12

def audit(rows):
    return [{"dimension":r["dimension"],"seed":r["seed"],"method":r["method"],
             "condition":m["condition"],"coverage":m["mode_coverage"],
             "valid_fraction":m["valid_fraction"],"mode_tv_to_test":m["mode_tv_to_test"]}
            for r in rows for m in r["conditions"]
            if m["mode_coverage"]<1 or m["valid_fraction"]<.9]
result["initial_stage"]={"source_commit":initial["source_commit"],
    "evidence_commit":"6b10c71a446e7a408957ee2cd685c25985b6800e",
    "steps":1800,"runs":24,"training_seconds":initial["training_seconds"],
    "recommendation":initial["recommendation"],"selection_scores":initial["selection_scores"]}
result["total_runs_both_stages"]=48
result["portable_interface"]={"continuous_features":"[B,condition_dim]","default_onehot_sampling_and_loss_bitwise_parity":True,"real_feature_training_tested":False}
result["final_stage_training_seconds"]=sum(r["training_seconds"] for r in long)
result["distribution_audit_threshold"]="coverage<1 OR valid_fraction<0.9; descriptive post-hoc audit, NOT a changed selection rule"
result["distribution_audit_initial"]=audit(short)
result["distribution_audit_final"]=audit(long)
result["interpretation"]="Recommendation follows the preregistered dimension-balanced energy-distance rule, not a claim that all correlation, mode, tail or real-image criteria are solved."
(ROOT/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
compare={"initial_1800":initial["summary"],"final_9000":result["summary"],
    "selection_rule_changed":False,"all_seeds_retained":True,"cumulative_training_seconds":total}
(ROOT/"results/comparison_stages.json").write_text(json.dumps(compare,indent=2)+"\n")
candidate=result["recommendation"]
names={"diag":"共享分量对角 GMM","full":"小型全协方差 GMM","ear":"EAR 式条件噪声 MLP"}
f=lambda x:f"{x:.4f}"
lines=["# C15 连续输出头：交接与中文结论","",
    f"本轮唯一建议进入下一轮的是 **{names[candidate]}（{candidate}）**。"
    "选择依据是训练前固定的、三种维度等权的独立测试能量距离；它只是下一轮实验候选，"
    "不表示已满足相关结构、局部细节或真实图像生成要求。两个 seed 也不足以证明普遍优越或统计显著。","",
    "## 已执行范围与预算","",
    "在标准公开仓库 GitHub Actions Linux CPU runner 执行；没有访问用户本机或 DCU，"
    "没有下载图像数据或模型权重。训练均设置 2 个 CPU 的 affinity、torch 2 线程及 interop 1。",
    f"初始阶段 1800 步 ×24 次；看到相关结构失真和模式间填充后，将所有方法统一重训至固定 9000 步 ×24 次。"
    f"累计实际训练 **{total:.1f} 秒**（首轮 {initial['training_seconds']:.1f}，"
    f"延长轮 {result['final_stage_training_seconds']:.1f}），低于 1800 秒。"
    "首轮全部输出、源码与失败现象均保留；没有挑最好 seed、没有按测试结果挑 checkpoint，也没有继续调预算。","",
    "每个维度 4 个离散条件，训练每条件独立 2048 个目标；测试每条件独立 1024 个目标。"
    "分布参数和两套数据种子先固定；同一条件明确包含多个合理目标，不是一个条件只配一个答案。"
    "固定 one-hot 条件编码，没有额外可训练主干。两层 MLP 宽度按输出维数调整，使参数量约 1.2 万。","",
    "## 9000 步主结果","",
    "|维度|输出头|实际参数|能量距离 seed2027 / seed2028|覆盖率均值 / 最差条件|有效比例|整体协方差相对误差|分量内协方差相对误差|",
    "|---|---|---:|---|---|---:|---:|---:|"]
for s in result["summary"]:
    lines.append(f"|{s['dimension']}|{s['method']}|{s['parameters'][0]}|"
        +" / ".join(f(v) for v in s["energy_u_per_seed"])
        +f"|{f(s['mode_coverage_mean'])} / {f(s['mode_coverage_worst'])}|"
        +f"{f(s['valid_fraction_mean'])}|{f(s['cov_relative_fro_mean'])}|{f(s['within_cov_relative_fro_mean'])}|")
lines += ["","能量距离越小越好，使用 alpha=1、除以 sqrt(d) 的无偏 U-statistic，允许有限样本负数。"
    "每条件 512 个生成样本与 512 个独立测试样本，距离块最大 128×128；"
    "没有构造大 N² 距离矩阵。真分布独立抽样基线见 results/oracle_baselines.json。","",
    "覆盖率：根据真实分量后验归类，进入该分量 99% Mahalanobis 椭球的生成样本占总样本至少 1%，才计为覆盖。"
    "因此覆盖率高也可能只有少数样本真正合理，必须结合有效比例阅读。"
    "真实潜在权重、测试分配比例、生成比例、比例 TV、每个坐标的 5/25/50/75/95 分位数和均值误差"
    "均保存在逐 seed、逐条件的 runs.json；重叠模式的后验比例不等同于真实混合权重。","",
    "## 延长训练是否改变观察","",
    "|维度|头|1800→9000 能量距离均值|1800→9000 有效比例|1800→9000 覆盖率|",
    "|---|---|---|---|---|"]
for a,b in zip(initial["summary"],result["summary"]):
    assert (a["dimension"],a["method"])==(b["dimension"],b["method"])
    lines.append(f"|{a['dimension']}|{a['method']}|{f(a['energy_u_mean'])} → {f(b['energy_u_mean'])}"
        +f"|{f(a['valid_fraction_mean'])} → {f(b['valid_fraction_mean'])}"
        +f"|{f(a['mode_coverage_mean'])} → {f(b['mode_coverage_mean'])}|")
s8={s["method"]:s for s in result["summary"] if s["dimension"]==8}
lines += ["",
    f"八维对角 GMM 有效比例为 {100*s8['diag']['valid_fraction_mean']:.1f}%，"
    f"全协方差为 {100*s8['full']['valid_fraction_mean']:.1f}%，"
    f"EAR 为 {100*s8['ear']['valid_fraction_mean']:.1f}%。"
    "这说明相近的整体均值或较小能量距离仍可能隐藏相关结构错误、模式间填充和不合理样本。"
    "请对照每个条件的两种 seed 图和比例，不能把首坐标散点或一个指标当成功标准。",
    f'候选 full 的八维有效比例，seed2027={100*s8["full"]["valid_fraction_per_seed"][0]:.1f}%，seed2028={100*s8["full"]["valid_fraction_per_seed"][1]:.1f}%；'
    "这项明显的 seed 不稳定性仍未解决。只展示较好 seed 会误导，因此两个 seed 和最差条件都保留。","",
    "MSE 的条件均值附近点输出无法表达同一条件下的多解；这不否定其在有唯一配对目标的重建任务中的价值。",
    "对角 GMM 的分量内部不表达非对角协方差，但混合均值可以贡献整体相关性；"
    "分量数增加可能近似更多相关结构，本轮未把增加分量数当作免费能力。",
    "全协方差 GMM 的真实分布恰在模型族内，是有利的相关性对照。"
    "它也可能落入差的局部解，且参数匹配时隐藏层更窄；本轮表现不能外推到高维视觉 token。",
    "EAR 的严格适当能量评分只约束总体目标最优解，不保证小 MLP、有限训练和随机优化能到达。"
    "连续 MLP 从连通噪声映射到分离模式时可以形成低密度连接，本实验保留了这种填充和漏模。","",
    "## NLL 与真实计算差异","",
    "|维度|头|宽度|参数|训练秒数（双 seed 合计）|每样本前向 dense MAC|测试 NLL/维 seed2027 / seed2028|",
    "|---|---|---:|---:|---:|---:|---|"]
for d in (1,2,8):
    for method in ("mse","diag","full","ear"):
        rr=[r for r in long if r["dimension"]==d and r["method"]==method]
        nlls=[statistics.mean(m["nll_per_dimension"] for m in r["conditions"]) for r in rr] if method in ("diag","full") else []
        lines.append(f"|{d}|{method}|{rr[0]['hidden_width']}|{rr[0]['parameters']}|"
          +f"{sum(r['training_seconds'] for r in rr):.1f}|{rr[0]['dense_macs_per_forward']}|"
          +(" / ".join(f(v) for v in nlls) if nlls else "无显式密度，null")+"|")
lines += ["",
    "各头训练 loss 不能互相排名。仅 GMM 有显式 Lebesgue 密度，因此报告向量 NLL 和每维 NLL；"
    "EAR/MSE 均为 null，不把训练能量或点质量伪装成 NLL。",
    "每次训练 batch128，9000 步，共 1,152,000 个目标抽样次数。EAR 每个目标生成两个独立样本，"
    "前向 MLP 计算约两倍；全协方差另需三角求解。MAC 只估算线性层前向，"
    "不含反向、激活、优化器、求解器。参数接近不等于等计算，实际耗时全保留。","",
    "## 数学、实现与检查","",
    "GMM 先在同一分量内组合维度，再对分量做 logsumexp；对角与全协方差实现均和 "
    "torch.distributions 对照。全协方差下三角 L 的对角使用 softplus+0.03，Sigma=LLᵀ；"
    "不用显式逆或行列式。共享分量和逐坐标混合的数值反例见 checks.json。",
    "EAR 官方公式采用 ||x−y||^alpha，0<alpha<2；本轮 alpha=1，两个独立噪声样本共享 condition 与 target，"
    "最小化两项到目标距离减样本间距离，温度固定1。没有运行完整 EAR，也没有实现官方大 AdaLN 网络。",
    "平方距离能量目标：本轮未除以2的 ED₂=2||E X−E Y||²，仅识别均值。"
    "对真实 ±1、塌缩0、错误方差±2 的精确数值例子，平方版本均为0，alpha=1分别为0、1、1。"
    "这不等同于“所有平方误差都不能用于重建”。详见 SOURCES.md，也注明了原文归一化常数的差异。",
    "公式/API、独立样本、梯度有限、采样 generator 可重现、块距离与 dense 参照均通过；"
    "仓库 tests/minimal_cpu.py 也在两个训练工作流通过。","",
    "## 失败与不确定性","",
    f"程序/数值失败：{len(result['failures'])}。生成分布失败另记："
    f"首轮有 {len(audit(short))} 个条件、延长轮有 {len(audit(long))} 个条件触发"
    "“覆盖率不足1或有效比例低于90%”的描述性审计标记。90%只是报告阈值，不是事后更改选型规则。"
    "全部明细保存在 RESULT.json，不用空的程序异常列表暗示没有模式失真。",
    "两次 seed、Gaussian-mixture 真分布、固定4条件和最多8维不足以验证通用生成能力。"
    "本轮没有自回归反馈，也无法确定历史 P1/P2 自由生成失败具体由哪项机制导致。","",
    "## 唯一候选的下一轮与仍缺证据","",
    f"只把 **{names[candidate]}** 带入下一轮真实冻结视觉特征的小实验。"
    "先测真实条件下多解、分量内相关与尾部偏差，再观察固定解码器的图像和小目标细节；"
    "不要直接铺开大模型训练。选择分数和全部对照都保留，其他头仍是本轮对照，不另推荐第二个候选。",
    "尚缺：真实特征的维数与分布适配、分量数与参数/计算前沿、非高斯条件多解、更多 seed、"
    "等时间预算、长序列自由生成误差积累，以及解码后的细节/画质证据。"
    "合成分布成绩不能当图像画质；教师前缀 NLL 也不代表自由生成可用。","",
    "## 可移植接口与交付","",
    "Head.sample(condition,n,generator) → [B,n,d]；scalar condition保留B=1。"
    "支持整数ID与 [B,condition_dim] 连续条件向量；Head(...,condition_dim=h) 可接外部 h 维特征。"
    "数值实验使用默认4维 one-hot；新增向量接口与默认 ID 版本做了采样和 loss 的逐位一致性检查，"
    "另验了16维连续条件、float64、生成形状和梯度有限。接口验证不等于已训练真实视觉条件。",
    "依赖与复现方式见 README.md，默认拒绝有训练记录的静默重跑。"
    "逐 seed/条件 metrics.csv、runs.json、训练 loss/梯度轨迹、所有散点/所有坐标边际图均交付。"
    "图范围固定，指标用未截断样本。未提交原始数据、模型权重或优化器状态。",
    "C15_DELIVERY.zip 同时包含 9000 步完整结果与 initial_1800/ 原始阶段；"
    "MANIFEST.json 覆盖两阶段每个包内文件，PACKAGING_CHECKS.json 记录数量、预算和完整性检查。"
]
(ROOT/"HANDOFF_ZH.md").write_text("\n".join(lines)+"\n")
(ROOT/"README.md").write_text(
    "# C15 final 9000-step delivery\n\n"
    "Read [HANDOFF_ZH.md](HANDOFF_ZH.md), [RESULT.json](RESULT.json), and [SOURCES.md](SOURCES.md).\n"
    "[C15_DELIVERY.zip](C15_DELIVERY.zip) includes both the 9000-step study and the complete initial_1800/ stage.\n\n"
    "From a clean checkout in this directory, install CPU torch==2.5.1 and requirements.txt, then run python run.py. "
    "Existing results/budget.json blocks silent retraining. For a new repeat, preserve these outputs and explicitly select a fresh C15_OUTPUT_DIR. "
    "The default config carries forward the initial-stage training cost when enforcing the 1800-second cumulative limit.\n\n"
    "The source repository base directory also contains config_extended.json and finalize.py for hosted execution/packaging. "
    "All code/data definitions are included; no trained weights or raw datasets are included.\n\n"
    "Portable interface: Head.sample(condition,n,generator) returns [B,n,d]. "
    "Accepts integer IDs or [B,condition_dim] continuous features. Use Head(...,condition_dim=h) for h-dimensional upstream features. "
    "Only the 4-dimensional one-hot conditions were trained in this study; vector interface checks do not validate real visual generation.\n")
checks={"training_runs":48,"initial_steps":1800,"final_steps":9000,
    "condition_metric_rows":192,"plot_files":24,"cumulative_training_seconds":total,
    "training_budget_seconds":1800,"cpu_affinity":result["affinity_cpu_count"],
    "threads":result["threads"],"all_formula_checks":"pass","all_run_statuses":"complete",
    "continuous_condition_interface_and_default_parity":"pass",
    "nll_only_for_explicit_density":"pass","raw_datasets_committed":False,
    "model_weights_committed":False,"initial_archive_crc":"pass"}
(ROOT/"PACKAGING_CHECKS.json").write_text(json.dumps(checks,indent=2)+"\n")
payload={}
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and "__pycache__" not in str(path) and path.suffix!=".zip" and path.name!="MANIFEST.json":
        payload[str(path.relative_to(ROOT))]=path.read_bytes()
# Include the exact original source and all figures, rather than replacing failed distributions.
with zipfile.ZipFile(BASE/"C15_DELIVERY.zip") as old:
    assert old.testzip() is None
    for name in old.namelist():
        if not name.endswith("/"):
            payload["initial_1800/"+name.removeprefix("C15/")]=old.read(name)
payload["packaging/finalize.py"]=Path(__file__).read_bytes()
payload["packaging/portability_checks.py"]=(BASE/"portability_checks.py").read_bytes()
manifest={name:hashlib.sha256(content).hexdigest() for name,content in sorted(payload.items())}
(ROOT/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
with zipfile.ZipFile(ROOT/"C15_DELIVERY.zip","w",zipfile.ZIP_DEFLATED) as z:
    for name,content in sorted(payload.items()):
        z.writestr("C15/"+name,content)
    z.writestr("C15/MANIFEST.json",(ROOT/"MANIFEST.json").read_bytes())
with zipfile.ZipFile(ROOT/"C15_DELIVERY.zip") as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(manifest)+1
    for name,digest in manifest.items():
        assert hashlib.sha256(z.read("C15/"+name)).hexdigest()==digest
print(json.dumps({"packaging":"pass","files":len(manifest),"bytes":(ROOT/"C15_DELIVERY.zip").stat().st_size,
                  "training_seconds":total,"recommendation":candidate}))
