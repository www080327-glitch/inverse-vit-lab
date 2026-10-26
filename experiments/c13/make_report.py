"""Build Chinese handoff from measured RESULT.json, never fabricate toy outcomes."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.loads((p/"RESULT.json").read_text())
assert r["status"]=="COMPLETED"
lines=["# C13：反向权重初始化的实测交接","",
"本报告来自一次真实 GitHub Actions CPU 执行；只涉及自造图形与小型 MLP。没有加载 Qwen、访问用户 Mac/DCU，也没有修改主训练。",""]
lines+=["## 实测结果","",
"| seed | 方案 | 初始验证 MSE | 最终训练 MSE | 最终验证 MSE | 最终验证边缘 MSE | 验证MSE曲线均值 | 首次记录≤0.02步 |",
"|---|---|---:|---:|---:|---:|---:|---:|"]
for run in r["runs"]:
    for mode,d in run["decoders"].items():
        lines.append(f'| {run["seed"]} | {mode} | {d["initial"]["val"]["mse"]:.6f} | {d["final"]["train"]["mse"]:.6f} | {d["final"]["val"]["mse"]:.6f} | {d["final"]["val"]["edge_mse"]:.6f} | {d["val_mse_auc"]:.6f} | {d["first_recorded_step_val_mse_le_002"]} |')
lines+=["","A=随机；B=三个线性层倒序转置；C=仅两端伪逆，中间层与A相同随机初始化。固定终点600步，不按验证集选择模型。输出未裁剪，MSE按原图[0,1]尺度。None/null表示在记录点没有达到阈值。","",
"## 中文结论",""]
summary={}
for mode in ("B","C"):
    ratios=[]; auc_ratios=[]; starts=[]
    for run in r["runs"]:
        a=run["decoders"]["A"]; b=run["decoders"][mode]
        ratios.append(b["final"]["val"]["mse"]/a["final"]["val"]["mse"])
        auc_ratios.append(b["val_mse_auc"]/a["val_mse_auc"])
        starts.append(b["initial"]["val"]["mse"]/a["initial"]["val"]["mse"])
    summary[mode]={"final_val_ratio_to_A_by_seed":ratios,"val_auc_ratio_to_A_by_seed":auc_ratios,
                   "initial_val_ratio_to_A_by_seed":starts}
    if all(q<1 for q in ratios):
        judgement="两个seed的固定末步验证MSE均低于A，构成值得复验的局部证据。"
    elif all(q>1 for q in ratios):
        judgement="两个seed的固定末步验证MSE均高于A，本次固定预算下没有最终优势。"
    else:
        judgement="固定末步结果随seed变化，本次没有一致的最终优势。"
    lines.append(f'- {mode}：{judgement} seed13/29末步验证MSE相对A分别为 {ratios[0]:.3f}× / {ratios[1]:.3f}×，曲线均值相对A为 {auc_ratios[0]:.3f}× / {auc_ratios[1]:.3f}×。初始比率为 {starts[0]:.3f}× / {starts[1]:.3f}×；初始好坏不能代替训练结论。')
r["paired_summary"]=summary
lines+=["","两seed是探索性证据，不做显著性宣称。固定600步不是渐近收敛证明；相同学习率是在相同方案预算下比较，不能证明每种初始化经过单独调参的最优性能。C只研究端点线性伪逆，保留随机中间层且有tanh，不能由C失败推出一般伪逆无用，也不能由B成功推出转置是逆。","",
"## 数学与数值检查",""]
m=r["math"]
lines+= [
f'- 一般满秩方阵：转置恢复 L2误差 {m["full_rank_square"]["transpose_error"]:.6g}；伪逆误差 {m["full_rank_square"]["pinverse_error"]:.3g}；正交正例转置误差 {m["full_rank_square"]["orthogonal_transpose_error"]:.3g}。',
f'- 已知二维子空间：环境伪逆误差 {m["known_subspace"]["ambient_pinverse_error"]:.6g}，已知子空间恢复误差 {m["known_subspace"]["known_subspace_recovery_error"]:.3g}。降维碰撞已列出，同时一维图像族经单测量恢复误差为 {m["dimension_reduction"]["one_dimensional_image_family_recovery_error"]:.3g}。',
f'- LayerNorm：平移碰撞最大差 {m["layer_norm"]["shift_collision_max_error"]:.3g}；提供额外均值/方差后恢复误差 {m["layer_norm"]["restore_with_mean_variance_sidechannel_error"]:.3g}。均值/方差明确算旁路。',
f'- 输入相关注意力：使用原P转置恢复误差 {m["attention"]["transpose_reconstruction_error"]:.6g}；另外提供原P后求伪逆恢复误差 {m["attention"]["pinverse_error_if_original_P_is_supplied"]:.3g}。原P是旁路；Q=K=0的不同输入碰撞另列。没有交换Q/K/V。',
"","完整推导与适用条件见 [MATH_ZH.md](MATH_ZH.md)，具体矩阵/输入/输出在 [RESULT.json](RESULT.json)。","",
"## 真实Qwen上值得复验的内容","",
"- 对完整镜像塔另建可随时停止的短程消融，使用同一冻结真实特征、相同解码器、配对seed/样本/优化器，比较随机初始化、形状兼容的线性层转置、仅线性边界的伪逆或训练集最小二乘。只建议研究设计；本任务不启动或安排该训练。",
"- 先检查每个真实张量的轴语义、投影方向、patch合并顺序和条件数；偏置、归一化和激活分别处理。Q/K/V应按真实计算图独立定义，形状兼容不是语义可逆的证据。",
"- 如果端点伪逆条件数很差，预注册截断SVD或ridge参数，再在相同预算下比较。最小二乘只可用训练划分特征/目标拟合；本实验C使用权重伪逆，没有使用验证校准。",
"- 只有在初始误差、曲线和独立验证末步都具有稳定收益时，才值得扩展；须同时评估真实细节、分布偏移和语言预测潜变量的误差。不要把本实验比率写成Qwen真实权重结论。","",
"## 已被小例子否定的具体说法","",
"- 任意满秩矩阵的转置就是逆。",
"- 普通反向传播梯度直接返回原输入。",
"- 不提供均值等额外信息，就能精确恢复所有LayerNorm输入。",
"- 仅凭维度减少就断言受限图像分布不可能近似重建。",
"- 在注意力输出上重算注意力，或把P转置，就自动成为逆Attention。",
"","这些是具体命题的反例；没有否定可训练结构镜像、数据先验恢复或有效初始化的可能性。","",
"## 执行与验证",""]
lines.append(f'- 累计训练 {r["cumulative_training_seconds"]:.2f} 秒；脚本总耗时 {r["total_script_seconds"]:.2f} 秒；CPU线程配置 {r["environment"]["threads"]}。')
for run in r["runs"]:
    e=run["encoder"]
    lines.append(f'- seed{run["seed"]}：编码器 {e["parameters_encoder"]} 参数，正式解码器 {run["decoder_parameters"]} 参数，预训练总参数 {e["parameters_encoder"]+e["parameters_aux_decoder"]+e["parameters_classifier"]}；编码器辅助验证MSE {e["val"]["mse"]:.6f}，分类准确率 {e["val"]["class_accuracy"]:.3f}。冻结哈希校验通过。')
lines+= [
f'- 有限差分反向传播最大绝对误差 {r["tests"]["finite_difference_max_abs_error"]:.3g}；数学断言、数据划分、参数量、冻结哈希、配对样本顺序检查通过。',
"- 主项目代码没有修改。现有仓库的PyTorch CPU smoke未运行：本实验环境没有安装/下载大型PyTorch，使用独立NumPy有限差分和端到端运行检查。",
"- 不提交真实/合成原始数据数组、模型权重、优化器状态或私人日志。固定种子和源码可再生成；验证图谱只展示事先固定的前8例，评分使用全部192例。",
f'- 执行来源：[GitHub Actions作业]({r["environment"]["github_run_url"]})；源commit：{r["environment"]["source_commit"]}。',
"","## 文件","",
"- [参数映射与预注册](PLAN_ZH.md)：全部参数如何复制，哪些保持随机。",
"- [完整结果](RESULT.json)：所有记录点、种子、统计、数据哈希、参数映射、数值例子。",
"- [学习曲线](learning_curves.svg)；[固定验证图谱](reconstructions.svg)。",
"- [experiment.py](experiment.py)、[math_examples.py](math_examples.py)：完整实现。",
"- [C13_DELIVERY.zip](C13_DELIVERY.zip)：交接、源码、结果、图谱和SHA256清单。",
"","仅C13实验；未修改或安排其他任务。"]
(p/"HANDOFF_ZH.md").write_text("\n".join(lines)+"\n")
(p/"RESULT.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
