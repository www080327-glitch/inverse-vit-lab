"""Generate the reviewable report from frozen measured results, never invent metrics."""
import json
import hashlib
from collections import defaultdict
from runtime import ROOT, np, write_json, read_jsonl

def main():
    load=lambda p:json.loads((ROOT/p).read_text())
    metrics=load('results/metrics.json'); costs=load('results/costs.json'); large=load('results/high_resolution.json')
    budget=load('results/training_budget.json'); train=load('results/training.json')
    generic=load('results/numerical_generic.json'); numerical=load('results/numerical_model.json')
    get=lambda split,method,k=2:next(r for r in metrics['results'] if r['split']==split and r['method']==method and r['k']==k)
    timing=costs['timing']; h=timing['hierarchical_whole_query_no_pack']['median'];d=timing['direct_dense_reader_no_router_no_pack']['median']
    lines=['# 问题驱动的多分辨率视觉记忆实验结果','',
      '本原型实际分开计算全局粗索引与选中位置的细 Attention，原图另存且可精确还原。固定普通场景中，学习路由只展开 2/16 个区域就命中全部目标；粗摘要相同的极小目标场景中，它接近随机覆盖率。这个结果同时展示条件计算的可行性和粗路由的漏检边界。','',
      f'所有张量在 CPU；intraop 2、interop 1，未调用加速设备。模型 {train["parameters"]:,} 个参数，合成训练累计 {budget["total_charged_seconds"]:.6f} 秒，限制 1,200 秒。代码、核查与验收的起止时间单列在 `results/acceptance.json`。','',
      '## 两次计算怎样衔接','',
      '输入是保存好的 32×32 RGB PNG。`PixelMemory.pack` 在必要时补零，然后使用 reshape、permute 与 contiguous 将像素排列为 `[16,8,8,3]`。反向排列并裁掉 padding 可逐字节还原；全局区域边界与有效像素掩码同时保留。细坐标由区域原点与局部整数坐标精确计算，协议标出根、区域和像素三个层级、尺度和父关系。插值与均值不在无损路径里。','',
      '每个区域的 4×4 灰度均值组成 16 维有损摘要，微型 MLP 生成 4 维 key。四维任务向量与全部 16 个 key 评分，先执行全局粗 softmax 得到上下文，再选 top-k。粗上下文用于展示全局读取，不额外给答案头一条绕过细读的通道。','',
      '只 gather 选中的 raw tile，先去重再编码。细类型 MLP 从每个选中区域的 64 个灰度位置预测类别，训练只使用类型标签；这一步本身读取该区域全部像素，成本已计入。细 K 为 `[log p(type=0..3), pixel_intensity]`，细 Q 是缩放后的 `[3×one_hot(task),7]`，Dk=5。V 是每个像素局部坐标对应的 64 维 one-hot，Dv=64。信息通过 K 决定的权重形成局部空间分布，再由小型答案头读取。','',
      '公式为 `O = softmax(Q K_selected^T / sqrt(5)) V_selected`。所有选中像素共用同一个 softmax。粗 token 留在索引路径，不再并入最终细 softmax，因此没有把同一区域的粗、细表示重复算作位置。当前区域固定且不重叠；任意重叠窗口的空间去重没有实现。','',
      '程序还实现用全局最大值与分区和合并的分块 Attention 参照；它验证统一 softmax，而不是把多个局部 softmax 输出直接相加。一个两位置反例会拒绝后一种做法。全局坐标是可核对的元数据；本模型的 V 使用局部空间 one-hot，没有实现 Qwen 的 M-RoPE。','',
      '## 固定数据与监督隔离','',
      '生成器保存 592 张原始 PNG：320 张普通训练、80 张粗盲训练、96 张普通测试、96 张粗盲测试。普通场景每图有眼睛、手臂、标记、连接四个对象与更大更亮的干扰块，目标同色，区分形状才能按问题选取。四个问题是任务编号；没有处理自然语言。','',
      '正常训练目标在棋盘格的一半区域，测试在互补区域；训练四种状态的组合之和为偶数，测试为奇数。单个形态仍由相同生成规则组成，所以这是受控的布局与组合迁移，不是开放世界泛化。粗盲组中，16 个区域的 2×2 池化结果完全相同，但原像素不同；前景每个目标只有 3 像素。该组不是自然比例的真实桌面样本，而是摘要不可辨识的构造反例。','',
      '`data/inputs` 只含图片路径与问题编号，`data/annotations` 单独保存训练和评测标签。训练器只打开 train/train_blind；推理 API 只接收像素库、任务编号、预算与策略。测试验证推理没有文件读取、没有生成器依赖，并拒绝 oracle 策略。oracle 只在评测器中使用目标区域。二分类准确率在漏检时仍可能靠猜测达到约 50%，因此主指标是区域召回率。','',
      '## 选两个区域的固定测试','',
      '| 测试组 | 路由 | 目标召回率 | 答案准确率 | 平均输出相对 L2 | 漏检问题数 |',
      '|---|---|---:|---:|---:|---:|']
    labels={'uniform':'均匀','random':'随机','oracle':'oracle','learned':'学习','brightest':'最亮区域'}
    for split in ['test','test_blind']:
        for method in labels:
            r=get(split,method)
            lines.append(f'| {"普通" if split=="test" else "粗盲"} | {labels[method]} | {r["target_region_recall"]:.2%} | {r["answer_accuracy"]:.2%} | {r["mean_relative_l2"]:.6g} | {r["misses"]}/384 |')
    r=get('test','learned');blind=get('test_blind','learned')
    lines.extend(['',f'每组为 96 张图、384 个问题。k=2 时只在 128/1,024 个细位置做 Attention。普通组目标前景占整图 {r["min_target_foreground_fraction"]:.3%} 至 {r["max_target_foreground_fraction"]:.3%}，均值 {r["mean_target_foreground_fraction"]:.3%}；粗盲组目标前景占比固定 {blind["mean_target_foreground_fraction"]:.3%}。区域本身占比为 6.25%，不能与目标前景占比混用。','',
      '预算 k=1、2、4 的完整结果、命中与漏检条件下的准确率、输出误差 p95、保留的 dense Attention 权重质量都保存在 `results/metrics.json` 和逐问题 JSONL。随机和均匀选择使用每图固定种子，并对同图所有问题保持相同选择；学习路由随问题改变。oracle 是选区上限，细读取和答案头仍使用同一模型。','',
      '## 失败案例','',
      f'粗盲组学习路由在 k=2 时漏掉 {blind["misses"]}/384 个问题，目标召回率 {blind["target_region_recall"]:.2%}，漏检条件下答案准确率 {blind["answer_accuracy_on_miss"]:.2%}。同一组的 dense 和 oracle 答案准确率均为 100%，说明这些样本的细节可由已训练的小读器区分，损失主要来自摘要看不见目标位置。','',
      '![失败案例](figures/failures.png)','',
      '图中蓝框是模型实际选择，红框只在评测叠加，绝不作为推理输入。示例同时包含实际答错，以及漏检后碰巧猜对的情况。四张原图及逐项真值保留在 `results/failure_cases.json`。普通固定测试这次没有观察到学习路由失败，不能据此保证新图像不会漏检。未实现多轮补救；提高到全选可以恢复细读取覆盖，但计算相应增加。','',
      '## 数值参照','',
      '| 定义 | 细位置数 | 最大绝对误差 |','|---|---:|---:|'])
    for row in numerical:lines.append(f'| 实际模型全选与独立 row-major dense PyTorch SDPA，float32 | {row["n"]} | {row["max_abs"]:.3g} |')
    for row in generic:lines.append(f'| 随机相同 Q/K/V 全选与 dense，float64 | {row["n"]} | {row["all_regions_max_abs"]:.3g} |')
    lines.extend(['',f'随机 Q/K/V 只选 1/8 位置时，相对完整 Attention 的 L2 误差分别为 {generic[0]["partial_vs_full_relative_l2"]:.4f} 和 {generic[1]["partial_vs_full_relative_l2"]:.4f}。这是实际近似误差，不要求它等于全选结果；它只与同样位置掩码的 dense 定义相等。普通场景的部分选择误差很小，是因为本小模型的类型 key 使目标外权重几乎为零，不能把它推广成任意 Q/K/V 的无损稀疏化。','',
      '## 存储与计算账本','',
      '| 实际张量 k 等于 2 | 字节 |','|---|---:|'])
    for name,record in costs['actual_tensor_allocations'].items():lines.append(f'| {name}，{record["dtype"]} {record["shape"]} | {record["storage_bytes"]:,} |')
    lines.extend(['',f'模型参数另占 {costs["model_parameter_bytes"]:,} 字节，固定空间 basis buffer 占 {costs["model_buffer_bytes"]:,} 字节。V 在模型内先以 view 表示，但展平多个区域时真实分配的 32,768 字节已计入。该教学 V 比原图大，不能据此宣称总内存减少。表是各张量的真实 storage 字节，既不是压缩 NPZ 文件大小，也不是进程峰值 RSS；Python、allocator 和临时工作区不在表内。','',
      '原像素一直保留。k=2 的 K/V 是本次问题的临时状态，不是从全图已计算 KV 中只保留一部分；未选区域根本不执行细类型编码或细 K/V 构造。没有跨问题或跨层 KV 共享。粗摘要与 key 在最小实现中每个问题重算，索引成本没有省略。','',
      '| 矩阵计算项 | k 等于 2 | 全部细位置 |','|---|---:|---:|',
      '| 全局粗 MLP | 20,480 | 直接 dense 读器跳过 |',
      '| 粗问题评分与上下文 | 128 加 512 | 直接 dense 读器跳过 |',
      '| 细类型编码 MLP | 13,248 | 105,984 |',
      '| QK 加 AV | 17,664 | 141,312 |',
      '| 答案头 | 6,720 | 6,720 |','',
      'FLOPs 以一次乘和加算两个操作，以上仅统计矩阵乘法。RGB 与池化扫描、归一化、tanh、log_softmax、细 softmax、排序与 gather 另外发生，不属于零成本。细 Attention 的理论乘法量是 `2×Q×N×(Dk+Dv)`，这里 Q=1，不是 N×N 的自注意力。减少 KV 记录数量、减少这项计算量和整个模型加速是不同口径。','',
      '## 实际 CPU 时间','',
      '| 范围 | 中位数 微秒 | p90 微秒 |','|---|---:|---:|'])
    for name,t in timing.items():lines.append(f'| {name} | {t["median"]:.3f} | {t["p90"]:.3f} |')
    lines.extend(['',f'这是一个固定图像和问题、预热 5 次后重复 101 次的 eager CPU 测量，不含 PNG I/O。分层整条查询为 {h:.3f} 微秒；直接 dense 读器跳过粗路由为 {d:.3f} 微秒。分层路径在这个小形状上反而慢约 {h/d:.2f} 倍。另一个全选区域但仍走粗路径的计时用于定位开销，不能代替直接 dense 基线。各阶段中位数也不能相加当作一次端到端测量。','',
      '因此本实验展示少做细 Attention，不建立整体同比例加速结论，更不提供 GPU、DCU 或 MPS 性能推断。','',
      '## 1920×1080 只验证真实图像存储','',
      f'NASA Blue Marble 2002 卫星图像复合产品从 2,048×2,048 原文件中心裁取 1,920×1,080，无插值；原 JPEG、PNG 和可逆分块 NPZ 都保存。共有 {large["theoretical_only"]["fine_positions"]:,} 像素，32,400 个区域，raw RGB 占 6,220,800 字节；区域坐标占 1,036,800 字节，有效掩码占 2,073,600 字节，实际粗摘要占 2,073,600 字节。手动取 8 个区域只用于验证 gather，共 512 像素、1,536 raw 字节，没有运行高分辨率模型路由或 Attention。','',
      '账本中的全像素自注意力分数矩阵有 4,299,816,960,000 个元素，float32 约 17.20 TB。代码没有构造它。单个问题 Query 读取所有像素的分数只有 2,073,600 个，不能与上述平方矩阵混同；其理论 QK 加 AV 为 286,156,800 FLOPs，选 512 位置则为 70,656 FLOPs，另有全局索引成本。这些使用原型维度的理论值，不是生产视觉模型的实测。','',
      '## 验收与交接','',
      '9 项机制测试覆盖 pack/unpack 含不整除边界、精确摘要盲区、dense 数值与分块 softmax、局部 softmax 错误反例、懒执行与去重、实际 Q/K/V 的像素参照、训练测试隔离、超过 1,024 像素的 Attention 拒绝，以及保留但未读取细节的干预。最后一项在未选区域改变一个答案像素而保持粗摘要相同：稀疏输出逐位相同，dense 输出改变约 0.327，直接展示“仍保存”与“本次已读取”不同。交接包含两个 NPZ 示例、独立真值文件、固定全量数据、模型、结果、结构图与最小代码。完成状态和文件 SHA256 位于 `exchange/READY.json`。','',
      '对照参考的具体实现与适用范围见 [SOURCE_MAP.md](SOURCE_MAP.md)。这个完成状态只覆盖原型机制；没有将合成任务成绩写成自然语言理解、真实桌面可靠性、论文完整复现或 CSA2 实现。'])
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Aggregate each canonical task without another training/evaluation pass.
    groups=defaultdict(list)
    for row in read_jsonl(ROOT/'results/per_question.jsonl'):groups[(row['split'],row['method'],row['k'],row['question'])].append(row)
    records=[]
    for (split,method,k,q),rows in groups.items():
        records.append(dict(split=split,method=method,k=k,question=q,n=len(rows),
                            recall=float(np.mean([r['target_region_recall'] for r in rows])),
                            answer_accuracy=float(np.mean([r['correct'] for r in rows]))))
    write_json(ROOT/'results/per_task_metrics.json',records)
    print('Report generated from measured JSON; per-task aggregates saved.')

if __name__=='__main__': main()
