# C09：现有分层原型最容易算错的三件事

1. **每块各自 softmax 后平均，丢掉了区域的真实概率。** 正确区域权重是区域内所有细 logits 的 logsumexp 再跨区域 softmax；不等长区域尤其容易错。区域均值 key 的分数不是区域质量。
2. **用精确区域质量选 Top-k，却把它当成无需读取细 Token 的路由。** 精确 logsumexp 和精确最大分数都需要本 query 对全部细 key 打分；只能省去部分 value 读取和聚合。均值 key 可以预先缓存，但有预处理、更新和存储成本。
3. **把粗 Token 与其细 Token 直接拼接，重复代表同一内容。** 这会改变归一化分母和区域先验。互斥替换、残差细化、直接拼接是不同模型语义，不能用一个未经训练的修正混为一谈。

本文是独立的小型数学/数值参照，不审计其他会话的实现，不接触真实视觉特征、Qwen 权重、用户本机或 DCU，不声称 GPU 更快或模型质量更高。实测结果在 [RESULT.json](RESULT.json)，完整测试输出在 [TEST_REPORT.txt](TEST_REPORT.txt)。

## 正确参考实现与使用

- [reference_attention.py](reference_attention.py)：Python 标准库，IEEE754 binary64，单 query、单 head。多头/多 query 可外层循环；没有申请 N×N 矩阵。
- dense_attention：直接参考；hierarchical_attention：区域内条件概率 × 区域概率。
- selected_attention：选中区域中的细 Token 精确重归一化，拒绝重复选区。
- prepare_centroids / route_attention：实际区分在线粗路由与全部细 key 扫描。
- approximate_region_logits：**全细 logits 已给定的诊断接口**，不是零成本粗路由。
- truncation_diagnostic：读取全体 K/V 的 oracle 诊断，不计作便宜部署能力。
- 最大 1024 个细 Token；空区域允许；全部为空或 masked 的 attention 拒绝；NaN、+∞ logits 拒绝，−∞ 可作 mask。+∞ 并列的极限依赖其趋近路径，不能任意规定为正常有限输入。

从本目录运行：

~~~sh
python3 -m unittest -v test_reference_attention.py
python3 run_experiment.py
~~~

第二条会重跑测试，并生成 RESULT.json、TEST_REPORT.txt、C09_DELIVERY.zip。只用标准库，不下载依赖、模型或数据。

## 1. 精确分层归一化

令非空的有限 token 集合被互不重叠区域 I_r 完整覆盖。注意力 logits 已包含缩放、位置偏置和 mask：

$$
s_i=q^\top k_i/\sqrt{d}+b_i,\qquad
Z=\sum_i e^{s_i},\quad
p_i=e^{s_i}/Z,\quad y=\sum_i p_i v_i.
$$

每区未归一化质量、log 质量、条件概率和条件输出为：

$$
Z_r=\sum_{i\in I_r}e^{s_i},\quad L_r=\log Z_r
=\operatorname{LSE}_{i\in I_r}(s_i),
$$

$$
a_{i|r}=e^{s_i-L_r},\qquad u_r=\sum_{i\in I_r}a_{i|r}v_i.
$$

跨区域概率和最终输出：

$$
w_r=\frac{Z_r}{\sum_t Z_t}
=\operatorname{softmax}_r(L_r),\qquad
y=\sum_r w_r u_r.
$$

因为 w_r a_{i|r}=p_i，所以此式在**所有区域都计算、同一组 logits 和 values、区域互斥且完整覆盖**时与 dense 完全是同一数学对象。没有假定每区长度相同，也没有均匀区域先验。每块单独 softmax 后平均只有在真实区域质量恰好相等时才成立，或者因特殊 values 偶然得到相同输出。

空区 Z_r=0、L_r=−∞、w_r=0；本实现把其无意义的条件输出记为零，跳过区域内部 softmax。全 masked 输入没有定义的归一化分母，抛异常。

**稳定实现不直接求 exp(s)。** 先取全局共同偏移 c=max_i s_i，求区域相对 log 质量 L'_r=LSE(s_i−c)，跨区 softmax(L'_r)。区内另做稳定 softmax。即使绝对偏移 1e300 使 LSE 的绝对表示丢失 log(n)，共同偏移后的区域质量仍能保留该差异。极小概率可能 binary64 下溢到零；不把此数值参照当任意实数的无限精度证明。

## 2. 不等长区域与四种打分

对有 n_r 个有效 token 的区域，令 μ_r 是 score 均值，m_r 是最大值，δ_i=s_i−μ_r：

$$
L_r=\mu_r+\log n_r+
\log\left(\frac1{n_r}\sum_{i\in I_r}e^{\delta_i}\right).
$$

最后一项记为 J_r≥0。它是区域内部 score 离散度对 log 质量的贡献。

| 区域打分 | 成立条件 / 局限 | 在线细 key 成本 |
|---|---|---|
| 均值 key 分数 μ_r | q·mean(k)/√d=mean(score)，只对一致的线性打分成立；不能忽略异长区域的 log n 或 J_r。区域大小、J_r 都相同时可只差公共常数 | 缓存 centroid 后 R 个粗 dot，选中 n_S 个细 dot；缓存更新另算 |
| 最大分数 m_r | m_r≤L_r≤m_r+log n_r；单一 token 主导时接近精确。有相同有效贡献数时区域排序/概率可能正确；否则不成立 | N 个细 dot，才能知道精确 max |
| 精确 LSE L_r | 所有区域精确质量；所有区域计算时给 dense 等价式 | N 个细 dot，N 次 exp 的区域归约，再聚合选中 values |
| μ_r+log n_r | 区内 logits 相同则精确；小离散度时较好；漏掉 J_r，藏在大区域里的尖峰会被平均掉 | 与缓存均值 key 相同，需有效大小和兼容 mask |

max+log n_r 是上界，只有区内 logits 相同时等于 L_r；它也不是普遍精确式。

若 s_i 位于区内区间 [a_r,b_r]，则 Jensen 与 Hoeffding 引理给出：

$$
0\le J_r\le (b_r-a_r)^2/8.
$$

这里使用“在该区域 token 上均匀采样 score”的有限分布，不要求实际 attention 均匀。区域 logits 离散度小时 μ+log n 合理；仅仅“key 均值很准”不能保证离散度小。若所有 J_r 恰好相同，它们在跨区 softmax 中也会抵消；否则近似会改变区域概率。代码用固定种子的 5 组范围检查该界。

位置编码后的 key 可以在线性点积层面求均值；但“原始 key 均值再加一个粗位置编码”一般不等于“先加各细位置编码再求均值”。任意 query 相关细 mask、位置 bias、区域内位置相关变换也不能被一个固定 centroid 普遍代表。便宜路由函数明确限定无 mask、无额外 bias 的点积；数学分层函数可以接收已算好的任意合法 logits。

## 3. Top-k：保留质量、误差、成本必须同时看

选择非空有效区域集合 S，把其细 token 并集记为 T_S。最终输出用选中细 logits **重新精确归一化**，不能继续拿近似区域概率混合就称为原 attention 的条件分布。

对每行实验记录：

- dense 概率下真实保留质量 α=Σ_{i∈T_S}p_i、遗漏质量 ε=Σ_{i∉T_S}p_i；
- 输出 L2 误差、value 最大范数、2Bε 上界；
- 选中区域、选中细 Token 数，在线 K/V 读取、dot/聚合标量乘加对和 exp 数量；
- centroid 的离线读取与存储成本；计算 oracle 质量/误差的全细 K/V 诊断成本另列。

为避免 1−α 的消减误差，遗漏质量直接加总被丢概率。输出误差本身也有浮点舍入，边界断言留 2e−12 绝对容差。

设 N 个细 token、R 个非空区域、n_S 个选中细 token、key 维度 d、value 维度 d_v：

| 方法 | 在线 qk 乘加对 | 在线 value 乘加对 | 额外信息 |
|---|---:|---:|---|
| Dense | Nd | Nd_v | 全细读取 |
| 缓存均值 key 路由 | (R+n_S)d | n_S d_v | 建缓存每次 K/partition 改变读取 Nd 标量，存 Rd；不假设免费摊销 |
| 精确 max/LSE oracle | Nd | n_S d_v | 全细打分；选中 logits 复用；LSE 另做全细 exp 归约 |
| 全细 oracle 诊断 | Nd | Nd_v | **额外诊断**，不算为部署可用的廉价能力 |

route_attention 实际执行上述在线分支：centroid 分支只对所选细 key 算 dot；max/LSE 分支确实算全部细 dot。run_experiment 的 dense 诊断另执行并单列成本。费用是解析操作/读取计数，**不是耗时或加速倍数**，也不包含所有 Python 校验、排序、缓存搬运或硬件通信。

固定 k 约束下 LSE top-k 最大化保留概率质量，但不保证输出误差最小：相似 values 即使被丢掉也可能无误差。不等长区域的 top-k 不是固定 token 预算或 FLOP 预算的最优选择，比较不同策略时要同时看 n_S。全部区域选中时，精确细归一化恢复 dense；粗路由此时可能只增加开销。

## 4. 粗细共存与重复计权

若某个完全相同的 logit/value 被复制 m 次，这份内容的总未归一化质量从 exp(s) 变为 m·exp(s)。其等效 logit 增加 log(m)。这是改变先验，不是增加证据。

两份内容 logits 均为 0、values 为 +1 与 −1 时，原输出 0；只复制 +1 这一份，输出变成 1/3。若所有 token 统一复制同样次数，归一化抵消，输出不变；局部非均匀复制才造成偏置。**重复索引/重叠区域**会被 partition 拒绝；**不同索引但内容重复**不可能由索引检查自动识别，另有失败示例。

| 语义 | 正确定义的含义 | 与原细 dense 的关系 |
|---|---|---|
| 互斥区域替换 | 选中区域用细 token，未选中区域用粗代表；同一区域不同时保留粗与细分支 | 若粗分支提供 query 相关的精确 L_r 和 u_r，则可恢复原细 attention；固定 mean key/value 一般不能提供它们 |
| 残差细化 | 例如 y=y_coarse+g·Δ_S；Δ_S 表示相对粗输出的修正，需定义基线、尺度与门控 | 不是同一 softmax 归一化语义，通常需训练/校准；把细输出直接加到粗输出不自动成为残差 |
| 直接拼接 | 对粗、细 token 的合集做 softmax；同一内容可以占多个 slots | 是一个新分布，可经训练变得有用，但没有默认 dense 等价性 |

若已知每一份只是完全相同的复制品，对每份减 log(m) 能消除**该特定复制机制**的质量增加。真实粗 token 是多细 token 的非线性摘要，其 logits、values、位置、分辨率可能都不同，不能套用该修正声称通用正确。精确 query 相关的粗摘要是数学充分统计量，但得到它通常仍需读全细信息。

## 5. 有条件的筛选误差界与反例

假设：

1. 有限实 logits 形成原 dense 分布 p，且选中集合概率 α>0。
2. 每个 value 的欧氏范数 ≤B，细 values 和 logits 本身不改变。
3. y_S 严格是 p 在选中细 token 上的条件期望；没有再改区域权重、value 或位置语义。

令 ε=1−α，被丢集合的条件输出为 y_D。当 0<α<1 时：

$$
y=\alpha y_S+\epsilon y_D,\qquad
y-y_S=\epsilon(y_D-y_S).
$$

因此：

$$
\|y-y_S\|_2\le\epsilon(\|y_D\|_2+\|y_S\|_2)\le2B\epsilon.
$$

ε=0 时输出完全相同。若所有 values 两两距离至多 D，可收紧为 Dε。该界控制绝对 L2 误差，不控制相对误差，也不保证视觉/语义质量。

**尖锐反例：** 标量 values +B、−B，概率 1−ε、ε，只保留 +B；误差正好 2Bε，常数 2 不能无条件缩小。

**去掉范数假设：** 概率 [1−10⁻⁶,10⁻⁶]，values [0,10⁶]，只保留第一份；遗漏质量极小但输出误差为 1。若错误地假设 B=1，会得到失效上界 2×10⁻⁶；真正 B=10⁶ 的上界仍成立。

**去掉精确条件归一化假设：** 区域长 1 与 9，所有 logits 为 0，values 分别 +1 与 −1；全部保留所以 ε=0，但把两区均匀混合输出 0，而 dense 输出 −0.8。2Bε 不适用于这个变了权重的算法。若另有选择内近似分布 p̃，则可加一项 B||p(.|S)−p̃||₁；本实验不凭空假设该项很小。

**该界不是双向界：** 两个 values 完全相同，遗漏一半概率也可误差为 0。实验包括该例。低保留质量并不必然导致大误差，准确保留质量也不能替代 value 信息。

便宜均值路由没有普遍的 ε 保证：隐藏峰反例能让它漏掉主要质量。只有额外获得区域质量上下界、分布假设或经过验证的路由校准后，才能把路由分数转成可靠误差证书。

## 6. 固定实验协议与交接

固定 seed=9027；单 query、d=8、d_v=6；区域大小 [1,3,12,52,188,768,0]，总细 token 1024。4 个 scenario：不等长均匀 logits、Gaussian、隐藏峰、±10000 极端 logits。4 个 router × k∈{1,2,4,6}，共 64 行；预先固定，不据结果挑行。values 归一化使范数 ≤1。

测试覆盖随机不同分区、最大 1024 token、空区、全 masked、部分 masked、极端/巨大公共 logits、只选一个、全部选中、重叠与重复区域、同内容重复、错误均匀混合、朴素 exp 溢出、条件误差界紧例及缺假设反例。参见 FAILURE_EXAMPLES.md。

RESULT.json 提供真实 CPU 时间、Python/binary64 信息、实际测试数、全部 64 行、失败例数值、源文件 SHA256。TEST_REPORT.txt 是测试实际输出。脚本在写包前检查 CPU 时间小于 600 秒，云端 job 另有 10 分钟超时；运行失败即退出，不生成成功声明。

**给现有路由原型的接入建议：**

- 先把同一份 q/k/v/位置 bias/mask 小样本送进 dense 与 hierarchical，逐层检查 w_r、a_{i|r}、最终输出，而非仅比较图像。
- 比较路由时先使用 selected_attention 的精确条件输出，将“选错区域”与“选中后归一化又算错”分开。
- 成本面板分开记录在线细 key 扫描、选中 values、粗缓存更新、全细 oracle 诊断；不要把最大分数路由标成无需看细 Token。
- 明确粗细替换/残差/拼接语义后再训练和评估；本任务没有改现有系统。

本包没有真实模型、GPU 基准、训练实验或视觉质量结论。若继续推进，先用现有真实特征测选区分布与 value 范数，再看误差界是否有用。
