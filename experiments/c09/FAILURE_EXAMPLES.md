# C09 失败示例

所有失败都是预定义的数学/数值反例；测试会验证正确实现并断言错误做法的差异。

| 例子 | 错误做法 / 失效假设 | 可复现结果 |
|---|---|---|
| 区域长 1 vs 9，logits=0，区域 values=+1/−1 | 每块 softmax 后均匀平均；均值 key 分数忽略区域大小 | dense=−0.8，错误输出=0；真实区域概率 0.1/0.9，错误 0.5/0.5 |
| 一 token score=3 vs 64 token score=0 | 最大分数代表总区域质量 | max 选小区，LSE 选大区；质量取决于 exp(3) vs 64 |
| 一 token score=1 vs 大区 [8,−8×63] | mean+log n 忽略离散度 | 近似选 score=1 的小区，oracle 选含 8 的大区 |
| 同分 +1/−1 两 token，仅复制 +1 | 直接拼接粗细表示而不改变语义 | 原输出 0，拼接输出 1/3；全体统一复制两遍则仍是 0 |
| 重复区域 [[0],[0,1]] 或重复选择 [0,0] | 重叠当互斥分区 | 正确实现 ValueError；不同索引同内容另作语义检查 |
| logits=[10000,−10000,9999] | 直接 exp(logit) | 朴素 math.exp(10000) 溢出；稳定实现有限且 dense/分层一致 |
| logits=1e300，相同 score 但区域长 1/9 | 先存绝对区域 LSE 再 softmax | log(9) 被舍入吃掉，错误区域概率 0.5/0.5；共同偏移后是 0.1/0.9 |
| 全 masked / 仅选择空区 | 把 0/0 当均匀概率 | 正确实现显式拒绝；空区混在有效分区则其概率为 0 |
| ε=0.1，values=+1/−1 | 认为 2Bε 太松可无条件缩小常数 | 只留 +1 时误差=0.2，恰好达到界 |
| ε=10⁻⁶，values=0/10⁶ | 未约束真实 value 范数却使用 B=1 | 输出误差=1；假的 2e−6 界失效，实际 B=10⁶ 的界成立 |
| 全区保留，仍把区间概率改成均匀 | 将 ε=0 的条件界套到改权重的算法 | 真实误差=0.8；假设不成立 |
| ε=0.5，values 相同 | 把保留质量低推成必然大输出误差 | 输出误差=0，说明误差界不是下界 |

完整数值来自 RESULT.json 的 approximation_counterexamples、bound_tight_example、
missing_norm_assumption_counterexample、duplicate_content；逐例断言在 test_reference_attention.py。
