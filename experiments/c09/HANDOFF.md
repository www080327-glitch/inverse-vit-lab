# C09 交接：现有分层原型最容易算错的三件事

1. 区域内 softmax 后平均会丢失真实区域质量；正确跨区权重来自 logsumexp。
2. 精确 logsumexp / 最大分数需要读取全部细 key，不能当作免费粗路由。
3. 粗细直接拼接会重复计权；替换、残差、拼接必须先定义模型语义。

## 实测与入口

- 22 项测试中实际运行 22 项，失败/错误均为 0。
- 64 组固定 Top-k 对照；最多 1024 个细 Token；seed=9027。
- Linux Python 3.12.3，binary64；实验墙钟 0.871705 秒，进程 CPU 0.871629 秒（不含打包和解释器启动）。
- 四个固定场景 dense/精确分层最大绝对输出差：1.1102230246251565e-16。
- 推导、条件、接口与接入建议：[REPORT.md](REPORT.md)。
- 实现：[reference_attention.py](reference_attention.py)。
- 失败反例：[FAILURE_EXAMPLES.md](FAILURE_EXAMPLES.md)。
- 所有 64 行、质量与成本：[RESULT.json](RESULT.json)。
- 实际测试输出：[TEST_REPORT.txt](TEST_REPORT.txt)。
- 完整包：[C09_DELIVERY.zip](C09_DELIVERY.zip)。

## Top-1 实测摘录（固定场景全部列出）

| 场景 | 区域打分 | 保留真实质量 | L2 输出误差 | 选中细 Token | 在线 qk+av 乘加对 |
|---|---|---:|---:|---:|---:|
| uniform_unequal | mean_key | 0.0009765625 | 1.0010721 | 1 | 62 |
| uniform_unequal | max | 0.0009765625 | 1.0010721 | 1 | 8198 |
| uniform_unequal | logsumexp | 0.75 | 0.0169989219 | 768 | 12800 |
| uniform_unequal | mean_log_size | 0.75 | 0.0169989219 | 768 | 10800 |
| gaussian | mean_key | 0.776397571 | 0.0463705267 | 768 | 10800 |
| gaussian | max | 0.776397571 | 0.0463705267 | 768 | 12800 |
| gaussian | logsumexp | 0.776397571 | 0.0463705267 | 768 | 12800 |
| gaussian | mean_log_size | 0.776397571 | 0.0463705267 | 768 | 10800 |
| hidden_peak | mean_key | 0.000910766719 | 1.27067711 | 1 | 62 |
| hidden_peak | max | 0.999011324 | 0.00120463201 | 768 | 12800 |
| hidden_peak | logsumexp | 0.999011324 | 0.00120463201 | 768 | 12800 |
| hidden_peak | mean_log_size | 0.000910766719 | 1.27067711 | 1 | 62 |
| extreme_logits | mean_key | 0.731058579 | 0.403963108 | 1 | 62 |
| extreme_logits | max | 0.731058579 | 0.403963108 | 1 | 8198 |
| extreme_logits | logsumexp | 0.731058579 | 0.403963108 | 1 | 8198 |
| extreme_logits | mean_log_size | 0.731058579 | 0.403963108 | 1 | 62 |

计数不是时间，不能从这张表推断 GPU 加速。精确 max/LSE 的全部细 key 成本已计入在线列；粗 key 缓存更新与独立的全细 oracle 诊断另列于 JSON，表中未把它们混进在线计数。不同区域长度下，Top-1 对应的细 Token 成本差别很大。

只要 ||v_i||₂≤B，且选中细 Token 沿用原 logits 精确重归一化，就有 ||y−y_S||₂≤2Bε。尖锐例达到等号；无 value 范数界、或选中后改权重的例子会使不当套用失效。详见报告。

## 范围与复现

本任务只验证数学与小型 CPU 数值，不含 Qwen 特征、模型训练、GPU 吞吐量或视觉质量结论。

运行 python3 run_experiment.py 即可重跑测试和实验并生成此交接、结果与包。只需 Python 标准库。源文件 SHA256 在 RESULT.json；其他会话或现有原型无需修改。

代码审阅与云端 CI：[PR #12](https://github.com/www080327-glitch/inverse-vit-lab/pull/12)。
