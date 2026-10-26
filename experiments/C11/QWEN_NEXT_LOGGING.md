# 下一轮真实 Qwen 训练最少记录字段

这些是诊断建议，不是正式重放比例决定，也不声称 toy 的比例可移植。

|类别|最低字段|
|---|---|
|运行身份|run_id、git commit、config hash、模型/教师标识、初始化快照、seed、数据版本与 split hash|
|更新边界|global optimizer step、microstep、accumulation、task、scheduler/lr、实际执行或跳过、AMP scaler/overflow|
|监督量|每任务样本数、输入/输出/有效 label token、图像数、patch/latent 位置、padding、loss mask、去重样本数|
|损失尺度|每任务 raw loss、reduction 与分母、加权系数、加权后 loss、KD 温度/T²、KL 方向、教师模式与冻结状态|
|梯度抽样|step、probe batch id、选定共享参数名、元素数、每任务未缩放 norm、加权 norm、dot/cosine、None/零梯度、clip 前后 norm、统计 rank/聚合方法|
|计算成本|每任务学生前向/反向时间、教师前向时间、优化耗时、诊断额外时间、显存峰值、吞吐、硬件与线程数|
|旧能力评估|固定指令集版本与 prompt/解码配置、基线答对集合、当前仍答对个数/分母、整体成功率、分类型指标、普通文本 CE|
|新能力评估|固定验证集与解码配置、新任务 raw loss、实际输出成功率/属性正确率、必要的视觉盲评与失败例|
|快照追踪|评估 step、训练样本累计、检查点 hash、固定保存与停止规则、恢复点与优化器状态是否恢复|

建议同时保留基线与中间时间点，尤其是新增任务前、早期和结束。新任务变好但旧指令退化，不能仅用普通 CE 改善抵消。旧教师错误与学生新增正确例另报，避免只看 retention 忽略可能改善。

先用小规模真实数据测日志开销、损失量级和实际性能轨迹，再决定后续对照与正式比例。这里不自动变更训练设置、数据混比、参数更新范围或部署策略。
