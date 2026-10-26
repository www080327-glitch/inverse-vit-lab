# C13：反向权重初始化

入口：[中文实测交接](HANDOFF_ZH.md)。原始数值：[RESULT.json](RESULT.json)。
交付包：[C13_DELIVERY.zip](C13_DELIVERY.zip)。

这是一个独立的纯NumPy CPU机制实验：自造16×16图形、小型冻结MLP视觉编码器、两seed、三种同结构解码器初始化。没有真实Qwen权重或特征。实验对照与参数映射见 [PLAN_ZH.md](PLAN_ZH.md)，数学参照见 [MATH_ZH.md](MATH_ZH.md)。

按顺序运行 python experiment.py、python make_report.py、python package_delivery.py；不自动下载数据或启动其他任务。唯一依赖见 requirements.txt。
