# C09 hierarchical attention reference

交接入口：[HANDOFF.md](HANDOFF.md)（实测摘要）。完整推导：[REPORT.md](REPORT.md)。先读前三条容易算错的地方。

- [reference_attention.py](reference_attention.py)：可复用 binary64 数学参照，最多 1024 细 Token。
- [test_reference_attention.py](test_reference_attention.py)：无下载、无模型、CPU 测试。
- [run_experiment.py](run_experiment.py)：固定 seed=9027，生成所有测量与交付包。
- [FAILURE_EXAMPLES.md](FAILURE_EXAMPLES.md)：预定义失败例与缺失假设。
- [RESULT.json](RESULT.json)：云端 CPU 实测、完整 64 行 top-k 结果、成本及源文件哈希。
- [TEST_REPORT.txt](TEST_REPORT.txt)：实际测试输出。
- [C09_DELIVERY.zip](C09_DELIVERY.zip)：上述源代码、报告、结果、测试输出。

~~~sh
cd experiments/c09
python3 -m unittest -v test_reference_attention.py
python3 run_experiment.py
~~~

Python 3 标准库即可。这里的“float64”指 Python float 的 IEEE754 binary64，
测试要求 mant_dig=53。不需要 numpy/torch，不安装依赖、不下载数据/模型，不访问本机/DCU。
每个 attention 仅单 query，最大 N=1024，不建立巨型 dense 矩阵。
运行脚本覆盖 RESULT.json、TEST_REPORT.txt、C09_DELIVERY.zip；不会修改其他代码或上传任何东西。

执行环境：GitHub Actions 的独立 Linux CPU job，10 分钟上限。
这是数值与数学参照，不是 GPU 吞吐量或模型能力评估。
