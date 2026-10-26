# C15 原文、实现与公式核对

## 固定来源

- GIVT: https://arxiv.org/html/2312.02116v2 ，§3.2、附录 A.1。
- GIVT 官方模型: https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/givt/givt.py
  本轮核对文件的 Git blob SHA 为 0bd4c84427fde5873c33dd7e76e6a289e163cb22。
  可用 https://api.github.com/repos/google-research/big_vision/git/blobs/0bd4c84427fde5873c33dd7e76e6a289e163cb22 固定读取。
- EAR: https://arxiv.org/html/2505.07812v1 ，§4.1 Eq.(8)–(10)。
- EAR 官方: https://github.com/shaochenze/EAR/blob/9256a9f450a6d52c4d22e72ea0e1b5f8b415dad1/models/scoreloss.py ，ScoreLoss.forward、energy_distance、energy_score。
- 历史 P2: https://github.com/www080327-glitch/inverse-vit-lab/blob/37b05311a06e7628b1752a79e8d9696fba5af20f/p2_qwen/probability.py ，nll、draw。历史代码只读参考，本轮未改动。
- 不复制第三方源码、论文或权重；以下为独立的小型实现和推导。

## GIVT 的默认独立假设与本轮 B 的区别

论文 §3.2 默认输出每个坐标一个 scalar GMM，共 3 d K 参数。这种密度为
p(x|c)=product_j sum_k pi_jk(c) Normal(x_j;mu_jk,s_jk)。

官方 givt.py 的 num_logits 及 get_pdf 实际支持不同分支：
per_channel_mixtures=True 对应上述逐坐标混合；
False 对应共享整向量分量的 Categorical + MultivariateNormalDiag，
输出 K+2Kd 参数。另一个 multivariate=True 分支要求 num_mixtures=1。
官方当前尺度转换用 _square_plus，本轮用 softplus(raw)+0.03，二者不能混称完全复现。
此处检查实际分支、reshape 和分布对象，不以注释判断。

按用户要求，本轮 B 采用共享整向量分量：
log p(x|c)=logsumexp_k[log pi_k(c) + sum_j log Normal(x_j;mu_kj,s_kj)]。
必须在每个分量内部先对维度组合，再对分量混合。
这不意味着分量内对角 GMM 的整体协方差为对角：
整体协方差还含不同分量均值的外积，因此它能表达一部分整体相关性。
本轮专门同时评估分量内相关结构。

历史 P2 的 nll 也先 sum(-1)，再 logsumexp；并使用 tanh 变量变换和对应 Jacobian。
本轮是无界合成向量，不使用 tanh、不使用像素去量化、不把像素 NLL 与向量 NLL 混比。

数值反例在 checks.json：
分量均值 (-2,-2)、(2,2)，各坐标尺度 0.2，权重各半；
对 x=(-2,2)，逐坐标混合会把两个不同分量拼在一起，得到很高密度；
共享分量联合密度则很低。两者对应不同模型，不能交换 sum/logsumexp。

## 全协方差 GMM

每个分量预测均值和下三角 L；对角 softplus(raw)+0.03，非对角不做正值限制，
Sigma=L L^T，因此稳定正定。自由参数 K+Kd+Kd(d+1)/2。
log N=-d/2 log(2pi)-sum_j log L_jj-1/2 ||L^{-1}(x-mu)||^2。
只做三角求解，不显式求逆或行列式。检查与 torch.distributions.MultivariateNormal
组成的 MixtureSameFamily 数值一致、梯度有限、Sigma 特征值为正。
这一对照是独立实现，并非声称官方 GIVT 已提供多分量全协方差。

## EAR：距离指数、独立样本与符号

论文采用 0<alpha<2，最大化
S(P,y)=E||X-X'||^alpha-2E||X-y||^alpha。
等价最小化双样本估计
L=||X1-y||^alpha+||X2-y||^alpha-||X1-X2||^alpha。

官方 forward 对同一 condition 重复两次，分别抽均匀噪声；
energy_distance 是 Euclidean norm 后 pow(alpha)，不是先 square 再 pow(alpha)；
energy_score 的样本对项乘 train_temperature，随后 loss=-score。
本轮 alpha=1、train_temperature=1；训练中两个独立噪声样本共享条件和目标，
不跨条件凑样本对。噪声 Uniform(-0.5,0.5)，不使用 CFG 或温度微调。
小型拼接 MLP 仅复现能量目标与隐式生成接口，不复现官方 AdaLN 大网络。
除以 sqrt(d) 只改变梯度尺度，不改变固定维度下最优分布。
没有显式密度，因此 EAR NLL 必须为 null。

## 平方距离：独立推导与数值校验

按本轮使用的、Eq.(9) 未加 1/2 的定义：
ED_2(P,Q)=2 E||X-Y||^2-E||X-X'||^2-E||Y-Y'||^2
        =2 ||E X-E Y||^2。

论文紧随 Eq.(9) 的文字写成均值差平方，没有该因子 2；
这是归一化常数差异，不影响“仅识别均值、不是严格适当评分”的结论。
本轮明确使用前式，避免直接照抄常数。

将 X=mu_P+epsilon，Y=mu_Q+eta 展开：
E||X-Y||^2=||mu_P-mu_Q||^2+tr Sigma_P+tr Sigma_Q；
两项自身距离分别为 2 tr Sigma_P 与 2 tr Sigma_Q，所以方差抵消。
对应期望训练目标 E L_2=2||mu_P-mu_Q||^2+2 tr Sigma_Q。

checks.py 对有限原子分布精确求和（非 Monte Carlo）：
Q 为 +/-1 等概率；P 分别为 Q、delta_0、+/-2 等概率。
alpha=2 的 ED 全为 0，alpha=1 则分别为 0、1、1。
这展示平方距离能量目标不能约束模式或方差。
它不否定确定性配对重建中的 MSE，也不否定以平方残差计算高斯 log likelihood：
这些目标的概率假设、配对条件及是否含样本间抵消项不同。

## 共同评价与解释边界

能量距离按 alpha=1 使用独立测试样本，无偏 U-statistic 不含自配对；
最大 128x128 block，无 N^2 大张量。checks.py 与小样本 dense 实现比对。
只对有密度的 diag/full 报告 NLL；共同排名不看训练 loss/NLL。
分布参数和数据种子先固定；按每个条件报告模式覆盖、比例、有效比例、
均值、整体/分量内协方差、所有坐标分位数。真分布独立抽样基线量化采样噪声。
真分布是 GMM，full 有模型族优势；所有模型参数接近不等于计算公平。
这里没有序列前缀、误差积累或图像解码器，不能据此解释完历史自由生成失败。
