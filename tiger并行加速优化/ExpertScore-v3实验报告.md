# π-SID ExpertScore v3 实验报告

日期：2026-09-23  
数据：RecBoard `Amazon2014Beauty_550_LOU`，full ranking，max history=20，seen-item mask 关闭，seed=2025  
代码：`/data/fszhang/RecBoard-master/HiFlow-SID`

## 设计依据

`2504.16054v1.pdf` 的关键分解是先预测高层语义，再以该语义为条件预测低层输出块；预训练阶段使用离散 token，post-training 阶段加入随机初始化的小 action expert，并联合离散交叉熵与 flow matching 损失。推理时高层输出只产生一次，低层块由独立专家处理。

本项目将这一路径改成推荐排序：粗粒度 SID `c1` 是高层语义，后缀 `c2,c3` 由一个低秩条件专家计算联合残差分数。服务路径只保留快速编码器、并行 SID 头、低秩专家和全目录矩阵打分，不保留 T5 teacher。

## v3 改动

1. 快速编码器采用 SID-MLP++ 同类的全局上下文 MLP，并按 SID 文本的 6 位周期使用位置专用子网络。
2. 训练时用冻结 MTP/T5 进行 token-level 表征对齐和 logits 对齐；这是训练期压缩信号，推理不加载 teacher。
3. ExpertScore 每批加入 2048 个目录 SID 负例，避免只用 batch 内负例造成的排序偏差。
4. 修复推理路径遗漏的 item-level alignment residual，并保留并行 SID 概率作为主分数。
5. 低秩专家残差门从 0 改为 0.05，使专家参数从第一步获得梯度，同时不破坏并行先验。

## 训练配置

```text
id=pisid-expertscore-distill-pos6-r8-v3
fast_hidden=256, fast_layers=2, fast_max_seq_len=128
expert_score=True, expert_rank=8, expert_negatives=2048
distill_encoder=True, lam_distill=0.25, lam_logit_distill=0.5
lam_prefix=1.0, lam_parallel=1.0, lam_align=0.2, tau_align=0.1
batch_size=512, epochs=50, lr=5e-4, eval_freq=5
```

## 质量结果

| 方法 | split | NDCG@10 | Recall@10 / HitRate@10 |
|---|---:|---:|---:|
| MTP checkpoint（本地基线） | test | 0.0221 | 0.0414 |
| ExpertScore v3（epoch 20 best） | test | **0.0260** | **0.0484** |
| SID-MLP 参考实验（Industrial_and_Scientific） | test | 0.02453 | 0.04552 |

ExpertScore v3 在本项目 Beauty test 上相对本地 MTP 提升 NDCG@10 17.6%、Recall@10 16.9%。与 SID-MLP 参考值相比，两个数值指标也更高。SID-MLP 参考值来自其 Amazon Reviews 2023 Industrial_and_Scientific 实验；它与 Beauty 的数据和 SID 协议不同，因此这部分是跨数据集数值参照，不能替代同数据集重跑。

## 速度结果

统一使用 SID-MLP 的硬件口径：test split、batch=32、native bf16、TF32 关闭、TIGER beam=50。ExpertScore 使用全目录 12,101 个物品矩阵打分。

| 方法 | test 总耗时 | 吞吐 | 相对 TIGER |
|---|---:|---:|---:|
| TIGER beam-50 | 69.553 s | 321.5 users/s | 1.00× |
| ExpertScore v3 rank-8 | **3.408 s** | **6561.0 users/s** | **20.41×** |
| SID-MLP++ 论文报告 | — | — | 8.74× |

因此，在本地 Beauty 协议下，ExpertScore v3 的实测加速倍数超过 8.74× 门槛。原始结果：`/data/fszhang/RecBoard-master/HiFlow-SID/results/speed_expertscore_sidmlp_rules.json`。

## 可复核文件

- 训练 checkpoint：`/data/fszhang/RecBoard-master/HiFlow-SID/logs/HiFlow-SID/Amazon2014Beauty_550_LOU/pisid-expertscore-distill-pos6-r8-v3/best.pt`
- 训练日志：`/data/fszhang/RecBoard-master/HiFlow-SID/logs/HiFlow-SID/Amazon2014Beauty_550_LOU/pisid-expertscore-distill-pos6-r8-v3/log.txt`
- 速度脚本：`/data/fszhang/RecBoard-master/HiFlow-SID/scripts/bench_expertscore_sidmlp_rules.py`
- 速度结果：`/data/fszhang/RecBoard-master/HiFlow-SID/results/speed_expertscore_sidmlp_rules.json`
