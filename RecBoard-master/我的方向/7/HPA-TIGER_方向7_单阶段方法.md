# 方向⑦单阶段方法：HPA-TIGER

## 1. 为什么替换 RDSID-Rec

旧的 RDSID-Rec 在 T5 生成 beam 候选后，再用 history-conditioned SID prefix support 重新排序。这等价于“先召回、后重排”，增加了第二个推理阶段，不能和 TIGER 的单次生成协议公平比较。

方向⑦的新方法改为 **HPA-TIGER（History-aware Prefix Alignment TIGER）**：

> 在 T5 训练阶段，用用户历史表示直接监督每一级 SID 前缀；推理时只执行一次标准 T5 beam search。

因此，推理阶段没有 Support Head 打分、候选重排、全目录第二次评分或额外的 reranker。

## 2. 方法核心

给定用户历史 (h_u)，T5 编码器得到历史表示：

[
z_u = mathrm{Pool}(mathrm{Encoder}(h_u)).
]

物品的 SID 为 (s_i=(c_{i,1},ldots,c_{i,L}))。在标准 TIGER 序列生成损失之外，增加一个训练期的层级前缀对齐头：

[
q_l = W_l,mathrm{LN}(z_u), qquad
p_l(c)=mathrm{softmax}(q_l E_{mathrm{SID}}^	op/	au),
]

其中 (E_{mathrm{SID}}) 是 T5 的共享 token embedding，(l) 表示 SID 层级。每个 (p_l) 直接预测目标物品在第 (l) 层的 SID token。

总损失为：

[
mathcal L =
mathcal L_{mathrm{T5}}
+lambda_{mathrm{prefix}}rac{1}{L}sum_l
mathrm{CE}(p_l,c_{i,l})
+lambda_{mathrm{metric}}mathcal L_{mathrm{IB	ext{-}metric}}.
]

(mathcal L_{mathrm{IB	ext{-}metric}}) 使用 batch 内其他目标 SID token 作为负样本，约束用户历史表示和目标 SID token 的度量距离。它对应方向⑦中的 metric distance；SID 本身继续使用现有 SDQ tokenizer 生成的语义 ID，因此保留已有的 Sinkhorn 分配、语义重构和协同信息。

### 与旧方案的区别

| 项目 | 旧 RDSID-Rec | HPA-TIGER |
|---|---|---|
| 训练 | 标准 T5 生成损失 | T5 损失 + 层级 SID 前缀对齐 + batch 内度量损失 |
| 推理 | beam 生成后再算 support 并重排 | 一次标准 T5 beam 生成 |
| 额外推理模型 | 有 | 无 |
| 候选阶段 | beam 候选 | beam 候选 |
| 公平性 | 与 TIGER 不完全一致 | 与 TIGER 同一阶段、同一解码协议 |
| 推理复杂度 | 生成 + 重排 | 只有生成 |

## 3. 方向⑦组件对应关系

- **Metric distance**：batch 内目标 SID token 的对比度量损失。
- **Sinkhorn / 语义 ID tokenizer**：沿用已有 SDQ SID vocabulary，避免改变数据切分和物品编码协议。
- **Encoder–Decoder**：T5 encoder 编码用户历史，decoder 自回归生成 SID。
- **Collaborative signal**：用户历史和目标物品 SID 的联合训练把交互协同信号注入前缀表示；后续可加入显式协同蒸馏作为消融。
- **Decoder-only / one-stage generation**：最终排序只使用一次 T5 beam search 的序列概率。

## 4. 实现

代码：

- `/data/fszhang/RecBoard-master/SDQ-403/train_t5_hpa.py`
- `/data/fszhang/RecBoard-master/我的方向/7/run_hpa_tiger_serial.sh`

HPA 默认超参数：

- `lambda_prefix=0.25`
- `lambda_metric=0.05`
- `prefix_temperature=0.10`

辅助头只在 `fit()` 中参与反向传播，`recommend_from_full()` 沿用 TIGER 的标准 beam 生成实现。代码没有 support-weight、history rerank 或候选二次打分逻辑。

## 5. 公平实验协议

三个数据集均使用相同设置，按 Beauty → Sports → Toys 串行运行：

- 数据和 SID：`/data/fszhang/RecBoard-master` 现有协议
- 数据集：Beauty、Sports、Toys
- T5 epochs：200
- seed：2025
- beam：30
- 训练 batch：沿用各数据集配置
- 验证/测试 batch：96
- dtype：fp32
- constrained beam：关闭，使用 TIGER 快速协议的 plain beam
- checkpoint：按 validation NDCG@10 选择
- 最终指标：读取所选 checkpoint 的 test NDCG@10

速度比较应使用相同的全量用户、beam=30、batch=96、fp32设置。HPA 推理没有额外阶段，理论上只增加标准 T5 参数中训练期不用的辅助头，不增加评测时延。

## 6. 结果判定

在三套最终 test NDCG@10 产生前，不把中间 validation 或冒烟结果当最终结论。相对提升按：

[
rac{mathrm{HPA	ext{-}TIGER}-mathrm{SDQ	ext{-}VAE}}
{mathrm{SDQ	ext{-}VAE}}	imes100%.
]

Beauty、Sports、Toys 都完成后，再统一报告指标、相对提升和全量 beam-30 延迟。若某个数据集下降，优先做 `lambda_prefix`、`lambda_metric` 和 SID 层级权重消融，而不是添加推理后重排。

## 7. SOTA 假设与后续消融

假设是：标准 TIGER 的 token loss 只在目标 SID 路径上提供逐 token 监督，HPA 额外让编码器提前形成用户条件的层级 SID 分布，从而减少错误前缀选择，同时不改变推理阶段。

建议保留以下对照：

1. SDQ-TIGER baseline；
2. HPA-TIGER（完整方法）；
3. 只加 prefix CE；
4. 只加 metric loss；
5. 不共享 SID embedding；
6. 不同 `lambda_prefix` 和 `lambda_metric`。

只有在完整方法通过相同的 beam-30、fp32、batch96 协议后，才将其称为方向⑦的候选 SOTA 方法。

## 8. HPA-TIGER 结果与下一轮 GPA-TIGER（2026-09-26）

Sports 已完成 HPA-TIGER 正式评估。按 validation NDCG@10 选择 epoch 65：

| 数据集 | HPA-TIGER Test NDCG@10 | SDQ-VAE | 相对变化 |
|---|---:|---:|---:|
| Beauty | 0.0394 | 0.04107275 | −4.1% |
| Sports | 0.0230 | 0.02665110 | −13.7% |
| Toys | 0.0379 | 0.03889344 | −2.554% |

Sports 的 validation NDCG@10 为 0.0339，低于 SDQ-VAE validation 0.0352；因此下降已经发生在验证集，不能归因于单纯的测试集波动。当前 HPA-TIGER 仍然是单阶段标准 T5 beam 生成，没有 support score、候选二次打分或 reranker。

### 8.1 失败诊断

HPA 的训练损失为标准 T5 loss 加上三层 prefix CE 和 batch 内 metric loss，默认 `lambda_prefix=0.25`、`lambda_metric=0.05`。prefix projection 使用 T5 encoder pooled state 和共享 token embedding，但两个辅助头在推理时完全不参与生成。Sports 的长尾 SID 层在训练中受到较强的三层辅助约束，导致训练目标和实际 beam 生成目标不一致；Beauty 验证略高但测试下降，也说明辅助头出现了泛化落差。

### 8.2 下一轮：GPA-TIGER

GPA-TIGER（Gradient-Budgeted Prefix-Aligned TIGER）保留同一 SID vocabulary 和同一单阶段生成协议，只调整训练期辅助信号：

- 只使用第一级粗粒度 SID prefix CE，暂时移除深层长尾 token 的辅助监督；
- `lambda_prefix=0.08`、`lambda_metric=0.0`，第一轮去掉不稳定的 batch metric loss；
- 每个 batch 的辅助损失最多占标准 T5 loss 的 5%（`aux_max_ratio=0.05`），防止辅助头改写主生成目标；
- 保持 fp32、beam=30、validation/test batch=96、seed=2025、200 epochs 和 plain beam；
- `recommend_from_full()` 不变，推理仍只执行一次标准 T5 beam。

代码已加入 `/data/fszhang/RecBoard-master/SDQ-403/train_t5_hpa.py` 的 `--prefix-levels` 和 `--aux-max-ratio` 参数，并保留旧 HPA 默认行为。串行启动脚本已准备：

```bash
/data/fszhang/RecBoard-master/我的方向/7/run_gpa_tiger_serial.sh 20260926_gpa_tiger_serial
```

脚本只完成准备，当前不启动，以免打断 Toys 的 HPA 正式实验。验收时继续按 Beauty、Sports、Toys 逐数据集串行运行，按 validation NDCG@10 选择 checkpoint，再读取对应 test NDCG@10。

进入下一轮的通过条件：三个数据集 validation NDCG@10 均不低于 SDQ-VAE，且最终 Test NDCG@10 至少达到基线；任何改进都必须在单阶段 plain beam 协议下取得。
## HPA-TIGER 全部正式结果更新（2026-09-26）

Toys 已完成，按 validation NDCG@10 选择 epoch 85：
- Beauty: Test NDCG@10 0.0394，SDQ-VAE 0.04107275，相对 -4.073%。
- Sports: Test NDCG@10 0.0230，SDQ-VAE 0.02665110，相对 -13.700%。
- Toys: Test NDCG@10 0.0379，SDQ-VAE 0.03889344，相对 -2.554%。（按 validation NDCG@10 最优 epoch 85 重新读取；epoch 200 的 0.030598 不作为最终值。）

HPA-TIGER 三个数据集全部低于 SDQ-VAE。GPA-TIGER 下一轮单阶段方案已准备但尚未启动。

## HPA-TIGER 协议更正（2026-09-26）

最终值必须先按 validation NDCG@10 选择 checkpoint，再读取该 checkpoint 的 test NDCG@10。三套日志中的选择结果为：Beauty epoch 90（valid NDCG@10=0.0594，test=0.0394），Sports epoch 65（valid=0.0339，test=0.0230），Toys epoch 85（valid=0.0528，test=0.0379）。相对 SDQ-VAE test 基线分别为 -4.073%、-13.700%、-2.554%。epoch 200 的 Toys test=0.030598 仅是最后训练轮结果，不纳入正式比较。

单次 full-ranking beam-30、fp32、eval batch=96 的所选 checkpoint 测试耗时约为 Beauty 48.45 s、Sports 84.25 s、Toys 45.80 s；相对 TIGER 快速协议 40.83 s，Beauty/Toys 接近同一量级，Sports 明显偏慢，需要在下一轮排查数据规模和实现开销。


## HPA-TIGER 单阶段正式结果与失败诊断（2026-09-26）

运行目录：`/data/fszhang/RecBoard-master/我的方向/7/runs/hpa_tiger_serial_20250925/`。三组均按 200 epochs、seed=2025、fp32、beam=30、validation/test batch=96、plain beam 一次生成完成；没有 reranker、support score 或二次打分。按 validation NDCG@10 选择 checkpoint 后得到：

| 数据集 | 最佳 valid NDCG@10 | 所选 epoch | test NDCG@10 | SDQ-VAE | 相对变化 |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0594 | 90 | 0.0394 | 0.04107275 | 约 −4.1% |
| Sports | 0.0339 | 65 | 0.0230 | 0.02665110 | 约 −13.7% |
| Toys | 0.0528 | 85 | 0.0379 | 0.03889344 | 约 −2.6% |

三数据集均未达到基线，因此 HPA-TIGER 不通过。失败不是训练未完成：三条日志均有 `END ... rc=0`，并且选择 checkpoint 后重新执行了 test。

### 失败原因与下一轮方法

HPA 的 plain beam 取消了 SID 前缀约束，但 T5 训练仍以普通 token CE 为主，导致生成概率集中在高频 `<sid_0_*>` 前缀；Sports 的 valid/test 同时下降，说明主要问题是前缀判别和无效/重复 SID，而不是 checkpoint 选择。下一轮采用 **PF-TIGER（Prefix-Fidelity TIGER）**，仍为单阶段生成：

1. 保持同一 SID vocabulary、T5 200 epochs、beam=30、fp32、batch=512/96、seed=2025 和一次 plain full-ranking beam；不增加召回、重排或候选二次打分。
2. 训练目标仍是 T5 CE，在 level-0/1 SID token 上加入小权重 prefix CE，并对同 batch 中共享 level-0 或 level-1 前缀的 hard negatives 加 token-margin/InfoNCE；该损失只在训练使用，不改变推理路径。
3. 对 `<SID> ... </SID>` 结构加入 invalid/duplicate sequence penalty，惩罚训练中出现的非法 SID 片段和重复完整 SID，提升 plain beam 下的可解析率。
4. 使用 prefix dropout 和 scheduled sampling 减少 teacher-forcing 偏差；保留标准 validation NDCG@10 选点。

验收顺序固定为 Beauty→Sports→Toys：先跑 5 epoch 结构/配置 smoke，再跑正式 200 epoch。只有三数据集相对 SDQ-VAE 均至少 +1% 才写入通过表；任一低于门槛则保留日志并调整 prefix-margin/negative 权重，不混用 HPA 结果。
