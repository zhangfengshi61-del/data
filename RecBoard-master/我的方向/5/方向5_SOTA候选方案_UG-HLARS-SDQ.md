# 方向⑤ SOTA 候选方案：UG-HLARS-SDQ

Utility-Gated Hierarchical LARS for Structure Diffusion Quantization

## 1. 结论先行

方向⑤不需要放弃。现有结果说明 LARS 的几何方向是有价值的，但当前目标只优化了稀疏重构，和最终 T5 推荐之间存在断层：

| 方法 | Beauty NDCG@10 | Sports NDCG@10 | Toys NDCG@10 |
|---|---:|---:|---:|
| SDQ baseline | 0.041073 | 0.026651 | 0.038893 |
| LARS-main | 0.042504（+3.48%） | 0.024331（−8.70%） | 0.037975（−2.36%） |
| LARS-gated | 0.041464（+0.95%） | 暂无 test | 0.037897（−2.56%） |
| RecLARS trust-region | 暂无 Beauty/Sports 完成结果 | 暂停 | 0.038871（−0.06%） |

失败的核心不是“LARS 没有作用”，而是：

1. LARS 只看语义残差，不看推荐目标；
2. LARS 选出多个方向，但最终 SID 仍只保留一个硬码字，稀疏系数被丢掉；
3. 当前 sequence_loss 在三个 codebook 层都鼓励相邻物品使用相似分布，容易损伤 Sports/Toys 的细粒度区分；
4. 当前 gate 只判断稀疏 MSE 是否优于硬码字，没有判断 T5 的 next-token loss 是否变好。

因此，新的主线是：

> LARS 提出稀疏方向，推荐目标决定该方向是否写入离散码本。

方法暂命名为 UG-HLARS-SDQ：Utility-Gated Hierarchical LARS-SDQ，中文为“推荐效用门控的层级 LARS-SDQ”。

---

## 2. 和原方向⑤的关系

原方向⑤是：

- 内层：LARS 在当前字典上选择多个方向并求系数；
- 外层：固定系数更新字典；
- 最终：把改进后的字典送入 SDQ，导出 3 个离散 SID token。

UG-HLARS-SDQ 保留这三点，并加入两个方向⑤缺失的机制：

1. Soft sparse bridge：把 LARS 的连续稀疏结果变成 codebook 分布，而不是只用硬码字桥接；
2. Recommendation utility gate：只有当稀疏结果能改善用户序列的推荐损失时，才让它影响该层码本。

这使得方法仍然是 LARS-SDQ，而不是换成纯 Gumbel tokenizer 或普通序列推荐模型。

---

## 3. 方法结构

### 3.1 原 SDQ 路径保持不变

对每个结构扩散层 l：

- Y_l：扩散后的残差；
- C_l：第 l 层 256 个码字；
- Q_l：原 SDQ 的硬量化结果；
- R_(l+1)：下一层残差；
- 导出的 SID 仍然为 3 个 token，T5 配置、beam size 和推理流程不变。

原始路径仍然是：

Y_l → nearest/Sinkhorn assignment → Q_l → residual update

### 3.2 LARS 稀疏教师

对停止梯度的 Y_l 和归一化字典 D_l 运行预算截断的 LAR：

A_l = LAR(stopgrad(Y_l), stopgrad(D_l))

得到连续稀疏重构：

U_l = A_l D_l

建议继续使用粗到细预算：

- 第 1 层：3 个方向；
- 第 2 层：2 个方向；
- 第 3 层：1 个方向。

这样保留原方案对结构层次的假设：早期层提供组合表达能力，后期层只做小幅细化。

### 3.3 Soft sparse bridge

当前实现的 bridge loss 是把稀疏重构直接拉向一个硬码字。它无法保留多个方向的相对信息。UG-HLARS-SDQ 改成两个分布：

稀疏教师分布：

q_l(k|i) = softmax(-||U_(i,l) - C_l[k]||² / tau_l)

当前量化分布：

p_l(k|i) = softmax(-||Y_(i,l) - C_l[k]||² / tau_l)

桥接损失：

L_softbridge = KL(q_l || p_l)

其中 q_l 停止梯度，p_l 仍然对编码器和码本传播梯度。导出 SID 时仍使用硬 argmax，所以推理接口不变。

这个变化解决了当前方向⑤的离散断层：LARS 的多个方向不会直接成为多个 token，但会以一个分布形式影响当前样本应该靠近哪些离散码字。

---

## 4. 层级推荐目标

### 4.1 当前 sequence loss 的问题

当前实现对每一个 codebook 单独计算：

similarity_l(i,j) = p_i,l · p_j,l

然后让用户序列中的相邻物品在每一层都相似。这相当于让 i → j 的物品在粗粒度层、细粒度层都共享相同倾向，后两层容易失去区分能力。

### 4.2 层级转移损失

计算累计前缀相似度：

s_l(i,j) = p_i,l · p_j,l

S_1(i,j) = s_1(i,j)

S_2(i,j) = s_1(i,j) s_2(i,j)

S_3(i,j) = s_1(i,j) s_2(i,j) s_3(i,j)

对真实相邻转移 i → j，在当前 batch 的候选物品中计算：

L_hier = CE(S_1 / tau_1, j)
      + 0.5 CE(S_2 / tau_2, j)
      + 0.25 CE(S_3 / tau_3, j)

含义是：

- 第一层主要学习哪些物品属于相同的用户兴趣区域；
- 第二层保留一定的行为相似性；
- 第三层不再被强迫完全共享，而是只在能区分正样本和 hard negative 时提供信息。

负样本需要混合三类：

1. batch 内随机负样本；
2. 共享第 1 层前缀、但用户没有点击的物品；
3. 语义相似但交互不同的 hard negative。

### 4.3 完整 SID 的碰撞抑制

对语义相似度低但完整 SID 重合概率高的物品对，增加：

L_collision = mean(ReLU(S_3(i,j) - m_col))

只对语义相似度低于阈值的物品对计算，避免拆散真实共购物品。

这比当前 diversity loss 更直接，因为它抑制的是完整 SID 的重合，同时不会要求所有相邻物品在每一个 level 都相似。

---

## 5. 推荐效用门控

### 5.1 为什么需要推荐门控

当前 gate 使用几何误差：

||U_l - Y_l||² < margin × ||Q_l - Y_l||²

它只能说明稀疏重构在几何空间中更近，不能说明 T5 更容易预测下一个物品。Beauty 上几何改进和推荐改进一致，但 Sports/Toys 上二者已经明显分离。

### 5.2 RecLite 教师

第一版建议训练一个很小的 SID transition teacher，不改变最终 T5：

- 2 层 Transformer 或 GRU；
- hidden size 128；
- 输入为用户历史 SID；
- 目标为下一个物品的 3 个 SID token；
- 先用 SDQ baseline 的硬 SID 训练；
- tokenizer 训练阶段冻结 teacher 参数。

对软 SID，使用每个 codebook token embedding 的期望值：

e_l(i) = sum_k p_l(k|i) E_l[k]

这样 RecLite 能接收连续的软 assignment，并把 next-token loss 的梯度传回 p_l、编码器和码本。

如果第一轮有效，再把 RecLite 替换为已经训练好的 baseline T5，使用 T5 的 inputs_embeds 计算同样的软 SID loss，作为最终版本的 downstream teacher。

### 5.3 效用差值

对样本 i 和第 l 层，分别计算：

- 使用当前分布 p_l 的推荐损失：L_rec(p_l)；
- 把第 l 层替换为 LARS 教师分布 q_l 的推荐损失：L_rec(q_l)。

定义推荐收益：

Delta_l(i) = L_rec(p_l) - L_rec(q_l)

若 Delta_l(i) 为正，说明 LARS 教师能降低推荐损失；若为负，说明它虽然改善几何重构，却伤害下游推荐。

门控改为连续形式：

g_l(i) = sigmoid((Delta_l(i) - m_l) / tau_g)

最终稀疏桥接为：

L_bridge = sum_l mean_i [g_l(i) KL(q_l(i) || p_l(i))]

这样，几何上有益但推荐上有害的样本不会继续改动码本。

### 5.4 总损失

L = L_SDQ
  + rho(e) [lambda_s L_sparse
          + lambda_b L_bridge
          + lambda_h L_hier
          + lambda_c L_collision
          + lambda_r L_rec
          + lambda_a L_anchor]

其中：

- L_sparse 保留原 LARS 稀疏重构；
- L_bridge 使用推荐效用门控；
- L_hier 替换当前 sequence loss；
- L_collision 负责完整 SID 区分；
- L_rec 让 tokenizer 看到下游推荐目标；
- L_anchor 保留 SDQ 的 codebook、encoder 和 SID trust region。

---

## 6. 推荐训练流程

### Stage 0：复现基线

固定现有协议：

- Amazon2014 Beauty、Sports、Toys；
- seed=2025；
- VAE 100 epochs；
- T5 200 epochs；
- beam=30；
- 按 validation NDCG@10 选 checkpoint；
- 读取对应 test NDCG@10。

同时保存：

- SID collision rate；
- 每层 PPL；
- 每层 prefix entropy；
- target prefix survival；
- RecLite/T5 的 next-token loss。

### Stage 1：训练推荐教师

1. 用 SDQ baseline 的 SID 训练 RecLite；
2. 只使用训练集中的相邻交互；
3. 冻结 teacher；
4. 记录 baseline SID 的 next-token loss，作为效用门控的参考。

### Stage 2：UG-HLARS tokenizer

建议起始设置：

- lars_steps = [3,2,1]；
- stage_weights = [1.0,0.5,0.25]；
- sparse_weight = [0.03,0.02,0.01]；
- softbridge_weight = [0.05,0.03,0.02]；
- hier_weight = 0.05；
- collision_weight = 0.02；
- anchor_weight = 0.5；
- sid_anchor_weight = 0.10；
- tau_l = [0.10,0.07,0.05]；
- 前 10 个 epoch 只训练 SDQ + LARS 几何损失；
- 第 11 个 epoch 开始加入层级推荐损失；
- 最后 30 个 epoch 逐渐降低 tau，导出硬 SID。

### Stage 3：原 T5 训练

导出硬 SID 后，完全复用现有 SDQ-403/train_t5.py。UG-HLARS 不把 RecLite、LARS 系数或额外连续向量输入 T5，保证推理成本和 baseline 一致。

---

## 7. 实验消融顺序

不要一开始同时跑大量组合。建议按下列顺序，每次只改变一个机制：

| 编号 | 方法 | 目的 |
|---|---|---|
| A0 | SDQ baseline | 参考下限 |
| A1 | 当前 RecLARS trust_region | 已有方法复核 |
| B1 | 只替换 hierarchical sequence loss | 验证当前 sequence loss 是否伤害细粒度层 |
| B2 | B1 + soft sparse bridge | 验证是否弥补 LARS 系数被丢弃 |
| B3 | B2 + utility gate | 验证推荐效用门控 |
| B4 | B3 + collision loss | 验证长尾/细粒度区分 |
| B5 | 完整 UG-HLARS-SDQ | 主结果 |
| C1 | B5 去掉 LARS，仅保留推荐 tokenizer loss | 证明 LARS 仍有独立价值 |
| C2 | B5 将 LARS 替换为 top-correlation/OMP | 证明等角 LARS 路径的必要性 |
| C3 | B5 使用固定 gate | 与效用门控比较 |

第一阶段可以只跑 VAE 和 RecLite 指标进行筛选。只有当以下三个条件同时满足，才进入原 T5 的 200 epoch 完整实验：

1. 完整 SID collision rate 不高于 SDQ baseline；
2. RecLite next-token loss 低于 SDQ baseline；
3. 每层 target prefix survival 至少不下降，尤其是第 2、3 层。

---

## 8. 代码落点

### SDQ-LARS/lars_quantizer.py

新增：

- sparse_teacher_probs：从 U_l 计算 q_l；
- soft_bridge_loss：计算 KL(q_l || p_l)；
- 保存每层 p_l 和 q_l，供训练器计算推荐效用；
- 将旧的硬码字 bridge 作为对照选项保留。

### SDQ-LARS/train_lars_vae.py

修改：

- 删除当前逐层独立的 sequence_loss；
- 加入累计 prefix similarity；
- 加入正样本转移、同前缀 hard negative、完整 SID collision；
- 接入冻结 RecLite；
- 计算每层 Delta_l 并生成 utility gate；
- 记录 utility_gain、gate_rate、prefix_survival_proxy。

### SDQ-LARS/run_experiment.py

新增 variant：

utility_hier

它需要保存：

- teacher checkpoint；
- tokenizer 配置；
- gate 统计；
- SID audit；
- 原版 T5 的 result.json。

T5 阶段继续调用原始 SDQ-403/train_t5.py，不复制一份新的推荐实现。

---

## 9. 评估口径

需要防止碰撞率变化造成虚假的推荐提升。近期的 Semantic-ID 评估工作指出，碰撞会使 item-level 与 SID-level 指标产生偏差，甚至改变 tokenizer 的方法排序。[Faithful Evaluation of Semantic-ID Tokenizers](https://arxiv.org/abs/2605.25330)

因此必须同时报告：

- 原有 full ranking NDCG@10；
- 对 SID collision 做 item-level 去重后的 NDCG@10；
- Recall@10、Hit@10、MRR；
- 每层 PPL 与 token utilization；
- 完整 SID collision rate；
- target prefix survival；
- RecLite/T5 next-token loss；
- 训练时间和推理延迟。

只有 item-level 指标、prefix survival 和 next-token loss 同时改善，才把结果写成方法有效。

---

## 10. 可能的论文贡献

如果 UG-HLARS-SDQ 在三个数据集都超过 SDQ-VAE，贡献可以写成：

1. 提出推荐效用门控的稀疏方向学习，把 LARS 的连续稀疏重构和下游生成式推荐联系起来；
2. 提出 soft sparse bridge，使多个 LARS 方向以离散 code distribution 的形式进入 Semantic ID 学习；
3. 提出层级非对称推荐目标，令早期 SID token 建立行为共享，后期 token 保持物品区分；
4. 在结构扩散量化中保持固定 SID 长度、固定 T5 规模和固定推理流程，同时改善 tokenizer 与推荐目标的匹配。

需要正面区分 DIGER：DIGER 通过可微 SID 和 Gumbel 探索让推荐损失更新 tokenizer，重点是端到端梯度与 codebook collapse；UG-HLARS-SDQ 的独立问题是如何在 SDQ 的结构扩散空间中保留 LARS 稀疏方向，并以层级效用门控决定哪些方向可以改变离散 SID。[DIGER](https://arxiv.org/abs/2601.19711)

---

## 11. 失败时的判定

如果 B1 已经改善 Sports/Toys，而 B3 没有改善，说明推荐教师或 gate 过强，应保留层级损失，降低 utility gate 权重。

如果 B3 改善了 RecLite 但 T5 变差，说明 surrogate 与真实 T5 不一致，应切换到冻结 baseline T5 的 inputs_embeds 版本。

如果完整方法仍只改善 Beauty，方向⑤不能再把“几何稀疏重构”作为主贡献，应将论文主线改成“推荐效用门控的结构化稀疏量化”，并报告数据集差异，而不是继续堆叠 LARS 超参数。

如果 full ranking 提升但 collision-robust item-level 指标不提升，不应宣称超过 SDQ-VAE。

---

## 12. 参考文献

- Efron et al. Least Angle Regression. 2004. https://arxiv.org/abs/math/0406456
- Mairal et al. Online Dictionary Learning for Sparse Coding. 2009. https://www.di.ens.fr/~fbach/mairal_icml09.pdf
- Fu et al. Differentiable Semantic ID for Generative Recommendation. 2026. https://arxiv.org/abs/2601.19711
- Xia et al. Unleash the Potential of Long Semantic IDs for Generative Recommendation. 2026. https://arxiv.org/abs/2602.13573
- Zhang et al. Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation. 2026. https://arxiv.org/abs/2605.25330
- Zheng et al. Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation. https://arxiv.org/abs/2311.09049

## 13. 当前代码落地状态

已在 `/data/fszhang/RecBoard-master/SDQ-LARS` 加入 `utility_hier` variant，当前正式实现包含：

- `[3,2,1]` 分层 LAR budget；
- soft sparse bridge；
- 累计 prefix transition loss；
- in-batch utility gate；
- dissimilar-item full-SID collision loss；
- SDQ codebook、encoder 和 baseline SID anchor。

RecLite 或冻结 baseline T5 的 inputs_embeds 反馈属于下一阶段增强；当前正式 run 先使用无数据泄漏的 in-batch transition proxy，保证与已有协议的训练规模和 T5 阶段一致。

正式 run：`20260924_ug_hlars`，Beauty、Sports、Toys 并行使用 GPU 2。VAE/T5 仍分别采用 100/200 epochs、beam=30、seed=2025，最终按 validation NDCG@10 选择 checkpoint。


## 7. UG-HLARS-SDQ 正式结果与失败诊断（2026-09-26）

Beauty 的正式结果已经完成，按 validation NDCG@10 选择 epoch 95：

| 数据集 | Utility-Hier Test NDCG@10 | SDQ-VAE | 相对变化 |
|---|---:|---:|---:|
| Beauty | 0.041210661 | 0.041072751 | +0.336% |
| Toys | 0.039517915 | 0.038893436 | +1.606% |
| Sports | 仍在训练 | 0.026651102 | 待定 |

Beauty 没有达到自动化要求的 1% 门槛，因此不能把当前版本判定为三数据集 SOTA。Beauty 的 validation NDCG@10 为 0.058967，T5 最佳 checkpoint 的 test NDCG@10 为 0.041211；epoch 200 的 test 0.036568 不作为正式结果。

### 7.1 失败原因

1. **连续重构改善没有完全传递到推荐。** Beauty 的 VAE PPL 从 SDQ 的 217.318 降到 201.680，level-0/1 PPL 从 181.537/214.638 降到 168.849/180.674，但 Test NDCG@10 只有 +0.336%。
2. **完整 SID 碰撞上升。** Beauty collision rate 从 0.142633 升到 0.193868；Toys 从 0.200939 升到 0.256122。当前 collision loss 只在随机 pair 上工作，且权重 0.02，无法约束导出的硬 SID。
3. **前缀使用变得集中。** Beauty 的 level-1 soft SID 熵从 5.369 降到 5.197，Toys 从 5.343 降到 5.231。唯一 code 数量接近 256 不能代表使用均衡，头部 code 频率已经明显增大。
4. **效用门槛过低。** 当前 utility margin=0、temperature=0.05，微小的 batch transition gain 也会推动稀疏教师桥接；这会让几何上有利但泛化不稳定的改动进入 SID。

因此下一轮需要把“可用的稀疏方向”限制在 SDQ 的 SID 信赖域内，同时显式控制软 SID 边际分布和完整 SID 碰撞。

### 7.2 下一轮方法：TC-UG-HLARS-SDQ

TC-UG-HLARS-SDQ（Trust-Calibrated Utility-Gated Hierarchical LARS-SDQ）保留 LARS 内层、SDQ 三层硬量化和原版 T5，只改变训练期约束：

- `sparse_weight=0.015`，降低连续目标对码本的漂移；
- `anchor_weight=0.75`、`sid_anchor_weight=0.25`，更强地保持 SDQ 的方向和离散 SID；
- `utility_margin=0.03`、`utility_temperature=0.02`，只有有明显推荐效用的稀疏教师才参与桥接；
- `collision_weight=0.10`、`collision_margin=0.01`，并启用 `diversity_weight=0.05`，直接压制 dissimilar item 的完整 SID 重合；
- 新增 `usage_balance_weight=0.02`，对每层 batch marginal soft SID 分布施加 KL-to-uniform 正则，缓解头部 prefix 集中；
- `soft_bridge_weight=0.03`、`hier_weight=0.02`，降低层级推荐辅助项的扰动；
- LARS budget 继续使用 `3,2,1`，T5 仍为 200 epochs、beam=30、seed=2025。

新增代码参数位于 `train_lars_vae.py`，运行器已加入 `tc_ug` variant。下一轮串行启动脚本为：

```bash
/data/fszhang/RecBoard-master/SDQ-LARS/run_tc_ug_hlars_serial.sh 20260926_tc_ug_hlars
```

该脚本目前只完成准备，未启动训练，以免打断当前 Sports 正式实验。启动后仍按 Beauty、Sports、Toys 顺序运行，并沿用当前 VAE 100 epoch、T5 200 epoch、beam=30 的协议。

### 7.3 下一轮验收条件

进入正式 T5 之前，必须同时检查：

- 三层 collision rate 不高于对应 SDQ baseline；
- level-0/1 的 usage entropy 不低于当前 utility_hier；
- SID anchor 的改变比例受控；
- T5 validation NDCG@10 不低于 SDQ baseline。

最终仍以 validation NDCG@10 选 checkpoint，并报告对应 Test NDCG@10；未完成的 Sports 和下一轮结果不写入当前主结果表。


## CP-UG-HLARS-SDQ：保守前缀校准方案（2026-09-26）

### 动机

utility_hier 在 Sports 的 test NDCG@10 为 0.0240369428，较 SDQ-VAE 0.02665110 低 9.809%。诊断显示 level-0/1 PPL 从 SDQ 的 172.94/225.19 降到 158.91/200.11，完整 SID collision 从 0.195893 升到 0.249224；原 utility gate 在零收益时仍输出约 0.5，导致前缀塌缩。短跑也确认，仅增加 collision 权重而不保护 assignment 会使 level-0 PPL 快速跌到 80，因此该配置被拒绝，不进入正式结果。

### 方法主体

CP-UG 仍然是 LARS/SDQ 的三层稀疏量化与 coarse-to-fine 训练。改动只用于保持推荐相关的 SID 决策边界：

1. 从 SDQ `model.pt` warm-start，前 100 个 VAE epoch 冻结 encoder/codebook，保证硬 SID 与基线一致；LARS 的结构、稀疏量化和训练预算仍执行。
2. 仅当 batch transition gain 大于 `utility_margin=0.001` 时开放 level-0 utility bridge；level-1/2 不参与 bridge，避免无效前缀更新。
3. 加入 baseline SID cross-entropy、逐层 marginal KL 和最近码 assignment hinge（权重 0.75、0.50、2.0），并直接约束 full-SID overlap。
4. T5 训练使用同一 SID vocabulary，在 teacher forcing 中对早期 SID token 加 0.50 小权重，改善前缀判别；推理仍只调用一次标准 beam=30 full-ranking。

### 公平协议

Beauty、Sports、Toys 串行；VAE 100 epochs、T5 200 epochs、seed=2025、fp32、训练 batch=512、validation/test batch=96、beam=30、`apply-constrained-beam-search=False`、num_workers=0。按 validation NDCG@10 选择 checkpoint，再读取对应 test NDCG@10；无 reranker、support score、候选二次打分。运行器：

```bash
/data/fszhang/RecBoard-master/SDQ-LARS/run_cp_ug_hlars_serial.sh 20260926_cp_ug_hlars
```

只有三个数据集都达到相对 SDQ-VAE 至少 +1% 才通过。任一数据集低于门槛，保留日志并基于 PPL、collision、SID 使用率、validation/test NDCG 进入下一轮方案；不把冒烟或未完成结果写入主表。

### 短跑验收

- `CP-UG-SMOKE2`（warmup=1）拒绝：Sports epoch 3 PPL#0=80.67、collision=0.2694。
- `CP-UG-SMOKE3`（warmup=10、freeze=10）通过安全性验收：epoch 10 仍为 SDQ PPL=217.9585、collision=0.1959；解冻后 pilot 继续监测 assignment 漂移。
- 正式候选采用 freeze=100，并把收益放在 T5 前缀校准项，避免改变 SDQ 硬 SID 分布。


## 2026-09-27 CP-UG-HLARS-SDQ Sports 正式结果与失败诊断

- 运行：`/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260926_cp_ug_hlars/Sports/`，沿用 SDQ VAE SID vocabulary；T5 200 epochs、seed=2025、fp32、beam=30、训练 batch=512、评估 batch=96、plain full-ranking beam（`apply-constrained-beam-search=False`）。
- 按 validation NDCG@10 选中的 checkpoint：`/data/fszhang/RecBoard-master/SDQ-LARS/logs/LARS-20260926_cp_ug_hlars-T5/Amazon2014Sports_550_LOU/t5/data/best.pkl`。
- 最佳 validation NDCG@10：`0.03473028106894046`（约第 17 个评估点）；对应 test NDCG@10：`0.02518133539533798`。
- 相对固定 SDQ-VAE test `0.02665110`：`-5.5148365533%`，未通过 +1% 门槛。
- 监控：Sports validation NDCG@10 在早期达到 0.03473028 后持续下降至约 0.02853；说明 prefix calibration 权重 0.5 在 Sports 上出现明显早期过拟合。VAE SID 诊断基线为 PPL 217.9585、collision 0.19589；CP-UG 只改变 T5 训练期 token CE，不改变 SID vocabulary 或一次生成解码，因此失败主要来自 T5 prefix loss 的泛化，而非二阶段排序。

## 下一轮 APC-UG-HLARS-SDQ（Adaptive Prefix Calibration）

为保持 LARS 主体和 TIGER 公平协议不变，新增训练期线性 prefix-loss annealing：prefix token 权重从 0.50 在前 60 个 epoch 线性退火到 0.10，之后保持 0.10；推理仍为一次标准 beam=30 full-ranking，禁止 constrained beam、support score、reranker 和二次打分。该退火降低 Sports 的早期过拟合，同时保留 Beauty 上 CP-UG 的 prefix 校准收益。

- 代码：`/data/fszhang/RecBoard-master/SDQ-403/train_t5_fp32.py` 新增 `--prefix-loss-min-weight` 与 `--prefix-loss-decay-epochs`。
- 串行脚本：`/data/fszhang/RecBoard-master/SDQ-LARS/run_cp_ug_adaptive_serial.sh`。
- 正式参数：max=0.50、min=0.10、decay=60；Beauty→Sports→Toys 串行，仍使用 batch=512/96、fp32、beam=30、seed=2025。
- 当前 CP-UG Toys 完成后，确认无重复进程和 GPU 资源，再启动新一轮；旧运行和 checkpoint 保留，不混用。


## CP-UG 本轮 Toys 完成与 APC-UG 启动（2026-09-27）

- CP-UG Toys：最佳 validation NDCG@10 `0.053197000799794655`，对应 test NDCG@10 `0.03980501310071969`，相对 SDQ-VAE `+2.343770828%`。
- CP-UG 三数据集正式结果：Beauty `0.04340491972472871`（`+5.678143598%`）、Sports `0.02518133539533798`（`-5.514836553%`）、Toys `0.03980501310071969`（`+2.343770828%`）；因 Sports 未达到 +1%，本轮不通过。
- 已完成无重复进程与 GPU 审计，GPU3 空闲后启动 APC-UG-HLARS-SDQ 串行轮次：`/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_cp_ug_adaptive/`。当前 Beauty 正在训练，Sports/Toys 尚未启动。
- APC 参数：prefix-loss max=0.50、min=0.10、linear decay=60 epochs；其余协议完全保持不变。


## LARS-only-HRQ：完全脱离 SDQ 的 SID 生成方案（2026-09-27）

### 方法动机

此前 CP/APC 仍然沿用了 SDQ 的 SID 词表或 SDQ 初始化。为验证方向5的 LARS 本身能否独立生成可用于生成式推荐的 SID，新增 **LARS-only-HRQ（Hierarchical Residual LARS Quantization）**。该方案不导入 SDQ `quantizer.py`、SDQ ODE/structure diffusion、Sinkhorn、SDQ codebook、SDQ checkpoint 或 SDQ `sid_vocab.json`。

### SID 生成

1. 从 item 的 sentence-T5 semantic feature 出发，训练一个轻量 LARS encoder。
2. 建立 3 个全新的 codebook，每层 256 个 codeword；初始 codeword 从当前数据集特征随机采样，不能使用 SDQ 初始化。
3. 对 encoder 输出逐层计算残差。每层用预算截断的 `lar_code`（LARS active-set）选择当前残差相关性最大的稀疏 atom，输出该 atom 的离散索引；再从残差中减去该 LARS atom，进入下一层。
4. 训练目标由 LARS hard residual reconstruction、commitment、codeword diversity 和轻量 usage entropy 组成，只用于防止残差失真与 codebook collapse。
5. 100 个 VAE epoch 后导出全量 `sid_vocab.json`，每个物品正好 3 个新 SID token：`<sid_0_i>, <sid_1_j>, <sid_2_k>`。

训练代码：`/data/fszhang/RecBoard-master/SDQ-LARS/train_lars_only_vae.py`。该文件的 `sdq_imports=false` 审计字段用于确认没有引入 SDQ 量化器。

### 下游协议

SID 生成后完全按原协议训练 T5：200 epochs、seed=2025、fp32、train batch=512、validation/test batch=96、beam=30、`apply-constrained-beam-search=False`，按 validation NDCG@10 选 checkpoint 并读取对应 test NDCG@10。T5 只读取本轮新生成的 LARS-only `sid_vocab.json`；不用 SDQ SID，不使用 reranker、support score 或二阶段精排。

### 初步审计

Beauty 100 epoch 原型：3 层均使用 256 个 codeword，三层 unique code 均为 256，完整 SID collision rate `0.1088339807`，低于 SDQ Beauty 基线 `0.1426328403`。该结果只证明 tokenizer 的结构可用，不能当作正式推荐结果；正式结果必须等待三数据集 T5 完成。

### 正式并行运行

正式运行目录：`/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_lars_only/`。Beauty、Sports、Toys 分别使用 GPU 0、1、2 并行执行 VAE 100 epoch 后 T5 200 epoch；GPU3 的 APC 任务保持不受影响。
