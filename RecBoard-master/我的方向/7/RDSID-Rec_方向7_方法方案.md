# 方向⑦：角色解耦与支持感知语义 ID 生成式推荐（RDSID-Rec）

## 0. 文档定位

本方案只围绕 /data/fszhang/RecBoard-master/我的方向/7/7.png 中列出的方向⑦展开。图片给出的关键词是：

- 度量距离（metric distance）
- Sinkhorn 分配
- Encoder–Decoder 式物品 Tokenizer
- 协同正则（collaborative regularization）
- Decoder-only 生成式推荐

本文把这些组件组织成一个可以在 RecBoard 代码库上实现和验证的研究方案，暂命名为 RDSID-Rec（Role-Decoupled, Support-Aware Semantic-ID Recommendation，角色解耦与支持感知语义 ID 生成式推荐）。

> 研究定位：这是一个基于方向⑦的研究假设和实验设计，不应在完成实验前宣称已经达到 SOTA。目标是把语义 ID 的“语义组织作用”和“用户条件下的候选筛选作用”分开建模，减少 Semantic ID 对生成式推荐造成的错误硬过滤。

---

## 1. 研究背景与问题

生成式推荐把用户历史直接映射为物品的 Semantic ID（SID）序列。例如，一个物品可以表示为：

s_i = (c_i1, c_i2, c_i3)

其中每一个 c_il 是一个 codebook token。Decoder-only 模型逐级生成这些 token：

p_phi(s_i | h_u) = Π_l p_phi(c_il | h_u, c_i<l)

这里 h_u 是用户历史，phi 是生成模型参数。

现有方法通常让同一个 SID 同时承担三个职责：

1. 组织语义相近的物品，让相似物品共享前缀；
2. 唯一标识一个物品；
3. 在树式解码过程中筛选候选物品。

这三个目标并不总是一致。语义相似的物品未必适合当前用户；一个全局有效的 SID 前缀也可能把用户真正喜欢的长尾物品排除在 beam search 之外。近期对 SID 的分析发现，SID 邻域与表示空间的近邻重合度有限，描述改写也可能改变 SID；在生成末端，许多原本合理的目标会因前缀或最后一个 token 选择而消失。[Understanding Semantic IDs](https://arxiv.org/abs/2607.24995) 将这个现象概括为 SID 的多重角色冲突，并通过推理阶段的 item-selection guidance 缓解它。

### 核心研究问题

> 能否让 SID 继续负责组织物品语义结构，同时由一个用户条件的支持分布负责候选筛选，从而降低 Decoder-only 生成过程中的错误前缀过滤？

与只在推理阶段重排候选的做法相比，RDSID-Rec 的重点是：在训练时就把“用户支持某个 SID 子树”的信号注入 prefix prediction，使生成器学到哪些前缀对当前用户更有价值。

---

## 2. 方法总览

RDSID-Rec 包含两个相互配合但职责不同的部分。

### 2.1 语义组织分支：SID Tokenizer

Tokenizer 将物品内容/多模态特征和协同行为特征编码成多级 SID。它的主要任务是形成稳定、均衡、可解释的物品语义层次：

x_i → z_i^sem → (c_i1, ..., c_iL)

该分支使用方向⑦中的度量距离、Sinkhorn 和协同正则：

- 度量距离：以向量距离或相似度定义量化目标，避免单纯依赖 token 分类损失；
- Sinkhorn 分配：在每个 codebook 中进行近似均衡的软分配，减少 codebook collapse 和热门 token 过载；
- Encoder–Decoder：Encoder 将物品特征压缩为离散 SID，Decoder 从 SID 重构物品语义；
- 协同正则：从用户—物品交互学习协同表示，并约束 SID 不完全偏离真实偏好结构。

Tokenizer 不直接承担用户条件下的最终排序。

### 2.2 用户支持分支：Support Head

给定用户历史 h_u，Support Head 计算每个物品在当前用户下的支持分数：

r_psi(i | h_u) = f_psi(h_u, e_i^sem, e_i^collab)

其中 e_i^sem 是物品语义向量，e_i^collab 是由交互学习的协同向量。为了支持冷启动，可让语义特征作为主要输入，并把协同向量作为可选增强。

对于 SID 前缀 v=(c_1,...,c_l)，定义它对应的物品子树为 I(v)，并聚合该子树的用户支持质量：

A_psi(v | h_u) = log Σ_{i∈I(v)} exp(r_psi(i | h_u))

这个量回答的是：当前用户是否支持这个前缀下面的物品集合。它不是重新生成一个 SID，而是为生成模型提供用户条件的子树证据。

### 2.3 Decoder-only 生成分支

Decoder-only 模型仍然自回归生成 SID，但每一个 prefix 的概率都同时接受两种信号：

1. 生成器根据用户历史预测的 token 概率；
2. Support Head 对该 prefix 子树给出的支持质量。

定义联合 prefix 分数：

S(v | h_u) = Σ_j log p_phi(c_j | h_u, c_<j>) + lambda_|v|(h_u) A_psi(v | h_u)

lambda_l(h_u) 控制支持信号在第 l 层的强度。一个可行实现是让生成熵越高时支持信号越强：

lambda_l(h_u) = lambda_l^max · sigmoid(a(H_l(h_u)-b_l))

这样，模型在不确定时更多借助支持分布，在已经非常确定时少做干预。

---

## 3. 训练目标

总损失由五部分组成：

L = L_SID + alpha L_item + beta L_prefix + gamma L_sem + delta L_collab

### 3.1 SID 自回归损失

L_SID = - Σ_l log p_phi(c_il | h_u, c_i<l)

这是标准的 Decoder-only SID 生成损失，用于保证方案可以与现有 TIGER/SDQ 类生成式推荐直接比较。

### 3.2 物品级支持损失

对正样本物品 i+ 和负样本集合 N(u)，使用 sampled softmax 或 in-batch contrastive loss：

L_item = -log [ exp(r_psi(i+|h_u)/tau) /
(exp(r_psi(i+|h_u)/tau) + Σ_j∈N(u) exp(r_psi(j|h_u)/tau)) ]

负样本应混合随机样本、同前缀样本、热门物品和 hard negatives。这样可以专门测试 Support Head 是否能区分“语义相近但用户不喜欢”的物品。

### 3.3 Prefix 对齐损失

由 Support Head 在每一级 SID 树上得到目标分布：

q_psi(c_l | h_u,c_<l>) ∝ Σ_{i:c_i<l=c_<l>, c_il=c_l} exp(r_psi(i|h_u))

让 Decoder 的 token 分布与该分布对齐：

L_prefix = Σ_l KL(q_psi(·|h_u,c_<l>) || p_phi(·|h_u,c_<l>))

这是 RDSID-Rec 的关键：支持信号在训练阶段进入 prefix prediction，而不只是在 beam search 后处理。

### 3.4 语义重构与度量损失

Tokenizer 使用 Encoder–Decoder 重构物品语义：

L_sem = d(Dec(s_i), x_i) + eta L_metric

其中 d 可以是 MSE、cosine distance 或适合多模态特征的距离。度量项应保持物品间的相对邻近关系，而非只记住每个物品的 ID。

### 3.5 协同正则

设协同表示为 e_i^collab，语义表示为 e_i^sem，可以使用行为相似度保持项：

L_collab = Σ_(i,j) w_ij [ max(0, m+d(e_i^collab,e_j^collab)-d(e_i^sem,e_j^sem)) ]

也可以替换为 BPR、对比学习或用户条件的协同蒸馏。建议第一版使用简单的 in-batch contrastive regularization，避免引入过多自由度。

---

## 4. Tokenizer 的具体实现建议

### 4.1 初始配置

为了与现有 SDQ 实验保持可比，第一版建议：

- SID 长度：L=3；
- 每级 codebook 大小：K=256；
- 物品表示维度：沿用现有 SDQ 配置；
- assignment：每个 batch 内使用 Sinkhorn 归一化的软分配；
- training：先单独训练/加载 tokenizer，再训练 Decoder-only；
- ablation：增加 no_sinkhorn、no_collab、metric_only 三组。

### 4.2 Sinkhorn 分配

对 batch 中的量化 logits Q 加入温度和熵正则，使用 Sinkhorn-Knopp 将其近似投影到行列边缘受限的运输计划：

P = Sinkhorn(Q / tau)， c = argmax_k P[:,k]

训练时使用 straight-through 或 Gumbel-softmax，推理时使用离散 argmax。需要记录每个 codebook 的 token 使用率、perplexity 和最大/最小桶占比。

### 4.3 协同信息的注入方式

不要让协同向量完全替代语义向量，否则会削弱新物品和冷启动物品的可编码性。推荐：

z_i = W_s e_i^sem + g_i ⊙ W_c e_i^collab

其中 g_i 是门控向量。对于没有交互的物品，把 g_i 置为 0 或使用由内容特征预测的默认门控。

---

## 5. 解码算法

### 5.1 Prefix-aware beam search

对每个候选 prefix v 计算：

score(v) = log p_phi(v | h_u) + lambda_|v|(h_u) A_psi(v | h_u)

保留 top-B 个 prefix，直到得到完整 SID。若多个完整 SID 指向同一物品，只保留最高分路径。

### 5.2 伪代码

输入：用户历史 h、SID trie T、beam size B

1. beam = {root: 0}
2. 对 level = 1,...,L：
   - 对 beam 中每个 prefix，调用 Decoder 得到 token logits；
   - 只遍历 T 中的合法子节点；
   - 新分数 = 旧分数 + token log probability + lambda[level] × SupportMass(prefix|h)；
   - 保留 top-B 个 prefix。
3. 对完整 SID 去重并返回物品。

为了公平比较，应同时报告三种设置：

1. 标准 Decoder-only beam search；
2. 只在推理阶段加入 Support Head；
3. RDSID-Rec 的训练时 prefix 对齐 + 推理时 prefix 支持分数。

---

## 6. 与方向⑦图片的对应关系

| 图片中的组件 | 在 RDSID-Rec 中的具体位置 | 作用 |
|---|---|---|
| 度量距离 | Tokenizer 的 L_metric 和量化距离 | 保留物品表示空间的相对结构 |
| Sinkhorn | 每一级 codebook 的平衡分配 | 降低 codebook collapse 和热门 token 过载 |
| Encoder–Decoder | 物品特征到 SID，再从 SID 重构物品特征 | 生成稳定、可重构的语义 ID |
| 协同正则 | L_collab 和协同表示 | 注入用户行为信号，同时保留语义/冷启动能力 |
| Decoder-only | 用户历史到 SID 的自回归生成 | 生成候选物品 |
| 本方案新增机制 | Support Head、prefix support mass、训练时 prefix 对齐 | 把用户条件的候选筛选从隐式硬过滤中解耦出来 |

因此，RDSID-Rec 不是脱离图片另起一个方向，而是把图片列出的五个组件串成一个明确的研究问题：语义 ID 负责组织结构，Decoder-only 负责生成，用户支持分支负责纠正前缀筛选。

---

## 7. 实验设计

### 7.1 数据集与任务

第一阶段建议使用现有工程已经支持的 Amazon Beauty、Sports and Outdoors、Toys and Games 数据。采用 leave-one-out 或时间切分，保持与 RecBoard 当前 SDQ 结果一致。

主任务：给定用户历史，生成 top-K 物品。

### 7.2 对比方法

至少包含：

- 现有 SDQ-403 tokenizer + Decoder-only baseline；
- 标准 Semantic-ID generation；
- 只使用 Support Head 的推理后处理；
- RDSID-Rec（训练时 prefix 对齐）；
- 去掉 Sinkhorn 的 RDSID-Rec；
- 去掉协同正则的 RDSID-Rec；
- 固定 lambda 与熵自适应 lambda；
- 可选：DIGER/UniGRec 等已有可微或软 SID 方法作为外部参考。

### 7.3 主要指标

常规指标：

- Recall@10、NDCG@10、HitRate@10、MRR；
- 生成延迟、beam size、显存。

机制指标：

- Target Prefix Survival：真实目标物品在每一级 beam 中仍存活的比例；
- Final-token retention：进入最后一级前可行、但最终未生成的目标比例；
- Prefix support calibration：prefix support mass 与真实点击概率的相关性；
- SID utilization/perplexity：各 codebook 的 token 使用率；
- collision rate：不同物品映射到相同 SID 的比例。

分组分析：

- 头部、腰部、长尾物品；
- 有协同行为与无协同行为的冷启动物品；
- 用户历史长度；
- 语义相近但偏好不同的 hard-negative 场景。

### 7.4 关键消融

建议按以下顺序完成：

1. Baseline；
2. + Sinkhorn；
3. + metric distance；
4. + collaborative regularization；
5. + Support Head；
6. + prefix alignment loss；
7. + entropy-adaptive lambda；
8. 完整 RDSID-Rec。

如果 Recall/NDCG 提升但 Target Prefix Survival 不提升，说明 Support Head 可能只是后处理重排；如果 Prefix Survival 提升但整体指标不提升，需要检查支持分支是否过度偏向热门物品。

---

## 8. 在 RecBoard 上的落地顺序

### Phase 1：复现基线

1. 固定当前 SDQ-403 的 tokenizer、SID 长度和数据切分；
2. 复现 Decoder-only baseline 的 Recall/NDCG；
3. 记录现有 SID 的 codebook 使用率和目标 prefix survival。

### Phase 2：实现支持分支

1. 从已生成 SID 构建 prefix trie；
2. 为每个用户—物品训练 Support Head；
3. 先只做推理阶段 prefix rerank，确认支持分数是否有独立预测能力；
4. 再加入 L_prefix 进行联合训练。

### Phase 3：接入方向⑦的 Tokenizer 组件

1. 在现有 tokenizer 中加入 Sinkhorn assignment；
2. 加入 metric distance loss；
3. 加入协同正则；
4. 比较 tokenizer 改进与支持分支改进的独立贡献。

### Phase 4：扩展与论文实验

1. 头部/长尾/冷启动分组；
2. 不同 SID 长度和 codebook 大小；
3. 不同 beam size 和推理延迟；
4. 与 ISD 类 inference-only guidance 对比；
5. 固化随机种子，至少重复 3 次并报告均值和标准差。

可优先复用工程中已有的 SDQ、CoFiRec、Pctx 和 tokenizer 实现，先把 Support Head 做成独立模块，避免第一版同时重写所有生成器。

---

## 9. 预期贡献与风险

### 9.1 预期贡献

如果实验成立，论文贡献可以表述为：

1. 提出一个角色解耦的 SID 生成框架，让 SID 负责语义组织、Support Head 负责用户条件的 prefix 支持；
2. 提出训练时 prefix alignment，使用户支持信号进入 Decoder-only 的中间 token 决策；
3. 把 Sinkhorn 平衡分配、度量量化和协同正则放入同一个可验证的 tokenizer 训练流程；
4. 用 target prefix survival、final-token retention 和 codebook utilization 解释生成式推荐的性能变化。

### 9.2 主要风险

- 支持分支过度热门化：加入 popularity-aware negative sampling、温度和长尾分组指标；
- Support Head 泄漏标签：只使用训练历史构造支持信号，验证集/测试集点击不能参与 trie 统计；
- Sinkhorn 牺牲语义邻近性：同时监控 reconstruction、neighbor recall 和 SID collision；
- 协同正则伤害冷启动：对无交互物品关闭或减小协同门控；
- prefix score 计算过慢：预计算每个 batch 的 prefix mass，或只在 beam 中计算合法子树；
- 提升来自参数量而非方法：所有对比方法保持 backbone、训练步数和负样本预算一致。

---

## 10. 最小可发表版本（MVP）

如果时间有限，建议只实现下面三项：

1. 复现 SDQ-403 + Decoder-only baseline；
2. 增加 Support Head，并在 SID trie 上计算 prefix support mass；
3. 加入 L_prefix，比较“只推理重排”和“训练时 prefix 对齐”。

这三个实验就能直接回答本方向最核心的问题：

> 用户条件的 prefix 支持是否应该在训练阶段进入 Semantic-ID 生成器，而不是等生成完成后再修正？

Sinkhorn、度量距离和协同正则可以作为第二阶段增强，并通过消融证明它们对 codebook 稳定性和推荐性能的独立作用。

---

## 11. 参考文献

1. Zheng et al. Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation (LC-Rec). [arXiv:2311.09049](https://arxiv.org/abs/2311.09049)
2. Li et al. End-to-End Generative Retrieval for Recommendation with Semantic IDs (ETEGRec). [arXiv:2409.05546](https://arxiv.org/abs/2409.05546)
3. Generative Recommendation with Semantic IDs: A Practitioner’s Handbook. [arXiv:2507.22224](https://arxiv.org/abs/2507.22224)
4. Differentiable Semantic IDs for Generative Recommendation (DIGER). [arXiv:2601.19711](https://arxiv.org/abs/2601.19711)
5. UniGRec: Unified Generative Recommendation with Soft Identifiers. [arXiv:2601.17438](https://arxiv.org/abs/2601.17438)
6. R3-VAE: Reference-anchored Representation Regularization for Generative Recommendation. [arXiv:2604.11440](https://arxiv.org/abs/2604.11440)
7. Understanding Semantic IDs: From Item Representation to Item Selection in Generative Recommendation. [arXiv:2607.24995](https://arxiv.org/abs/2607.24995)
8. What Makes a Good Semantic ID for Generative Recommendation? [arXiv:2609.24430](https://arxiv.org/abs/2609.24430)


## 7. 当前可运行首版：RDSID-PrefixSupport

为先验证方向⑦的核心假设，已在 RecBoard 上实现一个不改变基线 SID 词表的首版：

- 复用 `SDQ-403/logs/SDQ/<dataset>/vae/sid_vocab.json`，保证 Tokenizer 与已有 SDQ/T5 结果一致。
- T5 的训练、数据划分、随机种子、beam size 和评估指标保持 RecBoard 协议：`seed=2025`、`epochs=200`、`num_beams=30`、full ranking、`NDCG@10` 选最优。
- 从训练序列统计转移 `item -> SID prefix` 的支持质量。
- 对每个用户历史取最近 5 个物品，使用指数衰减聚合各级 SID 前缀的条件支持质量。
- 在 constrained beam 生成的候选上，将标准序列分数与归一化 prefix-support 分数相加，默认权重 `0.20`。

代码：

```text
SDQ-403/train_t5_rdsid.py
```

串行实验脚本：

```text
我的方向/7/run_rdsid_serial.sh
```

实验输出目录：

```text
我的方向/7/runs/rdsid_serial_20260924/
```

该首版用于验证“用户条件 prefix support 能否减少 SID 生成的错误前缀过滤”。完整论文版还需要把 prefix 对齐损失和可学习 Support Head 注入训练阶段；本轮先使用推理阶段的可复现实验隔离该因素，避免同时重训 Tokenizer 与 Decoder 导致比较不清楚。
