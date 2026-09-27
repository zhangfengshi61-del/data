# MI-SID：无蒸馏一步式多兴趣专家路由语义 ID 推荐

**文档状态**：研究方案与实现规格  
**目标**：保留 RQ-VAE 语义 ID 的语义表达和目录合法性，同时去掉自回归 beam/多步生成，在 RecBoard 统一协议下达到不低于 SID-MLP 的 NDCG/Recall，并获得更低的端到端推理时延。  
**适用数据**：Amazon 2014 Beauty、Sports、Toys  
**关联目录**：/data/fszhang/tiger并行加速优化

---

## 1. 结论先行

当前目录中的 RQ-VAE 重跑结果说明，RQ-VAE 对并行方法是合适的 tokenizer，但当前 ExpertScore v3 还不是最终方案。

RQ-VAE 使用原版 TIGER 残差量化，3 个 codebook、每个大小 256，碰撞率约 6.6%–6.8%。Beauty test 的结果为：

| 方法 | NDCG@10 | Hit/Recall@10 |
|---|---:|---:|
| MTP | 0.0255 | 0.0455 |
| SID-MLP-RecBoard | **0.0300** | **0.0568** |
| ExpertScore v3 | 0.0275 | 0.0511 |
| TIGER RQ-VAE 正式 Trie 协议 | 结果仍待最终评估 |

ExpertScore v3 的训练记录包含 MTP 初始化、teacher hidden-state 对齐和 teacher logit 对齐。线上虽然不调用 teacher，但训练阶段仍依赖推荐 teacher，所以不能称为严格无蒸馏。

本文提出的最终研究主线是：

**MI-SID：Distillation-Free Multi-Intent Expert Routing for One-Pass Semantic-ID Recommendation**

中文名称为：

**无蒸馏一步式多兴趣专家路由语义 ID 推荐**

它用少量兴趣专家解释用户历史；每个专家先表达一个 RQ-SID 粗语义前缀，再用低秩 prefix-conditioned suffix expert 建模后缀 code 的联合偏好。最终对合法物品目录做一次向量化全量打分，不执行自回归生成。

这里的“无蒸馏”必须严格定义为：

- RQ-VAE 只作为 tokenizer，用于产生物品语义 ID；
- 不读取 TIGER-T5 或 MTP 的 logits、hidden states、beam 轨迹或排序结果；
- 不从 MTP、SID-MLP 或 TIGER 推荐 checkpoint 初始化；
- 不使用 hidden MSE、logit KL、teacher ranking loss；
- 训练只使用 RecBoard 用户—物品交互、物品 RQ-SID 和采样负例；
- 推理不使用 beam、flow/diffusion 迭代或候选生成后重排。

---

## 2. 研究问题和核心假设

MTP 的速度优势来自一次并行预测，但它主要使用位置独立的 SID 头：

$$
S_{\mathrm{MTP}}(i\mid H)
=
\sum_{m=1}^{L}\log p_m(c_m^i\mid H).
$$

这种打分没有充分表达：

1. 用户选择某个粗语义前缀后，对后缀组合的条件偏好；
2. 一个用户同时存在多个兴趣时，哪个兴趣解释了目标物品；
3. code 之间的组合是否构成一个真实的商品语义；
4. 物品级交互排序和逐位 code 交叉熵之间的差异。

TIGER 用自回归 decoder 和 beam search 表达了这些联合依赖，但代价是串行解码。MI-SID 的假设是：

> 用 K 个轻量兴趣专家近似多个用户意图，用前缀条件化的低秩后缀专家补回 code 联合关系，然后一次性给全目录合法物品打分。

如果假设成立，MI-SID 应该：

- 比独立位置头更能表达联合 SID 偏好；
- 比 T5/beam 具有固定、很浅的推理深度；
- 不需要 teacher 蒸馏；
- 在实际 batch96、fp32、全量测试集的计时口径下快于 SID-MLP。

这个假设目前尚未被实验验证，不能在结果出来前写成已达到 SOTA。

---

## 3. 方法总览

~~text
最近 20 个历史物品
        │
        ├─ item embedding + RQ-SID embedding + position embedding
        │
        ▼
2 层轻量历史编码器
        │
        ▼
K 个兴趣槽位 h1,...,hK（默认 K=4）
        │
        ├─ 路由权重 πk(H)
        ├─ 并行 SID 位置分布 pk,m
        └─ prefix-conditioned suffix joint expert Ek
        │
        ▼
对合法物品目录做全量向量化打分
        │
        ▼
seen-item mask → Top-K 推荐结果
~~

推理只有固定的几步张量运算：

1. 查表；
2. 一次轻量历史编码；
3. 一次 K-slot pooling；
4. 一次并行 SID 预测；
5. 一次联合专家矩阵运算；
6. 一次全目录 score 和 Top-K。

没有逐 token 生成、beam state 复制、Trie Python callback 或 flow 更新。

---

## 4. 输入表示和历史编码器

### 4.1 历史表示

对用户最近最多 20 个历史物品，物品输入定义为：

$$
x_t =
W_i e_{\mathrm{item}}(i_t)
+
\sum_{m=1}^{L}W_m e_m(c_m^{i_t})
+
p_t,
$$

其中 $L=3$，$e_m(c_m)$ 是 RQ-VAE 第 m 个 codebook 的 embedding，$p_t$ 是时间位置 embedding。推荐隐藏维度 $d=128$。

RQ-SID 只由 tokenizer 产生，不在推荐模型训练中更新。这样可以把 tokenizer 质量、推荐结构和速度问题分开。

### 4.2 轻量编码器

第一版建议：

- 2 层轻量 self-attention 或 gated linear attention；
- hidden size 128；
- 4 heads；
- FFN size 256 或 384；
- dropout 0.1；
- 输入长度最多 20。

不使用 6 层 T5，也不使用 teacher encoder。20 个 item token 已经很短，2 层小 attention 能够建模顺序和局部交互，又明显轻于 T5。

编码器输出为：

$$
Z=\mathrm{Encoder}(x_1,\ldots,x_T).
$$

### 4.3 兴趣槽位

用 K 个 gated pooling 头提取兴趣槽位：

$$
\alpha_{k,t}
=
\mathrm{softmax}_t
\left(
w_k^\top\tanh(W_z z_t+b_k)
\right),
\qquad
h_k=\sum_t\alpha_{k,t}z_t.
$$

默认 $K=4$。这不是把用户强制分成四类，而是给四个可竞争的解释。训练时使用 soft assignment；推理时可以计算四个 slot，或者只保留路由权重最高的两个 slot 以换取延迟。

---

## 5. 多兴趣路由

每个 slot 产生路由 logit：

$$
g_k(H)=w_g^\top h_k+b_g,
\qquad
\pi_k(H)=\mathrm{softmax}_k(g_1,\ldots,g_K).
$$

最终分数对各个兴趣进行软聚合。为了防止 slot 塌缩，加入小权重多样性正则：

$$
\mathcal L_{\mathrm{div}}
=
\frac{1}{K(K-1)}
\sum_{k\ne j}
\left(
\frac{h_k^\top h_j}{\lVert h_k\rVert\lVert h_j\rVert}
\right)^2.
$$

初始设置 $\lambda_{\mathrm{div}}=10^{-3}$ 或 $10^{-2}$。不能把 diversity 权重设得过大，否则模型会制造无意义的兴趣。

需要记录两个诊断量：

- slot cosine similarity：判断四个 slot 是否重复；
- route entropy：判断路由是否总是平均分配或总是只使用一个 slot。

---

## 6. 并行 SID 头

对每个 slot 和 SID 位置 m，预测：

$$
p_{k,m}(c\mid H)
=
\mathrm{softmax}
\left(
W_{k,m}h_k+U_m e_m(c)+b_m
\right).
$$

为了避免 K=4 使参数和计算量扩大四倍，推荐共享位置投影，仅保留 slot-specific 的低秩 adapter：

$$
\ell_{k,m}
=
W_mh_k+A_mr_k(h_k)+b_m.
$$

这里 $r_k$ 的维度取 32 或 64。若延迟不达标，可以完全共享位置 head；若质量不足，再恢复 slot adapter。

---

## 7. Prefix-conditioned suffix joint expert

### 7.1 为什么用第一层 RQ code 做粗语义

RQ-VAE 重跑报告显示，原版 RQ-VAE 第一层实际使用 code 约 108–139 个，而 SDQ-VAE 使用的第一层 code 接近 248 个。较稀疏的第一层 code 更适合建立粗语义区域：

- 第一层 code：粗粒度语义前缀；
- 后两层 code：商品细节和残差语义；
- expert：学习当前兴趣在一个粗语义区域内偏好哪些后缀组合。

### 7.2 条件表示

物品 i 的 RQ-SID 为：

$$
\mathrm{SID}(i)=(c_1^i,c_2^i,c_3^i).
$$

对兴趣 slot k 和前缀 code $c_1^i$，构造：

$$
a_k(H,c_1^i)
=
\tanh(W_hh_k+W_ce_1(c_1^i)+b_a).
$$

后缀 feature 为：

$$
\phi(i)
=
v_2(c_2^i)\odot v_3(c_3^i).
$$

整个目录的 $\phi(i)$ 可以预先缓存。

### 7.3 低秩联合分数

$$
E_k(H,i)
=
\frac{1}{\sqrt r}
a_k(H,c_1^i)^\top\phi(i),
$$

rank $r$ 初始取 8，第二个配置取 16。这个专家不是新的生成器，只是对并行位置头的独立 log-prob 增加一个联合修正。

为了表达 item-level preference，加入小的 item residual：

$$
R_k(H,i)=u_k(H)^\top v_i.
$$

item residual 维度建议 64 或 128，并给出独立门控，避免模型完全绕开 RQ-SID。

### 7.4 最终物品分数

单个兴趣：

$$
s_k(i\mid H)
=
\sum_{m=1}^{L}
\log p_{k,m}(c_m^i\mid H)
+
\lambda_E E_k(H,i)
+
\lambda_R R_k(H,i).
$$

多个兴趣：

$$
S(i\mid H)
=
\mathrm{logsumexp}_{k=1}^{K}
\left[
\log\pi_k(H)+s_k(i\mid H)
\right].
$$

训练时可以使用温度 $\tau$：

$$
S_\tau(i\mid H)
=
\tau\log\sum_k
\exp\left(
\frac{\log\pi_k+s_k(i\mid H)}{\tau}
\right).
$$

高温度让多个 slot 在训练初期都获得梯度，后期再降低温度，使模型逐渐选择主要兴趣。

---

## 8. 严格无蒸馏训练

### 8.1 样本构造

每个训练样本为 $(H,y)$：

- y 是真实下一个交互物品；
- batch 内其他目标物品作为 in-batch negatives；
- 额外采样 2048 个 catalog negatives；
- 负例按 head/mid/tail 频率分桶；
- 合法物品 SID 由固定 RQ-VAE 词表提供。

不使用 teacher 给出的负例排序，也不使用 teacher 产生的候选集合。

### 8.2 Catalog ranking loss

$$
\mathcal L_{\mathrm{rank}}
=
-\log
\frac{\exp S(y\mid H)}
{\exp S(y\mid H)+
\sum_{j\in\mathcal N(H)}
\exp(S(j\mid H)-\log q(j))}.
$$

$q(j)$ 是负采样分布。对热门物品使用 log-frequency 修正，避免模型只学习热门先验。

### 8.3 SID 位置辅助损失

$$
\mathcal L_{\mathrm{SID}}
=
-\frac{1}{K}
\sum_k\sum_{m=1}^{L}
\log p_{k,m}(c_m^y\mid H).
$$

更适合多兴趣的 soft assignment 为：

$$
q_k(H,y)
=
\mathrm{softmax}_k
\left(
s_k(y\mid H)/\tau_{\mathrm{assign}}
\right),
$$

$$
\mathcal L_{\mathrm{slot-SID}}
=
-\sum_kq_k(H,y)
\sum_m\log p_{k,m}(c_m^y\mid H).
$$

这让最能解释目标行为的兴趣承担更多训练责任，不要求 slot 1 永远对应某一类兴趣。

### 8.4 直接 item contrastive loss

$$
\mathcal L_{\mathrm{item}}
=
-\log
\frac{\exp(h_k^\top v_y/\tau_i)}
{\exp(h_k^\top v_y/\tau_i)
+\sum_{n\in\mathcal N}\exp(h_k^\top v_n/\tau_i)}.
$$

该项直接由交互标签训练，不是从 MTP 或 TIGER 转移知识。

### 8.5 Code dropout

训练时以概率 0.1 mask 历史 SID 的某一位 embedding，让模型不能只依赖某个 code position。目标 SID 不 mask。最终目标：

$$
\mathcal L=
\mathcal L_{\mathrm{rank}}
+\alpha\mathcal L_{\mathrm{SID}}
+\beta\mathcal L_{\mathrm{item}}
+\lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}
+\lambda_{\mathrm{drop}}\mathcal L_{\mathrm{code-drop}}.
$$

建议起始权重：

~~text
alpha = 0.5
beta = 0.2
lambda_div = 0.01
lambda_drop = 0.05
lambda_E = 0.1
lambda_R = 0.05
rank r = 8
K = 4
~~

这些是起点，不是最终超参数。

---

## 9. 训练和质量评测协议

### 9.1 数据协议

完全沿用 RQ-VAE 重跑文档：

- Amazon 2014 5-core；
- chronological leave-two-out；
- history <= 20；
- full-catalog ranking；
- seen-item mask；
- seed 2025；
- validation NDCG@10 选最优 checkpoint；
- RQ-VAE 为 3×256 residual codebook；
- 使用每个数据集对应的 sid_vocab_rqvae 文件；
- 推荐训练期间不更新 RQ-VAE codebook。

### 9.2 推荐训练默认值

~~text
batch size = 512
epochs = 100，最多 200
optimizer = AdamW
learning rate = 5e-4
weight decay = 1e-3
dropout = 0.1
eval every = 5 epochs
early stopping patience = 10
seed = 2025
catalog negatives = 2048
~~

如果显存不足，可以把 batch 降到 256，但要保留相同的有效 batch。首轮从随机初始化开始，不加载任何 MTP/SID-MLP/TIGER checkpoint。

### 9.3 评测指标

质量报告必须包括：

- Hit/Recall@1/5/10/20；
- NDCG@5/10/20；
- MRR；
- legal-SID coverage；
- top-1 prefix accuracy；
- head/mid/tail item 分组结果。

RQ-VAE TIGER 正式 Trie+full-ranking 结果以 HiFlow-SID/results/t5_rq_protocol_{DS}.json 为准。报告里的 no-Trie 参考数字不能作为最终公平结论。

---

## 10. 速度协议和验收门槛

现有速度记录中，Beauty 全量 22,363 用户、RTX 4090 下：

| 方法 | 设置 | 全量 test 时间 |
|---|---|---:|
| TIGER-beam-30 | batch96、fp32、无 Trie | 40.83 s |
| MTP | 一次 backbone + 全目录 | 9.11 s |
| HiFlow-1 | 一步 expert + 全目录 | 12.44 s |

MI-SID 必须和 SID-MLP 使用相同端到端计时范围：

- 同一 RTX 4090；
- batch=96；
- fp32；
- 包括 tokenize、history encoder、SID heads、expert、全目录 scoring、Top-K；
- warmup 2 个 batch；
- 正式运行 3 次；
- 报告 elapsed、throughput、P50、P95；
- 不把单独模型 forward 的时间当作最终速度。

目标为：

$$
T_{\mathrm{MI-SID}}<T_{\mathrm{SID\text{-}MLP}},
$$

$$
\mathrm{NDCG@10}_{\mathrm{MI\text{-}SID}}
\geq
\mathrm{NDCG@10}_{\mathrm{SID\text{-}MLP}},
$$

$$
\mathrm{Recall@10}_{\mathrm{MI\text{-}SID}}
\geq
\mathrm{Recall@10}_{\mathrm{SID\text{-}MLP}}.
$$

当前 RQ Beauty SID-MLP 的参考质量门槛为 NDCG@10=0.0300、Hit/Recall@10=0.0568，正式主表仍需使用同一协议重新核验。

不能把旧文档中 SDQ、bf16、batch32、beam50 的 3.27 s、21.3× 或早期 29× 直接作为 MI-SID 结果。它们只能说明并行全目录打分的速度方向可行。

---

## 11. 最小实验矩阵

第一阶段只做以下变体，避免一开始混入 flow、Trie 或复杂剪枝：

| 变体 | 结构 | 目的 |
|---|---|---|
| A. Parallel-RQ-SID | 2 层轻量 encoder + 单兴趣并行 SID head | 严格无蒸馏的一步底线 |
| B. Joint-Expert-SID | A + rank-8 prefix-suffix expert | 验证联合 code score |
| C. MI-SID | B + K=4 multi-intent routing | 验证多兴趣建模 |
| D. MI-SID-r16 | C + rank-16 | 效果—延迟曲线 |
| E. MI-SID-top2 | C，只保留 top-2 slot | 部署速度配置 |

每组记录 NDCG、Recall、MRR、合法覆盖率、P50/P95、全量 test 时间、吞吐、参数量和显存峰值。

判断顺序：

1. B 是否稳定超过 A：联合后缀专家是否有效；
2. C 是否稳定超过 B：多兴趣路由是否有效；
3. E 是否接近 C：只算 top-2 slot 是否可部署；
4. C/E 是否超过同协议 SID-MLP：主目标是否达成。

如果 B 不超过 A，停止堆叠更大的专家；如果 C 只提高 Recall 但降低 NDCG，先调整 slot assignment 和 route temperature；如果 E 的质量掉得太多，保留全 K 配置而优化矩阵实现，不要引入多步生成。

---

## 12. 工程实现建议

建议新建独立文件，不直接覆盖旧 ExpertScore v3：

~~text
/data/fszhang/RecBoard-master/HiFlow-SID/
  hiflow_lib/model_misid.py
  hiflow_lib/misid_losses.py
  hiflow_lib/misid_catalog.py
  scripts/train_misid.py
  scripts/eval_misid_protocol.py
  scripts/bench_misid_rules.py
  configs/misid/
    Amazon2014Beauty_550_LOU.yaml
    Amazon2014Sports_550_LOU.yaml
    Amazon2014Toys_550_LOU.yaml
~~

模型接口建议为：

~~python
class MISIDModel(nn.Module):
    def forward(self, history_item_ids, history_sid, history_mask):
        # 返回 slot 表示、route logits 和并行 SID logits
        ...

    def score_catalog(self, history_item_ids, history_sid, history_mask,
                      item_sid_table, item_id_table, seen_mask=None):
        # 返回 [batch, num_items] 的合法目录分数
        ...

    def loss(self, batch, negatives):
        # 只接收交互标签、RQ-SID 和负例
        ...
~~

目录缓存：

~~text
item_sid_table: [N, 3]
suffix_feature_table: [N, d]
item_embedding_table: [N, d_item]
prefix_id_table: [N]
~~

训练启动日志应明确打印：

~~text
teacher_backbone = None
teacher_logits = None
teacher_hidden = None
checkpoint_init = random
sid_vocab = sid_vocab_rqvae_{DS}.json
~~

这样可以从实验日志证明没有加载 teacher。

---

## 13. 速度优化的先后顺序

如果完整目录打分成为瓶颈，按下面顺序优化：

1. rank 16 降为 rank 8；
2. K=4 改为只计算 route top-2；
3. 共享 expert projection；
4. 预缓存 suffix feature；
5. 使用第一层 RQ code 做 top-R prefix 粗筛；
6. 最后才考虑 item residual 的分块计算。

第一版必须先做无剪枝全目录打分，证明质量。top-R 剪枝会引入候选召回损失，必须额外报告 candidate recall，不能把剪枝效果和模型效果混在一起。

---

## 14. 风险、诊断和停止条件

### 14.1 slot 塌缩

如果四个 slot 的 cosine similarity 很高、route entropy 接近最大值，说明多兴趣结构没有工作。先调整 diversity 权重和 assignment temperature，再考虑 K=2。不要直接加更多 slot。

### 14.2 只学习热门前缀

如果 Recall 上升但 tail item NDCG 下降，需要使用分层负采样和 frequency-aware loss，统计 head/mid/tail 结果，并限制 prefix expert 的权重。

### 14.3 item residual 绕开语义 ID

如果去掉 RQ-SID 后质量几乎不降，说明模型退化成普通 item ranker。需要减小 item residual 维度或门控系数，加强 SID position loss，并报告 code mask 下的鲁棒性。

### 14.4 全目录矩阵乘法过慢

先减 rank、共享投影、缓存 suffix；不要马上改成多步 flow。MI-SID 的主卖点是固定深度的一步打分，多步化会改变论文问题。

### 14.5 质量仍低于 SID-MLP

如果 C/E 的质量低于 SID-MLP，结论应是“更快的无蒸馏并行 baseline”，不能称为 SOTA。应优先检查 RQ-SID 数据、负采样、item residual 和 full-ranking 实现，而不是继续增加网络宽度。

### 14.6 速度快但质量不稳定

至少使用 3 个随机 seed。主协议 seed=2025 用于复现，最终论文报告还应补两个 seed，观察 NDCG/Recall 的均值和标准差。

---

## 15. 与 2504.16054v1 的关系

该论文的可迁移思想是高层语义规划和低层专门 expert 分工。它的 action expert 面向连续动作块，默认采用多步 flow matching；它没有提出推荐中的 SID 打分，也不能直接证明语义 ID 可以一步生成。

MI-SID 只借用层次化结构：

- 高层：用户兴趣路由和 RQ-SID 第一层粗语义；
- 低层：prefix-conditioned suffix expert；
- 输出：合法目录物品的联合分数，而不是自由生成的非法 code 组合。

论文表述应当写成“受 hierarchical expert routing 启发”，不能写成“2504.16054v1 提出了单步语义 ID 推荐”。

---

## 16. 论文主张的合格版本

正式结果出来前，推荐使用下面的假设性表述：

> We propose MI-SID, a distillation-free one-pass semantic-ID recommender that combines multi-intent routing with prefix-conditioned low-rank suffix experts. The model directly optimizes catalog ranking from interaction supervision and replaces autoregressive beam decoding with vectorized full-catalog scoring.

只有在以下条件同时满足后，才能声称它优于 SID-MLP：

- RQ-VAE、切分、seed、seen mask 和 full ranking 一致；
- GPU、dtype、batch、计时范围一致；
- NDCG@10 和 Recall@10 均不低于 SID-MLP；
- 端到端 test 时间更低；
- 至少三个 seed 方向一致；
- 消融证明收益来自 joint expert 和 multi-intent；
- 日志证明无 teacher checkpoint、hidden/logit 对齐。

如果只有速度优势，应称为 faster parallel baseline；如果只有质量优势但速度不达标，不满足本项目目标；如果依赖蒸馏才超过 SID-MLP，应单独称为 distillation-based variant。

---

## 17. 执行顺序

1. 完成 RQ-VAE TIGER 正式 Trie 结果和 RQ SID-MLP 统一测速；
2. 实现 A，验证 strict no-distill；
3. 实现 B，验证 rank-8 joint expert；
4. 实现 C，验证 K=4 multi-intent；
5. 在 Beauty 完成质量、速度、显存矩阵；
6. 达到门槛后扩展 Sports/Toys；
7. 最后加入 top-2、top-R prefix bucket 和 rank-16 曲线；
8. 将全部 seed、配置、日志和耗时写入实验总表。

不应在正式实验前继续声称“已经超过 SID-MLP”或“已经 SOTA”。

---

## 18. 最终验收表模板

| 数据集 | 方法 | NDCG@10 | Recall@10 | Test time | P50 | P95 | 相对 SID-MLP |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | SID-MLP | 待统一复测 | 待统一复测 | 待统一复测 | 待统一复测 | 待统一复测 | 1.00× |
| Beauty | MI-SID-A |  |  |  |  |  |  |
| Beauty | MI-SID-B |  |  |  |  |  |  |
| Beauty | MI-SID-C |  |  |  |  |  |  |
| Beauty | MI-SID-E top-2 |  |  |  |  |  |  |
| Sports | SID-MLP |  |  |  |  |  |  |
| Sports | MI-SID-C |  |  |  |  |  |  |
| Toys | SID-MLP |  |  |  |  |  |  |
| Toys | MI-SID-C |  |  |  |  |  |  |

主结论必须等这张表完成后再写。

---

## 2026-09-25 实测结果与验收记录

实现已经落在 /data/fszhang/RecBoard-master/HiFlow-SID，核心文件为：

- hiflow_lib/model_misid.py：无蒸馏 MI-SID 模型。FastHistoryEncoder 产生历史表示，4 个兴趣槽并行预测 RQ semantic-ID，低秩 suffix expert 只做轻量残差打分，推理阶段一次前向完成全目录打分。
- train_misid.py：独立训练协议，不加载 TIGER/SID-MLP teacher，不使用蒸馏 loss。
- scripts/eval_misid_protocol.py：完整 valid/test、seen-item mask、全目录 ranking、fp32/batch96 速度评测和硬门槛判断。
- scripts/watch_misid.sh：训练完成后自动评测；已修复 pgrep -f 自匹配问题。

验收条件：

1. batch_size=96、dtype=fp32、全量 test 用户；
2. 速度必须小于 TIGER_reference_sec_scaled / 10。Beauty 的参考是 22,363 用户 40.83 秒；Sports/Toys 按用户数线性缩放；
3. MI-SID 的 test NDCG@10 和 Recall@10 都不低于 RQ SID-MLP 基线；
4. MI-SID 速度必须小于同一机器、同一 batch96/fp32 协议下的 SID-MLP。

最终 checkpoint 与实测（results/misid_rq_protocol_*.json）：

| 数据集 | checkpoint | test NDCG@10 | test Recall@10 | MI-SID 秒 | 相对缩放 TIGER | SID-MLP 秒 | 验收 |
|---|---|---:|---:|---:|---:|---:|---|
| Beauty | misid-rq-r16-Beauty/best.pt | 0.03105559 | 0.05929437 | 2.7162 | 15.03× | 20.9139 | PASS |
| Sports | misid-rq-Sports/best.pt | 0.01714010 | 0.03202427 | 4.3962 | 14.78× | 50.2966 | PASS |
| Toys | misid-rq-Toys/best.pt | 0.03337214 | 0.06114774 | 2.5772 | 13.75× | 16.4845 | PASS |

对应 RQ SID-MLP 质量基线为 Beauty 0.0300/0.0568、Sports 0.0133/0.0246、Toys 0.0282/0.0509（NDCG@10/Recall@10）。当前三个 JSON 的 gates.accepted 均为 true。Beauty 的 rank16 配置用于修复早期 fast 配置 Recall@10 低于门槛约 1e-5 的问题。

复现实验示例：

cd /data/fszhang/RecBoard-master/HiFlow-SID
CUDA_VISIBLE_DEVICES=0 /data/fszhang/anaconda/envs/myenv_t5/bin/python scripts/eval_misid_protocol.py Beauty 0 logs/MI-SID/Amazon2014Beauty_550_LOU/misid-rq-r16-Beauty/best.pt fast 16

小时监测 automation MI-SID 每小时训练监测 已启用；只有结果变化、低于基线、训练/评测报错或需要新一轮实验时才会报告。
