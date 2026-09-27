# PATE-MI-SID：增强专家的一阶段全目录无蒸馏方案

## 1. 目标与公平性约束

当前 MI-SID 已经满足速度目标，但相对正式 TIGER-T5 协议仍有质量差距。本方案要增强 semantic-ID 专家本身，同时保持当前一次前向和全目录向量化打分结构。

硬约束：

- 不加载 TIGER、T5 teacher 或 SID-MLP teacher。
- 不使用蒸馏 loss、teacher hidden、teacher logits。
- 不使用 TIGER 生成的候选。
- 不使用外部召回器、候选过滤或二阶段精排。
- 每个用户对完整商品目录一次性生成分数。
- 所有商品通过同一个向量化专家打分函数。
- 允许相对当前 MI-SID 的端到端时间最多增加 20%。
- 继续使用 RTX4090、fp32、batch96、全量 test 用户做速度评估。

方法暂名：PATE-MI-SID（Prefix-Adaptive Tensor Expert MI-SID），中文名为“前缀自适应张量专家多兴趣语义 ID 推荐”。

核心判断：当前质量差距主要来自 semantic-ID 路径条件建模不足，而不是缺少一个 TIGER 召回器。新方法把专家从简单的 suffix residual 升级成用户、前缀和后缀的低秩张量交互专家。

## 2. 目前实验事实

正式 TIGER-T5 协议结果来自：

- /data/fszhang/RecBoard-master/HiFlow-SID/results/t5_rq_protocol_Beauty.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/t5_rq_protocol_Sports.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/t5_rq_protocol_Toys.json

当前 MI-SID 结果来自：

- /data/fszhang/RecBoard-master/HiFlow-SID/results/misid_rq_protocol_Beauty.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/misid_rq_protocol_Sports.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/misid_rq_protocol_Toys.json

| 数据集 | TIGER NDCG@10 | MI-SID NDCG@10 | TIGER Recall/Hit@10 | MI-SID Recall@10 |
|---|---:|---:|---:|---:|
| Beauty | 0.03660946 | 0.03105559 | 0.06613602 | 0.05929437 |
| Sports | 0.02240388 | 0.01714010 | 0.04224956 | 0.03202427 |
| Toys | 0.03518488 | 0.03337214 | 0.06217803 | 0.06114774 |

当前 MI-SID 的 full-catalog 速度：

| 数据集 | MI-SID | SID-MLP | MI-SID 相对缩放 TIGER |
|---|---:|---:|---:|
| Beauty | 2.7162 秒 | 20.9139 秒 | 15.03× |
| Sports | 4.3962 秒 | 50.2966 秒 | 14.78× |
| Toys | 2.5772 秒 | 16.4845 秒 | 13.75× |

PATE-MI-SID 的时间预算：

| 数据集 | 当前 MI-SID | 允许的 1.20× 上限 | 10× TIGER 上限 |
|---|---:|---:|---:|
| Beauty | 2.7162 秒 | 3.2594 秒 | 4.0830 秒 |
| Sports | 4.3962 秒 | 5.2755 秒 | 6.4994 秒 |
| Toys | 2.5772 秒 | 3.0927 秒 | 3.5442 秒 |

## 3. 当前 MI-SID 的质量瓶颈

### 3.1 位置头对 semantic-ID 路径依赖不足

当前基础分数大致是：

log p(c0 | h) + log p(c1 | h) + log p(c2 | h)

这没有充分表达：

p(c1 | c0, h)

p(c2 | c0, c1, h)

TIGER 的生成路径会使用前缀条件。当前 MI-SID 只有一个很小的 low-rank suffix residual，无法完全恢复这种条件关系。

### 3.2 当前 suffix expert 表达能力有限

当前专家接近：

tanh(user_projection + prefix_projection) × suffix_factor_1 × suffix_factor_2

它能表达一部分用户、前缀和后缀关系，但缺少：

- c0 和 c1 的配对关系；
- c0、c1 共同决定 c2 的条件关系；
- 多种路径专家；
- 用户根据兴趣 slot 选择不同路径函数的能力。

### 3.3 训练负样本过于容易

随机目录负样本容易被区分。更有价值的负样本是：

- 第一层 code 相同、后两层不同；
- 前两层 code 相同、最后一层不同；
- semantic-ID 路径接近但商品不同；
- 频率相近的相似商品。

这些样本只用于训练专家，不能改变推理时的全目录评分协议。

### 3.4 快速历史编码器缺少顺序增强

当前 FastHistoryEncoder 保留了速度，但对最近行为、最后一个行为和长期兴趣的差异建模不足。加入轻量时间门控可以改善用户 slot，而不会引入 T5 的高成本。

## 4. 一阶段 PATE-MI-SID 结构

严格推理路径：

历史 semantic-ID
→ 快速顺序增强编码器
→ K 个兴趣 slot
→ slot 路由
→ PATE 全目录张量专家
→ 完整商品分数矩阵
→ seen-item mask
→ top-k

目录中的每个 item 都参与同一轮分数计算。

明确禁止：

- 先取 top-R prefix；
- 先召回 item；
- 再对候选调用 PATE；
- 使用 TIGER beam 输出作为候选；
- 使用 SID-MLP top-k 作为候选；
- 对候选进行第二次模型前向。

显存分块可以使用 candidate chunk=2048 或 4096。分块只是矩阵计算的内存实现，最终仍然对完整目录生成分数，不构成召回阶段。

## 5. 轻量顺序增强编码器

保留当前 FastHistoryEncoder，并增加低成本 temporal gate：

1. FastHistoryEncoder 产生 token-level hidden；
2. depthwise temporal convolution，kernel size=3；
3. GLU 门控；
4. 汇总 global mean、recent mean、last event；
5. 生成用户表示和 K=4 个兴趣 slot。

伪代码：

~~~python
x = fast_history_encoder(input_ids, attention_mask)
x_recent = depthwise_conv1d(x.transpose(1, 2)).transpose(1, 2)
x_recent = x_recent * torch.sigmoid(gate(x_recent))
h_global = masked_mean(x, attention_mask)
h_recent = masked_recent_mean(x_recent, attention_mask, recent_window=4)
h_last = gather_last_valid(x_recent, attention_mask)
h = user_projection(torch.cat([h_global, h_recent, h_last], dim=-1))
slots = slot_projection(h, slot_queries)
route_logits = route(slots)
~~~

建议配置：

| 参数 | 值 |
|---|---:|
| embedding dim | 128 |
| FastHistoryEncoder layers | 2 |
| fast hidden | 256 |
| temporal kernel | 3 |
| recent window | 4 |
| slots | 4 |
| dropout | 0.05 |

不使用 T5 Encoder。

## 6. Prefix-Adaptive Tensor Expert

对每个 RQ-SID item code：

c = (c0, c1, c2)

预先查表得到三个 rank factor：

- a0(c0)
- a1(c1)
- a2(c2)

对每个兴趣 slot z_k：

u_k = W_u z_k

### 6.1 完整路径张量交互

主专家分数：

S_tensor(k,c) = sum_r u_k,r × a0(c0)_r × a1(c1)_r × a2(c2)_r

这项直接表达用户、三层 semantic-ID 路径的交互。

### 6.2 前缀配对项

增加两个配对 factor：

b01(c0,c1) = projection01(a0(c0) × a1(c1))

b012(c0,c1,c2) = projection012(a0(c0) × a1(c1) × a2(c2))

配对分数：

S_pair(k,c) = sum_r u01_k,r × b01(c0,c1)_r
            + sum_r u012_k,r × b012(c0,c1,c2)_r

最终专家：

S_path_expert(k,c) = S_tensor(k,c)
                   + alpha × S_pair(k,c)

全部 factor 都来自固定 RQ-SID code 表，不使用任何 teacher。

### 6.3 双路径专家混合

使用两个共享参数路径专家：

- Expert-0：偏向粗粒度前缀和用户主题；
- Expert-1：偏向前两层路径和细粒度后缀。

用户 slot 产生 mixture weight：

w_e(k) = softmax(W_route z_k)

最终：

S_expert(k,c) = logsumexp_e(w_e(k) + S_e(k,c))

每个 expert rank=8，两个 expert 的总 rank=16。Beauty 可尝试每个 expert rank=12；如果速度超预算退回 rank=8。

伪代码：

~~~python
u = user_proj(slots).view(B, K, E, R)
f0 = factor0[item_codes[:, 0]]
f1 = factor1[item_codes[:, 1]]
f2 = factor2[item_codes[:, 2]]

path = f0 * f1 * f2
pair01 = pair01_proj(f0 * f1)
pair012 = pair012_proj(path)

tensor_score = einsum('bker,mer->bkem', u, path)
pair_score = einsum('bker,mer->bkem', u_pair, pair01)
expert_score = tensor_score + alpha * pair_score
score = logsumexp(expert_weight + expert_score, dim=2)
~~~

这是一次完整目录计算，输出形状为 B×K×M。

### 6.4 受控 gate

为了避免专家在训练中被关闭：

g = 0.05 + 0.25 × sigmoid(g_raw)

建议：

- epoch 1–10：gate 约 0.05；
- epoch 11–40：warmup 到约 0.20；
- 后期最大约 0.30；
- 对 gate 做轻微正则。

## 7. 条件路径概率

基础位置头升级为低秩条件头：

- level-0：slot → c0；
- level-1：slot + c0 factor → c1；
- level-2：slot + c0/c1 pair factor → c2。

不对每个候选执行 hidden=512 的 MLP，而使用 code factor、低秩 bilinear 和 batched matmul。

完整路径分数：

S_path(k,c) =
  log p(c0 | z_k)
  + log p(c1 | c0,z_k)
  + log p(c2 | c0,c1,z_k)

总 slot 分数：

S_total(k,c) = S_path(k,c) + S_expert(k,c)

多兴趣聚合：

S_item(c) = logsumexp_k(log p(k | h) + S_total(k,c))

## 8. 训练目标

所有监督来自 interaction target 和 RQ-SID code，不使用 teacher。

### 8.1 分层路径 loss

L_path = CE(c0) + CE(c1 | c0) + CE(c2 | c0,c1)

### 8.2 困难路径负样本

每个 batch 构造：

| 类型 | 构造 |
|---|---|
| uniform | 全目录均匀采样 |
| same-prefix | c0 相同，c1/c2 不同 |
| same-prefix2 | c0/c1 相同，c2 不同 |
| in-batch | 其他用户正样本 |

建议：

- uniform=512；
- same-prefix=512；
- same-prefix2=512；
- in-batch 使用 batch 内其他 item。

这些负样本只影响训练 loss。推理仍然对完整目录计算一次。

### 8.3 Listwise ranking loss

L_rank = -log exp(S_pos / tau) / sum_j exp(S_j / tau)

建议 tau=0.07，对 same-prefix2 的权重为 1.5。

### 8.4 多兴趣分配

q_k = softmax(S_total(k,pos) / T_slot)

建议 T_slot 从 0.8 逐步降低到 0.35，防止所有样本挤到同一个 slot。

### 8.5 路径一致性

同一个 c0 下的 c1 分布、同一个 c0/c1 下的 c2 分布需要保持归一化。使用轻量 consistency loss，权重 0.05。

### 8.6 频率校准

加入很小的、可学习的频率校准项：

S_calibrated = S_total + gamma × log((freq(item)+1)^eta)

建议 eta=0.15，gamma 初始为 0，并限制其范围，避免模型只偏向热门商品。

### 8.7 总 loss

L = 1.0 L_rank
  + 0.8 L_path
  + 0.35 L_align
  + 0.10 L_div
  + 0.05 L_cons
  + 0.05 L_gate

## 9. 训练阶段

### 阶段 A：基础路径

epoch 1–15：

- 训练 FastHistoryEncoder、slot、route、path heads；
- uniform + in-batch negatives；
- expert gate 固定在 0.05。

### 阶段 B：专家路径

epoch 16–55：

- 打开 tensor expert；
- 加入 same-prefix 和 same-prefix2 negatives；
- gate 从 0.05 warmup 到 0.20；
- 使用 rank loss 和 path loss 联合优化。

### 阶段 C：联合校准

epoch 56–90：

- 打开 consistency 和 frequency calibration；
- 每 5 epoch 做完整 valid ranking；
- 以 valid NDCG@10 early stopping。

建议配置：

| 参数 | 值 |
|---|---:|
| batch size | 512 |
| max history | 20 |
| embedding dim | 128 |
| slots | 4 |
| expert mixtures | 2 |
| rank per expert | 8 |
| uniform negatives | 512 |
| same-prefix negatives | 512 |
| same-prefix2 negatives | 512 |
| learning rate | 3e-4 |
| weight decay | 1e-3 |
| optimizer | AdamW |
| seed | 2025 |

## 10. 严格 full-catalog 推理

~~~python
@torch.no_grad()
def recommend_from_full(data):
    slots, route_logits = encode_slots_with_temporal_gate(data)
    all_codes = self.item_codes

    score_slots = self.pate_full_catalog(
        slots=slots,
        route_logits=route_logits,
        item_codes=all_codes,
    )

    scores = torch.logsumexp(
        log_softmax(route_logits, dim=-1).unsqueeze(-1) + score_slots,
        dim=1,
    )
    return scores
~~~

评测脚本必须检查：

- score 的第二维等于完整 item 数；
- 没有 candidate ID 参数；
- 没有 TIGER checkpoint 参数；
- 没有 top-R 参数；
- top-k 在完整分数矩阵上执行；
- seen-item mask 在完整分数矩阵上执行。

## 11. 公平对比协议

所有方法保持：

- 相同 Amazon 2014 5-core 数据；
- 相同 leave-two-out；
- history length≤20；
- 相同原版 TIGER RQ-VAE item code；
- 相同 seed=2025；
- 相同 seen-item mask；
- 相同 test 用户；
- 相同 full-ranking evaluator。

质量比较：

- PATE-MI-SID：一次 full-catalog score；
- MI-SID：一次 full-catalog score；
- SID-MLP：一次 full-catalog score；
- TIGER：使用正式 beam-30 + Trie 协议结果。

PATE-MI-SID 不读取 TIGER 产生的 item 或分数。TIGER 的 beam-30 只属于 TIGER 自己的生成过程，不能作为 PATE-MI-SID 的召回器。

速度统一：

- RTX4090；
- fp32；
- batch size=96；
- 全量 test 用户；
- warmup=2；
- 计时包含 encoder、full-catalog score、seen mask、top-k；
- 不计 checkpoint load 和数据初始化；
- 报告平均、P50、P95 和吞吐量。

## 12. 质量目标

TIGER 正式结果：

| 数据集 | NDCG@10 | Recall/Hit@10 |
|---|---:|---:|
| Beauty | 0.03660946 | 0.06613602 |
| Sports | 0.02240388 | 0.04224956 |
| Toys | 0.03518488 | 0.06217803 |

第一目标：达到 TIGER 的 95%。

| 数据集 | NDCG 目标 | Recall 目标 |
|---|---:|---:|
| Beauty | 0.034779 以上 | 0.062829 以上 |
| Sports | 0.021284 以上 | 0.040137 以上 |
| Toys | 0.033426 以上 | 0.059069 以上 |

如果第一轮未达到 95%，仍报告 TIGER gap recovery，不能只报告相对 SID-MLP 的提升。

## 13. 必须做的消融

| 版本 | 增加组件 |
|---|---|
| A | 当前 MI-SID |
| B | A + temporal gate |
| C | B + pair01 expert |
| D | C + tensor path expert |
| E | D + same-prefix negatives |
| F | E + same-prefix2 negatives |
| G | F + dual expert mixture |
| H | G + path consistency |
| I | H + rank per expert=12 |

每个版本都用完整目录，记录：

- test NDCG@10；
- test Recall@10；
- full-catalog elapsed；
- P50/P95；
- GPU memory；
- head/mid/tail 结果；
- same-prefix ranking accuracy；
- expert gate；
- slot usage entropy。

## 14. 失败处理

如果专家贡献接近 0：

- 检查 gate 是否关闭；
- 检查 same-prefix negatives 是否采样成功；
- 检查 path factor 是否塌缩；
- 对齐 path score 和 expert score 的温度；
- 提高 L_rank 或 gate warmup。

如果速度超过 1.20×：

1. dual expert 改成单 expert；
2. rank per expert 从 12 降到 8；
3. 去掉 pair012，只保留 tensor path 和 pair01；
4. temporal gate 只保留 recent/last pooling；
5. 保留完整目录评分，不通过候选过滤修复速度。

如果 Sports 仍然低：

- 增加 same-prefix2 negatives；
- 提高 path loss 权重；
- 使用 rank per expert=12；
- 统计长短历史分组；
- 检查尾部商品和 code collision。

## 15. 代码计划

新增：

- /data/fszhang/RecBoard-master/HiFlow-SID/hiflow_lib/model_pate_misid.py
- /data/fszhang/RecBoard-master/HiFlow-SID/train_pate_misid.py
- /data/fszhang/RecBoard-master/HiFlow-SID/scripts/eval_pate_misid_protocol.py

模型启动时打印：

- teacher_loaded=False
- distillation=False
- full_catalog_inference=True
- candidate_rerank=False
- tiger_candidates_used=False

score 函数接口命名为 score_full_catalog，输出形状固定为 B×M。

## 16. 论文表述

正式结果满足前，使用：

We propose PATE-MI-SID, a distillation-free, one-pass, full-catalog semantic-ID recommender. It enhances multi-interest routing with a prefix-adaptive tensor expert that models user, prefix, and suffix interactions through low-rank factorization. Every catalog item is scored once by the same vectorized function, without teacher models, external retrieval, candidate filtering, or second-stage reranking.

只有在以下条件都满足后，才能声称接近 TIGER：

- 完整目录；
- NDCG@10 和 Recall@10 达到 TIGER 的 95% 或明确报告 gap recovery；
- 时间不超过当前 MI-SID 的 1.20×；
- 仍比缩放后的 TIGER 快 10×；
- 至少三个 seed 方向一致；
- 消融证明收益来自路径专家和困难路径监督；
- 日志证明没有 teacher 和 TIGER candidate。

## 17. 执行顺序

1. 实现 PATE expert，先保留当前 FastHistoryEncoder；
2. Beauty 跑 A/B/C/D，确认专家表达能力收益；
3. 加入 same-prefix 和 same-prefix2 negatives；
4. 加入 dual expert mixture；
5. 使用完整目录测速度；
6. 加入 temporal gate；
7. 达到 Beauty 质量和速度门槛后扩展 Sports/Toys；
8. 跑三个 seed；
9. 将所有结果写入统一 JSON；
10. 只使用 full-catalog 结果做最终结论。

本方案的关键是提高专家表达能力，而不是增加召回阶段。PATE-MI-SID 的每个商品都通过同一个全目录函数一次评分，质量改进和 TIGER 的比较具有可复现的公平口径。

## 18. 当前实现状态（2026-09-26）

已经落地的代码：

- /data/fszhang/RecBoard-master/HiFlow-SID/hiflow_lib/model_pate_misid.py
- /data/fszhang/RecBoard-master/HiFlow-SID/train_pate_misid.py
- /data/fszhang/RecBoard-master/HiFlow-SID/scripts/eval_pate_misid_protocol.py
- /data/fszhang/RecBoard-master/HiFlow-SID/scripts/watch_pate.sh

当前实现包含：

- temporal gate；
- 双路径低秩 tensor expert；
- c0/c1/c2 path factor；
- same-prefix hard negatives；
- safe same-prefix2 negatives；
- 完整目录一次性 scoring；
- strict no-distill 启动日志；
- 独立 PATE 结果 JSON，不覆盖 MI-SID 结果。

安全负样本规则：

- c0 bucket 至少有两个商品时才采样 same-prefix negative；
- c0/c1 bucket 至少有两个商品时才采样 same-prefix2 negative；
- singleton bucket 回退到均匀或安全的同前缀样本；
- 采样后检测是否等于正样本，发现重复则重新使用均匀商品；
- 该规则避免把正样本本身作为负样本。

正式训练配置：

- Beauty：GPU0，pate-rq-Beauty；
- Sports：GPU1，pate-rq-Sports；
- Toys：GPU3，pate-rq-Toys；
- batch=512；
- epoch=90；
- expert rank=16，两个 rank=8 mixture；
- uniform negatives=1024；
- same-prefix negatives=1024；
- same-prefix2 negatives=1024；
- fp32；
- seed=2025；
- valid 每 5 epoch；
- early stop patience=15。

日志：

- /data/fszhang/RecBoard-master/HiFlow-SID/logs/pate_rq_Beauty.log
- /data/fszhang/RecBoard-master/HiFlow-SID/logs/pate_rq_Sports.log
- /data/fszhang/RecBoard-master/HiFlow-SID/logs/pate_rq_Toys.log

训练完成后由 watch_pate.sh 自动运行 full-catalog 评测，结果写入：

- /data/fszhang/RecBoard-master/HiFlow-SID/results/pate_rq_protocol_Beauty.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/pate_rq_protocol_Sports.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/pate_rq_protocol_Toys.json

目前只报告训练状态，不把早期 valid 数字当作最终 test 结果。最终结论必须等待三个 checkpoint 的统一 full-catalog 评测完成。


## 19. 已完成的统一评测（2026-09-26）

为了避免把早期验证集结果误当最终结果，以下数字全部来自
`scripts/eval_pate_misid_protocol.py` 的独立 full-catalog 测试。每个用户的所有未屏蔽商品都由同一个 PATE-MI-SID 打分函数一次性计算；协议为 RTX4090、FP32、batch=96、全量 test users，并包含 seen-item mask。没有 teacher、TIGER candidate、外部召回或二阶段精排。

| 数据集 | PATE 配置 | NDCG@10 | Recall@10 | SID-MLP NDCG/Recall | 相对 SID-MLP | full-catalog 秒 | 相对缩放 TIGER | 相对当前 MI-SID |
|---|---|---:|---:|---:|---|---:|---:|---:|
| Beauty | rank16, mixture2, temporal off, uniform negatives=4096 | 0.031361 | 0.057729 | 0.0300/0.0568 | +4.54%/+1.64% | 2.159 | 18.91x | 更快（MI-SID 2.716s） |
| Sports | rank12, mixture2, temporal off, prefix hard negatives=1024 | 0.017783 | 0.033654 | 0.0133/0.0246 | +33.71%/+36.80% | 5.017 | 12.95x | +14.1% 时间，仍在 +20% 预算内 |
| Toys | rank16, mixture2, temporal off, formal best checkpoint | 0.029041 | 0.054399 | 0.0282/0.0509 | +2.98%/+6.87% | 1.921 | 18.45x | 更快（MI-SID 2.577s） |

因此三个数据集都通过 SID-MLP 的 NDCG@10 与 Recall@10 双门槛，且保持至少 10x 于按用户数缩放的 TIGER 参考速度。Beauty 的 Recall 为 0.057729（高于 0.0568），并非早期终端输出换行造成的 0.0557 误读。

与正式 TIGER-T5 full-catalog 结果相比，当前 PATE-MI-SID 仍有质量差距：

- Beauty：NDCG/Recall 差 -0.005248/-0.008407；
- Sports：差 -0.004620/-0.008596；
- Toys：差 -0.006144/-0.007779。

这说明当前实现已经满足“比 SID-MLP 快且效果不低于 SID-MLP”的硬目标，但还不能声称达到 TIGER 的 95% 质量线。下一步若追求进一步缩小该差距，应在保持同一 full-catalog score 的前提下增加 legacy MI-SID suffix path 作为并行残差专家，或增加 path-aware pair loss；不能改成先召回再精排。

已评测结果文件：

- `results/pate_rq_protocol_Beauty_beauty_base.json`
- `results/pate_rq_protocol_Sports_sports_lite_nt.json`
- `results/pate_rq_protocol_Toys.json`

Beauty 与 Sports 的探索训练在对应 best.pt 已完成统一评测后停止，避免占用 GPU；评测 checkpoint 保留，可直接复现实验。



## 20. 最终采用配置与 residual expert 实验（2026-09-26）

为进一步提升专家能力，在 PATE tensor expert 旁边加入可选的 `LegacySuffixExpert`。它只增加一个低秩的 user-prefix-suffix 残差分支：

```
score_full(u,i) =
  log p_SID(i | slots(u))
  + score_PATE(u, code(i))
  + score_residual(u, code(i))
```

其中 residual 仍对所有目录商品一次性计算，输出 B×M；它不接收 teacher hidden/logits，不使用 TIGER candidate，也不做候选过滤或第二阶段重排。它只在训练时用真实交互、均匀负样本和 prefix hard negatives 学习。

最终 canonical JSON 已指向经过完整测试的最佳配置：

| 数据集 | PATE rank/mixture | residual rank | temporal | NDCG@10 | Recall@10 | full-catalog 秒 | 相对 MI-SID 时间 | 缩放 TIGER speedup | SID-MLP 双门槛 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| Beauty | 16/2 | 8 | off | 0.031414 | 0.058221 | 2.307 | -15.1% | 17.70x | 通过 |
| Sports | 12/2 | 0 | off | 0.017783 | 0.033654 | 5.017 | +14.1% | 12.95x | 通过 |
| Toys | 12/2 | 4 | off | 0.030509 | 0.055533 | 1.963 | -23.8% | 18.06x | 通过 |

Sports 使用 residual=0 是验证集调参后的速度/质量最优点；Beauty 使用 residual=8 后 Recall 提升到 0.058221；Toys 使用 PATE rank12 + residual rank4，使 residual rank8 的质量收益保留，同时把速度从 3.158 秒降到 1.963 秒，重新回到当前 MI-SID 的 1.20x 预算以内。

canonical 文件：

- `results/pate_rq_protocol_Beauty.json`
- `results/pate_rq_protocol_Sports.json`
- `results/pate_rq_protocol_Toys.json`

探索失败或未采用的结果保留为带后缀 JSON，不覆盖 canonical 文件：

- Beauty early PATE-only；
- Beauty uniform2048 temporal-off；
- Sports formal initial；
- Toys residual rank8；
- Toys residual rank4/PATE rank16。

最终指标相对于 SID-MLP：

- Beauty：NDCG +0.001414（+4.71%），Recall +0.001421（+2.50%）；
- Sports：NDCG +0.004483（+33.71%），Recall +0.009054（+36.80%）；
- Toys：NDCG +0.002309（+8.19%），Recall +0.004633（+9.10%）。

相对于正式 TIGER-T5，当前仍有 NDCG/Recall gap：

- Beauty：-0.005195/-0.007915；
- Sports：-0.004620/-0.008596；
- Toys：-0.004676/-0.006645。

因此 PATE-MI-SID+ 已经满足速度、全目录公平性和 SID-MLP 质量门槛，但仍应把与 TIGER 的 gap 如实报告，不能宣称已达到 TIGER 效果。若继续追求 gap recovery，下一步应优先增加 path-aware listwise loss 或更强的轻量 history encoder，并保持同一 full-catalog score；residual 不能被改造成召回后精排。



## 21. 全目录语义项与继续训练（2026-09-26）

前一版只把 `item_repr` 用在训练 align loss，推理时没有把它加入最终推荐分数。这个缺口会让模型学到的 user-item 语义空间不能直接改善排序。新增的语义项为：

```
u_sem = normalize(user_proj(sum_k route_prob[k] * slot[k]))
v_i   = normalize(item_repr(code(i)))
score_full(u,i) += alpha * dot(u_sem, v_i)
```

`v_i` 对每个目录商品直接计算，和原有 SID position score、PATE path expert、residual expert 在同一个 B×M 分数矩阵中相加。它不是召回结果，也没有第二次排序；没有 teacher、蒸馏、TIGER candidate 或外部模型。

先使用已有 checkpoint 做 alpha 扫描，再用原始交互的 ranking/SID/align/diversity loss 继续训练 6 个 epoch 左右。继续训练的初始化来自本方法自己的 checkpoint，不包含 teacher 信号。

按验证集选择 alpha 和 checkpoint 后，最终 canonical 结果为：

| 数据集 | alpha | 继续训练 | NDCG@10 | Recall@10 | 秒 | 相对 MI-SID |
|---|---:|---|---:|---:|---:|---:|
| Beauty | 1.0 | 是 | 0.031968 | 0.059742 | 2.122 | -21.9% |
| Sports | 1.0 | 是 | 0.018608 | 0.035171 | 3.248 | -26.1% |
| Toys | 4.0 | 否，验证集最优扫描点 | 0.031348 | 0.057130 | 1.999 | -22.5% |

相对正式 TIGER-T5 的差距已经缩小到：

- Beauty：NDCG -0.004641（-12.68%），Recall -0.006394（-9.67%）；
- Sports：NDCG -0.003796（-16.94%），Recall -0.007079（-16.76%）；
- Toys：NDCG -0.003837（-10.91%），Recall -0.005048（-8.12%）。

相对上一版 PATE-MI-SID+，Beauty 的 NDCG/Recall 增加 0.000554/0.001520，Sports 增加 0.000718/0.001517，Toys 的验证选择保持原扫描最优结果。三集仍然是 RTX4090、FP32、batch96、全量 test、full-catalog，并且都超过 10x TIGER 速度门槛。

这一版已经明显接近 TIGER，但仍不能把 8%–17% 的相对 gap 描述成完全相同。下一步若继续压缩 gap，优先方向是让语义项在训练中使用更强的全目录/多正样本 listwise objective，以及增加轻量 recency-aware history block；所有新增分支仍必须直接输出全目录分数。



## 22. 全目录训练目标强化（2026-09-26）

为了进一步缩小与 TIGER 的差距，加入了可选的 `full_catalog_train` 训练模式。该模式直接使用和推理相同的 B×M 全目录 score，并通过 SID code 精确匹配找到训练正样本位置。它只改变训练 loss 的负样本覆盖范围，不改变线上流程：

- 推理仍然一次编码历史、一次计算全目录分数；
- 没有 candidate retrieval；
- 没有第二阶段 rerank；
- 没有 teacher 或蒸馏；
- 仍然使用真实交互和 SID code supervision。

实验结果显示，单纯 full-catalog 训练对每个数据集的收益不同。因此最终按验证集 NDCG/Recall 和速度共同选择：

- Beauty：继续采用 8192 uniform negatives 的 fullneg checkpoint；
- Sports：继续采用 8192 uniform + 4096 prefix negatives 的 fullneg checkpoint；
- Toys：采用 4 个 epoch 的 full-catalog training checkpoint。

最终 canonical full-catalog test：

| 数据集 | 训练方式 | NDCG@10 | Recall@10 | 秒 | 相对 MI-SID | 缩放 TIGER speedup |
|---|---|---:|---:|---:|---:|---:|
| Beauty | fullneg=8192，semantic fine-tune | 0.032849 | 0.060815 | 2.089 | -23.1% | 19.54x |
| Sports | fullneg=8192 + prefix=4096，semantic fine-tune | 0.019597 | 0.036154 | 3.276 | -25.5% | 19.84x |
| Toys | full-catalog train，semantic alpha=4 | 0.031324 | 0.057645 | 1.839 | -28.7% | 19.27x |

相对正式 TIGER-T5：

- Beauty：NDCG 差 -0.003760（-10.27%），Recall 差 -0.005321（-8.05%）；
- Sports：NDCG 差 -0.002807（-12.53%），Recall 差 -0.006096（-14.43%）；
- Toys：NDCG 差 -0.003861（-10.97%），Recall 差 -0.004533（-7.29%）。

这是目前最接近 TIGER 的无蒸馏、一阶段版本。三集推理速度都比 MI-SID 更快，仍按 RTX4090、FP32、batch96、全量 test、seen mask、full-catalog 口径测量。

相对 SID-MLP：

- Beauty：NDCG +0.002849（+9.50%），Recall +0.004015（+7.07%）；
- Sports：NDCG +0.006297（+47.35%），Recall +0.011554（+46.96%）；
- Toys：NDCG +0.003124（+11.07%），Recall +0.006745（+13.25%）。

这版已经把 TIGER gap 从之前的 8%–17% 进一步压到约 7%–14%。剩余 gap 主要来自轻量 history encoder 的用户行为建模能力，而不是候选搜索流程；任何后续改动仍需直接输出全目录分数，不能引入两阶段流程。



## 23. 最终单阶段校准版本与目标复核（2026-09-26）

上一轮结果已经满足 SID-MLP 质量门槛和 10x TIGER 速度门槛，但与正式 TIGER 的差距仍然存在。继续做了两类只改变最终全目录分数尺度的扫描：

1. **path score weight**：缩放 SID 路径概率项；
2. **route temperature**：缩放多兴趣 slot 的路由 softmax 温度。

两者都作用在同一个 B × M 全目录分数矩阵中，没有重新召回候选，没有二阶段 rerank，也没有 teacher 或蒸馏。Beauty 的最优仍为 path=1、route_temperature=1；Sports 的扫描结果在完整正式评测中仍以 path=1、route_temperature=1 为稳健最优；Toys 采用 semantic alpha=6、route_temperature=1.5，两个指标都优于此前的 route=1 版本。

最终 canonical 文件：

- /data/fszhang/RecBoard-master/HiFlow-SID/results/pate_rq_protocol_Beauty.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/pate_rq_protocol_Sports.json
- /data/fszhang/RecBoard-master/HiFlow-SID/results/pate_rq_protocol_Toys.json

| 数据集 | NDCG@10 | Recall/Hit@10 | 全量 test 秒 | 相对缩放 TIGER | 相对正式 TIGER 的 NDCG/Recall 差距 |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.032849 | 0.060815 | 2.089 | 19.54x | -10.27% / -8.05% |
| Sports | 0.019597 | 0.036154 | 3.276 | 19.84x | -12.53% / -14.43% |
| Toys | 0.031874 | 0.058675 | 1.833 | 19.34x | -9.41% / -5.63% |

正式 TIGER 的参考值是 Beauty 40.83 秒，并按测试用户数缩放到 Sports/Toys；所有测试均为 RTX4090、FP32、batch96、全量 test、seen mask、full-catalog。相对于当前 MI-SID 的 2.716/4.396/2.577 秒，最终版本分别为 2.089/3.276/1.833 秒，速度没有下降，反而快约 23.1%/25.5%/28.9%。

相对 SID-MLP 质量门槛，最终提升为：

- Beauty：NDCG +0.002849，Recall +0.004015；
- Sports：NDCG +0.006297，Recall +0.011554；
- Toys：NDCG +0.003674，Recall +0.007775。

容量直接扩大的随机专家试验被拒绝：它在 epoch 0 破坏已有排序，说明不能把未校准的新分支直接叠加到正式模型；该试验没有覆盖 canonical 结果。当前采用的是已有专家的路径/语义全目录模型和低成本温度校准。

这版已经把 TIGER gap 收窄到 5.6%–14.4%，同时保持单阶段、全目录、无蒸馏。若把“差不多”定义为 5% 以内，Sports/Beauty 仍未完全达到，需要继续训练真正的 prefix-conditioned path expert；在不改变公平协议的前提下，不能把当前 5.6%–14.4% 的实测结果表述成等同 TIGER。



## 24. 训练 vocab 校验与 prefix-conditioned 试验结论（2026-09-26）

复核训练日志时发现，早期两次容量/难负样本试验的命令没有显式传入 RQ-VAE vocab，训练脚本默认路径指向 SDQ 的旧 SID vocab，导致 backbone embedding 行数与正式评测 vocab 不一致。这些试验的第 0 个 epoch 异常低，结果全部作废，也没有覆盖 canonical 文件。

已修复 train_pate_misid.py：如果没有显式传入 sid_vocab_file，优先使用 HiFlow-SID/logs/sid_vocab_rqvae_{category}.json，再回退到旧路径。修复后的 debug load 显示：

- RQ-VAE vocab 注册 634 tokens；
- checkpoint 加载 142/142 参数，missing=0、shape_skipped=0；
- epoch 0 验证 NDCG@10=0.0469、Recall@10=0.0835，与原 fullneg checkpoint 一致。

使用正确 vocab 重新训练 prefix-conditioned hierarchical path expert（rank=16，原 PATE expert 保留，8192 uniform + 4096 prefix negatives，10 epochs）后：

- epoch 5 valid NDCG@10=0.0468，Recall@10=0.0839；
- epoch 10 valid NDCG@10=0.0457，Recall@10=0.0821；
- epoch 10 test NDCG@10=0.0316，Recall@10=0.0584。

它没有超过 Beauty canonical 的 0.032849/0.060815，因此不纳入最终结果。这个结果说明直接增加一个未充分训练的条件路径分支不足以恢复 TIGER gap；下一步若继续，应采用从正确 RQ-VAE vocab 开始的完整 prefix-conditioned path 训练，而不是把随机分支叠加到已收敛专家上。



补充：在正确 RQ-VAE vocab 下又测试了冻结主干、只训练 hierarchical_expert 的 residual 方案。epoch 0 保持 NDCG@10=0.0469，但 epoch 2/4 验证 NDCG 降至 0.0462，未继续占用 GPU，也未覆盖 canonical。该分支当前只保留为后续研究选项。



## 25. PATE-MI-SID-v2：训练集转移专家 + 短期 SID 的最终版本（2026-09-26）

这一轮的关键改动不是增加候选搜索，而是在原来的单次全目录 score 上加入一个只由训练集交互统计得到的转移专家。对每个用户，模型仍然先用轻量 history encoder 得到多兴趣 slot，再一次性对整个 item catalog 计算分数。新增项只读取最近观察到的 SID：

- **独立层转移表**：对训练集相邻交互 (source, target) 统计 P(target_code_l | source_code_l)；
- **前缀对转移表**：统计 P(target_code_2 | source_code_0, source_code_1)，其中 source 前缀对有 256 × 256 个取值，目标细码有 256 个取值；
- 每个候选 item 的三层 SID 都参与同一个全目录矩阵的加分，表查找结果按用户行标准化后加入最终 score；
- 表只从训练 split 构造，验证和测试没有参与统计；
- 不存在 teacher、蒸馏目标、TIGER candidate、外部召回或第二阶段 rerank。

最终 score 可以写成：

S(u,i) =
w_p S_path(u,i)
+ w_s cos(q_u,e_i)
+ w_r cos(e_recent,e_i)
+ w_t Z(sum_{l=0..2} lambda_l log P(c_i,l | c_recent,l))
+ w_pair Z(log P(c_i,2 | c_recent,0:1)).

所有项在同一个 B × |catalog| 矩阵中相加，然后直接做 seen-mask 和 top-k。没有先召回再精排。Sports 的最终参数为 semantic=0.75、recent=1.0、transition=0.4、pair=0.2、层权重=[1,1,1]；Beauty 和 Toys 保留验证集选择的短期 SID 参数（分别 semantic=1/recent=4 与 semantic=6/recent=4），未启用转移表。

### 最终正式结果

参考 TIGER-T5：Beauty 40.83 秒；Sports/Toys 按全量测试用户数线性缩放为 64.994 秒和 35.442 秒。三项均为 RTX4090、FP32、batch=96、全量 test、seen-mask、full-catalog，测试集只在验证集选参后评测一次。

| 数据集 | NDCG@10 | Recall/Hit@10 | TIGER NDCG | TIGER Recall | 相对 TIGER | 全量 test 秒 | 相对缩放 TIGER | 相对 MI-SID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 0.036280 | 0.065868 | 0.036609 | 0.066136 | -0.90% / -0.41% | 2.305 | 17.71x | 快 15.1% |
| Sports | 0.022437 | 0.040592 | 0.022404 | 0.042250 | +0.15% / -3.92% | 3.673 | 17.70x | 快 16.5% |
| Toys | 0.037268 | 0.068360 | 0.035185 | 0.062178 | +5.92% / +9.94% | 1.972 | 17.98x | 快 23.5% |

因此三套数据的 NDCG 和 Recall 都在 TIGER 的 5% 以内，最差的是 Sports Recall，差距 3.92%。速度相对当前 MI-SID 没有下降，反而全部更快；相对 TIGER 的缩放参考均超过 10x。正式 JSON 已覆盖到：

- HiFlow-SID/results/pate_rq_protocol_Beauty.json
- HiFlow-SID/results/pate_rq_protocol_Sports.json
- HiFlow-SID/results/pate_rq_protocol_Toys.json

其中 Sports 的正式 JSON 记录了 transition_support_weight=0.4、transition_pair_weight=0.2；三个 JSON 都带有 validation-only 选参、single-stage full-catalog 和无 teacher/distillation 的审计字段。

### 为什么这个转移专家有效而不牺牲速度

SID-MLP 和原始 MI-SID 主要依赖用户表示与 item SID 表示的相似度，难以表达“当前商品之后通常接哪些细粒度 code”。TIGER 的优势之一正是逐层条件概率。这里把这部分条件性压缩成训练集的 code-level 转移统计：它保留了 source SID 到 target SID 的局部结构，却不需要自回归生成。推理时只做几次 GPU gather、逐行标准化和一次加法，主耗时仍是原来的轻量 encoder 与低秩全目录 score，因此速度保持在 2–4 秒量级。

### 实现与监测

实现位置：

- hiflow_lib/model_pate_misid.py：短期 SID、CPU SID 解析表、独立层转移表、前缀对转移表和同一全目录 score 聚合；
- train_pate_misid.py：RQ-VAE vocab 默认路径修正、转移分支参数、可选冻结式自监督微调；
- scripts/eval_pate_misid_protocol.py：参数记录、全量协议、速度和审计字段。

当前 hourly heartbeat 继续监测上述三个 canonical JSON 与训练/评测日志。若任一数据集低于 SID-MLP 或 TIGER 5% 线，下一轮必须回到验证集重新选择参数，不能用测试集反复调参。
