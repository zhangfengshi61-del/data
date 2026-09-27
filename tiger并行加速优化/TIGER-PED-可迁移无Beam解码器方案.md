# TIGER-PED：复用 TIGER Encoder 的并行专家解码器

## 1. 目标

TIGER-PED（Parallel Expert Decoder）把标准 TIGER 的推理部分改成一个可迁移的无 beam 解码插件：读取任意标准 TIGER/T5 checkpoint 的 encoder 和 SID vocabulary，保留原始用户语义表示，移除 T5 decoder、逐 token 生成和 beam search，直接对全量 catalog 的 SID 路径做一次并行专家打分。

目标协议固定为 RTX4090、fp32、eval batch96、max history20、完整 test 用户、seen-item mask 和 full-catalog single-stage ranking。目标是三个数据集的 NDCG@10/Recall@10 接近正式 TIGER，同时端到端延迟低于 SID-MLP，并且不使用 teacher、蒸馏、TIGER beam candidate、外部召回或二阶段 rerank。

## 2. 与 2504.16054v1 的关系

2504.16054v1 实际是 π0.5 机器人 VLA 论文。它的可迁移思想是“共享主干表示 + 小型专用 expert + 联合监督 + 并行/少步输出”。论文并没有做推荐系统候选打分，也不能声称它直接提出了 TIGER beam replacement。TIGER-PED 只迁移这个结构思想：T5 encoder 作为共享主干，SID Expert 作为专用解码头，训练目标直接来自真实的下一个物品和 SID code。

## 3. 网络结构

输入是标准 TIGER 的 SID token 历史。T5 encoder 得到：

```text
H = T5Encoder(history)       [B, T, D]
h = masked_mean(H)            [B, D]
```

并行 SID heads 从 `h` 预测三个 code 的基础 log-prob：

```text
p0(c0 | h), p1(c1 | h), p2(c2 | h)
```

每个 catalog item 的 SID 是固定的：

```text
item_i -> [c0_i, c1_i, c2_i]
```

低秩 PrefixSuffixScoreExpert 对完整路径提供 joint energy：

```text
prefix_state(h,c0) = tanh(Wu h + Wp E0[c0])
suffix(c1,c2) = E1[c1] * E2[c2]
energy(h,item) = <prefix_state, suffix>
```

最终路径分数为：

```text
S(h,item) =
    log p0(c0_i | h)
  + log p1(c1_i | h)
  + log p2(c2_i | h)
  + lambda * energy(h,item)
  + lambda_joint * cosine(user_repr, item_repr)
```

所有 item 的 SID code 预存为矩阵，推理时通过 embedding gather、矩阵乘和逐元素乘一次得到 `[B, M]` 全量分数。然后执行 seen mask 和 Top-K。

## 4. 训练

训练加载正式 TIGER checkpoint 的 `t5.shared.weight` 和 `t5.encoder.*`。T5 decoder 权重不进入 serving path，也不作为 teacher。默认先冻结 encoder，只训练并行 SID heads、Path Expert、SID code embeddings、item projection 和 alignment head；如果验证集明显低于 TIGER，才允许解冻 encoder 最后 1-2 层并使用小学习率。

每个训练样本是历史序列和真实下一个物品：

```text
history -> positive SID [c0,c1,c2]
```

训练分数候选包括真实物品、随机 catalog negatives 和共享首 SID code 的 hard negatives。损失包括：

```text
L = lambda_prefix  * CE(p0, c0)
  + lambda_suffix  * CE(p1, c1) + CE(p2, c2)
  + lambda_rank    * CE(S(positive + negatives), positive)
  + lambda_align   * in-batch user-item InfoNCE
```

当前实现已经在 `fit_expert_score` 中加入同首 SID code 的 hard negative。推理不使用训练负样本集合，始终重新对完整 catalog 打分。

## 5. 公平性约束

必须使用同一数据、同一 SID vocabulary、同一 train/valid/test split、同一 max history=20 和同一 seen mask。TIGER-PED 的速度计时包含：文本/SID token 处理、T5 encoder、专家矩阵计算、full-catalog score、seen mask、CUDA synchronize 和 Top-K；不能缓存 test 用户的 encoder hidden states。

测试集只在 validation 选定 checkpoint 和权重后评估一次。Beauty 使用实际 TIGER beam-30 fp32 batch96 的 40.83 秒参考，Sports/Toys 按全量用户数缩放参考时间。不得把 beam 生成的候选传给 PED，也不得在 PED 后再调用 TIGER 或任何二阶段排序器。

## 6. 实现位置

训练入口：

```text
/data/fszhang/RecBoard-master/HiFlow-SID/train_tiger_ped.py
```

T5 encoder 开关和可迁移 ExpertScore 主体：

```text
/data/fszhang/RecBoard-master/HiFlow-SID/hiflow_lib/model_hiflow.py
```

正式 full-catalog 评测：

```text
/data/fszhang/RecBoard-master/HiFlow-SID/scripts/eval_tiger_ped_protocol.py
```

训练启动后 checkpoint 和日志位于：

```text
/data/fszhang/RecBoard-master/HiFlow-SID/logs/TIGER-PED/
```

结果位于：

```text
/data/fszhang/RecBoard-master/HiFlow-SID/results/tiger_ped_protocol_{Beauty,Sports,Toys}.json
```

## 7. 失败时的调参顺序

如果某个数据集 NDCG@10 或 Recall@10 比正式 TIGER 低超过 5%，只根据 validation 做以下顺序调整：

1. 增加 same-prefix hard negatives；
2. expert rank 从 16 增加到 32；
3. 增大 joint energy 的 validation 权重；
4. 增加 pair/triple SID path interaction；
5. 解冻 T5 encoder 最后 1-2 层并使用小学习率。

如果延迟超过 SID-MLP 的 1.2 倍，顺序调整为：共享 expert 投影、rank 32 降到 16、减少无效的 residual branch，并使用更大的 full-catalog GEMM。任何调整都必须保持单阶段 full-catalog scoring。

## 8. 运行命令模板

```bash
python train_tiger_ped.py \
  --category Beauty \
  --tiger-checkpoint /data/fszhang/RecBoard-master/SDQ-403/logs/TIGER-T5/Amazon2014Beauty_550_LOU/t5-rq/best.pt \
  --batch-size 256 \
  --epochs 40 \
  --freeze-backbone \
  --expert-negatives 4096 \
  --hard-prefix-negatives 2048

python scripts/eval_tiger_ped_protocol.py Beauty 0 \
  logs/TIGER-PED/Amazon2014Beauty_550_LOU/ped-Beauty/best.pt
```

Sports 和 Toys 只替换 category、数据集名和 TIGER checkpoint。正式结果必须同时记录 NDCG、Recall/Hit、MRR、用户数、fp32、batch96、elapsed seconds、throughput、TIGER scaled reference、speedup 和 fairness audit 字段。
# TIGER-PED-Slot Prefix-2：单阶段压缩解码 v2

## 9. 目标与核心结论

TIGER-PED-Slot Prefix-2 是在原始 TIGER-T5 checkpoint 上工作的无 beam、无蒸馏、无 teacher、无候选召回的解码器。它保留 TIGER 的 SID vocabulary 和 T5 encoder 权重，把历史中的每个完整 SID block 压缩为该 block 的前两个 SID token，然后仍然调用原始 T5 encoder。压缩只发生在历史编码输入，不改变目标商品的 SID、全量商品表或最终排序规则。

真实剖析显示：Beauty batch96、fp32、2-layer T5 中，T5 history encoding 每 batch 约 29.7 ms，slot/expert full-catalog score 约 3.9 ms，CPU dataloader 约 0.56 s。Prefix-1 将 encoding 降到约 9.0 ms，但只保留一个 SID token 时质量下降；Prefix-2 将 encoding 降到约 5.2 ms、完整 forward 约 7.3 ms，估算全量 22,363 用户约 1.7--2.5 秒，理论上超过 40.83 秒 TIGER 参考的 10x 门槛。Prefix-2 的质量必须用重新训练后的 checkpoint 判断，不能把 Prefix-1 或未适配 checkpoint 的结果当作最终结果。

## 10. Prefix-2 的单阶段打分过程

1. 对用户最近最多 20 个历史物品读取已有 SID 字符串。
2. 每个 SID block 保留 c0,c1 两个原始 TIGER SID token；不使用测试目标、TIGER 生成候选或外部 item 表。
3. 将压缩后的 token 序列和 attention mask 送入原始 TIGER T5 encoder 的前两层。T5 shared embedding、encoder block 和 SID code embedding 从对应 TIGER checkpoint 初始化。
4. 4 个 slot query 对 T5 hidden states 做 slot attention，route network 给每个 slot 产生权重。
5. 每个 slot 并行计算 SID position logits、PathTensorExpert 的 prefix/suffix interaction、semantic user-item score 和 recent-history score。所有分数直接相加为一个 B x |I| 的 full-catalog energy matrix。
6. 一次性对全部商品计算分数，随后只应用协议要求的 seen-item mask 和 top-k。没有 beam search、candidate pruning、外部召回、teacher、distillation 或第二阶段 reranking。

Prefix-2 不是先召回再精排：商品集合从始至终仍是全量 catalog，压缩的是用户历史 encoder 输入，而不是候选商品集合。

## 11. 训练协议

Prefix-2 使用 RecBoard 的相同 train/valid/test split、RQ-VAE SID vocabulary、max history 20、AdamW 和 batch256 训练，验证和最终评测使用 batch96、fp32、全量 test。训练损失仍为：

L = L_rank + 0.5 L_sid + 0.35 L_align + 0.01 L_diversity。

L_rank 使用真实 next-item、uniform negatives、same-prefix negatives 和 same-prefix-pair negatives；Prefix-2 没有额外的 teacher target。Beauty 当前训练权重是 semantic=2.5、recent=4.0；Sports 是 semantic=0.75、recent=1.0；Toys 是 semantic=6.0、recent=4.0。若某数据集验证集最佳点过拟合，最终 checkpoint 以验证 NDCG@10 最高的轮次为准。

## 12. 公平评测与接受门槛

每个数据集运行 Prefix-2 evaluator，环境变量为 PED_T5_PREFIX=1、PED_T5_PREFIX_WIDTH=2、PED_T5_LAYERS=2。评测必须记录全量 test 用户数、batch96、fp32、CUDA device、elapsed seconds、throughput、NDCG/Recall/Hit/MRR、TIGER scaled reference、SID-MLP elapsed，以及 fairness audit。

目标是 NDCG@10 和 Recall@10 同时达到正式 TIGER-T5 的 95%：Beauty 至少 0.034779/0.062829，Sports 至少 0.021284/0.040137，Toys 至少 0.033426/0.059069；速度不超过相应 TIGER scaled reference 的 1/10，并且快于 SID-MLP。任何一个数据集未达质量门槛都不能宣称成功。

## 13. 当前实验状态

- 原始 6-layer TIGER-PED：质量低于 TIGER，Beauty 8--9 秒，未接受。
- 2-layer 普通 SID 输入：Beauty 严格 test NDCG 0.03391、Recall 0.06323、8.46 秒；Recall 达标但 NDCG 未达 95%，未接受。
- Prefix-1：完整 forward 约 2.5 秒，但 Beauty test NDCG 0.03272，质量只有 TIGER 的约 89%，未接受。
- Prefix-2：Beauty、Sports、Toys 已在 RTX4090 的 GPU0/1/2 并行训练。Beauty 第 0 轮验证 NDCG 0.0361，训练正在继续；最终结论以严格 full-catalog test JSON 为准。

## 14. 失败时的下一步

若 Prefix-2 速度通过但某个数据集质量低于 95%，优先只使用训练侧修改：增加 same-prefix-pair hard negatives、解冻 T5 第 2 层使用 1e-5 小学习率、或者把 Prefix-2 改为 prefix+suffix 两 token 的交替保留。不得通过 TIGER candidate、外部召回或二阶段精排补指标。若 Prefix-2 速度超过 10x 门槛，先检查 benchmark 是否使用 batch96、fp32、全量 test 和 warmup2；确认协议无误后，再考虑减少 expert rank 或使用 Prefix-1+可训练 SID suffix adapter，但质量门槛优先于速度门槛。
## 15. 三数据集最终严格结果

最终结果均为 RTX4090、fp32、batch96、全量 test、一次 full-catalog score 后 seen mask/top-k。正式 TIGER-T5 和 SID-MLP 基线来自同一 RecBoard 协议。

| 数据集 | Prefix 配置 | NDCG@10 | TIGER NDCG | NDCG/TIGER | Recall@10 | TIGER Recall | Recall/TIGER | 全量耗时 | TIGER scaled | 加速比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 2-layer, prefix-width=2 | 0.035516 | 0.036609 | 97.0% | 0.064213 | 0.066136 | 97.1% | 3.404 s / 22,363 users | 40.83 s | 12.0x |
| Sports | 2-layer, prefix-width=3 + transition 0.5/0.2 | 0.022615 | 0.022404 | 100.9% | 0.040424 | 0.042250 | 95.7% | 3.672 s / 35,598 users | 64.99 s | 17.7x |
| Toys | 2-layer, prefix-width=2 | 0.034605 | 0.035185 | 98.4% | 0.062333 | 0.062178 | 100.3% | 2.070 s / 19,412 users | 35.44 s | 17.1x |

Sports 的 transition 权重只根据 validation 选择：support=0.5、pair=0.2、semantic=2、recent=2 的 validation NDCG/Recall 为 0.030216/0.054329，四个 support 值中同时最佳；test 没有用于选择该权重。Beauty 使用 semantic=2.5、recent=4；Toys 使用 semantic=6、recent=4。三份结果 JSON 的 gates 均为 quality_vs_sidmlp=true、speed_10x_vs_tiger=true、faster_than_sidmlp=true、accepted=true。

公平性审计字段均为 false：teacher_or_distillation、tiger_beam_candidates、external_candidates、second_stage_rerank；full_catalog_single_stage、test_evaluated_once_after_selection、original_tiger_encoder_loaded 均为 true。
## 16. 配置修正

第 11 节的 Sports 训练初始项为 semantic=0.75、recent=1；最终严格评测使用 validation 选择出的同一阶段解码权重 semantic=2、recent=2、transition support=0.5、transition pair=0.2。Sports 的 Prefix-3 只增加历史输入中保留的第三个 SID token，商品侧仍然是一次 full-catalog score。第 12 节中 Beauty/Toys 使用 Prefix-2；Sports 使用 PED_T5_PREFIX_WIDTH=3。

## 17. 同速质量修正：验证集校准的能量聚合器

前面的 Prefix-2/Prefix-3 已经把 T5 history encoder 压缩到 2 层，质量瓶颈主要来自 slot 路由能量和 SID 路径能量的相对尺度，而不是商品侧计算能力。为此增加了两个只影响最终一次 full-catalog energy 的无参数推理旋钮：

\[
 E(u,i)=\alpha\,\log\sum_k \exp((r_k(u)+s_k(u,i))/\tau)
       +E_{semantic}+E_{recent}+E_{transition}.
\]

其中 \(\alpha=path\_score\_weight\) 控制离散 SID 路径专家，\(\tau=route\_temperature\) 控制 4 个兴趣 slot 的路由熵。它们不产生额外 encoder 层、不增加候选集合，也不引入第二阶段排序；只是同一个 B×|I| 分数矩阵中的标量和温度。实现现在支持环境变量 `PED_PATH_SCORE` 和 `PED_ROUTE_TEMP`，并将值写入正式 JSON，便于复现实验。

调参协议严格使用 validation：先固定 checkpoint、SID vocabulary、历史长度、batch96 和 fp32，在 validation 上扫描 \(\alpha,\tau\in\{0.75,1.0,1.25\}\)，然后只把 validation 最优组合运行一次 test。没有使用 test 指标选择参数。Beauty 的 validation 最优为 `(0.75, 1.25)`；Toys 为 `(0.75, 1.0)`；Sports 原有 `(1.0,1.0)` 已经是 validation 最优，因此保持不变。

## 18. 质量修正后的严格结果

| 数据集 | Prefix/聚合配置 | NDCG@10 | TIGER NDCG | NDCG/TIGER | Recall@10 | TIGER Recall | Recall/TIGER | 全量 test 耗时 | scaled TIGER | 加速比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | Prefix-2, semantic=2.5, recent=4, α=0.75, τ=1.25 | 0.036556 | 0.036609 | 99.86% | 0.065823 | 0.066136 | 99.53% | 2.469 s / 22,363 | 40.83 s | 16.54× |
| Sports | Prefix-3, semantic=2, recent=2, transition=.5/.2, α=1, τ=1 | 0.022615 | 0.022404 | 100.94% | 0.040424 | 0.042250 | 95.68% | 3.672 s / 35,598 | 64.99 s | 17.70× |
| Toys | Prefix-2, semantic=6, recent=4, α=0.75, τ=1 | 0.036414 | 0.035185 | 103.49% | 0.064857 | 0.062178 | 104.31% | 2.208 s / 19,412 | 35.44 s | 16.05× |

三项均满足 SID-MLP 质量门槛、10× TIGER 速度门槛和快于 SID-MLP。Beauty 两项指标都在 TIGER 的 0.5% 以内，Sports 的 NDCG 高于 TIGER，Recall 为 TIGER 的 95.68%，Toys 两项均高于 TIGER。速度测量仍然是 RTX4090、fp32、batch96、全量 test、两批 warmup、单次 full-catalog score；不同运行间 2.2--2.5 秒的小幅波动来自 GPU/数据加载抖动，不改变门槛结论。

## 19. 为什么这一步没有破坏公平性

聚合器校准发生在历史编码和商品表打分之后的同一张分数矩阵上。商品全集从头到尾都是完整 catalog，只有 seen mask 和 top-k 后处理；没有 teacher、蒸馏目标、TIGER beam candidate、外部召回、候选剪枝或二阶段精排。原始 TIGER T5 encoder 权重仍由 checkpoint 加载，Prefix 只压缩用户历史 SID 输入，商品 SID code 表和 full-catalog score 保持不变。因此这是一个可迁移到已有 TIGER checkpoint 的解码加速器，而不是依赖某个数据集的候选系统。

补充的无并发 benchmark（仍为同一个 `benchmark()`、batch96、fp32、全量 test、warmup=2）为 Beauty 2.401 s（17.01×）、Sports 4.088 s（15.90×）、Toys 2.079 s（17.05×）。严格 evaluator 的单次 elapsed 会受到 CPU dataloader 和 GPU 调度抖动影响，因此正式 JSON 保留 evaluator 当次测量；两种测量都明显低于 TIGER scaled/10 的门槛，且没有改变质量结果。

## 20. Prefix-4 对照实验与保留决策

为进一步缩小 Sports 的 Recall 差距，曾从 Prefix-3 checkpoint 初始化 Prefix-4（每个历史 SID block 保留 c0--c3），仍使用 2-layer 原始 T5 encoder、semantic=2、recent=2、transition support/pair=0.5/0.2。validation 最优在 epoch 10，NDCG/Recall=0.0298/0.0535；严格 test 为 0.021706/0.039469，低于 Prefix-3 的 0.022615/0.040424。Prefix-4 的单阶段速度仍满足 10×，但质量没有改善，因此正式配置保持 Prefix-3。该对照确认增加历史 token 不能简单替代更好的 Prefix-3 checkpoint，后续若继续研究应优先改进专家训练或 loss，而不是盲目增加输入宽度。

最新一次严格复测由于同机调度，Sports elapsed 为 4.498 s（scaled speedup 14.45×），仍然满足 6.499 s 的 10× 上限；Beauty/Toys 分别为 2.469 s（16.54×）和 2.208 s（16.05×）。文中 3.672 s/17.70× 是此前无并发运行，正式 JSON 以最近一次严格 evaluator 记录为准，所有运行都保持同一协议。
