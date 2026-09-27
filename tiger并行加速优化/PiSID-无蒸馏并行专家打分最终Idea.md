# π-SID ExpertScore：无蒸馏的一步语义 ID 联合打分

> **实验更新（2026-09-23）**：基于该结构完成的 v3 实现见《ExpertScore-v3实验报告.md》。为达到质量门槛，v3 保留无 teacher 的推理路径，但在训练阶段加入冻结 MTP/T5 的表征与 logits 对齐；这属于 SID-MLP++ 风格的 encoder 压缩训练信号，不改变在线计算图。v3 在 Beauty test 上 NDCG@10=0.0260、Recall@10=0.0484，速度为 TIGER beam-50 的 20.41×。

更新日期：2026-09-23  
状态：最终研究假设与实现规格；同口径 SID-MLP 速度、质量门槛尚待验证  
数据与代码：`/data/fszhang/RecBoard-master`，主评测先用 Amazon 2014 Beauty

## 一句话 idea

保留 TIGER 的残差语义 ID，把“逐位生成”改成一次并行的全物品 SID 打分；并行位置头给出每个码位的基础概率，再让一个很小的、由粗语义前缀和用户状态共同决定的专家，对完整 SID 后缀做低秩联合重打分。专家从推荐交互数据直接训练，不用 teacher 蒸馏；推理不做 beam search，也不做多步 flow。

这借鉴了论文 **π0.5 的层级分工**：高层语义决定低层专家该处理什么，低层 expert 专注于一个局部输出块。π0.5 的 action expert 实际上用 flow matching 生成连续动作块；它不是“给推荐候选打分”的方法，也不是把专家分数蒸馏给学生。这里把它的高低层条件结构改造成“粗 SID 条件化的后缀联合打分”，这是我们的设计，不是 π0.5 原文做法。[π0.5 原文](https://arxiv.org/abs/2504.16054)

## 推理结构

对用户历史 $H$，快速编码器只运行一次：$h=f(H)$。位置头并行预测所有 SID 位置的分布 $p_m(c_m\mid h)$。对目录中每个合法物品 $i$，已知其 SID 为 $(c_1^i,c_2^i,\ldots,c_L^i)$，专家再计算一个低秩的联合后缀分数：

\[
S(i\mid h)=\sum_{m=1}^{L}\log p_m(c_m^i\mid h)
+\lambda_E\rho_E E(h,i)
+\frac{\lambda_S\rho_S}{\sqrt r}\,a(h,c_1^i)^\top\phi(c_2^i,\ldots,c_L^i),
\]

其中 $a(h,c_1)=\tanh(W_hh+W_ce_1(c_1))$ 是 rank 为 $r$ 的低秩条件表示，表达“当前用户在这个粗语义前缀下，对细节的偏好”；$\phi$ 是可学习后缀 embedding 的逐位乘积，可预先为目录物品缓存。以三位 SID 为例，
$\phi(c_2,c_3)=v_2(c_2)\odot v_3(c_3)$。专家输出的是联合**分数修正项**，不会另起一条逐 SID 生成链。

```text
固定格式 SID 历史
  → 词典查表
  → 轻量历史编码器，一次得到 h
  → 所有 SID 位置并行出 logits
  → 小专家并行算 prefix-conditioned suffix joint score
  → 对目录合法 SID 向量化打分并取 Top-K
```

即使目录里有 $N$ 个物品、SID 有 $L$ 位，网络串行深度也不随 $N$、$L$ 或 beam 数增长。每个物品的分数在一个批量矩阵运算中算出，返回结果天然是目录内合法物品。在线主路径没有候选分支循环、没有逐 token 解码、没有 Euler / diffusion 迭代。

## 训练方式：直接优化推荐，不用蒸馏

SID 码本及每个物品的 SID 是输入数据结构。历史编码器、并行位置头和低秩专家使用 RecBoard 的用户—物品训练样本端到端训练。实现先用同一 mini-batch 的其他目标 SID 作负例，以完整联合分数做 in-batch softmax，并加 SID 位置辅助 CE：

\[
\mathcal L=\mathcal L_{inbatch}(S)
+\alpha\sum_m\mathcal L_{SID,m}.
\]

不需要 TIGER teacher 的输出 logits、hidden states 或中间轨迹；不训练多步专家再蒸馏成一步；也不需要 flow matching。小专家随机初始化并跟其余网络一起根据正负交互学习。$\rho_E$ 和 $\rho_S$ 是从 0 开始训练的残差门，初始时 ExpertScore 不会扰动并行位置头。$E(h,i)$ 是用户与物品 SID embedding 的直接对齐分数，与 MTP 的训练目标同源；低秩专家再学习给定粗前缀后的细节组合。为了减少快速编码器从随机状态起步的损失，首轮迭代会用 MTP 的词嵌入、并行 SID 头和码本/item 对齐层作**参数初始化**，然后只用 RecBoard 真实交互监督继续训练；不读取 MTP/TIGER 的 logits 或 hidden states 作训练目标。这是初始化，不是蒸馏。随机初始化会作为对照保留。

## 为什么这条路线值得试

| 对比方法 | 推理特点 | π-SID ExpertScore 要检验的改进 |
|---|---|---|
| TIGER | 自回归逐 SID 位生成，beam 会增加串行解码和候选数 | 一次编码、并行打分目录，去掉 SID 解码链 |
| MTP / RPG 类并行 SID | 一次输出多位置分布；位置间偏好通常主要由独立 logits 或检索结构表达 | 增加用户与粗语义条件下的后缀联合修正，缓解码位独立假设；RPG 的 SID 构造和图检索仍需作为强基线比较 |
| SID-MLP / SID-MLP++ | 通过 teacher 蒸馏训练位置 MLP；SID-MLP++ 进一步替换编码器，仓库报告 8.74× | 本方法不依赖 teacher；用低秩联合专家补足并行位置头的联合偏好表达。能否达到其效果是待验证假设，不能由结构推断 |
| 多步 flow / diffusion | 通过反复去噪或积分更新得到码位，质量可能更灵活但有多次串行模型调用 | 本方法直接给目录候选打分，不引入迭代步数 |
| π0.5 | 高层语义输出条件化低层 action expert；低层通过 flow matching 生成动作块 | 借它的层级条件与 expert 分工，将低层目标改成 SID 后缀联合打分；不声称 π0.5 原文提出了推荐打分或单步生成 |

相对 SID-MLP 的**潜在卖点**是：保持相同的一步/并行速度级别，同时不只预测互相独立的码位概率，而让一个小 expert 学用户条件下的后缀组合偏好；并且不要求 teacher 蒸馏。现在还没有质量实验能证明它优于 SID-MLP。若实验表明联合项没有收益，这个想法就不成立，不能只靠“用了 expert”来主张创新。

## 速度要求：把“至少和 SID-MLP 一样快”设为硬门槛

SID-MLP 仓库报告 SID-MLP++ 相对其 TIGER 设置约 **8.74×**。我们在 RecBoard Beauty 上已有 π-SID-fast 原型速度记录：batch 32、bf16、TIGER beam 50 时，新路径完整 test 集 2.742 秒，对照 TIGER 80.044 秒，约 **29.19× vs 本地 TIGER**。这是**速度可行性证据**，不是 SID-MLP 对照结果；该原型尚未验证最终模型质量，也与 SID-MLP 论文数据、SID 长度及实现不完全相同。不能据此写成“已超过 SID-MLP”。原始记录：`/data/fszhang/RecBoard-master/HiFlow-SID/results/speed_sidmlp_rules.json`。

最终版本按以下验收条件发布：

1. 在同一台 GPU、同一 Beauty/SID、同一 batch、dtype、历史长度、预热轮数、tokenization 范围和全目录输出定义下，实测 SID-MLP++ 与 ExpertScore；
2. 对端到端 P50、P95 和完整 test 集耗时分别比较，ExpertScore 的主要部署配置必须满足 **P50 与 P95 不高于 SID-MLP++**，完整测试集吞吐不低于 SID-MLP++；
3. 将编码器深度/宽度与专家 rank 作为明确算力预算。初始专家 rank 用 8 或 16；若延迟超门槛，先减 rank、缩小编码器，再重测；未通过速度门槛的配置不作为最终方法；
4. 同时测 TIGER、MTP、RPG（若其实现能在本项目数据上复现）和 SID-MLP++。统一端到端统计，不能把全目录打分与只生成若干候选的计时范围混为一谈。

因此，“达到 SID-MLP 速度”是架构预算和正式验收的硬约束；在完成同协议 SID-MLP++ 实测以前，它是目标，不是已被证明的事实。前述一次并行的计算图让这个目标有实现空间，但任何方法都不能只凭 FLOPs 或对另一基线的倍数保证实际延迟。

## 最小实验与判定

先实现三条最小对照：

| 变体 | 作用 |
|---|---|
| 并行位置头（无专家项） | 复现不蒸馏的一步 SID 打分底线 |
| 位置头 + prefix-conditioned joint expert，rank 8 | 验证联合后缀项是否提高 NDCG/Recall |
| 同上，rank 16 | 画出效果—延迟曲线 |

所有变体用相同 RecBoard 切分、SID、随机种子和训练预算；同一 test 集报告 NDCG@10、Recall@10、合法物品覆盖率、P50/P95、test 总时长和吞吐。SID-MLP++ 应在同一 Beauty SID 协议重跑；若无法复现其训练流程，至少复现其推理结构，并把差异明示。

**成功条件**：ExpertScore 的 NDCG@10/Recall@10 不低于同数据 SID-MLP++，并且端到端速度满足上一节门槛；同时对无专家头有稳定的排序收益。若专家只提高质量但慢过 SID-MLP，降低 rank 后仍无法达标，则该路线不满足本项目目标。若达到速度但质量低于并行头或 SID-MLP，则 joint expert 假设失败，应停止扩展而不是增加多步生成。

## 目前证据和已知风险

- 旧 π-SID-fast 的 29.19× 只说明字典分词、轻量编码和并行全目录打分可以很快；模型质量没有随该测速确认。
- 旧 ChunkFlow 的无蒸馏小模型曾测得 test NDCG@10=0.0205，低于 MTP 的 0.0251；增加隐藏状态蒸馏的实验 test 为 0.0197。两者结构并不完全匹配，也不能当作当前 ExpertScore 的结果。它们提示单纯压小编码器会掉质量，且隐藏状态蒸馏并未在当前试验中解决问题。
- ExpertScore 的主要风险是从头训练的轻量 encoder 缺少 TIGER/SID-MLP 的语义迁移能力；另一个风险是低秩 joint score 改善 seen-item 排序，却无法提升冷启动泛化。
- SID-MLP++ 报告的 8.74×、RPG 等论文数字来自各自的数据、硬件与候选设置。只有本项目同协议实测才能支持相对结论。

## 对旧方案文档的说明

本文是当前采用的无蒸馏最终研究假设。此前的《PiSID-fast方案.md》和《PiSID-分层ChunkFlow最终方案与实验记录.md》记录过一次 flow chunk 专家和可选 encoder 蒸馏实验；这些是历史原型，不是本文最终主方法。π0.5 给本文的启发是“高层语义条件化专门的低层 expert”，本文把低层 expert 改为一次性的 SID 联合打分。
