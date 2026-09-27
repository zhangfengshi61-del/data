# π-SID 分层 ChunkFlow：当前方案与实验记录

> **状态：历史原型记录，已被无蒸馏最终 idea 取代。** 当前研究方案见[π-SID ExpertScore：无蒸馏的一步语义 ID 联合打分](</data/fszhang/tiger并行加速优化/PiSID-无蒸馏并行专家打分最终Idea.md>)。本文保留 ChunkFlow/可选编码器蒸馏实验的原始记录，不代表当前主方法；ChunkFlow 的质量与速度结果不能转记为 ExpertScore 的结果。

更新日期：2026-09-23  
实现：`/data/fszhang/RecBoard-master/HiFlow-SID`  
数据：RecBoard 的 Amazon 2014 Beauty，22,363 用户、12,101 物品、3 位 SDQ SID，leave-two-out、full ranking。

## 当前结论

我们把目标定为：保留语义 SID 的共享码本与物品语义结构，用一次小型专家前向联合生成细粒度 SID 后缀，同时把用户历史编码和 SID 文本处理做轻量化。π0.5 提供的是“先预测高层语义，再由单独的小 expert 生成一个完整低层块”的架构启发；它的 action expert 在 post-training 开始时随机初始化，再用 flow matching 训练，并非先单独预训练或蒸馏出来。π0.5 本身默认仍用 10 次 flow integration，因此我们的一步推理是推荐侧待验证的改进，不是论文原结论。[π0.5 原文](https://arxiv.org/abs/2504.16054)

当前模型已证明速度路径有潜力，但**质量目标尚未达到**：训练后的蒸馏版在严格 batch=96、TIGER beam=30、fp32 对比中测得 P50 0.0445 ms/用户，对照 TIGER 1.4464 ms/用户，约 32.5×；但该模型 test NDCG@10=0.0197，低于当前 MTP 的 0.0251 和 TIGER 的 0.0411。故目前不能声称“同时达到 SID-MLP 的效率且质量相当”，更不能声称效果超过 SID-MLP。

## 方法定义

暂名 **π-SID 分层 ChunkFlow**。现有商品 SID 为 \((c_1,c_2,c_3)\)：\(c_1\) 表示高层粗语义，\((c_2,c_3)\) 构成低层细节块。用户历史 \(H\) 经快速编码器得到 \(h\)，prefix head 预测高层分布 \(p(c_1\mid h)\)。小型后缀专家在 \((h,g)\) 条件下，一次预测两个 suffix embedding 的向量场；向量经一次 Euler 更新，再投影到两个 SID 码本。

```text
SID 历史文本
   → 字典查表 tokenizer
   → 轻量无 attention 的全局上下文 MLP encoder → h
   → coarse SID prefix 分布 p(c1|h) → 条件表示 g
   → behavior prior 产生 z0
   → 小型 ChunkMLPExpert 一次预测 (c2,c3) 的 flow vector field
   → 一步 Euler + 两个码本投影
   → 与并行 SID 头及 item-level 能量合并，对全量物品排序
```

训练好的并行 suffix 头提供稳定的离散先验，低层 expert 学习在粗语义条件下对 suffix 联合结构作补充。当前推理分数为：

\[
S(i\mid h)=\log p(c_1^{(i)}\mid h)
 +\sum_{m=2}^{3}\left[\log p_{\mathrm{parallel}}(c_m^{(i)}\mid h)
 +q_{\mathrm{flow}}(c_m^{(i)}\mid h,g)\right]
 +\lambda E(h,\mathrm{SID}(i)).
\]

这里 \(q_{\mathrm{flow}}\) 来自一次 flow-expert 更新后的码本相似度；\(E\) 是 item-level 对齐能量。在线排序仍对目录内已有 SID 打分，因此候选一定是合法物品。实现默认 `prefix-mode=soft`，以 prefix 概率加权的 SID embedding 作为专家条件，避免训练时可微软条件与推理硬分支不一致。

## π0.5 启发与我们自己的改动

π0.5 将高层语义预测与低层连续动作块拆开：高层输出语义 subtask，低层 action expert 以该 subtask 为条件生成整块动作；后训练时新增随机初始化的 action expert，与已有模型一起训练。我们把 subtask 映射为粗 SID prefix，把 action chunk 映射为连续 SID suffix embedding 块。expert 本身随机初始化，使用 flow matching、码本分类和推荐排序目标训练；**不对 expert 做 teacher distillation**。

为了把 T5 历史编码器替换成小 MLP 而争取更高速度，当前方案另有一个可选的**编码器表征对齐项**：冻结已训练的 T5/MTP 编码器作为 teacher，让小编码器匹配 token-level hidden states。这个蒸馏针对的是被压缩的历史编码器，不是 π0.5 的低层 expert。它是否能提高 test 质量仍需消融；下面的两组现有结果配置不同，不能用于单独判断蒸馏的因果效果。匹配配置的蒸馏权重 0 对照正在运行，完成后补入。

## 训练配方

1. **离散 warm start**：从已有 MTP checkpoint 初始化 prefix head、并行 suffix heads、码本 embedding、item projection 和词表 embedding。
2. **新增快速编码器**：小型 global-context MLP 编码历史 SID，不执行 self-attention；用直接 next-SID 监督共同训练。
3. **新增低层 expert**：expert 随机初始化，以 \((h,g)\) 和带噪 suffix chunk 为输入，优化 flow matching 向量场；再加 suffix codebook CE 与 item-level 对比损失。
4. **可选 teacher 表征项**：训练快速编码器时冻结 T5/MTP encoder，最小化有效 token 上的 hidden-state MSE。该项只服务于编码器压缩，不用于训练 expert。

训练目标写成：

\[
\mathcal L=\lambda_{pre}\mathcal L_{prefix}
 +\lambda_{par}\mathcal L_{parallel\ suffix}
 +\lambda_{fm}\mathcal L_{flow}
 +\lambda_{code}\mathcal L_{codebook}
 +\lambda_{rank}\mathcal L_{item}
 +\lambda_{enc}\mathcal L_{encoder\ align}.
\]

## 为什么值得做，以及当前不能声称什么

- 相比 TIGER beam decoding，SID 全位不再逐个调用主干解码器，低层 suffix 由一次小 MLP chunk 专家输出；还去掉了高开销通用 tokenizer 和多层 self-attention 编码器，因此速度实测显著提高。
- 相比独立多头预测，专家以粗语义 prefix 为条件，目标是恢复 \(c_2,c_3\) 之间的联合偏好；并行头继续提供直接的离散 SID 概率，item-level loss 与目录内全物品打分负责推荐目标。
- 相比 SID-MLP，研究假设是“保留并行离散先验，再增加 prefix-conditioned joint suffix expert”，而不是仅靠位置 MLP 头复现 teacher。SID-MLP 的论文结果来自 Amazon Reviews 2023 与其四位 SID 协议；我们的 RecBoard Beauty 是 Amazon 2014、三位 SID。当前未在相同 SID、数据和硬件设置下跑通 SID-MLP，因此只能说结构目标不同，不能用两篇论文的公开倍数直接论证优劣。
- 目前 test 质量低于 MTP/TIGER，说明轻量编码器和 suffix flow 还没有保住效果。下一步优先做编码器蒸馏的配对消融、降低 flow 分数对 MTP 离散先验的干扰，并评估 prefix 条件方式；只有 test 质量追平后，速度比较才构成有效的质量—延迟优势。

## 已有实测

| 版本 | Fast encoder | Enc. 表征项 | suffix 头 | valid NDCG@10 | test NDCG@10 | 与 MTP test 0.0251 对照 |
|---|---:|---:|---:|---:|---:|---:|
| π-SID fast v1，30 epoch | 1 层、hidden 64 | 无 | 无（仅 flow codebook） | 0.0260 | 0.0205 | -0.0046 |
| π-SID distill v1，30 epoch | 2 层、hidden 128 | MSE，权重 1.0 | MTP 并行 suffix heads | 0.0279 | 0.0197 | -0.0054 |

注意：这两行**不是控制变量消融**，模型容量、suffix heads 等同时变化。权重 0 的同配置对照正在运行。

训练后速度测量使用同一采样用户和 `batch=96`，TIGER 为 `beam=30`、fp32、无 Trie，π-SID 包含快速分词、MLP 编码、一次 suffix expert 和全目录打分。TIGER P50=1.4464 ms，π-SID distill v1 P50=0.04445 ms，P95=0.04570 ms，加速 32.5×。TIGER 只生成 beam 候选，π-SID 对 12,101 个目录物品打分，输出工作量不同；这个数字是当前工程设置下的端到端延迟，不等同于同一检索计算量下的学术公平对比。结果文件：`/data/fszhang/RecBoard-master/HiFlow-SID/results/pisid_fast_speed.json`。

## 运行入口

模型代码：`/data/fszhang/RecBoard-master/HiFlow-SID/hiflow_lib/model_hiflow.py`  
训练入口：`/data/fszhang/RecBoard-master/HiFlow-SID/train_hiflow.py`  
严格速度脚本：`/data/fszhang/RecBoard-master/HiFlow-SID/scripts/bench_pisid_fast.py`  
profile 脚本：`/data/fszhang/RecBoard-master/HiFlow-SID/scripts/profile_pisid_fast.py`

当前 30 轮蒸馏版 checkpoint：`/data/fszhang/RecBoard-master/HiFlow-SID/logs/HiFlow-SID/Amazon2014Beauty_550_LOU/pisid-distill-v1/best.pt`。
