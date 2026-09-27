# LARS-CPQ：完全独立于 SDQ 的 LARS 组合模式语义 ID 学习

> 状态：候选新方案（Proposal），尚未完成正式实验验证  
> 更新时间：2026-09-27  
> 目标：沿方向 5 的纯 LARS 路线训练 Semantic ID tokenizer，在不依赖 SDQ-VAE、SDQ checkpoint、SDQ codebook、SDQ diffusion、SDQ SID 的前提下，提高最终生成式推荐的 NDCG / Recall。

---

## 0. 一句话概括

**现有 LARS-only-HRQ 用 LAR 联合选出了多个方向和系数，但最后每层 SID 只保留绝对系数最大的一个方向，导致 LARS 最有价值的“组合结构”没有真正进入离散 SID。**

LARS-CPQ 的核心改动是：

\[
\boxed{
\text{让一个 SID token 不再表示一个单独方向，而是表示一个可学习的 LARS 稀疏组合模式}
}
\]

每个组合模式由少量基础方向、符号和共享系数组成。训练和残差递推只能使用这个 token 对应的固定组合，禁止继续使用每个样本独有的连续 LARS 系数作为隐藏信息通道。

在此基础上，再加入一个**只读取硬 SID 表示的训练期推荐辅助头**，让组合模式不仅重构文本语义，还保留对 next-item prediction 有用的差异。

最终推理阶段只保留：

\[
\text{item}\rightarrow(s_1,s_2,s_3)\rightarrow\text{原版 T5}
\]

不增加 reranker，不增加 support score，不增加二阶段排序，也不在 T5 推理时运行 LARS。

---

# 1. 研究目标与约束

## 1.1 研究问题

方向 5 要回答的问题不是：

> 在 SDQ 上加一个 LARS loss 能不能更好？

而是：

> **LARS 本身能不能成为一个优秀的生成式推荐 Semantic ID tokenizer？**

即从 sentence-T5 等物品语义特征出发，通过 LARS 的稀疏方向选择与联合系数优化，直接产生高质量的三层离散 SID。

最终评价标准不是 tokenizer reconstruction loss，而是：

\[
\boxed{
\text{相同 T5、相同训练协议、相同 beam=30 下，
LARS SID 是否产生更高的 NDCG / Recall}
}
\]

## 1.2 方法独立性要求

LARS-CPQ 主方法不得使用：

- SDQ checkpoint；
- SDQ codebook；
- SDQ sid_vocab.json；
- SDQ graph diffusion / inverse diffusion；
- SDQ Sinkhorn assignment；
- SDQ quantizer.py；
- SDQ warm-start；
- SDQ SID anchor；
- SDQ reconstruction / commitment objective 作为隐式初始化。

允许使用：

- 与其他 tokenizer 相同的原始 item textual feature；
- RecBoard 相同的数据划分；
- train interactions；
- 相同的 T5 模型与训练协议；
- 相同的 SID 长度与 vocabulary budget；
- 相同的 collision converter；
- SDQ-VAE 作为**外部实验 baseline**。

因此 SDQ 与本方法只有“对比关系”，没有“方法依赖关系”。

---

# 2. 当前纯 LARS 基线：LARS-only-HRQ

当前独立实现位于：

~~~text
/data/fszhang/RecBoard-master/SDQ-LARS/train_lars_only_vae.py
~~~

它已经满足“完全不依赖 SDQ tokenizer”的基本要求。

当前流程可概括为：

\[
x_i
\rightarrow
z_i=E_\theta(x_i)
\rightarrow
\text{LAR residual quantization}
\rightarrow
(s_{i1},s_{i2},s_{i3})
\]

对第 \(\ell\) 层残差 \(r_{i\ell}\)，运行预算截断 LAR：

\[
a_{i\ell}
=
\operatorname{LAR}(r_{i\ell},D_\ell),
\]

其中：

- \(D_\ell\in\mathbb R^{M\times d}\) 为第 \(\ell\) 层方向字典；
- 当前 \(M=256\)；
- \(d=128\)；
- \(a_{i\ell}\in\mathbb R^M\) 是稀疏有符号系数；
- LAR budget 当前为 3。

现有代码随后执行：

\[
s_{i\ell}
=
\arg\max_k |a_{i\ell,k}|,
\]

取出：

\[
\alpha_{i\ell}
=
a_{i\ell,s_{i\ell}},
\]

并使用：

\[
q_{i\ell}
=
\alpha_{i\ell}D_\ell[s_{i\ell}]
\]

做连续重构和残差递推。

最终 SID 却只有：

\[
SID_i=(s_{i1},s_{i2},s_{i3}).
\]

---

# 3. 当前 LARS-only-HRQ 的关键瓶颈

## 3.1 LARS 找到的是“组合”，SID 保存的却只是“最大方向”

例如：

\[
a_A=[0.62,0.18,0.27,0,\cdots],
\]

\[
a_B=[0.62,-0.18,-0.27,0,\cdots].
\]

两个物品的稀疏组合明显不同。

但现有编码：

\[
\arg\max|a_A|
=
\arg\max|a_B|
=1.
\]

因此这一层都会输出相同的方向编号。

这会浪费 LAR 相比 nearest-centroid / greedy matching 最有价值的信息：

1. 多个 active directions；
2. active direction 之间的相对强度；
3. 系数符号；
4. LAR 等角更新产生的联合组合结构。

换句话说，LARS 做了一个比最终 SID 更丰富的推断，但在最后离散化时把大部分结果重新丢掉了。

## 3.2 训练阶段存在连续系数信息旁路

当前训练重构使用：

\[
q_{i\ell}=\alpha_{i\ell}D_\ell[s_{i\ell}],
\]

其中 \(\alpha_{i\ell}\) 是每个 item 自己的连续系数。

但最终 T5 只能看到 token \(s_{i\ell}\)，看不到这个连续值。

因此存在：

\[
\boxed{
\text{training representation information}
>
\text{exported SID information}
}
\]

一种可能的失败情况是：

- tokenizer reconstruction loss 明显下降；
- 连续表示很好；
- 但真正导出的 SID 并没有获得这些信息；
- T5 recommendation 因而没有同步提升。

新方法必须尽量满足：

\[
\boxed{
\text{训练时真正参与硬量化的主要信息}
=
\text{SID token 所能确定的信息}
}
\]

## 3.3 硬 bincount usage entropy 不能提供有效梯度

当前 LARS-only 代码通过硬 ID 统计：

\[
p_k=
\frac{\operatorname{count}(s_i=k)}{N}
\]

并计算 entropy regularization。

但 hard argmax + bincount 对 encoder / dictionary 不可微，因此该项主要是数值诊断，并不能可靠地推动 code usage balance。

后续应使用 soft assignment 的 batch marginal 来建立可微 usage objective，而最终前向仍采用 hard token。

## 3.4 训练编码与导出编码应完全一致

所有影响 SID 的预处理，例如：

\[
z=\operatorname{normalize}(E_\theta(x)),
\]

必须由统一的 encode() 接口执行。

训练、validation audit、export_vocab 不能分别写不同路径，否则即使理论上尺度变化不改变某些 LAR 选择，在系数截断、数值误差和多层残差下也可能出现不一致。

---

# 4. 新方法总览：LARS-CPQ

方法名称：

**LARS-CPQ: LARS Compositional Pattern Quantization**

中文：

**基于 LARS 稀疏组合模式的层次语义 ID 量化**

核心思想：

> LAR 仍负责发现“一个物品应该由哪些方向共同解释”；  
> 但 SID token 不再直接等于某一个方向的编号，而是对应一个**可复用的稀疏方向组合模式**。

因此每层存在两种结构：

### 基础方向字典

\[
D_\ell
=
[d_{\ell1},d_{\ell2},\ldots,d_{\ell M}]
\in\mathbb R^{M\times d}.
\]

LAR 在这个字典上进行 active-set selection。

### 组合模式词表

\[
B_\ell
=
[b_{\ell1},b_{\ell2},\ldots,b_{\ell K}]
\in\mathbb R^{K\times M}.
\]

其中每个：

\[
b_{\ell k}\in\mathbb R^M
\]

是一个稀疏有符号组合。

推荐默认：

\[
M=256,\qquad K=256.
\]

这样仍然保持每层只有 256 个 SID token。

每一个 pattern 最多保留 \(m_p\) 个非零方向：

\[
\|b_{\ell k}\|_0\le m_p.
\]

首轮建议：

\[
m_p=3.
\]

---

# 5. 为什么“组合模式”比“最大方向”更适合 LARS？

假设当前残差由：

\[
r
\approx
0.62d_1+0.18d_2+0.27d_3
\]

解释。

现有 LARS-only 会输出：

\[
SID=1.
\]

新的 LARS-CPQ 可以学习到一个 pattern：

\[
b_7=
[0.62,0.18,0.27,0,\cdots],
\]

于是：

\[
SID=7
\]

真正对应的是：

\[
c_7=b_7D.
\]

另一个物品：

\[
r'
\approx
0.62d_1-0.18d_2-0.27d_3
\]

可以匹配另一个 pattern：

\[
b_{21}=
[0.62,-0.18,-0.27,0,\cdots].
\]

于是：

\[
SID'=21.
\]

两个 item 仍然各自只输出一个 token，但这个 token 已经保留了部分：

- active-set 组合；
- coefficient sign；
- relative contribution。

注意：

> 这并没有增加 token 信息容量。

每层仍然只有 256 个离散状态。

LARS-CPQ 做的是：在固定的 256 个状态中，寻找比“256 个单独方向”更适合表示 LARS 结果的 256 个**组合原型**。

---

# 6. Pattern 参数化：禁止 item-specific coefficient bypass

为了让“给定 token 就能恢复训练使用的硬量化表示”，推荐把 pattern 写成：

\[
\tilde a_{\ell k}
=
g_{\ell k}
\bar b_{\ell k},
\]

其中：

\[
\|\bar b_{\ell k}\|_2=1,
\]

且：

\[
\|\bar b_{\ell k}\|_0\le m_p.
\]

\(g_{\ell k}>0\) 是这个 token 的**共享幅度**。

于是 token 对应的向量为：

\[
\boxed{
c_{\ell k}
=
\tilde a_{\ell k}D_\ell.
}
\]

对 item \(i\)，如果选择：

\[
s_{i\ell}=k,
\]

那么它的量化向量严格为：

\[
\boxed{
q_{i\ell}=c_{\ell k}.
}
\]

这里不存在每个 item 独有的：

\[
\alpha_i.
\]

也就是说：

\[
s_{i\ell}=k
\Rightarrow
q_{i\ell}
\text{完全由共享参数确定}.
\]

这就是整个方法最重要的 hard bottleneck。

---

# 7. LAR 内层仍然保留：它不是普通 VQ

对于当前残差 \(r_{i\ell}\)，先运行真正的 LAR：

\[
a_{i\ell}
=
\operatorname{LAR}_{m_\ell}
(
\operatorname{sg}(r_{i\ell}),
\operatorname{sg}(D_\ell)
).
\]

这里继续使用方向 5 当前已经验证过的 batched budget-truncated LAR：

- active direction 按 correlation 进入；
- active set 中的系数沿 equiangular direction 联合更新；
- 不把它写成 OMP；
- 不把它写成精确 Lasso；
- 不需要对 LAR solver 反向传播。

因此 LAR 继续扮演：

\[
\boxed{
\text{inner sparse inference}
}
\]

的角色。

而神经网络参数 / 字典参数通过外层 objective 更新。

---

# 8. 从 LAR 系数到组合 token

## 8.1 两个空间同时匹配

仅在 coefficient space 中找最近 pattern 可能出现问题：

两个 coefficient pattern 接近，并不一定代表在当前 dictionary 下真实 reconstruction 同样接近。

因此定义第 \(\ell\) 层 pattern energy：

\[
\boxed{
E_{i\ell k}
=
\frac{
\|r_{i\ell}-c_{\ell k}\|_2^2
}{
\sigma_{r,\ell}^2+\epsilon
}
+
\lambda_p
\frac{
\|a_{i\ell}-\tilde a_{\ell k}\|_2^2
}{
\sigma_{a,\ell}^2+\epsilon
}.
}
\]

其中：

- 第一项：真实 residual reconstruction；
- 第二项：LARS coefficient-pattern consistency；
- \(\sigma_r,\sigma_a\)：训练期运行统计，用来做尺度归一；
- \(\epsilon\)：防止早期方差过小。

最终 hard token：

\[
\boxed{
s_{i\ell}
=
\arg\min_k E_{i\ell k}.
}
\]

这样 token 既要“像 LARS 找到的组合”，又必须真正能解释当前 residual。

## 8.2 为什么不能只量化 \(a\)？

预算截断 LAR 本身并不保证：

\[
\|r-aD\|
\]

在所有样本上都比其他硬表示更低。

因此不能无条件把 LARS coefficient 当成真值。

真实残差 reconstruction term 可以防止：

> pattern 很像 LAR coefficient，但实际对 residual 的解释很差。

---

# 9. Hierarchical residual quantization

第 1 层：

\[
r_{i1}=z_i.
\]

选择 token：

\[
s_{i1}
=
\arg\min_k E_{i1k}.
\]

得到：

\[
q_{i1}=c_{1,s_{i1}}.
\]

下一层 residual：

\[
\boxed{
r_{i2}
=
r_{i1}
-
\operatorname{sg}(q_{i1}).
}
\]

继续：

\[
r_{i3}
=
r_{i2}
-
\operatorname{sg}(q_{i2}).
\]

最终：

\[
SID_i
=
(s_{i1},s_{i2},s_{i3}).
\]

总量化表示：

\[
q_i
=
q_{i1}+q_{i2}+q_{i3}.
\]

最重要的约束是：

\[
\boxed{
r_{i,\ell+1}
\text{只能减去 token 对应的固定 }q_{i\ell},
\text{不能减去 item-specific }a_{i\ell}D_\ell.
}
\]

否则连续 coefficient information bypass 会重新出现。

---

# 10. Hard-forward / soft-backward

token 选择是离散 argmin。

为了训练 encoder 与 pattern，可以定义：

\[
p_{i\ell k}
=
\operatorname{softmax}
(
-E_{i\ell k}/\tau_\ell
).
\]

hard one-hot：

\[
h_{i\ell}
=
\operatorname{onehot}
(
\arg\max_k p_{i\ell k}
).
\]

Straight-through assignment：

\[
\tilde h_{i\ell}
=
\operatorname{sg}
(
h_{i\ell}-p_{i\ell}
)
+
p_{i\ell}.
\]

前向数值上：

\[
\tilde h=h.
\]

所以实际使用 hard token。

反向时：

\[
\frac{\partial\tilde h}{\partial p}\ne0.
\]

于是：

\[
q_{i\ell}
=
\sum_k
\tilde h_{i\ell k}c_{\ell k}.
\]

这种 STE 是有偏梯度近似，因此它是工程训练机制，不应在论文中宣称为精确离散梯度。

---

# 11. 组合模式如何初始化

如果 \(B_\ell\) 从完全随机的 256 × 256 coefficient matrix 开始训练，早期会非常不稳定。

推荐采用 **LARS-derived initialization**。

## Phase A：LARS dictionary warm-up

前 \(E_w\) 个 epoch 只训练：

- encoder；
- decoder；
- base direction dictionary \(D_\ell\)；
- LAR reconstruction objective。

推荐首轮：

\[
E_w=10.
\]

收集每层 LAR coefficient：

\[
\mathcal A_\ell
=
\{a_{1\ell},a_{2\ell},\ldots\}.
\]

## Phase B：从真实 LAR path 初始化 pattern

对 coefficient 先做：

1. 保留 top-\(m_p\) absolute coefficients；
2. 保留 sign；
3. 对系数向量做 norm normalization；
4. 在 coefficient space 中选择 \(K=256\) 个 prototype。

实现可以优先使用：

- farthest-point / kmeans++ seed；
- 随后少量 coefficient-space clustering。

这里 clustering 只是初始化，不是最终 tokenizer。

每个 prototype 初始化：

\[
\bar b_{\ell k},
\]

共享 magnitude 初始化为该 cluster 中：

\[
g_{\ell k}
=
\operatorname{median}
(
\|a_{i\ell}\|_2
).
\]

初始化后进入端到端训练。

---

# 12. Pattern sparsity 的实现

首版不建议让 support 从第一天开始连续变化。

推荐：

### 初始化时确定 support

\[
S_{\ell k}
=
\operatorname{TopM}
(
|\bar b_{\ell k}|
).
\]

训练时只更新 support 内的参数：

\[
b_{\ell k,j}=0,\qquad j\notin S_{\ell k}.
\]

这样：

- 训练更稳定；
- pattern 更可解释；
- 真正保持“稀疏组合”；
- 不需要 differentiable top-k。

后续如果首版有效，再研究：

- periodic support refresh；
- learnable hard-concrete support；
- group sparse penalty。

这些不进入第一版正式方法，避免一次引入过多变量。

---

# 13. Dictionary 外层更新

LAR solver 本身 detached。

对于：

\[
a_{i\ell}
=
\operatorname{sg}
(
\operatorname{LAR}(r,D)
),
\]

定义：

\[
\boxed{
\mathcal L_{\mathrm{LAR}}
=
\frac{1}{NL}
\sum_{i,\ell}
\|
\operatorname{sg}(r_{i\ell})
-
a_{i\ell}D_\ell
\|_2^2.
}
\]

这里：

- \(a\) stop-gradient；
- residual target stop-gradient；
- \(D_\ell\) 保留梯度。

这继续保持方向 5 的核心思想：

\[
\boxed{
\text{固定内层 sparse code，更新外层 dictionary}
}
\]

而不是把 LARS 降级成只用于日志分析的模块。

---

# 14. Pattern consistency objective

pattern 必须逼近训练样本中真实出现的 LAR sparse structure。

定义：

\[
\boxed{
\mathcal L_{\mathrm{pattern}}
=
\frac1{NL}
\sum_{i,\ell}
\|
\operatorname{sg}(a_{i\ell})
-
\tilde a_{\ell,s_{i\ell}}
\|_2^2.
}
\]

该项直接推动：

> 同一个 SID token 表示一类可复用的 LARS 稀疏组合。

与普通 codebook VQ 的区别不只是：

\[
c_k
\]

是一个向量。

这里它具有明确的结构：

\[
\boxed{
c_k
=
\tilde a_kD.
}
\]

其中：

- \(D\)：共享基础方向；
- \(\tilde a_k\)：LARS-derived sparse composition。

---

# 15. Hard reconstruction objective

完整三层 hard token 表示：

\[
q_i
=
\sum_\ell q_{i\ell}.
\]

decoder：

\[
\hat x_i
=
G_\psi(q_i).
\]

如果输入 feature 已归一化，推荐继续使用 cosine reconstruction：

\[
\boxed{
\mathcal L_{\mathrm{recon}}
=
\frac1N
\sum_i
\left(
1-\hat x_i^\top x_i
\right).
}
\]

注意：

decoder 输入只能来自：

\[
(s_{i1},s_{i2},s_{i3})
\]

对应的 shared pattern vectors。

不能将：

- 原始 \(z_i\)；
- item-specific coefficient；
- item ID embedding；

拼接给 decoder。

---

# 16. 可微 code usage balance

对 soft assignments：

\[
p_{i\ell}
\in\mathbb R^K,
\]

batch marginal：

\[
\bar p_{\ell k}
=
\frac1N
\sum_i
p_{i\ell k}.
\]

希望它不要严重 collapse，可以使用：

\[
\boxed{
\mathcal L_{\mathrm{usage}}
=
\frac1L
\sum_{\ell}
D_{\mathrm{KL}}
(
\bar p_\ell
\|
U_K
).
}
\]

其中：

\[
U_K(k)=1/K.
\]

该项与当前 hard bincount 不同：

\[
\bar p
\]

对 assignment logits 可微，可以真正推动 encoder / pattern 改变。

不要求严格 uniform。

因此首轮 \(\lambda_u\) 应保持很小，只防止极端 collapse。

---

# 17. 推荐感知：只通过硬 SID 的训练期辅助头

仅靠 reconstruction 仍然不能保证最终 T5 recommendation 更好。

因此 LARS-CPQ 增加一个：

\[
\boxed{
\text{SID-only recommendation auxiliary objective}
}
\]

关键不是“再训练一个推荐模型”，而是：

> 让 tokenizer 知道哪些离散差异对真实 next-item behavior 有用。

## 17.1 SID item representation

定义：

\[
e_i^{SID}
=
\operatorname{LN}
\left(
W_q
[q_{i1}\Vert q_{i2}\Vert q_{i3}]
\right).
\]

也可以先用更简单版本：

\[
e_i^{SID}
=
\operatorname{LN}
(
q_{i1}+q_{i2}+q_{i3}
).
\]

第一版推荐先用 sum 版本，减少额外参数。

## 17.2 历史端也只能读取 SID

用户历史：

\[
(i_1,i_2,\ldots,i_t).
\]

训练期辅助模型只能获得：

\[
(
e_{i_1}^{SID},
e_{i_2}^{SID},
\ldots,
e_{i_t}^{SID}
).
\]

严格禁止给它：

- raw item ID embedding；
- sentence-T5 feature；
- continuous encoder output \(z_i\)；
- item-specific LARS coefficient。

否则辅助模型可能绕开 SID，自行学习推荐任务。

## 17.3 推荐第一版使用轻量 GRU

使用一层 GRU：

\[
h_t
=
\operatorname{GRU}
(
e_{i_1}^{SID},\ldots,e_{i_t}^{SID}
).
\]

score：

\[
f(i|h_t)
=
h_t^\top W_r e_i^{SID}.
\]

训练 next-item：

\[
\boxed{
\mathcal L_{\mathrm{rec}}
=
-\log
\frac{
\exp f(i^+|h)
}{
\exp f(i^+|h)
+
\sum_{j\in\mathcal N}
\exp f(j|h)
}.
}
\]

负样本使用：

- in-batch negatives；
- 或固定数量 random negatives。

为了方法归因干净，首版不引入另一个强 teacher。

---

# 18. 为什么推荐辅助头不会改变最终推理协议？

这个 GRU 只存在于 tokenizer training。

训练结束后：

1. 丢弃 GRU；
2. 丢弃 decoder；
3. 丢弃 LAR solver；
4. 只导出三层 hard SID；
5. 使用与基线完全相同的 T5 重新训练；
6. test 时仍然只运行 T5 beam search。

因此最终线上链路仍然是：

\[
\boxed{
\text{history SID}
\rightarrow
\text{T5}
\rightarrow
\text{beam=30}
}
\]

没有额外 inference latency。

---

# 19. 完整训练目标

第一版建议不要把历史上所有 gate / anchor / collision loss 再全部加入。

保持目标尽量干净：

\[
\boxed{
\mathcal L
=
\lambda_{\mathrm{recn}}
\mathcal L_{\mathrm{recon}}
+
\lambda_{\mathrm{lar}}
\mathcal L_{\mathrm{LAR}}
+
\lambda_{\mathrm{pat}}
\mathcal L_{\mathrm{pattern}}
+
\lambda_{\mathrm{rec}}
\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{use}}
\mathcal L_{\mathrm{usage}}
+
\lambda_{\mathrm{div}}
\mathcal L_{\mathrm{dir-div}}.
}
\]

其中：

### Reconstruction

\[
\mathcal L_{\mathrm{recon}}
\]

保证最终 hard SID 仍能表达 item semantics。

### LARS dictionary learning

\[
\mathcal L_{\mathrm{LAR}}
\]

让 base directions 按真实 LAR sparse inference 更新。

### Pattern consistency

\[
\mathcal L_{\mathrm{pattern}}
\]

把 item-specific LAR coefficient 压缩到有限的离散 composition prototypes。

### Recommendation

\[
\mathcal L_{\mathrm{rec}}
\]

让 hard SID 保留 next-item prediction 有用的信息。

### Usage balance

\[
\mathcal L_{\mathrm{usage}}
\]

避免 256 个 pattern 极度集中。

### Direction diversity

对于归一化方向：

\[
\hat D_\ell
=
\operatorname{normalize}(D_\ell),
\]

可保留轻量：

\[
\mathcal L_{\mathrm{dir-div}}
=
\operatorname{mean}
\left[
\operatorname{ReLU}
(
|\hat D\hat D^T|-\delta
)
\right].
\]

它只防止 base directions 大规模重复。

---

# 20. 第一版推荐超参数

以下只是第一轮起始配置，不应当在论文中写成理论最优。

| 参数 | 建议值 |
|---|---:|
| levels | 3 |
| base directions \(M\) | 256 |
| SID patterns \(K\) | 256 |
| code dim | 128 |
| LAR budget | 3 / 3 / 3 |
| pattern support | 3 |
| tokenizer epochs | 100 |
| warm-up | 10 |
| batch | 512 |
| optimizer | AdamW |
| encoder lr | \(5\times10^{-4}\) |
| dictionary lr | \(5\times10^{-4}\) |
| pattern lr | \(2\times10^{-4}\) |
| temperature start | 0.15 |
| temperature end | 0.05 |
| rec negative count | 128 或 in-batch |
| \(\lambda_{recon}\) | 1.0 |
| \(\lambda_{LAR}\) | 0.10 |
| \(\lambda_{pattern}\) | 0.10 |
| \(\lambda_{rec}\) | 0 → 0.10 线性 warm-up |
| \(\lambda_{usage}\) | 0.005 |
| \(\lambda_{div}\) | 0.01 |

推荐前 10 epoch：

\[
\lambda_{\mathrm{rec}}=0.
\]

pattern 初始化后再逐渐增加推荐目标。

---

# 21. 推荐训练时间表

## Epoch 1–10：Pure LARS warm-up

训练：

- encoder；
- decoder；
- base directions。

目标：

\[
\mathcal L
=
\mathcal L_{\mathrm{recon-warm}}
+
\lambda_{lar}\mathcal L_{\mathrm{LAR}}.
\]

此阶段不产生正式 SID。

主要目的是让 LAR coefficient distribution 稳定。

## Epoch 10：Pattern initialization

收集全量或大规模 coefficient bank：

\[
\mathcal A_1,\mathcal A_2,\mathcal A_3.
\]

初始化三层 256 个 composition patterns。

## Epoch 11–30：Hard pattern stabilization

启用：

- hard pattern assignment；
- pattern consistency；
- hard SID reconstruction；
- soft usage。

推荐 loss 从 0 缓慢升高。

## Epoch 31–100：Full training

完整目标训练。

保持：

\[
\text{hard forward}
+
\text{soft backward}.
\]

最终固定导出 epoch 100 SID。

除非后续有严格设计的 train-only tokenizer validation，否则不要根据 test 指标选择 tokenizer epoch。

---

# 22. 推理 / SID 导出

导出流程必须和训练完全一致：

~~~text
x
 -> encode_and_normalize()
 -> residual level 1
 -> LAR coefficients
 -> pattern energy
 -> hard pattern id s1
 -> q1 = pattern_vector[s1]
 -> residual -= q1

 -> level 2
 -> s2
 -> q2
 -> residual -= q2

 -> level 3
 -> s3
~~~

最终：

~~~text
item_0:
  <sid_0_s1>
  <sid_1_s2>
  <sid_2_s3>
~~~

禁止导出阶段重新：

- 使用 argmax(abs(LAR coef))；
- 使用不同 normalization；
- 使用 item-specific amplitude；
- 使用另一套 assignment。

---

# 23. 与 LARS-only-HRQ 的本质区别

| 模块 | LARS-only-HRQ | LARS-CPQ |
|---|---|---|
| LAR active set | 有 | 有 |
| LAR joint coefficient | 有 | 有 |
| 最终 token | 最大绝对系数方向 | 稀疏组合 pattern |
| coefficient sign | 大部分丢失 | pattern 中保留 |
| 多方向结构 | 大部分丢失 | pattern 中保留 |
| item-specific amplitude | 训练使用 | 禁止 |
| hard token 决定完整量化向量 | 否 | **是** |
| usage regularizer | hard bincount | soft differentiable |
| recommendation-aware tokenizer | 无 | **SID-only auxiliary head** |
| SDQ dependency | 无 | **无** |

---

# 24. 与旧 LARS-SDQ 系列的区别

main / gated / trust / utility / CP / APC 的共同特点是：

> LARS 被加入 SDQ 或 SDQ SID/T5 相关训练框架。

LARS-CPQ 的定位完全不同：

\[
\boxed{
\text{不是“改善 SDQ”，而是“独立设计 LARS tokenizer”。}
}
\]

方法中没有：

- diffusion；
- inverse diffusion；
- SDQ codebook；
- SDQ SID anchor；
- SDQ warm start；
- prefix calibration against SDQ SID。

因此论文叙述也应该围绕：

**Sparse Compositional Semantic Tokenization**

而不是：

**LARS-enhanced SDQ**。

---

# 25. 最关键的消融实验

为了证明真正有效的是 LARS，而不是简单增加参数或推荐 supervision，以下消融必须保留。

## A0. 当前 LARS-only-HRQ

现有纯 LARS 实现。

回答：

> 新方法是否超过自己的独立强基线？

## A1. LARS-only-HRQ-Fix

只修复：

- train/export normalization；
- soft usage；
- 其他明显工程不一致。

不增加 composition pattern。

回答：

> 提升是不是只来自工程修复？

## A2. LARS-CPQ w/o Rec

使用组合 pattern，但：

\[
\lambda_{rec}=0.
\]

回答：

> 保留 LAR composition 本身有没有价值？

## A3. Full LARS-CPQ

完整方法。

回答：

> recommendation-aware hard bottleneck 是否进一步提升？

## A4. Vanilla structured codebook + same Rec head

去掉 LAR coefficient consistency，仅学习 256 个普通 codeword，并使用相同辅助推荐头与参数预算。

回答：

> 提升是不是仅来自 rec auxiliary loss？

## A5. OMP-CPQ

LAR 替换成匹配预算 OMP。

回答：

> LAR 的 equiangular joint coefficient update 是否必要？

## A6. Greedy-correlation CPQ

每一步只选当前最大相关方向，不做 LAR joint update。

回答：

> 性能是否真正来自 LARS，而不只是 sparse dictionary？

---

# 26. 必须监控的 tokenizer 指标

不能只看 reconstruction。

每个 dataset 记录：

### Hard SID collision

\[
1-
\frac{
|\{SID_i\}|
}{
N
}.
\]

同时记录 converter 加 collision token 前后的长度统计。

### Per-level usage

每层：

- unique codes；
- entropy；
- perplexity：

\[
PPL_\ell
=
\exp(H(p_\ell)).
\]

这里明确称为：

**code usage perplexity**，

避免和语言模型 perplexity 混淆。

### Prefix statistics

记录：

- unique prefix@1；
- unique prefix@2；
- average items / prefix；
- max items / prefix；
- head 10% prefix coverage。

### LAR-to-pattern distortion

\[
D_{pat}
=
\mathbb E
\|a_{i\ell}-\tilde a_{\ell,s}\|^2.
\]

回答：

> 256 个 pattern 到底保留了多少 LAR coefficient information？

### Hard reconstruction

\[
D_{hard}
=
\mathbb E
\|z_i-q_i\|^2.
\]

### Auxiliary recommendation metric

在只用 train-derived split 的情况下记录：

- auxiliary Hit@10；
- auxiliary NDCG@10。

这只是 tokenizer 诊断，不作为最终论文主结果。

---

# 27. 最终 T5 公平协议

LARS-CPQ SID 导出之后，必须从头训练 T5。

正式协议保持当前 RecBoard 口径：

- Amazon2014 Beauty / Sports / Toys；
- 相同 leave-two-out split；
- history maxlen=20；
- tokenizer 100 epochs；
- T5 200 epochs；
- seed=2025；
- train batch=512；
- eval batch=96；
- fp32；
- beam=30；
- apply-constrained-beam-search=False；
- 无 reranker；
- 无 support score；
- 无二次打分；
- validation NDCG@10 选择 T5 checkpoint；
- 读取该 checkpoint 对应 test 指标。

主报告：

- Recall/Hit@5；
- Recall/Hit@10；
- Recall/Hit@20；
- NDCG@5；
- NDCG@10；
- NDCG@20；
- MRR（若评估脚本支持）。

同时重新测：

- 完整 test set inference time；
- 保持与 TIGER-beam-30 fp32 batch96 的 40.83 s 口径一致。

注意：

40.83 s 是已有 Beauty baseline 的一次实测值，不是任何新 SID 自动继承的耗时。

---

# 28. 成功标准

第一阶段目标不是直接宣称 SOTA，而是依次回答：

## Gate 1：超过当前纯 LARS

\[
\boxed{
\text{LARS-CPQ}
>
\text{LARS-only-HRQ}
}
\]

三个数据集至少大部分稳定提升。

## Gate 2：证明 composition 有用

\[
\boxed{
\text{LARS-CPQ w/o Rec}
>
\text{LARS-only-HRQ-Fix}
}
\]

否则“组合模式”核心假设不成立。

## Gate 3：证明 recommendation auxiliary 有用

\[
\boxed{
\text{Full LARS-CPQ}
>
\text{LARS-CPQ w/o Rec}
}
\]

## Gate 4：证明 LARS 有独立贡献

\[
\boxed{
\text{Full LARS-CPQ}
>
\text{Vanilla codebook + same Rec head}
}
\]

如果 Gate 4 不成立，则论文不能把主要提升归因于 LARS。

## Gate 5：外部对照

最终再与：

- RQ-VAE/TIGER tokenizer；
- SDQ-VAE；
- 其他 SID tokenizer；

在同一 RecBoard 协议下比较。

SDQ 在这里仅是 baseline。

---

# 29. 潜在风险与对应措施

## 风险 1：组合 pattern 退化成普通 codebook

因为：

\[
C=BD,
\]

从函数表达能力上看，最终确实可以被视为结构化 codebook。

因此方法贡献不能写成：

> 首次把 codebook 分解成两个矩阵。

必须证明：

- pattern 是由 LAR coefficient path 初始化和持续约束的；
- LAR coefficient consistency 对性能有贡献；
- 替换为普通 codebook 后结果下降；
- 替换 LAR 为 OMP/greedy 后结果变化。

真正贡献应表述为：

> **利用 LARS active-set path 学习离散可复用的 sparse composition prototypes，使最终有限 SID token 保留多方向联合结构。**

## 风险 2：256 个 pattern 不足以覆盖 coefficient space

解决：

- 先保持 K=256 公平比较；
- 如果 distortion 明显过大，再做 K=512 的 capacity study；
- 正式主实验仍优先保持和现有 256 vocabulary 一致。

## 风险 3：推荐 loss 导致语义结构崩塌

解决：

- rec warm-up；
- \(\lambda_{rec}\) 从 0 逐渐增加；
- 保留 hard reconstruction；
- 监控 SID collision、usage entropy、semantic neighbor consistency。

## 风险 4：辅助 GRU 太强，掩盖 tokenizer

解决：

- 历史端和目标端都只能读取 SID representation；
- 禁止 item ID embedding；
- 禁止 raw semantic feature；
- 设置一层、hidden=128 的轻量模型；
- 用相同辅助头训练 vanilla codebook baseline。

## 风险 5：hard assignment 不稳定

解决：

- 先 warm-up dictionary；
- coefficient-derived pattern initialization；
- soft temperature annealing；
- pattern learning rate低于 encoder/dictionary；
- 必要时对 pattern 使用 EMA target，但不作为首轮默认。

---

# 30. 代码落地建议

建议不要继续把新逻辑塞进现有 train_lars_only_vae.py。

保留原文件作为强 baseline。

新增：

~~~text
RecBoard-master/SDQ-LARS/
├── train_lars_only_vae.py          # 保留：LARS-only-HRQ baseline
├── lars_solver.py                  # 复用：现有 LAR solver
├── lars_cpq_quantizer.py           # 新增：composition pattern quantizer
├── lars_cpq_rec.py                 # 新增：SID-only recommendation auxiliary head
├── train_lars_cpq.py               # 新增：完整 tokenizer trainer
├── run_lars_cpq.py                 # 新增：单数据集 runner
├── run_lars_cpq_all.sh             # 新增：Beauty/Sports/Toys
└── runs/
    └── <run_name>/
        └── lars_cpq/
            ├── Beauty/
            ├── Sports/
            └── Toys/
~~~

---

# 31. lars_cpq_quantizer.py 建议接口

~~~python
class LARSCPQ(nn.Module):
    def __init__(
        self,
        dim=128,
        levels=3,
        num_directions=256,
        num_patterns=256,
        lars_steps=3,
        pattern_support=3,
    ):
        ...

    def lar_infer(self, residual, level):
        # return detached item-specific LAR coefficients
        ...

    def pattern_energy(self, residual, lar_coef, level):
        # residual reconstruction + coefficient consistency
        ...

    def hard_assign(self, energy):
        # hard forward / soft backward
        ...

    def quantize_level(self, residual, level):
        # IMPORTANT:
        # q only comes from selected shared pattern
        ...

    def forward(self, z):
        # hierarchical residual quantization
        ...
~~~

必须有单元测试验证：

\[
SID_i=SID_j
\Rightarrow
q_i=q_j
\]

在同一层严格成立。

这是当前 LARS-only 不满足的关键性质之一。

---

# 32. 必做单元测试

## Test 1：同 token 同向量

两个不同输入只要产生相同 token：

~~~text
sid_a == sid_b
~~~

则：

~~~text
q_a == q_b
~~~

不能因为 item-specific amplitude 不同而变化。

## Test 2：符号 pattern 可区分

构造：

\[
a=[0.6,0.3,0],
\qquad
a'=[0.6,-0.3,0].
\]

当 pattern bank 包含这两个 prototype 时，应能分配不同 token。

## Test 3：导出一致

训练模式关闭后：

~~~text
model.encode_ids(x)
~~~

和：

~~~text
export_vocab()
~~~

必须产生完全相同的 ID。

## Test 4：无 SDQ import

自动扫描：

~~~text
quantizer.py
StructureDiffusionQuantizer
SDQ checkpoint
sid_vocab baseline
~~~

都不应成为本方法运行依赖。

config 中保留：

~~~json
"sdq_imports": false
~~~

## Test 5：recommendation head 无旁路

检查 auxiliary recommender 参数中不存在：

~~~text
nn.Embedding(num_items, ...)
~~~

candidate 与 history item representation 必须来自 hard SID。

---

# 33. 建议的第一轮实验顺序

不要一开始三个数据集完整 100+200 全跑。

## Step 1：Beauty tokenizer sanity

只训练 tokenizer。

必须满足：

- 不 collapse；
- unique_per_level 合理；
- collision 不异常；
- hard reconstruction 收敛；
- LAR-pattern distortion 持续下降。

## Step 2：Sports tokenizer sanity

Sports 是历史上 LARS-SDQ 最敏感的数据集。

如果这里：

- usage 严重集中；
- collision 快速升高；
- auxiliary rec 崩塌；

直接修 tokenizer，不进入 200 epoch T5。

## Step 3：短 T5 validation screening

只用于排除明显失败配置。

不使用 test 指标调参。

## Step 4：冻结正式配置

只根据：

- tokenizer train diagnostics；
- validation；

确定一套固定配置。

## Step 5：三个数据集完整正式 T5

再产生论文结果。

---

# 34. 论文级核心叙事

建议论文不要再从：

> “SDQ 很好，但我们给 SDQ 加 LARS。”

展开。

而是：

## Observation 1

Generative recommenders require a discrete item interface.

## Observation 2

Conventional residual quantization represents each semantic token with one codeword.

## Observation 3

Sparse coding suggests that an item residual may be better characterized by a small **combination of directions**.

## Problem

Directly exporting the strongest sparse direction discards most of the LARS solution, while preserving item-specific continuous coefficients creates a train–serve representation mismatch.

## Proposed principle

\[
\boxed{
\text{Discretize sparse compositions, not individual sparse coefficients.}
}
\]

## LARS-CPQ

1. LARS discovers sparse active-direction structure;
2. sparse coefficient paths are compressed into reusable discrete composition prototypes;
3. each SID token deterministically maps to one shared composition vector;
4. hierarchical residual coding preserves coarse-to-fine capacity;
5. a SID-only auxiliary recommendation objective makes the discrete bottleneck behavior-aware;
6. all training-only modules are removed before generative recommendation.

这是比“给已有量化器加辅助损失”更独立、也更容易建立方法身份的路线。

---

# 35. 当前方法的核心创新点候选

在实验真正验证之前，只能作为 **candidate contributions**。

### Contribution 1：LARS-derived compositional SID

不是把 LAR 最大方向直接作为 token，而是将多方向有符号 sparse path 聚合成有限的 reusable discrete composition prototypes。

### Contribution 2：No-bypass discrete bottleneck

训练残差递推和重构严格使用 token 决定的共享组合，不使用 item-specific continuous coefficient，减少 tokenizer training objective 与最终 SID interface 的信息错位。

### Contribution 3：Recommendation-aware compositional quantization

通过只能读取 hard SID 表示的辅助 next-item objective，让组合模式同时保留 semantic reconstruction 和 recommendation utility。

### Contribution 4：Alternating sparse dictionary / discrete pattern learning

LAR 负责 inner sparse inference，神经网络和 dictionary/pattern 负责 outer optimization，保持方向 5 的核心稀疏方向学习思想。

---

# 36. 最终方法示意

~~~text
                 item textual feature x_i
                          |
                          v
                     Encoder E
                          |
                          v
                     latent z_i
                          |
                +---------+---------+
                |                   |
                v                   |
         LAR sparse inference       |
       a_i1 over base D_1           |
                |                   |
                v                   |
       match composition bank B_1   |
                |                   |
                v                   |
        hard token s_i1             |
                |                   |
                v                   |
      q_i1 = B_1[s_i1] D_1          |
                |                   |
                +--> residual r_i2 -+
                          |
                          v
                  repeat level 2
                          |
                          v
                  repeat level 3
                          |
                          v
                SID = (s1,s2,s3)
                    /         \
                   /           \
                  v             v
          semantic decoder   SID-only GRU
              train only      train only
                   \           /
                    \         /
                     \       /
                     tokenizer losses

After tokenizer training:

SID vocabulary
      |
      v
fresh original T5
      |
      v
beam=30 recommendation
~~~

---

# 37. 最终必须坚持的原则

整个方案最重要的四句话：

\[
\boxed{
\textbf{1. LARS 必须决定 token 结构，而不只是做 auxiliary loss。}
}
\]

\[
\boxed{
\textbf{2. 最终硬 SID 必须承载训练时真正使用的信息，不能靠连续系数旁路。}
}
\]

\[
\boxed{
\textbf{3. 推荐监督只能帮助训练 tokenizer，最终推理仍然只有 SID + 原版 T5。}
}
\]

\[
\boxed{
\textbf{4. SDQ-VAE 只作为实验 baseline，不参与 LARS-CPQ 的训练、初始化或推理。}
}
\]

---

# 38. 下一步实现优先级

按照风险最低到最高：

1. 修复并冻结 LARS-only-HRQ baseline；
2. 实现 shared composition pattern；
3. 完成 no-bypass hard residual recursion；
4. 实现 coefficient-derived initialization；
5. 加 soft usage；
6. 跑 LARS-CPQ w/o Rec；
7. 确认 composition 本身有效后，再加入 SID-only recommendation head；
8. 做 Vanilla-codebook / OMP / greedy 消融；
9. 最后运行 Beauty / Sports / Toys 的正式 T5；
10. 使用统一 TIGER beam=30、fp32、batch96 口径重新测质量与速度。

在第 6 步没有证明组合模式本身有效以前，不建议继续叠加大量复杂 loss。

---

## 结论

LARS-CPQ 的核心不是“把更多 loss 加到 LARS 上”，而是重新定义：

\[
\boxed{
\text{LARS 的连续 sparse solution 应该怎样变成一个真正有信息的离散 SID。}
}
\]

现有 LARS-only-HRQ 已经证明可以完全脱离 SDQ 构造三层 SID。

下一步最值得验证的问题是：

> **能否把 LARS 发现的多方向联合结构压缩成固定预算的离散组合 token，并让这些 token 在不依赖任何连续旁路的条件下，真正提高下游生成式推荐效果？**

这就是 LARS-CPQ 的主要研究假设。
