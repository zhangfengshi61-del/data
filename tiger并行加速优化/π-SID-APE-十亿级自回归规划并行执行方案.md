# π-SID-APE：面向十亿级物品的自回归规划—并行执行可扩展生成式推荐

> **π-SID-APE = Autoregressive Planning + Parallel Execution for Scalable Semantic-ID Recommendation**
>
> 更新日期：2026-09-27  
> 状态：下一阶段核心研究方案 / 待实现验证  
> 参考现有实现：/data/fszhang/RecBoard-master/HiFlow-SID  
> 参考现有方案：/data/fszhang/tiger并行加速优化/PATE-MI-SID-增强专家一阶段全目录方案.md  
> 主要外部启发：/data/fszhang/tiger并行加速优化/2504.16054v1.pdf（π0.5）

---

## 1. 研究背景与当前问题

当前已经实现了一条很有价值的路线：使用轻量历史编码器、多兴趣 slot、并行 SID 位置头、prefix-conditioned 路径专家、短期兴趣与训练集转移统计，一次性对完整目录进行打分。

这条路线的优点很明确：

1. 不需要 TIGER 的完整自回归 beam search；
2. 不依赖推理阶段 teacher；
3. 大量计算可以变成 GPU 上的并行矩阵运算；
4. 在目前 Beauty / Sports / Toys 规模上，速度明显快于 SID-MLP，并且效果已经接近 TIGER；
5. 当前联合评分函数本身已经具备较好的推荐质量，因此不应该轻易推倒重来。

但是当前结构存在一个根本规模瓶颈：

\[
S(u,i), \qquad \forall i\in\mathcal I
\]

仍然需要显式或分块地遍历整个 item catalog。

当目录只有几万或几十万物品时，全目录并行打分非常快；但是当目录扩展到：

\[
10^7,\ 10^8,\ 10^9
\]

个物品时，问题不再是“矩阵乘法是不是够快”，而是：

> **是否仍然必须对所有物品做一次计算。**

因此当前方法实际上是：

\[
\boxed{\text{用并行计算解决了串行生成，但没有解决全目录线性扫描}}
\]

这就是下一阶段需要解决的核心问题。

---

## 2. 十亿级物品下，全目录并行为什么会失效

设：

- batch size 为 \(B\)；
- 物品数为 \(N\)；
- 兴趣 slot 数为 \(K\)；
- 每个 float32 占 4 bytes。

仅一个 \(B\times N\) score matrix 的存储量就是：

\[
4BN\ \text{bytes}.
\]

若：

\[
B=96,\qquad N=10^9,
\]

则仅最终分数矩阵就需要：

\[
96\times10^9\times4
=384\text{ GB}.
\]

如果保留 4 个兴趣槽的中间分数：

\[
B\times K\times N,
\]

则：

\[
96\times4\times10^9\times4
=1.536\text{ TB}.
\]

即使使用 chunk：

~~~text
10 亿物品
→ 每次只处理 4096 / 8192 个
→ 最终循环处理所有 chunk
~~~

只能解决显存峰值，不能改变总计算量：

\[
O(N).
\]

因此下一阶段必须从：

\[
\boxed{\text{对所有物品并行算}}
\]

升级为：

\[
\boxed{\text{只对值得计算的语义区域进行精确联合评分}}
\]

同时尽可能保留当前 full-catalog scorer 的排序能力。

---

# 3. 核心思想

本方案暂命名：

# **π-SID-APE**

即：

> **Autoregressive Planning and Parallel Execution for Semantic-ID Recommendation**

一句话概括：

> **让自回归模型负责决定“去哪里搜索”，让并行专家负责决定“这个区域里的哪些完整 SID / item 最好”。**

也可以理解成：

\[
\boxed{
\text{自回归负责搜索空间压缩}
+
\text{并行专家负责局部高质量排序}
}
\]

而不是二选一。

---

## 4. 为什么参考 π0.5，而不是直接照搬 π0.5

π0.5 的关键思想不是 Flow Matching 本身，而是：

> **高层语义决策与低层高维输出执行解耦。**

在 π0.5 中，可以抽象为：

~~~text
高层模型
    ↓
预测语义 subtask
    ↓
低层 action expert
    ↓
生成连续动作块
~~~

迁移到推荐系统后：

~~~text
用户历史
    ↓
高层自回归语义规划器
    ↓
选择值得进入的 SID prefix / semantic region
    ↓
低层并行 SID 专家
    ↓
对该局部区域内合法完整 item 并行联合打分
~~~

因此，本方案不直接照搬 flow matching。

原因是：

- 如果 flow 最后仍然需要与十亿个 item 做 nearest-neighbor / full ranking，则目录瓶颈依然存在；
- 我们真正需要的是让大部分 item 根本不进入昂贵的 item-level score；
- π0.5 最有价值的启发是“高层决定低层专家在哪里工作”。

---

# 5. 总体架构

完整在线流程：

~~~text
用户历史 H
   │
   ▼
共享轻量 History Encoder
   │
   ├──────────────► multi-interest slots z1...zK
   │
   ├──────────────► parallel SID heads
   │
   └──────────────► semantic / recent / transition user state
   │
   ▼
Autoregressive Planner
   │
   ├── 展开高价值 SID prefix
   │
   ├── 判断节点容量
   │
   ├── 判断节点上界
   │
   └── 动态决定继续展开 or 交给并行专家
   │
   ▼
若节点仍过大：继续自回归细分
若节点足够小：进入 Parallel Executor
   │
   ▼
对节点中的合法完整 item / SID 一次并行评分
   │
   ▼
更新全局 Top-K
   │
   ▼
检查未访问节点的 score upper bound
   │
   ├── 仍可能超过当前 Top-K → 回补搜索
   │
   └── 不可能超过 → 安全停止
   ▼
最终推荐
~~~

这里最重要的一点是：

> **自回归部分不直接生成所有最终 item；它只负责规划搜索路径。**

真正的完整 item ranking 仍然交给当前已经有效的联合专家。

---

# 6. 第一原则：保留现有评分器，不首先改变质量来源

第一版 π-SID-APE 的目标不应该是重新训练一个完全不同的推荐器。

应该先把当前 scorer 看作 oracle：

\[
S_{\text{old}}(u,i).
\]

其结构可以抽象为：

\[
S_{\text{old}}(u,i)
=
S_{\text{path}}(u,i)
+
\alpha_s S_{\text{semantic}}(u,i)
+
\alpha_r S_{\text{recent}}(u,i)
+
\alpha_t S_{\text{transition}}(u,i)
+
\alpha_p S_{\text{pair-transition}}(u,i).
\]

第一阶段的目标：

\[
\boxed{
\text{尽量少访问 item}
\quad
\text{但得到与 }
\operatorname{TopK}_{i\in\mathcal I}S_{\text{old}}(u,i)
\text{ 相同的结果}
}
\]

即：

> 先只改“执行方式”，不改“模型本身”。

这样可以把两个问题分开：

1. 原模型质量是否好；
2. 搜索加速是否损失原模型结果。

---

# 7. Capacity-Aware SID Index：容量自适应 SID 索引

## 7.1 为什么不能简单 top-prefix 后扫描整个桶

假设三层 SID，每层 codebook size 为 256。

如果有 10 亿物品，并假设分布完全均匀：

### 第一层

\[
10^9 / 256
\approx3.91\times10^6.
\]

每个 c0 桶平均约 391 万物品。

### 前两层

\[
10^9 / 256^2
\approx15,259.
\]

每个 \((c_0,c_1)\) 桶平均仍然约 1.5 万物品。

### 前三层

\[
10^9 / 256^3
\approx60.
\]

这说明：

> 只选几个 prefix 并不自动意味着计算量很小。

而且真实 SID 分布通常不均匀，热门前缀可能远大于平均值。

---

## 7.2 关键改动：停止深度由桶容量决定

为每个索引节点 \(v\) 定义：

\[
\mathcal D(v)
=
\{i:\text{item }i\text{ 属于该 SID prefix}\}.
\]

设置最大执行块：

\[
S_{\max}.
\]

例如初始实验：

\[
S_{\max}\in\{32,64,128,256\}.
\]

规则：

\[
|\mathcal D(v)|\le S_{\max}
\Rightarrow
\text{交给 Parallel Executor}
\]

否则：

\[
|\mathcal D(v)|>S_{\max}
\Rightarrow
\text{继续展开}
\]

因此不同节点可以具有不同深度：

~~~text
prefix A
 └── 只有 31 item
     → 直接并行评分

prefix B
 └── 20 万 item
     → 必须继续细分

prefix C
 └── 412 item
     → 再展开一层
~~~

这比固定所有请求都生成 2 位或 3 位 SID 更适合十亿规模。

---

# 8. 索引不应该改变原始 SID

一个重要设计原则：

> **搜索索引可以比模型 SID 更细，但不要求改变已经训练好的原始 SID。**

例如某个完整 semantic SID：

\[
(c_0,c_1,c_2)
\]

仍然对应大量 collision items。

可以继续引入：

\[
(c_0,c_1,c_2,d_1,d_2,\ldots)
\]

其中 \(d\) 可以是：

- collision code；
- bucket-local code；
- learned disambiguation code；
- compact hash；
- item identity routing code。

这些 code 的目的主要是：

\[
\boxed{\text{缩小执行块}}
\]

而不是替代原始 semantic SID 的语义表示。

因此：

- 前几层负责语义；
- 深层负责检索粒度 / collision resolution。

这能避免为了扩展到十亿物品，必须重新训练整个 RQ-VAE / SDQ SID。

---

# 9. Autoregressive Planner：轻量自回归规划器

## 9.1 自回归不再负责完整推荐生成

定义：

\[
h=f_\theta(H)
\]

为用户历史表示。

对节点 \(v_t=(c_0,\ldots,c_t)\)，规划器预测：

\[
p_\theta(c_{t+1}\mid h,c_{\le t}).
\]

但这里的作用不是最终生成 SID，而是：

\[
\boxed{\text{给搜索节点排序}}
\]

因此 Planner 可以做得非常小，例如：

~~~python
state = user_state

for selected prefix depth:
    prefix_emb = prefix_encoder(prefix)
    state = small_gru_or_mlp(state, prefix_emb)
    route_logits = route_head(state)
~~~

用户历史 encoder 只运行一次。

不能为每个节点重复运行完整 T5/TIGER encoder。

---

## 9.2 并行扩展多个 prefix

虽然 Planner 有自回归依赖：

\[
c_{t+1}\mid c_{\le t},
\]

但是同一个深度的多个 active nodes 可以组成 batch：

~~~text
node-1 prefix
node-2 prefix
node-3 prefix
...
node-P prefix
         │
         ▼
一次 batched planner forward
~~~

所以这里是：

\[
\boxed{\text{层间自回归，层内并行}}
\]

而不是完全串行 beam search。

---

# 10. Planner 不应该只学习 prefix probability

普通自回归 SID 模型倾向于优化：

\[
P(v\mid u).
\]

但对于 Top-K retrieval，更关键的问题不是：

> “这个桶的总概率有多大？”

而是：

> “这个桶里面有没有一个非常高分的 item？”

举例：

### Region A

总概率：

\[
0.60
\]

但是包含 10,000 个相似 item：

\[
\max_i P(i)\approx0.00006.
\]

### Region B

总概率：

\[
0.05
\]

但其中一个 item：

\[
P(i^*)=0.05.
\]

如果只按照 prefix probability 搜索：

\[
A>B.
\]

但如果目标是 Top-1：

\[
B>A.
\]

因此我们引入：

# **Top-K Value Head**

定义：

\[
V^*(u,v)
=
\max_{i\in\mathcal D(v)}
S_{\text{old}}(u,i).
\]

Planner 不只预测：

\[
P(v\mid u),
\]

还预测：

\[
\hat V(u,v).
\]

最终节点 priority 可以写为：

\[
Q(u,v)
=
\lambda_p\log P(v\mid u)
+
\lambda_v\hat V(u,v)
-
\lambda_c C(v),
\]

其中 \(C(v)\) 表示访问该节点的预计计算成本。

这样 Planner 学习的不是“最热门的语义区域”，而是：

\[
\boxed{\text{最可能产生高分 Top-K item 的区域}}
\]

---

# 11. Value Head 的训练

第一阶段冻结当前推荐 scorer。

对训练样本或离线采样用户：

1. 使用当前 full-catalog scorer / 大规模候选 oracle；
2. 记录 item score \(S_{\text{old}}(u,i)\)；
3. 对每个 prefix 节点构造：
   \[
   y(u,v)
   =
   \max_{i\in\mathcal D(v)}S_{\text{old}}(u,i)
   \]
4. 训练：
   \[
   \mathcal L_{\text{value}}
   =
   \operatorname{Huber}(
   \hat V(u,v),
   y(u,v)
   ).
   \]

也可以学习 Top-R：

\[
y_R(u,v)
=
\operatorname{TopRMean}_{i\in\mathcal D(v)}
S_{\text{old}}(u,i),
\]

避免单点最大值过于尖锐。

重要：

\[
\hat V(u,v)
\]

只是 learned predictor。

它可以用来：

- 节点排序；
- 搜索优先级；
- budget allocation。

**不能直接当成严格剪枝证书。**

---

# 12. Parallel Executor：局部完整路径联合专家

当：

\[
|\mathcal D(v)|\le S_{\max},
\]

就不再继续逐 token 生成。

直接获得该节点下所有合法 item：

\[
\mathcal C_v=\mathcal D(v).
\]

然后计算：

\[
S_{\text{old}}(u,i),
\qquad i\in\mathcal C_v.
\]

由于：

\[
|\mathcal C_v|\le S_{\max},
\]

因此可以一次批量执行当前 PATE / joint expert。

例如：

~~~text
selected prefix = [12, 87]

合法 item：
[12,87,9,0]
[12,87,9,1]
[12,87,31,0]
[12,87,44,0]
...

       ↓

一次 parallel executor

       ↓

score(item1)
score(item2)
score(item3)
...
~~~

不是独立预测每一个 suffix 位置后再做笛卡尔积。

这样可以继续保留完整 SID 路径之间的联合结构。

---

# 13. 自回归与并行各自发挥什么优势

## 自回归 Planner 的优势

自回归：

\[
P(c_t\mid c_{<t},u)
\]

非常适合表达：

- 已经进入哪个粗语义；
- 后续应该如何继续细分；
- 不同 prefix 下不同 suffix 的条件关系。

它最适合解决：

\[
\boxed{\text{结构化搜索}}
\]

## Parallel Executor 的优势

并行专家适合：

- 同时处理数十/数百合法完整 item；
- 计算用户 × prefix × suffix 联合交互；
- 充分利用 GPU；
- 避免每一个完整 SID 都逐 token decode。

它最适合解决：

\[
\boxed{\text{局部高精度排序}}
\]

## π-SID-APE 的关键分工

\[
\boxed{\text{AR Planner：减少搜索空间}}
\]

\[
\boxed{\text{Parallel Executor：保住 ranking quality}}
\]

这就是本方案最核心的架构逻辑。

---

# 14. 最大风险：候选漏掉之后无法恢复

如果：

~~~text
Planner 只选择 10 个 prefix
→ Executor 只处理这些 prefix
→ 真实 Top item 在第 11 个 prefix
~~~

那么无论 Executor 多强，正确 item 都无法回来。

因此不能简单采用：

\[
\text{Top-P prefix}
\rightarrow
\text{局部 rerank}
\]

作为最终方法。

必须增加：

# **Adaptive Recovery / Safe Search**

---

# 15. Score Upper Bound：未展开节点的分数上界

我们的目标是为每个节点 \(v\) 构造：

\[
U(u,v)
\]

满足：

\[
U(u,v)
\ge
\max_{i\in\mathcal D(v)}
S_{\text{old}}(u,i).
\]

这样才能知道：

> 某个未访问节点是否仍然有机会产生新的 Top-K item。

---

## 15.1 对点积项的上界

如果某一项：

\[
s(u,i)
=
q(u)^T\phi(i),
\]

则为每个节点预存：

\[
m_j^+(v)
=
\max_{i\in\mathcal D(v)}\phi_j(i)
\]

以及：

\[
m_j^-(v)
=
\min_{i\in\mathcal D(v)}\phi_j(i).
\]

定义：

\[
q_j^+=\max(q_j,0),
\qquad
q_j^-=\min(q_j,0).
\]

则：

\[
q^T\phi(i)
\le
\sum_j
\left(
q_j^+m_j^+(v)
+
q_j^-m_j^-(v)
\right).
\]

因此：

\[
U_{\text{dot}}(u,v)
=
\sum_j
q_j^+m_j^+(v)
+
q_j^-m_j^-(v).
\]

这个 upper bound 不需要遍历节点内所有 item。

---

# 16. 如何处理当前多个 score 分支

当前 scorer 不是一个简单 dot product。

它包含：

1. SID position score；
2. multi-interest route aggregation；
3. tensor/path expert；
4. optional residual expert；
5. semantic score；
6. recent score；
7. transition support；
8. prefix-pair transition。

因此 upper bound 应该分项构造：

\[
U_{\text{total}}
=
U_{\text{path}}
+
U_{\text{semantic}}
+
U_{\text{recent}}
+
U_{\text{transition}}
+
U_{\text{pair}}.
\]

---

## 16.1 Multi-interest logsumexp

如果对每个 slot：

\[
s_k(u,i)\le U_k(u,v),
\]

且最终：

\[
S_{\text{path}}
=
w_p
\log\sum_k
\pi_k(u)e^{s_k(u,i)},
\]

那么因为 logsumexp 单调：

\[
S_{\text{path}}
\le
w_p
\log\sum_k
\pi_k(u)e^{U_k(u,v)}.
\]

因此：

\[
U_{\text{path}}
=
w_p
\log\sum_k
\pi_k(u)e^{U_k(u,v)}.
\]

---

## 16.2 Semantic / Recent

semantic 和 recent 本质上接近：

\[
q_u^T e_i.
\]

因此可以直接使用节点 embedding min/max 构造 bound。

---

## 16.3 Transition

transition score 来自离散 code table：

\[
T(c_{\text{recent}},c_i).
\]

节点内 code 范围是已知的，因此可以在离线阶段存：

\[
U_{\text{transition}}(u,v)
=
\max_{c\in\mathcal C(v)}
T(c_{\text{recent}},c).
\]

因为 codebook 比 item 数小很多，这部分非常便宜。

---

# 17. Exact / Certified Stop Condition

维护当前已经访问到的 Top-K：

\[
\mathcal T_K.
\]

令当前第 K 名分数为：

\[
\tau_K.
\]

对于所有还没有展开的 frontier nodes：

\[
\mathcal F.
\]

如果：

\[
\boxed{
\max_{v\in\mathcal F}U(u,v)
<
\tau_K
}
\]

则可以证明：

\[
\forall i\in
\bigcup_{v\in\mathcal F}\mathcal D(v),
\qquad
S_{\text{old}}(u,i)<\tau_K.
\]

因此剩余 item 不可能进入 Top-K。

此时：

\[
\boxed{
\operatorname{TopK}_{\text{APE}}
=
\operatorname{TopK}_{\text{full-scan}}
}
\]

前提：

1. 索引覆盖完整目录；
2. bound 是严格安全的；
3. 所有 score 项都包含在 bound 中；
4. transition / normalization 与 full catalog 定义一致；
5. 浮点误差使用保守 margin；
6. tie-breaking 与 full scan 一致。

这就是：

# **Certified Mode**

---

# 18. 两种推理模式

## 18.1 Certified Mode

目标：

\[
\text{尽可能复现 full-catalog Top-K}.
\]

停止条件：

\[
U_{\max}<\tau_K.
\]

优点：

- 可以证明没有搜索损失；
- 非常适合论文分析；
- 可以直接计算“多少请求在访问多少 item 后已经获得证书”。

缺点：

- 如果 bound 很松，某些请求可能需要继续展开大量节点；
- 最坏情况仍可能退化。

## 18.2 Budget Mode

设置：

- 最大 node expansion；
- 最大 executor blocks；
- 最大 scored items；
- 最大 latency budget。

例如：

~~~text
max_nodes = 128
max_leaf_blocks = 32
S_max = 64
max_scored_items = 2048
~~~

超过 budget 就直接输出当前 Top-K。

优点：

- latency 可控；
- 更适合线上系统。

缺点：

- 不能保证完全等价 full scan。

因此论文中必须分开报告：

~~~text
Certified Search
Budgeted Search
~~~

不能把两者混为一谈。

---

# 19. 一个非常重要的实现漏洞：当前 transition normalization

当前模型中 transition support 使用类似：

~~~python
support = support - support.mean(dim=1, keepdim=True)
support = support / support.std(dim=1, keepdim=True)
~~~

如果 mean/std 是在当前传入的 candidate set 上计算，则：

\[
S(u,i\mid\text{full catalog})
\neq
S(u,i\mid\text{local bucket}).
\]

这意味着：

> 直接把 full-catalog scorer 改成 local candidate scorer，会悄悄改变原来的排序函数。

这是 π-SID-APE 实现前必须修复的问题。

---

## 19.1 解决方案 A：使用全局固定统计

对每个 source SID / transition condition 预计算 full-catalog：

\[
\mu(u)
\]

与：

\[
\sigma(u).
\]

由于 transition 只依赖离散 source code，不真正依赖连续 user vector，所以可以预计算：

\[
\mu(c_{\text{recent}})
\]

\[
\sigma(c_{\text{recent}}).
\]

局部执行时：

\[
Z(T)
=
\frac{T-\mu_{\text{global}}}
{\sigma_{\text{global}}}.
\]

从而保证：

\[
\text{同一个 item 的 transition 分数与 candidate set 无关}.
\]

---

## 19.2 必做单元测试

随机选择一个用户和 item：

~~~text
score_full(item_x)
score_bucket(item_x)
score_singleton(item_x)
~~~

必须满足：

\[
|s_{\text{full}}
-
s_{\text{bucket}}|
<\epsilon.
\]

建议：

\[
\epsilon=10^{-5}\sim10^{-4}.
\]

只有这个测试通过，才能开始比较 indexed search 与 full scan。

---

# 20. 索引结构：不能继续使用 padded bucket

当前小规模实现中类似：

~~~text
prefix_bucket_ids:
[num_prefix, max_bucket_size]
~~~

十亿级会产生巨大的 padding 浪费。

应该改成：

# **CSR-like Compact Prefix Index**

存储：

~~~text
sorted_item_ids

node_offset
node_length

child_offset
child_count

node_feature_min
node_feature_max

node_transition_bound
~~~

节点 \(v\)：

~~~python
start = node_offset[v]
length = node_length[v]

items = sorted_item_ids[start:start+length]
~~~

空间近似：

\[
O(N)
\]

而不是：

\[
O(\#node\times\max bucket).
\]

---

# 21. 上层索引与下层索引分层存储

十亿物品不可能把所有 dense item embedding 都放在单张 GPU。

因此建议：

## GPU Resident

常驻：

- 第一层 / 第二层索引；
- 高频节点摘要；
- codebook；
- planner；
- user encoder；
- top-level bound features。

## CPU / Host Memory

存：

- 深层节点；
- compact item ids；
- collision mapping；
- node metadata。

## NVMe / Distributed KV（未来）

冷门叶节点可以存：

- item ids；
- sparse metadata；
- bucket-local codes。

请求只加载实际访问的叶块。

---

# 22. 不建议存十亿个完整 item embedding

如果每个 item embedding：

\[
128\times FP16,
\]

十亿 item 需要：

\[
10^9\times128\times2
=
256\text{ GB}.
\]

而当前方法的一个优势是：

> item 表示可以由 SID codebook / low-rank factor 组合得到。

因此建议叶块主要存：

~~~text
item_id
SID
collision code
compact metadata
~~~

需要评分时：

\[
\text{SID}
\rightarrow
\text{codebook lookup}
\rightarrow
\text{item feature}.
\]

这样可以把大部分存储从：

\[
O(Nd)
\]

压到：

\[
O(NL_{\text{code}}).
\]

---

# 23. 搜索算法伪代码

~~~python
def recommend_ape(user_history, K):
    # 1. 用户侧只编码一次
    hidden = history_encoder(user_history)
    slots, route = build_interest_slots(hidden)

    # 2. 当前 top-k
    topk = TopKHeap(K)

    # 3. frontier
    root = index.root
    root_priority = planner_priority(slots, root)
    frontier = PriorityQueue()
    frontier.push(root, root_priority)

    while frontier:

        node = frontier.pop()

        # 4. certified pruning
        ub = score_upper_bound(slots, node)

        if topk.full() and ub < topk.kth_score():
            continue

        # 5. 小块交给并行 expert
        if node.item_count <= S_max:

            items = index.items(node)

            scores = score_local_exact(
                slots,
                items,
                global_transition_stats=True
            )

            topk.update(items, scores)
            continue

        # 6. 大块继续自回归展开
        children = index.children(node)

        priorities = planner_batch(
            slots,
            prefix=node.prefix,
            children=children
        )

        child_bounds = score_bound_batch(
            slots,
            children
        )

        for child in children:
            if (
                not topk.full()
                or child_bounds[child] >= topk.kth_score()
            ):
                frontier.push(
                    child,
                    priorities[child]
                )

        # 7. Budget Mode
        if budget_exceeded():
            break

    return topk
~~~

---

# 24. 为什么它不是简单的传统“两阶段召回 + 重排”

需要谨慎表述。

从执行形式上，它确实存在：

~~~text
搜索空间缩小
→ 局部精确打分
~~~

因此不能声称：

> “完全没有候选选择”。

更准确的论文表述应该是：

> π-SID-APE performs **score-consistent indexed search** under the same recommendation scorer, rather than using an independent retrieval model followed by a separate reranker.

核心区别希望体现为：

### 传统两阶段

\[
S_{\text{retriever}}
\neq
S_{\text{reranker}}.
\]

召回器漏掉 item 后无法知道。

### π-SID-APE

搜索优先级可以由 planner 提供，但最终安全剪枝由：

\[
U_{\text{same scorer}}
\]

决定。

Certified Mode 下：

\[
\operatorname{TopK}_{\text{APE}}
=
\operatorname{TopK}_{S_{\text{old}}}.
\]

因此贡献应该落在：

\[
\boxed{\text{score-consistent search + parallel semantic execution}}
\]

而不是“我们没有候选”。

---

# 25. 与 TIGER 的关系

TIGER：

~~~text
history
  ↓
autoregressive decoder
  ↓
c0
  ↓
c1 | c0
  ↓
c2 | c0,c1
  ↓
beam search
  ↓
item
~~~

优点：

- 条件依赖强；
- 不需要全目录 scan。

缺点：

- 每一步需要模型 forward；
- beam 带来额外串行与分支成本；
- 完整 item SID 越长，decode 成本越高。

π-SID-APE：

~~~text
history
  ↓
encoder once
  ↓
small autoregressive planner
  ↓
adaptive semantic regions
  ↓
parallel full-path expert
  ↓
top-k
~~~

核心目标：

\[
\boxed{
\text{保留 TIGER 的结构化自回归搜索优势}
+
\text{保留当前并行专家的低延迟优势}
}
\]

---

# 26. 与 SID-MLP 的关系

SID-MLP 的核心方向是：

- 用轻量 MLP 减少生成模型计算；
- 保留 prefix conditioning；
- 约束在合法 SID 空间中解码。

π-SID-APE 不应该只 claim：

> “MLP 比 Transformer 快”。

真正差异应该集中在：

1. 多兴趣用户表示；
2. 高层 AR Planner 与低层 parallel executor 的职责解耦；
3. capacity-aware variable-depth expansion；
4. Top-K value routing；
5. scorer-consistent bound；
6. Certified / Budget dual-mode search。

---

# 27. 与普通 tree retrieval 的关系

树检索本身不是新东西。

因此论文不能把创新点写成：

> “我们提出了一棵 SID tree”。

真正需要证明的是：

### 1. 树节点来自已有 semantic-ID 结构

而不是额外训练完全独立的 retrieval tree。

### 2. Planner 预测的是 Top-K value / search utility

而不仅仅是 node classification。

### 3. 叶节点交给已有联合 SID expert

不是简单 dot product。

### 4. 剪枝依据来自最终 scorer 的 upper bound

而不是仅靠 learned probability。

### 5. 支持 certified full-score equivalence

这是最有研究价值的方向之一。

---

# 28. 建议的训练阶段

## Stage 0：保留 current scorer

完全加载现有最好 checkpoint。

冻结：

- history encoder；
- slot module；
- path expert；
- semantic scorer；
- transition scorer。

先不改变推荐质量。

## Stage 1：建立离线 SID index

输入：

~~~text
item_id
SID codes
collision information
expert static factors
~~~

构造：

~~~text
node children
node item range
node capacity
node factor min/max
node transition bound
~~~

并保存：

~~~text
index_version
checkpoint_hash
sid_vocab_hash
~~~

保证 index 与 checkpoint 对齐。

## Stage 2：实现 exact local scorer

要求：

\[
S_{\text{local}}(u,i)
\approx
S_{\text{full}}(u,i).
\]

优先修正：

- transition normalization；
- semantic normalization；
- recent normalization；
- candidate-set-dependent operation。

## Stage 3：先不用 learned planner

先使用：

~~~text
exact bound best-first search
~~~

判断：

> 当前 scorer 本身是否具有“可索引性”。

如果 exact bound 必须访问 80% item 才能确定 Top-K：

> 问题不是 planner，而是 bound / item representation 不够紧。

这是必须首先知道的事实。

## Stage 4：训练 Planner

Planner 目标：

\[
\mathcal L_{\text{planner}}
=
\lambda_{\text{route}}\mathcal L_{\text{route}}
+
\lambda_{\text{value}}\mathcal L_{\text{value}}
+
\lambda_{\text{cost}}\mathcal L_{\text{cost-aware}}.
\]

其中：

### route loss

真实 target item 所在路径：

\[
\mathcal L_{\text{route}}
=
-\sum_t
\log
P(c_t^+\mid u,c_{<t}^+).
\]

### value loss

\[
\mathcal L_{\text{value}}
=
\operatorname{Huber}(
\hat V(u,v),
V^*(u,v)
).
\]

### cost-aware ranking

希望高价值低成本节点优先：

\[
\mathcal L_{\text{cost-aware}}
=
\max(
0,
m-Q(u,v^+)+Q(u,v^-)
).
\]

---

# 29. 第二阶段模型增强：Search-Aware Expert Training

只有当第一阶段 indexed execution 已经跑通后，再进一步训练推荐模型。

可以引入：

# **Search-aware hard negatives**

Planner 经常混淆的邻近节点：

~~~text
正确 prefix
vs
高 priority 错误 prefix
~~~

里面的 item 是非常强的 hard negatives。

训练：

\[
\mathcal L_{\text{rank}}
=
-\log
\frac{
e^{S(u,i^+)}
}{
e^{S(u,i^+)}
+
\sum_{j\in\mathcal N_{\text{search}}}
e^{S(u,j)}
}.
\]

这比 uniform negatives 更接近真实推理错误。

---

# 30. Boundary Tightness Regularization：让模型更容易被检索

一个进一步值得研究的新方向：

> 不只训练 item score，还训练同一 semantic region 内的 item feature 更紧凑。

因为 upper bound tightness 决定搜索效率。

若节点内：

\[
m_j^+(v)-m_j^-(v)
\]

很大，则 upper bound 很松。

因此可以加入：

\[
\mathcal L_{\text{bound}}
=
\sum_v
w_v
\sum_j
(
m_j^+(v)-m_j^-(v)
).
\]

实际训练不能直接对 max/min 做大规模计算，可以用：

- sampled variance；
- quantile range；
- feature radius；
- center-radius surrogate。

例如：

\[
\mathcal L_{\text{compact}}
=
\sum_{i\in v}
\|
\phi(i)-\mu_v
\|^2.
\]

但要同时防止语义表示塌缩。

这可以形成一个非常有价值的研究问题：

\[
\boxed{
\text{不仅学习高质量推荐表示，还学习“可高效搜索”的推荐表示}
}
\]

即：

# **Retrieval-friendly Scorer**

---

# 31. 复杂度分析

当前 full scan：

\[
T_{\text{full}}
=
T_{\text{encoder}}
+
O(NC_{\text{score}}).
\]

π-SID-APE：

设：

- \(P\)：访问内部节点数；
- \(J\)：执行叶块数；
- \(S_{\max}\)：每块最大 item 数；
- \(V\)：平均分支数。

则：

\[
T_{\text{APE}}
=
T_{\text{encoder}}
+
O(PVC_{\text{planner}})
+
O(P C_{\text{bound}})
+
O(JS_{\max}C_{\text{score}}).
\]

如果：

\[
P\ll N
\]

且：

\[
JS_{\max}\ll N,
\]

则可以把物品级昂贵计算从：

\[
O(N)
\]

降低到：

\[
O(JS_{\max}).
\]

注意：

> 不能声称严格 \(O(1)\)。

因为：

- tree depth 可能随 catalog 增长；
- index size 仍然是 \(O(N)\)；
- certified search 最坏情况仍可能访问很多节点。

论文应该强调：

\[
\boxed{\text{sublinear practical item scoring}}
\]

而不是不严谨地说 \(O(1)\)。

---

# 32. 一个十亿物品的示例

假设：

\[
N=10^9.
\]

设置：

\[
S_{\max}=64.
\]

某用户有 4 个兴趣 slot。

Planner 最终探索 32 个叶块。

则执行专家实际评分 item：

\[
32\times64=2048.
\]

相比 full scan：

\[
10^9
\]

理论 item-score reduction：

\[
\frac{10^9}{2048}
\approx488,281\times.
\]

但这不是最终 latency speedup。

实际延迟还包括：

- user encoder；
- planner；
- upper-bound；
- index lookup；
- host-to-device transfer；
- top-k heap；
- cache miss。

所以正式论文必须报告真实 wall-clock latency，而不是只报告 item-count reduction。

---

# 33. 第一阶段最重要的实验不是 NDCG

当前 scorer 已经有质量。

第一阶段必须先回答：

> **这个 scorer 能否被高效索引？**

建议新增以下指标：

### Search Efficiency

\[
\text{ScoredItemRatio}
=
\frac{
\#\text{exactly scored items}
}{
N
}.
\]

### Node Visit

\[
\#\text{visited nodes}.
\]

### Full-Score Top-K Recall

\[
R_{\text{oracle@K}}
=
\frac{
|
TopK_{\text{APE}}
\cap
TopK_{\text{Full}}
|
}{K}.
\]

### Exact Match Rate

\[
\frac{
\#\{
u:
TopK_{\text{APE}}(u)
=
TopK_{\text{Full}}(u)
\}
}{
\#u
}.
\]

### Certification Rate

在给定：

- 512 item；
- 1024 item；
- 2048 item；
- 4096 item；

预算下，有多少用户已经满足：

\[
U_{\max}<\tau_K.
\]

这将是非常关键的一张图。

---

# 34. 推荐的核心图

## 图 1：Quality–Latency Curve

X：

~~~text
P50 latency
~~~

Y：

~~~text
NDCG@10
~~~

方法：

- TIGER；
- SID-MLP；
- current PATE / current scorer；
- π-SID-APE Budget；
- π-SID-APE Certified。

## 图 2：Catalog Scaling

X：

\[
10^4
\rightarrow
10^5
\rightarrow
10^6
\rightarrow
10^7
\rightarrow
10^8
\rightarrow
10^9
\]

Y：

~~~text
latency
~~~

比较：

- full scan；
- TIGER；
- SID-MLP；
- APE。

这是证明本工作的关键图。

## 图 3：Scored Items vs Full-score Recall

X：

~~~text
exactly scored items
~~~

Y：

~~~text
Top-K recall against full scorer
~~~

用于证明 Planner 的搜索效率。

## 图 4：Certification Curve

X：

~~~text
search budget
~~~

Y：

~~~text
fraction of users certified exact
~~~

这个图非常有辨识度。

---

# 35. Ablation

至少包含：

| 版本 | AR Planner | Value Head | Capacity-Adaptive | Upper Bound | Parallel Expert |
|---|---|---|---|---|---|
| A | × | × | × | × | ✓ |
| B | ✓ | × | fixed depth | × | ✓ |
| C | ✓ | ✓ | fixed depth | × | ✓ |
| D | ✓ | ✓ | ✓ | × | ✓ |
| E | ✓ | ✓ | ✓ | ✓ | ✓ |

重点回答：

### Q1

AR Planner 比 independent prefix prediction 是否更好？

### Q2

Value Head 是否比 prefix probability 更容易找到 full scorer 的 Top-K？

### Q3

Capacity-adaptive depth 是否比固定 Prefix-2 / Prefix-3 更适合大目录？

### Q4

Upper Bound recovery 是否显著提高 Full-score Recall？

### Q5

Parallel Executor 是否比完全 AR 生成更快？

---

# 36. 规模实验不能只复制 item

要区分：

## 1. Engineering Stress Test

可以复制 / 合成大量 item 测：

- latency；
- memory；
- index；
- I/O；
- throughput。

这种实验可以做到 \(10^9\)。

但只能证明工程扩展性。

## 2. Recommendation Quality Test

必须使用真实交互 / 合理负样本 / 有意义的物品分布。

不能因为复制 10 亿个随机容易负样本仍然保持 NDCG，就声称：

> 十亿物品推荐 SOTA。

论文中应明确区分：

~~~text
quality benchmark
scaling benchmark
~~~

---

# 37. 公平速度协议

未来比较：

- TIGER；
- SID-MLP；
- current PATE；
- APE。

必须统一：

- 同一 GPU；
- 同一 dtype；
- 同一 batch；
- 同一用户历史；
- 同一 item catalog；
- 同一 Top-K；
- 同一 seen mask；
- 同一 tokenizer 范围；
- warmup；
- P50 / P95 / P99；
- end-to-end wall clock。

尤其不能继续根据用户数线性缩放 TIGER 时间作为最终大规模速度结论。

必须真正运行对应规模下的协议，或者明确说明哪些数字只是 extrapolation。

---

# 38. 需要新增的代码模块

建议新增：

~~~text
/data/fszhang/RecBoard-master/HiFlow-SID/
├── hiflow_lib/
│   ├── model_pate_misid.py
│   ├── ape_index.py
│   ├── ape_planner.py
│   ├── ape_bounds.py
│   └── ape_search.py
│
├── scripts/
│   ├── build_ape_index.py
│   ├── verify_local_score_equivalence.py
│   ├── eval_ape_oracle_recall.py
│   ├── eval_ape_protocol.py
│   ├── bench_ape_scaling.py
│   └── profile_ape_search.py
│
└── train_ape_planner.py
~~~

---

# 39. 推荐实现顺序

## Step 1

把当前 full scorer 拆成：

~~~python
encode_user(...)
score_items(user_state, item_ids)
~~~

必须允许：

~~~text
同一个 user_state
+ 任意局部 item_ids
→ 得到和 full catalog 一致的 item score
~~~

## Step 2

实现 verify_local_score_equivalence.py。

在所有数据集随机抽：

~~~text
100 users
1000 items/user
~~~

检查：

\[
S_{\text{local}}
\approx
S_{\text{full}}.
\]

## Step 3

实现 CSR SID index。

第一版只做：

~~~text
SID prefix
+ collision leaf
~~~

不训练 Planner。

## Step 4

实现 naive best-first：

~~~text
node priority = exact / simple optimistic bound
~~~

评估 indexed scorer 的可索引性。

## Step 5

实现 upper bound。

必须有单元测试：

对随机节点：

\[
U(u,v)
\ge
\max_{i\in\mathcal D(v)}
S(u,i).
\]

测试至少：

~~~text
1000 users × 1000 nodes
~~~

不能出现一次 violation。

## Step 6

实现 Certified Search。

与 full scan 对比：

~~~text
Top-10 item ids
Top-10 scores
~~~

应该一致。

## Step 7

训练 Planner / Value Head。

目标不是先优化 NDCG，而是：

\[
\text{减少达到同样 Top-K 所需的 node visit}.
\]

## Step 8

再做 Budget Search。

画：

~~~text
budget → quality → latency
~~~

## Step 9

最后才考虑 Search-aware fine-tuning。

避免一开始同时改变 scorer、index、planner、loss 和 search，否则实验不可解释。

---

# 40. 成功条件

第一阶段至少满足：

### Correctness

Certified Mode：

\[
TopK_{\text{APE}}
=
TopK_{\text{full scorer}}
\]

在绝大多数、理想情况下全部 test users 上成立。

### Efficiency

在目录扩大时：

\[
\frac{\#\text{scored items}}{N}
\]

快速下降。

### Quality

Budget Mode 在少量 item score 下，NDCG / Recall 仍然接近 current scorer / TIGER。

### Latency

目录从：

\[
10^4\rightarrow10^8/10^9
\]

时，不出现 full scan 的近线性延迟增长。

---

# 41. 失败条件

以下任意一种情况都说明当前版本需要重新设计。

## Failure 1

Certified Search 需要访问 \(>30\%\sim50\%\) 目录才能停止。

说明：

> bound 太松 / scorer 不适合索引。

## Failure 2

Value Planner 相比 prefix probability 没有减少访问节点。

说明：

> value target / training 设计没有帮助 Top-K search。

## Failure 3

局部 score 与 full score 不一致。

说明：

> candidate-dependent normalization 仍然存在。

这种情况下不能继续做公平对照。

## Failure 4

Budget Search 很快，但 target recall 大幅下降。

说明：

> planner candidate coverage 不够。

应该先解决搜索，而不是继续加深 executor。

## Failure 5

index I/O 比模型计算更慢。

需要：

- GPU cache；
- hot node cache；
- compressed leaf；
- asynchronous prefetch；
- batch coalescing。

---

# 42. 最值得强调的潜在创新点

如果实验成立，建议论文创新点不要写成普通的：

> “结合自回归和并行生成”。

这太泛。

更具体地写成：

### Innovation 1：Hierarchical Role Decomposition

将 semantic-ID recommendation 分解成：

\[
\text{autoregressive semantic planning}
+
\text{parallel full-path execution}.
\]

AR 不再承担完整 item generation，而只承担结构搜索。

### Innovation 2：Capacity-Adaptive Semantic Execution

不同 SID region 根据真实 item capacity 动态选择：

\[
\text{继续 AR}
\quad\text{or}\quad
\text{并行 execute}.
\]

避免固定 Prefix-2 / Prefix-3 在十亿目录下失效。

### Innovation 3：Top-K Value Routing

Planner 不只学习：

\[
P(prefix\mid u),
\]

而学习：

\[
\max_{i\in prefix}S(u,i),
\]

直接针对 Top-K retrieval utility。

### Innovation 4：Score-Consistent Certified Search

利用最终联合 scorer 的安全 upper bound，实现：

\[
TopK_{\text{APE}}
=
TopK_{\text{full scorer}}
\]

的可验证停止。

### Innovation 5：Search-Friendly Generative Recommender

进一步通过 compactness / bound-tightness training，让推荐表示不仅准确，而且可被高效检索。

这是后续最值得深入发展成论文核心理论 / 训练贡献的方向。

---

# 43. 论文 Method 的推荐组织

正式论文可以这样组织：

~~~text
3.1 Problem Formulation

3.2 Base Parallel Semantic-ID Scorer

3.3 Capacity-Aware Semantic Index

3.4 Autoregressive Semantic Planner

3.5 Parallel Full-Path Executor

3.6 Score Upper Bounds

3.7 Certified Adaptive Search

3.8 Planner Training

3.9 Search-aware Fine-tuning

3.10 Complexity Analysis
~~~

---

# 44. 最终方法公式

可以把 π-SID-APE 抽象成：

## 用户编码

\[
Z=f_\theta(H).
\]

## 节点规划

\[
P_\psi(v_{t+1}\mid Z,v_{\le t}).
\]

## 搜索价值

\[
\hat V_\psi(Z,v)
\approx
\max_{i\in\mathcal D(v)}
S_\theta(Z,i).
\]

## 局部精确执行

\[
s_i=S_\theta(Z,i),
\qquad
i\in\mathcal D(v),
\quad
|\mathcal D(v)|\le S_{\max}.
\]

## 安全上界

\[
S_\theta(Z,i)
\le
U_\theta(Z,v),
\qquad
\forall i\in\mathcal D(v).
\]

## 停止条件

\[
\max_{v\in\mathcal F}
U_\theta(Z,v)
<
\tau_K.
\]

于是：

\[
\operatorname{TopK}_{\text{APE}}
=
\operatorname{TopK}_{i\in\mathcal I}
S_\theta(Z,i).
\]

这组公式能够非常清楚地表达本方案：

\[
\boxed{
\text{AR 负责 search}
,\qquad
\text{parallel scorer 负责 ranking}
}
\]

以及：

\[
\boxed{
\text{bound 负责保证两者结合后不会盲目漏掉高分区域}
}
\]

---

# 45. 当前最优先的三个实验

如果接下来马上开始实现，不建议同时铺开所有模块，先做下面三件事。

## Experiment A：Local Score Equivalence

目的：

> 确定现有 scorer 能不能在任意局部 item subset 上保持与 full catalog 完全相同的分数。

成功条件：

\[
|S_{\text{local}}-S_{\text{full}}|<10^{-4}.
\]

如果失败，先修 normalization，不能继续。

## Experiment B：Exact Bound Tightness

不用 learned planner，只使用真实安全 upper bound 做 best-first search。

观察：

- Top-10 exact match；
- 平均访问节点；
- 平均 exact-scored items；
- certification rate。

这一步直接判断这个研究方向是否真正可行。

## Experiment C：Value Planner

在 B 可行以后训练 value head。

比较：

~~~text
prefix probability
vs
value prediction
vs
value + cost
~~~

看达到相同 full-score Top-K recall 需要访问多少节点。

如果 value head 能大幅减少 node visits，就形成一个很强的结果。

---

# 46. 最终判断

当前方法最宝贵的东西不是“全目录一次矩阵乘法”，而是：

> **已经训练出一个速度很快、效果接近 TIGER 的非自回归联合 SID scorer。**

全目录扫描只是它目前的执行方式，不应该成为方法本身不可改变的限制。

下一阶段最有潜力的研究路线就是：

\[
\boxed{
\text{Current PATE / Joint SID Scorer}
+
\text{Autoregressive Planner}
+
\text{Capacity-Adaptive Index}
+
\text{Parallel Executor}
+
\text{Safe Recovery / Certified Bound}
}
\]

目标不是简单做到：

> “10 亿物品也能跑”。

而是形成一个更强的研究命题：

> **Can a high-quality parallel generative recommender retain its ranking quality while avoiding full-catalog scoring through autoregressive semantic planning and score-consistent certified search?**

如果这个问题的答案通过实验被证明为 yes，那么这条路线不仅可以解决当前“小物品集才快”的漏洞，而且有机会把当前工作从“并行解码加速器”推进为：

# **面向超大规模目录的可扩展生成式推荐架构。**
