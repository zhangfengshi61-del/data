# LARS-CPQ 失败分析与下一轮诊断方案

> 状态：CPQ 第一轮正式结果已完成，当前版本判定为失败候选  
> 更新时间：2026-09-28  
> 目标：解释 LARS-CPQ 为什么在 Beauty / Sports / Toys 上均明显退化，并给出完全不依赖 SDQ-VAE 的下一轮诊断与改进路线。

---

# 1. 当前正式结果

本轮 CPQ 运行目录：

~~~text
/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_lars_cpq/
~~~

| 数据集 | 验证最佳 NDCG@10 | 测试 NDCG@10 | 相对 SDQ-VAE |
|---|---:|---:|---:|
| Beauty | 0.04610360 | 0.03190002 | -22.333% |
| Sports | 0.02601080 | 0.01815959 | -31.862% |
| Toys | 0.03937938 | 0.02673292 | -31.266% |

结果文件：

~~~text
/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_lars_cpq/Beauty/result.json
/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_lars_cpq/Sports/result.json
/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_lars_cpq/Toys/result.json
~~~

T5 日志目录：

~~~text
/data/fszhang/RecBoard-master/SDQ-LARS/logs/LARS-20260927_lars_cpq-CPQ-T5/
~~~

Beauty 与 Toys 的 status.json 标记为 failed，Sports 标记为 complete，但三个数据集均已经生成 result.json 和 T5 结果文件。

因此首先可以确认：

\[
\boxed{
\text{CPQ 的明显退化不能仅用 failed 状态解释}
}
\]

因为 Sports 正常完成却同样下降约 32%。

Beauty / Toys 的 failed 状态仍需要单独检查：究竟是“质量验收未通过”，还是 test 完成之后在汇总、画图、复制文件、cleanup 等阶段报错。

---

# 2. 当前版本应如何定性

这一轮不能简单解释为：

> CPQ 思路一定正确，只是参数没有调好。

更准确的结论是：

> **当前 LARS-CPQ 设计在三个数据集上均显著退化，说明“把 LARS 多方向系数组合压缩成 composition token”这一设计至少在当前实现、初始化和损失下没有产生更好的生成式推荐 SID。**

当前可以确认的是：

\[
\boxed{
\text{CPQ 第一版失败}
}
\]

但不能由此推出：

\[
\boxed{
\text{纯 LARS 路线失败}
}
\]

因为 CPQ 相比 LARS-only-HRQ 同时改变了多个关键变量：

1. token 从最大方向改为组合 pattern；
2. 新增 coefficient-space matching；
3. 改变 residual recursion；
4. 引入 shared pattern magnitude；
5. 使用 hard-forward / soft-backward；
6. 新增 soft usage regularization；
7. 新增 SID-only recommendation auxiliary head；
8. 使用新的 pattern initialization；
9. 固定 sparse support。

因此必须通过消融定位是哪一个环节破坏了 SID。

---

# 3. 第一优先级怀疑：Coefficient-space matching 可能是错误的 SID 分组标准

CPQ 的核心分配能量为：

\[
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
\]

其中：

- \(r_{i\ell}\)：当前 residual；
- \(a_{i\ell}\)：LAR 求出的 item-specific sparse coefficient；
- \(\tilde a_{\ell k}\)：第 \(k\) 个 composition pattern 的 coefficient prototype；
- \(c_{\ell k}=\tilde a_{\ell k}D_\ell\)。

第二项实际上假设：

\[
\boxed{
\text{LARS coefficient 接近}
\Rightarrow
\text{语义表示或推荐意义接近}
}
\]

但这个假设并不一般成立。

假设两个 dictionary direction 几乎重合：

\[
d_1=(1,0),
\]

\[
d_2\approx(0.99995,0.01).
\]

两个 coefficient 分别为：

\[
a_A=(1,0),
\]

\[
a_B=(0,1).
\]

在 coefficient space：

\[
\|a_A-a_B\|_2^2=2.
\]

但真实向量空间：

\[
\|a_AD-a_BD\|_2^2
=
\|d_1-d_2\|_2^2
\approx 0.0001.
\]

即：系数看起来差得很大，真实表示却几乎一样。

更一般地：

\[
\|(a-b)D\|_2^2
=
(a-b)DD^\top(a-b)^\top.
\]

真正的 coefficient geometry 应受 dictionary Gram matrix：

\[
DD^\top
\]

影响。

而普通的：

\[
\|a-b\|_2^2
\]

等价于默认 dictionary directions 接近正交。

LARS dictionary 并不保证严格正交，所以：

\[
\boxed{
\text{普通 coefficient L2 很可能不是合适的 SID clustering metric}
}
\]

这是下一轮必须优先做消融的因素之一。

---

# 4. 第二个核心问题：保留更多 LARS 组合信息，不等于保留更多推荐信息

CPQ 原始动机是：

> LARS-only-HRQ 只保留最大方向，因此丢掉 LARS 的符号、组合和相对权重。

这个观察成立。

但从：

\[
\text{组合信息更多}
\]

不能直接推出：

\[
\text{推荐效果更好}.
\]

CPQ 每层仍然只有：

\[
K=256
\]

个 token。

离散接口的状态数并没有增加：

\[
\log_2 256 = 8\text{ bits}.
\]

CPQ 真正改变的是：

> 哪些 item 应该共享同一个 token。

原来的 token 更接近一个基础方向。

CPQ token 对应一个稀疏组合 pattern。

因此 CPQ 不是“让每个 token 自动携带更多比特”，而是：

\[
\boxed{
\text{重新划分了 256 个离散区域}
}
\]

新的划分可能更适合重构 LARS coefficient，却更难被 T5 按 prefix 顺序预测。

所以真正值得研究的问题应从：

> 如何把更多 LARS coefficient 塞进 SID？

改成：

\[
\boxed{
\text{哪些 LARS 信息值得进入 SID，哪些信息应该主动丢掉？}
}
\]

---

# 5. 第三个核心问题：辅助推荐头仍然可能只改善连续码字，而不是离散 SID

CPQ 训练期辅助推荐头读取：

\[
q_i
=
\sum_\ell b_{\ell,s_{i\ell}}D_\ell
\]

进行 next-item prediction。

虽然它没有读取 raw item ID embedding 或 item-specific LARS amplitude，但仍存在一个问题。

假设整个训练过程中，item 的 SID assignment 基本不变：

~~~text
A -> (12, 37, 9)
B -> (12, 41, 8)
C -> (21, 6, 55)
~~~

与此同时：

\[
B_\ell,\ D_\ell
\]

仍然持续更新。

那么：

\[
q_i
\]

的连续向量可以不断改善，辅助 GRU 也可以不断改善。

于是可能出现：

\[
\mathcal L_{\mathrm{aux-rec}}\downarrow
\]

但：

\[
SID_i
\]

几乎没有变化。

最终重新训练 T5 时：

- T5 不继承 GRU；
- T5 不继承 CPQ 连续 code-vector geometry；
- T5 只看到离散 SID token。

因此：

\[
\boxed{
\text{辅助推荐头变好}
\not\Rightarrow
\text{离散 SID 变好}
}
\]

这说明当前 CPQ 仍然存在：

\[
\text{continuous pattern-vector objective}
\rightarrow
\text{discrete SID utility}
\]

之间的目标断层。

---

# 6. 第四个风险：Pattern 初始化 residual 与正式 residual 可能不一致

CPQ 先进行 LARS warm-up，再收集 coefficient 初始化 pattern。

如果 warm-up 阶段 residual recursion 使用：

\[
r_{\ell+1}^{warm}
=
r_\ell^{warm}
-
a_{i\ell}D_\ell
\]

而正式 CPQ 阶段使用：

\[
r_{\ell+1}^{CPQ}
=
r_\ell^{CPQ}
-
b_{\ell,s_{i\ell}}D_\ell,
\]

那么从第 2 层开始，输入 distribution 已经变化。

如果第 2 / 3 层 pattern 是依据旧的 warm-up residual 初始化：

\[
r_2^{warm},
\quad
r_3^{warm},
\]

正式训练却面对：

\[
r_2^{CPQ},
\quad
r_3^{CPQ},
\]

就会产生 distribution mismatch。

如果 sparse support 又在初始化后固定，后层可能尤其难以恢复。

需要检查当前实现：

### 方案 A

三层 pattern 一次性由 warm-up residual 初始化；

### 方案 B

先确定 level-1 hard pattern，再重新计算 residual，之后初始化 level 2，再初始化 level 3。

如果当前实现接近 A，应把它视为高风险点。

---

# 7. 第五个风险：Soft usage 健康，不代表 hard SID 健康

CPQ 使用：

\[
p_{i\ell k}
=
\operatorname{softmax}(-E_{i\ell k}/\tau)
\]

计算 soft marginal：

\[
\bar p_{\ell k}
=
\frac1N\sum_i p_{i\ell k}
\]

并做 usage balance。

但最终导出：

\[
s_{i\ell}
=
\arg\max_k p_{i\ell k}.
\]

即使 soft distribution 看起来均衡，hard argmax 仍可能严重集中。

因此必须同时监控：

- soft entropy；
- hard entropy；
- hard PPL；
- top-1 token share；
- top-10 token share；
- unique pattern；
- prefix@1 / prefix@2 unique；
- full SID collision。

不能只看 soft usage loss。

---

# 8. Beauty / Toys 的 failed 状态应如何检查

由于 Sports 标记 complete 但同样严重下降：

\[
\boxed{
\text{failed status 不是 CPQ 整体退化的主解释}
}
\]

但 Beauty / Toys 仍需要明确发生了什么。

重点读取：

~~~text
runs/20260927_lars_cpq/Beauty/status.json
runs/20260927_lars_cpq/Toys/status.json
launcher.log
t5.log
~~~

需要回答：

## 8.1 test 是否真正执行完成

日志是否明确包含：

- best validation checkpoint；
- load best checkpoint；
- test evaluation；
- final metrics；
- result.json written。

## 8.2 failed 发生在 test 前还是 test 后

如果 test 后才在：

- write summary；
- plot；
- copy result；
- cleanup；
- status update；

阶段报错，则 test 数值可能仍是有效结果。

如果 test 前失败，则必须核查 result.json 是否来自当前运行。

## 8.3 SID 与 T5 是否一一对应

应保存：

- SID vocab hash；
- T5 run name；
- T5 checkpoint path；
- config；
- dataset；
- best validation epoch。

## 8.4 test 是否真的加载 validation best checkpoint

不能仅因为目录里存在 best.pt 就默认 test 使用了它。

必须从日志或结果 metadata 确认。

---

# 9. 下一轮优先做四个诊断实验

下一步不建议立刻增加更多 loss，而应该做减法。

---

## 9.1 Baseline A：当前纯 LARS 最好版本

保持已有 LARS-only-HRQ 不变。

它是所有后续独立纯 LARS 新方法必须首先超过的 baseline。

---

## 9.2 Ablation B：CPQ-NoCoef

去掉分配能量中的：

\[
\lambda_p
\|a_i-\tilde a_k\|^2
\]

同时关闭独立的 coefficient pattern consistency loss。

只保留：

\[
\boxed{
E_{ik}
=
\|r_i-c_k\|^2
}
\]

但仍保留：

\[
c_k=b_kD.
\]

它回答：

> 这次退化是否主要来自“把 LARS coefficient structure 强行当作 SID clustering target”？

如果：

\[
\text{CPQ-NoCoef}
\gg
\text{CPQ},
\]

说明 coefficient matching 是关键问题。

---

## 9.3 Ablation C：CPQ-NoRec

完全关闭：

\[
\mathcal L_{\mathrm{rec}}.
\]

其余 CPQ 保持不变。

它回答：

> recommendation auxiliary 是否主要优化了连续 pattern vector，却破坏或没有改善最终 SID？

如果：

\[
\text{CPQ-NoRec}
\gg
\text{CPQ},
\]

说明当前辅助推荐训练与最终 T5 interface 存在明显不一致。

---

## 9.4 Ablation D：CPQ-Minimal

同时关闭：

- coefficient matching；
- recommendation auxiliary；
- usage balance；
- direction diversity。

仅保留：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{hard-recon}}
+
\lambda_{\mathrm{lar}}
\mathcal L_{\mathrm{LAR}}.
}
\]

token 仍然是 composition pattern。

这个实验直接回答：

> “composition pattern token”本身是否有价值？

如果：

\[
\text{CPQ-Minimal}
<
\text{LARS-only-HRQ},
\]

则应认真考虑停止 CPQ 路线。

---

# 10. 下一轮不要直接重新跑三个完整 200 epoch T5

先做 tokenizer-only diagnostics：

| 指标 | LARS-only | CPQ | CPQ-NoCoef | CPQ-NoRec | CPQ-Minimal |
|---|---:|---:|---:|---:|---:|
| hard collision | | | | | |
| hard PPL L0 | | | | | |
| hard PPL L1 | | | | | |
| hard PPL L2 | | | | | |
| top-1 token share | | | | | |
| top-10 token share | | | | | |
| prefix@1 unique | | | | | |
| prefix@2 unique | | | | | |
| hard reconstruction | | | | | |
| semantic kNN preservation | | | | | |
| LAR reconstruction | | | | | |

随后进行短 T5 validation screening。

只有 tokenizer 结构和 validation 都健康的版本才进入正式 200 epoch T5。

禁止根据 test 指标反复选择配置。

---

# 11. 必须增加：真实 beam prefix survival

最终需要解释：

\[
\text{为什么 CPQ SID 更难被 T5 推荐？}
\]

对于真实 target：

\[
SID^*
=
(s_1^*,s_2^*,s_3^*),
\]

记录实际 beam=30 生成过程中：

### Prefix-1 survival

真实：

\[
s_1^*
\]

是否仍在 beam 内。

### Prefix-2 survival

真实：

\[
(s_1^*,s_2^*)
\]

是否仍在 beam 内。

### Full SID survival

完整：

\[
(s_1^*,s_2^*,s_3^*)
\]

是否仍在 beam 内。

对比：

| Method | Prefix-1 survival | Prefix-2 survival | Full SID survival |
|---|---:|---:|---:|
| LARS-only-HRQ | | | |
| CPQ | | | |

如果 CPQ 在 level 1 就大幅下降：

> 第一层 token clustering / frequency / conditional entropy 很可能有问题。

如果 level 1 正常，而 level 2 / 3 大幅下降：

> hierarchical residual organization 更值得怀疑。

---

# 12. 必须增加：条件熵诊断

好的 SID 不只是 collision 少。

还需要每一层比较容易被自回归模型预测。

建议统计训练序列上的：

\[
H(S_1^{next}\mid History),
\]

\[
H(S_2^{next}\mid S_1^{next},History),
\]

\[
H(S_3^{next}\mid S_1^{next},S_2^{next},History).
\]

CPQ 可能提高了：

\[
\text{item discrimination}
\]

但恶化了：

\[
\text{prefix predictability}.
\]

这两者并不等价。

---

# 13. 下一轮研究问题应该发生变化

原来的问题是：

> 如何尽可能完整地把 LARS coefficient 压进 SID？

更合理的新问题是：

\[
\boxed{
\text{如何从 LARS path 中选择值得离散化的信息，
使 SID 同时保持 item discrimination 与 autoregressive prefix predictability？}
}
\]

这个问题更贴近 TIGER 类生成式推荐。

---

# 14. 如果 CPQ-Minimal 仍失败：停止“组合 token”路线

如果：

\[
\text{CPQ-Minimal}
<
\text{LARS-only-HRQ},
\]

则不建议继续通过：

- 增大 codebook；
- 增加更多 loss；
- 增加 teacher；
- 增加训练 epoch；

来修补 CPQ。

应该回到：

\[
\boxed{
\text{一个 SID token 对应一个基础 LARS direction}
}
\]

但改进：

> LARS active set 中到底选择哪个 direction 作为最终 token。

一个值得进一步研究的候选方向是：

# Predictability-Aware LARS Selection

LAR 仍然得到：

\[
a_{i\ell}
\]

和 active set：

\[
\mathcal A_i.
\]

最终 token 不再直接取：

\[
\arg\max |a_k|.
\]

而是在 active set 中选择：

\[
\boxed{
s_i
=
\arg\max_{k\in\mathcal A_i}
\left[
\alpha\cdot
\mathrm{ResidualGain}(k)
+
\beta\cdot
\mathrm{PrefixPredictability}(k)
+
\gamma\cdot
\mathrm{UsageRegularity}(k)
\right].
}
\]

这样：

- LARS 仍负责发现候选 directions；
- 最终 SID 仍是 direction index；
- 不引入 composition-token parameterization；
- 不增加离散状态数；
- 直接优化“哪个 LARS direction 更适合成为 autoregressive token”。

这个方向应等 CPQ 诊断完成以后再正式设计。

---

# 15. CPQ 失败真正带来的研究信息

这轮实验至少提供了三个重要信息。

## 15.1 更完整保留 LARS 连续组合不是充分条件

\[
\boxed{
\text{more sparse-code fidelity}
\not\Rightarrow
\text{better generative recommendation}
}
\]

## 15.2 SID 不是普通 reconstruction code

一个生成式推荐 SID 至少同时涉及：

1. semantic structure；
2. item discrimination；
3. collision；
4. code usage；
5. prefix frequency；
6. hierarchical organization；
7. autoregressive conditional predictability；
8. next-item behavior。

因此仅优化重构或 coefficient fidelity 都可能失败。

## 15.3 LARS 的价值可能更多体现在 candidate discovery

LARS 回答的是：

> 当前 residual 中哪些 directions 最重要？

而最终 SID 要回答的是：

> 哪一个离散 token 最值得让 T5 去预测？

两者相关，但并不是同一个问题。

---

# 16. 当前实验优先级

建议严格按照以下顺序：

~~~text
1. 核查 Beauty / Toys failed 的真实原因
2. 核查 result / best checkpoint / SID vocab 是否一一对应
3. 对 CPQ 与 LARS-only 做 hard SID audit
4. 做 prefix survival 分析
5. 做 conditional entropy 分析
6. 跑 CPQ-NoCoef
7. 跑 CPQ-NoRec
8. 跑 CPQ-Minimal
9. 只有诊断健康的版本才跑完整 200 epoch T5
10. 如果 CPQ-Minimal 仍明显失败，停止 CPQ 路线
11. 回到纯 LARS direction-token，研究 predictability-aware selection
~~~

---

# 17. 当前版本最终判定

当前：

~~~text
20260927_lars_cpq
~~~

应标记为：

\[
\boxed{
\text{正式失败候选，但具有诊断价值}
}
\]

不能作为优于现有方法的结果。

也不应当立即通过增加更多复杂模块继续修补。

当前优先任务是拆清：

\[
\boxed{
\text{coefficient matching}
\quad vs\quad
\text{recommendation auxiliary}
\quad vs\quad
\text{composition pattern}
\quad vs\quad
\text{初始化 / 实现问题}
}
\]

分别承担了多少退化。

---

# 18. 一句话总结

这次 CPQ 最值得记住的是：

\[
\boxed{
\text{LARS 能找到更丰富的连续稀疏组合，
但把更多 LARS 信息塞进 SID，并不自动让 SID 更适合自回归推荐。}
}
\]

下一轮真正值得优化的是：

\[
\boxed{
\text{LARS 信息的离散选择质量}
+
\text{SID prefix predictability}
}
\]

而不是继续单纯追求：

\[
\text{LARS coefficient preservation}.
\]
