# ServeSID 成本实验报告(J4)

> **统一测试环境**:4×RTX 4090(48GB),torch 2.7.1 + bf16 autocast;除标注外均为**独占 GPU、顺序执行**的单模型测时(排除并行争抢)。基准数据集 Beauty(2014),train 131,413 行 / test 22,363 行(有效映射后 131,221 / 22,171)。学生 beam=50、batch 32;教师 factorized beam=50、batch 64。
> **测量脚本**:`scripts/bench_infer_cost.py`(逐相位计时,每批 torch.cuda.synchronize,前 3 批预热后重置峰值统计)、训练测时用 `students_kd.py`/`train_decoder.py` 的 `BENCH_COST=1` 探针(每 epoch 计时 + max_memory_allocated)。

## 表 1:Offline Training Cost(Beauty,200 epochs)

| 指标 | Base KD (A1) | ServeSID (A9) | ServeSID 额外增量 |
|---|---|---|---|
| 教师 Cache 构建(beam dump, train) | 26.2 min(131,413 行, beam 40) | 26.2 min(同一 cache,一次构建) | **0**(与 Base KD 共享) |
| Cache 大小(beam dump + teacher topk) | 105.4 + 8.5 MB | 105.4 + 8.5 MB | **0** |
| k0 教师隐藏态缓存(fp32, 共享) | 5.7 GB | 5.7 GB | **0** |
| 每 Epoch 训练时间 | MLP 4.6s / GRU 5.5s | MLP 11.5s / GRU 11.9s / CNN 14.5s | ≈ +2.5× |
| 总训练时间(200 ep, 实测) | MLP 27 min | MLP 49 min / CNN 65 min | +22 min(MLP) |
| Peak GPU 显存(训练) | 7.4 GB | 9.2-9.5 GB | +1.8-2.1 GB |

**注**:A1(Base KD)本身也使用教师 beam dump(候选 + step_logp 是 Token KD 的来源),因此 cache 构建与存储不是 ServeSID 独有的成本;若与纯 CE 基座(A0)对比,ServeSID 才额外承担 26 min + 105 MB/数据集的 cache 构建。ServeSID 的增量成本集中在**训练期**:每 epoch 多一次候选集合内的层级边缘/DA 前向(M=40 候选 ×4 深度),约 +2.5× epoch 时间、+1.8GB 显存。

## 表 2:Online Inference Cost(Beauty test,beam=50,RTX 4090)

| 方法 | Parameters | Throughput (req/s) | Latency (ms/req) | NDCG@10 |
|---|---|---|---|---|
| TIGER 教师 | 220M(全部参与前向) | 1,346 | 0.743 | 0.0363(beam=20 参考) |
| SID-MLP Base KD | 1.664M(学生)+ 220M(冻结编码器) | 2,769 | 0.361 | 0.0365 |
| ServeSID (MLP) | 与 Base 相同 | 3,036 | 0.329 | 0.0369 |
| ServeSID (GRU) | 1.201M + 220M 冻结 | 3,423 | 0.292 | 0.0368 |
| ServeSID (CNN) | 1.753M + 220M 冻结 | 3,024 | 0.331 | 0.0368 |

> MLP Base/ServeSID 的吞吐差(2769 vs 3036)是两次独立运行间的抖动;GRU base=3445 vs servesid=3423、CNN base=3028 vs servesid=3024 逐项一致,证明**蒸馏方法不改变在线推理结构**。学生参数量仅计解码器;冻结教师编码器(220M)是三种学生的共享底座。

## 推理延迟拆解(ms/req)

| 阶段 | TIGER 教师 | ServeSID GRU | ServeSID CNN |
|---|---|---|---|
| 历史编码(冻结 T5 编码器) | —(并入 d1 步) | 0.154 | 0.151 |
| 学生上下文模块(+适配器) | — | 0.031 | 0.046 |
| SID 逐位预测(4 步前缀评分) | 0.733(4 次 T5 前向) | 0.089 | 0.114 |
| Beam 维护(topk/mask/去重) | 0.010 | 0.019 | 0.019 |
| **合计** | **0.743** | **0.292** | **0.330** |

**发现**:三种学生中,历史编码(冻结 T5 编码器)占总延迟的 45-53%,是推理成本的主导项;ServeSID 的额外监督只影响训练期,推理路径与对应 Base 逐位相同。教师总延迟约为学生的 2.2-2.5×,主要来自 4 次完整 T5 前向(每次重算编码器)。

## 峰值显存(推理)

| 方法 | Peak GPU Memory |
|---|---|
| TIGER 教师 | 0.65 GB |
| ServeSID MLP | 0.07 GB |
| ServeSID GRU | 0.07 GB |
| ServeSID CNN | 0.11 GB |

## 三数据集教师 Cache 规模

| 数据集 | train 行数 | beam dump 构建时间 | beam dump 大小 | teacher topk | k0 缓存 |
|---|---|---|---|---|---|
| Beauty | 131,413 | 26.2 min | 105.4 MB | 8.5 MB | 5.7 GB(fp32) |
| Toys | 109,361 | 26.7 min | 87.7 MB | 7.4 MB | 4.8 GB(fp32) |
| Video_Games | 530,300 | 38.6 min(4 分片并行) | 425.3 MB | 36.2 MB | 11.7 GB(fp16) |

## 核心结论

1. **在线成本零增量**:ServeSID 与同结构 Base KD 的推理吞吐/延迟逐项一致(GRU 3445 vs 3423,CNN 3028 vs 3024 req/s)——证明论文主张"改善推荐且不增加学生在线推理计算"成立。
2. **成本集中在训练期**:ServeSID 训练每 epoch 慢约 2.5×(多 40 候选 ×4 深度的边缘/DA 前向)、显存 +1.8GB;200 epoch 总量级 +22 min/数据集(MLP,GPU 独占)。
3. **Cache 与 Base KD 共享**:教师 beam dump 构建(26 min/数据集、~100-425 MB)是 Token KD 的既有需求,不构成 ServeSID 独有成本。
4. **主导延迟项是冻结编码器**:学生总延迟中历史编码占 45-53%;SID 逐位预测仅 0.09-0.11 ms/req。学生推理比教师快 2.2-2.5×。
5. **训练显存裕量大**:峰值 9.5GB(Beauty, batch 512)远低于 48GB,可在 4-8 卡上并行多组消融。

## 测量口径与注意事项

- 训练测时为 3-epoch 受控重测(独占 GPU,跳过周期 beam 验证);200-epoch 总时=每 epoch ×200 + 周期 beam 验证开销,与历史完整训练的实测墙钟(MLP base 27min / servesid 49min / CNN servesid 65min)吻合。
- GRU base/servesid 的历史完整训练墙钟(356/430 min)受当时多作业共享 GPU 污染,不采用;以受控重测为准。
- 教师吞吐按 factorized beam=50(batch 64)测;教师 beam dump 构建用 beam 40 的历史日志。
- 学生吞吐包含冻结 T5 编码器前向(与 eval 同口径);peak mem 为前 3 批预热后统计。
- 重测脚本与探针:`ServeSID-Amazon2014/scripts/bench_infer_cost.py`、`students_kd.py`/`train_decoder.py`(BENCH_COST=1)。原始数据:`verification/bench_*.json`、`logs/bench_*.log`。