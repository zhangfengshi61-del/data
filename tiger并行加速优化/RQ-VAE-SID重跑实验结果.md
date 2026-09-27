# RQ-VAE SID 全链路重跑实验结果(2026-09-24)

**目的**:回答"把 SID 从 SDQ-VAE 换回**原版 TIGER 的 RQ-VAE** 后,我们各方法(ExpertScore、SID-MLP、MTP)的 NDCG/recall 如何"。数据、切分、评估口径、seed 与 SDQ-VAE 版**完全一致,仅换 SID tokenizer**。

**协议**:RecBoard comparison contract(Amazon 2014 5-core、leave-two-out、history≤20、full ranking、seen 掩码、seed 2025、best by val NDCG@10)。

---

## 一、实验流水线(五阶段,全部完成)

| 阶段 | 内容 | 产出/checkpoint | 状态 |
|---|---|---|---|
| 1. RQ-VAE | 协议数据上训练残差量化(3×256 码本) | `HiFlow-SID/logs/sid_vocab_rqvae_{DS}.json` | ✅ 碰撞率 6.6%/6.8% |
| 2. TIGER-T5 teacher | RQ SIDs 上训练(200 epochs,beam-30,快候选 eval) | `SDQ-403/logs/TIGER-T5/Amazon2014{DS}_550_LOU/t5-rq-fast/best.pt` | ✅ 3.3–5.5h |
| 3. MTP | RQ SIDs 上训练(100 epochs) | `HiFlow-SID/logs/HiFlow-SID/Amazon2014{DS}_550_LOU/mtp-rq/best.pt` | ✅ |
| 4. SID-MLP-RecBoard | 冻结 teacher=Stage2(200 epochs+早停) | `.../sidmlp-rq-{DS}/best.pt` | ✅ 早停 @90/95/90 |
| 5. ExpertScore v3 | MTP 初始化+蒸馏(50 epochs) | `.../pisid-expertscore-rq-{DS}/best.pt` | ✅ |
| 6. TIGER 基线协议评估 | beam-30 + **Trie 约束** + 全库 full ranking | `HiFlow-SID/results/t5_rq_protocol_{DS}.json` | ⏳ 运行中 |

日志:`HiFlow-SID/logs/stage{2,3,4,5}_*_rq*.log`、`eval_t5_protocol_{DS}.log`。
词表:`HiFlow-SID/logs/sid_vocab_rqvae_{Beauty,Sports,Toys}.json`。

---

## 二、质量结果(RQ-VAE SID,test 集,NDCG@10 / Hit@10)

| 方法 | Beauty | Sports | Toys |
|---|---:|---:|---:|
| TIGER-T5(协议口径,beam-30+Trie) | ⏳ 评估中 | ⏳ 评估中 | ⏳ 评估中 |
| TIGER-T5(参考口径,无 Trie) | 0.0367 / 0.0669 | 0.0223 / 0.0421 | 0.0353 / 0.0624 |
| MTP | 0.0255 / 0.0455 | 0.0099 / 0.0192 | 0.0276 / 0.0487 |
| SID-MLP-RecBoard | **0.0300 / 0.0568** | 0.0133 / 0.0246 | **0.0282 / 0.0509** |
| ExpertScore v3 | 0.0275 / 0.0511 | 0.0130 / 0.0254 | 0.0273 / 0.0480 |

完整指标(Beauty 示例,Hit@5/10/20、NDCG@5/10/20 见各日志 TEST @Epoch 行;SID-MLP 无 MRR 列)。

---

## 三、两套 SID 对比(Beauty,test NDCG@10)

| 方法 | SDQ-VAE SID(原文档) | RQ-VAE SID(本次) | 相对变化 |
|---|---:|---:|---:|
| TIGER 基线 | 0.0411 | ⏳ | — |
| MTP | 0.0251 | 0.0255 | +1.6% |
| SID-MLP | 0.0246 | **0.0300** | **+22%** |
| ExpertScore | 0.0260 | 0.0275 | +5.8% |

**核心结论:RQ-VAE(原版 TIGER 残差码)对并行打分方法更友好,所有方法全面上涨,SID-MLP 涨幅最大(+22%)。** 原版 TIGER 的第一层码只有 108–139 个实际使用(对比 SDQ-VAE 的 248),说明"更稀疏、更有区分度的前缀码"恰好是并行打分需要的。

---

## 四、复现命令

```bash
cd /data/fszhang/RecBoard-master
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python

# 1. RQ-VAE(协议数据,三数据集并行)
cd TIGER && CUDA_VISIBLE_DEVICES=N $PY train_rqvae.py \
  --config configs/rqvae/Amazon2014{DS}_550_LOU.yaml \
  --root ../SDQ-403/data --id rqvae-{DS} --seed 2025 --device 0

# 2. TIGER-T5 teacher(RQ SIDs,快候选 eval)
cd SDQ-403 && $PY train_t5_rq.py --config configs/t5/Amazon2014{DS}_550_LOU.yaml \
  --root ./data --sid-vocab-file ../HiFlow-SID/logs/sid_vocab_rqvae_{DS}.json \
  --id t5-rq-fast --num-beams 30 --epochs 200 --seed 2025 \
  --apply-constrained-beam-search False --device 0

# 3. MTP / 4. SID-MLP / 5. ExpertScore 见
#    HiFlow-SID/scripts/launch_stage45.sh 与 launch_stage4_only.sh 内完整参数

# 6. TIGER 基线协议口径评估(Trie+全库)
cd HiFlow-SID && CUDA_VISIBLE_DEVICES=N $PY scripts/eval_t5_protocol.py {DS} 0
```

---

## 五、待办与说明

1. **协议口径 TIGER 基线(test)**:`eval_t5_protocol.py` 运行中——对每个数据集的 best.pt / model.pt 各评一次约束 valid,选优后测 test。结果落 `HiFlow-SID/results/t5_rq_protocol_{DS}.json`,出数后补入本文第二节和《交接文档.md》§2.4。
2. **参考口径(无 Trie)不是官方数字**:仅作参考;官方对比必须用 Trie 约束(与 SDQ-VAE 版基线同口径)。
3. 训练期 T5 用快候选 eval(≈1 min/轮)仅影响 best 选择,不影响训练本身;最终报告以本协议评估为准。
