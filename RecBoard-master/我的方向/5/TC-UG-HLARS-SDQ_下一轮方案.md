
## 7. UG-HLARS-SDQ 正式结果与失败诊断（2026-09-26）

Beauty 的正式结果已经完成，按 validation NDCG@10 选择 epoch 95：

| 数据集 | Utility-Hier Test NDCG@10 | SDQ-VAE | 相对变化 |
|---|---:|---:|---:|
| Beauty | 0.041210661 | 0.041072751 | +0.336% |
| Toys | 0.039517915 | 0.038893436 | +1.606% |
| Sports | 仍在训练 | 0.026651102 | 待定 |

Beauty 没有达到自动化要求的 1% 门槛，因此不能把当前版本判定为三数据集 SOTA。Beauty 的 validation NDCG@10 为 0.058967，T5 最佳 checkpoint 的 test NDCG@10 为 0.041211；epoch 200 的 test 0.036568 不作为正式结果。

### 7.1 失败原因

1. **连续重构改善没有完全传递到推荐。** Beauty 的 VAE PPL 从 SDQ 的 217.318 降到 201.680，level-0/1 PPL 从 181.537/214.638 降到 168.849/180.674，但 Test NDCG@10 只有 +0.336%。
2. **完整 SID 碰撞上升。** Beauty collision rate 从 0.142633 升到 0.193868；Toys 从 0.200939 升到 0.256122。当前 collision loss 只在随机 pair 上工作，且权重 0.02，无法约束导出的硬 SID。
3. **前缀使用变得集中。** Beauty 的 level-1 soft SID 熵从 5.369 降到 5.197，Toys 从 5.343 降到 5.231。唯一 code 数量接近 256 不能代表使用均衡，头部 code 频率已经明显增大。
4. **效用门槛过低。** 当前 utility margin=0、temperature=0.05，微小的 batch transition gain 也会推动稀疏教师桥接；这会让几何上有利但泛化不稳定的改动进入 SID。

因此下一轮需要把“可用的稀疏方向”限制在 SDQ 的 SID 信赖域内，同时显式控制软 SID 边际分布和完整 SID 碰撞。

### 7.2 下一轮方法：TC-UG-HLARS-SDQ

TC-UG-HLARS-SDQ（Trust-Calibrated Utility-Gated Hierarchical LARS-SDQ）保留 LARS 内层、SDQ 三层硬量化和原版 T5，只改变训练期约束：

- `sparse_weight=0.015`，降低连续目标对码本的漂移；
- `anchor_weight=0.75`、`sid_anchor_weight=0.25`，更强地保持 SDQ 的方向和离散 SID；
- `utility_margin=0.03`、`utility_temperature=0.02`，只有有明显推荐效用的稀疏教师才参与桥接；
- `collision_weight=0.10`、`collision_margin=0.01`，并启用 `diversity_weight=0.05`，直接压制 dissimilar item 的完整 SID 重合；
- 新增 `usage_balance_weight=0.02`，对每层 batch marginal soft SID 分布施加 KL-to-uniform 正则，缓解头部 prefix 集中；
- `soft_bridge_weight=0.03`、`hier_weight=0.02`，降低层级推荐辅助项的扰动；
- LARS budget 继续使用 `3,2,1`，T5 仍为 200 epochs、beam=30、seed=2025。

新增代码参数位于 `train_lars_vae.py`，运行器已加入 `tc_ug` variant。下一轮串行启动脚本为：

```bash
/data/fszhang/RecBoard-master/SDQ-LARS/run_tc_ug_hlars_serial.sh 20260926_tc_ug_hlars
```

该脚本目前只完成准备，未启动训练，以免打断当前 Sports 正式实验。启动后仍按 Beauty、Sports、Toys 顺序运行，并沿用当前 VAE 100 epoch、T5 200 epoch、beam=30 的协议。

### 7.3 下一轮验收条件

进入正式 T5 之前，必须同时检查：

- 三层 collision rate 不高于对应 SDQ baseline；
- level-0/1 的 usage entropy 不低于当前 utility_hier；
- SID anchor 的改变比例受控；
- T5 validation NDCG@10 不低于 SDQ baseline。

最终仍以 validation NDCG@10 选 checkpoint，并报告对应 Test NDCG@10；未完成的 Sports 和下一轮结果不写入当前主结果表。
## 正式结果更新（2026-09-26）

Sports 已完成：best validation epoch 65，Test NDCG@10=0.0240369428，较 SDQ-VAE 0.02665110 为 -9.809%。Beauty=0.0412106614（+0.336%），Toys=0.0395179150（+1.606%）。由于 Beauty 未达 +1% 且 Sports 低于基线，UG-HLARS-SDQ 未通过；本文件中的下一轮 TC-UG-HLARS-SDQ 方案保持不变，尚未启动。

## CP-UG-HLARS-SDQ：保守前缀校准方案（2026-09-26）

### 动机

utility_hier 在 Sports 的 test NDCG@10 为 0.0240369428，较 SDQ-VAE 0.02665110 低 9.809%。诊断显示 level-0/1 PPL 从 SDQ 的 172.94/225.19 降到 158.91/200.11，完整 SID collision 从 0.195893 升到 0.249224；原 utility gate 在零收益时仍输出约 0.5，导致前缀塌缩。短跑也确认，仅增加 collision 权重而不保护 assignment 会使 level-0 PPL 快速跌到 80，因此该配置被拒绝，不进入正式结果。

### 方法主体

CP-UG 仍然是 LARS/SDQ 的三层稀疏量化与 coarse-to-fine 训练。改动只用于保持推荐相关的 SID 决策边界：

1. 从 SDQ `model.pt` warm-start，前 100 个 VAE epoch 冻结 encoder/codebook，保证硬 SID 与基线一致；LARS 的结构、稀疏量化和训练预算仍执行。
2. 仅当 batch transition gain 大于 `utility_margin=0.001` 时开放 level-0 utility bridge；level-1/2 不参与 bridge，避免无效前缀更新。
3. 加入 baseline SID cross-entropy、逐层 marginal KL 和最近码 assignment hinge（权重 0.75、0.50、2.0），并直接约束 full-SID overlap。
4. T5 训练使用同一 SID vocabulary，在 teacher forcing 中对早期 SID token 加 0.50 小权重，改善前缀判别；推理仍只调用一次标准 beam=30 full-ranking。

### 公平协议

Beauty、Sports、Toys 串行；VAE 100 epochs、T5 200 epochs、seed=2025、fp32、训练 batch=512、validation/test batch=96、beam=30、`apply-constrained-beam-search=False`、num_workers=0。按 validation NDCG@10 选择 checkpoint，再读取对应 test NDCG@10；无 reranker、support score、候选二次打分。运行器：

```bash
/data/fszhang/RecBoard-master/SDQ-LARS/run_cp_ug_hlars_serial.sh 20260926_cp_ug_hlars
```

只有三个数据集都达到相对 SDQ-VAE 至少 +1% 才通过。任一数据集低于门槛，保留日志并基于 PPL、collision、SID 使用率、validation/test NDCG 进入下一轮方案；不把冒烟或未完成结果写入主表。

### 短跑验收

- `CP-UG-SMOKE2`（warmup=1）拒绝：Sports epoch 3 PPL#0=80.67、collision=0.2694。
- `CP-UG-SMOKE3`（warmup=10、freeze=10）通过安全性验收：epoch 10 仍为 SDQ PPL=217.9585、collision=0.1959；解冻后 pilot 继续监测 assignment 漂移。
- 正式候选采用 freeze=100，并把收益放在 T5 前缀校准项，避免改变 SDQ 硬 SID 分布。


## 2026-09-27 CP-UG-HLARS-SDQ Sports 正式结果与失败诊断

- 运行：`/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260926_cp_ug_hlars/Sports/`，沿用 SDQ VAE SID vocabulary；T5 200 epochs、seed=2025、fp32、beam=30、训练 batch=512、评估 batch=96、plain full-ranking beam（`apply-constrained-beam-search=False`）。
- 按 validation NDCG@10 选中的 checkpoint：`/data/fszhang/RecBoard-master/SDQ-LARS/logs/LARS-20260926_cp_ug_hlars-T5/Amazon2014Sports_550_LOU/t5/data/best.pkl`。
- 最佳 validation NDCG@10：`0.03473028106894046`（约第 17 个评估点）；对应 test NDCG@10：`0.02518133539533798`。
- 相对固定 SDQ-VAE test `0.02665110`：`-5.5148365533%`，未通过 +1% 门槛。
- 监控：Sports validation NDCG@10 在早期达到 0.03473028 后持续下降至约 0.02853；说明 prefix calibration 权重 0.5 在 Sports 上出现明显早期过拟合。VAE SID 诊断基线为 PPL 217.9585、collision 0.19589；CP-UG 只改变 T5 训练期 token CE，不改变 SID vocabulary 或一次生成解码，因此失败主要来自 T5 prefix loss 的泛化，而非二阶段排序。

## 下一轮 APC-UG-HLARS-SDQ（Adaptive Prefix Calibration）

为保持 LARS 主体和 TIGER 公平协议不变，新增训练期线性 prefix-loss annealing：prefix token 权重从 0.50 在前 60 个 epoch 线性退火到 0.10，之后保持 0.10；推理仍为一次标准 beam=30 full-ranking，禁止 constrained beam、support score、reranker 和二次打分。该退火降低 Sports 的早期过拟合，同时保留 Beauty 上 CP-UG 的 prefix 校准收益。

- 代码：`/data/fszhang/RecBoard-master/SDQ-403/train_t5_fp32.py` 新增 `--prefix-loss-min-weight` 与 `--prefix-loss-decay-epochs`。
- 串行脚本：`/data/fszhang/RecBoard-master/SDQ-LARS/run_cp_ug_adaptive_serial.sh`。
- 正式参数：max=0.50、min=0.10、decay=60；Beauty→Sports→Toys 串行，仍使用 batch=512/96、fp32、beam=30、seed=2025。
- 当前 CP-UG Toys 完成后，确认无重复进程和 GPU 资源，再启动新一轮；旧运行和 checkpoint 保留，不混用。


## CP-UG 本轮 Toys 完成与 APC-UG 启动（2026-09-27）

- CP-UG Toys：最佳 validation NDCG@10 `0.053197000799794655`，对应 test NDCG@10 `0.03980501310071969`，相对 SDQ-VAE `+2.343770828%`。
- CP-UG 三数据集正式结果：Beauty `0.04340491972472871`（`+5.678143598%`）、Sports `0.02518133539533798`（`-5.514836553%`）、Toys `0.03980501310071969`（`+2.343770828%`）；因 Sports 未达到 +1%，本轮不通过。
- 已完成无重复进程与 GPU 审计，GPU3 空闲后启动 APC-UG-HLARS-SDQ 串行轮次：`/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_cp_ug_adaptive/`。当前 Beauty 正在训练，Sports/Toys 尚未启动。
- APC 参数：prefix-loss max=0.50、min=0.10、linear decay=60 epochs；其余协议完全保持不变。


## LARS-only-HRQ：完全脱离 SDQ 的 SID 生成方案（2026-09-27）

### 方法动机

此前 CP/APC 仍然沿用了 SDQ 的 SID 词表或 SDQ 初始化。为验证方向5的 LARS 本身能否独立生成可用于生成式推荐的 SID，新增 **LARS-only-HRQ（Hierarchical Residual LARS Quantization）**。该方案不导入 SDQ `quantizer.py`、SDQ ODE/structure diffusion、Sinkhorn、SDQ codebook、SDQ checkpoint 或 SDQ `sid_vocab.json`。

### SID 生成

1. 从 item 的 sentence-T5 semantic feature 出发，训练一个轻量 LARS encoder。
2. 建立 3 个全新的 codebook，每层 256 个 codeword；初始 codeword 从当前数据集特征随机采样，不能使用 SDQ 初始化。
3. 对 encoder 输出逐层计算残差。每层用预算截断的 `lar_code`（LARS active-set）选择当前残差相关性最大的稀疏 atom，输出该 atom 的离散索引；再从残差中减去该 LARS atom，进入下一层。
4. 训练目标由 LARS hard residual reconstruction、commitment、codeword diversity 和轻量 usage entropy 组成，只用于防止残差失真与 codebook collapse。
5. 100 个 VAE epoch 后导出全量 `sid_vocab.json`，每个物品正好 3 个新 SID token：`<sid_0_i>, <sid_1_j>, <sid_2_k>`。

训练代码：`/data/fszhang/RecBoard-master/SDQ-LARS/train_lars_only_vae.py`。该文件的 `sdq_imports=false` 审计字段用于确认没有引入 SDQ 量化器。

### 下游协议

SID 生成后完全按原协议训练 T5：200 epochs、seed=2025、fp32、train batch=512、validation/test batch=96、beam=30、`apply-constrained-beam-search=False`，按 validation NDCG@10 选 checkpoint 并读取对应 test NDCG@10。T5 只读取本轮新生成的 LARS-only `sid_vocab.json`；不用 SDQ SID，不使用 reranker、support score 或二阶段精排。

### 初步审计

Beauty 100 epoch 原型：3 层均使用 256 个 codeword，三层 unique code 均为 256，完整 SID collision rate `0.1088339807`，低于 SDQ Beauty 基线 `0.1426328403`。该结果只证明 tokenizer 的结构可用，不能当作正式推荐结果；正式结果必须等待三数据集 T5 完成。

### 正式并行运行

正式运行目录：`/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260927_lars_only/`。Beauty、Sports、Toys 分别使用 GPU 0、1、2 并行执行 VAE 100 epoch 后 T5 200 epoch；GPU3 的 APC 任务保持不受影响。
