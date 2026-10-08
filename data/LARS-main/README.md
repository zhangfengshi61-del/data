# LARS-main

LARS（Learned Adaptive Residual Sparse）语义 ID 生成方法的可复现代码包，以及与 SDQ-VAE 的对照、三数据集调参流水线。

## 目录结构

```
LARS-main/
├── SDQ-LARS/                  # LARS 算法与 lars-main 实验流水线
│   ├── lars_solver.py         # LARS 稀疏最小二乘求解器（核心）
│   ├── lars_quantizer.py      # LARS 结构扩散量化器（三层 SID codebook）
│   ├── train_lars_vae.py      # LARS VAE 训练器（主入口，--lars-steps/sparse/bridge/warmup）
│   ├── run_lars_main_vs_sdq.py   # LARS-main vs SDQ-VAE 同协议对照（串行 4 阶段）
│   ├── tune_lars_main_binary.py  # lars-main 参数调优调度器（多 GPU 排队）
│   ├── run_lars_tune_one.py      # 单候选 worker（VAE→T5，幂等/锁保护）
│   ├── run_lars_vae_batch.py     # VAE 批量预跑器（跳过已跑/在跑候选）
│   ├── monitor_sports_results.py # 调参结果汇总监控
│   ├── watch_sports.py           # 结果后台监视器
│   ├── run_lars_common_clothing_fix.sh   # Clothing 单数据集对照
│   ├── run_lars_stable_prefix_*.sh       # StablePrefix 变体
│   └── grid_2026*.json          # 各轮调参候选网格（st/ty 12 组合、sports r2-r4）
└── SDQ-403/                   # SDQ 基准与 T5 评估协议（lars-main 的依赖）
    ├── train_sdq_vae.py       # SDQ-VAE 基线训练
    ├── train_t5_fp32.py       # T5 生成式排序（同协议评估，beam 30 / eval batch 96）
    ├── converter.py / partition.py / quantizer.py / sampler.py / utils.py
    ├── eval_t5_ckpt.py        # 任意 best.pt 的离屏评估工具
    ├── prepare_amazon_550_lou.py / prepare_local_data.py   # 数据预处理
    └── configs/{sdq,t5}/Amazon2014*_550_LOU.yaml           # Beauty/Sports/Toys/Clothing/Electronics 配置
```

## 依赖

- Python 环境需安装：`freerec`、`torch`、`transformers`（T5）、`numpy/pandas`
  （开发环境为 conda `myenv_t5`；可用 `PY=<python路径>` 环境变量覆盖脚本内的解释器）
- 数据：Amazon 2014 5-core，550_LOU 处理（预处理脚本见 `SDQ-403/prepare_*`）
- 已处理数据目录结构放在 `SDQ-403/data/`，VAE/T5 日志写入 `SDQ-LARS/logs/`、`SDQ-403/logs/`，
  调参运行产物写入 `SDQ-LARS/runs/`（这些目录已从仓库中排除）

## 快速使用

**1. LARS-main vs SDQ-VAE 对照（单数据集）**

```bash
python SDQ-LARS/run_lars_main_vs_sdq.py --dataset Electronics --gpu 0 --run-name <名字>
# 依次执行 sdq_vae → sdq_t5 → lars_vae → lars_t5，已完成阶段自动跳过
# 结果：SDQ-LARS/runs/<名字>/<数据集>/result.json
```

**2. lars-main 参数网格调参（多卡并行）**

```bash
# 启动调度器（4 卡、每个候选 VAE 100ep → T5 200ep）
python SDQ-LARS/tune_lars_main_binary.py \
    --run-name <名字> --dataset Sports --stage 0 --gpus 0,1,2,3 \
    --candidates-json SDQ-LARS/grid_20261001_sports_r4.json

# 可选：先用批量预跑器把 VAE 阶段并行摊开（调度器遇到已完成 VAE 会直接接 T5）
python SDQ-LARS/run_lars_vae_batch.py \
    --candidates-json SDQ-LARS/grid_20261001_sports_r4.json \
    --dataset Sports --run-name <名字> --start 0 --slots 1,2,3,0
```

候选 json 字段：`name / lr / dropout / weight_decay`，可选 `lars_steps / sparse_weight / bridge_weight / lars_warmup`（默认 3 / 0.1 / 0.1 / 10）。

**3. 离屏评估任意 T5 best.pt**

```bash
python SDQ-403/eval_t5_ckpt.py \
    --best-pt <路径>/t5/best.pt \
    --sid-vocab-file <路径>/vae/sid_vocab.json \
    --description EVAL-xxx --device 0
```

## 协议（固定）

VAE 100 epochs；T5 200 epochs、fp32、train batch 512、eval batch 96、beam 30、
`apply-constrained-beam-search=False`、seed 2025；按 validation NDCG@10 选 checkpoint，
报告对应 test 指标。

## 主要结果（详见 `我的方向/5/LARS-main_三数据集最优结果汇总.md`）

| 数据集 | 最优参数 | test NDCG@10 | vs SDQ-VAE |
|---|---|---:|---|
| Beauty | lr 5e-4, d0, wd0 | 0.043046 | +4.80% |
| Toys | lr 1e-3, d0, wd1e-4 | 0.041315 | +6.23% |
| Sports | 39 组参数全负，最优未复现 | — | — |
