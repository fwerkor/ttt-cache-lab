# TTT Cache Lab

TTT Cache Lab is an experiment framework for studying **versioned KV cache management under inference-time adapter evolution**.

The project focuses on the following setting:

```text
adapter version v0
  -> train/update during inference
adapter version v1
  -> train/update again
adapter version v2
  -> ...
```

A KV cache block is produced by a specific model/adaptor version. Once the adapter changes, the old cache may be exact, approximately usable, stale-but-tolerable, or invalid. This repository provides the code structure needed to measure that drift and evaluate cache-maintenance strategies.

The framework supports lightweight toy diagnostics and Hugging Face experiments on CPU or CUDA GPUs.

## Research question

> During inference-time adapter training, when an adapter evolves from `v0` to `v1`, `v2`, and later versions, which KV cache blocks can be reused, which can be corrected, which should be refreshed, and which must be recomputed?

The project is not just about Q-only qTTT. Existing work already covers parts of static adapter reuse and multi-LoRA serving. This repository targets the dynamic case where the **same adapter keeps changing during inference**.

## Current implementation status

Implemented:

- controlled retrieval, rejection, multi-hop, aggregation, set-reasoning, and state-tracking diagnostics plus JSONL and Hugging Face loaders;
- deterministic calibration/validation/test partitions, LongBench and LongBench v2 field mapping, multiple-choice, numeric, set, QA, summarization, and code scoring;
- exact tokenizer-level context sizing with explicit `error`, `left`, and `middle` truncation policies;
- Q/K/V/O/QV/attention/MLP/Norm/output-head update targets, layer-positioned LoRA variants, and separate MoE router/shared-expert/routed-expert targets;
- answer-supervised online LoRA updates, global-L2 update-norm control, and multi-token exact-match generation scoring;
- versioned per-layer cache metadata, adapter/version indexing, relative update norm since the last refresh, and executable rejection semantics;
- full recomputation, stale/frozen reuse, LoRA K/V delta correction, native Llama-like/Gemma 3/GPT-2 layer restart, periodic/threshold policies, measured oracle selection, and adaptive planning;
- fixed-adapter E1 baselines, including executable aLoRA-style invocation-prefix reuse, LRAgent-style per-adapter caches, and ForkKV-style base/delta decomposition;
- real KV tensor byte counts, peak allocated memory, adaptation/cache/decode/end-to-end latency, throughput, cache-entry counts, and configurable tensor/task metrics;
- lightweight E1-E7 templates plus a frozen E1-E8 paper matrix with controlled, LongBench, LongBench v2, code, scaling, ablation, and cache-pressure workloads;
- single-model decoder-layer sharding across CUDA GPUs and Ascend NPUs, including a hardware-validated 4×NPU Qwen2.5-7B 8K path and 6×GPU/8×NPU Qwen2.5-32B templates;
- condition-preserving E1-E7 analyses that retain model, context, rank, update norm, seed, and sweep axes and pair every strategy with the exact full-recompute reference;
- E3-calibrated E4 planning with failure-map artifact hashes, runtime action-latency budgets, explicit cache-manager scopes, and measured-oracle provenance;
- E5 correction/fallback diagnostics, E6 latency/speedup/task-drop scaling plots, E7 paired ablation effects, E8 tail-latency/cache-pressure reports, and adaptation-gain/update-scale reports;
- cluster-bootstrap confidence intervals, paired comparisons, Wilson false-safe bounds, warm-up/repeated timing, and p50/p95 latency reporting;
- a shardable 84-configuration, 252-job manifest spanning 1.5B, 3B, 4B, 7B, 14B, and 32B models, dense/sliding-window/local-global/MoE architecture transfer, and a 7B code model;
- atomic per-target checkpoints, record-level resume, cross-run record merging, structured failure manifests, and run metadata with config/git/package provenance;
- baseline-only task probes that preserve generated answers, score distributions, latency, memory, and explicit all-zero/all-one degeneracy flags before expensive runs;
- CI for linting, strict type checking, unit tests, and offline tiny-Llama integration tests that execute real LoRA, KV delta correction, and native layer restart paths.

Remaining work is paper-scale hardware validation rather than placeholder implementation: Qwen2.5-7B has passed real Ascend smoke, two-NPU feasibility, and four-NPU 8K stress runs, while the full long-context matrix and 14B/32B configurations still need complete measurements on the selected accelerator infrastructure. The aLoRA/LRAgent/ForkKV-style methods are explicitly labeled as paper reimplementations and should still be compared with official upstream implementations where licensing and environments permit.

## 论文实验进度 / Paper experiment progress

> 最后人工核对：**2026-07-13 10:36 +08:00**。本节是论文数据的人工维护清单；不做复杂自动同步，需要更新时直接核对运行产物并修改勾选。

### 状态规则

- `✅`：已验收为论文数据。task probe 通过；配置要求的全部 sample × target × strategy × version 条件齐全；存在 `run_metadata.json` 和汇总文件；没有未解决的 `run_failure.json`。
- `◐`：正在运行或仅部分完成。备注中必须写清尚缺的 target、strategy、version、sample 或失败条件。
- `⬜`：尚未验收。Smoke、少样本探索、旧协议结果和被新配置替代的结果都不能勾选。
- `⚠`：配置本身与冻结协议不一致，必须先修配置再运行。
- 每个 seed 单元对应 `runs/paper/study/<job-name>/seed-<seed>/`；一项配置只有三个 seed 全部为 `✅` 才算完成。

### 正式数据总览

| 实验组 | 配置数 | Seed 运行数 | 已验收 | 剩余 |
|---|---:|---:|---:|---:|
| E1 · 静态适配器基线 | 1 | 3 | 0 | 3 |
| E2 · 参数版本漂移 | 6 | 18 | 0 | 18 |
| E3 · 失效图校准 | 24 | 72 | 0 | 72 |
| E4-V · Planner 验证集调参/冻结 | 4 | 12 | 0 | 12 |
| E4-T · Planner 最终测试 | 11 | 33 | 0 | 33 |
| E5 · Delta correction 安全域 | 12 | 36 | 0 | 36 |
| E6 · 上下文与模型扩展 | 11 | 33 | 0 | 33 |
| E7 · Planner 消融与失效边界 | 1 | 3 | 0 | 3 |
| E8 · 容量压力与尾延迟 | 2 | 6 | 0 | 6 |
| A1 · 跨架构迁移筛查 | 12 | 36 | 0 | 36 |
| **冻结主矩阵合计** | **84** | **252** | **0** | **252** |

> 这里的“已验收”刻意采用保守口径。已有 Ascend smoke/stress 与 B/W 探索结果用于验证实现和选择研究路线；在最终论文规模产物完成核对前，不计入 252 个正式 seed 运行。

### 当前执行批次：`formal_20260712`

该批次是正式矩阵的分阶段执行子集，共 **219 个 seed-run**。批次结果只有通过完整性和质量验收后，才会计入上方冻结主矩阵的“已验收”列。

| 队列 | 覆盖范围 | 总数 | 成功 | 失败 | 运行中 | 未开始 | 当前状态 |
|---|---|---:|---:|---:|---:|---:|---|
| `small0` | Qwen2.5-1.5B：W1-W4 + E3 calibration | 30 | 9 | 0 | 1 | 20 | W1/W2/W3 的 3 个 seed 已完成；W4 seed 7 正在生成 blockwise oracle 产物 |
| `seven13` | Qwen2.5-7B：E1/E2/W1-W3/E3/E5/E6 | 63 | 0 | 6 | 1 | 56 | E2 controlled seed 7 在答案专属 loss 修复后又于 output-head 随机扰动范数计算处 OOM；已修复大张量范数/Delta 临时分配，归档失败目录并启动干净重跑。E2 LongBench-v2 seed 7 仅有部分产物，未验收 |
| `fourteen4567` | Qwen2.5-14B：E2/E3/E6 | 30 | 0 | 24 | 0 | 6 | 已结束的 E2/E3 共 24 个 seed 全部在 LoRA 更新阶段 NPU OOM；E6 8K seed 7 已停止且仅有部分产物，无 `.success`，不计为完成 |
| `arch13` | Llama/Gemma/Mistral/MoE 架构筛查 | 36 | 0 | 0 | 0 | 36 | 尚未启动 |
| `sevenlong4567` | Qwen2.5-7B：32K/64K E6 | 6 | 0 | 0 | 0 | 6 | 尚未启动 |
| `longall` | Qwen2.5-14B：32K E6 | 3 | 0 | 0 | 0 | 3 | 尚未启动 |
| `thirtysix` | Qwen2.5-32B：E2/E3/E5/E6 | 51 | 0 | 0 | 0 | 51 | 尚未启动 |
| **合计** |  | **219** | **9** | **30** | **2** | **178** | 当前仅 W4 seed 7 与修复后的 E2 7B controlled seed 7 在运行；无新增正式验收完成项 |

稳定性处理原则：暂停会继续批量产生 OOM 的大模型队列；保留可恢复的 W4；先用单个 seed、最小目标集合和内存峰值保护验证 7B，再逐步放开 14B/32B。2026-07-13 已进一步消除 output-head 随机扰动路径中的 float64 全量临时张量和重复 Delta 缓冲。失败的 seed 不作为论文数据，修复后从检查点或干净目录重跑。

### W/B：机制发现与 Planner 探索线

| ID | 模型 / 任务 | 精确条件 | 数据状态 | 最终验收产物 / 备注 |
|---|---|---|---|---|
| W1-1.5B | Qwen2.5-1.5B · multi-hop · 4K · n=16 | q/k/v/mlp 的 early/middle/late 位置；window=1/2/4/8/16/32；gap=1/4/16；full/stale/suffix 配对参照 | ◐ | `formal_20260712` 的 seed 7/17/29 均有 `.success`，但顶层 `run_metadata.json` 与 `minimal_safe_windows.csv` 均缺失，暂不验收 |
| W1-7B | Qwen2.5-7B · multi-hop · 8K · n=16 | 与 W1-1.5B 相同的配对 window 矩阵 | ⬜ | 当前批次尚未启动；需先通过稳定版 7B 单-seed 内存预检 |
| W2-1.5B | Qwen2.5-1.5B · multi-hop · 4K · n=16 | 8 个 target/position 条件；gap=1/4/16；逐层 hidden/K/V drift；32 probe tokens | ◐ | `formal_20260712` 的 seed 7/17/29 均有 `.success`、metadata 和原始传播记录，但缺少要求的 `propagation_profiles.csv`，暂不验收 |
| W2-7B | Qwen2.5-7B · multi-hop · 8K · n=16 | 与 W2-1.5B 相同的逐层传播矩阵 | ⬜ | 当前批次尚未启动；需先通过稳定版 7B 单-seed 内存预检 |
| W3-1.5B | Qwen2.5-1.5B · multi-hop · 4K · n=8 | 8 个 target/position 条件；gap=1/4/16；local-boundary 与 stale-suffix 信号；held-out predictor | ◐ | `formal_20260712` 的 seed 7/17/29 均有 `.success`、metadata 和原始边界记录，但缺少要求的 `boundary_predictor_summary.csv`，暂不验收 |
| W3-7B | Qwen2.5-7B · multi-hop · 8K · n=8 | 与 W3-1.5B 相同的 boundary predictor 矩阵 | ⬜ | 当前批次尚未启动；需先通过稳定版 7B 单-seed 内存预检 |
| W4/B1 oracle | Qwen2.5-1.5B · multi-hop · 4K · n=16 | target=k/q/mlp/v-middle；gap=4；block=32/64/128；budget=1/14、2/14；random/raw-drift/attention-weighted/layer-prefix/greedy/per-token oracle | ◐ | seed 7 正在运行且产物持续增长；seed 17/29 待运行；需 `block_frontier.csv`、`block_masks.csv`、`blockwise_report.md` |
| B2 static ranker | W4 calibration artifacts | zero-probe sparse block ranker；跨样本切分；confidence/safety gate | ◐ | 代码已实现；需 held-out KL 收益、误伤率、选中 cells 与 planner latency |
| B3 one-probe router | W4 calibration artifacts | prompt-anchor probe length=1/2/4；reference/baseline-reference policy | ◐ | 代码已实现；需 probe 成本、总延迟与 held-out quality |
| B4 committed router | W4 calibration + 独立 guard split | zero-probe direct commit/recompute gate；trust-band calibration | ◐ | 代码已实现；需无 KL runtime 评估与 false-safe 置信上界 |
| B5 dynamic controller | W4 calibration artifacts | risk-scaled 动态预算；activation threshold；max cells；marginal-gain stopping | ◐ | 代码已实现；需 held-out budget/quality/latency frontier |
| B6 planner 正式证据 | 晋级后的模型/任务矩阵 | 完整计时 probe + selection + repair；对比 full/stale/periodic/threshold/delta/suffix/oracle | ⬜ | B2-B5 选定可部署策略后冻结精确配置，并加入 `study.yaml` |

### 冻结的 84 项正式配置

下表逐项列出所有正式配置。“内部条件”是每个 seed 内部必须完整覆盖的实验轴；若某个 seed 只完成一部分，必须在该 seed 或备注中写明剩余条件。

<details>
<summary><strong>E1 · 静态适配器基线</strong> — 1 个配置 / 3 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`baseline_e1_qwen_7b_longbench_v2`](configs/paper/baseline/e1_qwen_7b_longbench_v2.yaml) | Qwen2.5-7B | LongBench-v2 · validation | 16K | n=96; offset=0; target=q/k/v/qv; norm=1e-3; r=8; v=0; cache=full/base-reuse/per-adapter/aLoRA/LRAgent/ForkKV/base+delta; repeat=0w+1t; parallel=single; adapter-seq=0/1/2/3/0/2/1/3 | ⬜ | ⬜ | ⬜ |  |

</details>

<details>
<summary><strong>E2 · 参数版本漂移</strong> — 6 个配置 / 18 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`drift_e2_qwen_14b_controlled`](configs/paper/drift/e2_qwen_14b_controlled.yaml) | Qwen2.5-14B | common_words · test · hard | 16K | n=48; offset=96; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16/32; cache=full/stale/frozen; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`drift_e2_qwen_14b_longbench_v2`](configs/paper/drift/e2_qwen_14b_longbench_v2.yaml) | Qwen2.5-14B | LongBench-v2 · test | 16K | n=96; offset=96; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`drift_e2_qwen_32b_controlled`](configs/paper/drift/e2_qwen_32b_controlled.yaml) | Qwen2.5-32B | common_words · test · hard | 16K | n=32; offset=96; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16/32; cache=full/stale/frozen; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`drift_e2_qwen_32b_longbench_v2`](configs/paper/drift/e2_qwen_32b_longbench_v2.yaml) | Qwen2.5-32B | LongBench-v2 · test | 16K | n=96; offset=96; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`drift_e2_qwen_7b_controlled`](configs/paper/drift/e2_qwen_7b_controlled.yaml) | Qwen2.5-7B | variable_tracking · test · easy | 16K | n=64; offset=96; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16/32; cache=full/stale/frozen; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`drift_e2_qwen_7b_longbench_v2`](configs/paper/drift/e2_qwen_7b_longbench_v2.yaml) | Qwen2.5-7B | LongBench-v2 · test | 16K | n=96; offset=96; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |

</details>

<details>
<summary><strong>E3 · 失效图校准</strong> — 24 个配置 / 72 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`calibration_e3_qwen_14b_aggregation`](configs/paper/calibration/e3_qwen_14b_aggregation.yaml) | Qwen2.5-14B | aggregation · calibration · hard | 16K | n=48; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_14b_multi_hop_tracing`](configs/paper/calibration/e3_qwen_14b_multi_hop_tracing.yaml) | Qwen2.5-14B | multi_hop_tracing · calibration · medium | 16K | n=48; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_14b_multi_needle`](configs/paper/calibration/e3_qwen_14b_multi_needle.yaml) | Qwen2.5-14B | multi_needle · calibration · hard | 16K | n=48; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_14b_variable_tracking`](configs/paper/calibration/e3_qwen_14b_variable_tracking.yaml) | Qwen2.5-14B | variable_tracking · calibration · hard | 16K | n=48; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_14b_common_words`](configs/paper/calibration/e3_qwen_14b_common_words.yaml) | Qwen2.5-14B | common_words · calibration · hard | 16K | n=48; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_14b_needle_absent`](configs/paper/calibration/e3_qwen_14b_needle_absent.yaml) | Qwen2.5-14B | needle_absent · calibration · hard | 16K | n=48; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_1_5b_aggregation`](configs/paper/calibration/e3_qwen_1_5b_aggregation.yaml) | Qwen2.5-1.5B | aggregation · calibration · easy | 4K | n=96; offset=0; target=q/k_early/k_middle/k_late/v_early/v_middle/v_late/qv/o/mlp_early/mlp_middle/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_1_5b_common_words`](configs/paper/calibration/e3_qwen_1_5b_common_words.yaml) | Qwen2.5-1.5B | common_words · calibration · easy | 4K | n=96; offset=0; target=q/k_early/k_middle/k_late/v_early/v_middle/v_late/qv/o/mlp_early/mlp_middle/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_1_5b_multi_hop_tracing`](configs/paper/calibration/e3_qwen_1_5b_multi_hop_tracing.yaml) | Qwen2.5-1.5B | multi_hop_tracing · calibration · easy | 4K | n=96; offset=0; target=q/k_early/k_middle/k_late/v_early/v_middle/v_late/qv/o/mlp_early/mlp_middle/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_1_5b_multi_needle`](configs/paper/calibration/e3_qwen_1_5b_multi_needle.yaml) | Qwen2.5-1.5B | multi_needle · calibration · medium | 4K | n=96; offset=0; target=q/k_early/k_middle/k_late/v_early/v_middle/v_late/qv/o/mlp_early/mlp_middle/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_1_5b_needle_absent`](configs/paper/calibration/e3_qwen_1_5b_needle_absent.yaml) | Qwen2.5-1.5B | needle_absent · calibration · hard | 4K | n=96; offset=0; target=q/k_early/k_middle/k_late/v_early/v_middle/v_late/qv/o/mlp_early/mlp_middle/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_1_5b_variable_tracking`](configs/paper/calibration/e3_qwen_1_5b_variable_tracking.yaml) | Qwen2.5-1.5B | variable_tracking · calibration · easy | 4K | n=96; offset=0; target=q/k_early/k_middle/k_late/v_early/v_middle/v_late/qv/o/mlp_early/mlp_middle/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_32b_aggregation`](configs/paper/calibration/e3_qwen_32b_aggregation.yaml) | Qwen2.5-32B | aggregation · calibration · hard | 16K | n=32; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_32b_multi_hop_tracing`](configs/paper/calibration/e3_qwen_32b_multi_hop_tracing.yaml) | Qwen2.5-32B | multi_hop_tracing · calibration · hard | 16K | n=32; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_32b_multi_needle`](configs/paper/calibration/e3_qwen_32b_multi_needle.yaml) | Qwen2.5-32B | multi_needle · calibration · hard | 16K | n=32; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_32b_variable_tracking`](configs/paper/calibration/e3_qwen_32b_variable_tracking.yaml) | Qwen2.5-32B | variable_tracking · calibration · hard | 16K | n=32; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_32b_common_words`](configs/paper/calibration/e3_qwen_32b_common_words.yaml) | Qwen2.5-32B | common_words · calibration · hard | 16K | n=32; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_32b_needle_absent`](configs/paper/calibration/e3_qwen_32b_needle_absent.yaml) | Qwen2.5-32B | needle_absent · calibration · hard | 16K | n=32; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_7b_aggregation`](configs/paper/calibration/e3_qwen_7b_aggregation.yaml) | Qwen2.5-7B | aggregation · calibration · medium | 8K | n=64; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_7b_multi_hop_tracing`](configs/paper/calibration/e3_qwen_7b_multi_hop_tracing.yaml) | Qwen2.5-7B | multi_hop_tracing · calibration · medium | 8K | n=64; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_7b_multi_needle`](configs/paper/calibration/e3_qwen_7b_multi_needle.yaml) | Qwen2.5-7B | multi_needle · calibration · hard | 8K | n=64; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_7b_variable_tracking`](configs/paper/calibration/e3_qwen_7b_variable_tracking.yaml) | Qwen2.5-7B | variable_tracking · calibration · easy | 8K | n=64; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_7b_common_words`](configs/paper/calibration/e3_qwen_7b_common_words.yaml) | Qwen2.5-7B | common_words · calibration · hard | 8K | n=64; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |
| [`calibration_e3_qwen_7b_needle_absent`](configs/paper/calibration/e3_qwen_7b_needle_absent.yaml) | Qwen2.5-7B | needle_absent · calibration · hard | 8K | n=64; offset=0; target=q/k/v/qv/mlp_early/mlp_late/norm/output_head; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/frozen/periodic/threshold/delta/suffix/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ |  |

</details>

<details>
<summary><strong>E4-V · Planner 验证集调参/冻结</strong> — 4 个配置 / 12 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`validation_e4_mistral_7b_longbench_v2_validation`](configs/paper/validation/e4_mistral_7b_longbench_v2_validation.yaml) | Mistral-7B-v0.3 | LongBench-v2 · validation | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`validation_e4_qwen_14b_longbench_v2_validation`](configs/paper/validation/e4_qwen_14b_longbench_v2_validation.yaml) | Qwen2.5-14B | LongBench-v2 · validation | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`validation_e4_qwen_32b_longbench_v2_validation`](configs/paper/validation/e4_qwen_32b_longbench_v2_validation.yaml) | Qwen2.5-32B | LongBench-v2 · validation | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`validation_e4_qwen_7b_longbench_v2_validation`](configs/paper/validation/e4_qwen_7b_longbench_v2_validation.yaml) | Qwen2.5-7B | LongBench-v2 · validation | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |

</details>

<details>
<summary><strong>E4-T · Planner 最终测试</strong> — 11 个配置 / 33 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`test_e4_mistral_7b_longbench_v2_test`](configs/paper/test/e4_mistral_7b_longbench_v2_test.yaml) | Mistral-7B-v0.3 | LongBench-v2 · test | 16K | n=96; offset=64; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_14b_longbench_v2_test`](configs/paper/test/e4_qwen_14b_longbench_v2_test.yaml) | Qwen2.5-14B | LongBench-v2 · test | 16K | n=96; offset=64; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_32b_longbench_v2_test`](configs/paper/test/e4_qwen_32b_longbench_v2_test.yaml) | Qwen2.5-32B | LongBench-v2 · test | 16K | n=96; offset=64; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_7b_2wikimqa`](configs/paper/test/e4_qwen_7b_2wikimqa.yaml) | Qwen2.5-7B | LongBench/2wikimqa · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_7b_gov_report`](configs/paper/test/e4_qwen_7b_gov_report.yaml) | Qwen2.5-7B | LongBench/gov_report · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_7b_hotpotqa`](configs/paper/test/e4_qwen_7b_hotpotqa.yaml) | Qwen2.5-7B | LongBench/hotpotqa · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_7b_longbench_v2_test`](configs/paper/test/e4_qwen_7b_longbench_v2_test.yaml) | Qwen2.5-7B | LongBench-v2 · test | 16K | n=256; offset=96; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_7b_passage_count`](configs/paper/test/e4_qwen_7b_passage_count.yaml) | Qwen2.5-7B | LongBench/passage_count · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_7b_passage_retrieval_en`](configs/paper/test/e4_qwen_7b_passage_retrieval_en.yaml) | Qwen2.5-7B | LongBench/passage_retrieval_en · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_coder_7b_lcc`](configs/paper/test/e4_qwen_coder_7b_lcc.yaml) | Qwen2.5-Coder-7B | LongBench/lcc · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`test_e4_qwen_coder_7b_repobench_p`](configs/paper/test/e4_qwen_coder_7b_repobench_p.yaml) | Qwen2.5-Coder-7B | LongBench/repobench-p · test | 16K | n=96; offset=0; target=q/k/v/qv/mlp_late; norm=1e-3; r=8; v=0/1/2/4/8; cache=no-adapt/full/stale/periodic/threshold/delta/suffix/planner/oracle; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |

</details>

<details>
<summary><strong>E5 · Delta correction 安全域</strong> — 12 个配置 / 36 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`delta_e5_qwen_32b_r16_n1e3`](configs/paper/delta/e5_qwen_32b_r16_n1e3.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=24; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-3; r=16; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_32b_r16_n1e4`](configs/paper/delta/e5_qwen_32b_r16_n1e4.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=24; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-4; r=16; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_32b_r4_n1e3`](configs/paper/delta/e5_qwen_32b_r4_n1e3.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=24; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-3; r=4; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_32b_r4_n1e4`](configs/paper/delta/e5_qwen_32b_r4_n1e4.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=24; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-4; r=4; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_32b_r8_n1e3`](configs/paper/delta/e5_qwen_32b_r8_n1e3.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=24; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_32b_r8_n1e4`](configs/paper/delta/e5_qwen_32b_r8_n1e4.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=24; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-4; r=8; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_7b_r16_n1e3`](configs/paper/delta/e5_qwen_7b_r16_n1e3.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-3; r=16; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_7b_r16_n1e4`](configs/paper/delta/e5_qwen_7b_r16_n1e4.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-4; r=16; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_7b_r4_n1e3`](configs/paper/delta/e5_qwen_7b_r4_n1e3.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-3; r=4; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_7b_r4_n1e4`](configs/paper/delta/e5_qwen_7b_r4_n1e4.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-4; r=4; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_7b_r8_n1e3`](configs/paper/delta/e5_qwen_7b_r8_n1e3.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`delta_e5_qwen_7b_r8_n1e4`](configs/paper/delta/e5_qwen_7b_r8_n1e4.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=160; target=k_early/k_middle/k_late/v_early/v_middle/v_late/qv; norm=1e-4; r=8; v=0/1/2/4/8/16; cache=full/stale/base+delta/delta/periodic/planner; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |

</details>

<details>
<summary><strong>E6 · 上下文与模型扩展</strong> — 11 个配置 / 33 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`scaling_e6_qwen_14b_16384`](configs/paper/scaling/e6_qwen_14b_16384.yaml) | Qwen2.5-14B | multi_hop_tracing · test · hard | 16K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_14b_32768`](configs/paper/scaling/e6_qwen_14b_32768.yaml) | Qwen2.5-14B | multi_hop_tracing · test · hard | 32K | n=16; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_14b_8192`](configs/paper/scaling/e6_qwen_14b_8192.yaml) | Qwen2.5-14B | multi_hop_tracing · test · hard | 8K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_32b_16384`](configs/paper/scaling/e6_qwen_32b_16384.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 16K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_32b_32768`](configs/paper/scaling/e6_qwen_32b_32768.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 32K | n=16; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_32b_8192`](configs/paper/scaling/e6_qwen_32b_8192.yaml) | Qwen2.5-32B | multi_hop_tracing · test · hard | 8K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_7b_16384`](configs/paper/scaling/e6_qwen_7b_16384.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 16K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_7b_32768`](configs/paper/scaling/e6_qwen_7b_32768.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 32K | n=16; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_7b_4096`](configs/paper/scaling/e6_qwen_7b_4096.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 4K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_7b_65536`](configs/paper/scaling/e6_qwen_7b_65536.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 64K | n=16; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`scaling_e6_qwen_7b_8192`](configs/paper/scaling/e6_qwen_7b_8192.yaml) | Qwen2.5-7B | multi_hop_tracing · test · medium | 8K | n=32; offset=128; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8; cache=full/stale/periodic/threshold/planner; repeat=3w+10t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |

</details>

<details>
<summary><strong>E7 · Planner 消融与失效边界</strong> — 1 个配置 / 3 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`ablation_e7_qwen_7b_longbench_v2`](configs/paper/ablation/e7_qwen_7b_longbench_v2.yaml) | Qwen2.5-7B | LongBench-v2 · test | 16K | n=96; offset=352; target=q/k/v/qv/mlp_late/norm; norm=1e-3; r=8; v=0/1/2/4/8/16; cache=no-adapt/full/stale/planner/-version/-target/-norm/-delta/-partial/-periodic; repeat=0w+1t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |

</details>

<details>
<summary><strong>E8 · 容量压力与尾延迟</strong> — 2 个配置 / 6 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`workload_e8_qwen_32b`](configs/paper/workload/e8_qwen_32b.yaml) | Qwen2.5-32B | variable_tracking · test · hard | 16K | n=64; offset=256; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8/16/32; cache=full/stale/periodic/threshold/delta/planner; repeat=1w+3t; parallel=model_shard | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |
| [`workload_e8_qwen_7b`](configs/paper/workload/e8_qwen_7b.yaml) | Qwen2.5-7B | variable_tracking · test · medium | 16K | n=128; offset=256; target=q/k/v/qv; norm=1e-3; r=8; v=0/1/2/4/8/16/32; cache=full/stale/periodic/threshold/delta/planner; repeat=1w+3t; parallel=single | ⬜ | ⬜ | ⬜ | 依赖 E3 failure map |

</details>

<details>
<summary><strong>A1 · 跨架构迁移筛查</strong> — 12 个配置 / 36 个 seed 运行</summary>

| 配置 | 模型 | 任务 / 分区 | 上下文 | 内部条件 | Seed 7 | Seed 17 | Seed 29 | 备注 |
|---|---|---|---:|---|:---:|:---:|:---:|---|
| [`architecture_a1_gemma_3_4b_multi_hop_tracing`](configs/paper/architecture/a1_gemma_3_4b_multi_hop_tracing.yaml) | Gemma-3-4B | multi_hop_tracing · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_gemma_3_4b_multi_needle`](configs/paper/architecture/a1_gemma_3_4b_multi_needle.yaml) | Gemma-3-4B | multi_needle · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_gemma_3_4b_variable_tracking`](configs/paper/architecture/a1_gemma_3_4b_variable_tracking.yaml) | Gemma-3-4B | variable_tracking · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_llama_3_2_3b_multi_hop_tracing`](configs/paper/architecture/a1_llama_3_2_3b_multi_hop_tracing.yaml) | Llama-3.2-3B | multi_hop_tracing · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_llama_3_2_3b_multi_needle`](configs/paper/architecture/a1_llama_3_2_3b_multi_needle.yaml) | Llama-3.2-3B | multi_needle · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_llama_3_2_3b_variable_tracking`](configs/paper/architecture/a1_llama_3_2_3b_variable_tracking.yaml) | Llama-3.2-3B | variable_tracking · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_mistral_7b_v0_1_multi_hop_tracing`](configs/paper/architecture/a1_mistral_7b_v0_1_multi_hop_tracing.yaml) | Mistral-7B-v0.1 | multi_hop_tracing · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_mistral_7b_v0_1_multi_needle`](configs/paper/architecture/a1_mistral_7b_v0_1_multi_needle.yaml) | Mistral-7B-v0.1 | multi_needle · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_mistral_7b_v0_1_variable_tracking`](configs/paper/architecture/a1_mistral_7b_v0_1_variable_tracking.yaml) | Mistral-7B-v0.1 | variable_tracking · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_early/k_middle/v_middle/mlp_middle/norm_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_qwen1_5_moe_a2_7b_multi_hop_tracing`](configs/paper/architecture/a1_qwen1_5_moe_a2_7b_multi_hop_tracing.yaml) | Qwen1.5-MoE-A2.7B | multi_hop_tracing · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_middle/v_middle/moe.router_middle/moe_shared_expert_middle/moe.routed_experts_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_qwen1_5_moe_a2_7b_multi_needle`](configs/paper/architecture/a1_qwen1_5_moe_a2_7b_multi_needle.yaml) | Qwen1.5-MoE-A2.7B | multi_needle · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_middle/v_middle/moe.router_middle/moe_shared_expert_middle/moe.routed_experts_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |
| [`architecture_a1_qwen1_5_moe_a2_7b_variable_tracking`](configs/paper/architecture/a1_qwen1_5_moe_a2_7b_variable_tracking.yaml) | Qwen1.5-MoE-A2.7B | variable_tracking · calibration · easy | 4K | n=8; offset=0; target=q_middle/k_middle/v_middle/moe.router_middle/moe_shared_expert_middle/moe.routed_experts_middle; norm=1e-3; r=8; v=1/4/16; cache=full/stale/windowed_recompute_4/suffix; repeat=0w+1t; parallel=single | ⚠ | ⚠ | ⚠ | ⚠ 当前配置 n=8，低于冻结协议要求的 n≥48，且缺 task_viability；正式运行前必须修复 |

</details>

### 已完成但不计入正式矩阵的验证

| 验证项 | 状态 | 不勾正式配置的原因 |
|---|:---:|---|
| Qwen2.5-7B Ascend 真实 smoke | ✅ | 仅验证实现与硬件路径 |
| Qwen2.5-7B 双 NPU 可行性 | ✅ | 仅可行性，不是冻结论文条件 |
| Qwen2.5-7B 四 NPU 8K QV / late-MLP stress | ✅ | 仅部分 stress 条件，不等于完整 E2/E6 seed 矩阵 |
| BF16 delta quantization investigation | ✅ | 已完成探索性负结果/诊断结果；正式 E5 网格仍单独计算 |

更新本节时只能修改有产物证据的状态。seed 若部分完成，使用 `◐`，并精确写出剩余 target、strategy、version gap、sample 或失败条件。


## Repository layout

```text
src/ttt_cache_lab/
  cache/         cache validity semantics, strategies, and planner
  data/          synthetic, JSONL, and Hugging Face task loaders
  experiments/  single-step runner, versioned runner, sweep, summaries, reports
  metrics/      cache/logit/task metrics
  models/       toy and Hugging Face backends
  updates/      update targets and executable random/LoRA updaters
configs/
  feasibility_*.yaml          early single-step configs
  sweep_*.yaml                single-step sweep configs
  versioned_sweep_*.yaml      E5/E6 versioned sweep configs
  experiments/                lightweight E1-E7 templates
  paper/                      frozen E1-E8, multi-model, multi-seed paper matrix
docs/
  project_plan.md             full research and experiment roadmap
  paper_experiment_protocol.md frozen tasks, partitions, statistics, and model roles
  runbook.md                  general run instructions
scripts/
  run_toy_study.sh            run all E1-E7 toy templates
  run_model_sharded.sh        one-model multi-GPU/NPU layer-sharding launcher
tests/
  unit tests for configs, runners, metrics, planner, reports, and backends
```

## Install

For local development and toy experiments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For Hugging Face experiments:

```bash
pip install -e '.[dev,hf]'
```

## Validate the repository

```bash
ruff check src tests
mypy src tests
pytest
```

The base CI runs the lightweight suite. A separate offline integration job constructs a tiny Llama locally and executes real HF cache paths without downloading model weights.

## Toy experiments

Toy experiments are useful for validating the pipeline without model downloads.

Run one versioned experiment:

```bash
python -m ttt_cache_lab.cli versioned-run \
  --config configs/experiments/e2_version_drift_toy.yaml \
  --version-summary
```

Generate the report:

```bash
python -m ttt_cache_lab.cli version-report \
  --input runs/e2_version_drift/summary.csv \
  --output-dir runs/e2_version_drift/report
```

Generate E3 and then run the E4 planner against that exact artifact:

```bash
scripts/run_calibrated_planner.sh \
  configs/experiments/e3_failure_map_toy.yaml \
  configs/experiments/e4_planner_main_toy.yaml
```

The script verifies that `cache.failure_map_path` points to the E3 artifact before starting E4.

Run all E1-E7 toy templates:

```bash
scripts/run_toy_study.sh
```

## Single-step feasibility commands

The original single-step runner is still available for quick checks.

```bash
python -m ttt_cache_lab.cli run --config configs/feasibility_toy.yaml
python -m ttt_cache_lab.cli summarize --input runs/feasibility-toy/summary.csv
python -m ttt_cache_lab.cli first-table --input runs/feasibility-toy/summary.csv
```

Sweep examples:

```bash
python -m ttt_cache_lab.cli sweep --config configs/sweep_toy_update_norm.yaml
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/versioned_sweep_e5_delta_qwen_0_5b.yaml
python -m ttt_cache_lab.cli versioned-sweep \
  --config configs/versioned_sweep_e6_context_qwen_1_5b.yaml
```

## Main experiment groups

The full plan is in [`docs/project_plan.md`](docs/project_plan.md). The runnable templates currently cover:

| Group | Purpose | Example config |
|---|---|---|
| E1 | Static adapters and aLoRA/LRAgent/ForkKV-style baselines | `configs/experiments/e1_static_adapter_baseline_qwen_0_5b.yaml` |
| E2 | Adapter-version drift and task-ability retention | `configs/experiments/e2_version_drift_qwen_1_5b.yaml`, `e2_task_ability_qwen_1_5b.yaml` |
| E2 cross-family | Dense, sliding-window, local/global, and MoE architecture generality | `configs/experiments/e2_version_drift_llama_3_2_3b.yaml`, `e2_version_drift_mistral_7b_v0_1.yaml`, `e2_version_drift_gemma_3_4b.yaml`, and `e2_version_drift_qwen1_5_moe_a2_7b.yaml` |
| E3 | Update-target × version-gap failure map | `configs/experiments/e3_failure_map_qwen_0_5b.yaml` |
| E4 | Versioned planner main experiment | `configs/experiments/e4_planner_main_qwen_0_5b.yaml` |
| E5 | Delta correction and rank/update-norm sweep | `configs/versioned_sweep_e5_delta_qwen_0_5b.yaml` |
| E6 | Exact 4K-32K context and model-scale scaling | `configs/versioned_sweep_e6_context_qwen_1_5b.yaml` |
| E7 | Planner-component ablations and failure boundaries | `configs/paper/ablation/e7_qwen_7b_longbench_v2.yaml` |
| E8 | Sustained cache-capacity and tail-latency workload | `configs/paper/workload/e8_qwen_32b.yaml` |
| A1 | Lightweight cross-architecture screening on three controlled tasks | `configs/paper/architecture/a1_*_{multi_hop_tracing,multi_needle,variable_tracking}.yaml` |

## Task viability preflight

Every non-toy E1-E8 configuration and the W1/W2/W4 discovery configurations enable a baseline task probe before the expensive experiment starts. The probe reuses the already-loaded model and writes artifacts to `runs/<experiment>/task_probe/`. Quality-facing E1/E2/E4/E6/E7/E8 runs fail fast at either a floor or ceiling. Diagnostic E3/E5/W runs always reject floor effects but may allow a perfect baseline because their primary endpoints are KL, propagation, and repair fidelity. The A1 and W3 gaps are explicitly flagged in the progress table and must be fixed before those runs are accepted as paper evidence.

Synthetic tasks use task-appropriate scorers rather than universal exact match. Passkey and key-value difficulty now changes the retrieval structure, including one-, two-, and three-hop version routing. The checked-in E1-E8 matrix enforces sample-count floors: real benchmark results use at least 96 samples, controlled quality experiments generally use 32-96, and 32K/64K cost studies use at least 16. The current A1 screening configs remain below their frozen `n >= 48` requirement and are therefore marked `⚠` rather than runnable paper conditions.

A standalone calibration can be run with:

```bash
python -m ttt_cache_lab.cli task-probe \
  --config configs/experiments/e2_version_drift_qwen_1_5b.yaml \
  --output-dir runs/task_probe/e2_qwen_1_5b \
  --min-mean-score 0.05 --max-mean-score 0.95 \
  --min-nonzero-fraction 0.10 --max-perfect-fraction 0.90
```

E2 has two explicit scales: `e2_version_drift_*` retains small updates for sensitivity/consistency measurements, while `e2_task_ability_*` and the paper drift configs use `update_norm: 1e-3` on held-out samples to create measurable fresh-model task changes. E2 analysis reports task change versus the pre-update model, the fraction of conditions below the pre-update baseline, and retained fresh-cache adaptation gain.

## Output files

Most experiment runs write:

```text
runs/<experiment>/records.jsonl          raw records
runs/<experiment>/summary.csv            flat CSV records
runs/<experiment>/run_metadata.json       config hash, git state, platform, packages
runs/<experiment>/run_failure.json        structured exception manifest, only on failure
runs/<experiment>/version_summary.csv    condition-preserving grouped means
runs/<experiment>/report/report.md       Markdown report
runs/<experiment>/report/*.svg           metric-vs-version plots
runs/e3_failure_map/failure_map/*        E3 policy table and heatmap
runs/e4_planner_main/pareto/*            E4 quality-cost Pareto table
```

Important columns include:

```text
experiment_id
update_target
cache_strategy
action
adapter_version
cached_version
version_gap
accumulated_update_norm
accumulated_raw_update_norm
update_scale
baseline_task_score
full_task_score
adaptation_gain_vs_base
task_score
logits_kl
top1_agreement
relative_error
hidden_relative_error
update_norm_since_cache
cache_bytes
physical_cache_bytes
peak_memory_allocated
adaptation_latency
cache_maintenance_latency
decode_latency
end_to_end_latency
throughput_tokens_per_s
cache_entry_count
recompute_fraction
cache_hit
refresh_count
false_safe
strategy_mode
strategy_available
strategy_fallback
planner_source
baseline_fidelity
baseline_source
run_config_sha256
```

## Backends

| Backend | Config value | Purpose |
|---|---|---|
| Toy | `toy` | CI, dry runs, pipeline validation |
| HuggingFace | `hf` | CPU/CUDA experiments; `model.parallelism: model_shard` partitions decoder layers across visible GPUs |

## Example config fragment

```yaml
model:
  backend: hf
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  revision: <immutable-model-commit>
  device: cuda:0
  torch_dtype: bfloat16
  attention_implementation: eager  # required when attention metrics are enabled
  max_length: 4096
  trust_remote_code: true

adapter:
  update_mode: lora_train
  norm_control: target_l2  # use none to preserve raw learning-rate updates
  lora_rank: 8
  lora_alpha: 16.0
  learning_rate: 0.0002
  train_steps_per_version: 1
  freeze_base_model: true

cache:
  manager_scope: condition  # sample or global_workload must be explicit

resume: true
checkpoint_each_target: true
version_steps: [0, 1, 2, 4, 8]
```

## Cross-run scaling analysis

Merge completed model/context runs before generating one E6 report:

```bash
python -m ttt_cache_lab.cli merge-records \
  --input runs/e6_qwen_1_5b/records.jsonl runs/e6_qwen_7b/records.jsonl \
  --output-dir runs/e6_merged
python -m ttt_cache_lab.cli version-report \
  --input runs/e6_merged/summary.csv \
  --output-dir runs/e6_merged/report
```

`resume: true` merges existing checkpoint records by a stable condition identity. It preserves completed records but still replays model updates needed to reconstruct in-memory adapter/cache state.

## Documentation

- [`docs/project_plan.md`](docs/project_plan.md): complete research plan, experiment dependency graph, deliverables, and decision gates.
- [`docs/runbook.md`](docs/runbook.md): detailed command reference.
- [`docs/design.md`](docs/design.md): cache semantics and strategy design notes.
- [`docs/experiment_plan.md`](docs/experiment_plan.md): earlier feasibility experiment plan.

## Paper-scale study

The frozen protocol is in [`docs/paper_experiment_protocol.md`](docs/paper_experiment_protocol.md). The checked-in matrix contains 84 configurations and expands to 252 jobs over seeds 7, 17, and 29. Every Qwen calibration scale covers all six controlled task families. The A1 gate adds Mistral-7B-v0.1, Llama-3.2-3B, Gemma-3-4B, and Qwen1.5-MoE-A2.7B without placing them in any default launcher. Qwen2.5-7B remains the complete main evaluation; 14B and 32B are explicit primary scaling evidence.

Generate the stable job matrix:

```bash
python -m ttt_cache_lab.cli study-plan --manifest configs/paper/study.yaml
```

Run calibration, finalize its immutable failure map, then run held-out stages:

```bash
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag calibration
scripts/finalize_paper_calibration.sh
python -m ttt_cache_lab.cli study-run --manifest configs/paper/study.yaml --tag test
scripts/finalize_paper_stage.sh test
```

Use `scripts/run_paper_shard.sh` to distribute the manifest over accelerator groups. Dependent stages fail early when the calibrated failure-map artifact is absent.

## License

MIT.
