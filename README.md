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

以下状态基于 **2026 年 7 月 14 日实时核对**：

- 仓库：`main`
- 正式冻结矩阵：**84 个配置 × 3 seeds = 252 个正式 seed-run**
- 严格完成配置：**1/84**
- 已正式验收的单 seed：**4/252**
  - E1：seed 7、17、29；该配置三 seed 已全部验收
  - E3：Qwen2.5-1.5B aggregation 的 seed 7
- 当前正在运行：
  - E2 Qwen2.5-7B controlled seed 7
  - E3 1.5B aggregation seed 17
- 当前策略：**只推进 E1、E2、E3；A、W、B 暂停**

状态含义：

- `[完成]`：产物齐全并通过论文数据验收
- `[运行]`：任务正在运行
- `[部分]`：有部分结果，但尚不能作为论文数据
- `[待做]`：尚未开始或没有有效正式结果
- `[阻塞]`：依赖上游产物
- `[配置错误]`：当前配置不满足冻结协议，不能直接运行
- `[暂停]`：按当前资源调度主动暂停

---

# 一、论文实验整体依赖树

```text id="63bocr"
KV Cache Consistency and Reuse under Inference-Time Parameter Evolution
│
├── 0. 共同前置条件与验收协议
│   │
│   ├── 0.1 Task viability probe
│   │   ├── 在正式实验前验证模型确实具备任务能力
│   │   ├── 防止把模型不会做任务造成的 floor effect 误判为 cache 失效
│   │   ├── 质量型实验默认要求：
│   │   │   ├── 平均任务分数 ∈ [0.05, 0.95]
│   │   │   ├── 非零分数样本比例 ≥ 10%
│   │   │   └── 满分样本比例 ≤ 90%
│   │   └── 状态：[代码完成]；正式运行时必须逐配置保留 probe 记录
│   │
│   ├── 0.2 固定数据划分
│   │   ├── LongBench-v2 validation：offset 0，n=96
│   │   ├── LongBench-v2 main test：
│   │   │   ├── Qwen2.5-7B：offset 96，n=256
│   │   │   └── 其他模型：offset 96，n=96
│   │   ├── E7 ablation test：offset 352，n=96
│   │   └── selection seed 固定为 2027
│   │
│   ├── 0.3 固定随机种子
│   │   ├── seed 7
│   │   ├── seed 17
│   │   └── seed 29
│   │
│   └── 0.4 单个 seed 的验收条件
│       ├── 覆盖配置要求的所有：
│       │   ├── sample
│       │   ├── update target
│       │   ├── cache strategy
│       │   └── adapter version
│       ├── 必须存在：
│       │   ├── .success
│       │   ├── run_metadata.json
│       │   ├── records.jsonl
│       │   ├── summary.csv
│       │   └── 实验要求的专用汇总文件
│       ├── 不得存在未解决的 run_failure.json
│       └── 三个 seed 全部通过后，一个配置才计为完成
│
├── 1. 基础现象与主问题成立性
│   │
│   ├── E1：静态适配器缓存基线
│   └── E2：参数版本漂移
│
├── 2. 失效机制与可修复边界发现
│   │
│   ├── W1：有限重算窗口
│   ├── W2：逐层误差传播
│   ├── W3：失效边界预测
│   ├── W4/B1：层—token 块级 oracle
│   └── E3：正式 failure map 校准
│
├── 3. Planner 构建与冻结
│   │
│   ├── B2：静态块排序器
│   ├── B3：单次 probe router
│   ├── B4：直接提交/重算门控
│   ├── B5：动态预算控制器
│   ├── B6：Planner 正式证据配置
│   └── E4-V：验证集调参与最终冻结
│
├── 4. Planner 最终有效性
│   └── E4-T：独立测试集 quality-cost frontier
│
├── 5. 替代修复方法与扩展实验
│   │
│   ├── E5：Delta correction 安全域
│   ├── E6：上下文长度与模型规模扩展
│   ├── E7：Planner 消融与失效边界
│   └── E8：容量压力与尾延迟
│
├── 6. 外部有效性
│   └── A1：跨架构迁移筛查
│
└── 7. 统计、图表与论文证据闭环
    ├── 三 seed 合并
    ├── paired comparison
    ├── 95% cluster-bootstrap CI
    ├── false-safe Wilson CI
    ├── p50/p90/p95/p99 latency
    ├── failure map
    ├── quality-cost Pareto frontier
    └── 最终表格、图和复现实验包
```

---

# 二、正式冻结实验矩阵

## 1. E1：静态适配器缓存基线

**研究问题：已有面向静态 adapter 的缓存复用方法，能够解决多少参数变化下的缓存问题？**

```text id="fbze05"
E1 静态适配器基线
│
├── 正式规模：1 个配置 / 3 个 seed-run
│
├── 模型
│   └── Qwen2.5-7B
│
├── 数据
│   └── LongBench-v2 validation
│       ├── context：16K
│       ├── n=96
│       └── offset=0
│
├── 参数更新条件
│   ├── target：q / k / v / qv
│   ├── update norm：1e-3
│   ├── LoRA rank：8
│   ├── adapter version：0
│   └── adapter sequence：0/1/2/3/0/2/1/3
│
├── 对比方法
│   ├── full recomputation
│   ├── base reuse
│   ├── per-adapter cache
│   ├── aLoRA
│   ├── LRAgent
│   ├── ForkKV
│   └── base+delta
│
├── 论文作用
│   ├── 建立静态 adapter 方法基线
│   ├── 区分静态 adapter 切换与持续参数演化
│   └── 证明后续 versioned-cache 问题并未被已有方法完全覆盖
│
└── 当前状态
    ├── seed 7：[完成]
    ├── seed 17：[完成]
    ├── seed 29：[完成]
    │   ├── .success 存在
    │   ├── run_metadata.json、records.jsonl、summary.csv、version_summary.csv 均完整且非空
    │   └── 无 run_failure.json 或 .failed
    └── 配置完成度：1/1
        单-seed 完成度：3/3
```

这是当前首个完成全部三 seed 并通过正式验收的冻结配置。

---

## 2. E2：参数版本漂移

**研究问题：随着 adapter 版本不断演化，旧 KV cache 的误差和任务能力损失如何增长？**

```text id="tvk1sv"
E2 参数版本漂移
│
├── 正式规模：6 个配置 / 18 个 seed-run
│
├── 2.1 Qwen2.5-7B
│   ├── controlled
│   │   ├── variable_tracking · test · easy
│   │   ├── context：16K
│   │   ├── n=64
│   │   └── versions：0/1/2/4/8/16/32
│   └── realistic
│       ├── LongBench-v2 · test
│       ├── context：16K
│       ├── n=96
│       └── versions：0/1/2/4/8/16
│
├── 2.2 Qwen2.5-14B
│   ├── controlled
│   │   ├── common_words · test · hard
│   │   └── versions：0/1/2/4/8/16/32
│   └── realistic
│       └── LongBench-v2 · test
│
├── 2.3 Qwen2.5-32B
│   ├── controlled
│   │   ├── common_words · test · hard
│   │   └── versions：0/1/2/4/8/16/32
│   └── realistic
│       └── LongBench-v2 · test
│
├── controlled target
│   ├── q
│   ├── k
│   ├── v
│   ├── qv
│   ├── mlp_early
│   ├── mlp_late
│   ├── norm
│   └── output_head
│
├── cache 策略
│   ├── full
│   ├── stale
│   └── frozen
│
├── 必须报告
│   ├── fresh-cache adaptation gain
│   ├── stale-cache task drop
│   ├── KL / distribution drift
│   ├── gain-retention ratio
│   ├── 首次越过失效阈值的版本
│   └── target × version-gap 漂移曲线
│
└── 当前状态
    ├── Qwen2.5-7B
    │   ├── controlled seed 7：[运行]
    │   └── realistic：[已排队]
    ├── Qwen2.5-14B：[待重跑]
    │   └── 旧 E2/E3 队列中存在失败记录，不计论文数据
    ├── Qwen2.5-32B：[暂停]
    └── 正式完成度：0/6 配置，0/18 seed
```

E2 不依赖 Planner，也不依赖 failure map，是当前应直接推进的主线。

---

## 3. E3：失效图校准

**研究问题：在哪些模型、任务、更新位置和版本差距下，可以安全复用、近似修复或必须重算？**

```text id="8qsu43"
E3 Failure Map
│
├── 正式规模：24 个配置 / 72 个 seed-run
│
├── 模型轴：4 个模型
│   │
│   ├── Qwen2.5-1.5B
│   │   ├── context：4K
│   │   ├── n=96
│   │   └── 用于最全面的 target/layer 扫描
│   │
│   ├── Qwen2.5-7B
│   │   ├── context：8K
│   │   ├── n=64
│   │   └── 论文主要完整评估规模
│   │
│   ├── Qwen2.5-14B
│   │   ├── context：16K
│   │   ├── n=48
│   │   └── 中等规模迁移验证
│   │
│   └── Qwen2.5-32B
│       ├── context：16K
│       ├── n=32
│       └── 大模型 headline evidence
│
├── 每个模型包含 6 个 controlled task
│   ├── multi_needle
│   ├── needle_absent
│   ├── multi_hop_tracing
│   ├── aggregation
│   ├── common_words
│   └── variable_tracking
│
├── 共计
│   └── 4 模型 × 6 任务 = 24 配置
│
├── update target
│   ├── 1.5B 完整 target 扫描
│   │   ├── q
│   │   ├── k_early / k_middle / k_late
│   │   ├── v_early / v_middle / v_late
│   │   ├── qv
│   │   ├── o
│   │   ├── mlp_early / mlp_middle / mlp_late
│   │   ├── norm
│   │   └── output_head
│   └── 7B/14B/32B 核心 target
│       ├── q / k / v / qv
│       ├── mlp_early / mlp_late
│       ├── norm
│       └── output_head
│
├── version gap
│   └── 0 / 1 / 2 / 4 / 8 / 16
│
├── cache 策略
│   ├── full
│   ├── stale
│   ├── frozen
│   ├── periodic
│   ├── threshold
│   ├── delta
│   ├── suffix
│   └── planner
│
├── 核心产物
│   └── failure_map.csv
│       ├── model
│       ├── context
│       ├── task
│       ├── target
│       ├── layer position
│       ├── version gap
│       ├── update norm / rank
│       └── 各策略是否安全
│
├── 依赖关系
│   ├── failure map 是 E4-E8 唯一允许使用的校准依据
│   ├── E4 测试开始后不能再根据测试结果修改 failure map
│   └── 一个策略必须在所有兼容 calibration cell 上安全，才能被 Planner 使用
│
└── 当前状态
    ├── 1.5B aggregation
    │   ├── seed 7：[完成]
    │   ├── seed 17：[运行]
    │   └── seed 29：[待做]
    ├── 其他 1.5B 任务：[待做]
    ├── 7B 全部 6 任务：[待做]
    ├── 14B 全部 6 任务：[待重跑/待做]
    ├── 32B 全部 6 任务：[暂停]
    └── 正式完成度
        ├── 完整配置：0/24
        └── 已验收单 seed：1/72
```

仓库 README 的 E3 seed 状态格存在一次错位：表格把 seed 17 标成了完成，但服务器实际产物显示：

- seed 7 完成；
- seed 17 正在运行；
- seed 29 尚不存在。

上面的树以服务器实际产物为准。

---

# 三、机制发现与 Planner 探索线

这些实验目前不计入冻结的 252 个正式 seed-run，但决定 Planner 是否有足够强的机制依据。

## 4. W1：有限重算窗口

**研究问题：参数发生变化后，是否只重算有限层数就可以恢复一致性，而不必一直重算到模型末尾？**

```text id="gmclpi"
W1 有限重算窗口
│
├── 条件
│   ├── target：q/k/v/mlp
│   ├── position：early/middle/late
│   ├── window：1/2/4/8/16/32
│   ├── gap：1/4/16
│   └── full/stale/suffix 成对比较
│
├── 必须产出
│   ├── smallest safe window
│   ├── smallest quality-improving window
│   ├── minimal_safe_windows.csv
│   └── 随窗口增大时的 KL 非单调异常
│
├── W1-1.5B
│   ├── seed 7/17/29 均运行成功
│   ├── 缺少顶层 run_metadata.json
│   ├── 缺少 minimal_safe_windows.csv
│   └── 状态：[部分][暂停]
│
└── W1-7B
    └── 状态：[待做][暂停]
```

当前有原始数据，但还没有形成可直接用于论文的完整分析产物。

---

## 5. W2：逐层误差传播

**研究问题：旧缓存产生的扰动沿后续网络层传播时，是衰减、持续还是放大？**

```text id="fsh4of"
W2 逐层传播
│
├── 观测量
│   ├── hidden-state drift
│   ├── K drift
│   ├── V drift
│   └── 32 probe token 上的逐层曲线
│
├── target/position 条件：8 组
├── gap：1/4/16
│
├── W2-1.5B
│   ├── seed 7/17/29 均运行成功
│   ├── metadata 和原始传播记录存在
│   ├── 缺少 propagation_profiles.csv
│   └── 状态：[部分][暂停]
│
└── W2-7B
    └── 状态：[待做][暂停]
```

它决定有限窗口重算是否具有理论和经验依据。

---

## 6. W3：失效边界预测

**研究问题：能否通过低成本局部信号预测某个缓存块是否需要修复？**

```text id="eaottm"
W3 Boundary Predictor
│
├── 信号
│   ├── local-boundary signal
│   └── whole stale-suffix signal
│
├── 评估
│   ├── sample-held-out split
│   ├── false-safe
│   ├── false-positive
│   └── predictor calibration
│
├── W3-1.5B
│   ├── seed 7/17/29 均运行成功
│   ├── metadata 和原始边界记录存在
│   ├── 缺少 boundary_predictor_summary.csv
│   └── 状态：[部分][暂停]
│
└── W3-7B
    └── 状态：[待做][暂停]
```

---

## 7. W4/B1：层—token 块级 Oracle

**研究问题：在相同重算预算下，选择正确的层—token 缓存块，理论上最多可以改善多少？**

```text id="u4tkxb"
W4/B1 Blockwise Oracle
│
├── 模型：Qwen2.5-1.5B
├── task：multi-hop
├── context：4K
├── n=16
│
├── target
│   ├── k
│   ├── q
│   ├── mlp
│   └── v-middle
│
├── block size
│   ├── 32
│   ├── 64
│   └── 128 tokens
│
├── budget
│   ├── 1/14 cache cells
│   └── 2/14 cache cells
│
├── selector
│   ├── random
│   ├── raw drift
│   ├── attention weighted
│   ├── layer prefix
│   ├── greedy
│   └── per-token oracle
│
├── 已有产物
│   ├── block_frontier.csv
│   ├── block_masks.csv
│   └── blockwise_report.md
│
└── 当前状态
    ├── seed 7：[完成探索]
    ├── seed 17：[部分][暂停]
    ├── seed 29：[待做]
    └── 整体：[部分][暂停]
```

这里获得的是 **oracle 上界**。在真正实现稀疏重算或 copy-on-write 数据路径之前，不能把 oracle 的收益直接报告为可部署加速。

---

# 四、Planner 开发线

## 8. B2：静态块排序器

```text id="a5rn4l"
B2 Zero-probe Static Ranker
│
├── 输入：W4 calibration artifacts
├── 方法：不进行额外前向 probe，直接预测高风险块
├── 需要验证
│   ├── held-out KL improvement
│   ├── 误伤率
│   ├── 选中 cache cells
│   ├── confidence/safety gate
│   └── planner latency
└── 状态：[代码完成][正式证据待做][暂停]
```

## 9. B3：单次 Probe Router

```text id="51w5ms"
B3 One-probe Router
│
├── probe length：1/2/4
├── reference policy
│   ├── direct reference
│   └── baseline-reference
├── 需要验证
│   ├── probe 成本
│   ├── 总延迟
│   ├── held-out quality
│   └── 相比 zero-probe 的额外收益
└── 状态：[代码完成][正式证据待做][暂停]
```

## 10. B4：直接提交/重算门控

```text id="oua3ze"
B4 Committed Router
│
├── zero-probe direct commit/recompute gate
├── trust-band calibration
├── 独立 guard split
├── 需要验证
│   ├── 不使用真实 KL 的运行时决策
│   ├── false-safe rate
│   └── false-safe Wilson confidence upper bound
└── 状态：[代码完成][正式证据待做][暂停]
```

## 11. B5：动态预算控制器

```text id="h7eacv"
B5 Dynamic Controller
│
├── risk-scaled budget
├── activation threshold
├── max selected cells
├── marginal-gain stopping
├── 需要验证
│   ├── budget-quality frontier
│   ├── latency-quality frontier
│   ├── 动态预算是否优于固定预算
│   └── 全局失效时是否自动扩大重算范围
└── 状态：[代码完成][正式证据待做][暂停]
```

## 12. B6：Planner 正式证据

```text id="9tg7p4"
B6 Planner Formal Evidence
│
├── 前置条件
│   ├── 从 B2-B5 中确定最终可部署策略
│   ├── 冻结全部阈值和预算规则
│   └── 写入正式 study.yaml
│
├── 正式对比
│   ├── full
│   ├── stale
│   ├── periodic
│   ├── threshold
│   ├── delta
│   ├── suffix
│   ├── planner
│   └── oracle
│
├── 完整计时
│   ├── probe latency
│   ├── selection latency
│   ├── repair latency
│   ├── decode latency
│   └── end-to-end latency
│
└── 状态：[待做][依赖 B2-B5]
```

---

# 五、E4：Planner 主实验

## 13. E4-V：验证集调参与冻结

**用途：只在 validation 上选择 Planner、periodic、threshold 等方法的超参数，随后完全冻结。**

```text id="3s8ix2"
E4-V Planner Validation
│
├── 正式规模：4 配置 / 12 seed-run
│
├── 模型
│   ├── Qwen2.5-7B
│   ├── Qwen2.5-14B
│   ├── Qwen2.5-32B
│   └── Mistral-7B-v0.3
│
├── 数据
│   └── LongBench-v2 validation
│       ├── context：16K
│       ├── n=96
│       └── offset=0
│
├── 对比策略
│   ├── no-adapt
│   ├── full
│   ├── stale
│   ├── periodic
│   ├── threshold
│   ├── delta
│   ├── suffix
│   ├── planner
│   └── oracle
│
├── 输出
│   ├── frozen planner hyperparameters
│   ├── frozen periodic interval
│   ├── frozen threshold baseline
│   └── 固定 failure-map hash
│
└── 当前状态
    ├── 0/4 配置
    ├── 0/12 seed
    └── [阻塞：等待 E3 failure map 和 Planner 路线冻结]
```

---

## 14. E4-T：最终独立测试

**研究问题：Planner 是否在未用于调参的任务和样本上，优于完整重算以外的低成本基线？**

```text id="pnowno"
E4-T Planner Final Test
│
├── 正式规模：11 配置 / 33 seed-run
│
├── 14.1 LongBench-v2 跨规模/跨家族
│   ├── Qwen2.5-7B
│   │   └── n=256，论文主要结果
│   ├── Qwen2.5-14B
│   ├── Qwen2.5-32B
│   └── Mistral-7B-v0.3
│
├── 14.2 Qwen2.5-7B 开放式 LongBench
│   ├── 2wikimqa
│   ├── hotpotqa
│   ├── gov_report
│   ├── passage_count
│   └── passage_retrieval_en
│
├── 14.3 Qwen2.5-Coder-7B 代码任务
│   ├── lcc
│   └── repobench-p
│
├── 所有任务
│   ├── context：16K
│   ├── target：q/k/v/qv/mlp_late
│   ├── versions：0/1/2/4/8
│   └── 九种 cache/adaptation 策略
│
├── 主要论文结论
│   ├── Planner 是否改善 held-out quality-cost frontier
│   ├── 是否优于 tuned periodic
│   ├── 是否优于 update-norm threshold
│   ├── 是否接近 oracle
│   ├── false-safe 是否足够低
│   └── 收益是否能迁移到其他模型和任务
│
└── 当前状态
    ├── 0/11 配置
    ├── 0/33 seed
    └── [阻塞：E3 → B2-B6 → E4-V]
```

---

# 六、替代修复方法与扩展实验

## 15. E5：Delta Correction 安全域

**研究问题：低秩 K/V delta correction 是否存在一个稳定、非平凡的安全区域？**

```text id="byte9b"
E5 Delta Correction
│
├── 正式规模：12 配置 / 36 seed-run
│
├── 模型轴
│   ├── Qwen2.5-7B
│   └── Qwen2.5-32B
│
├── LoRA rank 轴
│   ├── r=4
│   ├── r=8
│   └── r=16
│
├── update norm 轴
│   ├── 1e-4
│   └── 1e-3
│
├── 组合
│   └── 2 模型 × 3 rank × 2 norm = 12 配置
│
├── task
│   ├── multi_hop_tracing · test
│   └── context：16K
│
├── target
│   ├── k_early / k_middle / k_late
│   ├── v_early / v_middle / v_late
│   └── qv
│
├── 对比
│   ├── full
│   ├── stale
│   ├── base+delta
│   ├── delta
│   ├── periodic
│   └── planner
│
├── 输出
│   ├── correction error
│   ├── task quality
│   ├── safe-region map
│   └── 无安全域时的明确负面结论
│
└── 当前状态
    ├── 0/12 配置
    ├── 0/36 seed
    └── [阻塞：等待 E3 failure map]
```

即使最终证明 delta correction 基本无效，清楚划定其失效边界也可以作为论文证据，不能事后删除。

---

## 16. E6：上下文长度与模型规模扩展

**研究问题：上下文越长、模型越大时，选择性缓存维护的成本优势是否增强？**

```text id="yerwas"
E6 Scaling
│
├── 正式规模：11 配置 / 33 seed-run
│
├── 16.1 Qwen2.5-7B
│   ├── 4K
│   ├── 8K
│   ├── 16K
│   ├── 32K
│   └── 64K
│
├── 16.2 Qwen2.5-14B
│   ├── 8K
│   ├── 16K
│   └── 32K
│
├── 16.3 Qwen2.5-32B
│   ├── 8K
│   ├── 16K
│   └── 32K
│
├── 合计
│   └── 5 + 3 + 3 = 11 配置
│
├── task
│   └── multi_hop_tracing · test
│
├── target
│   └── q/k/v/qv
│
├── 对比
│   ├── full
│   ├── stale
│   ├── periodic
│   ├── threshold
│   └── planner
│
├── 专用性能协议
│   ├── 3 次 warm-up
│   ├── 10 次正式计时
│   ├── p50 主延迟
│   ├── p95 尾延迟
│   ├── adaptation latency
│   ├── cache-maintenance latency
│   └── decode latency
│
└── 当前状态
    ├── 14B 8K seed 7 曾产生部分产物，但不验收
    ├── 7B 32K/64K：[暂停]
    ├── 14B 32K：[暂停]
    ├── 32B 全部：[暂停]
    └── 正式完成度：0/11 配置，0/33 seed
```

论文最低要求之一是：**16K 以上上下文必须显示出比短上下文更明显的成本优势**。

---

## 17. E7：Planner 消融与失效边界

**研究问题：Planner 的哪些输入和组件真正有贡献？在哪些条件下必然失效？**

```text id="oxqxf9"
E7 Ablation
│
├── 正式规模：1 配置 / 3 seed-run
├── 模型：Qwen2.5-7B
├── 数据：LongBench-v2 ablation test
│   ├── context：16K
│   ├── offset=352
│   └── n=96
│
├── target
│   ├── q/k/v/qv
│   ├── mlp_late
│   └── norm
│
├── versions：0/1/2/4/8/16
│
├── 消融项
│   ├── 完整 planner
│   ├── -version feature
│   ├── -target feature
│   ├── -update-norm feature
│   ├── -delta option
│   ├── -partial-recompute option
│   └── -periodic baseline information
│
├── 输出
│   ├── 各组件质量贡献
│   ├── 各组件延迟贡献
│   ├── false-safe 变化
│   └── update norm × version gap 失效边界
│
└── 当前状态
    ├── 0/1 配置
    ├── 0/3 seed
    └── [阻塞：最终 Planner 冻结后才能运行]
```

---

## 18. E8：容量压力与尾延迟

**研究问题：当缓存容量持续不足、请求不断到达时，Planner 的收益是否会被 eviction、维护开销和尾延迟抵消？**

```text id="zijasc"
E8 Capacity-limited Workload
│
├── 正式规模：2 配置 / 6 seed-run
│
├── Qwen2.5-7B
│   ├── variable_tracking · test · medium
│   ├── context：16K
│   └── n=128
│
├── Qwen2.5-32B
│   ├── variable_tracking · test · hard
│   ├── context：16K
│   └── n=64
│
├── versions：0/1/2/4/8/16/32
│
├── 策略
│   ├── full
│   ├── stale
│   ├── periodic
│   ├── threshold
│   ├── delta
│   └── planner
│
├── 系统指标
│   ├── global cache manager 行为
│   ├── LRU eviction
│   ├── cache hit/miss
│   ├── p50/p95/p99 latency
│   ├── throughput
│   ├── false-safe rate
│   └── 容量压力下的 quality degradation
│
└── 当前状态
    ├── 0/2 配置
    ├── 0/6 seed
    └── [阻塞：等待 Planner 和 failure map 冻结]
```

E8 不能只报告平均延迟，必须公开尾延迟和容量不足时的失败。

---

# 七、A1：跨架构迁移筛查

**研究问题：Qwen 上发现的 failure boundary 是否能够迁移到不同 decoder 结构？**

```text id="a7x2cs"
A1 Architecture Transfer
│
├── 正式规模：12 配置 / 36 seed-run
│
├── 架构 1：Llama-3.2-3B
│   └── 独立 dense GQA 架构
│
├── 架构 2：Mistral-7B-v0.1
│   └── sliding-window attention
│
├── 架构 3：Gemma-3-4B
│   └── alternating local/global attention
│
├── 架构 4：Qwen1.5-MoE-A2.7B
│   ├── sparse MoE
│   ├── router target
│   ├── shared expert target
│   └── routed experts target
│
├── 每个架构 3 个任务
│   ├── multi_hop_tracing
│   ├── multi_needle
│   └── variable_tracking
│
├── 合计
│   └── 4 架构 × 3 任务 = 12 配置
│
├── 当前配置问题
│   ├── 每项只有 n=8
│   ├── 冻结协议要求 n≥48
│   └── 缺少 task_viability probe
│
└── 当前状态
    ├── 12/12 配置：[配置错误]
    ├── 36/36 seed 暂不能启动
    ├── Llama seed 7 曾中断并留下部分产物
    ├── 这些产物不能作为论文数据
    └── 整体：[暂停]
```

A1 必须先统一修正为至少 48 个样本并补充 task probe，再重新正式运行。

---

# 八、最终统计和论文产物

```text id="vb7rri"
论文证据闭环
│
├── 每个配置三 seed 合并
│   ├── seed 7
│   ├── seed 17
│   └── seed 29
│
├── 配对维度必须完全一致
│   ├── dataset sample
│   ├── seed
│   ├── update target
│   ├── adapter version
│   ├── LoRA rank
│   ├── update norm
│   └── context
│
├── 统计
│   ├── mean / median / std
│   ├── 5,000 次 cluster bootstrap
│   ├── 95% confidence interval
│   ├── paired task-score difference
│   ├── paired latency difference
│   ├── paired speedup
│   └── false-safe Wilson confidence interval
│
├── 延迟
│   ├── p50
│   ├── p90
│   ├── p95
│   └── p99
│
├── 核心图
│   ├── E1：静态 adapter 方法对比
│   ├── E2：version drift curves
│   ├── E3：target × gap failure heatmap
│   ├── E4：quality-cost Pareto frontier
│   ├── E5：delta safe-region map
│   ├── E6：context/model scaling curves
│   ├── E7：component ablation
│   ├── E8：capacity/latency trace
│   └── A1：cross-architecture transfer table
│
└── 最低论文成立条件
    ├── adaptation 至少在部分任务上优于 no-adapt
    ├── drift 随 target/layer/gap/norm 呈现系统差异
    ├── Planner 优于 tuned periodic 和 norm-threshold
    ├── false-safe 上界足够低
    ├── 收益在 7B 成立，并在 14B 或 32B 得到确认
    ├── 长上下文成本优势明显增强
    ├── E5 找到安全域，或可靠证明不存在
    └── E8 不掩盖容量不足导致的尾延迟和失败
```

---

# 九、当前完成情况总树

```text id="2o4eei"
全部论文实验
│
├── 正式矩阵：84 configs / 252 seeds
│   │
│   ├── E1：1 config / 3 seeds
│   │   ├── 完成：3
│   │   ├── 运行：0
│   │   └── 完整配置：1/1
│   │
│   ├── E2：6 configs / 18 seeds
│   │   ├── 完成：0
│   │   ├── 7B：controlled seed 7 运行中，realistic 已排队
│   │   ├── 14B：旧失败待重跑
│   │   └── 32B：暂停
│   │
│   ├── E3：24 configs / 72 seeds
│   │   ├── 正式完成 seed：1
│   │   ├── 运行：1
│   │   ├── 完整配置：0/24
│   │   └── 其余：待做/待重跑/暂停
│   │
│   ├── E4-V：4 configs / 12 seeds
│   │   └── 阻塞于 E3 + Planner
│   │
│   ├── E4-T：11 configs / 33 seeds
│   │   └── 阻塞于 E4-V
│   │
│   ├── E5：12 configs / 36 seeds
│   │   └── 阻塞于 E3
│   │
│   ├── E6：11 configs / 33 seeds
│   │   └── 0 正式完成，少量旧部分产物
│   │
│   ├── E7：1 config / 3 seeds
│   │   └── 阻塞于 Planner 冻结
│   │
│   ├── E8：2 configs / 6 seeds
│   │   └── 阻塞于 Planner 冻结
│   │
│   └── A1：12 configs / 36 seeds
│       └── 全部配置需要修复，当前暂停
│
└── W/B 探索线
    ├── W1-1.5B：原始运行 3/3，缺分析产物
    ├── W2-1.5B：原始运行 3/3，缺分析产物
    ├── W3-1.5B：原始运行 3/3，缺分析产物
    ├── W4/B1：seed 7 完成，seed 17 部分，seed 29 未做
    ├── B2：代码完成，正式评估未做
    ├── B3：代码完成，正式评估未做
    ├── B4：代码完成，正式评估未做
    ├── B5：代码完成，正式评估未做
    └── B6：未冻结正式配置
```

# 十、当前关键路径

```text id="no9ydj"
当前
│
├── E1 三个 seed 已完成
│   └── 已得到首个完整三-seed 正式配置
│
├── E2 7B controlled + LongBench-v2
│   └── 形成版本漂移主结果
│
├── E3 1.5B 六任务
│   ├── 完成 aggregation seed 17/29
│   ├── 完成其余五个任务
│   └── 生成第一版完整 failure map
│
├── 判断 W1-W4 机制是否足够强
│   ├── 补齐 W1/W2/W3 汇总文件
│   ├── 完成 W4 剩余 seeds
│   └── 决定 B2-B5 最终路线
│
├── 扩展 E3 到 7B
│
├── 扩展 E3 到 14B/32B
│
├── 冻结 Planner
│   └── E4-V
│
├── 运行最终测试
│   ├── E4-T
│   ├── E5
│   ├── E6
│   ├── E7
│   └── E8
│
└── 修复并运行 A1
    └── 完成跨架构外部有效性证据
```

当前最准确的概括是：**实验设计和代码框架已经相当完整，论文正式数据已完成首个三-seed 冻结配置。严格口径下已有 1/84 个配置完成；单 seed 层面已正式验收 4 个，另有 2 个正在运行。**


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
