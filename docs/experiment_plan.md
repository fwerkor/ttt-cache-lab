# Experiment plan

## Phase 1: feasibility

The first phase answers whether the direction is worth continuing.

### E1. Minimal qTTT-style loop

- Model: toy backend first, then a small HuggingFace causal LM.
- Task: synthetic passkey / key-value retrieval.
- Compare no adaptation, Q-only update, and full recompute.

### E2. Cache invalidation heatmap

Update targets:

- Q, K, V, O projection;
- late MLP;
- LoRA-Q, LoRA-K, LoRA-V, LoRA-MLP-late.

Strategies:

- full recompute;
- stale reuse;
- frozen reuse;
- layer-wise recompute.

Metrics:

- KV relative error;
- hidden-state relative error;
- logits KL divergence;
- top-1 agreement;
- synthetic task exact match.

### E3. Update-space usefulness

Run all update targets with full recompute to determine whether broader updates provide stronger adaptation than Q-only.

### E4. Low-cost maintenance

For update targets that look useful, test whether layer-wise recomputation, periodic refresh, or delta correction can approximate full recompute.

## Phase 2: paper-scale experiments

Research questions:

1. Which parameter updates invalidate which cache blocks?
2. Is there a useful update space beyond Q-only qTTT?
3. Does the planner improve the accuracy-latency Pareto frontier?
4. Does the benefit grow with context length?
5. When should cache reuse be rejected?

## Intended plots

- update target × layer-position heatmap;
- accuracy-latency Pareto curve;
- context-length scaling curve;
- cache error vs task score;
- planner ablation table;
- failure-boundary table.
