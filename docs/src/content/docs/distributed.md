---
title: "Distributed training"
description: "3D parallelism, interconnect rooflines and per-device memory sharding."
section: "Models"
order: 2
---

When scaling large models across multi-node clusters, training throughput is governed by the interaction between compute rooflines, network interconnect fabrics, and 3D parallelism strategy.

`neural-cost` models **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, and **Data Parallelism / ZeRO / FSDP** communication volumes, Hockney (α-β) network transfer times, 1F1B schedule bubble fractions, and per-device memory sharding.

- With `activation_checkpointing=True`, the recomputed forward re-issues its TP collectives (`tp_recompute_bytes_per_step`), and step FLOPs are 8×params×tokens instead of 6×.
- `pp_bytes_per_step` is the per-rank volume an interior stage carries; `pp_total_bytes_per_step` sums all `pp_boundaries` stage boundaries across TP ranks and DP replicas. With sequence parallelism each TP rank sends only its 1/TP shard of the boundary tensor. `pp_virtual_stages` models Megatron's interleaved 1F1B schedule, which divides the bubble by *v* and multiplies per-rank PP traffic by *v*.

```python
from neural_cost import (
    ClusterTopology,
    HardwareSpec,
    analyze_distributed_gap,
    estimate_parallelism,
    get_interconnect_preset,
)

# 128x H100 cluster across 16 nodes (8 GPUs per node)
h100 = HardwareSpec(
    "H100", peak_flops=989e12, memory_bandwidth=3.35e12, memory_capacity=80 * 1024**3
)
topo = ClusterTopology(
    device=h100,
    num_nodes=16,
    devices_per_node=8,
    intra_node=get_interconnect_preset("nvlink4"),  # 900 GB/s
    inter_node=get_interconnect_preset("infiniband_ndr"),  # 50 GB/s
)

# Train Llama-3 70B with TP=8, PP=4, DP=4 (ZeRO-3 / FSDP)
cost = estimate_parallelism(
    total_parameters=70_000_000_000,
    batch_size=128,
    seq_len=4096,
    embed_dim=8192,
    num_layers=80,
    tp_degree=8,
    pp_degree=4,
    dp_degree=4,
    num_microbatches=8,
    dp_mode="zero3_fsdp",
)

gap = analyze_distributed_gap(cost, topo, overlap_efficiency=0.85)
print(gap.render())
# Reports MFU, step time, communication overhead, pipeline bubble fraction, and memory fit
```
