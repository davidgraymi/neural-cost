---
title: "Hugging Face & roofline plots"
description: "Profile open-weight models from their config and export SVG roofline charts."
section: "Models"
order: 3
---

Profile any open-weight model directly by its Hugging Face repository ID (or local `config.json` path) with **zero external dependencies** and zero PyTorch/Transformers overhead:

```python
from neural_cost import HardwareSpec, from_huggingface, save_roofline_svg, RooflinePoint

# 1. Ingest model architecture directly from Hugging Face Hub (cached locally)
model_cfg = from_huggingface("meta-llama/Meta-Llama-3-8B")
print(f"Architecture: {model_cfg.architecture}, Dim: {model_cfg.embed_dim}, Heads: {model_cfg.num_heads}/{model_cfg.num_kv_heads}")

# 2. Render publication-quality vector SVG roofline charts with workload overlays
hardware = HardwareSpec(name="NVIDIA A100-SXM4-80GB", peak_flops=312e12, memory_bandwidth=2039e9)
point = RooflinePoint(
    arithmetic_intensity=14.2,
    flops=14.2 * hardware.memory_bandwidth,
    label="LLaMA-3-8B Decode (b=1)",
    bottleneck="memory",
    latency_ms=1.1,
)

# Export standalone, responsive vector SVG
save_roofline_svg("llama3_roofline.svg", hardware, points=point, theme="dark")
```
