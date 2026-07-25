---
title: The short version
order: 1
---

Prism is an ECS-native compiler and runtime for local, heterogeneous
AI deployment. A model artifact becomes a searchable representation
plan, a target-aware spatial/execution plan, and a sealed
ComputeImage; the runtime routes work across CPU, GPU, and NPU
capabilities while preserving receipts and validation evidence.

The path:

- **Model** — GGUF or SafeTensors
- **ECS Compiler** — quantize, ternarize, lower, search
- **ComputeImage** — weights, views, plans, evidence
- **Runtime** — CPU, Metal/ROCm GPU, XDNA NPU

## If you remember one thing

Prism searches representations and placement together, then makes
the chosen deployment contract inspectable.
