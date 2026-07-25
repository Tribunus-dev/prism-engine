---
title: The relationship
order: 1
---

Complementary by construction. Bonsai and Prism Engine do not need
to compete for the same layer. Prism ML can explore training,
calibration, progressive quantization, ternarization, mixed
precision, and model-quality tradeoffs. Prism's ECS-native
compiler/runtime accepts the resulting artifact and solves the
deployment questions that appear when those weights meet
heterogeneous CPU, GPU, and NPU capabilities.

- **Prism ML / Bonsai** — Training + representation (calibration, ternary, mixed precision, quality)
- **Prism Engine** — Deployment + execution (admission, CImage, KV, routing, receipts)
