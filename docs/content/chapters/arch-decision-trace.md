---
title: One decision, made visible
order: 4
---

The ECS-native compiler treats representation and placement as
searchable data. A tensor can remain FP16/BF16, move through
INT8/NF4, or be progressively ternarized when calibration and
admission gates permit; the resulting plan can route different
regions to CPU, ROCm/HIP GPU, Metal, or XDNA/XDNA2 spatial resources.

## The trace

- **Model** — logical meaning (shape, role, graph edge)
- **Representation** — mixed precision (quantize, ternarize, calibrate)
- **Spatial plan** — route and resources (CPU, GPU, NPU, KV budget)
- **Proof** — sealed contract (CImage, receipt, validation boundary)

## Invariant

The logical graph remains attributable. Formats, execution views,
routing, and KV-cache policy are selected only when the evidence and
target legality support them.
