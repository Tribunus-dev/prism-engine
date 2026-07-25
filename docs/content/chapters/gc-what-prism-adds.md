---
title: What Prism adds
order: 3
---

The accelerator stays itself. Prism does not flatten MI300X
ROCm/HIP, XDNA/XDNA2, or other providers into a generic GPU-shaped
abstraction. It preserves provider-specific capabilities while
making representation, spatial planning, KV ownership, routing, and
validation implications visible to the ECS-native planner.

- **Capabilities** — What phases, dtypes, layouts, spatial resources, and workloads the provider can execute.
- **Placement** — Which phase or tensor region should run on CPU, GPU, or NPU under latency, memory, and compatibility constraints.
- **Handoffs** — Where activations, KV state, or compressed cache pages cross a physical memory or provider boundary.
- **Evidence** — What was admitted, planned, executed, measured, and actually validated for replay.
