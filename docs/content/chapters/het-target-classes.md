---
title: Target classes
order: 3
---

Prism does not assume that a "GPU backend" is one universal target.
A target profile constrains memory, bandwidth, scratch, resident
views, context, sharding, KV ownership, and preferred execution
lanes.

- **CPU** — Portable reference. Correctness oracle, fallback execution, and Linux hardening.
- **GPU** — Apple + MI300X. Metal on Apple Silicon; ROCm/HIP and gfx942-oriented validation on MI300X.
- **NPU** — XDNA / XDNA2. Tile, FIFO, DMA, barrier, and resource legalization for spatial plans.
- **Heterogeneous** — Cross-device plan. Route phases, activations, and KV state with explicit handoffs and evidence.
