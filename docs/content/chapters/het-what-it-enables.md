---
title: What this enables
order: 5
---

A single model family can have multiple deployment identities: a
mixed-precision or ternarized CImage, a Metal or MI300X GPU image,
an XDNA/XDNA2 spatial plan, or a heterogeneous composition with
explicit KV-cache ownership. The logical model remains comparable
while the physical artifact reflects the machine.

- **Representation** — FP16/BF16, INT8/NF4, progressive ternarization
- **GPU** — Metal or ROCm/HIP, MI300X validation
- **NPU** — XDNA/XDNA2, spatial legality
- **State** — KV search/compression, ownership + receipts

## What is implemented today?

Apple Silicon and MI300X validation paths are active. ECS-native
compilation, mixed-precision and ternary search, CImage assembly,
and heterogeneous capability modeling are implemented across the
repository. XDNA/XDNA2 planning is compile-verified and
resource/legalization tested; general Ryzen AI hardware execution
remains dependent on access to a matching XDNA-capable system and
runtime stack.
