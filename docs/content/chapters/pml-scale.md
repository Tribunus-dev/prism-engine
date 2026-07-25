---
title: Deployment scale
order: 3
---

Prism Engine is the deployment scale. The same Bonsai model family
can become different physical artifacts depending on where it will
run. The logical model and quality contract remain attributable to
Prism ML; the CImage reflects representation admission, KV policy,
hardware capabilities, routing, and execution evidence for the
target.

- **CPU** — Reference image. Differential checks, conservative fallback, and bounded memory.
- **GPU** — MI300X ROCm/HIP. AMD memory, kernels, queues, and active validation evidence.
- **NPU** — XDNA / XDNA2. Spatial planning, tile legality, and explicit hardware boundary.
- **Heterogeneous** — Distributed image. CPU/GPU/NPU routing, KV ownership, transfers, and replay.
