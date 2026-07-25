---
title: What the evidence supports today
order: 8
---

Apple Silicon and MI300X ROCm/HIP validation paths are active. CPU
execution and ECS-native compilation continue under hardening and
migration gates. XDNA/XDNA2 spatial planning is compile-verified and
resource/legalization tested, but general Ryzen AI hardware
execution is not claimed without an XDNA-capable system and the
matching userspace stack. Progressive quantization, ternarization,
mixed precision, KV-cache search/compression, and CImages are real
compiler/runtime concerns; a valid artifact or plan is not by
itself proof of production readiness on every target.
