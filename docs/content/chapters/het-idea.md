---
title: One semantic workload, many physical homes
order: 1
---

Prism separates the model's logical execution requirements from the
hardware that will satisfy them. The ECS-native compiler preserves
workload semantics while selecting mixed-precision representations,
memory tiers, execution lanes, KV-cache policy, transfers, and
validation gates for a particular target.

## Scale line

- **CPU** — portable fallback, reference + hardening
- **GPU** — Metal or MI300X, ROCm / HIP execution
- **NPU** — XDNA / XDNA2, spatial plan + legality
- **Handoff** — KV ownership, residency + evidence
