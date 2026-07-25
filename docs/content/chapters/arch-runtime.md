---
title: The ECS-native runtime
order: 6
---

Prism's ECS runtime queries canonical components and capability
resources, chooses an admitted execution plan, manages residency and
KV-cache state, then routes work to CPU, GPU, or NPU systems when the
target path is validated. Handoffs and terminal outcomes become
evidence rather than hidden backend side effects.

## Flow

- Canonical ECS state
- ECS router + scheduler
- CPU, MI300X / ROCm-HIP, XDNA / XDNA2 spatial plan
- Residency + KV cache
- Execution receipt + validation scope

## Runtime contract

- **Input** — sealed CImage + request + capabilities
- **Work** — query ECS state, route, make resident, dispatch
- **Output** — tokens, metrics, KV updates, receipt, failure class

Canonical state owns accepted work, model identity, ownership,
leases, committed state, and terminal outcomes. Caches, queues,
backend handles, and projections remain derived or ephemeral.
XDNA/XDNA2 is a spatial-planning and legalization boundary unless a
matching hardware/runtime path has been validated; a compiled plan
is not itself a claim of device execution.
