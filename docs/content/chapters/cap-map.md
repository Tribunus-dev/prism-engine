---
title: The capability map
order: 2
---

Filter by domain or status. Every entry carries source paths and an
explicit limitation. The categories are:

- **Runtime** — CPU, Metal, ROCm/HIP, ANE, NPU
- **Compiler** — representation search, ternarization, KV policy
- **Artifact** — ComputeImage, ABI, payloads, receipts
- **Authority** — constitutional transactions, projection rebuild
- **Evidence** — replay, comparison, regression, recovery
- **Models** — supported architectures, ingest paths

Each entry's status is one of `Implemented`, `Compile-verified`,
`Measured`, `Verified`, or `Planned`. A claim without source paths
fails the constitutional rule and is rejected at build time.
