---
title: The ECS-native compiler
order: 3
---

Compiler state is represented as ECS entities and components so
systems can analyze the graph, search progressive quantization and
ternarization candidates, assign mixed precision, plan layouts and
KV policy, and emit target-aware execution views without hiding
policy in a monolithic runtime.

## Pipeline

1. **Ingest** — Model identity, metadata, weights, graph, quality contract.
2. **Analyze** — Tensor classes, sensitivity, candidate representations, KV pressure.
3. **Search** — Quantization, ternarization, mixed precision, cache compression.
4. **Plan** — Layouts, views, residency, CPU/GPU/NPU placement and handoffs.
5. **Seal** — CImage plus provenance, receipts, and explicit validation gates.

## Compiler researcher detail

```text
Model → ECS entities/components → TensorAnalysis → RepresentationSearch → KVPolicy →
Admission → PhysicalLayout → ExecutionViews → PrismIR → HeterogeneousPlan → CImage
```
