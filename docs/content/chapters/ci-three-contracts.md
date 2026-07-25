---
title: Three contracts
order: 3
---

Meaning, storage, and consumption stay separate.

- **LogicalTensor** — What the tensor means in the model graph.
- **PhysicalTileLayout** — How its data and metadata are represented in memory.
- **ExecutionView** — How a specific CPU, GPU, or NPU lane consumes it.

## Invariant

Changing a physical view must not erase logical identity or fabricate
evidence.
