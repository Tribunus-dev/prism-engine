---
title: One tensor, three questions
order: 2
---

Prism separates concerns that model formats and runtimes often blend
together.

## LogicalTensor

What does this tensor mean? Identity, class, shape, logical operation,
orientation, data type, and graph boundary: the semantic contract.

## PhysicalTileLayout

How are its bits stored? Codec, tile family, group axis, metadata,
padding, alignment, and interleave: the storage and kernel ABI.

## ExecutionView

How does a lane consume it? Lane, offsets, metadata ranges, repacking,
and residency. Multiple views can derive from one physical layout.
