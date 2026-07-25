---
title: A deployment artifact, not a renamed checkpoint
order: 5
---

A ComputeImage is the boundary between compilation and execution. The
runtime loads it without reconstructing deployment policy from the
original model.

The artifact holds:

- **Metadata** — identity, version, manifest
- **Logical tensors** — semantic tensor table
- **Physical layouts** — codec and tile contracts
- **Execution views** — lane-specific materializations
- **Plan + receipts** — decisions and validation
- **Payloads** — mapped tensor bytes

## ABI detail

The CImage layout ABI makes logical identity, physical storage, and
execution views independently inspectable. A target profile can
select resident or mutually exclusive views without changing the
logical graph.
