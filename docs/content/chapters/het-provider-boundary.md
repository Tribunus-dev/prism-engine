---
title: Provider boundary
order: 4
---

Prism does not pretend to be the accelerator.

## Prism owns

- semantic workload
- physical plan
- KV ownership
- admission policy
- execution receipts

## Provider owns

- kernels
- queues
- memory handles
- fabric transport
- device topology

## Design constraint

Backend handles may execute work. They cannot silently become the
authority for placement, lifecycle, or accepted outcome.
