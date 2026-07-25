---
title: Ownership
order: 4
---

Clear boundaries prevent fake portability.

## General Compute owns

- SambaNova topology
- Accelerator execution
- Memory and queues
- Provider transport
- Kernel behavior

## Prism Engine owns

- Workload semantics
- Capability matching
- Phase placement
- KV ownership
- Receipts and recovery

## Design constraint

A provider may reject a plan, but it cannot silently redefine the
workload or publish an unaccounted outcome.
