---
title: Serving phases
order: 2
---

One serving request. Several physical phases. A frontier-model
serving path does not have to assign prefill, KV state, decode, and
streaming to the same provider. Prism can represent the phase
requirements and ask the capability model for the lowest-latency
legal composition.

- **Phase 01** — Prefill. Process prompt, produce KV state.
- **Boundary** — KV handoff. Explicit transfer, ownership + cost.
- **Phase 02** — Decode. Token generation, state residency.
- **Phase 03** — Stream. Observable output, terminal receipt.

## Researcher detail

The repository's heterogeneous-serving fixture models provider
capabilities for prefill, KV handoff, decode, and token streaming.
The serving requirement remains fixed while changing capability
descriptions changes the physical plan. The SambaNova path is a
planning and integration surface, not a claim of general production
support.
