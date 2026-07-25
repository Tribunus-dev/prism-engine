---
title: Same graph, different machines
order: 2
---

The serving requirement can stay constant while the physical plan
changes. Prefill may run on one provider, KV state may cross an
explicit boundary, and decode or streaming may run on another
provider. The planner chooses a legal composition from capability
descriptions rather than baking one vendor into the workload.

## Phases

- **Workload** — Serve frontier model (latency, memory, quality policy)
- **Phase 01** — Prefill (provider capability + admission)
- **Handoff** — KV transfer (explicit memory boundary)
- **Phase 02** — Decode + stream (provider capability + receipt)

## Researcher detail

The heterogeneous-serving fixture expresses a fixed serving
requirement and provider phase capabilities for prefill, KV
handoff, decode, and token streaming. Replacing the capability
descriptions changes the plan without changing the serving
requirement or orchestration contract.
