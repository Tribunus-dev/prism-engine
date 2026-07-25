---
title: The validation loop
order: 2
---

Bonsai models made the architecture concrete. The Bonsai Ternary and
binary model work gave Prism Engine a demanding upstream
representation to compile against. That relationship helps validate
the architecture's central boundary: numerical representation can
evolve upstream while deployment layout, execution views,
scheduling, and evidence remain explicit downstream.

1. **Bonsai model** — ternary or binary representation
2. **Prism ingestion** — identity + graph + tensor classes
3. **Deployment admission** — quality, target, resource gates
4. **Execution artifact** — CImage + views + receipts

## Architectural result

The upstream model can change its numerical representation without
forcing the runtime to become the owner of every deployment
decision.
