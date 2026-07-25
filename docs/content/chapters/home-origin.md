---
title: Observe Origin
order: 2
---

A model enters as bytes. Through the constitutional compiler it
becomes a `ComputeImage` — a sealed artifact whose identity is the
content hash of its source, weights, and binding ABI. The artifact
is inspectable; the compiler is the witness.

## The path

1. **Ingest**. The source model is read into a typed artifact
   entity. The artifact is content-addressed; the bytes determine
   the identity.
2. **Compile**. The compiler produces a `ComputeImage`. Every
   stage is a typed system; the output is a typed value.
3. **Seal**. The artifact is sealed. From this point, the
   identity is fixed; no silent substitution is possible.
4. **Admit**. The sealed artifact is admitted into the catalog.
   The admission is itself a typed event in the event store.

## What this page proves

That the architecture is content-addressed from end to end. A
replay from durable events reconstructs the same canonical world;
a runtime reconciliation projects the same DOM; the SSG emits
the same HTML.
