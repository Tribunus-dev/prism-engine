---
title: Staged workflow
order: 2
---

One path. Four gates.

1. **Ingest** — GGUF or SafeTensors, identity + digest
2. **Compile** — PrismIR, representation search
3. **Realize** — ComputeImage, .cimage artifact
4. **Prove** — Metal run, receipt + replay fields

## Release principle

A demo milestone is complete only when a second engineer can
reproduce the path and see which parts are measured, which are
fallback behavior, and which remain unproven.
