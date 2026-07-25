---
title: Current surface
order: 3
---

What is implemented today.

- **Replay** — the durable event log is read; the canonical world is re-derived.
- **Projection** — the world is projected to the chosen surface.
- **Reconcile** — the projection is diffed against the live surface and a minimal change is applied.

The SSG runs this loop at build time. The browser hydration runs
the same loop with a live DOM substrate.
