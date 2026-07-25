---
title: How it works
order: 2
---

Three phases: replay, project, reconcile.

1. **Replay** — Read the durable event log and re-derive the canonical ECS world. The replay is deterministic; it does not call any provider, device, or network.
2. **Project** — A renderer (the projection) reads the world and produces a typed view. The view is independent of the surface; the same world can be projected to HTML, JSON, or a visualization.
3. **Reconcile** — The renderer compares the projection to the live surface and applies a minimal diff. On the SSG, the surface is the file system; on the browser, the live DOM.

The result is a representation that survives process restarts,
matches the canonical world, and is the only path that produces
visible state.
