---
title: Failure
order: 4
---

Failure must be recorded, not disguised.

- **Stale outcome** — A result produced for an expired or superseded lease is rejected.
- **Provider failure** — The backend error is journaled with its execution boundary.
- **Restart recovery** — Durable history reconstructs canonical state and unfinished work.
- **Projection loss** — Derived views are rebuilt rather than promoted to hidden authority.
