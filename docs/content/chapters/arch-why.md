---
title: Why not extend an inference runtime?
order: 7
---

Existing engines can be excellent execution backends. Prism makes a
different architectural choice: representation admission, layout
planning, execution views, residency, scheduling metadata,
validation, and artifact sealing happen once during compilation.

## Compare

- **Typical runtime** — Load model, derive strategy dynamically, dispatch.
- **Prism** — Compile policy, seal artifact, execute explicit plan.

## Operational consequence

Two CImages from the same source can be compared, audited, and
reproduced without reconstructing hidden runtime heuristics.
