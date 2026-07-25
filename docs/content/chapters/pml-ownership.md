---
title: What each system owns
order: 4
---

Different expertise. One handoff.

## Prism ML / Bonsai owns

- Training behavior
- Quantization strategy
- Ternary / binary representation
- Calibration and numerical validation

## Prism Engine owns

- Representation admission
- Physical tile layout
- Execution views
- Scheduling, residency, and receipts

## What the integration can become

Prism Engine can expose deployment evidence back upstream: tensor
sensitivity, layout pressure, target-specific failures, and runtime
observations. Prism ML can use that evidence to improve the next
model or representation. The result is a feedback loop without
collapsing training and deployment into one system.
