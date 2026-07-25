---
title: Validation
order: 5
---

Compare plans, not slogans. The useful question is not whether a
provider behaves like a GPU. It is whether Prism can represent its
capabilities, produce a legal ECS-native plan, execute the selected
phases, and compare the resulting evidence against another
deployment.

- **Input** — same model, same workload, same quality policy
- **Plan A** — MI300X / ROCm-HIP, GPU validation path
- **Plan B** — XDNA/XDNA2, spatial planning path
- **Output** — latency, memory, KV, quality, receipts

## Current boundary

General Compute and provider integrations remain bounded by
backend, driver, device, and conformance evidence. A legal plan or
compiled CImage does not by itself claim production execution on
every provider.
