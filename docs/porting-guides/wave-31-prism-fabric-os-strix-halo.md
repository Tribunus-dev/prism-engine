# Wave 31: Prism Fabric OS on Strix Halo — Porting Guide

**Status:** Architecture spec  
**Target:** AMD Strix Halo (Ryzen AI Max)  
**Cluster fabric:** NTB over commodity ethernet  
**Dependencies:** ADR-030, ADR-028 (NTB), ADR-027 (memory model), Waves 13-27 (compiler pipeline)

## Overview

Strix Halo is the ideal first Fabric OS target because it eliminates the hardest problems: unified memory (no copy management), standard x86 (no custom toolchain), mature platform security (IOMMU/SME), and commodity NTB clustering. The compiler pipeline built in Waves 13-27 maps directly — per-tensor (format, operation) assignment targeting RDNA GPU, XDNA NPU, or Zen CPU within the same address space.

## Key differences from Blackhole

| Aspect | Blackhole approach | Strix Halo approach | Why |
|---|---|---|---|
| Boot | Host-assisted image load | EFI stub + ACPI | Standard PC boot |
| GPU dispatch | Tensix NoC doorbell | ROCm queue write | Mature API |
| NPU dispatch | N/A | XDNN mailbox | Hardware exists |
| Memory | TLB leases across domains | Shared pointers, hardware SVM | Hardware provides it |
| Isolation | Software capabilities | IOMMU/SME/SEV-SNP | Hardware provides it |
| Toolchain | Custom Tensix compiler | LLVM/ROCm/XDNN | Well-lit path |
| Kernel size | ~50 crates, full VMM | ~15 crates, thin ECS runtime | Hardware does the heavy lifting |

## Shared substrate

The following layers are target-independent and shared between Blackhole and Strix Halo builds:

- ECS world (prism-ecs-core, prism-ecs-schema, prism-ecs-schedule)
- Cimage format and admission
- Host bridge protocol
- Receipt format
- NTB cluster protocol
- Telemetry and recovery systems

The target-specific layers are only: device discovery, queue dispatch, and memory model assumptions.

## Milestone gates

| Gate | Deliverable | Hardware dependency |
|---|---|---|
| Gate 0 | EFI stub boots Rust executable on Strix Halo | Strix Halo board |
| Gate 1 | ACPI table scan discovers GPU (RDNA) + NPU (XDNA) | Strix Halo board |
| Gate 2 | Unified buffer allocated, both GPU and NPU validated | Strix Halo board |
| Gate 3 | GPU kernel dispatched via ROCm queue from ECS runtime | Strix Halo board |
| Gate 4 | NPU kernel dispatched via XDNN mailbox from ECS runtime | Strix Halo board |
| Gate 5 | Small cimage admitted and executed with per-layer heterogeneous dispatch | Strix Halo board |
| Gate 6 | NTB peer discovery across two nodes | 2× Strix Halo + NTB NICs |
| Gate 7 | Cross-node work packet with address translation | 2× Strix Halo + NTB NICs |
| Gate 8 | Cluster executes partitioned graph with matching receipts | 2× Strix Halo + NTB NICs |

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| NTB latency higher than expected | Medium | Compiler cost model accepts measured value; evolutionary search optimizes around it |
| XDNN NPU kernel compilation unclear | Low | Prism compiles to XDNN graph format; AMD toolchain is documented |
| ROCm queue management from non-Linux | Medium | Use host Linux for GPU dispatch in rev 0 (AMD kernel driver); move to direct queue writes in rev 1 |
| Power management | Low | Standard ACPI C-states; GPU power managed by AMD PMFW as normal |
| Firmware compatibility across Strix Halo SKUs | Low | ACPI DSDT provides platform parameters; minimal hardcoded assumptions |
