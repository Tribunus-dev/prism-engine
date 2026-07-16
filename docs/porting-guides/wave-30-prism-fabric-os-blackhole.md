# Prism Fabric OS for Tenstorrent Blackhole — Scaffold Specification

**Status:** Draft architecture specification  
**Primary target:** Blackhole p150a/p150b  
**Secondary target:** Blackhole p100a  
**Execution CPU:** One SiFive X280 four-core coherent cluster initially  
**Accelerator substrate:** Blackhole Tensix grid, GDDR6, NoC, TT-Fabric  
**Implementation language:** Rust, `no_std`, tightly isolated unsafe hardware code  
**Native workload format:** Prism `.cimage` + Blackhole execution bundles  
**Compatibility target:** None — no Linux, POSIX, GNU, ELF userspace, or TT-NN compatibility at the OS boundary

## 1. Executive summary

Blackhole is the rare commercially available accelerator that can plausibly become an autonomous computer running Prism's operating substrate. It combines 16 general-purpose RISC-V control cores with 120 openly programmable Tensix execution islands, explicit NoC routing, distributed memory domains, a programmable inter-card fabric, and an open software stack. The card already resembles the architecture Prism Fabric OS describes: control processors above, deterministic execution islands below, explicit data movement between typed memory domains.

This scaffold documents the hardware baseline, architecture, kernel structure, ECS systems, memory/NoC model, execution subsystem, cimage profile, host bridge, TT-Fabric integration, security model, recovery model, testing strategy, milestone gates, and revision 0 definition of done.

## 2. Hardware baseline

### 2.1 Card specifications

| Resource | p100a | p150a/p150b |
|---|---|
| Advertised Tensix cores | 120 | 120 |
| X280 cores | 16 | 16 |
| Distributed SRAM | 180 MB | 180 MB |
| GDDR6 | 28 GB | 32 GB |
| Memory bandwidth | 448 GB/s | 512 GB/s |
| Host interface | PCIe 5.0 x16 | PCIe 5.0 x16 |
| Board power | 300 W | 300 W |
| Inter-card ports | None | 4 × 800 Gbit/s QSFP-DD |

### 2.2 X280 topology

Four L2CPU blocks, each with 4 coherent X280 cores and 4 GB locally attached DRAM. Blocks are NOT coherent with each other. Cross-block communication uses explicit message passing.

Control domain: one selected L2CPU block with local DRAM. Provisioner must avoid blocks with harvested DRAM controllers.

### 2.3 Tensix execution model

Each Tensix core: 5 × RV32IM control processors, matrix & vector engines, pack/unpack units, NoC interfaces, ~1.5 MB local SRAM. Normal operation uses reader/compute/writer with hardware-assisted circular buffers.

### 2.4 Alignment requirements

| Access | Alignment |
|---|---|
| Tensix L1 access | 16-byte |
| DRAM read | 64-byte |
| DRAM write | 16-byte |
| PCIe read | 64-byte |
| PCIe write | 16-byte |
| Native compute tile | 32×32 elements |
| X280 small TLB | 2 MB |
| X280 large TLB | 128 GB |

## 3. Repository scaffold

```
prism-fabric-os/
├── boot/        — prism-bh-stage0, linker, image-builder
├── kernel/      — prism-machine-frame (MMU, sched, IPC, realms, panic)
├── platform/    — X280, Blackhole address/NoC/TLB/topology/interrupt/cache/reset
├── ecs/         — core, schema, schedule, receipts
├── services/    — fabric-init, host-bridge, topology, region, artifact, cimage, tensix, telemetry, recovery
├── execution/   — program format, loader, circular-buffer, command-queue, placement
├── artifacts/   — cimage-core, cimage-blackhole, signatures, admission
├── protocols/   — boot-abi, host-abi, domain-abi, fabric-abi, receipt-abi
├── host/        — provisioner, compiler, artifact-store, gateway, CLI
├── simulation/  — mock, QEMU, TTSim, fault-injection
├── kernels/     — smoke, memory, matmul, collectives, fabric
├── tests/       — boot, ABI, memory, NoC, Tensix, cimage, fabric, recovery
└── docs/        — architecture, hardware-evidence, ABI, safety, validation
```

## 4. Revision 0 milestone gates

| Gate | Deliverable |
|---|---|
| Gate 0 | Versioned hardware evidence map, firmware compatibility table, ABI ownership |
| Gate 1 | Rust boot image on one X280 hart, host-visible log ring, panic record |
| Gate 2 | 4-hart coherent machine frame with CLINT, PLIC, scheduler, capability table, IPC |
| Gate 3 | TLB lease manager, harvesting map, NoC read/write smoke tests |
| Gate 4 | Bank-aware GDDR6 allocator, interleaved/sharded regions, transfer validation |
| Gate 5 | First Tensix reader/compute/writer program, circular-buffer setup, completion |
| Gate 6 | Blackhole cimage deployment, model-region placement, sequential graph operations |
| Gate 7 | Autonomous inference session with persistent dispatcher, KV cache, host gateway |
| Gate 8 | Two-card TT-Fabric execution, region transfer, distributed graph, matching receipts |

## 5. Referenced ADR documents

- ADR-028: NTB cluster coordination — distributed heterogeneous compute across machines
- ADR-029: Prism Fabric OS on Tenstorrent Blackhole — this document
