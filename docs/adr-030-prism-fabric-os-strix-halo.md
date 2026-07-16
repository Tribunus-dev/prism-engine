# ADR-030: Prism Fabric OS on AMD Strix Halo

**Status:** Draft — architecture specification  
**Primary target:** AMD Strix Halo (Ryzen AI Max series)  
**Secondary target:** AMD Krackan Point, Hawk Point  
**Cluster fabric:** NTB over commodity ethernet  
**Implementation language:** Rust, with minimal platform-specific assembly  

## 1. Thesis

Strix Halo is the ideal first target for Prism Fabric OS because it eliminates the hardest problems before they start:

- **Unified memory.** CPU, GPU, and NPU share one physical address space. A pointer allocated by any processor is valid on all processors. Zero-copy dispatch is the default. The kernel does not need a memory management unit for accelerators — the platform already provides one.
- **Standard x86.** No custom RISC-V toolchain, no undocumented registers, no boot path reverse-engineering. Prism can boot directly on Zen 5 cores using existing EFI, ACPI, and IOMMU infrastructure.
- **NTB over commodity ethernet.** NTB bridges map PCIe address spaces across standard network links. Each Strix Halo node's unified pool extends into the cluster address space. The compiler sees one flat address space with latency gradients — same problem as within a node, just at different cost.

## 2. Comparative analysis

| Dimension | Blackhole | Strix Halo | Winner |
|---|---|---|---|
| Memory model | Discrete GDDR6 + SRAM, explicit NoC | Unified pool across CPU/GPU/NPU | Strix Halo |
| Control CPU | X280 RISC-V, non-coherent clusters | Zen 5 x86, fully coherent | Strix Halo |
| GPU | Tensix mesh (open RISC-V + ISA docs) | AMD RDNA 3.5+ (ROCm) | Tie (open vs ecosystem) |
| NPU | None (Tensix does everything) | AMD XDNA 2 (standard XDNN) | Strix Halo |
| Toolchain | Custom Tensix kernel compiler | LLVM/ROCm via amdxdna mailbox ioctl | Tie (NPU userspace is version-fragile) |
| Platform maturity | OpenSBI demo, gap-filled docs | Mature AMD platform, OEM support | Strix Halo |
| Interconnect | QSFP-DD (accelerator-only) | NTB over standard ethernet | Tie (NTB is cheaper) |
| Kernel complexity | Full M-mode/S-mode, MMU, TLB | Lightweight ECS runtime on Linux/host | Strix Halo |
| Autonomous boot | Requires host | Standard EFI boot | Strix Halo |
| Memory isolation | Software-managed NoC windows | IOMMU/SME — hardware-backed | Strix Halo |
| Per-tensor evolution fit | Isomorphic (per-core = per-tensor) | Same theory, GPU dispatch | Blackhole (Tensix is more granular) |

**Winner: Strix Halo for revision 0 by a wide margin.** Blackhole's per-core granularity is architecturally elegant, but Strix Halo eliminates 80% of the kernel engineering while keeping the compiler pipeline (Waves 13-29) nearly identical.

## 3. Architecture

### 3.1 Single-node model

```
Strix Halo SoC
┌────────────────────────────────────────────┐
│ Zen 5 CCD (up to 16 cores)                │
│  Prism Fabric OS control plane             │
│   ├── ECS world (admission, scheduling)    │
│   ├── host bridge                          │
│   ├── cimage service                       │
│   └── NTB cluster coordination             │
├────────────────────────────────────────────┤
│ RDNA 3.5+ GPU (40 CU)                     │
│  Shared memory — same pointers as CPU     │
│  ROCm dispatch via queue writes            │
│  Large matmul / attention                  │
├────────────────────────────────────────────┤
│ XDNA 2 NPU (up to 50 TOPS)               │
│  Shared memory — same pointers as CPU      │
│  XDNN dispatch via mailbox                 │
│  Quantized layers (ternary, binary)        │
├────────────────────────────────────────────┤
│ Unified DDR5 / LPDDR5X memory             │
│  Up to 128 GB shared pool                  │
│  No copies — every processor sees it       │
├────────────────────────────────────────────┤
│ PCIe root complex                          │
│  NTB bridge → ethernet → other nodes       │
└────────────────────────────────────────────┘
```

### 3.2 Multi-node cluster

```
NTB fabric (commodity ethernet + NTB NIC)
├── Node 1: Strix Halo — Prefill + attention
├── Node 2: Strix Halo — Decode + KV cache  
├── Node 3: Strix Halo — Draft / speculative
├── Node 4: Strix Halo — Vision / encoders
└── Node 5: Storage + network gateway (thin)
```

NTB extends the unified memory model across nodes. A buffer allocated on node 1's memory is addressable from node 2's GPU at the cost of NTB latency (500 ns–2 µs vs 0 ns for same-node). The evolutionary compiler treats cross-node dispatch as a higher-latency memory hop — exactly the same model as within-node dispatch, just with a different cost parameter.

## 4. Why unified memory changes the kernel

A discrete memory system (NVIDIA dGPU, Blackhole) requires the kernel to:

1. Track which device owns each allocation
2. Insert explicit copy operations at ownership boundaries
3. Manage TLB mappings per device
4. Flush caches across coherency domains
5. Handle page faults for on-demand migration

A unified memory system (Strix Halo) requires none of this. The hardware already provides:

- **CPU and GPU share page tables** (AMD IOMMUv2 / SVM)
- **NPU shares the same pool** (XDNN uses host pointers)
- **NTB maps remote memory into the local address space** (PCIe BAR windows)
- **Coherency is hardware-managed** (no software cache flushes)

The kernel becomes a **thin admission and scheduling layer** on top of hardware that already works.

## 5. Strix Halo cimage profile

```rust
pub struct StrixHaloCimageProfile {
    // Inherited from universal cimage
    pub manifest: CimageManifest,
    pub tensors: Vec<TensorPayload>,

    // Strix Halo specific
    pub gpu_kernels: Vec<RocmKernelBundle>,
    pub npu_kernels: Vec<XdnKernelBundle>,
    pub memory_plan: UnifiedMemoryPlan,  // single address space, no copies
    pub affinity_map: Vec<ProcessorAffinity>,  // per-layer: GPU, NPU, or CPU

    // NTB cluster
    pub ntb_topology: NtbTopology,
    pub placement_alternatives: Vec<NtbPlacement>,
}
```

The memory plan is trivial compared to discrete: allocate at compile time, use the same addresses at runtime.

## 6. Kernel structure

| Component | Strix Halo kernel | Blackhole kernel |
|---|---|---|
| Boot | EFI stub + ACPI | Host-assisted image load |
| Memory | Existing page tables + allocate from pool | Custom VMM, TLB leases |
| Scheduler | Same ECS scheduler | Same ECS scheduler |
| GPU dispatch | Write ROCm queue doorbell | Write Tensix NoC doorbell |
| NPU dispatch | Write XDNN mailbox | N/A |
| Isolation | IOMMU/SME (hardware) | Software capability enforcement |
| Cluster | NTB descriptor rings | TT-Fabric |
| Recovery | Standard PCIe AER + RAS | Custom card reset path |

The Strix Halo kernel is the same ECS runtime as Blackhole would use, but it delegates memory management, page tables, and device isolation to hardware that already provides them correctly. **The kernel is simpler not because Prism changes, but because AMD already did the hard part.**

## 7. Boot sequence

1. Standard UEFI boot
2. Prism Fabric OS ELF loaded by EFI stub
3. ACPI table scan detects GPU, NPU, NTB device
4. IOMMU configured for SVM/Shared Virtual Memory
5. ECS world initialized with device entities
6. NTB bridge driver discovers cluster peers
7. Host bridge started on management interface
8. READY published to cluster
9. Cimage admission + dispatch loop begins

No custom boot protocol. No host dependency. No firmware updates.

## 8. NTB cluster coordination

NTB maps each node's memory into every other node's PCIe address space. The cluster scheduler (from ADR-028) reads NTB latency measurements and assigns work:

```
Node 1 renders attention output to buffer at address 0x7f00_..._a000
Node 2's GPU directly reads that address via NTB window
Node 2 applies matmul to the attention output
Node 2 writes result to address 0x7f00_..._b000
Node 1's CPU reads the final output at that address
```

No RPC. No serialization. No copy between nodes. The NTB card translates PCIe transactions into ethernet packets transparently. The compiler baked the address plan into the cimage.

## 9. Security model

## 9.1 NPU driver risk

The `amdxdna` kernel driver is mainlined since Linux 6.14, but the full userspace stack (XRT, Peano compiler, IRON API, MLIR AI Engine) is version-fragile and not uniformly packaged across distributions.

Prism mitigates this by targeting a narrower interface:
- **Kernel interface only.** The `amdxdna` mailbox ioctl is the stable ABI — open source, mainlined.
- **No XRT dependency.** Submit compiled graph handles, not generic programs.
- **Unified memory eliminates buffer IOCTLs.** No DMA-buf imports — one pointer for all processors.
- **Offline compilation.** Peano/MLIR-AIE toolchain runs at build time, packaged into cimage.
- **Fallback path.** NPU-targeted tensors run on CPU (Zen 5 codegen) if NPU unavailable.


Trusted in revision 0: AMD PSP (Platform Security Processor), IOMMU/SME, standard UEFI Secure Boot, Prism signing root.

The hardware trust chain on Strix Halo is far stronger than Blackhole because AMD's platform security infrastructure (PSP, IOMMU, SME, SEV-SNP) is mature and battle-tested. Prism Fabric OS on Strix Halo can claim hardware-backed isolation from day one, which Blackhole could not.

## 10. Revision 0 gate

1. Boot Prism Fabric OS on Strix Halo hardware via EFI stub
2. Discover GPU (RDNA 3.5+) and NPU (XDNA 2) via ACPI
3. Allocate a unified shared buffer
4. Dispatch a GPU kernel via ROCm queue from the ECS runtime
5. Dispatch an NPU kernel via XDNN from the same runtime
6. Verify zero-copy: GPU and NPU modify the same buffer without explicit data movement
7. Admit and execute a small cimage with per-layer (format, operation) targeting different processors
8. Discover a second Strix Halo node via NTB
9. Route a work packet across nodes with address translation
10. Return matched execution receipts to the cluster coordinator

## 11. Repository scaffold

```
prism-fabric-os/
├── kernel/
│   └── prism-fabric-amd/    — EFI stub, ACPI, IOMMU, ECS runtime
├── platform/
│   ├── prism-amd-gpu/       — ROCm queue dispatch, queue management
│   ├── prism-amd-npu/       — XDNN mailbox, graph submission
│   └── prism-ntb/           — NTB bridge, address window management, peer discovery
├── ecs/                     — Shared with Blackhole path
├── services/                — Shared: host-bridge, cimage, telemetry, recovery
├── artifacts/               — Shared: cimage format, signatures
├── protocols/               — Shared: boot ABI, host ABI, receipt ABI, NTB cluster ABI
├── host/                    — Shared: provisioner, compiler, CLI
├── tests/                   — boot, memory, gpu, npu, cimage, ntb, recovery
└── docs/
```

~15 crates vs Blackhole's ~50. Most of the complexity is shared between both targets.
