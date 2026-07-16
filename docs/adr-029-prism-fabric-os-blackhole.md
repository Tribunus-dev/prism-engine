# ADR-029: Prism Fabric OS on Tenstorrent Blackhole

**Status:** Draft — architecture specification  
**Dependencies:** ADR-028 NTB cluster coordination, ADR-027 memory model  
**Owner:** distributed, tenstorrent

## 1. Executive decision

Blackhole cards are the canonical hardware target for Prism Fabric OS. A Blackhole card combines:

- **16 SiFive X280 RISC-V cores** (four coherent 4-core L2CPU clusters) running general-purpose code
- **120 Tensix cores** each with 5 RV32IM processors (reader, compute unpack, math, pack, writer) and ~1.5 MB local SRAM
- **Up to 32 GB GDDR6** across 8 DRAM banks
- **p150 models**: 4 × 800 Gbit/s QSFP-DD inter-card fabric ports
- **TT-Fabric** — programmable Ethernet-core routing fabric for inter-card data plane
- **Open source stack** — OpenSBI + Linux boot demo, Luwen host driver, TT-Metalium

The card already resembles the architecture Prism needs: control processors above, Tensix execution islands below, explicit NoC routing, distributed memory domains, and a programmable inter-card fabric. No forcing required.

## 2. Three execution classes

| Class | Hardware | Runs |
|---|---|---|
| X280 control plane | 4 coherent X280 cores (1 L2CPU cluster) | Fabric OS: ECS world, cimage admission, scheduling, host bridge, telemetry |
| Tensix data plane | 120 Tensix cores × 5 RV32IM processors | Precompiled reader/compute/writer kernel programs |
| TT-Fabric data plane | Ethernet core RISC-V processors | Inter-card packet routing, transport |

## 3. Prism Fabric OS architecture

```
Linux host
┌───────────────────────────────────────────────┐
│ prism-bh-provisioner                          │
│  Artifact storage, card boot, emergency reset │
└────────────────────┬──────────────────────────┘
                     │ PCIe BAR / DMA / polling bridge
Blackhole card       │
┌────────────────────▼──────────────────────────┐
│ Vendor firmware (ROM → MCUboot → CMFW/DMFW)   │
├───────────────────────────────────────────────┤
│ OpenSBI (M-mode)                              │
├───────────────────────────────────────────────┤
│ Prism Machine Frame (S-mode)                  │
│  MMU · interrupts · timers · SMP · capabilities│
├───────────────────────────────────────────────┤
│ Prism control services                        │
│  ECS world · cimage admission                 │
│  topology · TLB management · program loader   │
│  graph scheduler · host bridge                │
├───────────────────────────────────────────────┤
│ Blackhole execution substrate                 │
│  Tensix grid · GDDR6 · NoC · Ethernet cores   │
└───────────────────────────────────────────────┘
```

### 3.1 Host dependencies (revision 0)

The host remains responsible for:
- PCIe enumeration and card reset
- Writing the Prism boot image into X280 DRAM
- Programming the X280 reset vector
- Supplying precompiled Tensix binaries
- External networking and persistent storage
- Emergency whole-card reset

### 3.2 Hypervisor break

When Blackhole exposes enough documented autonomous-device infrastructure, the host shrinks to a provisioning + storage + network gateway.

## 4. Machine model

### 4.1 X280 topology

Blackhole has four L2CPU blocks, each containing four coherent X280 cores and 4 GB locally attached DRAM. Blocks are **not coherent** with each other — cross-block communication uses explicit message passing.

Revision 0 uses one L2CPU block:
```
Hart 0: interrupt handling, kernel coordination, recovery
Hart 1: Tensix dispatch and completion processing
Hart 2: host bridge and artifact transfer
Hart 3: placement, telemetry, background ECS systems
```

### 4.2 Memory topology

| Memory class | Size | Use |
|---|---|---|
| X280 local DRAM | 4 GB per L2CPU block | Kernel, ECS state, host mailbox, deployment metadata |
| GDDR6 (8 banks) | 28-32 GB total | Cimage weights, KV cache, activation arenas |
| Tensix SRAM | 180 MB total | Circular buffers, kernel code, semaphores |
| Tensix L1 per core | ~1.5 MB | Local execution state |

No general heap in Tensix SRAM. No general heap across non-coherent L2CPU blocks.

### 4.3 Address space

X280 address map includes:
- CLINT at `0x0200_0000`
- PLIC at `0x0C00_0000`
- 256 MB peripheral port (uncached control)
- 64 TiB memory port (coherent cached)
- 64 TiB system port (uncached)

NoC TLB provides: 224 windows of 2 MB, 32 windows of 128 GB. These are kernel-managed scarce capabilities (TlbLease), not generic pointer mappings.

## 5. ECS world

### 5.1 Core entities

```
BlackholeDevice
 ├── L2CpuCluster[4]
 ├── TensixCore[120]
 ├── DramBank[8]
 ├── EthernetCore[]
 ├── NoC0 / NoC1
 ├── TlbWindow[256]
 ├── FabricLink[]
 └── ThermalDomain[]
```

### 5.2 Core components

```rust
struct TensixCore {
    coordinate: MeshCoordinate,
    local_sram: RegionId,
    supported_formats: FormatSet,
    resident_program: Option<ProgramId>,
    queue_state: QueueState,
    health: HealthState,
}
```

### 5.3 Mutation authority

Every mutable hardware resource has exactly one authoritative ECS system:

| Resource | System |
|---|---|
| NoC TLB windows | `TlbLeaseSystem` |
| Physical X280 pages | `RegionAllocationSystem` |
| Tensix L1 ranges | `ProgramMaterializationSystem` |
| Tensix launch state | `ExecutionDispatchSystem` |
| Fabric routing | `FabricRouteSystem` |
| Realm capabilities | Kernel capability subsystem |
| Artifact admission | `CimageAdmissionSystem` |

## 6. Cimage as native executable format

A Blackhole cimage carries architecture-specific sections alongside the universal model manifest:

```
.cimage
├── Universal manifest
├── Tensor payloads
├── Blackhole requirements
│   ├── firmware range
│   ├── required usable cores
│   ├── required DRAM banks
│   └── topology constraints
├── Program bundles (reader/compute/writer per core group)
├── Circular-buffer plans
├── NoC route templates
├── TT-Fabric route templates
├── Placement alternatives (dense/alternate/multi-card/fallback)
├── Validation vectors
└── Signatures
```

Admission checks: signature, architecture match, firmware range, precision format availability, topology fit after harvesting, DRAM/L1 fit, alignment validity, kernel target validity, NoC destination permissions.

## 7. TT-Fabric integration

Prism layers above TT-Fabric, not replacing it:

```
Prism capability/session protocol
        ↓
Prism object and execution messages
        ↓
TT-Fabric sockets
        ↓
TT-routing and transport
        ↓
Blackhole Ethernet cores
        ↓
QSFP-DD card links
```

Prism owns: identity, capabilities, workloads, artifact routing, execution graph routing, KV ownership, receipts. TT-Fabric owns: physical topology, packet transport, routing primitives, flow control, virtual channels.

## 8. Multi-card deployment

The strongest deployment model — a rack of p150 cards with no Linux host orchestrating inference:

```
Host: provisioning, storage, network gateway, emergency reset

Card 0: Prism OS · model ingress · prefill
Card 1: Prism OS · decode · KV shard A
Card 2: Prism OS · drafter · speculative execution
Card 3: Prism OS · vision/audio encoders
```

Cards discover each other, exchange signed inventory, establish TT-Fabric routes, and partition the execution graph autonomously. The host only sends requests and receives outputs.

## 9. Security model

**Trusted in revision 0:** physical card, Tenstorrent ROM/firmware, Linux host kernel, Luwen, provisioner, signing root, Prism machine frame.

**Not trusted:** cimages, model kernels, user workloads, host clients, remote Fabric cards.

Every executable object carries: content hash, format version, target architecture, firmware range, producer/compiler identity, signatures.

**Multi-tenancy is NOT claimed in revision 0.** Hard isolation requires verified X280 MMU boundaries, NoC addressability, TLB lease cleanup, Tensix L1 cleanup, GDDR6 cleanup, host DMA isolation, firmware mailbox isolation, and inter-card traffic isolation.

## 10. Core Rust interfaces

```rust
pub trait ExecutionIsland {
    fn capabilities(&self) -> ExecutionCapabilities;
    fn reserve(&self, request: ReservationRequest) -> Result<Reservation, ExecutionError>;
    fn install(&self, reservation: &Reservation, program: &VerifiedProgramBundle) -> Result<InstalledProgram, ExecutionError>;
    fn launch(&self, program: &InstalledProgram, arguments: &RuntimeArguments) -> Result<ExecutionToken, ExecutionError>;
    fn poll(&self, token: &ExecutionToken) -> Result<ExecutionState, ExecutionError>;
    fn recover(&self, token: &ExecutionToken, policy: RecoveryPolicy) -> Result<RecoveryReceipt, ExecutionError>;
}
```

Every low-level operation has `MockBackend | TtSimBackend | HostLuwenBackend | NativeX280Backend`.

## 11. Revision 0 gate

PASS when:
1. Boot from Linux host onto one healthy L2CPU cluster
2. All four coherent X280 harts running
3. Versioned host bridge established
4. Harvested topology discovered (not hardcoded)
5. X280 NoC TLB windows managed through leases
6. Local X280 memory + accelerator GDDR6 allocated
7. Signed Blackhole cimage verified
8. One precompiled reader/compute/writer Tensix program launched
9. Model data preserved across repeated requests
10. Outputs returned through host bridge
11. Signed admission + execution receipts
12. Stuck execution detected and crash snapshot preserved
13. Host-driven card reset survived
14. Same control logic runs against mock, simulator, and physical backends

NOT required in revision 0: hostless boot, on-card kernel compilation, Linux compatibility, hostile multi-tenancy, 16-core SMP, multi-card TT-Fabric, firmware replacement, direct NVMe/external networking.
