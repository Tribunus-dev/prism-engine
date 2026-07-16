# ADR-028: NTB Cluster Coordination — Distributed Heterogeneous Compute

**Status:** Draft
**Dependencies:** ADR-027 memory model, Wave 28 vendor runtimes, Wave 29 scheduler
**Owner:** distributed

## 1. Thesis

NTB (Non-Transparent Bridge) maps PCIe address spaces across ethernet. Each node's unified memory + discrete accelerators become one globally addressable pool. The compiler treats the cluster as one `HardwareDescriptor` with a topology constraint — cross-node hops cost more than same-node hops, but the addressing model is the same.

## 2. Topology

```
Node 1 (Apple Silicon):    ANE + GPU + NPU + CPU
Node 2 (x86 + NVIDIA):     GPU (discrete, 24GB VRAM)
Node 3 (x86 + Tenstorrent): Wormhole mesh (12x10 RISCV cores)
Node 4 (AMD Strix Halo):   GPU + NPU + CPU (unified)
       ↓ NTB (PCIe over ethernet)
     All memory addressable from any node
```

Each node runs `prism-server`. A coordinator node runs the global `NtbScheduler`.

## 3. Design

### NtbNode resource (one per node in the cluster)

```rust
pub struct NtbNode {
    pub id: u32,
    pub address: String,            // IP:port
    pub capabilities: Vec<HardwareDescriptor>,
    pub current_load: f32,          // 0.0 — 1.0
    pub ntb_latency_ns: u64,        // measured round-trip
}
```

### WorkPacket entity

The unit of work that crosses node boundaries. Contains buffer handles, not data.

```rust
pub struct WorkPacket {
    pub source_node: u32,
    pub dest_node: u32,
    pub input_buffers: Vec<Entity>,    // references to remote buffers
    pub output_buffers: Vec<Entity>,
    pub kernel: Entity,                // the HalExecutable to run
}
```

### NtbScheduler

Extends the ContinuousBatchingScheduler with node topology awareness. For each batch, assigns layers to nodes based on: (1) which node has the fastest hardware for that format+operation, (2) current load per node, (3) NTB latency penalty.

```rust
pub struct NtbScheduler {
    pub nodes: Vec<NtbNode>,
    pub local_scheduler: Scheduler,
    pub topology: NtbTopology,
}
```

## 4. File map

| File | Contents |
|---|---|
| `prism-ecs-server/src/ntb_cluster.rs` | NtbNode, NtbTopology, cluster discovery |
| `prism-ecs-server/src/ntb_scheduler.rs` | NtbScheduler, WorkPacket dispatch |
| `prism-ecs-server/src/ntb_transport.rs` | NTB address translation, buffer sharing |

## 5. Gate

- Two machines running prism-server discover each other over NTB
- A work packet crosses from node 1's ANE to node 2's GPU and returns correct output
- The evolutionary search assigns layers across nodes when the latency budget allows
- Single-node performance is preserved when no other nodes are present
