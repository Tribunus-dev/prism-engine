# ADR-027: Memory Model — Unified vs Discrete Topology

**Status:** Draft — architecture formalization
**Dependencies:** Wave 28 vendor runtime crates, Wave 29 heterogeneous dispatch
**Owner:** systems

## 1. Problem

The compiler currently treats all hardware as equivalent — emit source, compile, dispatch. But there are two fundamentally different memory topologies with different data movement rules:

**Unified memory** (Apple Silicon, AMD Strix Halo, Intel Lunar Lake):
- CPU + GPU + NPU share one physical address space
- A pointer allocated by CPU is valid on GPU and NPU
- No explicit copy needed — just pass the address
- Zero-copy dispatch is the default and the optimal path

**Discrete memory** (NVIDIA dGPU, AMD dGPU, Tenstorrent, Intel Arc dGPU):
- Each device has its own VRAM, isolated by PCIe
- A CPU pointer is NOT valid on the GPU
- Data must be copied via `cudaMemcpy` / `hipMemcpy` / `zeCommandListAppendMemoryCopy`
- The only thing that crosses the PCIe bus is: (1) input buffers before dispatch, (2) output buffers after dispatch, (3) sync signals

The compiler must know which model each target uses, and the heterogeneous scheduler must insert copy operations at discrete boundaries.

## 2. Design

### MemoryModel enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryModel {
    /// Unified: CPU + accelerator share physical memory. Zero-copy pointers.
    Unified,
    /// Discrete: accelerator has isolated VRAM. Explicit copies required.
    Discrete {
        pcie_gen: u8,       // 3, 4, 5
        pcie_lanes: u8,     // 4, 8, 16
        bandwidth_gbs: f32, // theoretical peak GB/s
    },
}
```

### HardwareDescriptor component

Attached to each device entity in the World. The heterogeneous scheduler reads this to decide whether copy operations are needed.

```rust
pub struct HardwareDescriptor {
    pub name: String,                    // "Apple M5 Max GPU", "NVIDIA RTX 5090"
    pub kind: HardwareKind,              // Gpu, Npu, Cpu, Ane
    pub memory_model: MemoryModel,
    pub memory_size_bytes: u64,          // total VRAM or shared pool
}
```

### Compile-time constraint

The evolutionary search (Wave 27) includes memory model as a constraint. A CompilePlan that assigns a tensor to a discrete GPU must: (1) allocate VRAM on that GPU, (2) insert copy operations at dispatch boundaries, (3) account for PCIe bandwidth in the cost model.

### Discrete dispatch flow (inserted by the scheduler)

```
Input buffer in shared/CPU memory
    → cudaMemcpyAsync(input, device_buffer, size, H2D)
    → kernel<<<grid, block>>>(device_buffer, ...)
    → cudaMemcpyAsync(output, host_buffer, size, D2H)
    → Return output to caller
```

### Unified dispatch flow (zero-copy)

```
Input buffer in shared memory
    → kernel<<<grid, block>>>(input, ...)    // same pointer, no copy
    → Return output (same memory)
```

## 3. File map

| File | Change |
|---|---|
| `prism-ecs-ir/src/backend_dispatch.rs` | Add `MemoryModel` to `HalFormat` or as a separate resource |
| `prism-ecs-server/src/heterogeneous_dispatch.rs` | Read `HardwareDescriptor` on each device, insert copy ops at discrete boundaries |
| `prism-ecs-ir/src/evolution.rs` | Add `MemoryModel` constraint to `CompilePlan` mutation operators |

## 4. Gate

- CompilePlan targeting unified memory produces zero-copy dispatch
- CompilePlan targeting discrete memory inserts explicit copy ops
- Cross-device work packets (e.g., ANE → GPU over PCIe) serialize only buffer handles, not data
