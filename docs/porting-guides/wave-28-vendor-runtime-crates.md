# Wave 28: Vendor Runtime Crates — Production HAL Backends

**Status:** Draft (pre-implementation)
**Dependency:** Wave 16 codegen backends, Wave 27 evolutionary pass pipeline
**Owner:** hardware

## 1. Scope

Deliver one vendor runtime crate per target that compiles emitted source → device binary → dispatch → synchronize → evidence. The codegen modules (Wave 16) produce source strings; these crates make them run.

## 2. Crate structure

| Crate | Target | Vendor runtime |
|---|---|---|
| `prism-cuda-runtime` | NVIDIA GPU | `ptxas` compilation, CUDA driver API, stream dispatch |
| `prism-rocm-runtime` | AMD GPU | ROCm compiler, HIP runtime, HSA dispatch |
| `prism-igc-runtime` | Intel GPU | Intel Graphics Compiler, Level Zero API |
| `prism-tt-runtime` | Tenstorrent | TT-Metalium runtime, RISCV core dispatch |

Each crate follows the same pattern as `prism-metal-runtime`.

## 3. Tenstorrent isomorphism

Tenstorrent's Wormhole architecture:
- 12x10 mesh of RISCV-32 cores (6,528 total), each with 64KB SRAM
- No cache coherence — explicit DRAM↔SRAM DMA via `tt::dma::tensor_to_core()`
- Cores communicate via asynchronous ethernet-like links (no shared memory)
- Programming model: compile a kernel per core, describe the data movement graph

**ECS mapping:**

| Tenstorrent concept | ECS form |
|---|---|
| Core | `TTCore { x: u32, y: u32, kernel: Entity }` component |
| DRAM buffer | `TTBuffer { address, size, format }` resource |
| Data movement | `TTTransfer { src: Entity, dst: Entity, tensor_slice }` component |
| Kernel binary | Compiled RISCV-32 ELF from a codegen module |

The per-tensor evolution search assigns each tensor to a specific core or group of cores. A `linalg.matmul` that spans 4 Tenstorrent cores gets 4 separate kernel entities, one per core, each with its own (format, operation, tile) plan.

## 4. Test gate per crate

- Emitted source compiles to a valid device binary via the vendor toolchain
- The compiled binary dispatches and produces correct output on a 4x4 matmul
- Timing evidence matches the vendor profiler within 5%
- No CUDA/ROCm/TT-Metalium/IGC headers linked at compile time — loaded dynamically or via FFI stubs
