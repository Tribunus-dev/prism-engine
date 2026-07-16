# Waves 24: Compiler Pipeline — Fusion Systems, Codegen Backends, TableGen Parser

**Status:** Draft (pre-implementation)
**Dependency:** Waves 13-23 (ECS-native IR kernel, dialects, Metal codegen, assessments, evolution scaffolding)
**Owner:** kernel

## 1. Scope

Consolidates ADR-005 Waves 17-20, 22-23 into a single parallel dispatch. All are the same thing: building compiler passes and codegen backends as ECS systems, plus the `.td` parser that generates dialect definitions.

## 2. Parallel work streams

All streams target `prism-ecs-ir`. Each writes to its own files — no conflicts.

### Stream A: Fusion/partition system

Reads `linalg.matmul` + `arith.*` ops, analyzes data dependencies via `Uses` components, writes `FusionGroup` components.

| File | What |
|---|---|
| `src/fusion.rs` | `FusionGroup` component, `analyze_dataflow` system, `partition_fusion_groups` system |

### Stream B: codegen_cpu.rs

Reads same ECS IR as `codegen_metal.rs`, emits C/Accelerate source or Rayon dispatch.

| File | What |
|---|---|
| `src/codegen_cpu.rs` | `lower_matmul_to_cpu()`, `lower_to_cpu()`, `CpuLowerError` |

### Stream C: codegen_nvvm.rs

Reads same ECS IR, emits PTX source for NVIDIA GPUs.

| File | What |
|---|---|
| `src/codegen_nvvm.rs` | `lower_matmul_to_nvvm()`, `lower_to_nvvm()`, `NvvmLowerError` |

### Stream D: codegen_amdgpu.rs

Reads same ECS IR, emits AMDGCN source for AMD GPUs.

| File | What |
|---|---|
| `src/codegen_amdgpu.rs` | `lower_matmul_to_amdgpu()`, `lower_to_amdgpu()`, `AmdgpuLowerError` |

### Stream E: TableGen parser (`prism-tblgen` crate)

Parses MLIR-style `.td` files, emits Rust ECS dialect definitions matching `arith.rs` pattern.

| File | What |
|---|---|
| `crates/prism-tblgen/` | Full crate: lexer, parser, resolver, emitter, CLI |

### Stream F: Bonsai codec integration

Extends existing `evolution.rs` + `bonsai.rs` with codec implementations that each codegen module can call.

| File | What |
|---|---|
| `src/bonsai_codec.rs` | `ternary_dot_product()`, `binary_popcount_dot()`, codec dispatch table |

### Stream G: C++ dependency resolution

Update `Cargo.toml` feature flags so the full pipeline builds without optional C++ deps. Gated on Streams A-F passing.

## 3. Shared contract — HalExecutable

All codegen modules produce the same type:

```rust
#[derive(Debug, Clone)]
pub struct HalExecutable {
    pub format: HalFormat,     // Metal, PTX, AMDGCN, CSource
    pub source: String,        // The kernel source code
    pub entry_point: String,   // Kernel function name
    pub grid_dims: (u32, u32, u32),
    pub block_dims: (u32, u32, u32),
}

pub enum HalFormat {
    Metal,
    Ptx,
    AmdGcn,
    CSource,
}
```

## 4. Gate

1. All 7 streams produce compilable, tested code
2. `codegen_nvvm.rs` emits valid PTX for `linalg.matmul` on f32 (test: output contains `.version`, `.target`, `ldg`, `fma`)
3. `codegen_cpu.rs` emits compilable C with loop nests
4. `prism-tblgen` parses a multi-op `.td` file and generates Rust that compiles
5. `HalExecutable` round-trips through serde JSON unchanged
6. `cargo check --no-default-features -p prism-ecs-ir` succeeds (no optional C++ deps needed)
