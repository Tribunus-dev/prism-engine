# Wave 17 Assessment: IREE Flow → Stream → HAL Integration Path

**Date:** 2026-07-15
**Scout:** IreeScout
**Target:** ADR-005 Wave 17 — IREE Flow → Stream → HAL vertical slice

---

## 1. Executive Summary

**No IREE dependency exists anywhere in the workspace.** There is no `iree`, `iree-sys`, `iree-rs`, or `iree-rs` crate in any `Cargo.toml`, no IREE source files, no IREE-related configuration, and no IREE CI integration. The only MLIR-related dependency is **melior** (Rust MLIR bindings), gated behind the `mlir-runtime` feature flag and used solely for module parsing and verification — not for IREE's Flow/Stream/HAL pipeline semantics.

The existing compilation infrastructure provides **four independent building blocks** for Wave 17:
1. **ECS-native IR** (`prism-ecs-ir` crate) — fully operational linalg, arith, func, scf dialects with Metal codegen (151 tests pass)
2. **MlirExecutionContract** (`compute-core/src/ecs/mlir.rs`) — struct-level transform schedule → Metal lowering (13 kernel variants)
3. **Compiler systems** (`compute-core/src/ecs/system/compiler_systems.rs`) — GraphOptimizer, GraphEqualizer, BackendAssessment, CompileSchedule
4. **CPU runtime** (`compute-core/src/ecs/cpu_runtime/lowering.rs`) — Accelerate+Rayon program lowering

Wave 17 must **introduce IREE's pipeline semantics as ECS-native systems**, not as a crate dependency. The deliverable is a small `linalg.matmul` passing through all three IREE stages ported as Prism ECS phases.

---

## 2. Current State of Each Relevant Component

### 2.1 ECS-Native IR (prism-ecs-ir crate)

| Dialect/Module | Status | File |
|---|---|---|
| `arith` | Implemented | `crates/prism-ecs-ir/src/arith.rs` |
| `func` | Implemented | `crates/prism-ecs-ir/src/func.rs` |
| `scf` | Implemented | `crates/prism-ecs-ir/src/scf.rs` |
| `linalg` | Implemented (matmul, batch_matmul, fill) | `crates/prism-ecs-ir/src/linalg.rs` |
| `block`, `region` | Implemented | `crates/prism-ecs-ir/src/{block,region}.rs` |
| `value` (SSA) | Implemented | `crates/prism-ecs-ir/src/value.rs` |
| `ir_types` | Float, Integer, Tensor, Function types | `crates/prism-ecs-ir/src/ir_types.rs` |
| `lowering` | linalg.matmul → scf.for (scaffold) | `crates/prism-ecs-ir/src/lowering.rs` |
| `codegen_metal` | Lower to Metal MSL source | `crates/prism-ecs-ir/src/codegen_metal.rs` |
| `rewrite_driver` | Pattern-based rewrite engine | `crates/prism-ecs-ir/src/rewrite_driver.rs` |
| `dominance` | Dominator tree + frontier | `crates/prism-ecs-ir/src/dominance.rs` |
| `symbol_table` | Symbol resolution | `crates/prism-ecs-ir/src/symbol_table.rs` |
| `type_inference` | Type inference | `crates/prism-ecs-ir/src/type_inference.rs` |
| `evolution` | Evolutionary search | `crates/prism-ecs-ir/src/evolution.rs` |
| `serde` | Serialization | `crates/prism-ecs-ir/src/serde.rs` |
| `builder` | OpBuilder | `crates/prism-ecs-ir/src/builder.rs` |

**Key gap:** No IREE-specific constructs — no dispatch regions, no memory planning, no HAL command buffers. The crate has **zero IREE dependencies** and its `Cargo.toml` only depends on `prism-ecs-core`, `fxhash`, `serde`, `serde_json`, and `thiserror`.

### 2.2 MlirExecutionContract (`compute-core/src/ecs/mlir.rs`)

The existing MLIR contract system defines:

```rust
pub struct MlirExecutionContract {
    pub semantic_id: KernelSemanticId,
    pub inputs: Vec<MlirTensorType>,
    pub output: MlirTensorType,
    pub dialects: Vec<MlirDialect>,
    pub schedule: MlirTransformSchedule,
    pub target: MlirLoweringTarget,
}
```

Where `MlirLoweringTarget` already enumerates `Metal | Nvidia | Amd | Cpu | HetGpu`. The `TransformStep` enum provides `Tile | Fuse | Parallelize | Vectorize | DecomposeReduction`.

**Current limitation:** `lower_to_metal()` is the **only implemented lowering path**. `MlirLoweringTarget::Cpu` is declared but produces an error if used. There is no CPU codegen path in the contract — only hardcoded Metal kernel templates for 13 quantization variants.

The `mlir-runtime` feature (`dep:melior`) only parses and verifies the generated MLIR module text. It does **not** perform any IREE-specific transform, scheduling, or HAL lowering.

### 2.3 Existing Compiler Systems (`compute-core/src/ecs/system/compiler_systems.rs`)

| System | Phase | What it does |
|---|---|---|
| `GraphOptimizerSystem` | FusionDispatch | Constant folding, shape propagation, DCE |
| `GraphEqualizationSystem` | FusionDispatch | NF4 scale migration legality checks |
| `CompileScheduleSystem` | Compilation | Translates model manifest → ScheduledModule |
| `BackendAssessmentSystem` | Compilation | Scores ops against Metal/Accelerate/ANE/MLX/CPU backends |

These are **not IREE pipeline stages**. They operate at the model compilation level (CImage assembly), not at the linalg → Flow → Stream → HAL IR lowering level.

### 2.4 CPU Runtime (`compute-core/src/ecs/cpu_runtime/lowering.rs`)

The CPU runtime has a complete lowering pipeline from `FusedGroup` → `AccelerateRayonProgram` with `CpuProgramOp` enum supporting:
- `VdspRmsNorm`, `VforceSilu`, `VdspMul`, `VdspAdd`
- `CblasSgemm`, `CblasSgemv`
- `CustomInt8TileGemv`, `CustomNf4TileGemv`, `LayoutConvert`

This is where Wave 17.3's HAL → CPU backend would wire into — but currently the boundary is `FusedGroup` (from the execution plan fusion system), not `HALExecutable` entities from the IREE Stream stage.

### 2.5 Compile Session (`compute-core/src/ecs/compile_session.rs`)

The `CompileSession` registers: `DispatchFormationSystem`, `ScalarDispatchSystem`, `GraphEqualizationSystem`, `GraphOptimizerSystem`, `RegionCatalogueSystem`, `ExecutionGraphSystem`, `StagingSystem`, `TriLaneSystem`, `CompileScheduleSystem`, `BackendAssessmentSystem`, `TertiaryPipelineSystem`.

These are compile-time CImage assembly systems, not IREE-equivalent lowering systems. They share the same `SchedulePhase` infrastructure that Wave 17 would use for its Flow/Stream/HAL phases.

---

## 3. IREE Pipeline Architecture (from ADR-005)

The ADR maps IREE's pipeline to ECS-native form:

| IREE Stage | ECS Form | ADR Reference |
|---|---|---|
| Flow → dispatch region formation | ECS system: partition model entity into dispatch region entities | ADR § Wave 17.1, also § Decision mapping table |
| Stream → resource scheduling | ECS systems: memory planning + task ordering | ADR § Wave 17.2 |
| HAL → CPU backend | ECS systems: command buffer, buffer allocation, executable ABI | ADR § Wave 17.3 |
| HAL backends (Metal, CUDA, etc.) | Backend dispatch systems reading `HALExecutable` components | ADR § Decision mapping table |

The ADR explicitly says these stages map to **ECS phase graph: one phase per pipeline stage**. The existing `SchedulePhase` enum in the codebase would need extension.

---

## 4. Concrete Next Steps for Wave 17

### Step 1: Create IREE pipeline infrastructure in prism-ecs-ir

**Files to create:**
- `crates/prism-ecs-ir/src/flow.rs` — Flow dialect types: `DispatchRegion`, `FlowTensor`, `FlowPartitionOp`, dispatch region membership components
- `crates/prism-ecs-ir/src/stream.rs` — Stream dialect types: `StreamResource`, `StreamTask`, `StreamSchedule`, memory planning components
- `crates/prism-ecs-ir/src/hal.rs` — HAL dialect types: `HALExecutable`, `HALCommandBuffer`, `HALBufferAllocation`, device capability components

**Because:** The existing ECS-native IR has dialects for arith, func, scf, and linalg but nothing for the IREE pipeline stages. These three new modules define the ECS components that Flow/Stream/HAL systems operate on.

**Registration:** Each module must register its ops in the `OpRegistry` (following `register_linalg_ops` pattern in `linalg.rs`) and export component types for runtime queries.

### Step 2: Extend the prism-ecs-ir lowering pipeline

The existing `lowering.rs` has a scaffold `lower_matmul` that produces an `scf.for` loop. Wave 17 needs:

**17.1 — Flow dispatch formation:**
- Add `flow_partition` function that takes a linalg.matmul entity and produces `DispatchRegion` entities
- Each dispatch region entity carries: `FlowTensor` components (input/output shapes), a reference to the source linalg op, and backend capability constraints
- Test: partition a single matmul → verify exactly one dispatch region

**17.2 — Stream resource scheduling:**
- Add `stream_schedule` function that reads dispatch region entities + HAL target capability entities
- Produces `StreamTask` entities with memory allocation plans (buffer sizes, lifetimes)
- Produces `StreamSchedule` entity ordering the tasks
- Test: single dispatch region → verify one task with correct buffer allocation

**17.3 — HAL CPU backend:**
- Add `hal_cpu_lower` function that reads `StreamSchedule` entities
- Produces `HALExecutable` entities wrapping the existing `AccelerateRayonProgram`
- Wires into existing `cpu_runtime/lowering.rs` by mapping `HALExecutable` → `FusedGroup` → `AccelerateRayonProgram`
- Test: executable produces same numerical output as CPU reference

### Step 3: Add IREE pipeline systems to compute-core

**Files to create/update:**
- `compute-core/src/ecs/system/flow_systems.rs` — `FlowPartitionSystem` (ECS system that reads model IR entities, produces dispatch region entities)
- `compute-core/src/ecs/system/stream_systems.rs` — `StreamResourceSystem` + `StreamScheduleSystem`
- `compute-core/src/ecs/system/hal_systems.rs` — `HalCpuBackendSystem` (reads stream schedule, writes HAL executable entities)

**Register in `compile_session.rs`:**
- Add new phases to `SchedulePhase` if needed (e.g., `Flow`, `Stream`, `Hal` phases)
- Register the three new systems in `CompileSession::new()` or create a separate `IreePipelineSession`

### Step 4: Extend MlirExecutionContract for CPU codegen

The existing `MlirExecutionContract` has `MlirLoweringTarget::Cpu` but no lowering path. For Wave 17.3:

- Add `lower_to_cpu()` method that reads the contract's tensor geometry and transform schedule
- Emit a CPU program descriptor compatible with `CpuProgramOp`
- Wire into `cpu_runtime/lowering.rs` through a new `lower_contract_to_cpu_program` bridge function

### Step 5: Add Lift/Lower adapters between prism-ecs-ir and MlirExecutionContract

The existing `MlirExecutionContract` operates at a different level than the ECS-native IR:
- `MlirExecutionContract` has its own `MlirTensorType`, `TransformStep`, `QuantizationAttribute` types
- `prism-ecs-ir` has `Type`, `LinalgOp`, etc. as ECS entities

For Wave 17's end-to-end path, we need:
- `lift_ecs_ir_to_contract(world, linalg_op)` — reads a linalg.matmul entity and produces an `MlirExecutionContract`
- `lower_contract_to_flow_regions(contract)` — produces Flow dispatch region entities from the contract

### Step 6: Gate and test

The ADR gate for Wave 17:
```
linalg.matmul input
→ Prism ECS-native Flow → Stream → HAL (CPU)
runner output satisfies the same numerical policy as upstream IREE on the same input
→ normalized Flow, Stream, and HAL state matches
→ executable ABI and replay behavior are equivalent; container bytes need not be identical
```

**Test plan (in order):**
1. Unit test: linalg.matmul entity → Flow dispatch region (exactly one region, correct shapes)
2. Unit test: Flow dispatch region → Stream schedule (correct memory allocation, task ordering)
3. Unit test: Stream schedule → HAL executable (CPU program descriptor matches Accelerate+Rayon ops)
4. Integration test: Full pipeline on a small matmul (e.g., [4×4] @ [4×4]) → compare output to Accelerate `cblas_sgemm`
5. Serialization round-trip test for each new entity type

---

## 5. Dependency Map

```
prism-ecs-ir (crate)
├── flow.rs          [NEW] → DispatchRegion, FlowTensor components
├── stream.rs        [NEW] → StreamResource, StreamTask, StreamSchedule components
├── hal.rs           [NEW] → HALExecutable, HALCommandBuffer, HALBufferAllocation components
└── lowering.rs      [EXTEND] → add flow_partition(), stream_schedule(), hal_cpu_lower()

compute-core (crate)
├── ecs/system/
│   ├── flow_systems.rs      [NEW] → FlowPartitionSystem
│   ├── stream_systems.rs    [NEW] → StreamResourceSystem, StreamScheduleSystem
│   └── hal_systems.rs       [NEW] → HalCpuBackendSystem
├── ecs/mlir.rs              [EXTEND] → add lower_to_cpu(), lift/lower adapters
├── ecs/compile_session.rs   [EXTEND] → register IREE pipeline systems
└── ecs/cpu_runtime/
    └── lowering.rs          [EXTEND] → accept HALExecutable → AccelerateRayonProgram bridge
```

---

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No upstream IREE to compare against for differential testing | Cannot declare Wave 17 fully gated | Install IREE as a pinned binary or build from source in a conformance lane. The ADR permits this for Waves 18+. For Wave 17, use a CPU reference (Accelerate sgemm) as the oracle instead of upstream IREE. |
| MlirExecutionContract has no CPU codegen path | 17.3 blocked | Add `lower_to_cpu()` as a direct CPU program emitter, or bridge through AccelerateRayon. The contract already has all the tensor geometry. |
| ECS-native IR and MlirExecutionContract use different type systems | Lift/lower adapters needed | Write bridge functions as identified in Step 5. Can be thin wrappers since both describe same tensor shapes and operations. |
| Stream memory planning requires HAL capability entities | 17.2 depends on 17.1 completing first | The ADR already specifies sequential ordering. Ensure 17.1 produces all the entities 17.2 needs before starting 17.2. |
| Existing 151 tests may break | Regressions | Run full test suite after each sub-wave. The prism-ecs-ir crate is test-only with `prism-ecs-core` as its sole dependency, so risk is low. |

---

## 7. Recommendation

**Start with sub-wave 17.1 (Flow → dispatch region formation) immediately.** It has no dependency on CPU codegen or the existing `MlirExecutionContract`. The implementation path is:

1. Add `flow.rs` to `prism-ecs-ir` with `DispatchRegion` component and a `flow_partition()` function
2. Add `FlowPartitionSystem` to `compute-core/src/ecs/system/`
3. Write a test that creates a `linalg.matmul` entity and verifies flow partitioning produces correct dispatch regions
4. Verify 151 existing tests still pass

This builds on the existing ECS-native IR without touching any IREE-specific codegen or memory planning — establishing the Flow semantics cleanly before layering Stream and HAL on top.
