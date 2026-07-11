# Fusion Compiler IR — Architecture Spec v2

## Pipeline

PolicyResolver → LayoutResolver → DataflowGraphBuilder → FusionScheduler → BackendLowering → ExecutionPlanner → ExecutionRegion → RegionEncoder

## Phase 1: Backend-Neutral Dataflow Graph (FusionCore)

File: `compute-core/src/execution_plan/fusion.rs`

### Core types

```rust
pub type DataflowBufferId = String;
pub type DataflowTensorRef = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowValue {
    pub id: DataflowBufferId,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub current_residency: ValueResidency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueResidency {
    CpuResident,
    GpuResident,
    AneResident,
    SharedUnified,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowEdge {
    pub producer: usize,            // index into DataflowGraph.nodes
    pub consumer: usize,
    pub value: DataflowBufferId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowGraph {
    pub nodes: Vec<DataflowNode>,
    pub edges: Vec<DataflowEdge>,
    pub values: HashMap<DataflowBufferId, DataflowValue>,
    pub layer_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowNode {
    pub id: usize,
    pub op: DataflowOp,
    pub inputs: Vec<DataflowBufferId>,
    pub outputs: Vec<DataflowBufferId>,
}
```

### DataflowOp

```rust
pub enum DataflowOp {
    LoadWeight {
        tensor: DataflowTensorRef,
        codec: CodecFamily,
        layout: PhysicalTileLayout,
    },
    Dequantize {
        input: DataflowBufferId,
        output_dtype: DType,
    },
    MatMul {
        lhs: DataflowBufferId,
        rhs: DataflowBufferId,
        output: DataflowBufferId,
        contract: MatMulContract,
    },
    RmsNorm {
        input: DataflowBufferId,
        weight: DataflowTensorRef,
        output: DataflowBufferId,
        epsilon: f32,
    },
    SiLU {
        input: DataflowBufferId,
        output: DataflowBufferId,
    },
    Gelu {
        input: DataflowBufferId,
        output: DataflowBufferId,
    },
    Mul {
        lhs: DataflowBufferId,
        rhs: DataflowBufferId,
        output: DataflowBufferId,
    },
    Add {
        lhs: DataflowBufferId,
        rhs: DataflowBufferId,
        output: DataflowBufferId,
    },
    ResidualAdd {
        residual: DataflowBufferId,
        update: DataflowBufferId,
        output: DataflowBufferId,
    },
    StoreActivation {
        slot: String,
        input: DataflowBufferId,
    },
    KvRead {
        slot: String,
        output: DataflowBufferId,
    },
    KvWrite {
        slot: String,
        input: DataflowBufferId,
    },
}
```

### FusedGroup

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedGroup {
    pub id: String,
    pub body: Vec<DataflowNode>,
    pub inputs: Vec<DataflowBufferId>,
    pub outputs: Vec<DataflowBufferId>,
    pub internal_values: Vec<DataflowBufferId>,
}
```

### Utilities

- `DataflowGraph::topological_sort() -> Vec<usize>` — returns node indices in topological order
- `DataflowGraph::producer_of(value_id) -> Option<usize>` — which node produces a given value
- `DataflowGraph::consumers_of(value_id) -> Vec<usize>` — which nodes consume a given value
- `DataflowGraph::materialization_boundaries() -> Vec<usize>` — node indices where values must be materialized (cross-layer aliasing, KV cache, etc.)
- `DataflowGraphBuilder::build_mlp(layer_config, resolved_layouts) -> DataflowGraph` — builds a Gemma decoder MLP graph from resolved layouts

### Gemma MLP test graph

The canonical test graph:
1. RMSNorm(activation) → normalized
2. Gate MatMul(normalized, gate_proj.weight) → gate_out
3. Up MatMul(normalized, up_proj.weight) → up_out
4. SiLU(gate_out) → gated
5. Mul(gated, up_out) → gated_up
6. Down MatMul(gated_up, down_proj.weight) → down_out
7. ResidualAdd(layer_input, down_out) → layer_output

## Phase 2: Backend Capabilities (BackendCapabilities)

File: `compute-core/src/execution_plan/backend_capability.rs`

### BackendLoweringTarget

```rust
pub enum BackendLoweringTarget {
    MetalFusedGpu,
    MetalTensorApi,
    AnePlanarEngine,
    CoreMlHighLevel,
    AccelerateRayonCpu,
}

pub enum BackendRole {
    ProductionHotPath,
    PressureFallback,
    DeterministicReference,
    ValidationProbe,
    LayoutConversion,
}

pub enum UnsupportedFusionReason {
    UnsupportedCodec(CodecFamily),
    UnsupportedOp(String),
    UnsupportedLayout(String),
    CrossLaneMaterialization,
    ExceedsMaxOps(usize),
    QuantMismatch,
    PrecisionMismatch,
    TileShapeMismatch,
    NestedParallelismRisk,
    HugeDenseMaterialization(u64),
}

pub struct FusionSupport {
    pub supported: bool,
    pub target: BackendLoweringTarget,
    pub reason: Option<UnsupportedFusionReason>,
    pub estimated_latency_us: Option<f64>,
    pub estimated_memory_bytes: Option<u64>,
    pub estimated_scratch_bytes: Option<u64>,
    pub estimated_power_class: PowerClass,
    pub precision_class: PrecisionClass,
    pub requires_materialization: bool,
    pub supports_in_place: bool,
    pub supports_aliasing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapability {
    pub target: BackendLoweringTarget,
    pub supported_codecs: Vec<CodecFamily>,
    pub supported_roles: Vec<BackendRole>,
    pub max_ops_per_group: usize,
    pub max_tile_elements: u64,
    pub rules: Vec<BackendFusionRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendFusionRule {
    pub pattern: Vec<String>,        // sequence of DataflowOp variant names
    pub requires_same_codec: bool,
    pub requires_same_precision: bool,
    pub max_tile_elements: Option<u64>,
    pub requires_same_lane: bool,
}

pub struct BackendCapabilityRegistry {
    entries: HashMap<BackendLoweringTarget, BackendCapability>,
}

impl BackendCapabilityRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, cap: BackendCapability);
    pub fn supports(&self, target: BackendLoweringTarget, group: &FusedGroup) -> FusionSupport;
    pub fn evaluate(&self, target: BackendLoweringTarget, group: &FusedGroup, role: BackendRole) -> FusionSupport;
    pub fn all_targets(&self) -> Vec<BackendLoweringTarget>;
}
```

### Backend capability rules

MetalFusedGpu:
- supported_codecs: [RawF32, Fp16, Int8, Nf4]
- max_ops_per_group: 4
- patterns: matmul+add, gate+up+silu+mul, matmul+residual
- NF4 dequant in shader

AnePlanarEngine:
- supported_codecs: [Fp16, Int8]
- max_ops_per_group: 4
- patterns: matmul+add, gate+up+silu+mul, matmul+residual
- REJECTS: Nf4, SymInt4, Ternary (UnsupportedCodec)

CoreMlHighLevel:
- supported_codecs: [Fp16, Int8]
- max_ops_per_group: 1 (no fusion)
- conservative MIL fallback path

AccelerateRayonCpu:
- supported_codecs: [RawF32, Fp16, Int8]
- supported_roles: [ProductionHotPath, PressureFallback, DeterministicReference, ValidationProbe, LayoutConversion]
- patterns: rmsnorm, matmul(rawf32/fp16), matmul+add, int8+matmul(custom tile)
- max_ops_per_group: 3
- REJECTS: Nf4, Ternary without custom kernel
- REJECTS: huge dense materialization (configurable threshold)

## Phase 3: Fusion Scheduler (FusionScheduler)

File: `compute-core/src/execution_plan/fusion_scheduler.rs`

```rust
pub struct FusionScheduler {
    pub capabilities: BackendCapabilityRegistry,
}

pub struct FusionPolicy {
    pub max_group_size: Option<usize>,
    pub allow_materialization: bool,
    pub forbid_cross_lane: bool,
    pub allow_research_fusions: bool,
    pub phase: ExecutionPhase,
}

pub struct FusionCandidate {
    pub group: FusedGroup,
    pub target: BackendLoweringTarget,
    pub support: FusionSupport,
    pub lowering_cost: Option<LoweringCost>,
}

pub struct FusionRejection {
    pub group_id: String,
    pub target: BackendLoweringTarget,
    pub reason: UnsupportedFusionReason,
}

pub struct FusionEvaluation {
    pub source_nodes: Vec<usize>,
    pub candidates: Vec<FusionCandidate>,
    pub selected: Option<FusionCandidate>,
    pub rejected: Vec<FusionRejection>,
}

pub struct FusionSchedule {
    pub groups: Vec<FusionEvaluation>,
    pub receipts: Vec<FusionScheduleReceipt>,
}

pub struct FusionSelectionPolicy {
    pub phase: ExecutionPhase,
    pub prefer_low_power: bool,
    pub prefer_low_latency: bool,
    pub prefer_low_memory: bool,
    pub allow_cpu_lane: bool,
    pub allow_pressure_fallback: bool,
    pub require_deterministic_reference: bool,
}
```

### Algorithm

1. Topologically walk graph
2. Per contiguous run of compatible nodes, try to grow candidate group
3. For each candidate group, ask every viable backend for support
4. Score candidates by policy (power, latency, memory)
5. Select best candidate, record rejections
6. Emit FusionSchedule with one FusionEvaluation per group
7. If no backend accepts, produce a rejection and use CPU fallback

## Phase 4: Metal Lowering (MetalFusionLowering)

File: `compute-core/src/metal_runtime/fusion_lowering.rs`

- Consumes FusedGroup → emits ScheduledKernelOp with KernelSpecializationKey
- Every codec/layout parameter in specialization key (group_size, tile_elements, metadata_layout, fusion_pattern_id)
- Function constants from PhysicalTileLayout + FusionPattern
- PSO cache key includes fusion_pattern_id
- NF4 g32 and g128 produce distinct keys
- FusedGateUpSiLU and DownResidual produce distinct fusion pattern IDs
- Unsupported fusion → error, not silent fallback

## Phase 5: ANE Planar Lowering (AnePlanarLowering)

File: `compute-core/src/ane_runtime/planar_lowering.rs`

```rust
pub struct PlanarProgramDescriptor {
    pub program_id: String,
    pub inputs: Vec<PlanarInput>,
    pub outputs: Vec<PlanarOutput>,
    pub ops: Vec<PlanarOp>,
    pub accumulation_dtype: DType,
    pub tile_policy: PlanarTilePolicy,
    pub iosurface_bindings: Vec<IOSurfaceBinding>,
}

pub enum PlanarOp {
    LoadMatrix { source: PlanarInputId, shape: Vec<usize> },
    MatMul { a: PlanarBufferId, b: PlanarBufferId, output: PlanarBufferId },
    ElementWise { op: PlanarElementwise, input: PlanarBufferId, output: PlanarBufferId },
    StoreMatrix { source: PlanarBufferId, dest: PlanarOutputId },
}
```

Initial support: FP16/INT8 matmul, matmul+add, matmul+elementwise, gate/up+silu
Rejects: NF4, SymInt4, Ternary, dynamic control flow, cross-lane IOSurface inside group

## Phase 6: Planner Integration

Files:
- `compute-core/src/execution_plan/planner.rs`
- `compute-core/src/execution_plan/model_plan.rs`
- `compute-core/src/execution_profile/mod.rs`

New pipeline from planner:
  resolved policy/layout → build layer dataflow graph → schedule fused groups → lower → to ExecutionRegion

Add `ExecutionMode`:
```rust
pub enum ExecutionMode {
    OpByOp,
    RegionBatched,
    MegakernelExperimental,
}
```

RegionBatched is opt-in. Add `fusion_mode: FusionMode` to ExecutionProfile.

## Phase 7: Fusion Receipts

File: `compute-core/src/execution_plan/receipts.rs`

```rust
pub struct DataflowGraphReceipt { pub node_count: usize, pub edge_count: usize, pub value_count: usize }
pub struct FusionScheduleReceipt { pub group_count: usize, pub evaluations: Vec<FusionEvaluationReceipt> }
pub struct FusionEvaluationReceipt { pub source_nodes: Vec<usize>, pub selected_target: BackendLoweringTarget, pub rejected: Vec<UnsupportedFusionReason>, pub materialization_saved_bytes: u64 }
pub struct BackendLoweringReceipt { pub target: BackendLoweringTarget, pub specialization_key_digest: String, pub fusion_pattern_id: String }
```

## Phase 8: Equivalence Tests

Tests:
1. dataflow_toposort_gemma_mlp
2. metal_fuses_nf4_gate_up_silu_when_supported
3. ane_rejects_nf4_fusion
4. ane_accepts_int8_bridge_projection
5. fusion_boundary_inserted_at_iosurface_materialization
6. region_batched_plan_matches_op_by_op_plan_shape
7. missing_backend_capability_fails_closed
8. unsupported_codec_fails_closed
9. cpu_capability_registered_as_first_class_backend
10. cpu_accepts_rawf32_rmsnorm_reference
11. cpu_rejects_nf4_without_custom_kernel
12. cpu_candidate_competes_in_fusion_evaluation

## CPU Backend (AccelerateRayonFusionBackend)

File: `compute-core/src/cpu_runtime/`

Types:
```rust
pub enum CpuProgramOp { VdspRmsNorm, VforceSilu, VdspMul, VdspAdd, CblasSgemm, CblasSgemv, CustomInt8TileGemv, CustomNf4TileGemv, LayoutConvert }
pub enum RayonStrategy { Disabled, ParallelRows { chunk_rows: usize }, ParallelOutputTiles { tile_count_per_task: usize }, ParallelInputBlocks { block_size: usize }, ParallelTensorBatch, ParallelLayerRange }
pub enum CpuThreadingPolicy { AccelerateOwnsThreads, RayonOwnsThreads, SingleThreadedForDeterminism, HybridOuterRayonInnerVector }

pub struct AccelerateRayonProgram {
    pub program_id: String,
    pub ops: Vec<CpuProgramOp>,
    pub parallel_strategy: RayonStrategy,
    pub accelerate_calls: Vec<AccelerateCallSpec>,
    pub scratch_plan: CpuScratchPlan,
    pub deterministic: bool,
}
```

Capability registration:
```rust
pub fn register_accelerate_rayon_capabilities(registry: &mut BackendCapabilityRegistry);
```
