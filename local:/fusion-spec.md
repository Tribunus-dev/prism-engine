# Fusion Scheduling Engine — Design Spec

## Goal
Add a dataflow-graph-based fusion scheduling phase to the compiler pipeline between `LayoutResolver` and `ExecutionPlanner`.

## Pipeline change
```
Before:  PolicyResolver → LayoutResolver → ExecutionPlanner → ExecutionRegion[]
After:   PolicyResolver → LayoutResolver → FusionScheduler → ExecutionPlanner → ExecutionRegion[]
```

## Types

### DataflowOp (the atomic ops in the dataflow graph)
- `LoadWeight { tensor, codec, layout }` — quantized bytes to compute precision
- `LoadActivation { arena_slot, dtype }` — read arena buffer
- `MatMul { weight, activation, output }` — dense projection
- `FusedGateUp { activation, gate_weight, up_weight, output }` — SiLU/GELU gated MLP
- `FusedQkv { activation, q_weight, k_weight, v_weight, outputs: [q, k, v] }` — fused QKV
- `Add { lhs, rhs, output }` — residual
- `RmsNorm { input, output, weight }` — RMSNorm
- `RoPE { input, output, cos, sin }` — rotary position embedding
- `StoreActivation { arena_slot, input }` — write arena buffer

### DataflowGraph
- `ops: Vec<DataflowOp>` — all ops for one layer
- `edges: Vec<(usize, usize)>` — (producer_idx, consumer_idx)
- `buffers: HashMap<BufferId, BufferSpec>` — intermediate buffers
- `tensors: Vec<TensorRef>` — weights used
- `layer_id: String`

### FusedGroup
- `id: String`
- `body: Vec<DataflowOp>` — the fused sequence
- `lane: ExecutionLane` — target backend
- `tile_spec: Option<TileFamilySpec>` — tile shape if tiled
- `function_constants: FunctionConstantSet` — for kernel specialization
- `input_buffers: Vec<BufferId>` — external inputs to this group
- `output_buffers: Vec<BufferId>` — external outputs from this group

### FusionCapabilities (per backend)
- `lane: ExecutionLane`
- `supported_patterns: Vec<FusionPattern>` — e.g., "matmul+add", "gate+up+silu+matmul"
- `max_ops_per_group: usize`
- `requires_same_quantization: bool` — all weights must use same codec
- `requires_same_precision: bool` — all computations must use same dtype

### FusionPattern (pattern matching for fusion)
- `name: &'static str` — e.g., "nf4_matmul_silu_add"
- `pattern: Vec<DataflowOpKind>` — sequence of op kinds to match
- `resulting_template: &'static str` — kernel template name to use

### FusionScheduler
- `schedule(graph: &DataflowGraph, capabilities: &[FusionCapabilities]) -> Vec<FusedGroup>`
- Algorithm: walk ops in topological order, greedily fuse adjacent matching patterns per backend, insert boundary at incompatible transitions

## Backend capability tables

### Metal
- Supports: NF4 matmul, INT8 matmul, FP16 matmul, FusedGateUp, Add, RmsNorm
- Can fuse: matmul+add, gate_up+matmul (gated MLP), matmul+rmsnorm
- max_ops: 4 per group
- requires_same_quantization: true
- requires_same_precision: true

### ANE (planar engine)
- Supports: INT8 matmul, FP16 matmul, Add, RmsNorm, RoPE
- Can fuse: matmul+add, qkv (3 matmuls fused), gate_up+matmul
- max_ops: 4 per group
- requires_same_quantization: true
- requires_same_precision: true

### CPU (fallback)
- Supports: all at RawF32
- Can fuse: none (each op dispatches separately)
- max_ops: 1

## Integration points
- `ScheduledKernelOp` already has `function_constants` field — set from `FusedGroup.function_constants`
- `KernelSpecialization` already has codec/layout params — augment with fusion pattern identifier
- `ExecutionRegion` already groups ops — a `FusedGroup` maps to one or more `ScheduledKernelOp`s in one `ExecutionRegion`

## Constraints
- No fusion across layer boundaries
- No fusion across memory barrier points (hazard plan boundaries)
- Backend capability table is a registry, not hardcoded — new backends add entries
- Fusion is optional: scheduler must produce one FusedGroup per DataflowOp when no pattern matches
