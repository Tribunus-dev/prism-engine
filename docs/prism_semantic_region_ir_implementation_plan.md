# Prism Semantic Region IR
## Research-grounded implementation plan for `Tribunus-dev/prism-engine`

**Prepared:** 2026-07-24  
**Repository baseline reviewed:** `main` at commit `43b83e252bc365165063f11c0c2e5facbe264116`  
**Commit:** `Complete heterogeneous workload evaluation propagation and graph-canonical runtime wiring`  
**Primary goal:** introduce a persistent semantic sub-tensor abstraction without destabilizing the current heterogeneous execution, evidence, and ComputeImage work before the demo.

---

## 1. Executive decision

Implement **Semantic Region IR** as a new data-oriented compiler abstraction attached to a `LogicalTensorId`.

Do **not** extend the existing `RegionKind` in `crates/prism-ecs-ir/src/region.rs`. That file represents structural IR regions containing blocks and currently distinguishes graph regions from SSA control-flow regions. A semantic tensor region is a different concept: it identifies a stable subset of a logical tensor, not a nested control-flow container.

Do **not** add one genome axis per semantic region. The current `CandidateGenome` already spans global representation, packing, Metal geometry, decomposition, memory, fusion, Engram, runtime, and ANE choices. A region-local Cartesian product would make search size explode with the number of regions. Use a hierarchical search:

```text
model and graph semantics
→ semantic region discovery
→ partition normalization and verification
→ region-policy search
→ region-local representation/layout/placement candidates
→ adjacent-region coalescing
→ physical tiling and backend lowering
→ ComputeImage manifest and receipts
```

For the demo, implement only the first vertical slice:

```text
real mapped tensor
→ explicit or graph-derived semantic partition
→ verified, deterministic SemanticRegionPlan
→ region-level representation assignments
→ compile-verified receipt
→ human-readable and JSON output
```

Do not claim performance gains tomorrow unless a real measured execution produces them.

---

## 2. Precise research claim

The weak claims are already occupied by prior work:

> Different parts of one tensor can use different precision.

> A tensor can be partitioned into sub-tensors, blocks, channels, or sensitive regions.

> Semantics can influence quantization boundaries.

> Fine-grained assignments can be regularized into hardware-friendly blocks.

> Data layout and compute schedule can be optimized jointly.

The defensible Prism claim is narrower and more systems-oriented:

> **Prism introduces a persistent Semantic Region IR that preserves model-derived sub-tensor identity across representation search, physical layout, heterogeneous placement, residency planning, execution, and evidence-bearing ComputeImage construction.**

The proposed optimization problem is:

```text
For each semantic region, jointly select:

representation
codec
physical layout
execution target
residency policy
materialization boundaries
validation contract

subject to:

quality constraints
legality constraints
memory budget
latency budget
transfer budget
layout regularity
backend capabilities
evidence requirements
```

The likely novelty is not any one decision. It is the persistent identity and provenance chain across all of them.

---

## 3. Literature synthesis and what Prism should take from each paper

| Paper | Existing contribution | Prism implementation takeaway | What Prism must not claim |
|---|---|---|---|
| [SegQuant](https://arxiv.org/abs/2507.14811) | Uses graph-derived semantic segments and spatial heterogeneity to quantize segments independently. | Derive candidate boundaries from graph operations and model roles, not only fixed group sizes. Preserve the source operation that established each boundary. | Semantic segmentation for quantization is new. |
| [MoR: Mixture of Representations](https://arxiv.org/abs/2512.22804) | Dynamically chooses representations at tensor and sub-tensor granularity from numerical properties. | Separate region identity from the selected representation; representation is one realization of a region. | Per-block or per-sub-tensor representation choice is new. |
| [DRQ](https://doi.org/10.1109/ISCA45697.2020.00086) | Changes precision dynamically for sensitive feature-map regions. | Make sensitivity-derived regions a distinct provenance class and prevent them from masquerading as graph-semantic regions. | Region-level dynamic precision is new. |
| [Tender](https://arxiv.org/abs/2406.12930) | Decomposes activation tensors into range-compatible sub-tensors and co-designs runtime requantization. | Track conversion and partial-sum compatibility explicitly; score materialization and requantization costs. | Decomposed tensor quantization or sub-tensor scale groups are new. |
| [SemanticDialect](https://arxiv.org/abs/2603.02883) | Selects block-wise formats and shares a formatbook among semantically correlated tokens. | Allow behavioral or semantic correlation to propose region overlays in a later phase, but keep v0 static and disjoint. | Semantics-aware block format selection is new. |
| [PolyQ](https://arxiv.org/abs/2607.14618) | Assigns per-channel bit widths, then performs compile-time permutation and clustering into homogeneous blocks. | Add a mandatory layout-regularization/coalescing pass after region assignment; keep irregularity out of the hot path. | Compiler regularization of fine-grained bit assignments is new. |
| [AWQ](https://arxiv.org/abs/2306.00978) | Uses activation saliency to protect important channels while retaining hardware-friendly execution. | Region discovery can consume activation-aware saliency, but sensitivity is evidence, not semantic identity by itself. | Saliency-aware protection of channels is new. |
| [SpQR](https://arxiv.org/abs/2306.03078) | Separates outlier weights into a sparse higher-precision representation. | Support a dense base region plus an optional sparse exception sidecar in a later codec phase. | High-precision outlier side paths are new. |
| [SqueezeLLM](https://arxiv.org/abs/2306.07629) | Combines sensitivity-aware nonuniform quantization with dense-and-sparse decomposition. | Add sparse-exception cost and sensitivity evidence to the region policy, not just a bit-width field. | Sensitivity-based dense/sparse decomposition is new. |
| [Atom](https://arxiv.org/abs/2310.19102) | Combines mixed precision, channel reordering, fine-grained grouping, and serving kernels. | Search results must be executable, not merely numerically attractive. Include packing, reorder, and kernel availability in admission. | Fine-grained mixed precision combined with runtime kernels is new. |
| [UniSparse](https://arxiv.org/abs/2403.05802) | Separates logical sparse representation from physical memory layout and supports heterogeneous code generation. | Keep `SemanticRegionDescriptor` independent from `PhysicalRegionRealization`. A region ID must survive repacking. | Separating logical representation from physical sparse layout is new. |
| [Cypress: Task-Based Tensor Computations](https://arxiv.org/abs/2504.07004) | Separates logical task/tensor semantics from a mapping specification for execution placement and memory materialization. | Region semantics should feed a later mapping contract; do not encode GPU tile geometry in semantic identity. | First-class sub-tensors or explicit mapping specifications are new. |
| [TensorIR](https://arxiv.org/abs/2207.04296) | Makes tensor computations and blocks first-class for hardware-aware optimization. | Maintain a clean boundary between semantic-region analysis and target-specific tensor-program lowering. | First-class tensor regions or blocks in compiler IR are new. |
| [Relax](https://arxiv.org/abs/2311.02103) | Preserves symbolic and cross-level information across graph, tensor-program, and library-call boundaries. | Region identity and constraints should remain available after lowering rather than disappearing at the first target pass. | Cross-level information preservation is new. |
| [DCC](https://arxiv.org/abs/2511.15503) | Jointly optimizes tensor partition/data rearrangement and compute schedules for PIM. | Search must price data rearrangement and region-boundary conversions together with compute cost. | Joint partition, movement, and compute search is new. |
| [nncase](https://arxiv.org/abs/2512.21571) | Globally explores computation and data movement using e-graphs across heterogeneous storage. | Avoid greedy region decisions that prevent later global layout or placement optimization. Keep transformations replayable. | Global computation/data-movement exploration is new. |
| [TAPAS](https://arxiv.org/abs/2302.00247) | Folds search using repeated network substructures. | Share discovered region templates across repeated layers instead of independently searching every layer. | Repeated-subgraph search-space folding is new. |
| [Axe](https://arxiv.org/abs/2601.19092) | Maps logical coordinates to named physical axes across devices, memories, and threads. | Express region selectors in logical coordinates; lower them through a separate named-axis physical map. | Unified logical-to-physical axis mapping is new. |
| [Tensor Processing Primitives](https://arxiv.org/abs/2104.05755) | Defines a compact 2D tensor virtual ISA for portable backend implementations. | Coalesce semantic regions into backend-executable primitives instead of scheduling arbitrary tiny regions directly. | Sub-tensor primitives or a virtual tensor ISA are new. |
| [Ladder](https://www.usenix.org/conference/osdi24/presentation/wang-lei) | Makes evolving low-precision types first-class and jointly schedules storage, access, and conversion. | Treat representation and conversion as compiler-visible contracts; reject plans requiring unsupported hidden conversion paths. | Hardware-aware custom-type transformation is new. |

### Combined design rule from the literature

Prism should preserve two independent identities:

```text
Semantic identity:
why this subset exists and which model behavior or graph role it represents

Physical realization:
how that subset is packed, tiled, placed, transferred, and executed on a target
```

The compiler may change the physical realization many times. The semantic identity and provenance must remain stable.

---

## 4. Repository audit at commit `43b83e2`

### 4.1 Existing pieces that should be reused

The latest push already supplies most of the substrate:

| Existing surface | Current responsibility | Semantic-region use |
|---|---|---|
| `LogicalTensorId` in `prism-ecs-ir/src/evolution/foundation.rs` | Stable logical tensor identity. | Parent identity for every semantic region. |
| `TensorSensitivityReceipt` and `SensitivityAnalysisSystem` | Tensor-wide sensitivity and search-budget classification. | Basis for a new region-scoped sensitivity receipt. |
| `CandidateGenome` | Joint global search across representation, packing, geometry, decomposition, memory, fusion, Engram, runtime, and ANE. | Remains the global candidate. Region choices must be hierarchical, not a flat extension. |
| `DecompositionSystem` | Splits genome axes into search sub-problems. | Later coordinator for region-policy subproblems, after its axis representation is fixed. |
| `JointEvolutionSystem` | Multi-objective global evolution. | Outer search over region policy and global backend plan. |
| `CompilePlanRef`, `FormatAssignment`, `TileSizes` | ECS-native compile-plan components. | Add region-scoped assignment components parallel to tensor-scoped assignments. |
| `MappedTensorProbeContext` | Reproducible mapped SafeTensors probe for one tensor. | Extend with a selector and partition digest for bounded region probes. |
| `WorkloadThroughputEvidence` | Measured/profiled workload evidence with representation and tiling digest. | Add region-plan and materialization evidence fields. |
| `Qwen36TensorRole` / `TensorRole` plumbing | Initial architecture-aware tensor classification. | Starting point for model-specific semantic discovery. |
| `prism-spatial-ir` tiling, memory, execution plan | Target-side physical planning. | Destination for `SemanticRegionPlan → PhysicalRegionRealization`. |
| ComputeImage, assembly, forensic, observability, evidence schema | Artifact construction and evidence. | Persist region manifest, mapping digest, and receipt references. |

### 4.2 Naming collision that must be avoided

`crates/prism-ecs-ir/src/region.rs` defines structural control-flow regions:

```text
Region
→ ordered Blocks
→ Operations
```

Its `RegionKind` is `Graph` or `SSACFG`.

Do not add `RegionKind::Semantic`. That would conflate a control-flow ownership unit with a subset of tensor coordinates. Use `semantic_region.rs` and explicit types such as `SemanticRegionDescriptor`, `SemanticRegionPartition`, and `SemanticRegionPlan`.

### 4.3 Existing axis-count defect to fix first

`CandidateGenome` is documented as eight-dimensional, but the current struct contains nine axes after `ane_unit` was added. `SubProblem.active_axes` is a `u8`, and `DimensionVariance` tests still use eight entries.

Before adding any new search dimension:

```rust
#[repr(u8)]
pub enum GenomeAxis {
    Representation,
    Packing,
    MetalGeometry,
    Decomposition,
    Memory,
    Fusion,
    Engram,
    Runtime,
    AneUnit,
}

pub const GENOME_AXIS_COUNT: usize = 9;

#[derive(Clone, Copy, Default)]
pub struct GenomeAxisSet(u16);
```

Replace every hard-coded dimension count and `u8` axis mask with `GenomeAxisSet`.

Do not add `SemanticRegion` as axis ten. Region policy is an object containing a bounded collection of assignments, not one scalar axis.

### 4.4 Evidence discipline from the latest push

The commit immediately follows work that removed fabricated inference and Metal receipts and rejected non-authoritative legacy receipts. Semantic-region output must follow that direction.

Tomorrow’s result should be classified as:

```text
region boundaries:
repository-verified or explicit-user-supplied

partition legality:
compile-verified

representation assignment:
planned / compile-verified

quality:
unproven unless a real behavioral probe ran

latency and throughput:
unproven unless a real backend execution was measured
```

---

## 5. Canonical terminology

Use these terms consistently.

| Term | Definition |
|---|---|
| `LogicalTensor` | Existing model-level tensor identity independent of physical packing. |
| `SemanticRegion` | Stable logical subset of a tensor with shared semantic, behavioral, numerical, or hardware-relevant properties. |
| `SemanticRegionPartition` | Verified set of disjoint regions covering all or a declared subset of one logical tensor. |
| `RegionSelector` | Logical-coordinate description of the subset. |
| `RegionOrigin` | Evidence class explaining how the region boundary was discovered. |
| `RegionRole` | Model-level role such as query, key, value, router, expert, gate, up projection, outlier sidecar, or generic channel group. |
| `RegionPolicy` | Candidate choices and constraints for representations, codecs, placement, and residency. |
| `SemanticRegionPlan` | Admitted set of region assignments before target-specific lowering. |
| `PhysicalRegionRealization` | Backend-specific mapping from semantic regions to packed blocks, tiles, buffers, queues, and device locations. |
| `RegionReceipt` | Evidence record explaining discovery, admission, lowering, or execution. |

Avoid “tile” in semantic-region APIs. A semantic region can lower to many tiles, and multiple semantic regions can be co-packed into one physical block.

---

## 6. IR design

Create `crates/prism-ecs-ir/src/semantic_region.rs`.

### 6.1 Core identity and selectors

```rust
use serde::{Deserialize, Serialize};
use crate::evolution::foundation::LogicalTensorId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticRegionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionSelector {
    WholeTensor,
    AxisSpan {
        axis: u32,
        start: u64,
        end: u64,
    },
    Rect {
        offsets: Vec<u64>,
        extents: Vec<u64>,
    },
}
```

For the first implementation, accept only `WholeTensor` and one-dimensional `AxisSpan`. Keep `Rect` serialized but gate it behind validation until all consumers support it.

Do not support arbitrary masks, overlapping overlays, token-dynamic regions, or index lists in v0.

### 6.2 Origin and role

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionOrigin {
    GraphDerived {
        operation: String,
        source_value: String,
    },
    ArchitectureDerived {
        model_family: String,
        rule: String,
    },
    SensitivityDerived {
        probe_digest: String,
    },
    Explicit {
        source: String,
    },
    Hybrid {
        sources: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionRole {
    QueryProjection,
    KeyProjection,
    ValueProjection,
    AttentionHeadGroup { first: u32, count: u32 },
    Router,
    RoutedExpertGroup { first: u32, count: u32 },
    SharedExpert,
    GateProjection,
    UpProjection,
    DownProjection,
    EmbeddingShard,
    OutlierSidecar,
    SensitiveChannelGroup,
    Generic { label: String },
}
```

### 6.3 Constraints and descriptor

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConstraints {
    pub allowed_formats: Vec<String>,
    pub allowed_codecs: Vec<String>,
    pub preferred_lanes: Vec<String>,
    pub max_error: Option<f64>,
    pub alignment_elements: u64,
    pub must_be_contiguous: bool,
    pub may_materialize: bool,
    pub may_reorder: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionDescriptor {
    pub id: SemanticRegionId,
    pub parent: LogicalTensorId,
    pub selector: RegionSelector,
    pub role: RegionRole,
    pub origin: RegionOrigin,
    pub constraints: RegionConstraints,
    pub provenance_refs: Vec<String>,
}
```

### 6.4 Partition and plan

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionPartition {
    pub parent: LogicalTensorId,
    pub parent_shape: Vec<u64>,
    pub regions: Vec<SemanticRegionDescriptor>,
    pub exhaustive: bool,
    pub disjoint: bool,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRepresentationAssignment {
    pub region: SemanticRegionId,
    pub representation: String,
    pub codec: Option<String>,
    pub preferred_lane: Option<String>,
    pub residency: Option<String>,
    pub assignment_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionPlan {
    pub partition: SemanticRegionPartition,
    pub assignments: Vec<RegionRepresentationAssignment>,
    pub compile_verified: bool,
    pub plan_digest: String,
}
```

### 6.5 ECS components

Use the descriptor types as serializable values and add small ECS components:

```rust
pub struct SemanticRegionMarker;
pub struct SemanticRegionIdComp(pub SemanticRegionId);
pub struct SemanticRegionParent(pub LogicalTensorId);
pub struct SemanticRegionSelectorComp(pub RegionSelector);
pub struct SemanticRegionRoleComp(pub RegionRole);
pub struct SemanticRegionOriginComp(pub RegionOrigin);
pub struct SemanticRegionConstraintsComp(pub RegionConstraints);
pub struct SemanticRegionPlanRef(pub prism_ecs_core::Entity);
```

Do not put one large mutable plan blob on every region entity. Store one partition/plan entity and use references.

---

## 7. Partition verifier

Implement `SemanticRegionPartition::verify()` with fail-closed behavior.

The verifier must check:

| Invariant | Required behavior |
|---|---|
| Parent identity | Every descriptor references the partition’s `LogicalTensorId`. |
| Bounds | Every span is inside `parent_shape`. |
| Nonempty | `end > start`; extents are nonzero. |
| Stable IDs | IDs are unique and deterministically derived. |
| Disjointness | v0 regions cannot overlap. |
| Exhaustiveness | When `exhaustive=true`, the selected axis is fully covered without gaps. |
| Ordering | Regions use canonical axis/start/end ordering before hashing. |
| Constraint consistency | A region must have at least one allowed representation or inherit an explicit default. |
| Provenance | Every non-generic semantic role must have graph, architecture, or explicit provenance. |
| Digest | Canonical serialization regenerates the stored digest. |

Use the repository’s existing digest conventions if available. Otherwise use BLAKE3 over canonical JSON or a fixed binary encoding. Never hash `Debug` output.

Stable ID format:

```text
sr:<tensor-digest>:axis:<axis>:<start>:<end>:<role-digest>
```

---

## 8. Discovery architecture

Create a trait in `prism-ecs-compile`:

```rust
pub trait SemanticRegionDiscoverer: Send + Sync {
    fn discover(
        &self,
        tensor: &LogicalTensorDescriptor,
        graph: &ModelGraph,
        manifest: &ModelManifest,
    ) -> Result<Vec<SemanticRegionDescriptor>, SemanticRegionError>;
}
```

Run discoverers in confidence order.

### 8.1 Graph-explicit discoverer

Recognize boundaries established by operations such as:

```text
split
chunk
slice
concat
stack
fused projection unpacking
expert-bank indexing
router/expert dispatch
```

This is the SegQuant-aligned path. Record operation IDs and source values in `RegionOrigin::GraphDerived`.

### 8.2 Model-family discoverer

Add targeted rules for verified model families.

For a fused QKV projection:

```text
Q region
K region
V region
```

Derive sizes from `num_attention_heads`, `num_key_value_heads`, and `head_dim`; do not assume equal thirds when grouped-query attention is present.

For a fused gate/up projection:

```text
gate region
up region
```

Derive the split from model configuration or the graph contract.

For MoE:

```text
router
shared experts
routed expert groups
```

If experts are already independent tensors, keep them as separate tensors rather than fabricating sub-tensor regions.

The current Qwen 3.6 classifier is name-based and defaults layer numbers and shapes. Upgrade it to parse layer/expert indices and consume real tensor shapes before treating its output as semantic evidence.

### 8.3 Sensitivity discoverer

Add later, after the static path works.

It should segment channels or blocks according to real probe results and emit `RegionOrigin::SensitivityDerived`.

Do not label sensitivity clusters as query/key/value or other semantic roles unless graph evidence confirms that role.

### 8.4 Explicit spec discoverer

Support a versioned JSON file for demos and research:

```json
{
  "schema": "prism.semantic-regions.v1",
  "tensor": "model.layers.0.self_attn.qkv_proj.weight",
  "shape": [6144, 2048],
  "regions": [
    {
      "role": "query_projection",
      "axis": 0,
      "start": 0,
      "end": 4096
    },
    {
      "role": "key_projection",
      "axis": 0,
      "start": 4096,
      "end": 5120
    },
    {
      "role": "value_projection",
      "axis": 0,
      "start": 5120,
      "end": 6144
    }
  ]
}
```

The explicit path is appropriate for tomorrow’s demo if the current graph does not preserve a fused split operation. The receipt must identify it as explicit architecture evidence, not automatic discovery.

---

## 9. Region sensitivity and behavioral probes

Do not replace `MappedTensorProbeContext`. Add a parallel context:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedTensorRegionProbeContext {
    pub model_dir: PathBuf,
    pub tensor_name: String,
    pub selector: RegionSelector,
    pub partition_digest: String,
    pub region_id: SemanticRegionId,
}
```

Add:

```rust
pub struct RegionSensitivityReceipt {
    pub tensor_id: LogicalTensorId,
    pub region_id: SemanticRegionId,
    pub selector_digest: String,
    pub format_variance: f64,
    pub operation_variance: f64,
    pub geometry_variance: f64,
    pub memory_variance: f64,
    pub probe_valid: bool,
    pub evidence_source: String,
}
```

The provider should map the source tensor once and expose a bounded logical view. Avoid copying an entire tensor merely to probe one region. If the SafeTensors provider cannot express a strided view, the receipt must record materialized bytes.

Cache key:

```text
model digest
tensor identity
region selector digest
candidate representation
probe version
calibration corpus digest
```

For the first implementation, support contiguous row or channel ranges only.

---

## 10. Search architecture

### 10.1 Fix genome-axis representation first

Replace `u8 active_axes`, hard-coded “8-dimensional” comments, and eight-entry variance assumptions.

This is a prerequisite because the current struct already contains nine axes.

### 10.2 Do not flatten region choices into `CandidateGenome`

Use two levels:

```text
GlobalGenome
    representation family
    backend and lane policy
    packing family
    memory policy
    fusion and runtime policy

RegionalPlan
    partition template
    per-region representation
    per-region codec
    per-region preferred lane
    per-region residency
```

### 10.3 Regional search candidate

```rust
pub struct RegionalCandidate {
    pub global: CandidateGenome,
    pub partition_digest: String,
    pub assignments: Vec<RegionRepresentationAssignment>,
    pub regularization: RegionRegularizationPolicy,
}
```

### 10.4 Bounded search strategy

Use this sequence:

```text
1. Discover and verify regions.
2. Build a small candidate palette per region.
3. Prune candidates violating hard quality or backend constraints.
4. Coalesce equivalent repeated-layer region templates.
5. Search region assignments under global budgets.
6. Regularize adjacent compatible regions.
7. Lower and evaluate the complete executable plan.
```

Do not independently evolve every region in every layer. Use a `RegionTemplateId` so repeated decoder layers share search results unless evidence shows a layer is exceptional. This applies the TAPAS insight and avoids model-depth-linear duplication.

### 10.5 Objectives

Add these objectives or constraints:

```text
quality loss
measured latency
measured throughput
packed bytes
materialized bytes
conversion bytes
region boundary count
kernel variant count
layout fragmentation
cross-lane transfer bytes
compile time
receipt completeness
```

The regularity penalty is essential. A numerically optimal plan with hundreds of tiny format boundaries is not an executable win.

---

## 11. Compile-plan integration

Add parallel components in `evolution/compile_plan.rs`:

```rust
pub struct RegionFormatAssignment {
    pub region: SemanticRegionId,
    pub format: TensorFormat,
}

pub struct RegionCodecAssignment {
    pub region: SemanticRegionId,
    pub codec: String,
}

pub struct RegionPlacementAssignment {
    pub region: SemanticRegionId,
    pub lane: String,
    pub residency: String,
}

pub struct SemanticRegionPlanComp(pub SemanticRegionPlan);
```

Keep `FormatAssignment(pub TensorFormat)` for whole-tensor fallback.

Resolution order:

```text
region assignment
→ tensor assignment
→ compile default
```

A region plan is invalid if two assignments target the same region and property without an explicit priority rule.

---

## 12. Physical lowering

Add a physical mapping layer rather than teaching semantic regions about backend tiles.

Proposed type in `prism-spatial-ir`:

```rust
pub struct PhysicalRegionRealization {
    pub semantic_region: SemanticRegionId,
    pub logical_selector_digest: String,
    pub packed_buffer: String,
    pub byte_ranges: Vec<std::ops::Range<u64>>,
    pub tile_ids: Vec<String>,
    pub execution_lane: String,
    pub residency_class: String,
    pub materialized_bytes: u64,
    pub conversion_ops: Vec<String>,
    pub realization_digest: String,
}
```

### 12.1 Lowering sequence

```text
SemanticRegionPlan
→ region compatibility analysis
→ adjacent assignment coalescing
→ logical-to-physical axis mapping
→ packing and sparse-sidecar selection
→ target tile selection
→ buffer assignment
→ conversion/materialization insertion
→ backend executable views
```

### 12.2 Required guarantees

The lowering verifier must prove:

```text
every logical element is represented exactly once,
unless an explicit overlay/sidecar contract says otherwise

the semantic selector maps to the declared physical byte ranges

all inserted conversions are represented in the execution graph

all materializations contribute to the cost model

backend kernels support the resulting homogeneous blocks

region-to-physical provenance survives into the artifact
```

Use PolyQ’s general lesson: irregular decisions should be clustered into homogeneous execution blocks at compile time. Use UniSparse and Axe’s lesson: logical identity must remain distinct from physical layout.

---

## 13. ComputeImage integration

Do not change the binary payload ABI before the demo unless the existing format has a versioned optional metadata section.

The preferred long-term manifest is:

```rust
pub struct SemanticRegionManifest {
    pub schema_version: u32,
    pub model_digest: String,
    pub partitions: Vec<SemanticRegionPartition>,
    pub plans: Vec<SemanticRegionPlan>,
    pub realizations: Vec<PhysicalRegionRealization>,
    pub receipt_refs: Vec<String>,
    pub manifest_digest: String,
}
```

### Safe demo path

Attach a versioned JSON sidecar or optional metadata record:

```text
model.cimage
model.semantic-regions.json
model.semantic-regions.receipt.json
```

Seal the sidecar digest into the demo output if the current ComputeImage envelope supports extension metadata. Otherwise state clearly that the demo emits a compile-plan sidecar, not a production ABI addition.

### Production path

The ComputeImage must eventually answer:

```text
Which semantic regions exist?
Why do they exist?
Which representations were selected?
How were they packed?
Where do they execute?
Which conversions were inserted?
Which evidence supports each choice?
Which claims are measured versus compile-verified?
```

---

## 14. Evidence and receipts

Introduce four receipt categories.

### Discovery receipt

```text
parent tensor
region ID
selector
semantic role
origin
source graph operation or model rule
confidence/evidence class
partition digest
```

### Admission receipt

```text
allowed representation set
selected representation
quality constraint
backend legality
budget impact
rejected alternatives
plan digest
```

### Lowering receipt

```text
semantic region ID
physical buffers and tiles
reordering
materialization
conversion operations
backend kernel contract
realization digest
```

### Execution receipt

```text
artifact identity
region-plan digest
execution fingerprint
lane/provider
measured latency/throughput
input/calibration digest
validation result
```

Extend `WorkloadThroughputEvidence` with optional fields:

```rust
pub semantic_region_plan_digest: Option<String>,
pub region_count: Option<u32>,
pub materialized_bytes: Option<u64>,
pub conversion_bytes: Option<u64>,
pub layout_fragmentation_score: Option<f64>,
pub region_execution_fingerprint: Option<String>,
```

A profile is not valid measured evidence merely because these fields exist. Preserve the current requirement that real measurements and valid execution fingerprints are present.

---

## 15. Tomorrow-morning implementation slice

This slice is intentionally bounded. It produces something demonstrable without modifying the evolutionary search, runtime scheduler, backend kernels, or stable ComputeImage ABI.

### Milestone D0 — protect the current branch

```bash
git switch -c feat/semantic-region-ir-demo
git rev-parse HEAD
cargo test -p prism-ecs-ir -p prism-ecs-compile
```

Record the baseline commit in the demo receipt.

### Milestone D1 — fix genome-axis bookkeeping

Change:

```text
hard-coded 8 dimensions
u8 active_axes
eight-value DimensionVariance tests
```

to:

```text
GenomeAxis enum
GENOME_AXIS_COUNT
GenomeAxisSet(u16)
tests derived from GENOME_AXIS_COUNT
```

Do not change search behavior.

Acceptance:

```bash
cargo test -p prism-ecs-ir evolution
```

passes and every current axis can be represented.

### Milestone D2 — add Semantic Region IR

Create:

```text
crates/prism-ecs-ir/src/semantic_region.rs
```

Export it from the crate.

Implement:

```text
SemanticRegionId
RegionSelector
RegionOrigin
RegionRole
RegionConstraints
SemanticRegionDescriptor
SemanticRegionPartition
RegionRepresentationAssignment
SemanticRegionPlan
verification
canonical digest
```

Tests:

```text
accept exhaustive Q/K/V partition
reject overlap
reject gap when exhaustive
reject out-of-bounds region
reject duplicate ID
produce stable digest regardless of input ordering
round-trip serde
```

Acceptance:

```bash
cargo test -p prism-ecs-ir semantic_region
```

passes.

### Milestone D3 — add explicit spec loader

Create:

```text
crates/prism-ecs-compile/src/semantic_region_spec.rs
```

It loads `prism.semantic-regions.v1`, validates tensor identity and shape, constructs the partition, and emits a discovery receipt.

Add one checked-in example:

```text
examples/semantic-regions/qkv-gqa.example.json
```

Do not name it after a real model unless its dimensions come from that model’s verified configuration.

### Milestone D4 — add a region-plan demo example

Create:

```text
crates/prism-ecs-compile/examples/semantic_region_plan.rs
```

Suggested interface:

```bash
cargo run -p prism-ecs-compile \
  --example semantic_region_plan -- \
  --model-dir "$MODEL_DIR" \
  --tensor "$TENSOR_NAME" \
  --spec examples/semantic-regions/qkv-gqa.example.json \
  --assign query_projection=fp16 \
  --assign key_projection=int8 \
  --assign value_projection=int8 \
  --json-out /tmp/semantic-region-plan.json
```

The example must:

```text
verify the tensor exists in SafeTensors
read the real tensor shape
load the explicit semantic spec
verify the partition
apply bounded representation assignments
produce a deterministic plan digest
emit human-readable output
emit machine-readable JSON
emit a compile-verified receipt
```

It must not claim quality or performance.

Expected demo output:

```text
Tensor: model.layers.0.self_attn.qkv_proj.weight
Shape: [6144, 2048]
Semantic partition: 3 regions
Coverage: 100%
Overlap: none
Plan:
  query_projection [0,4096) -> fp16
  key_projection   [4096,5120) -> int8
  value_projection [5120,6144) -> int8
Plan digest: ...
Evidence:
  tensor source: repository-backed mapped checkpoint
  boundaries: explicit architecture contract
  legality: compile-verified
  numerical quality: unproven
  execution performance: unmeasured
```

### Milestone D5 — attach observability

Add a JSON receipt with:

```text
commit SHA
model/tensor source
tensor digest if available
partition digest
plan digest
region descriptors
claim classes
unproven fields
```

Add one sentence to demo material:

> Prism can now preserve semantic sub-tensor identity separately from the physical layout that a backend will eventually select.

### Milestone D6 — final gate

Run:

```bash
cargo fmt --all -- --check
cargo clippy -p prism-ecs-ir -p prism-ecs-compile --all-targets -- -D warnings
cargo test -p prism-ecs-ir
cargo test -p prism-ecs-compile
git diff --check
```

Do not merge backend, scheduler, or stable ABI changes into the demo branch.

---

## 16. Milestones after the demo

### Milestone 1 — static semantic discovery

**Goal:** replace explicit specs for supported model families.

Implement graph-explicit and architecture-derived discoverers. Begin with fused QKV and fused gate/up projections. Upgrade Qwen tensor-role parsing to recover real layer/expert indices and actual shapes.

**Exit criteria:**

```text
the same model produces the same partition digest across runs
every non-generic role has provenance
unsupported models fall back to whole-tensor regions
no inferred boundary is accepted without shape validation
```

### Milestone 2 — region-aware sensitivity

**Goal:** perform real, bounded probes on region views.

Implement `MappedTensorRegionProbeContext`, region view extraction, region cache keys, and `RegionSensitivityReceipt`.

**Exit criteria:**

```text
probe reads only the declared region or records materialization bytes
cache keys include selector and calibration digests
all quality claims identify the probe and source tensor
whole-tensor results remain reproducible
```

### Milestone 3 — hierarchical region search

**Goal:** choose per-region representations without flattening the global genome.

Implement regional candidates, template sharing, candidate pruning, and global budget constraints.

**Exit criteria:**

```text
search supports whole-tensor baseline
search supports fixed-group baseline
search supports semantic-only regions
search supports sensitivity-only regions
search supports hybrid regions
region count and kernel variants remain bounded
```

### Milestone 4 — layout regularization and physical realization

**Goal:** turn irregular region policies into backend-executable blocks.

Implement adjacency coalescing, permutation legality, named-axis mapping, buffer assignment, and conversion accounting.

**Exit criteria:**

```text
no hidden runtime permutation
every materialization appears in the plan and cost model
region-to-byte provenance is reversible
all backend blocks are homogeneous for their kernel contract
```

### Milestone 5 — ComputeImage manifest

**Goal:** preserve semantic-region identity in the executable artifact.

Add a versioned manifest, digest sealing, realization map, and receipt references.

**Exit criteria:**

```text
artifact round-trip retains region identities
manifest digest participates in artifact identity
old artifacts remain readable
new runtimes can ignore optional semantic metadata safely
```

### Milestone 6 — backend integration

**Goal:** execute coalesced region plans on real targets.

Start with the backend that already has the strongest measured path. Do not attempt Metal, ANE, ROCm, and Intel simultaneously.

Implement:

```text
homogeneous mixed-format blocks
explicit conversion operations
region-aware packed buffer views
execution fingerprint including plan digest
measured receipt propagation
```

**Exit criteria:**

```text
end-to-end output matches baseline within declared error
latency and throughput are measured
copy and conversion bytes are reported
failure falls back to an admitted whole-tensor plan
```

### Milestone 7 — scheduler and residency

**Goal:** allow placement and residency decisions while avoiding micro-task explosion.

The scheduler should consume coalesced physical execution views, not one work item per semantic region.

Implement region-aware residency only when it changes an actual placement or transfer boundary.

**Exit criteria:**

```text
scheduler queue size is bounded independently of raw region count
cross-lane transfers are explicit
residency ownership is deterministic
receipts name the physical realization, not only the semantic region
```

### Milestone 8 — evaluation and paper-quality study

**Goal:** determine whether the abstraction improves a real frontier.

Baselines:

```text
uniform per-tensor
fixed group-wise
sensitivity-only
SegQuant-style graph semantic
MoR-style numerical block selection
PolyQ-style channel assignment plus regularization
Prism semantic-only
Prism semantic + sensitivity
Prism semantic + sensitivity + placement
```

Metrics:

```text
perplexity or task quality
model size
prefill latency
decode latency
tokens/second
energy/token where available
compile time
search time
materialization bytes
conversion bytes
kernel count
layout fragmentation
region count
fallback frequency
receipt completeness
```

Ablations:

```text
remove semantic provenance
remove sensitivity
remove layout regularization
remove joint placement
remove region-template sharing
remove sparse sidecar
```

---

## 17. Required test matrix

### IR tests

```text
selector bounds
disjointness
coverage
ordering
stable identity
stable digest
serde compatibility
unknown enum/version rejection
```

### Discovery tests

```text
fused QKV with MHA
fused QKV with GQA
gate/up split
separate Q/K/V tensors
MoE experts as independent tensors
unsupported tensor fallback
ambiguous name rejection
```

### Probe tests

```text
contiguous row range
contiguous channel range
strided unsupported path
materialization accounting
cache separation by region
calibration digest separation
```

### Search tests

```text
whole-tensor baseline remains legal
hard quality limit rejects candidate
backend capability rejects format
region-count budget enforced
coalescing reduces block count
template sharing is deterministic
```

### Lowering tests

```text
logical-to-physical element conservation
byte-range non-overlap
sparse sidecar explicit overlay
conversion insertion
materialization accounting
backend block homogeneity
```

### Artifact tests

```text
old ComputeImage compatibility
new manifest round-trip
digest tamper detection
receipt reference resolution
region-plan fingerprint propagation
```

### Execution tests

```text
baseline equivalence
measured evidence only after real run
fallback path
provider failure receipt
cross-lane transfer accounting
no hidden copy
```

---

## 18. Research and engineering risks

### Search-space explosion

Per-region independent evolution is not viable. Use region templates, bounded palettes, hard region-count budgets, and post-search coalescing.

### Hardware inefficiency

Fine-grained semantic boundaries can destroy dense kernels. The semantic plan must be regularized before execution. A region is not automatically a dispatch unit.

### False semantic claims

Name matching is weak evidence. Treat it as fallback classification. Prefer graph operations, architecture configuration, and verified manifests.

### Dynamic behavior

Token-dependent or activation-dependent regions can create unstable layouts and dispatch overhead. Keep v0 static. Add dynamic overlays only after static execution is measured.

### ABI instability

Avoid changing the stable ComputeImage payload before the region manifest is validated. Begin with optional metadata or a sidecar.

### Evidence inflation

A verified partition does not prove accuracy or speed. Keep compile verification, behavioral validation, and measured execution as separate receipt states.

### Terminology collision

Structural IR regions and semantic tensor regions must remain separate modules and types.

### Existing genome-axis drift

Fix the current eight-versus-nine axis mismatch before extending search infrastructure.

---

## 19. Non-goals for the initial implementation

Do not implement:

```text
arbitrary boolean masks
overlapping semantic overlays
token-dynamic region boundaries
runtime dispatch per semantic region
full per-region evolutionary Cartesian search
all backends at once
stable ABI changes before compatibility tests
automatic performance claims
a new quantization algorithm
a new GPU kernel family
```

The initial contribution is the IR, identity, verifier, plan, and evidence path.

---

## 20. Definition of done

The semantic-region architecture is complete when:

```text
a region has stable identity independent of layout

a partition is verified, deterministic, and reproducible

graph-, architecture-, sensitivity-, and explicit-derived origins are distinguishable

regional assignments inherit from tensor defaults and can override them legally

search is hierarchical and bounded

physical lowering records every reorder, conversion, and materialization

ComputeImage preserves semantic-to-physical provenance

execution fingerprints include the region-plan digest

measured claims require real execution

whole-tensor fallback remains available and tested

the implementation beats or improves a measured quality/latency/memory frontier,
not merely a compile-time visualization
```

---

## 21. File-by-file implementation map

| File or module | Tomorrow | Full implementation |
|---|---|---|
| `prism-ecs-ir/src/semantic_region.rs` | Add core types, verification, digest, tests. | Add ECS helpers, versioning, overlays if justified. |
| `prism-ecs-ir/src/lib.rs` | Export module. | Preserve public API stability. |
| `evolution/foundation.rs` | Fix axis count and mask representation. | Add region template IDs, not flat region axes. |
| `evolution/sensitivity.rs` | No behavior change beyond axis-count fix. | Add region-scoped receipts and analysis. |
| `evolution/decompose.rs` | Migrate to `GenomeAxisSet`. | Add regional subproblem coordination. |
| `evolution/joint.rs` | No demo integration. | Add hierarchical regional search. |
| `evolution/compile_plan.rs` | Add optional regional assignment components if needed by demo. | Full region format/codec/placement plan. |
| `prism-ecs-compile/src/semantic_region_spec.rs` | Add explicit spec loader. | Add model/graph discoverer registry. |
| `prism-ecs-compile/src/evaluator.rs` | Optionally validate mapped tensor and shape. | Add mapped region probes and cache. |
| `prism-ecs-compile/src/model_manifest.rs` | Expose enough config for verified explicit demo. | Add model-family semantic discovery contracts. |
| `prism-ecs-compile/src/qwen3_6_moe.rs` | Do not rely on current placeholder layer/shape values. | Parse real roles, layers, experts, and shapes. |
| `prism-ecs-compile/src/workload_search.rs` | No demo behavior change. | Add region-plan evidence and materialization metrics. |
| `prism-ecs-compile/src/assembly.rs` / `cimage.rs` | Sidecar only. | Versioned manifest and sealed digest. |
| `prism-spatial-ir` | No demo integration. | Physical region realization and verifier. |
| `prism-ecs-runtime` | No demo integration. | Execute coalesced views and propagate receipts. |
| `evidence-schema` | Add demo receipt schema only if low-risk. | Canonical discovery/admission/lowering/execution schemas. |
| `docs` | Add research note and explicitly scoped claim. | Add architecture visualization after execution exists. |

---

## 22. Suggested demo narrative

Use one tensor and one question:

> Must every value inside a tensor have the same compiled life?

Show:

```text
one logical fused tensor
→ three verified semantic regions
→ independent representation contracts
→ one deterministic region-plan digest
→ explicit evidence boundary
```

Then explain:

> Prism does not schedule arbitrary fragments. It preserves semantic identity, regularizes the plan into hardware-friendly physical blocks, and only then lowers it to a target.

Do not present projected speedups. The demo is successful if the audience understands the new compiler boundary and sees a real tensor-backed, verified plan.

---

## 23. Source index

1. SegQuant — https://arxiv.org/abs/2507.14811  
2. MoR — https://arxiv.org/abs/2512.22804  
3. DRQ — https://doi.org/10.1109/ISCA45697.2020.00086  
4. Tender — https://arxiv.org/abs/2406.12930  
5. SemanticDialect — https://arxiv.org/abs/2603.02883  
6. PolyQ — https://arxiv.org/abs/2607.14618  
7. AWQ — https://arxiv.org/abs/2306.00978  
8. SpQR — https://arxiv.org/abs/2306.03078  
9. SqueezeLLM — https://arxiv.org/abs/2306.07629  
10. Atom — https://arxiv.org/abs/2310.19102  
11. UniSparse — https://arxiv.org/abs/2403.05802  
12. Cypress — https://arxiv.org/abs/2504.07004  
13. TensorIR — https://arxiv.org/abs/2207.04296  
14. Relax — https://arxiv.org/abs/2311.02103  
15. DCC — https://arxiv.org/abs/2511.15503  
16. nncase — https://arxiv.org/abs/2512.21571  
17. TAPAS — https://arxiv.org/abs/2302.00247  
18. Axe — https://arxiv.org/abs/2601.19092  
19. Tensor Processing Primitives — https://arxiv.org/abs/2104.05755  
20. Ladder — https://www.usenix.org/conference/osdi24/presentation/wang-lei  

---

## 24. Immediate order of execution

```text
1. Create the branch and preserve the baseline.
2. Repair genome-axis bookkeeping.
3. Add Semantic Region IR and verifier.
4. Add explicit region-spec loader.
5. Add mapped-tensor-backed demo example.
6. Emit deterministic plan and compile-verified receipt.
7. Run formatting, lint, unit, and compile tests.
8. Freeze the demo branch.
9. After the demo, implement static discovery.
10. Only after real region probes exist, integrate regional search.
```

This order produces a defensible demonstration without destabilizing the latest heterogeneous workload and graph-canonical runtime changes.
