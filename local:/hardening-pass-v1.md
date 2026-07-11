# Prism Hardening Pass 0001 — with PrecisionPlan

## Design Correction

Mixed precision is not an exception to CodecFamily. It is a first-class strategy managed by PrecisionPlan. CodecFamily::Mixed requires a PrecisionPlan to be valid.

## New Types

### PrecisionPlan (execution_plan/precision_plan.rs or fusion.rs)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPlan {
    pub plan_id: String,
    pub scope: PrecisionScope,
    pub default_codec: CodecFamily,
    pub overrides: Vec<PrecisionOverride>,
    pub selection_basis: PrecisionSelectionBasis,
    pub evidence_level: RequiredEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionScope {
    WholeTensor, TensorFamily, LayerRange, Tile, Group,
    InputAxisSlice, OutputAxisSlice, Expert, FusedGroup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOverride {
    pub selector: PrecisionSelector,
    pub codec: CodecFamily,
    pub reason: PrecisionOverrideReason,
    pub byte_cost: u64,
    pub expected_error_reduction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionSelector {
    TileIds(Vec<u32>),
    GroupIds(Vec<u32>),
    InputColumns(Vec<u32>),
    OutputRows(Vec<u32>),
    LayerRange { start: u32, end: u32 },
    TopErrorTiles { fraction: f64 },
    OutlierColumns { max_fraction: f64 },
    ActivationWeightedTopK { fraction: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionOverrideReason {
    OperatorTailRescue,
    ActivationWeightedOutlier,
    ZeroCollapseRescue,
    ByteSavingsFallback,
    BackendCompatibility,
    RawF32Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionSelectionBasis {
    StaticPolicy, WeightError, OperatorError,
    ActivationWeightedError, OutlierMagnitude,
    ZeroCollapseRisk, HardwareProfile, LearnedProfile,
}
```

### CodecFamily gets Mixed variant
In the existing CodecFamily enum, add Mixed.
Add validation: `CodecFamily::Mixed` is invalid if not accompanied by a PrecisionPlan.

### MixedPrecisionCapability (backend_capability.rs)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionCapability {
    pub supports_mixed_precision: bool,
    pub supported_scopes: Vec<PrecisionScope>,
    pub supported_base_codecs: Vec<CodecFamily>,
    pub supported_override_codecs: Vec<CodecFamily>,
    pub max_override_fraction: Option<f64>,
    pub requires_sidecar: bool,
    pub supports_inline_mixed_tiles: bool,
}
```

Add `pub mixed_precision: MixedPrecisionCapability` to BackendCapability.

### Updated evaluate() behavior

In BackendCapabilityRegistry::evaluate():
- If group.semantics.mixed_codec is true:
  - Group must have a PrecisionPlan → reject with MissingPrecisionPlan if not present
  - Backend MixedPrecisionCapability.supports_mixed_precision must be true
  - Plan scope must be in supported_scopes
  - Plan base_codec must be in supported_base_codecs
  - Each override codec must be in supported_override_codecs
  - Override fraction must not exceed max_override_fraction
- If group is NOT mixed_codec, evaluate normally (unchanged)

### Backend default mixed precision matrix

MetalFusedGpu:
- supports_mixed_precision: true
- scopes: Tile, Group, InputAxisSlice, OutputAxisSlice
- base: NF4, INT8, FP16
- overrides: INT8, FP16, RawF32
- max_override_fraction: Some(0.10)
- requires_sidecar: true
- inline_mixed_tiles: true

AnePlanarEngine:
- supports_mixed_precision: true (limited)
- scopes: WholeTensor, FusedGroup
- base: FP16, INT8
- overrides: FP16
- max_override_fraction: None (any fraction)
- requires_sidecar: false
- inline_mixed_tiles: false

CoreMlHighLevel:
- supports_mixed_precision: false
- scopes: [] (empty)
- base: FP16, INT8
- overrides: []
- remains effectively single-codec

AccelerateRayonCpu:
- supports_mixed_precision: true
- scopes: Tile, Group, AxisSlice
- base: RawF32, FP16, INT8
- overrides: RawF32, FP16, INT8 (NF4 only if custom kernel registered)
- max_override_fraction: Some(0.25)
- requires_sidecar: true
- inline_mixed_tiles: true (via custom tile iterators)

### Updated FusedGroupSemantics
Add `precision_plan: Option<PrecisionPlan>` field.

### MixedPrecisionPlanner (execution_plan/mixed_precision.rs)

```rust
pub struct MixedPrecisionPlanner;
impl MixedPrecisionPlanner {
    pub fn plan(
        tensor: &TensorDescriptor,
        base_candidate: &SubstitutionExperimentReceipt,
        attribution: &AttributionReceipt,
        policy: &MixedPrecisionPolicy,
    ) -> Result<PrecisionPlan, MixedPrecisionError> {
        // 1. base codec from candidate
        // 2. per-tile/group error attribution
        // 3. sort by error contribution descending
        // 4. promote top units to rescue codec
        // 5. recompute effective byte cost
        // 6. re-run operator validation
        // 7. stop when gates pass or byte-savings floor violated
    }
}
```

### MixedPrecisionLayout (cimage layout descriptor)

```rust
pub struct MixedPrecisionLayout {
    pub base_layout: PhysicalTileLayout,
    pub override_table: PrecisionOverrideTable,
    pub sidecars: Vec<PrecisionSidecar>,
    pub dispatch_policy: MixedDispatchPolicy,
}
pub struct PrecisionOverrideTable {
    pub granularity: PrecisionScope,
    pub entries: Vec<PrecisionOverrideEntry>,
}
pub struct PrecisionOverrideEntry {
    pub unit_id: u32,
    pub codec: CodecFamily,
    pub payload_offset: u64,
    pub metadata_offset: u64,
    pub element_count: u32,
}
pub struct PrecisionSidecar {
    pub codec: CodecFamily,
    pub payload_bytes: u64,
    pub metadata_bytes: u64,
    pub alignment_bytes: u32,
}
```

### MixedPrecisionReceipt

```rust
pub struct MixedPrecisionReceipt {
    pub tensor_key: String,
    pub tensor_class: String,
    pub base_codec: CodecFamily,
    pub override_count: usize,
    pub override_fraction: f64,
    pub effective_bytes: u64,
    pub raw_f32_bytes: u64,
    pub byte_savings_ratio: f64,
    pub operator_nrmse: Option<f64>,
    pub operator_cosine: Option<f64>,
    pub operator_max_abs_error: Option<f64>,
    pub selected: bool,
    pub rejection_reason: Option<String>,
    pub precision_plan_digest: String,
}
```

### MixedPrecisionTrainingTarget

```rust
pub struct MixedPrecisionTrainingTarget {
    pub tensor_class: String,
    pub base_codec: CodecFamily,
    pub allowed_override_codecs: Vec<CodecFamily>,
    pub max_override_fraction: f64,
    pub target_override_fraction: f64,
    pub loss_terms: Vec<TargetedLossTerm>,
}
```

## Revised Test Set (10 tests)

1. nf4_group_selects_metal_only — NF4 group, no PrecisionPlan, ANE/CPU reject, Metal selected (single codec)
2. mixed_codec_without_precision_plan_rejected — group has mixed_codec=true but no PrecisionPlan, all backends reject
3. mixed_precision_nf4_base_int8_rescue_metal_supported — NF4 base + INT8 rescue with PrecisionPlan, Metal accepts
4. mixed_precision_nf4_base_int8_rescue_ane_rejected — same PrecisionPlan, ANE rejects because NF4 not in supported_base_codecs
5. mixed_precision_int8_base_fp16_rescue_cpu_supported — CPU accepts
6. compile_mode_fails_without_backend — empty registry, Compile mode
7. explore_mode_records_without_backend — empty registry, Explore mode
8. profile_template_not_gate_eligible — template evidence cannot promote
9. training_target_layout_parses_policy — PhysicalTileLayout parsed from policy
10. metal_lowering_requires_bindings — ScheduledKernelOp without bindings fails validate_lowered()
