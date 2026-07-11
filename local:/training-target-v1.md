# Training-Aware Compilation — Implementation Spec v1

## Module
`compute-core/src/training_target/`

## Files
```
training_target/
  mod.rs           — module declarations, re-exports
  spec.rs          — TrainingTargetSpec, WeightTrainingTarget, KvCacheTrainingTarget, etc.
  gates.rs         — WeightTrainingGates, RequiredEvidenceLevel, TrainingTargetStatus
  resolve.rs       — TrainingTargetResolver (policy → spec)
  feedback.rs      — TrainingFeedbackReport, TrainingFeedbackBuilder, TrainingFailureMode, TargetedLossTerm
  receipts.rs      — TrainingTargetReceipt, TrainingFeedbackReceipt
  export.rs        — JSON serialization/deserialization helpers, deterministic digest
  tests.rs         — integration tests
```

## Types (all in spec.rs unless noted)

### TrainingTargetSpec
- spec_version: u32
- model_family: String
- model_digest: Option<String>
- source_policy_digest: String
- target_cimage_profile: String
- weight_targets: Vec<WeightTrainingTarget>
- kv_cache_target: Option<KvCacheTrainingTarget>
- speculative_targets: Vec<SpeculativeTrainingTarget>
- engram_targets: Vec<EngramTrainingTarget>
- attention_shape_targets: Vec<AttentionShapeTrainingTarget>
- evidence_gates: Vec<TrainingEvidenceGate>

### WeightTrainingTarget
- target_id: String
- tensor_class: String
- tensor_key_match: Vec<String>
- target_codec: CodecFamily
- physical_layout: PhysicalTileLayout
- training_method: QuantTrainingMethod
- gates: WeightTrainingGates
- priority: TrainingTargetPriority

### QuantTrainingMethod (gates.rs)
```rust
pub enum QuantTrainingMethod {
    ShadowWeightsSte,
    GradualBitTransition {
        start_bits: f32,
        target_bits: f32,
        schedule_steps: usize,
    },
    SoftTernarization {
        temperature_start: f32,
        temperature_end: f32,
        learnable_modulation: bool,
    },
    ActivationWeighted {
        profile_required: bool,
        objective: ActivationWeightedObjective,
    },
}
```

### WeightTrainingGates (gates.rs)
```rust
pub struct WeightTrainingGates {
    pub max_weight_nrmse: Option<f64>,
    pub max_zero_collapse_ratio: Option<f64>,
    pub max_operator_nrmse: Option<f64>,
    pub min_operator_cosine: Option<f64>,
    pub max_operator_abs_error: Option<f64>,
    pub min_byte_savings_ratio: Option<f64>,
    pub required_evidence_level: RequiredEvidenceLevel,
}
```

### RequiredEvidenceLevel (gates.rs)
```rust
pub enum RequiredEvidenceLevel {
    WeightSpace,
    SyntheticOperator,
    HardwareOperator,
    ModelQuality,
    RuntimeProfiled,
    ProductionPromoted,
}
```

### TrainingTargetStatus (gates.rs)
```rust
pub enum TrainingTargetStatus {
    Draft,
    ReadyForTraining,
    EvidenceIncomplete,
    PartiallySatisfied,
    Satisfied,
    Failed,
}
```

### TrainingTargetPriority
```rust
pub enum TrainingTargetPriority {
    Required,
    Recommended,
    Experimental,
    Research,
}
```

### TrainingFailureMode (feedback.rs)
```rust
pub enum TrainingFailureMode {
    WeightNrmseTooHigh,
    ZeroCollapseTooHigh,
    OperatorNrmseTooHigh,
    OperatorCosineTooLow,
    OperatorAbsTailTooHigh,
    ByteSavingsTooLow,
    ActivationProfileMissing,
    HardwareEvidenceMissing,
    RolloutEvidenceMissing,
    QualityDriftTooHigh,
    RuntimeHealthFailed,
}
```

### TargetedLossTerm (feedback.rs)
```rust
pub enum TargetedLossTerm {
    ReduceZeroCollapse,
    ReduceWeightReconstructionError,
    ReduceActivationWeightedError,
    ReduceOperatorTailError,
    PreserveHiddenDirection,
    PreserveLogitTopK,
    PreserveAttentionScores,
    IncreaseDraftAcceptance,
}
```

### TrainingFeedbackItem (feedback.rs)
- target_id: String
- tensor_key: String
- tensor_class: String
- failed_gate: String
- failure_mode: TrainingFailureMode
- observed_value: Option<f64>
- required_value: Option<f64>
- severity: f64
- suggested_loss: Option<TargetedLossTerm>

### TrainingFeedbackReport (feedback.rs)
- report_version: u32
- spec_digest: String
- checkpoint_digest: String
- evidence_ledger_digest: String
- status: TrainingTargetStatus
- items: Vec<TrainingFeedbackItem>
- summary: TrainingFeedbackSummary

### TrainingFeedbackSummary (feedback.rs)
- total_targets: usize
- satisfied: usize
- failed: usize
- warnings: usize
- gate_results: HashMap<String, f64>  // gate_name → fraction_passing

### TrainingTargetReceipt (receipts.rs)
- spec_digest: String
- source_policy_digest: String
- generated_at_unix_ms: u64
- target_count: usize
- weight_target_count: usize
- warnings: Vec<String>

### TrainingFeedbackReceipt (receipts.rs)
- report_digest: String
- status: TrainingTargetStatus
- total_items: usize
- failed_items: usize
- satisfied_targets: usize

## Evidence Ledger Integration
- Extend EvidenceLedgerEntry enum (from substitution sweep) with a new variant for training feedback
- Add `target_id: String` field to sweep experiment receipts that reference training targets

## CLI Integration
- Add to `tribunus-compute-image`:
  - `training-target export --policy <path> --out <path>` — generate TrainingTargetSpec JSON
  - `training-target check <spec>` — validate internal consistency
  - `training-target feedback --target <spec> --evidence <ledger> --checkpoint-digest <hash> --out <path>` — generate feedback

## Resolver Rules (resolve.rs)
1. Scan compiler_policy for tensors with ternary, nf4, int8 codecs marked as training-eligible
2. RawF32-required → no target (unless experimental override)
3. FP16 → optional calibration target
4. INT8 → optional activation-aware calibration target
5. NF4 → optional training/calibration target
6. Ternary → explicit QAT target (GradualBitTransition or SoftTernarization)
7. Generate PhysicalTileLayout from TensorFamilySpec / codec defaults

## Test Plan (tests.rs)
1. `test_spec_serde_roundtrip` — create, serialize, deserialize, verify integrity
2. `test_resolver_generates_ternary_target` — policy with ternary entry produces WeightTrainingTarget
3. `test_resolver_skips_rawf32` — RawF32-required produces no target
4. `test_feedback_zero_collapse` — evidence with high zero_collapse creates ReduceZeroCollapse feedback
5. `test_feedback_operator_error` — evidence with high operator NRMSE creates ReduceOperatorTailError
6. `test_feedback_satisfied_target` — evidence meeting all gates marks target Satisfied
7. `test_training_target_digest_changes_when_gate_changes` — deterministic digest test
8. `test_export_deterministic` — same spec serializes identically twice
9. `test_check_validates_consistency` — invalid combination is rejected

## Integration with existing types
- Use existing CodecFamily from compute-core/src/execution_profile/mod.rs
- Use existing PhysicalTileLayout from compute-core/src/execution_plan/mod.rs
- Use existing EvidenceLedger from compute-core/src/sweep/mod.rs
- Add target_id field to existing SubstitutionExperimentReceipt
