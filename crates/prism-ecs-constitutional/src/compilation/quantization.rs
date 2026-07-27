//! `QuantizationPlan`, `QuantizationResultComponent`, and the
//! quantization-side submit command.
//!
//! **Single authority:** owns the constitutional shape of a
//! quantization result and the bridge that converts a
//! `prism_ecs_quantization::QuantizationResult` into the canonical
//! per-tensor component. Also owns `SubmitQuantizationResultCommand`,
//! which is the chokepoint between per-tensor compilation (which
//! lives in `prism_ecs_quantization`) and the constitutional
//! `CompilationJob`.
//!
//! The `prism_ecs_quantization` crate stays free of constitutional
//! dependencies: the conversion goes one way, and the
//! `quantization-bridge` feature flag on this crate is the only
//! dependency edge.

use crate::compilation::job::CompilationError;
use crate::schema::SchemaRegistry;
use crate::types::{EntityKindId, MessageId, SchemaKey};
use crate::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTransitExt, WorldTxn,
};
use prism_ecs_core::{Component, Entity, World};
use serde::{Deserialize, Serialize};

use super::job::JobLifecycle;

// ── Component types ─────────────────────────────────────────────────────────

/// A quantization plan — describes how weights should be quantized.
///
/// `QuantizationPlan` is the long-lived "shape" of a quantization
/// strategy attached to a job; the per-tensor decisions live in
/// [`QuantizationResultComponent`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub codec: String,
    pub group_size: u32,
    pub target_bitwidth: u8,
    pub validation_gate: String,
}

impl Component for QuantizationPlan {}
impl ClassifiedComponent for QuantizationPlan {
    type Class = DurableClass;
}
impl DurableComponent for QuantizationPlan {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 37,
        version: 1,
    };
}

/// Per-tensor result of a quantization pass, attached to a
/// `CompilationJob`.
///
/// This is the constitutional counterpart of
/// `prism_ecs_quantization::QuantizationResult`. It is the structured
/// evidence that the per-tensor codecs ran and produced specific
/// representations. The job cannot transition from `Compiling` to
/// `Validating` without this component attached.
///
/// `selections` is the canonical ordered list of decisions. Its digest
/// is what `CImage` emission later seals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationResultComponent {
    /// Content digest of the source model the selections were derived
    /// from.
    pub source_digest: String,
    /// Target hardware identifier (e.g. "apple-m1", "cpu").
    pub target_hardware: String,
    /// Per-tensor selections, in source-graph iteration order.
    pub selections: Vec<QuantizedTensorSelectionComponent>,
    /// Default format used for tensors that did not appear in the
    /// caller-supplied `FormatPlan`. Surfaced explicitly so receipts
    /// can prove the default policy was applied.
    pub default_format: String,
    /// Schema version of the originating `prism_ecs_quantization`
    /// plan. Bumped if the on-wire shape changes.
    pub schema_version: u32,
}

impl Component for QuantizationResultComponent {}
impl ClassifiedComponent for QuantizationResultComponent {
    type Class = DurableClass;
}
impl DurableComponent for QuantizationResultComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 39,
        version: 1,
    };
}

/// A single per-tensor selection — the constitutional counterpart of
/// `prism_ecs_quantization::QuantizedTensorSelection`. Stores the
/// payload bytes and the format that was actually applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedTensorSelectionComponent {
    pub key: String,
    /// `TensorFormat` discriminant, encoded as a string for
    /// forward-compatibility. Decoded via
    /// `prism_ecs_quantization::TensorFormat::discriminant_byte` /
    /// `from_discriminant_byte` (the latter is added when the first
    /// version of this component ships).
    pub format_discriminant: u8,
    /// CImage-ready payload bytes for the applied codec.
    pub payload: Vec<u8>,
    /// `TensorType` discriminant (CImage physical type).
    pub tensor_type_discriminant: u8,
    pub dim_m: u32,
    pub dim_n: u32,
    pub effective_bpp: f32,
    pub payload_bytes: u64,
}

impl Component for QuantizedTensorSelectionComponent {}
impl ClassifiedComponent for QuantizedTensorSelectionComponent {
    type Class = DurableClass;
}
impl DurableComponent for QuantizedTensorSelectionComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 40,
        version: 1,
    };
}

// ── Submit command ──────────────────────────────────────────────────────────

/// Command to submit a `QuantizationResultComponent` for a compilation
/// job.
///
/// This is the chokepoint between per-tensor compilation (which lives
/// in `prism_ecs_quantization`) and the constitutional `CompilationJob`
/// entity. After this command, the job has a `Planned` lifecycle and
/// a structured plan that downstream validation and emission can read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmitQuantizationResultCommand {
    pub id: MessageId,
    /// Entity ID of the job receiving the plan. See [`Entity`] for the
    /// canonical generational entity handle.
    pub job_entity: u64,
    /// The structured per-tensor result.
    pub result: QuantizationResultComponent,
}

impl SubmitQuantizationResultCommand {
    /// Preflight: validate schemas and that the job exists in a
    /// compatible lifecycle (`Compiling` is the only accepted source).
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        super::validate_compilation_schemas(schema_registry).map_err(CompilationError::SchemaError)?;

        let entity = Entity::new(self.job_entity, 0);
        if !world.has_entity(entity) {
            return Err(CompilationError::JobNotFound(self.job_entity));
        }
        let lifecycle = world
            .get_component::<JobLifecycle>(entity)
            .ok_or(CompilationError::JobNotFound(self.job_entity))?;
        if *lifecycle != JobLifecycle::Compiling {
            return Err(CompilationError::InvalidState {
                job_id: self.job_entity,
                expected: JobLifecycle::Compiling,
                actual: *lifecycle,
            });
        }
        Ok(())
    }

    /// Execute: attach the result and transition the lifecycle to
    /// `Planned`.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, crate::command::DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let entity = Entity::new(self.job_entity, 0);
        let mut txn = WorldTxn::new(world);

        txn.put_durable(entity, self.result.clone());
        txn.put_durable(entity, JobLifecycle::Planned);

        let event = crate::command::DomainEvent {
            id: self.id,
            kind: "quantization_result_submitted".to_string(),
            entity_id: Some(EntityKindId(self.job_entity)),
            payload: serde_json::json!({
                "job_entity": self.job_entity,
                "selection_count": self.result.selections.len(),
                "default_format": self.result.default_format,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

        Ok((epoch, event))
    }
}

// ── Quantization bridge (optional feature) ──────────────────────────────────

/// Convert a `prism_ecs_quantization::QuantizationResult` into the
/// constitutional `QuantizationResultComponent`.
///
/// This lives in the constitutional crate because it is the only
/// place that knows the wire-format on the ECS side. The
/// `prism_ecs_quantization` crate stays pure of constitutional
/// dependencies.
#[cfg(feature = "quantization-bridge")]
pub fn quantization_result_to_component(
    result: &prism_ecs_quantization::QuantizationResult,
) -> QuantizationResultComponent {
    use prism_ecs_quantization::cimage::TensorType;
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;

    fn tensor_type_discriminant(t: &TensorType) -> u8 {
        t.discriminant_byte()[0]
    }

    QuantizationResultComponent {
        source_digest: result.source_digest.clone(),
        target_hardware: result.target_hardware.clone(),
        selections: result
            .selections
            .iter()
            .map(|s| QuantizedTensorSelectionComponent {
                key: s.key.clone(),
                format_discriminant: s.format.discriminant_byte(),
                payload: s.payload.clone(),
                tensor_type_discriminant: tensor_type_discriminant(&s.tensor_type),
                dim_m: s.dim_m,
                dim_n: s.dim_n,
                effective_bpp: s.effective_bpp,
                payload_bytes: s.payload_bytes,
            })
            .collect(),
        default_format: format!("{:?}", result.default_format),
        schema_version: 1,
    }
}

/// Reference to the default format for runtime use. Re-exported for
/// code that needs the typed value but cannot import the IR enum.
#[must_use]
pub fn default_format_name() -> &'static str {
    "Palettized4Bit"
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_quantization_plan_construction ───────────────────────────

    #[test]
    fn quantization_plan_construction() {
        let plan = QuantizationPlan {
            codec: "nf4".to_string(),
            group_size: 64,
            target_bitwidth: 4,
            validation_gate: "quantization_admission".to_string(),
        };

        assert_eq!(plan.codec, "nf4");
        assert_eq!(plan.group_size, 64);
        assert_eq!(plan.target_bitwidth, 4);
        assert_eq!(plan.validation_gate, "quantization_admission");
    }

    #[test]
    fn quantization_plan_serde_roundtrip() {
        let plan = QuantizationPlan {
            codec: "nf4".to_string(),
            group_size: 64,
            target_bitwidth: 4,
            validation_gate: "quantization_admission".to_string(),
        };
        let json = serde_json::to_string(&plan).expect("serialize");
        let back: QuantizationPlan = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(plan, back);
    }

    #[test]
    fn quantization_result_component_construction() {
        let result = QuantizationResultComponent {
            source_digest: "sha256:deadbeef".to_string(),
            target_hardware: "apple-m1".to_string(),
            selections: vec![QuantizedTensorSelectionComponent {
                key: "model.layer.0.weight".to_string(),
                format_discriminant: 4, // Nf4
                payload: vec![0u8; 16],
                tensor_type_discriminant: 0,
                dim_m: 64,
                dim_n: 64,
                effective_bpp: 4.5,
                payload_bytes: 16,
            }],
            default_format: "Palettized4Bit".to_string(),
            schema_version: 1,
        };

        assert_eq!(result.selections.len(), 1);
        assert_eq!(result.selections[0].format_discriminant, 4);
        assert_eq!(result.selections[0].key, "model.layer.0.weight");
        assert_eq!(result.default_format, "Palettized4Bit");
    }

    #[test]
    fn default_format_name_is_palettized4bit() {
        assert_eq!(default_format_name(), "Palettized4Bit");
    }

    // ── test_submit_quantization_result_command ───────────────────────

    #[test]
    fn submit_quantization_result_rejects_non_compiling_state() {
        let mut world = World::new();
        let mut reg = SchemaRegistry::new();
        super::super::register_compilation_schemas(&mut reg);

        // Spawn an entity as a job in Pending state.
        let entity_id = WorldTxn::next_entity_id(&world);
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(entity_id, prism_ecs_core::EntityKind::Executable);
        txn.add_component(
            entity_id,
            crate::types::ComponentSchemaId(super::super::schema_ids::SCHEMA_JOB_LIFECYCLE),
            crate::types::SchemaVersion(1),
            JobLifecycle::Pending,
        );
        world.transit(txn).unwrap();

        let cmd = SubmitQuantizationResultCommand {
            id: MessageId::compute(b"qplan_1"),
            job_entity: entity_id.id(),
            result: QuantizationResultComponent {
                source_digest: "sha256:beef".to_string(),
                target_hardware: "cpu".to_string(),
                selections: Vec::new(),
                default_format: default_format_name().to_string(),
                schema_version: 1,
            },
        };

        let result = cmd.preflight(&world, &reg);
        assert!(matches!(result, Err(CompilationError::InvalidState { .. })));
    }
}
