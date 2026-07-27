//! Compilation authority — the constitutional surface for compilation
//! jobs, validation receipts, quantization plans, and CImage promotion.
//!
//! ## Sub-modules (single authority per file)
//!
//! | Sub-module            | Authority                                       |
//! |-----------------------|-------------------------------------------------|
//! | [`schema_ids`]        | Schema ID constants 31-39 (durable wire IDs).   |
//! | [`job`]               | `CompilationJob` shape, lifecycle, create cmd,  |
//! |                       | replay, and the shared `CompilationError`.       |
//! | [`validation`]        | `ValidationReceipt` and receipt-submit command.  |
//! | [`quantization`]      | `QuantizationPlan`, `QuantizationResultComponent` |
//! |                       | (and selection), submit command, bridge.         |
//! | [`cimage_promotion`]  | `CimagePromotion` and `PromoteCimageCommand`.    |
//! | [`observation`]       | `CompileProgress` projection (absorbed from     |
//! |                       | engine's `core/compile_progress.rs`).           |
//!
//! ## Schema IDs
//!
//! `prism.compilation` namespace, IDs 31-39 (selection at 40). See
//! [`schema_ids`] for the full allocation table. The schema IDs are
//! the durable contract for cross-process replay; bumping any of
//! them is a wire-format break.
//!
//! ## Engine boundary
//!
//! The engine's `compute-core/src/ecs/core/compile_state.rs` remains
//! **execution-boundary**: it owns the `CompileState::write` /
//! `read` methods that open files on disk (file descriptor I/O per
//! AGENTS.md criterion 1). The data types it carries
//! (`CompileStage`, `SegmentCompletion`, `SchedulerConfig`,
//! `SchedulerPolicy`) are not absorbed; the parallel observation
//! surface in this crate is [`observation::CompileProgress`], and
//! the lifecycle mapping is documented in [`job::JobLifecycle`].
//!
//! The engine's `compute-core/src/ecs/compile/{audio,vision}.rs`
//! entry points and the `archive_ane_modelc` helper in
//! `compile/pipeline.rs` are also execution-boundary (cimage
//! pipeline orchestration; the `compile/pipeline.rs` re-exports
//! `compute_image::compile::archive_ane_modelc`, a side-effecting
//! function that writes a tar archive to disk). Engine consumers
//! that import `compute_core::ecs::compile::audio::compile_audio_model`
//! continue to compile — those paths are unchanged.
//!
//! The engine's `compute-core/src/ecs/core/compile_pipeline.rs` is
//! also execution-boundary: it spawns `tokio::task::spawn_blocking`
//! workers and owns `mpsc::channel` senders/receivers (criterion 3,
//! process-local state). The constitutional crate has no
//! equivalent; the canonical pipeline is described by the
//! `CompilationJob` state machine and the per-stage commands
//! defined in the sub-modules above.

pub mod cimage_promotion;
pub mod job;
pub mod observation;
pub mod quantization;
pub mod schema_ids;
pub mod validation;

// ── Re-exports (preserve `prism_ecs_constitutional::compilation::*`
//    import surface from the pre-decomposition godfile) ──────────────

pub use cimage_promotion::{CimagePromotion, PromoteCimageCommand};
pub use job::{
    CompilationError, CompilationJob, CreateCompilationJobCommand, JobConfig, JobInput,
    JobLifecycle, JobOutput, replay_compilation_job_created,
};
pub use observation::CompileProgress;
pub use quantization::{
    QuantizationPlan, QuantizationResultComponent, QuantizedTensorSelectionComponent,
    SubmitQuantizationResultCommand, default_format_name,
};
#[cfg(feature = "quantization-bridge")]
pub use quantization::quantization_result_to_component;
pub use schema_ids::{
    SCHEMA_CIMAGE_PROMOTION, SCHEMA_COMPILATION_JOB, SCHEMA_JOB_CONFIG, SCHEMA_JOB_INPUT,
    SCHEMA_JOB_LIFECYCLE, SCHEMA_JOB_OUTPUT, SCHEMA_QUANTIZATION_PLAN, SCHEMA_QUANTIZATION_RESULT,
    SCHEMA_VALIDATION_RECEIPT,
};
pub use validation::{SubmitValidationReceiptCommand, ValidationReceipt};

// ── Cross-cutting helpers ──────────────────────────────────────────────────

use crate::schema::SchemaRegistry;
use crate::types::ComponentSchemaId;

/// Validate that every compilation sub-module's component types are
/// registered against the expected schema ID in `reg`.
///
/// Used at startup and inside each command's `preflight` to catch
/// schema-id drift before it can corrupt a durable event.
pub fn validate_compilation_schemas(reg: &SchemaRegistry) -> Result<(), String> {
    reg.verify_type::<CompilationJob>(ComponentSchemaId(SCHEMA_COMPILATION_JOB))
        .map_err(|e| format!("CompilationJob schema: {e}"))?;
    reg.verify_type::<JobInput>(ComponentSchemaId(SCHEMA_JOB_INPUT))
        .map_err(|e| format!("JobInput schema: {e}"))?;
    reg.verify_type::<JobConfig>(ComponentSchemaId(SCHEMA_JOB_CONFIG))
        .map_err(|e| format!("JobConfig schema: {e}"))?;
    reg.verify_type::<JobOutput>(ComponentSchemaId(SCHEMA_JOB_OUTPUT))
        .map_err(|e| format!("JobOutput schema: {e}"))?;
    reg.verify_type::<JobLifecycle>(ComponentSchemaId(SCHEMA_JOB_LIFECYCLE))
        .map_err(|e| format!("JobLifecycle schema: {e}"))?;
    reg.verify_type::<ValidationReceipt>(ComponentSchemaId(SCHEMA_VALIDATION_RECEIPT))
        .map_err(|e| format!("ValidationReceipt schema: {e}"))?;
    reg.verify_type::<QuantizationPlan>(ComponentSchemaId(SCHEMA_QUANTIZATION_PLAN))
        .map_err(|e| format!("QuantizationPlan schema: {e}"))?;
    reg.verify_type::<CimagePromotion>(ComponentSchemaId(SCHEMA_CIMAGE_PROMOTION))
        .map_err(|e| format!("CimagePromotion schema: {e}"))?;
    reg.verify_type::<QuantizationResultComponent>(ComponentSchemaId(SCHEMA_QUANTIZATION_RESULT))
        .map_err(|e| format!("QuantizationResultComponent schema: {e}"))?;
    Ok(())
}

/// Register all compilation sub-module component types against a
/// [`SchemaRegistry`] using the canonical schema IDs. Test and
/// production startup code should call this exactly once.
///
/// The list is intentionally explicit (rather than derived from a
/// `linkme` / inventory collector) so that the schema catalogue is
/// statically checkable: any new component type added to a
/// sub-module shows up as a missing `register_for_type` call here,
/// which fails the next `validate_compilation_schemas` invocation.
pub fn register_compilation_schemas(reg: &mut SchemaRegistry) {
    use crate::schema::ComponentDurability;
    use crate::types::SchemaVersion;
    reg.register_for_type::<CompilationJob>(
        ComponentSchemaId(SCHEMA_COMPILATION_JOB),
        SchemaVersion(1),
        "CompilationJob",
        "Compilation job metadata",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<JobInput>(
        ComponentSchemaId(SCHEMA_JOB_INPUT),
        SchemaVersion(1),
        "JobInput",
        "Compilation job input",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<JobConfig>(
        ComponentSchemaId(SCHEMA_JOB_CONFIG),
        SchemaVersion(1),
        "JobConfig",
        "Compilation job config",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<JobOutput>(
        ComponentSchemaId(SCHEMA_JOB_OUTPUT),
        SchemaVersion(1),
        "JobOutput",
        "Compilation job output",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<JobLifecycle>(
        ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
        SchemaVersion(1),
        "JobLifecycle",
        "Compilation job lifecycle",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<ValidationReceipt>(
        ComponentSchemaId(SCHEMA_VALIDATION_RECEIPT),
        SchemaVersion(1),
        "ValidationReceipt",
        "Validation receipt",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<QuantizationPlan>(
        ComponentSchemaId(SCHEMA_QUANTIZATION_PLAN),
        SchemaVersion(1),
        "QuantizationPlan",
        "Quantization plan",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<CimagePromotion>(
        ComponentSchemaId(SCHEMA_CIMAGE_PROMOTION),
        SchemaVersion(1),
        "CimagePromotion",
        "CImage promotion record",
        ComponentDurability::Durable,
    );
    reg.register_for_type::<QuantizationResultComponent>(
        ComponentSchemaId(SCHEMA_QUANTIZATION_RESULT),
        SchemaVersion(1),
        "QuantizationResultComponent",
        "Per-tensor quantization result attached to a job",
        ComponentDurability::Durable,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_then_validate_succeeds() {
        let mut reg = SchemaRegistry::new();
        register_compilation_schemas(&mut reg);
        assert!(validate_compilation_schemas(&reg).is_ok());
    }

    #[test]
    fn validate_fails_when_registry_is_empty() {
        let reg = SchemaRegistry::new();
        assert!(validate_compilation_schemas(&reg).is_err());
    }

    #[test]
    fn all_schemas_have_durable_storage_class() {
        let mut reg = SchemaRegistry::new();
        register_compilation_schemas(&mut reg);
        for id in [
            SCHEMA_COMPILATION_JOB,
            SCHEMA_JOB_INPUT,
            SCHEMA_JOB_CONFIG,
            SCHEMA_JOB_OUTPUT,
            SCHEMA_JOB_LIFECYCLE,
            SCHEMA_VALIDATION_RECEIPT,
            SCHEMA_QUANTIZATION_PLAN,
            SCHEMA_CIMAGE_PROMOTION,
            SCHEMA_QUANTIZATION_RESULT,
        ] {
            let entry = reg
                .get(&ComponentSchemaId(id))
                .expect("schema registered");
            assert_eq!(
                entry.durability,
                crate::schema::ComponentDurability::Durable,
                "schema {id} must be durable"
            );
        }
    }
}
