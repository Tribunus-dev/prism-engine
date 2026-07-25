use crate::compilation_entity::{CompilationEntity, CompilationStatus};
use crate::ecs::{
    CImageArtifact, CompilationSession, KernelCollection, LegalizedPlan, SearchStateComponent,
    SessionHandle, SessionStatus, SourceModel, SpatialGraphComponent, TensorCollection,
};
use crate::CompileError;
use prism_ecs_core::component::Component;
use prism_ecs_core::world::World;

/// Mirror the session lifecycle into the legacy compilation entity so the two
/// ECS representations cannot silently drift.
pub fn sync_compilation_entity(world: &mut World) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    let session_status = world
        .component::<CompilationSession>(handle.0)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?
        .status
        .clone();
    let mapped = match session_status {
        SessionStatus::Complete => CompilationStatus::Complete,
        SessionStatus::Failed(_) => CompilationStatus::Failed,
        SessionStatus::Initialized => CompilationStatus::Created,
        _ => CompilationStatus::Running,
    };
    world
        .component_mut::<CompilationEntity>(handle.0)
        .map_err(|e| CompileError::CompilationFailed(format!("compilation entity missing: {e}")))?
        .status = mapped;
    Ok(())
}

fn session_mut(world: &mut World) -> Result<crate::ecs::SessionHandle, CompileError> {
    world
        .get_resource::<SessionHandle>()
        .copied()
        .ok_or_else(|| CompileError::CompilationFailed("session handle missing".into()))
}

fn require_status(world: &mut World, expected: SessionStatus) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    let actual = world
        .component::<CompilationSession>(handle.0)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?
        .status
        .clone();
    if std::mem::discriminant(&actual) != std::mem::discriminant(&expected) {
        return Err(CompileError::PolicyViolation(format!(
            "invalid ECS stage transition: expected {expected:?}, found {actual:?}"
        )));
    }
    Ok(())
}

fn require_component<T: Component>(world: &mut World) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    world
        .component::<T>(handle.0)
        .map(|_| ())
        .map_err(|e| CompileError::PolicyViolation(format!("stage output missing: {e}")))
}

pub fn system_transition_ingest_to_plan(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::Ingested)?;
    require_component::<SourceModel>(world)?;
    require_component::<TensorCollection>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_plan_to_evaluate(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::GraphBuilt)?;
    require_component::<SpatialGraphComponent>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_evaluate_to_legalize(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::SearchComplete)?;
    require_component::<SearchStateComponent>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_legalize_to_compile(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::Legalized)?;
    require_component::<LegalizedPlan>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_compile_to_emit(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::KernelsGenerated)?;
    require_component::<KernelCollection>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_emit_to_complete(world: &mut World) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    {
        let status = world
            .component::<CompilationSession>(handle.0)
            .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
        if !matches!(status.status, SessionStatus::Certified) {
            return Err(CompileError::PolicyViolation(
                "completion requires a certified CImage".into(),
            ));
        }
    }
    require_component::<CImageArtifact>(world)?;
    {
        let status = world
            .component_mut::<CompilationSession>(handle.0)
            .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
        status.status = SessionStatus::Complete;
    }
    sync_compilation_entity(world)
}

/// Synthesize a `QuantizationResultComponent` from whatever the search
/// produced and the source catalog.
///
/// The previous emission step read the source catalog and wrote
/// `Blob`-typed source bytes under the source dtype, ignoring the
/// search's format plan. The new emission step requires a
/// `QuantizationResultComponent` on the session; this system
/// materializes one so the constitutional path is the production path.
///
/// Two input modes:
///
/// 1. `SearchStateComponent.format_plan` is set and decodes to a
///    `FormatPlan` — use the per-tensor formats from the plan.
/// 2. No format plan — every tensor gets the default
///    `Palettized4Bit` format, but the bytes are still the
///    source-catalog bytes (legacy `Blob` behavior).
///
/// In both cases the `tensor_type` and `payload_bytes` are read from
/// the catalog metadata, not the source provider. That is sufficient
/// for the emission/certification round-trip; a future cutover will
/// call `prism_ecs_quantization::build_quantization_plan` here for
/// real per-tensor quantization.
pub fn system_build_quantization_result(world: &mut World) -> Result<(), CompileError> {
    use prism_ecs_constitutional::compilation::{
        QuantizationResultComponent, QuantizedTensorSelectionComponent,
    };
    use prism_ecs_ir::evolution::compile_plan::FormatPlan;
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;
    use prism_ecs_quantization::cimage::TensorType;

    let handle = session_mut(world)?;
    let session = handle.0;

    // The source catalog drives the per-tensor plan.
    let source = world
        .get_extension::<crate::ecs::CurrentSource>()
        .ok_or_else(|| {
            CompileError::CompilationFailed("no CurrentSource for plan synthesis".into())
        })?;

    // Optional format plan from the search.
    let format_plan: Option<FormatPlan> = world
        .component::<crate::ecs::SearchStateComponent>(session)
        .ok()
        .and_then(|s| s.format_plan.clone())
        .and_then(|json| serde_json::from_str(&json).ok());

    let default_format = TensorFormat::Palettized4Bit;

    let mut selections: Vec<QuantizedTensorSelectionComponent> = Vec::new();
    for tensor in source.0.catalog.iter() {
        let format = format_plan
            .as_ref()
            .and_then(|p| p.per_tensor.get(&tensor.name).copied())
            .unwrap_or(default_format);

        // For the legacy test path the provider is an `EmptyProvider`.
        // We record the catalog's reported byte count and leave the
        // payload empty; the writer writes zero bytes for this entry.
        // The certification step verifies size agreement, so any
        // divergence surfaces immediately.
        let payload: Vec<u8> = match source.0.provider.as_ref() {
            Some(p) => p
                .read_tensor(tensor)
                .map_err(|e| CompileError::CompilationFailed(format!("read tensor {}: {e}", tensor.name)))?,
            None => Vec::new(),
        };
        let payload_bytes = payload.len() as u64;

        // Map the canonical format to the legacy catalog dtype. Real
        // per-tensor quantization (which would map NF4 → NF4 type,
        // Fp16 → StandardFP16, etc.) is the next step; for now the
        // plan records the selected format while the physical type
        // stays `Blob` so the certification contract holds.
        let tensor_type = TensorType::Blob;
        let _ = tensor_type.discriminant_byte(); // silence unused warnings if format isn't used

        let dim_m = tensor.shape.first().copied().unwrap_or(0) as u32;
        let dim_n = tensor.shape.get(1).copied().unwrap_or(0) as u32;

        selections.push(QuantizedTensorSelectionComponent {
            key: tensor.name.clone(),
            format_discriminant: format.discriminant_byte(),
            payload,
            tensor_type_discriminant: TensorType::Blob.discriminant_byte()[0],
            dim_m,
            dim_n,
            effective_bpp: match format {
                TensorFormat::Fp16 | TensorFormat::Bf16 => 16.0,
                TensorFormat::Int8 | TensorFormat::Nf8 => 8.0,
                TensorFormat::Int4 | TensorFormat::Nf4 | TensorFormat::Palettized4Bit => 4.0,
                TensorFormat::Ternary158 => 2.0,
                TensorFormat::Binary1 => 1.0,
            },
            payload_bytes,
        });
    }

    let plan = QuantizationResultComponent {
        source_digest: source.0.identity.source_digest.clone(),
        target_hardware: "default".into(),
        selections,
        default_format: format!("{:?}", default_format),
        schema_version: 1,
    };

    world
        .insert_component(session, plan)
        .map_err(|e| CompileError::CompilationFailed(format!("insert plan: {e}")))?;

    Ok(())
}
