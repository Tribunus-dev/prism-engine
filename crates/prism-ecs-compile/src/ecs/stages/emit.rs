//! CImage emission stage.
//!
//! Emits the CImage binary artifact from the constitutional
//! `QuantizationResultComponent` (the per-tensor plan), attaches the search
//! trace, legalization report, and kernel artifacts as auxiliary header
//! metadata, and binds the artifact to its plan via the
//! [`CImagePlanDigest`] extension.
//!
//! The previous version of this function iterated
//! `CurrentSource.catalog` and wrote the *source* tensor bytes into
//! the CImage under the *original* dtype, completely ignoring whatever
//! the search / legalization stages had selected. The artifact's body
//! and the artifact's header disagreed.
//!
//! This rewrite consumes the constitutional
//! `QuantizationResultComponent` (attached to the session by
//! `SubmitQuantizationResultCommand`) as the sole source of truth for
//! which tensors are in the artifact and in what representation.
//! Search trace, legalization report, and kernel artifacts remain as
//! auxiliary metadata in the header — they describe the *decisions*
//! but are not the *artifact body*.
//!
//! `set_source` is still called so the header carries the source
//! identity for provenance. The `provider` is no longer used at
//! emission time.

use std::path::PathBuf;

use sha2::Digest;

use prism_ecs_core::world::World;
use prism_ecs_constitutional::compilation::{
    QuantizationResultComponent, QuantizedTensorSelectionComponent,
};

use crate::cimage::{TensorPayloadEntry, UniversalCImageWriter};
use crate::ecs::components::{
    CImageArtifact, CompilationSession, KernelCollection, LegalizedPlan, SearchStateComponent,
    SessionStatus,
};
use crate::ecs::orchestrator::{read_session_config, session_entity};
use crate::ecs::resources::{CImagePlanDigest, CurrentSource, ModelManifestResource, VecEventSink};
use crate::CompileError;

/// Run the **CImage emission** stage.
pub fn system_emit_cimage(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let config = read_session_config(world, session)?;

    let legal = world
        .component::<LegalizedPlan>(session)
        .map_err(|e| CompileError::CImageEmitFailed(format!("legalized plan missing: {e}")))?;
    if !legal.is_valid {
        return Err(CompileError::CImageEmitFailed(
            "cannot emit CImage from an invalid legalized plan".into(),
        ));
    }
    world
        .component::<KernelCollection>(session)
        .map_err(|e| CompileError::CImageEmitFailed(format!("kernel collection missing: {e}")))?;
    if config.enable_search {
        world
            .component::<SearchStateComponent>(session)
            .map_err(|e| CompileError::CImageEmitFailed(format!("search state missing: {e}")))?;
    }

    // The constitutional `QuantizationResultComponent` is the plan. If
    // it is missing, the compile pipeline has not actually selected any
    // per-tensor representations yet — refusing to emit is the only
    // honest move.
    let plan = world
        .component::<QuantizationResultComponent>(session)
        .map_err(|e| {
            CompileError::CImageEmitFailed(format!(
                "no QuantizationResultComponent on session; cannot emit: {e}"
            ))
        })?
        .clone();

    // Source is still used for the output path and for header
    // provenance; we do not iterate its catalog at emission time.
    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::CImageEmitFailed("no current source resource".into()))?;

    let output_path = config
        .target_backends
        .first()
        .map(|_| PathBuf::from(format!("{}.cimage", source.0.identity.source_digest)))
        .unwrap_or_else(|| PathBuf::from("output.cimage"));

    let mut writer = UniversalCImageWriter::new(&output_path);
    writer.set_source(&source.0);

    // Plan digest — content-addressed identifier for the entire
    // per-tensor decision set. Stored alongside the file digest so
    // certification can verify the plan and the bytes agree.
    let plan_digest = quantization_plan_digest(&plan);

    for sel in &plan.selections {
        let entry = selection_to_payload_entry(sel).map_err(|e| {
            CompileError::CImageEmitFailed(format!(
                "plan selection {} invalid: {e}",
                sel.key
            ))
        })?;
        writer
            .add_tensor_payload(entry)
            .map_err(CompileError::CImageEmitFailed)?;
    }

    // Attach search trace if available.
    if let Ok(search) = world.component::<SearchStateComponent>(session) {
        writer.set_search_trace(search.trace.clone());
        writer.set_selection_receipt(search.selection_receipt.clone());
        if let Some(evidence) = &search.best_joint_tiling {
            writer.set_joint_tiling_evidence(evidence.clone());
        }
        if let Some(evidence) = &search.heterogeneous_workload_evidence {
            writer.set_heterogeneous_workload_evidence(evidence.clone());
        }
    }

    // Attach legalization report if available.
    if let Ok(legal) = world.component::<LegalizedPlan>(session) {
        writer.set_legalization_report(legal.report.clone());
    }

    // Attach kernel artifacts if available.
    if let Ok(kernels) = world.component::<KernelCollection>(session) {
        if let Some(receipt) = &kernels.uop_tuning_receipt {
            writer.set_uop_tuning_receipt(receipt.clone());
        }
        if let Some(capture) = &kernels.uop_capture {
            writer
                .add_uop_capture(capture)
                .map_err(CompileError::CImageEmitFailed)?;
            if !kernels.uop_strategy_captures.is_empty() {
                writer
                    .add_uop_strategy_captures(&kernels.uop_strategy_captures)
                    .map_err(CompileError::CImageEmitFailed)?;
            }
        } else {
            for artifact in &kernels.artifacts {
                writer.add_kernel_artifact(artifact.clone());
            }
            if let Some(manifest) = kernels.lowered_manifests.first() {
                let plan_json = serde_json::to_string(manifest).map_err(|error| {
                    CompileError::CImageEmitFailed(format!("serialize execution plan: {error}"))
                })?;
                writer.set_execution_plan(plan_json);
            }
        }
    }
    if let Some(manifest) = world.get_extension::<ModelManifestResource>() {
        writer
            .set_model_manifest(manifest.0.clone())
            .map_err(CompileError::CImageEmitFailed)?;
    }

    // Attach events.
    if let Some(sink) = world.get_resource::<VecEventSink>() {
        writer.set_events(sink.events().to_vec());
    }

    // Finalize and capture the digest.
    let _digest = writer
        .finalize()
        .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;
    let artifact_digest = hex::encode(sha2::Sha256::digest(
        std::fs::read(&output_path)
            .map_err(|e| CompileError::CImageEmitFailed(format!("read emitted CImage: {e}")))?,
    ));

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Emitted;
    }

    world
        .insert_component(
            session,
            CImageArtifact {
                output_path: output_path.clone(),
                digest: artifact_digest,
                schema_version: "1.1".into(),
            },
        )
        .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;

    // Also store the plan digest on the artifact receipt so
    // certification and downstream receipts can verify the plan ↔
    // bytes correspondence without re-deriving the plan.
    world.set_extension(CImagePlanDigest(plan_digest));

    Ok(())
}

/// Compute a stable digest of the constitutional
/// `QuantizationResultComponent`. Used to bind a CImage artifact to
/// the plan that produced it. `pub(super)` so the certification stage
/// can re-derive the digest and compare.
pub(super) fn quantization_plan_digest(plan: &QuantizationResultComponent) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(plan.source_digest.as_bytes());
    hasher.update([0u8]);
    hasher.update(plan.target_hardware.as_bytes());
    hasher.update([0u8]);
    for sel in &plan.selections {
        hasher.update(selection_digest_bytes(sel));
        hasher.update([0xffu8]);
    }
    hasher.update(plan.default_format.as_bytes());
    hasher.update([0u8]);
    hasher.update(plan.schema_version.to_le_bytes());
    hasher.finalize().into()
}

fn selection_digest_bytes(sel: &QuantizedTensorSelectionComponent) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(sel.key.as_bytes());
    hasher.update([0u8]);
    hasher.update([sel.format_discriminant]);
    hasher.update([sel.tensor_type_discriminant]);
    hasher.update(sel.dim_m.to_le_bytes());
    hasher.update(sel.dim_n.to_le_bytes());
    hasher.update(sel.effective_bpp.to_le_bytes());
    hasher.update(sel.payload_bytes.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(&sel.payload);
    hasher.finalize().to_vec()
}

/// Convert a constitutional `QuantizedTensorSelectionComponent` to a
/// `TensorPayloadEntry` ready for `UniversalCImageWriter`.
///
/// Returns an error if either discriminant is unrecognized (i.e. the
/// plan was written by a newer or older schema version than this
/// compiler understands). That is a hard failure, not a silent
/// fallback.
fn selection_to_payload_entry(
    sel: &QuantizedTensorSelectionComponent,
) -> Result<TensorPayloadEntry, String> {
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;
    let format = TensorFormat::from_discriminant_byte(sel.format_discriminant)
        .ok_or_else(|| {
            format!(
                "unknown TensorFormat discriminant {}",
                sel.format_discriminant
            )
        })?;
    let tensor_type = crate::cimage::TensorType::from_discriminant_byte(sel.tensor_type_discriminant)
        .ok_or_else(|| {
            format!(
                "unknown TensorType discriminant {}",
                sel.tensor_type_discriminant
            )
        })?;

    if (sel.payload.len() as u64) != sel.payload_bytes {
        return Err(format!(
            "payload length {} does not match recorded payload_bytes {}",
            sel.payload.len(),
            sel.payload_bytes
        ));
    }

    Ok(TensorPayloadEntry {
        name: sel.key.clone(),
        payload: sel.payload.clone(),
        representation: format!("{:?}", format),
        effective_bpp: sel.effective_bpp,
        dim_m: sel.dim_m,
        dim_n: sel.dim_n,
        tensor_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileConfig;
    use crate::compilation_systems::system_build_quantization_result;
    use crate::ecs::components::{
        CImageArtifact, CompilationSession, KernelCollection, LegalizedPlan, SearchStateComponent,
        SessionStatus, SourceModel, SpatialGraphComponent, TensorCollection,
    };
    use crate::ecs::orchestrator::CompilationOrchestrator;
    use crate::ecs::resources::{SessionHandle, SourceAdapterList};
    use crate::legalize::LegalizationReport;
    use crate::SearchTrace;
    use crate::cimage::CImageReader;
    use prism_ecs_constitutional::compilation::{
        QuantizationResultComponent, QuantizedTensorSelectionComponent,
    };
    use prism_ecs_core::entity::{Entity, EntityKind};
    use prism_ecs_core::identity::SourceFormat;
    use prism_ecs_core::world::World;
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;
    use prism_ecs_kernel::BackendKind;
    use prism_ecs_quantization::cimage::TensorType;
    use prism_ecs_source::{SourceIdentity, TensorCatalog, TensorDataProvider, TensorDescriptor};
    use std::io::{Read, Seek, SeekFrom};

    /// Minimal test source with one tensor whose provider returns
    /// bytes that look like the source dtype. The plan then supplies
    /// different bytes; the CImage must contain the plan's bytes.
    struct MarkerProvider;

    impl TensorDataProvider for MarkerProvider {
        fn read_tensor(&self, tensor: &TensorDescriptor) -> Result<Vec<u8>, prism_ecs_source::SourceError> {
            Ok(vec![0xAA; tensor.data_size_bytes as usize])
        }
    }

    fn make_test_source_with_provider(unique: &str) -> prism_ecs_source::CanonicalSource {
        let tensors = vec![TensorDescriptor {
            name: "weight".into(),
            shape: vec![4, 4],
            dtype: "f16".into(),
            byte_offset: 0,
            byte_length: 4 * 4 * 2,
            element_size: 2,
            original_dtype: "F16".into(),
            data_offset: None,
            data_size_bytes: 4 * 4 * 2,
            layout: "row-major".into(),
        }];
        let catalog = TensorCatalog {
            tensors,
            ..Default::default()
        };
        prism_ecs_source::CanonicalSource {
            identity: SourceIdentity {
                format: SourceFormat::SafeTensors,
                source_digest: format!("test-digest-{unique}"),
                model_family: "test".into(),
                architecture: "test".into(),
            },
            catalog,
            provider: Some(std::sync::Arc::new(MarkerProvider)),
            capabilities: Default::default(),
        }
    }

    fn install_session(world: &mut World, source: prism_ecs_source::CanonicalSource) -> Entity {
        let spawned = world
            .spawn(EntityKind::Session, None)
            .expect("spawn session")
            .entity;
        world
            .insert_component(
                spawned,
                CompilationSession {
                    config: CompileConfig {
                        target_backends: vec![BackendKind::CPU],
                        ..Default::default()
                    },
                    status: SessionStatus::KernelsGenerated,
                    session_id: "test".into(),
                },
            )
            .expect("insert session");
        let _ = world.insert_component(
            spawned,
            LegalizedPlan {
                report: LegalizationReport {
                    valid: true,
                    tensor_layout_valid: vec![],
                },
                is_valid: true,
            },
        );
        let _ = world.insert_component(
            spawned,
            KernelCollection {
                artifacts: vec![],
                kernel_count: 0,
                lowered_manifests: vec![],
                uop_capture: None,
                uop_strategy_captures: vec![],
                uop_tuning_receipt: None,
            },
        );
        let _ = world.insert_component(
            spawned,
            SearchStateComponent {
                trace: SearchTrace::default(),
                candidates_evaluated: 0,
                generations_completed: 0,
                format_plan: None,
                best_joint_tiling: None,
                selection_receipt: crate::search::SearchSelectionReceipt {
                    schema_version: "test".into(),
                    search_id: "test".into(),
                    evaluator: "test".into(),
                    evidence_source: "test".into(),
                    production_evidence: false,
                    candidates_evaluated: 0,
                    measured_candidates: 0,
                    selected_candidate_digest: None,
                    fallback_reason: None,
                    receipt_digest: "test".into(),
                },
                heterogeneous_workload_evidence: None,
                deployment_archive: Default::default(),
                selected_deployment_digest: None,
            },
        );
        world.set_extension(CurrentSource(source));
        world
            .insert_resource(SessionHandle(spawned))
            .expect("session handle");
        spawned
    }

    /// `system_emit_cimage` must fail if no `QuantizationResultComponent`
    /// is on the session. The previous bug was that the emitter
    /// silently read the source catalog instead.
    #[test]
    fn emit_requires_quantization_result_component() {
        let mut world = World::new();
        let source = make_test_source_with_provider("emit-requires");
        let _session = install_session(&mut world, source.clone());
        let result = system_emit_cimage(&mut world);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("QuantizationResultComponent"),
            "error must name the missing component: {msg}"
        );
    }

    /// `system_emit_cimage` writes the plan's bytes, not the source's.
    /// The plan supplies 0xCC bytes; the source provider supplies 0xAA
    /// bytes. The CImage header record for the tensor must record
    /// 0xCC bytes.
    #[test]
    fn emit_writes_plan_bytes_not_source_bytes() {
        let mut world = World::new();
        let source = make_test_source_with_provider("write-plan-bytes");
        let session = install_session(&mut world, source.clone());

        let plan = QuantizationResultComponent {
            source_digest: source.identity.source_digest.clone(),
            target_hardware: "default".into(),
            selections: vec![QuantizedTensorSelectionComponent {
                key: "weight".into(),
                format_discriminant: TensorFormat::Palettized4Bit.discriminant_byte(),
                payload: vec![0xCC; 8],
                tensor_type_discriminant: TensorType::Blob.discriminant_byte()[0],
                dim_m: 4,
                dim_n: 4,
                effective_bpp: 4.0,
                payload_bytes: 8,
            }],
            default_format: "Palettized4Bit".into(),
            schema_version: 1,
        };
        world
            .insert_component(session, plan)
            .expect("insert plan");

        system_emit_cimage(&mut world).expect("emit ok");

        let artifact = world
            .component::<CImageArtifact>(session)
            .expect("artifact")
            .clone();
        assert!(artifact.output_path.exists(), "CImage must exist on disk");

        let reader = CImageReader::open(&artifact.output_path).expect("open");
        let entry = reader
            .header
            .tensors
            .get("weight")
            .expect("weight record");
        assert_eq!(entry.size, 8, "header size must match plan payload_bytes");
        assert_eq!(entry.dim_m, 4);
        assert_eq!(entry.dim_n, 4);

        let mut file = std::fs::File::open(&artifact.output_path).expect("open file");
        file.seek(SeekFrom::Start(entry.offset))
            .expect("seek to payload");
        let mut buf = vec![0u8; entry.size as usize];
        file.read_exact(&mut buf).expect("read payload");
        assert!(buf.iter().all(|&b| b == 0xCC), "payload must be 0xCC");
    }

    /// `system_build_quantization_result` produces a plan that the
    /// emitter accepts. The legacy search path that did not call
    /// `prism_ecs_quantization::build_quantization_plan` still works
    /// because this system synthesizes the plan from the source
    /// catalog.
    #[test]
    fn build_plan_synthesizes_from_source_catalog() {
        let mut world = World::new();
        let source = make_test_source_with_provider("build-plan");
        let session = install_session(&mut world, source);

        system_build_quantization_result(&mut world).expect("build plan ok");

        let plan = world
            .component::<QuantizationResultComponent>(session)
            .expect("plan must be attached");
        assert_eq!(plan.selections.len(), 1);
        let sel = &plan.selections[0];
        assert_eq!(sel.key, "weight");
        assert_eq!(sel.dim_m, 4);
        assert_eq!(sel.dim_n, 4);
        assert!(sel.payload.iter().all(|&b| b == 0xAA));
        assert_eq!(sel.payload_bytes, 32); // 4*4*2 from the catalog

        system_emit_cimage(&mut world).expect("emit ok");
    }
}
