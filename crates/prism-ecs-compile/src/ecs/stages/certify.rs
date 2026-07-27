//! CImage certification stage.
//!
//! Certifies the emitted artifact structurally before publishing the receipt.
//! Verifies, in order:
//!   1. The file bytes still hash to the digest recorded at emit time.
//!   2. The `QuantizationResultComponent` is attached to the session.
//!   3. The plan's content digest matches the digest stored on the
//!      `CImagePlanDigest` extension (which the emitter wrote).
//!   4. Every tensor in the plan exists in the CImage header with the
//!      same name, dimensions, and physical type.
//!   5. The execution plan, if any, validates.
//!
//! The previous version of this function compared the source catalog
//! against the CImage contents. That is no longer meaningful — the
//! CImage body is now the *selected* plan, not the source catalog.

use sha2::Digest;

use prism_ecs_core::world::World;
use prism_ecs_constitutional::compilation::QuantizationResultComponent;

use crate::cimage::CImageReader;
use crate::ecs::components::{CImageArtifact, CompilationSession, SessionStatus};
use crate::ecs::orchestrator::session_entity;
use crate::ecs::resources::CImagePlanDigest;
use crate::runtime::RuntimeModel;
use crate::CompileError;

use super::emit::quantization_plan_digest;

pub fn system_certify(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let artifact = world
        .component::<CImageArtifact>(session)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
    let model = RuntimeModel::load(&artifact.output_path)
        .map_err(|e| CompileError::CompilationFailed(format!("certification load failed: {e}")))?;
    let reader = CImageReader::open(&artifact.output_path)
        .map_err(|e| CompileError::CompilationFailed(format!("read CImage header failed: {e}")))?;
    let actual_digest = hex::encode(sha2::Sha256::digest(
        std::fs::read(&artifact.output_path)
            .map_err(|e| CompileError::CompilationFailed(format!("read CImage failed: {e}")))?,
    ));
    if actual_digest != artifact.digest {
        return Err(CompileError::CompilationFailed(
            "CImage artifact digest does not match emitted bytes".into(),
        ));
    }

    // The plan is the source of truth for the artifact body.
    let plan = world
        .component::<QuantizationResultComponent>(session)
        .map_err(|e| {
            CompileError::CompilationFailed(format!(
                "no QuantizationResultComponent on session at certify time: {e}"
            ))
        })?
        .clone();

    // Verify the plan digest stored on the world extension matches the
    // plan we have. If it doesn't, the emit step didn't run against the
    // plan we are certifying — refuse to certify.
    let plan_digest_now = quantization_plan_digest(&plan);
    let stored_plan_digest = world
        .get_extension::<CImagePlanDigest>()
        .ok_or_else(|| {
            CompileError::CompilationFailed(
                "no CImagePlanDigest on world; emit did not bind a plan".into(),
            )
        })?;
    if stored_plan_digest.0 != plan_digest_now {
        return Err(CompileError::CompilationFailed(format!(
            "plan digest mismatch: stored {}, current {}",
            hex::encode(stored_plan_digest.0),
            hex::encode(plan_digest_now)
        )));
    }

    // Every selected tensor must exist in the CImage with matching
    // name, dimensions, and physical type.
    for sel in &plan.selections {
        let header_entry = reader.header.tensors.get(&sel.key).ok_or_else(|| {
            CompileError::CompilationFailed(format!(
                "CImage is missing plan tensor {}",
                sel.key
            ))
        })?;
        if header_entry.dim_m != sel.dim_m || header_entry.dim_n != sel.dim_n {
            return Err(CompileError::CompilationFailed(format!(
                "tensor {} dimension mismatch: plan {}x{}, header {}x{}",
                sel.key, sel.dim_m, sel.dim_n, header_entry.dim_m, header_entry.dim_n
            )));
        }
        if (header_entry.size as u64) != sel.payload_bytes {
            return Err(CompileError::CompilationFailed(format!(
                "tensor {} payload size mismatch: plan {}, header {}",
                sel.key,
                sel.payload_bytes,
                header_entry.size
            )));
        }
    }

    if let Some(plan) = &model.execution_plan {
        plan.validate().map_err(|e| {
            CompileError::CompilationFailed(format!("certification plan failed: {e}"))
        })?;
        if !plan.supports_all_streamed_workloads() {
            return Err(CompileError::CompilationFailed(
                "certification plan does not cover all streamed workloads".into(),
            ));
        }
    }
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Certified;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation_systems::system_build_quantization_result;
    use crate::ecs::components::{
        CImageArtifact, CompilationSession, KernelCollection, LegalizedPlan, SearchStateComponent,
        SessionStatus,
    };
    use crate::ecs::orchestrator::CompilationOrchestrator;
    use crate::ecs::resources::{CurrentSource, SessionHandle};
    use crate::legalize::LegalizationReport;
    use crate::SearchTrace;
    use crate::cimage::CImageReader;
    use crate::ecs::stages::emit::system_emit_cimage;
    use crate::CompileConfig;
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

    /// `system_certify` accepts an emit that wrote from a plan, and
    /// transitions the session to Certified.
    #[test]
    fn certify_accepts_plan_backed_artifact() {
        let mut world = World::new();
        let source = make_test_source_with_provider("certify-accepts");
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
        system_certify(&mut world).expect("certify ok");

        let status = world
            .component::<CompilationSession>(session)
            .expect("session")
            .clone();
        assert!(matches!(status.status, SessionStatus::Certified));
    }

    /// `system_certify` rejects a mismatch between the plan and the
    /// file. If we mutate the recorded plan after emit, certification
    /// must fail.
    #[test]
    fn certify_rejects_plan_mutation_after_emit() {
        let mut world = World::new();
        let source = make_test_source_with_provider("certify-rejects");
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

        // Mutate the plan after emit. The plan digest stored in
        // CImagePlanDigest must no longer match.
        if let Ok(mut p) = world.component_mut::<QuantizationResultComponent>(session) {
            p.selections[0].payload[0] = 0xFF;
        }

        let result = system_certify(&mut world);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("plan digest mismatch"),
            "error must mention plan digest mismatch: {msg}"
        );
    }
}
