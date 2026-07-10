use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::ecs::cimage::receipts::EvidenceReceiptV0;
use crate::ecs::cimage::{
    CImageArtifactKind, CImageManifestV0, CImagePayloadKind, CImagePayloadRef,
    CImageReceiptRef, CImageTensorEntry, CImageWriter,
    ModelExecutionPlanSummary, PendingPayload, PendingReceipt, PhysicalTileLayout,
};
use crate::ecs::{
    component::{
        backend::CompiledBinary,
        quality::{AOTProfileMatch, AdmissionReceipt, QualityGateResult},
        tensor::{CanonicalRoleComp, CodecFamilyComp, DataType, LayerIndex, Shape},
    },
    CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase,
};
use crate::ecs::plan::{CodecFamily, DType as PlanDType, HardwareProfileId};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map the ECS component DType to the coarser execution-plan DType.
fn map_dtype(dt: crate::ecs::component::tensor::DType) -> PlanDType {
    use crate::ecs::component::tensor::DType;
    match dt {
        DType::F32 => PlanDType::F32,
        DType::F16 => PlanDType::F16,
        DType::BF16 => PlanDType::F16,
        DType::I8 => PlanDType::I8,
        DType::I4 => PlanDType::I8,
        DType::I2 => PlanDType::I8,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------
// CImageAssemblySystem
// ---------------------------------------------------------------------------

/// Assembles a CImage V0 artifact from the ECS world state.
///
/// Collects tensor manifests from Tensor entities, payload blobs from
/// Executable entities (CompiledBinary), and quality/admission receipts
/// from all entities.  Writes the result via [`CImageWriter::write_v0`].
pub struct CImageAssemblySystem {
    pub output_path: PathBuf,
}

impl CompilerSystem for CImageAssemblySystem {
    fn name(&self) -> &str {
        "CImageAssemblySystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // ── Model info ────────────────────────────────────────────────────
        let model_entities = world.entities_of_kind(EntityKind::Model);
        let model_family = model_entities
            .first()
            .and_then(|&e| world.name(e))
            .unwrap_or("unknown")
            .to_string();

        // ── Tensor entries ────────────────────────────────────────────────
        let tensor_entities = world.entities_of_kind(EntityKind::Tensor);
        let mut tensor_entries: Vec<CImageTensorEntry> = Vec::new();

        for &tensor in &tensor_entities {
            let name = world.name(tensor).unwrap_or("unnamed");
            let shape = world.get_component::<Shape>(tensor);
            let dtype = world.get_component::<DataType>(tensor);
            let codec = world.get_component::<CodecFamilyComp>(tensor);
            let role = world.get_component::<CanonicalRoleComp>(tensor);
            let layer = world.get_component::<LayerIndex>(tensor);

            let logical_shape = shape.map(|s| s.0.clone()).unwrap_or_default();
            let source_dtype = dtype.map_or(PlanDType::F32, |d| map_dtype(d.0));
            let (codec_family, group_size) = codec
                .map(|c| (c.0, c.1))
                .unwrap_or((CodecFamily::RawF32, 0));
            let tensor_class = role.map_or("Unknown".into(), |r| format!("{:?}", r.0));
            let layer_idx = layer.map_or(0, |l| l.0);

            let tile_n = 64u32;
            let groups_per_tile = if group_size > 0 {
                tile_n / group_size
            } else {
                0
            };

            let physical_layout = PhysicalTileLayout {
                tile_m: 64,
                tile_n,
                tiles_per_row: 1,
                total_tiles: 1,
                padded_cols: tile_n,
                group_size,
                groups_per_tile,
                packed_bytes_per_tile: 4096,
                metadata_f32_per_tile: 16,
            };

            tensor_entries.push(CImageTensorEntry {
                tensor_id: format!("tensor_{}", tensor.0),
                tensor_key: format!("layer_{}/{}", layer_idx, name),
                tensor_class,
                logical_shape,
                source_dtype,
                codec: codec_family,
                precision_plan: None,
                physical_layout,
                payload_ref: CImagePayloadRef::Single {
                    payload_id: format!("p_{}", name),
                },
                raw_f32_reference_ref: None,
                tensor_sha256: sha256_hex(&[]),
                validation_digest: None,
            });
        }

        // ── Payloads from compiled binaries ───────────────────────────────
        let exec_entities = world.entities_of_kind(EntityKind::Executable);
        let mut pending_payloads: Vec<PendingPayload> = Vec::new();

        for &exec in &exec_entities {
            if let Some(binary) = world.get_component::<CompiledBinary>(exec) {
                let name = world.name(exec).unwrap_or("unnamed");
                pending_payloads.push(PendingPayload {
                    payload_id: format!("payload_{}", name),
                    payload_kind: CImagePayloadKind::PackedTensorCodes,
                    codec: Some(format!("{:?}", binary.format)),
                    alignment_bytes: 1,
                    bytes: binary.data.clone(),
                });
            }
        }

        // ── Quality gate / admission receipts from all entities ───────────
        let mut receipt_refs: Vec<CImageReceiptRef> = Vec::new();
        let mut pending_receipts: Vec<PendingReceipt> = Vec::new();
        let entity_count = world.entity_count();

        for id in 1..=entity_count {
            let entity = CompEntity(id as u64);

            if let Some(qg) = world.get_component::<QualityGateResult>(entity) {
                let rid = format!("quality_gate_{}", id);
                let json = serde_json::to_vec(qg)?;
                receipt_refs.push(CImageReceiptRef {
                    receipt_id: rid.clone(),
                    receipt_kind: "QualityGateResult".into(),
                });
                pending_receipts.push(PendingReceipt {
                    receipt_id: rid,
                    receipt_kind: "QualityGateResult".into(),
                    bytes: json,
                });
            }

            if let Some(aot) = world.get_component::<AOTProfileMatch>(entity) {
                let rid = format!("aot_profile_{}", id);
                let json = serde_json::to_vec(aot)?;
                receipt_refs.push(CImageReceiptRef {
                    receipt_id: rid.clone(),
                    receipt_kind: "AOTProfileMatch".into(),
                });
                pending_receipts.push(PendingReceipt {
                    receipt_id: rid,
                    receipt_kind: "AOTProfileMatch".into(),
                    bytes: json,
                });
            }

            if let Some(adm) = world.get_component::<AdmissionReceipt>(entity) {
                let rid = format!("admission_{}", id);
                let json = serde_json::to_vec(adm)?;
                receipt_refs.push(CImageReceiptRef {
                    receipt_id: rid.clone(),
                    receipt_kind: "AdmissionReceipt".into(),
                });
                pending_receipts.push(PendingReceipt {
                    receipt_id: rid,
                    receipt_kind: "AdmissionReceipt".into(),
                    bytes: json,
                });
            }
        }

        // ── Execution plan summary ────────────────────────────────────────
        let tensor_refs: Vec<String> = tensor_entries
            .iter()
            .map(|t| t.tensor_key.clone())
            .collect();

        let plan = ModelExecutionPlanSummary {
            plan_id: "plan_0".into(),
            region_count: exec_entities.len() as u32,
            total_kernel_ops: exec_entities.len() as u32,
            total_input_bytes: 0,
            total_output_bytes: 0,
            tensor_refs,
        };

        // ── Manifest ──────────────────────────────────────────────────────
        let manifest = CImageManifestV0 {
            schema_version: 0,
            model_family,
            artifact_kind: CImageArtifactKind::ModelShard,
            source_model_digest: None,
            compiler_policy_digest: "policy_digest_0".into(),
            layout_profile: HardwareProfileId::AppleMProBalanced,
            tensors: tensor_entries,
            execution_plan: plan,
            receipts: receipt_refs,
            assistant_graph: None,
            state_store_schema: None,
        };

        // ── Write via CImageWriter ────────────────────────────────────────
        CImageWriter::write_v0(
            &self.output_path,
            manifest,
            pending_payloads,
            pending_receipts,
        )
        .map_err(|e| anyhow::anyhow!("CImage write_v0 failed: {e}"))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ReceiptSigningSystem
// ---------------------------------------------------------------------------

/// Collects quality-gate, AOT-profile, and admission receipts from all
/// entities, signs each bundle with a SHA-256 digest of its serialized
/// evidence, and writes [`EvidenceReceiptV0`] JSON records to a designated
/// output directory.
pub struct ReceiptSigningSystem {
    pub output_dir: PathBuf,
}

impl CompilerSystem for ReceiptSigningSystem {
    fn name(&self) -> &str {
        "ReceiptSigningSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entity_count = world.entity_count();
        let mut evidence_receipts: Vec<EvidenceReceiptV0> = Vec::new();

        for id in 1..=entity_count {
            let entity = CompEntity(id as u64);
            let qg = world.get_component::<QualityGateResult>(entity);
            let aot = world.get_component::<AOTProfileMatch>(entity);
            let adm = world.get_component::<AdmissionReceipt>(entity);
            if qg.is_none() && aot.is_none() && adm.is_none() {
                continue;
            }

            // Combine all evidence for this entity and compute a signature.
            let mut hasher = Sha256::new();
            let mut kind_parts: Vec<&str> = Vec::new();
            if let Some(qg) = qg {
                let json = serde_json::to_vec(qg)?;
                hasher.update(&json);
                kind_parts.push("QualityGateResult");
            }
            if let Some(aot) = aot {
                let json = serde_json::to_vec(aot)?;
                hasher.update(&json);
                kind_parts.push("AOTProfileMatch");
            }
            if let Some(adm) = adm {
                let json = serde_json::to_vec(adm)?;
                hasher.update(&json);
                kind_parts.push("AdmissionReceipt");
            }

            let digest = format!("{:x}", hasher.finalize());

            evidence_receipts.push(EvidenceReceiptV0 {
                receipt_id: format!("evidence_{}", id),
                receipt_kind: kind_parts.join("|"),
                manifest_digest: digest,
                shard_validation: None,
                load_receipt: None,
            });
        }

        if evidence_receipts.is_empty() {
            return Ok(());
        }

        // Write each evidence receipt as its own JSON file.
        std::fs::create_dir_all(&self.output_dir)?;
        for evidence in &evidence_receipts {
            let json = serde_json::to_vec_pretty(evidence)?;
            let filename = format!("{}.json", evidence.receipt_id);
            std::fs::write(self.output_dir.join(&filename), &json)?;
        }

        // Write a summary manifest listing all evidence receipts.
        let summary_json = serde_json::to_vec_pretty(&evidence_receipts)?;
        std::fs::write(
            self.output_dir.join("evidence_manifest.json"),
            &summary_json,
        )?;

        Ok(())
    }
}
