use crate::ecs::adapter::CanonicalRole;
use crate::ecs::component::tensor::{CanonicalRoleComp, CodecFamilyComp, Shape};
use crate::ecs::plan::precision_plan::{
    PrecisionOverride, PrecisionOverrideReason, PrecisionPlan, PrecisionScope,
    PrecisionSelectionBasis, PrecisionSelector,
};
use crate::ecs::plan::CodecFamily;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, Component, EntityKind, SchedulePhase};
use serde::{Deserialize, Serialize};

// ── Component wrapper ─────────────────────────────────────────────────────

/// Component wrapper for attaching a PrecisionPlan to a model entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPlanComponent(pub PrecisionPlan);
impl Component for PrecisionPlanComponent {}

// ── Codec Selection System ────────────────────────────────────────────────

/// Selects a quantization codec family for each tensor based on its canonical role.
///
/// Mapping:
///   Q / K / V / O / QNorm / KNorm (attention) → Int8
///   Gate / Up / Down (MLP)                    → Q8_0
///   GateEx / UpEx / DownEx (routed experts)   → Q4_K
///   Embedding / LmHead                        → Fp16
///   RouterWeight / shared experts             → Int8
pub struct CodecSelectionSystem;

impl CompilerSystem for CodecSelectionSystem {
    fn name(&self) -> &str {
        "CodecSelectionSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Quantization
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let tensors: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for tensor in tensors {
            // Only process tensors with both Shape and CanonicalRoleComp.
            let _shape = match world.get_component::<Shape>(tensor) {
                Some(s) => s.clone(),
                None => continue,
            };
            let role = match world.get_component::<CanonicalRoleComp>(tensor) {
                Some(r) => r.0,
                None => continue,
            };

            let (codec, group_size) = select_codec_for_role(role);
            world.add_component(tensor, CodecFamilyComp(codec, group_size));
        }

        Ok(())
    }
}

/// Map a canonical role to its default codec family and group size.
fn select_codec_for_role(role: CanonicalRole) -> (CodecFamily, u32) {
    match role {
        // ── Attention projections — Int8 offers good perf/accuracy ─────
        CanonicalRole::Q(_)
        | CanonicalRole::K(_)
        | CanonicalRole::V(_)
        | CanonicalRole::O(_)
        | CanonicalRole::QNorm(_)
        | CanonicalRole::KNorm(_) => (CodecFamily::Int8, 0),

        // ── Normalisation layers — keep fp16 for stability ────────────
        CanonicalRole::AttnNorm(_) | CanonicalRole::MlpNorm(_) => (CodecFamily::Fp16, 0),

        // ── MLP hidden projections — Q8_0 for good compute density ────
        CanonicalRole::Gate(_) | CanonicalRole::Up(_) | CanonicalRole::Down(_) => {
            (CodecFamily::Q8_0, 32)
        }

        // ── Routed MoE experts — aggressive Q4_K for memory savings ───
        CanonicalRole::GateEx(_, _) | CanonicalRole::UpEx(_, _) | CanonicalRole::DownEx(_, _) => {
            (CodecFamily::Q4_K, 128)
        }

        CanonicalRole::RouterWeight(_) => (CodecFamily::Q8_0, 32),

        // ── Shared experts — Int8 for balanced density ─────────────────
        CanonicalRole::SharedGate
        | CanonicalRole::SharedUp
        | CanonicalRole::SharedDown
        | CanonicalRole::SharedGateL(_)
        | CanonicalRole::SharedUpL(_)
        | CanonicalRole::SharedDownL(_) => (CodecFamily::Int8, 0),

        // ── Input / output — keep fp16 or raw for fidelity ────────────
        CanonicalRole::Embedding | CanonicalRole::LmHead => (CodecFamily::Fp16, 0),
        CanonicalRole::FinalNorm => (CodecFamily::RawF32, 0),

        // ── Compressed / sparse attention ──────────────────────────────
        CanonicalRole::CompressWeight(_)
        | CanonicalRole::IndexerWeight(_)
        | CanonicalRole::WindowK(_)
        | CanonicalRole::WindowV(_) => (CodecFamily::Int8, 0),

        CanonicalRole::HCWeight(_) => (CodecFamily::Int8, 0),
    }
}

// ── Precision Plan System ─────────────────────────────────────────────────

/// Aggregates per-tensor codec assignments into a model-level PrecisionPlan.
/// Attaches the plan to each ModelEntity for downstream fusion scheduling
/// and backend capability evaluation.
pub struct PrecisionPlanSystem;

impl CompilerSystem for PrecisionPlanSystem {
    fn name(&self) -> &str {
        "PrecisionPlanSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Quantization
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let models: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        let tensors: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        let mut overrides: Vec<PrecisionOverride> = Vec::new();
        let mut default_codec = CodecFamily::RawF32;
        let mut has_any = false;

        for tensor in &tensors {
            if let Some(codec_comp) = world.get_component::<CodecFamilyComp>(*tensor) {
                has_any = true;
                overrides.push(PrecisionOverride {
                    selector: PrecisionSelector::TileIds(vec![tensor.0 as u32]),
                    codec: codec_comp.0,
                    reason: PrecisionOverrideReason::OperatorTailRescue,
                    byte_cost: 0,
                    expected_error_reduction: None,
                });
            }
        }

        if !has_any {
            default_codec = CodecFamily::RawF32;
        }

        let plan_id = {
            let hash = models.iter().fold(0u64, |acc, e| acc.wrapping_add(e.0));
            format!("plan-{:x}", hash)
        };

        let plan = PrecisionPlan {
            plan_id,
            scope: PrecisionScope::FusedGroup,
            default_codec,
            overrides,
            selection_basis: PrecisionSelectionBasis::StaticPolicy,
            evidence_level: crate::training_target::RequiredEvidenceLevel::WeightSpace,
        };

        for model in &models {
            world.add_component(*model, PrecisionPlanComponent(plan.clone()));
        }

        Ok(())
    }
}
