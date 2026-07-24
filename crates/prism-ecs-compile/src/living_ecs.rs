//! Final constitutional ECS propagation for all Living CImage refinement domains.

use crate::adapter_training::{AdapterArtifact, AdapterTrainingReceipt, AdapterTrainingRequest};
use crate::agentic_workload::AgenticCalibrationCorpus;
use crate::engram_learning::{EngramAdmissionReceipt, EngramGeneration};
use crate::execution_graph_evolution::{ExecutionGraphAdmissionReceipt, TargetExecutionGraph};
use crate::knowledge_correction::{KnowledgeCorrectionProposal, KnowledgeCorrectionReceipt};
use crate::kv_cache_compaction::{KvCompactionCandidate, KvCompactionReceipt};
use crate::kv_cache_evolution::{KvCacheCandidate, KvCacheEvaluationReceipt};
use crate::living_cimage::{
    CImageGeneration, LivingCImage, LivingCImageGeneration, LivingCImageId,
};
use crate::living_promotion::{LivingGenerationPromotionReceipt, LivingGenerationRollbackReceipt};
use crate::model_export::{ModelExportReceipt, ModelExportRequest};
use crate::progressive_ternary::{ProgressiveTernaryPlan, ProgressiveTernaryReceipt};
use crate::shadow_calibration::ShadowCalibrationReceipt;
use crate::speculative_inference::{SpeculativeInferenceCandidate, SpeculativeInferenceReceipt};
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LivingEntityKind {
    LivingCImage,
    Generation,
    AdaptationJob,
    CalibrationCorpus,
    EngramGeneration,
    KnowledgeCorrection,
    ModelExport,
    ExecutionGraphSearch,
    PromotionDecision,
    RollbackDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LivingLifecycle {
    Proposed,
    Calibrating,
    Training,
    Evaluating,
    Admitted,
    Promoted,
    Rejected,
    Superseded,
    RolledBack,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingCImageAuthority(pub LivingCImage);
impl Component for LivingCImageAuthority {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingGenerationAuthority(pub LivingCImageGeneration);
impl Component for LivingGenerationAuthority {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingCalibrationAuthority(pub AgenticCalibrationCorpus);
impl Component for LivingCalibrationAuthority {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingAdaptationIndex {
    pub living_cimage_id: LivingCImageId,
    pub generation: CImageGeneration,
    pub progressive_ternary: Option<(ProgressiveTernaryPlan, ProgressiveTernaryReceipt)>,
    pub speculative_inference: Option<(SpeculativeInferenceCandidate, SpeculativeInferenceReceipt)>,
    pub kv_cache_evolution: Option<(KvCacheCandidate, KvCacheEvaluationReceipt)>,
    pub kv_cache_compaction: Option<(KvCompactionCandidate, KvCompactionReceipt)>,
    pub adapter_training: Option<(
        AdapterTrainingRequest,
        AdapterArtifact,
        AdapterTrainingReceipt,
    )>,
    pub engram_generation: Option<(EngramGeneration, EngramAdmissionReceipt)>,
    pub knowledge_correction: Option<(KnowledgeCorrectionProposal, KnowledgeCorrectionReceipt)>,
    pub model_exports: Vec<(ModelExportRequest, ModelExportReceipt)>,
    pub execution_graphs: Vec<(TargetExecutionGraph, ExecutionGraphAdmissionReceipt)>,
    pub shadow_receipt: Option<ShadowCalibrationReceipt>,
    pub promotion_receipt: Option<LivingGenerationPromotionReceipt>,
    pub rollback_receipts: Vec<LivingGenerationRollbackReceipt>,
}
impl Component for LivingAdaptationIndex {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingLifecycleComponent {
    pub entity_kind: LivingEntityKind,
    pub lifecycle: LivingLifecycle,
    pub generation_digest: String,
    pub last_event_sequence: u64,
}
impl Component for LivingLifecycleComponent {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LivingCommandKind {
    CreateLivingCImage,
    ProposeGeneration,
    AttachCalibrationCorpus,
    AttachProgressiveTernary,
    AttachSpeculativeInference,
    AttachKvCacheEvolution,
    AttachKvCacheCompaction,
    AttachAdapterTraining,
    AttachEngramGeneration,
    AttachKnowledgeCorrection,
    AttachModelExport,
    AttachExecutionGraph,
    AttachShadowEvaluation,
    PromoteGeneration,
    RejectGeneration,
    RollbackGeneration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingCommand {
    pub command_id: String,
    pub kind: LivingCommandKind,
    pub living_cimage_id: String,
    pub target_generation_digest: Option<String>,
    pub expected_active_generation_digest: Option<String>,
    pub artifact_digest: Option<String>,
    pub receipt_digest: Option<String>,
    pub actor: String,
    pub idempotency_key: String,
    #[serde(default)]
    pub command_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingEvent {
    pub sequence: u64,
    pub event_id: String,
    pub command_digest: String,
    pub living_cimage_id: String,
    pub generation_digest: Option<String>,
    pub event_kind: String,
    pub artifact_digest: Option<String>,
    pub receipt_digest: Option<String>,
    pub previous_active_generation_digest: Option<String>,
    pub next_active_generation_digest: Option<String>,
    pub actor: String,
    pub event_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LivingReplayRegistry {
    pub appliers: BTreeMap<String, String>,
}

impl LivingReplayRegistry {
    pub fn canonical() -> Self {
        let mut appliers = BTreeMap::new();
        for kind in [
            "living_cimage_created",
            "generation_proposed",
            "calibration_attached",
            "progressive_ternary_attached",
            "speculative_inference_attached",
            "kv_cache_evolution_attached",
            "kv_cache_compaction_attached",
            "adapter_training_attached",
            "engram_generation_attached",
            "knowledge_correction_attached",
            "model_export_attached",
            "execution_graph_attached",
            "shadow_evaluation_attached",
            "generation_promoted",
            "generation_rejected",
            "generation_rolled_back",
        ] {
            appliers.insert(kind.into(), format!("replay_{kind}"));
        }
        Self { appliers }
    }

    pub fn supports(&self, event_kind: &str) -> bool {
        self.appliers.contains_key(event_kind)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingProjection {
    pub living_cimage_id: String,
    pub active_generation_digest: String,
    pub generation_count: usize,
    pub admitted_domains: Vec<String>,
    pub latest_receipts: BTreeMap<String, String>,
    pub rollback_targets: Vec<String>,
    pub projection_checkpoint: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LivingEcsError {
    #[error("command identity is incomplete")]
    MissingCommandIdentity,
    #[error("command digest mismatch")]
    CommandDigestMismatch,
    #[error("stale active generation")]
    StaleGeneration,
    #[error("event is not replay registered")]
    ReplayRegistrationGap,
    #[error("promotion lacks an admitted receipt")]
    PromotionReceiptMissing,
    #[error("rollback target is not known")]
    UnknownRollbackTarget,
}

impl LivingCommand {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.command_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, LivingEcsError> {
        if self.command_id.is_empty()
            || self.living_cimage_id.is_empty()
            || self.actor.is_empty()
            || self.idempotency_key.is_empty()
        {
            return Err(LivingEcsError::MissingCommandIdentity);
        }
        self.command_digest = self.canonical_digest();
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), LivingEcsError> {
        if self.command_digest != self.canonical_digest() {
            return Err(LivingEcsError::CommandDigestMismatch);
        }
        Ok(())
    }
}

pub fn validate_command_against_authority(
    command: &LivingCommand,
    authority: &LivingCImage,
) -> Result<(), LivingEcsError> {
    command.verify()?;
    let active = authority
        .generations
        .get(authority.active_generation.0 as usize)
        .map(|generation| generation.generation_digest.as_str())
        .unwrap_or_default();
    if let Some(expected) = command.expected_active_generation_digest.as_deref() {
        if expected != active {
            return Err(LivingEcsError::StaleGeneration);
        }
    }
    if matches!(command.kind, LivingCommandKind::PromoteGeneration)
        && command
            .receipt_digest
            .as_deref()
            .unwrap_or_default()
            .is_empty()
    {
        return Err(LivingEcsError::PromotionReceiptMissing);
    }
    Ok(())
}

pub fn event_from_command(
    sequence: u64,
    command: &LivingCommand,
    previous_active_generation_digest: Option<String>,
    next_active_generation_digest: Option<String>,
) -> Result<LivingEvent, LivingEcsError> {
    command.verify()?;
    let event_kind = match command.kind {
        LivingCommandKind::CreateLivingCImage => "living_cimage_created",
        LivingCommandKind::ProposeGeneration => "generation_proposed",
        LivingCommandKind::AttachCalibrationCorpus => "calibration_attached",
        LivingCommandKind::AttachProgressiveTernary => "progressive_ternary_attached",
        LivingCommandKind::AttachSpeculativeInference => "speculative_inference_attached",
        LivingCommandKind::AttachKvCacheEvolution => "kv_cache_evolution_attached",
        LivingCommandKind::AttachKvCacheCompaction => "kv_cache_compaction_attached",
        LivingCommandKind::AttachAdapterTraining => "adapter_training_attached",
        LivingCommandKind::AttachEngramGeneration => "engram_generation_attached",
        LivingCommandKind::AttachKnowledgeCorrection => "knowledge_correction_attached",
        LivingCommandKind::AttachModelExport => "model_export_attached",
        LivingCommandKind::AttachExecutionGraph => "execution_graph_attached",
        LivingCommandKind::AttachShadowEvaluation => "shadow_evaluation_attached",
        LivingCommandKind::PromoteGeneration => "generation_promoted",
        LivingCommandKind::RejectGeneration => "generation_rejected",
        LivingCommandKind::RollbackGeneration => "generation_rolled_back",
    };
    let registry = LivingReplayRegistry::canonical();
    if !registry.supports(event_kind) {
        return Err(LivingEcsError::ReplayRegistrationGap);
    }
    let mut event = LivingEvent {
        sequence,
        event_id: format!("living-event-{sequence}"),
        command_digest: command.command_digest.clone(),
        living_cimage_id: command.living_cimage_id.clone(),
        generation_digest: command.target_generation_digest.clone(),
        event_kind: event_kind.into(),
        artifact_digest: command.artifact_digest.clone(),
        receipt_digest: command.receipt_digest.clone(),
        previous_active_generation_digest,
        next_active_generation_digest,
        actor: command.actor.clone(),
        event_digest: String::new(),
    };
    event.event_digest = digest_json(&event);
    Ok(event)
}

pub fn project_living_cimage(
    authority: &LivingCImage,
    index: &LivingAdaptationIndex,
    checkpoint: u64,
) -> LivingProjection {
    let active_generation_digest = authority
        .generations
        .get(authority.active_generation.0 as usize)
        .map(|generation| generation.generation_digest.clone())
        .unwrap_or_default();
    let mut admitted_domains = Vec::new();
    let mut latest_receipts = BTreeMap::new();
    macro_rules! domain {
        ($option:expr, $name:literal, $receipt:expr) => {
            if let Some(value) = $option {
                admitted_domains.push($name.into());
                latest_receipts.insert($name.into(), $receipt(value));
            }
        };
    }
    domain!(
        index.progressive_ternary.as_ref(),
        "progressive_ternary",
        |value: &(ProgressiveTernaryPlan, ProgressiveTernaryReceipt)| value
            .1
            .receipt_digest
            .clone()
    );
    domain!(
        index.speculative_inference.as_ref(),
        "speculative_inference",
        |value: &(SpeculativeInferenceCandidate, SpeculativeInferenceReceipt)| value
            .1
            .receipt_digest
            .clone()
    );
    domain!(
        index.kv_cache_evolution.as_ref(),
        "kv_cache_evolution",
        |value: &(KvCacheCandidate, KvCacheEvaluationReceipt)| value.1.receipt_digest.clone()
    );
    domain!(
        index.kv_cache_compaction.as_ref(),
        "kv_cache_compaction",
        |value: &(KvCompactionCandidate, KvCompactionReceipt)| value.1.receipt_digest.clone()
    );
    domain!(
        index.adapter_training.as_ref(),
        "adapter_training",
        |value: &(
            AdapterTrainingRequest,
            AdapterArtifact,
            AdapterTrainingReceipt
        )| value.2.receipt_digest.clone()
    );
    domain!(
        index.engram_generation.as_ref(),
        "engram_generation",
        |value: &(EngramGeneration, EngramAdmissionReceipt)| value.1.receipt_digest.clone()
    );
    domain!(
        index.knowledge_correction.as_ref(),
        "knowledge_correction",
        |value: &(KnowledgeCorrectionProposal, KnowledgeCorrectionReceipt)| value
            .1
            .receipt_digest
            .clone()
    );
    LivingProjection {
        living_cimage_id: authority.id.0.clone(),
        active_generation_digest,
        generation_count: authority.generations.len(),
        admitted_domains,
        latest_receipts,
        rollback_targets: index
            .rollback_receipts
            .iter()
            .map(|receipt| receipt.to_generation_digest.clone())
            .collect(),
        projection_checkpoint: checkpoint,
    }
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("living ecs canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
