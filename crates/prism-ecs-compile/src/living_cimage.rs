//! Constitutional Living CImage generations and adaptation lifecycle.

use prism_ecs_ir::semantic_region::SemanticRegionPlan;
use prism_spatial_ir::semantic_region::PhysicalRegionPlan;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LivingCImageId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CImageGeneration(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationKind {
    ProgressiveTernarization,
    ResidualProtection,
    LowRankFineTune,
    EngramCalibration,
    RoutingCalibration,
    KnowledgeCorrection,
    ExecutionGraphEvolution,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationLifecycle {
    Proposed,
    Calibrating,
    Training,
    Evaluating,
    Admitted,
    Promoted,
    Rejected,
    Superseded,
    RolledBack,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingCImageGeneration {
    pub living_cimage_id: LivingCImageId,
    pub generation: CImageGeneration,
    pub parent_generation: Option<CImageGeneration>,
    pub base_artifact_digest: String,
    pub semantic_region_plan: SemanticRegionPlan,
    pub physical_region_plan: Option<PhysicalRegionPlan>,
    pub adaptation_kind: AdaptationKind,
    pub adaptation_manifest_digest: String,
    pub engram_generation_digest: Option<String>,
    pub calibration_corpus_digest: Option<String>,
    pub admission_receipt_digest: Option<String>,
    pub lifecycle: AdaptationLifecycle,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub generation_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingCImage {
    pub id: LivingCImageId,
    pub base_artifact_digest: String,
    pub active_generation: CImageGeneration,
    pub generations: Vec<LivingCImageGeneration>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LivingCImageError {
    #[error("base artifact digest is empty")]
    MissingBase,
    #[error("generation sequence is invalid")]
    InvalidSequence,
    #[error("parent generation is missing")]
    MissingParent,
    #[error("active generation is not promoted")]
    ActiveNotPromoted,
    #[error("generation digest mismatch")]
    DigestMismatch,
    #[error("generation not found")]
    GenerationNotFound,
    #[error("candidate generation is not admitted")]
    NotAdmitted,
}

impl LivingCImageGeneration {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.generation_digest.clear();
        let bytes = serde_json::to_vec(&canonical).expect("living cimage serialization");
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    pub fn seal(mut self) -> Result<Self, LivingCImageError> {
        self.generation_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), LivingCImageError> {
        if self.base_artifact_digest.is_empty() {
            return Err(LivingCImageError::MissingBase);
        }
        self.semantic_region_plan
            .verify()
            .map_err(|_| LivingCImageError::InvalidSequence)?;
        if !self.generation_digest.is_empty() && self.generation_digest != self.canonical_digest() {
            return Err(LivingCImageError::DigestMismatch);
        }
        Ok(())
    }
}

impl LivingCImage {
    pub fn verify(&self) -> Result<(), LivingCImageError> {
        if self.base_artifact_digest.is_empty() || self.generations.is_empty() {
            return Err(LivingCImageError::MissingBase);
        }
        for (index, generation) in self.generations.iter().enumerate() {
            generation.verify()?;
            if generation.generation.0 != index as u64 {
                return Err(LivingCImageError::InvalidSequence);
            }
            if index == 0 {
                if generation.parent_generation.is_some() {
                    return Err(LivingCImageError::InvalidSequence);
                }
            } else if generation.parent_generation != Some(CImageGeneration(index as u64 - 1)) {
                return Err(LivingCImageError::MissingParent);
            }
        }
        let active = self
            .generations
            .get(self.active_generation.0 as usize)
            .ok_or(LivingCImageError::GenerationNotFound)?;
        if active.lifecycle != AdaptationLifecycle::Promoted {
            return Err(LivingCImageError::ActiveNotPromoted);
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        mut candidate: LivingCImageGeneration,
    ) -> Result<CImageGeneration, LivingCImageError> {
        candidate.generation = CImageGeneration(self.generations.len() as u64);
        candidate.parent_generation = Some(self.active_generation);
        candidate.lifecycle = AdaptationLifecycle::Proposed;
        candidate = candidate.seal()?;
        let id = candidate.generation;
        self.generations.push(candidate);
        Ok(id)
    }

    pub fn promote(
        &mut self,
        generation: CImageGeneration,
        receipt_digest: String,
    ) -> Result<(), LivingCImageError> {
        let candidate = self
            .generations
            .get_mut(generation.0 as usize)
            .ok_or(LivingCImageError::GenerationNotFound)?;
        if candidate.lifecycle != AdaptationLifecycle::Admitted {
            return Err(LivingCImageError::NotAdmitted);
        }
        candidate.lifecycle = AdaptationLifecycle::Promoted;
        candidate.admission_receipt_digest = Some(receipt_digest);
        candidate.generation_digest = candidate.canonical_digest();
        self.active_generation = generation;
        Ok(())
    }

    pub fn rollback(&mut self, generation: CImageGeneration) -> Result<(), LivingCImageError> {
        let target = self
            .generations
            .get_mut(generation.0 as usize)
            .ok_or(LivingCImageError::GenerationNotFound)?;
        target.lifecycle = AdaptationLifecycle::Promoted;
        target.generation_digest = target.canonical_digest();
        self.active_generation = generation;
        Ok(())
    }
}
