//! Versioned optional Semantic Region manifest for ComputeImage sidecars and metadata.

use prism_ecs_ir::semantic_region::{SemanticRegionPartition, SemanticRegionPlan};
use prism_spatial_ir::semantic_region::PhysicalRegionRealization;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const SEMANTIC_REGION_MANIFEST_V1: &str = "prism.semantic-region-manifest.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRegionManifest {
    pub schema: String,
    pub schema_version: u32,
    pub model_digest: String,
    pub partitions: Vec<SemanticRegionPartition>,
    pub plans: Vec<SemanticRegionPlan>,
    pub realizations: Vec<PhysicalRegionRealization>,
    pub receipt_refs: Vec<String>,
    pub manifest_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SemanticRegionManifestError {
    #[error("unsupported semantic region manifest schema")]
    UnsupportedSchema,
    #[error("manifest model digest is empty")]
    EmptyModelDigest,
    #[error("manifest references an unknown partition or plan")]
    UnknownPlan,
    #[error("manifest realization references an unknown semantic region")]
    UnknownRegion,
    #[error("manifest digest mismatch")]
    DigestMismatch,
    #[error("manifest contains duplicate receipt references")]
    DuplicateReceipt,
}

impl SemanticRegionManifest {
    pub fn verify(&self) -> Result<(), SemanticRegionManifestError> {
        if self.schema != SEMANTIC_REGION_MANIFEST_V1 || self.schema_version != 1 {
            return Err(SemanticRegionManifestError::UnsupportedSchema);
        }
        if self.model_digest.is_empty() {
            return Err(SemanticRegionManifestError::EmptyModelDigest);
        }
        let partition_digests = self
            .partitions
            .iter()
            .map(|partition| partition.digest.as_str())
            .collect::<BTreeSet<_>>();
        for plan in &self.plans {
            if !partition_digests.contains(plan.partition.digest.as_str()) {
                return Err(SemanticRegionManifestError::UnknownPlan);
            }
        }
        let region_ids = self
            .partitions
            .iter()
            .flat_map(|partition| partition.regions.iter().map(|region| region.id.clone()))
            .collect::<BTreeSet<_>>();
        for realization in &self.realizations {
            if !region_ids.contains(&realization.semantic_region) {
                return Err(SemanticRegionManifestError::UnknownRegion);
            }
        }
        let receipts = self.receipt_refs.iter().collect::<BTreeSet<_>>();
        if receipts.len() != self.receipt_refs.len() {
            return Err(SemanticRegionManifestError::DuplicateReceipt);
        }
        if self.manifest_digest != self.canonical_digest() {
            return Err(SemanticRegionManifestError::DigestMismatch);
        }
        Ok(())
    }

    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.manifest_digest.clear();
        canonical.partitions.sort_by(|a, b| a.digest.cmp(&b.digest));
        canonical.plans.sort_by(|a, b| a.plan_digest.cmp(&b.plan_digest));
        canonical.realizations.sort_by(|a, b| a.semantic_region.cmp(&b.semantic_region));
        canonical.receipt_refs.sort();
        let bytes = serde_json::to_vec(&canonical).expect("semantic region manifest serialization");
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    pub fn seal(mut self) -> Result<Self, SemanticRegionManifestError> {
        self.manifest_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_ir::evolution::foundation::LogicalTensorId;
    use prism_ecs_ir::semantic_region::{
        RegionConstraints, RegionOrigin, RegionRepresentationAssignment, RegionRole,
        RegionSelector, SemanticRegionDescriptor, SemanticRegionId,
    };

    fn manifest() -> SemanticRegionManifest {
        let partition = SemanticRegionPartition {
            parent: LogicalTensorId("tensor".into()),
            parent_shape: vec![1],
            regions: vec![SemanticRegionDescriptor {
                id: SemanticRegionId("r".into()),
                parent: LogicalTensorId("tensor".into()),
                selector: RegionSelector::AxisSpan { axis: 0, start: 0, end: 1 },
                role: RegionRole::Generic { label: "whole".into() },
                origin: RegionOrigin::Explicit { source: "test".into() },
                constraints: RegionConstraints { allowed_formats: vec!["fp16".into()], ..Default::default() },
                provenance_refs: vec![],
            }],
            exhaustive: true,
            disjoint: true,
            digest: String::new(),
        }
        .seal()
        .unwrap();
        let plan = SemanticRegionPlan {
            partition: partition.clone(),
            assignments: vec![RegionRepresentationAssignment {
                region: SemanticRegionId("r".into()),
                representation: "fp16".into(),
                codec: None,
                preferred_lane: None,
                residency: None,
                assignment_evidence: vec![],
            }],
            compile_verified: true,
            plan_digest: String::new(),
        }
        .seal()
        .unwrap();
        SemanticRegionManifest {
            schema: SEMANTIC_REGION_MANIFEST_V1.into(),
            schema_version: 1,
            model_digest: "model".into(),
            partitions: vec![partition],
            plans: vec![plan],
            realizations: vec![],
            receipt_refs: vec!["receipt:discovery".into()],
            manifest_digest: String::new(),
        }
    }

    #[test]
    fn manifest_roundtrip_and_digest() {
        let sealed = manifest().seal().unwrap();
        let json = sealed.to_pretty_json().unwrap();
        let decoded: SemanticRegionManifest = serde_json::from_str(&json).unwrap();
        assert!(decoded.verify().is_ok());
    }

    #[test]
    fn duplicate_receipts_are_rejected() {
        let mut m = manifest();
        m.receipt_refs.push("receipt:discovery".into());
        m.manifest_digest = m.canonical_digest();
        assert_eq!(m.verify(), Err(SemanticRegionManifestError::DuplicateReceipt));
    }
}
