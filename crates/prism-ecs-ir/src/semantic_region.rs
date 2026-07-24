//! Persistent semantic sub-tensor identity independent of physical layout.

use crate::evolution::foundation::LogicalTensorId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SemanticRegionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RegionSelector {
    WholeTensor,
    AxisSpan {
        axis: u32,
        start: u64,
        end: u64,
    },
    Rect {
        offsets: Vec<u64>,
        extents: Vec<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RegionOrigin {
    GraphDerived {
        operation: String,
        source_value: String,
    },
    ArchitectureDerived {
        model_family: String,
        rule: String,
    },
    SensitivityDerived {
        probe_digest: String,
    },
    Explicit {
        source: String,
    },
    Hybrid {
        sources: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RegionRole {
    QueryProjection,
    KeyProjection,
    ValueProjection,
    AttentionHeadGroup { first: u32, count: u32 },
    Router,
    RoutedExpertGroup { first: u32, count: u32 },
    SharedExpert,
    GateProjection,
    UpProjection,
    DownProjection,
    EmbeddingShard,
    OutlierSidecar,
    SensitiveChannelGroup,
    Generic { label: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionConstraints {
    #[serde(default)]
    pub allowed_formats: Vec<String>,
    #[serde(default)]
    pub allowed_codecs: Vec<String>,
    #[serde(default)]
    pub preferred_lanes: Vec<String>,
    pub max_error: Option<f64>,
    #[serde(default = "default_alignment")]
    pub alignment_elements: u64,
    #[serde(default)]
    pub must_be_contiguous: bool,
    #[serde(default)]
    pub may_materialize: bool,
    #[serde(default)]
    pub may_reorder: bool,
}

fn default_alignment() -> u64 {
    1
}

impl Default for RegionConstraints {
    fn default() -> Self {
        Self {
            allowed_formats: vec!["fp16".into()],
            allowed_codecs: Vec::new(),
            preferred_lanes: Vec::new(),
            max_error: None,
            alignment_elements: 1,
            must_be_contiguous: true,
            may_materialize: false,
            may_reorder: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRegionDescriptor {
    pub id: SemanticRegionId,
    pub parent: LogicalTensorId,
    pub selector: RegionSelector,
    pub role: RegionRole,
    pub origin: RegionOrigin,
    pub constraints: RegionConstraints,
    #[serde(default)]
    pub provenance_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRegionPartition {
    pub parent: LogicalTensorId,
    pub parent_shape: Vec<u64>,
    pub regions: Vec<SemanticRegionDescriptor>,
    pub exhaustive: bool,
    pub disjoint: bool,
    #[serde(default)]
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionRepresentationAssignment {
    pub region: SemanticRegionId,
    pub representation: String,
    pub codec: Option<String>,
    pub preferred_lane: Option<String>,
    pub residency: Option<String>,
    #[serde(default)]
    pub assignment_evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRegionPlan {
    pub partition: SemanticRegionPartition,
    pub assignments: Vec<RegionRepresentationAssignment>,
    pub compile_verified: bool,
    #[serde(default)]
    pub plan_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SemanticRegionError {
    #[error("partition parent shape is empty")]
    EmptyShape,
    #[error("region {0} references a different parent tensor")]
    ParentMismatch(String),
    #[error("duplicate semantic region id: {0}")]
    DuplicateId(String),
    #[error("region {0} is empty or out of bounds")]
    InvalidBounds(String),
    #[error("rectangular selectors are not admitted in semantic region v0")]
    RectUnsupported,
    #[error("semantic regions overlap")]
    Overlap,
    #[error("exhaustive partition contains a gap")]
    CoverageGap,
    #[error("region {0} has no allowed representation")]
    MissingRepresentation(String),
    #[error("region {0} lacks semantic provenance")]
    MissingProvenance(String),
    #[error("partition digest mismatch")]
    DigestMismatch,
    #[error("assignment references unknown region {0}")]
    UnknownRegion(String),
    #[error("duplicate representation assignment for region {0}")]
    DuplicateAssignment(String),
    #[error("assigned representation {representation} is not allowed for region {region}")]
    RepresentationNotAllowed {
        region: String,
        representation: String,
    },
}

impl SemanticRegionPartition {
    pub fn verify(&self) -> Result<(), SemanticRegionError> {
        if self.parent_shape.is_empty() {
            return Err(SemanticRegionError::EmptyShape);
        }
        let mut ids = BTreeSet::new();
        let mut spans: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
        for region in &self.regions {
            if region.parent != self.parent {
                return Err(SemanticRegionError::ParentMismatch(region.id.0.clone()));
            }
            if !ids.insert(region.id.0.clone()) {
                return Err(SemanticRegionError::DuplicateId(region.id.0.clone()));
            }
            if region.constraints.allowed_formats.is_empty() {
                return Err(SemanticRegionError::MissingRepresentation(
                    region.id.0.clone(),
                ));
            }
            if !matches!(region.role, RegionRole::Generic { .. })
                && region.provenance_refs.is_empty()
            {
                return Err(SemanticRegionError::MissingProvenance(region.id.0.clone()));
            }
            match region.selector {
                RegionSelector::WholeTensor => {
                    if self.regions.len() != 1 {
                        return Err(SemanticRegionError::Overlap);
                    }
                }
                RegionSelector::AxisSpan { axis, start, end } => {
                    let Some(&axis_len) = self.parent_shape.get(axis as usize) else {
                        return Err(SemanticRegionError::InvalidBounds(region.id.0.clone()));
                    };
                    if start >= end || end > axis_len {
                        return Err(SemanticRegionError::InvalidBounds(region.id.0.clone()));
                    }
                    spans.entry(axis).or_default().push((start, end));
                }
                RegionSelector::Rect { .. } => return Err(SemanticRegionError::RectUnsupported),
            }
        }
        for (axis, entries) in &mut spans {
            entries.sort_unstable();
            let mut cursor = 0;
            for &(start, end) in entries.iter() {
                if self.disjoint && start < cursor {
                    return Err(SemanticRegionError::Overlap);
                }
                if self.exhaustive && start != cursor {
                    return Err(SemanticRegionError::CoverageGap);
                }
                cursor = cursor.max(end);
            }
            if self.exhaustive && cursor != self.parent_shape[*axis as usize] {
                return Err(SemanticRegionError::CoverageGap);
            }
        }
        if !self.digest.is_empty() && self.digest != self.canonical_digest() {
            return Err(SemanticRegionError::DigestMismatch);
        }
        Ok(())
    }

    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.digest.clear();
        canonical.regions.sort_by_key(selector_key);
        let bytes =
            serde_json::to_vec(&canonical).expect("semantic region canonical serialization");
        hex_digest(&bytes)
    }

    pub fn seal(mut self) -> Result<Self, SemanticRegionError> {
        self.digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }
}

impl SemanticRegionPlan {
    pub fn verify(&self) -> Result<(), SemanticRegionError> {
        self.partition.verify()?;
        let by_id: BTreeMap<_, _> = self.partition.regions.iter().map(|r| (&r.id, r)).collect();
        let mut assigned = BTreeSet::new();
        for assignment in &self.assignments {
            let Some(region) = by_id.get(&assignment.region) else {
                return Err(SemanticRegionError::UnknownRegion(
                    assignment.region.0.clone(),
                ));
            };
            if !assigned.insert(assignment.region.0.clone()) {
                return Err(SemanticRegionError::DuplicateAssignment(
                    assignment.region.0.clone(),
                ));
            }
            if !region
                .constraints
                .allowed_formats
                .iter()
                .any(|f| f == &assignment.representation)
            {
                return Err(SemanticRegionError::RepresentationNotAllowed {
                    region: assignment.region.0.clone(),
                    representation: assignment.representation.clone(),
                });
            }
        }
        if !self.plan_digest.is_empty() && self.plan_digest != self.canonical_digest() {
            return Err(SemanticRegionError::DigestMismatch);
        }
        Ok(())
    }

    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.plan_digest.clear();
        canonical
            .assignments
            .sort_by(|a, b| a.region.cmp(&b.region));
        let bytes = serde_json::to_vec(&canonical).expect("semantic region plan serialization");
        hex_digest(&bytes)
    }

    pub fn seal(mut self) -> Result<Self, SemanticRegionError> {
        self.partition = self.partition.seal()?;
        self.plan_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }
}

fn selector_key(region: &SemanticRegionDescriptor) -> (u32, u64, u64, String) {
    match region.selector {
        RegionSelector::WholeTensor => (0, 0, u64::MAX, region.id.0.clone()),
        RegionSelector::AxisSpan { axis, start, end } => (axis, start, end, region.id.0.clone()),
        RegionSelector::Rect { .. } => (u32::MAX, 0, 0, region.id.0.clone()),
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(id: &str, role: RegionRole, start: u64, end: u64) -> SemanticRegionDescriptor {
        SemanticRegionDescriptor {
            id: SemanticRegionId(id.into()),
            parent: LogicalTensorId("qkv".into()),
            selector: RegionSelector::AxisSpan {
                axis: 0,
                start,
                end,
            },
            role,
            origin: RegionOrigin::Explicit {
                source: "test".into(),
            },
            constraints: RegionConstraints {
                allowed_formats: vec!["fp16".into(), "int8".into()],
                ..Default::default()
            },
            provenance_refs: vec!["test:fixture".into()],
        }
    }

    fn partition(regions: Vec<SemanticRegionDescriptor>) -> SemanticRegionPartition {
        SemanticRegionPartition {
            parent: LogicalTensorId("qkv".into()),
            parent_shape: vec![6, 2],
            regions,
            exhaustive: true,
            disjoint: true,
            digest: String::new(),
        }
    }

    #[test]
    fn accepts_exhaustive_qkv_partition() {
        assert!(partition(vec![
            region("q", RegionRole::QueryProjection, 0, 4),
            region("k", RegionRole::KeyProjection, 4, 5),
            region("v", RegionRole::ValueProjection, 5, 6)
        ])
        .seal()
        .is_ok());
    }
    #[test]
    fn rejects_overlap() {
        assert_eq!(
            partition(vec![
                region("q", RegionRole::QueryProjection, 0, 4),
                region("k", RegionRole::KeyProjection, 3, 5),
                region("v", RegionRole::ValueProjection, 5, 6)
            ])
            .verify(),
            Err(SemanticRegionError::Overlap)
        );
    }
    #[test]
    fn rejects_gap() {
        assert_eq!(
            partition(vec![
                region("q", RegionRole::QueryProjection, 0, 3),
                region("k", RegionRole::KeyProjection, 4, 5),
                region("v", RegionRole::ValueProjection, 5, 6)
            ])
            .verify(),
            Err(SemanticRegionError::CoverageGap)
        );
    }
    #[test]
    fn rejects_out_of_bounds() {
        assert!(matches!(
            partition(vec![region("q", RegionRole::QueryProjection, 0, 7)]).verify(),
            Err(SemanticRegionError::InvalidBounds(_))
        ));
    }
    #[test]
    fn rejects_duplicate_id() {
        assert!(matches!(
            partition(vec![
                region("x", RegionRole::QueryProjection, 0, 4),
                region("x", RegionRole::KeyProjection, 4, 6)
            ])
            .verify(),
            Err(SemanticRegionError::DuplicateId(_))
        ));
    }
    #[test]
    fn digest_is_stable_across_input_order() {
        let a = partition(vec![
            region("q", RegionRole::QueryProjection, 0, 4),
            region("k", RegionRole::KeyProjection, 4, 5),
            region("v", RegionRole::ValueProjection, 5, 6),
        ]);
        let b = partition(vec![
            region("v", RegionRole::ValueProjection, 5, 6),
            region("q", RegionRole::QueryProjection, 0, 4),
            region("k", RegionRole::KeyProjection, 4, 5),
        ]);
        assert_eq!(a.canonical_digest(), b.canonical_digest());
    }
    #[test]
    fn serde_roundtrip() {
        let p = partition(vec![region("q", RegionRole::QueryProjection, 0, 6)]);
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(p, serde_json::from_str(&json).unwrap());
    }
}
