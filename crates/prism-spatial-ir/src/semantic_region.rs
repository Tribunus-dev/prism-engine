//! Physical realization of persistent semantic tensor regions.

use prism_ecs_ir::semantic_region::{RegionSelector, SemanticRegionId, SemanticRegionPlan};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalRegionRealization {
    pub semantic_region: SemanticRegionId,
    pub logical_selector_digest: String,
    pub packed_buffer: String,
    pub byte_ranges: Vec<Range<u64>>,
    pub tile_ids: Vec<String>,
    pub execution_lane: String,
    pub residency_class: String,
    pub materialized_bytes: u64,
    pub conversion_ops: Vec<String>,
    pub realization_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalRegionPlan {
    pub semantic_plan_digest: String,
    pub realizations: Vec<PhysicalRegionRealization>,
    pub total_materialized_bytes: u64,
    pub total_conversion_bytes: u64,
    pub digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PhysicalRegionError {
    #[error("semantic region {0} is missing a physical realization")]
    MissingRegion(String),
    #[error("semantic region {0} is realized more than once")]
    DuplicateRegion(String),
    #[error("physical byte ranges overlap")]
    OverlappingByteRanges,
    #[error("physical realization contains an empty byte range")]
    EmptyByteRange,
    #[error("hidden conversion or materialization is not represented")]
    HiddenConversion,
    #[error("realization digest mismatch")]
    DigestMismatch,
}

impl PhysicalRegionRealization {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.realization_digest.clear();
        canonical
            .byte_ranges
            .sort_by_key(|range| (range.start, range.end));
        canonical.tile_ids.sort();
        canonical.conversion_ops.sort();
        digest(&serde_json::to_vec(&canonical).expect("physical region serialization"))
    }

    pub fn seal(mut self) -> Result<Self, PhysicalRegionError> {
        for range in &self.byte_ranges {
            if range.start >= range.end {
                return Err(PhysicalRegionError::EmptyByteRange);
            }
        }
        if self.materialized_bytes > 0 && self.conversion_ops.is_empty() {
            return Err(PhysicalRegionError::HiddenConversion);
        }
        self.realization_digest = self.canonical_digest();
        Ok(self)
    }
}

impl PhysicalRegionPlan {
    pub fn verify(&self, semantic: &SemanticRegionPlan) -> Result<(), PhysicalRegionError> {
        let expected = semantic
            .partition
            .regions
            .iter()
            .map(|region| region.id.clone())
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut ranges = Vec::new();
        for realization in &self.realizations {
            if !expected.contains(&realization.semantic_region) {
                return Err(PhysicalRegionError::MissingRegion(
                    realization.semantic_region.0.clone(),
                ));
            }
            if !seen.insert(realization.semantic_region.clone()) {
                return Err(PhysicalRegionError::DuplicateRegion(
                    realization.semantic_region.0.clone(),
                ));
            }
            if realization.realization_digest != realization.canonical_digest() {
                return Err(PhysicalRegionError::DigestMismatch);
            }
            ranges.extend(realization.byte_ranges.iter().cloned());
        }
        if let Some(missing) = expected.difference(&seen).next() {
            return Err(PhysicalRegionError::MissingRegion(missing.0.clone()));
        }
        ranges.sort_by_key(|range| (range.start, range.end));
        for pair in ranges.windows(2) {
            if pair[1].start < pair[0].end {
                return Err(PhysicalRegionError::OverlappingByteRanges);
            }
        }
        if self.digest != self.canonical_digest() {
            return Err(PhysicalRegionError::DigestMismatch);
        }
        Ok(())
    }

    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.digest.clear();
        canonical
            .realizations
            .sort_by(|a, b| a.semantic_region.cmp(&b.semantic_region));
        digest(&serde_json::to_vec(&canonical).expect("physical region plan serialization"))
    }

    pub fn seal(mut self, semantic: &SemanticRegionPlan) -> Result<Self, PhysicalRegionError> {
        self.realizations = self
            .realizations
            .into_iter()
            .map(PhysicalRegionRealization::seal)
            .collect::<Result<Vec<_>, _>>()?;
        self.total_materialized_bytes =
            self.realizations.iter().map(|r| r.materialized_bytes).sum();
        self.total_conversion_bytes = self
            .realizations
            .iter()
            .filter(|r| !r.conversion_ops.is_empty())
            .flat_map(|r| r.byte_ranges.iter())
            .map(|range| range.end - range.start)
            .sum();
        self.digest = self.canonical_digest();
        self.verify(semantic)?;
        Ok(self)
    }
}

pub fn lower_contiguous_axis0(
    semantic: &SemanticRegionPlan,
    element_size: u64,
    row_elements: u64,
    buffer: &str,
) -> Result<PhysicalRegionPlan, PhysicalRegionError> {
    let mut realizations = Vec::new();
    for assignment in &semantic.assignments {
        let descriptor = semantic
            .partition
            .regions
            .iter()
            .find(|region| region.id == assignment.region)
            .ok_or_else(|| PhysicalRegionError::MissingRegion(assignment.region.0.clone()))?;
        let (start, end) = match descriptor.selector {
            RegionSelector::WholeTensor => (0, semantic.partition.parent_shape[0]),
            RegionSelector::AxisSpan {
                axis: 0,
                start,
                end,
            } => (start, end),
            _ => return Err(PhysicalRegionError::HiddenConversion),
        };
        let byte_start = start
            .saturating_mul(row_elements)
            .saturating_mul(element_size);
        let byte_end = end
            .saturating_mul(row_elements)
            .saturating_mul(element_size);
        realizations.push(PhysicalRegionRealization {
            semantic_region: assignment.region.clone(),
            logical_selector_digest: selector_digest(&descriptor.selector),
            packed_buffer: buffer.into(),
            byte_ranges: std::iter::once(byte_start..byte_end).collect(),
            tile_ids: vec![format!("tile:{}", assignment.region.0)],
            execution_lane: assignment
                .preferred_lane
                .clone()
                .unwrap_or_else(|| "cpu".into()),
            residency_class: assignment
                .residency
                .clone()
                .unwrap_or_else(|| "resident".into()),
            materialized_bytes: 0,
            conversion_ops: Vec::new(),
            realization_digest: String::new(),
        });
    }
    PhysicalRegionPlan {
        semantic_plan_digest: semantic.plan_digest.clone(),
        realizations,
        total_materialized_bytes: 0,
        total_conversion_bytes: 0,
        digest: String::new(),
    }
    .seal(semantic)
}

fn selector_digest(selector: &RegionSelector) -> String {
    digest(&serde_json::to_vec(selector).expect("selector serialization"))
}

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_ir::evolution::foundation::LogicalTensorId;
    use prism_ecs_ir::semantic_region::{
        RegionConstraints, RegionOrigin, RegionRepresentationAssignment, RegionRole,
        SemanticRegionDescriptor, SemanticRegionPartition,
    };

    fn semantic() -> SemanticRegionPlan {
        let regions = vec![
            SemanticRegionDescriptor {
                id: SemanticRegionId("q".into()),
                parent: LogicalTensorId("qkv".into()),
                selector: RegionSelector::AxisSpan {
                    axis: 0,
                    start: 0,
                    end: 4,
                },
                role: RegionRole::Generic { label: "q".into() },
                origin: RegionOrigin::Explicit {
                    source: "test".into(),
                },
                constraints: RegionConstraints {
                    allowed_formats: vec!["fp16".into()],
                    ..Default::default()
                },
                provenance_refs: vec![],
            },
            SemanticRegionDescriptor {
                id: SemanticRegionId("kv".into()),
                parent: LogicalTensorId("qkv".into()),
                selector: RegionSelector::AxisSpan {
                    axis: 0,
                    start: 4,
                    end: 6,
                },
                role: RegionRole::Generic { label: "kv".into() },
                origin: RegionOrigin::Explicit {
                    source: "test".into(),
                },
                constraints: RegionConstraints {
                    allowed_formats: vec!["fp16".into()],
                    ..Default::default()
                },
                provenance_refs: vec![],
            },
        ];
        SemanticRegionPlan {
            partition: SemanticRegionPartition {
                parent: LogicalTensorId("qkv".into()),
                parent_shape: vec![6, 2],
                regions,
                exhaustive: true,
                disjoint: true,
                digest: String::new(),
            },
            assignments: vec![
                RegionRepresentationAssignment {
                    region: SemanticRegionId("q".into()),
                    representation: "fp16".into(),
                    codec: None,
                    preferred_lane: Some("metal".into()),
                    residency: None,
                    assignment_evidence: vec![],
                },
                RegionRepresentationAssignment {
                    region: SemanticRegionId("kv".into()),
                    representation: "fp16".into(),
                    codec: None,
                    preferred_lane: Some("metal".into()),
                    residency: None,
                    assignment_evidence: vec![],
                },
            ],
            compile_verified: true,
            plan_digest: String::new(),
        }
        .seal()
        .unwrap()
    }

    #[test]
    fn lowering_conserves_nonoverlapping_ranges() {
        let s = semantic();
        let p = lower_contiguous_axis0(&s, 2, 2, "weights").unwrap();
        assert_eq!(p.realizations[0].byte_ranges[0], 0..16);
        assert_eq!(p.realizations[1].byte_ranges[0], 16..24);
        assert!(p.verify(&s).is_ok());
    }

    #[test]
    fn materialization_requires_explicit_conversion() {
        let r = PhysicalRegionRealization {
            semantic_region: SemanticRegionId("q".into()),
            logical_selector_digest: "x".into(),
            packed_buffer: "b".into(),
            byte_ranges: std::iter::once(0..4).collect(),
            tile_ids: vec![],
            execution_lane: "cpu".into(),
            residency_class: "resident".into(),
            materialized_bytes: 4,
            conversion_ops: vec![],
            realization_digest: String::new(),
        };
        assert_eq!(r.seal(), Err(PhysicalRegionError::HiddenConversion));
    }
}
