//! Hierarchical and bounded semantic-region assignment search.

use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::semantic_region::{
    RegionRepresentationAssignment, SemanticRegionId, SemanticRegionPartition, SemanticRegionPlan,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegionTemplateId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCandidatePalette {
    pub region: SemanticRegionId,
    pub template: RegionTemplateId,
    pub representations: Vec<String>,
    pub codecs: Vec<String>,
    pub preferred_lanes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRegularizationPolicy {
    pub max_regions: usize,
    pub max_kernel_variants: usize,
    pub max_conversion_boundaries: usize,
    pub prefer_adjacent_coalescing: bool,
}

impl Default for RegionRegularizationPolicy {
    fn default() -> Self {
        Self {
            max_regions: 64,
            max_kernel_variants: 8,
            max_conversion_boundaries: 16,
            prefer_adjacent_coalescing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalCandidate {
    pub global: CandidateGenome,
    pub partition_digest: String,
    pub assignments: Vec<RegionRepresentationAssignment>,
    pub regularization: RegionRegularizationPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalSearchObjectives {
    pub quality_loss: f64,
    pub packed_bytes: u64,
    pub materialized_bytes: u64,
    pub conversion_bytes: u64,
    pub region_boundary_count: usize,
    pub kernel_variant_count: usize,
    pub layout_fragmentation: f64,
    pub cross_lane_transfer_bytes: u64,
    pub receipt_complete: bool,
}

#[derive(Debug, Error)]
pub enum RegionalSearchError {
    #[error("partition exceeds region budget")]
    RegionBudgetExceeded,
    #[error("region {0} has no legal candidate representation")]
    EmptyPalette(String),
    #[error("assignment exceeds kernel variant budget")]
    KernelVariantBudgetExceeded,
    #[error("assignment exceeds conversion-boundary budget")]
    ConversionBudgetExceeded,
    #[error(transparent)]
    InvalidPlan(#[from] prism_ecs_ir::semantic_region::SemanticRegionError),
}

pub fn build_palettes(
    partition: &SemanticRegionPartition,
    templates: &BTreeMap<SemanticRegionId, RegionTemplateId>,
) -> Result<Vec<RegionCandidatePalette>, RegionalSearchError> {
    partition.verify()?;
    partition
        .regions
        .iter()
        .map(|region| {
            let representations = region.constraints.allowed_formats.clone();
            if representations.is_empty() {
                return Err(RegionalSearchError::EmptyPalette(region.id.0.clone()));
            }
            Ok(RegionCandidatePalette {
                region: region.id.clone(),
                template: templates
                    .get(&region.id)
                    .cloned()
                    .unwrap_or_else(|| RegionTemplateId(region.id.0.clone())),
                representations,
                codecs: region.constraints.allowed_codecs.clone(),
                preferred_lanes: region.constraints.preferred_lanes.clone(),
            })
        })
        .collect()
}

pub fn select_bounded_plan(
    partition: SemanticRegionPartition,
    global: CandidateGenome,
    palettes: &[RegionCandidatePalette],
    regularization: RegionRegularizationPolicy,
) -> Result<RegionalCandidate, RegionalSearchError> {
    if partition.regions.len() > regularization.max_regions {
        return Err(RegionalSearchError::RegionBudgetExceeded);
    }
    let by_region: BTreeMap<_, _> = palettes.iter().map(|p| (&p.region, p)).collect();
    let mut template_choices: BTreeMap<&RegionTemplateId, String> = BTreeMap::new();
    let mut assignments = Vec::new();
    for region in &partition.regions {
        let palette = by_region
            .get(&region.id)
            .ok_or_else(|| RegionalSearchError::EmptyPalette(region.id.0.clone()))?;
        let representation = template_choices
            .entry(&palette.template)
            .or_insert_with(|| palette.representations[0].clone())
            .clone();
        assignments.push(RegionRepresentationAssignment {
            region: region.id.clone(),
            representation,
            codec: palette.codecs.first().cloned(),
            preferred_lane: palette.preferred_lanes.first().cloned(),
            residency: None,
            assignment_evidence: vec![format!("region-template:{}", palette.template.0)],
        });
    }
    enforce_regularization(&assignments, &regularization)?;
    let plan = SemanticRegionPlan {
        partition,
        assignments: assignments.clone(),
        compile_verified: true,
        plan_digest: String::new(),
    }
    .seal()?;
    Ok(RegionalCandidate {
        global,
        partition_digest: plan.partition.digest,
        assignments,
        regularization,
    })
}

pub fn enforce_regularization(
    assignments: &[RegionRepresentationAssignment],
    policy: &RegionRegularizationPolicy,
) -> Result<(), RegionalSearchError> {
    let variants = assignments
        .iter()
        .map(|a| (a.representation.as_str(), a.codec.as_deref(), a.preferred_lane.as_deref()))
        .collect::<BTreeSet<_>>()
        .len();
    if variants > policy.max_kernel_variants {
        return Err(RegionalSearchError::KernelVariantBudgetExceeded);
    }
    let boundaries = assignments
        .windows(2)
        .filter(|pair| {
            pair[0].representation != pair[1].representation
                || pair[0].codec != pair[1].codec
                || pair[0].preferred_lane != pair[1].preferred_lane
        })
        .count();
    if boundaries > policy.max_conversion_boundaries {
        return Err(RegionalSearchError::ConversionBudgetExceeded);
    }
    Ok(())
}

pub fn objective_score(objectives: &RegionalSearchObjectives) -> f64 {
    if !objectives.receipt_complete || !objectives.quality_loss.is_finite() {
        return f64::NEG_INFINITY;
    }
    -(objectives.quality_loss * 1_000.0
        + objectives.packed_bytes as f64 / 1_000_000.0
        + objectives.materialized_bytes as f64 / 1_000_000.0
        + objectives.conversion_bytes as f64 / 1_000_000.0
        + objectives.region_boundary_count as f64 * 2.0
        + objectives.kernel_variant_count as f64 * 5.0
        + objectives.layout_fragmentation * 10.0
        + objectives.cross_lane_transfer_bytes as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_ir::evolution::foundation::LogicalTensorId;
    use prism_ecs_ir::semantic_region::{
        RegionConstraints, RegionOrigin, RegionRole, RegionSelector, SemanticRegionDescriptor,
    };

    fn partition() -> SemanticRegionPartition {
        let regions = (0..3)
            .map(|i| SemanticRegionDescriptor {
                id: SemanticRegionId(format!("r{i}")),
                parent: LogicalTensorId("qkv".into()),
                selector: RegionSelector::AxisSpan { axis: 0, start: i, end: i + 1 },
                role: RegionRole::Generic { label: format!("r{i}") },
                origin: RegionOrigin::Explicit { source: "test".into() },
                constraints: RegionConstraints { allowed_formats: vec!["int8".into(), "fp16".into()], ..Default::default() },
                provenance_refs: vec![],
            })
            .collect();
        SemanticRegionPartition { parent: LogicalTensorId("qkv".into()), parent_shape: vec![3, 2], regions, exhaustive: true, disjoint: true, digest: String::new() }.seal().unwrap()
    }

    #[test]
    fn template_sharing_reuses_assignment() {
        let p = partition();
        let template = RegionTemplateId("decoder-qkv".into());
        let templates = p.regions.iter().map(|r| (r.id.clone(), template.clone())).collect();
        let palettes = build_palettes(&p, &templates).unwrap();
        let candidate = select_bounded_plan(p, CandidateGenome::default(), &palettes, RegionRegularizationPolicy::default()).unwrap();
        assert!(candidate.assignments.iter().all(|a| a.representation == "int8"));
    }

    #[test]
    fn region_budget_is_enforced() {
        let p = partition();
        let palettes = build_palettes(&p, &BTreeMap::new()).unwrap();
        let policy = RegionRegularizationPolicy { max_regions: 2, ..Default::default() };
        assert!(matches!(select_bounded_plan(p, CandidateGenome::default(), &palettes, policy), Err(RegionalSearchError::RegionBudgetExceeded)));
    }

    #[test]
    fn objective_rejects_incomplete_receipt() {
        let score = objective_score(&RegionalSearchObjectives { quality_loss: 0.0, packed_bytes: 0, materialized_bytes: 0, conversion_bytes: 0, region_boundary_count: 0, kernel_variant_count: 1, layout_fragmentation: 0.0, cross_lane_transfer_bytes: 0, receipt_complete: false });
        assert_eq!(score, f64::NEG_INFINITY);
    }
}
