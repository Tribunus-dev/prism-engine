use prism_ecs_ir::semantic_region::{
    RegionRepresentationAssignment, SemanticRegionId, SemanticRegionPartition, SemanticRegionPlan,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectedRepresentation {
    SparseFp16Residual,
    LowRankResidual { rank: u32 },
    DenseResidual,
    Int8Fallback,
    Nf4Fallback,
    Bf16Protected,
    Fp16Protected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionTernaryEvidence {
    pub region: SemanticRegionId,
    pub weight_nrmse: Option<f64>,
    pub operator_nrmse: Option<f64>,
    pub cosine_mean: Option<f64>,
    pub activation_divergence: Option<f64>,
    pub router_flip_rate: Option<f64>,
    pub logit_kl: Option<f64>,
    pub rollout_success_delta: Option<f64>,
    pub measured: bool,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressiveTernaryPolicy {
    pub native_ternary_format: String,
    pub max_weight_nrmse_initializer: f64,
    pub max_operator_nrmse: f64,
    pub min_cosine_mean: f64,
    pub max_activation_divergence: f64,
    pub max_router_flip_rate: f64,
    pub max_logit_kl: f64,
    pub min_rollout_success_delta: f64,
    pub default_protection: ProtectedRepresentation,
}

impl Default for ProgressiveTernaryPolicy {
    fn default() -> Self {
        Self {
            native_ternary_format: "ternary158".into(),
            max_weight_nrmse_initializer: 0.10,
            max_operator_nrmse: 0.15,
            min_cosine_mean: 0.995,
            max_activation_divergence: 0.05,
            max_router_flip_rate: 0.01,
            max_logit_kl: 0.02,
            min_rollout_success_delta: -0.01,
            default_protection: ProtectedRepresentation::Int8Fallback,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressiveTernaryDecision {
    pub region: SemanticRegionId,
    pub native_ternary: bool,
    pub protection: Option<ProtectedRepresentation>,
    pub admitted_by: Vec<String>,
    pub rejected_by: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressiveTernaryPlan {
    pub semantic_plan: SemanticRegionPlan,
    pub decisions: Vec<ProgressiveTernaryDecision>,
    pub native_ternary_elements: u64,
    pub protected_elements: u64,
    pub evidence_complete: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProgressiveTernaryError {
    #[error("evidence references unknown region {0}")]
    UnknownRegion(String),
    #[error("duplicate evidence for region {0}")]
    DuplicateEvidence(String),
    #[error("missing evidence for region {0}")]
    MissingEvidence(String),
    #[error("semantic plan is invalid")]
    InvalidSemanticPlan,
}

pub fn build_progressive_ternary_plan(
    partition: SemanticRegionPartition,
    evidence: Vec<RegionTernaryEvidence>,
    policy: &ProgressiveTernaryPolicy,
    require_measured_behavior: bool,
) -> Result<ProgressiveTernaryPlan, ProgressiveTernaryError> {
    let known = partition
        .regions
        .iter()
        .map(|region| region.id.clone())
        .collect::<BTreeSet<_>>();
    let mut by_region = BTreeMap::new();
    for sample in evidence {
        if !known.contains(&sample.region) {
            return Err(ProgressiveTernaryError::UnknownRegion(sample.region.0));
        }
        if by_region.insert(sample.region.clone(), sample).is_some() {
            return Err(ProgressiveTernaryError::DuplicateEvidence("duplicate".into()));
        }
    }

    let mut assignments = Vec::new();
    let mut decisions = Vec::new();
    let mut native_ternary_elements = 0;
    let mut protected_elements = 0;
    let mut evidence_complete = true;

    for region in &partition.regions {
        let sample = by_region
            .get(&region.id)
            .ok_or_else(|| ProgressiveTernaryError::MissingEvidence(region.id.0.clone()))?;
        let mut admitted_by = Vec::new();
        let mut rejected_by = Vec::new();
        check_max(sample.weight_nrmse, policy.max_weight_nrmse_initializer, "weight_initializer", &mut admitted_by, &mut rejected_by);
        check_max(sample.operator_nrmse, policy.max_operator_nrmse, "operator", &mut admitted_by, &mut rejected_by);
        check_min(sample.cosine_mean, policy.min_cosine_mean, "cosine", &mut admitted_by, &mut rejected_by);
        check_max(sample.activation_divergence, policy.max_activation_divergence, "activation", &mut admitted_by, &mut rejected_by);
        check_max(sample.router_flip_rate, policy.max_router_flip_rate, "router", &mut admitted_by, &mut rejected_by);
        check_max(sample.logit_kl, policy.max_logit_kl, "logit", &mut admitted_by, &mut rejected_by);
        check_min(sample.rollout_success_delta, policy.min_rollout_success_delta, "rollout", &mut admitted_by, &mut rejected_by);
        if require_measured_behavior && !sample.measured {
            rejected_by.push("behavior_unmeasured".into());
        }
        if sample.evidence_refs.is_empty() {
            evidence_complete = false;
            rejected_by.push("missing_evidence_refs".into());
        }
        let native_ternary = rejected_by.is_empty();
        let protection = (!native_ternary).then(|| policy.default_protection.clone());
        let representation = if native_ternary {
            policy.native_ternary_format.clone()
        } else {
            protection_name(protection.as_ref().expect("protected representation"))
        };
        assignments.push(RegionRepresentationAssignment {
            region: region.id.clone(),
            representation,
            codec: None,
            preferred_lane: None,
            residency: None,
            assignment_evidence: sample.evidence_refs.clone(),
        });
        let elements = selector_elements(&partition.parent_shape, &region.selector);
        if native_ternary {
            native_ternary_elements += elements;
        } else {
            protected_elements += elements;
        }
        decisions.push(ProgressiveTernaryDecision {
            region: region.id.clone(),
            native_ternary,
            protection,
            admitted_by,
            rejected_by,
        });
    }

    let semantic_plan = SemanticRegionPlan {
        partition,
        assignments,
        compile_verified: true,
        plan_digest: String::new(),
    }
    .seal()
    .map_err(|_| ProgressiveTernaryError::InvalidSemanticPlan)?;

    Ok(ProgressiveTernaryPlan {
        semantic_plan,
        decisions,
        native_ternary_elements,
        protected_elements,
        evidence_complete,
    })
}

fn check_max(value: Option<f64>, limit: f64, label: &str, admitted: &mut Vec<String>, rejected: &mut Vec<String>) {
    match value {
        Some(value) if value.is_finite() && value <= limit => admitted.push(label.into()),
        Some(_) => rejected.push(label.into()),
        None => rejected.push(format!("{label}_missing")),
    }
}

fn check_min(value: Option<f64>, limit: f64, label: &str, admitted: &mut Vec<String>, rejected: &mut Vec<String>) {
    match value {
        Some(value) if value.is_finite() && value >= limit => admitted.push(label.into()),
        Some(_) => rejected.push(label.into()),
        None => rejected.push(format!("{label}_missing")),
    }
}

fn protection_name(protection: &ProtectedRepresentation) -> String {
    match protection {
        ProtectedRepresentation::SparseFp16Residual => "ternary+sparse_fp16_residual",
        ProtectedRepresentation::LowRankResidual { .. } => "ternary+low_rank_residual",
        ProtectedRepresentation::DenseResidual => "ternary+dense_residual",
        ProtectedRepresentation::Int8Fallback => "int8",
        ProtectedRepresentation::Nf4Fallback => "nf4",
        ProtectedRepresentation::Bf16Protected => "bf16",
        ProtectedRepresentation::Fp16Protected => "fp16",
    }
    .into()
}

fn selector_elements(shape: &[u64], selector: &prism_ecs_ir::semantic_region::RegionSelector) -> u64 {
    use prism_ecs_ir::semantic_region::RegionSelector;
    match selector {
        RegionSelector::WholeTensor => shape.iter().product(),
        RegionSelector::AxisSpan { axis, start, end } => {
            let axis_len = shape.get(*axis as usize).copied().unwrap_or(1).max(1);
            shape.iter().product::<u64>() / axis_len * (end - start)
        }
        RegionSelector::Rect { extents, .. } => extents.iter().product(),
    }
}
