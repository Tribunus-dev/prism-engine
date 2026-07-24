//! Workload-level heterogeneous execution evidence.
//!
//! Representation quality is not enough for deployment: a profile must be
//! measured under the phase, batch, and concurrency pressure it is intended
//! to serve.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceWorkloadPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceClass {
    Realtime,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionLane {
    Ane,
    Accelerate,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrecisionSensitiveOp {
    Attention,
    Router,
    Norm,
    ExpertProjection,
    OutputHead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionExecutionGraph {
    pub graph_id: String,
    pub assignments: std::collections::BTreeMap<PrecisionSensitiveOp, String>,
    pub ternary_native_ops: Vec<PrecisionSensitiveOp>,
    pub conversion_boundaries: Vec<String>,
}

pub fn mixed_precision_graphs() -> Vec<MixedPrecisionExecutionGraph> {
    use PrecisionSensitiveOp::*;
    let specs = [
        (
            "ternary-expert-int8-attention",
            vec![
                (ExpertProjection, "Ternary158"),
                (Attention, "Int8"),
                (Router, "Int8"),
                (Norm, "Bf16"),
                (OutputHead, "Int8"),
            ],
        ),
        (
            "ternary-expert-fp16-attention",
            vec![
                (ExpertProjection, "Ternary158"),
                (Attention, "Fp16"),
                (Router, "Int8"),
                (Norm, "Bf16"),
                (OutputHead, "Fp16"),
            ],
        ),
        (
            "nf4-expert-int8-attention",
            vec![
                (ExpertProjection, "Nf4"),
                (Attention, "Int8"),
                (Router, "Int8"),
                (Norm, "Bf16"),
                (OutputHead, "Int8"),
            ],
        ),
        (
            "int4-expert-bf16-attention",
            vec![
                (ExpertProjection, "Int4"),
                (Attention, "Bf16"),
                (Router, "Int8"),
                (Norm, "Bf16"),
                (OutputHead, "Bf16"),
            ],
        ),
    ];
    specs
        .into_iter()
        .map(|(graph_id, entries)| {
            let assignments = entries
                .into_iter()
                .map(|(op, format)| (op, format.into()))
                .collect::<std::collections::BTreeMap<_, _>>();
            let ternary_native_ops = assignments
                .iter()
                .filter_map(|(op, format)| (format == "Ternary158").then_some(*op))
                .collect();
            MixedPrecisionExecutionGraph {
                graph_id: graph_id.into(),
                assignments,
                ternary_native_ops,
                conversion_boundaries: vec![
                    "ane-planar-int8-in".into(),
                    "ane-planar-int8-out".into(),
                ],
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub phase: InferenceWorkloadPhase,
    pub service_class: ServiceClass,
    pub batch_size: u32,
    pub concurrency: u32,
    pub primary_lane: ExecutionLane,
    pub attention_lane: ExecutionLane,
    pub interleaved_metal: bool,
    pub stateless_shared_arena: bool,
    pub ane_compute_int8: bool,
    pub planar_input_conversion: bool,
    pub planar_output_conversion: bool,
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self {
            phase: InferenceWorkloadPhase::Prefill,
            service_class: ServiceClass::Realtime,
            batch_size: 1,
            concurrency: 1,
            primary_lane: ExecutionLane::Ane,
            attention_lane: ExecutionLane::Ane,
            interleaved_metal: false,
            stateless_shared_arena: true,
            ane_compute_int8: true,
            planar_input_conversion: false,
            planar_output_conversion: false,
        }
    }
}

impl WorkloadProfile {
    pub fn validate(self) -> Result<(), String> {
        if self.batch_size == 0 || self.concurrency == 0 {
            return Err("workload batch and concurrency must be nonzero".into());
        }
        if self.service_class == ServiceClass::Realtime && self.batch_size != 1 {
            return Err("realtime profiles require batch size one".into());
        }
        if self.phase == InferenceWorkloadPhase::Prefill && self.primary_lane != ExecutionLane::Ane
        {
            return Err("prefill profiles must use ANE as the primary lane".into());
        }
        if self.interleaved_metal
            && self.primary_lane != ExecutionLane::Metal
            && self.attention_lane != ExecutionLane::Metal
        {
            return Err("interleaved Metal requires a Metal execution lane".into());
        }
        if self.primary_lane == ExecutionLane::Ane && !self.stateless_shared_arena {
            return Err("ANE profiles require stateless shared-arena binding".into());
        }
        if self.primary_lane == ExecutionLane::Ane
            && (!self.ane_compute_int8
                || !self.planar_input_conversion
                || !self.planar_output_conversion)
        {
            return Err(
                "ANE profiles require INT8 compute with planar input/output conversions".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadThroughputEvidence {
    pub profile: WorkloadProfile,
    pub representation: String,
    pub tiling_digest: String,
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub measured: bool,
    pub evidence_source: String,
    pub execution_fingerprint: String,
    pub projected: bool,
    pub projection_basis: String,
    pub mixed_precision_graph: String,
    #[serde(default)]
    pub semantic_region_plan_digest: Option<String>,
    #[serde(default)]
    pub region_count: Option<u32>,
    #[serde(default)]
    pub materialized_bytes: Option<u64>,
    #[serde(default)]
    pub conversion_bytes: Option<u64>,
    #[serde(default)]
    pub layout_fragmentation_score: Option<f64>,
    #[serde(default)]
    pub region_execution_fingerprint: Option<String>,
}

impl Default for WorkloadThroughputEvidence {
    fn default() -> Self {
        Self {
            profile: WorkloadProfile::default(),
            representation: String::new(),
            tiling_digest: String::new(),
            tokens_per_second: 0.0,
            latency_ms: 0.0,
            measured: false,
            evidence_source: String::new(),
            execution_fingerprint: String::new(),
            projected: false,
            projection_basis: String::new(),
            mixed_precision_graph: String::new(),
            semantic_region_plan_digest: None,
            region_count: None,
            materialized_bytes: None,
            conversion_bytes: None,
            layout_fragmentation_score: None,
            region_execution_fingerprint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SelectedExecutionGraph {
    pub profiles: Vec<WorkloadThroughputEvidence>,
    pub route_sequence: Vec<ExecutionLane>,
    pub fused_interleaved_metal: bool,
    pub stateless_ane: bool,
    pub ane_int8_planar_boundaries: bool,
    pub mixed_precision_graphs: Vec<MixedPrecisionExecutionGraph>,
}

impl WorkloadThroughputEvidence {
    pub fn valid(&self) -> bool {
        self.measured
            && self.tokens_per_second.is_finite()
            && self.tokens_per_second > 0.0
            && self.profile.validate().is_ok()
            && !self.execution_fingerprint.is_empty()
            && self.semantic_region_evidence_valid()
    }

    fn semantic_region_evidence_valid(&self) -> bool {
        match &self.semantic_region_plan_digest {
            None => true,
            Some(digest) => {
                !digest.is_empty()
                    && self.region_count.unwrap_or(0) > 0
                    && self
                        .region_execution_fingerprint
                        .as_ref()
                        .is_some_and(|fingerprint| !fingerprint.is_empty())
            }
        }
    }
}

pub fn default_profile_grid() -> Vec<WorkloadProfile> {
    let mut profiles = Vec::new();
    for phase in [
        InferenceWorkloadPhase::Prefill,
        InferenceWorkloadPhase::Decode,
    ] {
        for service_class in [ServiceClass::Realtime, ServiceClass::Batch] {
            let batches = if service_class == ServiceClass::Realtime {
                vec![1]
            } else {
                vec![1, 4, 8, 16]
            };
            for batch_size in batches {
                for concurrency in [1, 2, 4, 8, 16] {
                    let primary_lanes = if phase == InferenceWorkloadPhase::Prefill {
                        vec![ExecutionLane::Ane]
                    } else {
                        vec![
                            ExecutionLane::Ane,
                            ExecutionLane::Accelerate,
                            ExecutionLane::Metal,
                        ]
                    };
                    for primary_lane in primary_lanes {
                        for attention_lane in [
                            ExecutionLane::Ane,
                            ExecutionLane::Accelerate,
                            ExecutionLane::Metal,
                        ] {
                            let profile = WorkloadProfile {
                                phase,
                                service_class,
                                batch_size,
                                concurrency,
                                primary_lane,
                                attention_lane,
                                interleaved_metal: primary_lane == ExecutionLane::Metal
                                    || attention_lane == ExecutionLane::Metal,
                                stateless_shared_arena: primary_lane == ExecutionLane::Ane,
                                ane_compute_int8: primary_lane == ExecutionLane::Ane,
                                planar_input_conversion: primary_lane == ExecutionLane::Ane,
                                planar_output_conversion: primary_lane == ExecutionLane::Ane,
                            };
                            if profile.validate().is_ok() {
                                profiles.push(profile);
                            }
                        }
                    }
                }
            }
        }
    }
    profiles
}

pub fn benchmark_profiles<F>(
    profiles: &[WorkloadProfile],
    representation: &str,
    tiling_digest: &str,
    mut runner: F,
) -> Vec<WorkloadThroughputEvidence>
where
    F: FnMut(WorkloadProfile) -> Result<f64, String>,
{
    profiles
        .iter()
        .copied()
        .filter_map(|profile| {
            if profile.validate().is_err() {
                return None;
            }
            let latency_ms = runner(profile).ok()?;
            let logical_tokens = profile.batch_size.max(1);
            let tokens_per_second = (logical_tokens as f64 * profile.concurrency as f64 * 1_000.0)
                / latency_ms.max(0.001);
            Some(WorkloadThroughputEvidence {
                profile,
                representation: representation.into(),
                tiling_digest: tiling_digest.into(),
                tokens_per_second,
                latency_ms,
                measured: true,
                evidence_source: format!(
                    "native-{:?}-{:?}",
                    profile.primary_lane, profile.attention_lane
                ),
                execution_fingerprint: format!(
                    "{:?}:{:?}:{}:{}:{}",
                    profile.phase,
                    profile.primary_lane,
                    profile.batch_size,
                    profile.concurrency,
                    tiling_digest
                ),
                projected: true,
                projection_basis: "bounded representative kernel timing".into(),
                mixed_precision_graph: "mixed-precision-candidate-set".into(),
                semantic_region_plan_digest: None,
                region_count: None,
                materialized_bytes: None,
                conversion_bytes: None,
                layout_fragmentation_score: None,
                region_execution_fingerprint: None,
            })
        })
        .collect()
}

pub fn benchmark_mixed_precision_profiles<F>(
    profiles: &[WorkloadProfile],
    tiling_digest: &str,
    mut runner: F,
) -> Vec<WorkloadThroughputEvidence>
where
    F: FnMut(WorkloadProfile, &MixedPrecisionExecutionGraph) -> Result<f64, String>,
{
    let graphs = mixed_precision_graphs();
    profiles
        .iter()
        .copied()
        .filter_map(|profile| {
            let (graph, latency_ms) = graphs
                .iter()
                .filter_map(|graph| runner(profile, graph).ok().map(|latency| (graph, latency)))
                .min_by(|(_, a), (_, b)| a.total_cmp(b))?;
            let tokens_per_second =
                profile.batch_size.max(1) as f64 * profile.concurrency.max(1) as f64 * 1_000.0
                    / latency_ms.max(0.001);
            Some(WorkloadThroughputEvidence {
                profile,
                representation: "mixed".into(),
                tiling_digest: tiling_digest.into(),
                tokens_per_second,
                latency_ms,
                measured: true,
                evidence_source: "native-mixed-precision-representative".into(),
                execution_fingerprint: format!(
                    "{}:{:?}:{}:{}",
                    graph.graph_id, profile.phase, profile.batch_size, profile.concurrency
                ),
                projected: true,
                projection_basis: "bounded mixed-precision representative kernel timing".into(),
                mixed_precision_graph: graph.graph_id.clone(),
                semantic_region_plan_digest: None,
                region_count: None,
                materialized_bytes: None,
                conversion_bytes: None,
                layout_fragmentation_score: None,
                region_execution_fingerprint: None,
            })
        })
        .collect()
}

pub fn attach_semantic_region_evidence(
    mut evidence: WorkloadThroughputEvidence,
    plan_digest: impl Into<String>,
    region_count: u32,
    materialized_bytes: u64,
    conversion_bytes: u64,
    layout_fragmentation_score: f64,
    region_execution_fingerprint: impl Into<String>,
) -> WorkloadThroughputEvidence {
    evidence.semantic_region_plan_digest = Some(plan_digest.into());
    evidence.region_count = Some(region_count);
    evidence.materialized_bytes = Some(materialized_bytes);
    evidence.conversion_bytes = Some(conversion_bytes);
    evidence.layout_fragmentation_score = Some(layout_fragmentation_score);
    evidence.region_execution_fingerprint = Some(region_execution_fingerprint.into());
    evidence
}

pub fn select_best_profile(
    evidence: &[WorkloadThroughputEvidence],
) -> Option<WorkloadThroughputEvidence> {
    evidence
        .iter()
        .filter(|sample| sample.valid())
        .max_by(|a, b| a.tokens_per_second.total_cmp(&b.tokens_per_second))
        .cloned()
}

pub fn select_execution_graph(evidence: &[WorkloadThroughputEvidence]) -> SelectedExecutionGraph {
    let mut selected = Vec::new();
    let lane_preference = |sample: &WorkloadThroughputEvidence| {
        let attention = match sample.profile.attention_lane {
            ExecutionLane::Accelerate => 3,
            ExecutionLane::Metal => 2,
            ExecutionLane::Ane => 1,
        };
        let primary = match sample.profile.primary_lane {
            ExecutionLane::Metal => 3,
            ExecutionLane::Ane => 2,
            ExecutionLane::Accelerate => 1,
        };
        attention * 10 + primary
    };
    for phase in [
        InferenceWorkloadPhase::Prefill,
        InferenceWorkloadPhase::Decode,
    ] {
        for service in [ServiceClass::Realtime, ServiceClass::Batch] {
            if let Some(best) = evidence
                .iter()
                .filter(|sample| {
                    sample.profile.phase == phase && sample.profile.service_class == service
                })
                .filter(|sample| sample.valid())
                .max_by(|a, b| {
                    a.tokens_per_second
                        .total_cmp(&b.tokens_per_second)
                        .then_with(|| lane_preference(a).cmp(&lane_preference(b)))
                })
            {
                selected.push(best.clone());
            }
        }
    }
    let route_sequence = selected
        .iter()
        .map(|sample| sample.profile.primary_lane)
        .collect();
    SelectedExecutionGraph {
        fused_interleaved_metal: selected
            .iter()
            .any(|sample| sample.profile.interleaved_metal),
        stateless_ane: selected
            .iter()
            .any(|sample| sample.profile.primary_lane == ExecutionLane::Ane),
        ane_int8_planar_boundaries: selected.iter().any(|sample| {
            sample.profile.ane_compute_int8
                && sample.profile.planar_input_conversion
                && sample.profile.planar_output_conversion
        }),
        profiles: selected,
        route_sequence,
        mixed_precision_graphs: mixed_precision_graphs(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_evidence_requires_region_fingerprint() {
        let profile = WorkloadProfile {
            phase: InferenceWorkloadPhase::Decode,
            service_class: ServiceClass::Realtime,
            batch_size: 1,
            concurrency: 1,
            primary_lane: ExecutionLane::Metal,
            attention_lane: ExecutionLane::Metal,
            interleaved_metal: true,
            stateless_shared_arena: false,
            ane_compute_int8: false,
            planar_input_conversion: false,
            planar_output_conversion: false,
        };
        let mut evidence = benchmark_profiles(&[profile], "fp16", "tile", |_| Ok(1.0)).remove(0);
        evidence.semantic_region_plan_digest = Some("plan".into());
        evidence.region_count = Some(3);
        assert!(!evidence.valid());
        let evidence =
            attach_semantic_region_evidence(evidence, "plan", 3, 0, 0, 0.0, "region-run");
        assert!(evidence.valid());
    }
}
