//! Evolutionary search contracts for target-machine execution graphs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableUnitKind {
    MetalKernel,
    AccelerateProgram,
    MlModelcProgram,
    CpuKernel,
    ConversionKernel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutableUnitCandidate {
    pub unit_id: String,
    pub kind: ExecutableUnitKind,
    pub operation_ids: Vec<String>,
    pub semantic_region_ids: Vec<String>,
    pub input_layouts: Vec<String>,
    pub output_layouts: Vec<String>,
    pub compile_options: BTreeMap<String, String>,
    pub estimated_compile_ms: f64,
    pub estimated_resident_bytes: u64,
    pub artifact_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionGraphEdge {
    pub from_unit: String,
    pub to_unit: String,
    pub buffer: String,
    pub synchronization: String,
    pub zero_copy: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetExecutionGraph {
    pub graph_id: String,
    pub target_profile_digest: String,
    pub semantic_region_plan_digest: String,
    pub units: Vec<ExecutableUnitCandidate>,
    pub edges: Vec<ExecutionGraphEdge>,
    pub residency_policy: BTreeMap<String, String>,
    pub workload_class: String,
    #[serde(default)]
    pub graph_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionGraphMutation {
    FuseAdjacent,
    SplitFusedUnit,
    MoveToMetal,
    MoveToAccelerate,
    MoveToMlModelc,
    MoveToCpu,
    ChangeMetalGeometry,
    ChangePacking,
    ChangeCommandBufferBoundary,
    InsertConversion,
    RemoveConversion,
    ChangeResidencyWindow,
    ChangeSynchronizationEdge,
    SpecializeMlModelcShape,
    MergeMlModelcPrograms,
    SplitMlModelcProgram,
    ChangeAneExecutionUnit,
    CoalesceSemanticRegions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGraphMeasurement {
    pub graph_digest: String,
    pub measured: bool,
    pub execution_fingerprint: String,
    pub behavioral_quality: f64,
    pub agentic_success: f64,
    pub prefill_latency_ms: f64,
    pub decode_latency_ms: f64,
    pub tokens_per_second: f64,
    pub energy_per_token: Option<f64>,
    pub peak_resident_bytes: u64,
    pub metal_utilization: f64,
    pub cpu_utilization: f64,
    pub ane_utilization: f64,
    pub copy_bytes: u64,
    pub conversion_bytes: u64,
    pub synchronization_ms: f64,
    pub kernel_count: u32,
    pub mlmodelc_load_ms: f64,
    pub compile_ms: f64,
    pub fallback_frequency: f64,
    pub receipt_completeness: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGraphAdmissionPolicy {
    pub min_behavioral_quality: f64,
    pub min_agentic_success: f64,
    pub max_prefill_latency_ms: Option<f64>,
    pub max_decode_latency_ms: Option<f64>,
    pub min_tokens_per_second: Option<f64>,
    pub max_peak_resident_bytes: Option<u64>,
    pub max_copy_bytes: Option<u64>,
    pub max_conversion_bytes: Option<u64>,
    pub max_synchronization_ms: Option<f64>,
    pub max_kernel_count: Option<u32>,
    pub max_fallback_frequency: f64,
    pub min_receipt_completeness: f64,
}

impl Default for ExecutionGraphAdmissionPolicy {
    fn default() -> Self {
        Self {
            min_behavioral_quality: 0.99,
            min_agentic_success: 0.98,
            max_prefill_latency_ms: None,
            max_decode_latency_ms: None,
            min_tokens_per_second: None,
            max_peak_resident_bytes: None,
            max_copy_bytes: None,
            max_conversion_bytes: None,
            max_synchronization_ms: None,
            max_kernel_count: None,
            max_fallback_frequency: 0.05,
            min_receipt_completeness: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGraphAdmissionReceipt {
    pub graph_digest: String,
    pub admitted: bool,
    pub measured: bool,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutionGraphError {
    #[error("execution graph identity is incomplete")]
    MissingIdentity,
    #[error("execution graph contains no executable units")]
    EmptyGraph,
    #[error("execution graph contains an invalid edge")]
    InvalidEdge,
    #[error("execution graph digest mismatch")]
    DigestMismatch,
    #[error("execution graph measurement is not authoritative")]
    Unmeasured,
}

impl TargetExecutionGraph {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.graph_digest.clear();
        canonical.units.sort_by(|a, b| a.unit_id.cmp(&b.unit_id));
        canonical.edges.sort_by(|a, b| (&a.from_unit, &a.to_unit, &a.buffer).cmp(&(&b.from_unit, &b.to_unit, &b.buffer)));
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, ExecutionGraphError> {
        self.graph_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), ExecutionGraphError> {
        if self.graph_id.is_empty() || self.target_profile_digest.is_empty() || self.semantic_region_plan_digest.is_empty() {
            return Err(ExecutionGraphError::MissingIdentity);
        }
        if self.units.is_empty() {
            return Err(ExecutionGraphError::EmptyGraph);
        }
        let unit_ids = self.units.iter().map(|unit| unit.unit_id.as_str()).collect::<std::collections::BTreeSet<_>>();
        if self.edges.iter().any(|edge| !unit_ids.contains(edge.from_unit.as_str()) || !unit_ids.contains(edge.to_unit.as_str())) {
            return Err(ExecutionGraphError::InvalidEdge);
        }
        if !self.graph_digest.is_empty() && self.graph_digest != self.canonical_digest() {
            return Err(ExecutionGraphError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn mutate_execution_graph(
    graph: &TargetExecutionGraph,
    mutation: ExecutionGraphMutation,
) -> Result<TargetExecutionGraph, ExecutionGraphError> {
    graph.verify()?;
    let mut candidate = graph.clone();
    candidate.graph_id = format!("{}:{mutation:?}", graph.graph_id);
    candidate.graph_digest.clear();
    match mutation {
        ExecutionGraphMutation::FuseAdjacent if candidate.units.len() >= 2 => {
            let right = candidate.units.remove(1);
            candidate.units[0].operation_ids.extend(right.operation_ids);
            candidate.units[0].semantic_region_ids.extend(right.semantic_region_ids);
            candidate.edges.retain(|edge| edge.from_unit != right.unit_id && edge.to_unit != right.unit_id);
        }
        ExecutionGraphMutation::MoveToMetal => candidate.units[0].kind = ExecutableUnitKind::MetalKernel,
        ExecutionGraphMutation::MoveToAccelerate => candidate.units[0].kind = ExecutableUnitKind::AccelerateProgram,
        ExecutionGraphMutation::MoveToMlModelc => candidate.units[0].kind = ExecutableUnitKind::MlModelcProgram,
        ExecutionGraphMutation::MoveToCpu => candidate.units[0].kind = ExecutableUnitKind::CpuKernel,
        ExecutionGraphMutation::InsertConversion => {
            candidate.units.push(ExecutableUnitCandidate { unit_id: format!("conversion-{}", candidate.units.len()), kind: ExecutableUnitKind::ConversionKernel, operation_ids: vec!["convert".into()], semantic_region_ids: vec![], input_layouts: vec![], output_layouts: vec![], compile_options: BTreeMap::new(), estimated_compile_ms: 0.0, estimated_resident_bytes: 0, artifact_digest: None });
        }
        other => {
            candidate.units[0].compile_options.insert("last_mutation".into(), format!("{other:?}"));
        }
    }
    candidate.seal()
}

pub fn admit_execution_graph(
    graph: &TargetExecutionGraph,
    measurement: &ExecutionGraphMeasurement,
    policy: &ExecutionGraphAdmissionPolicy,
) -> Result<ExecutionGraphAdmissionReceipt, ExecutionGraphError> {
    graph.verify()?;
    if !measurement.measured || measurement.execution_fingerprint.is_empty() || measurement.graph_digest != graph.graph_digest {
        return Err(ExecutionGraphError::Unmeasured);
    }
    let mut reasons = Vec::new();
    if measurement.behavioral_quality < policy.min_behavioral_quality { reasons.push("behavioral quality below gate".into()); }
    if measurement.agentic_success < policy.min_agentic_success { reasons.push("agentic success below gate".into()); }
    if let Some(limit) = policy.max_prefill_latency_ms { if measurement.prefill_latency_ms > limit { reasons.push("prefill latency exceeds gate".into()); } }
    if let Some(limit) = policy.max_decode_latency_ms { if measurement.decode_latency_ms > limit { reasons.push("decode latency exceeds gate".into()); } }
    if let Some(limit) = policy.min_tokens_per_second { if measurement.tokens_per_second < limit { reasons.push("throughput below gate".into()); } }
    if let Some(limit) = policy.max_peak_resident_bytes { if measurement.peak_resident_bytes > limit { reasons.push("resident bytes exceed gate".into()); } }
    if let Some(limit) = policy.max_copy_bytes { if measurement.copy_bytes > limit { reasons.push("copy bytes exceed gate".into()); } }
    if let Some(limit) = policy.max_conversion_bytes { if measurement.conversion_bytes > limit { reasons.push("conversion bytes exceed gate".into()); } }
    if let Some(limit) = policy.max_synchronization_ms { if measurement.synchronization_ms > limit { reasons.push("synchronization cost exceeds gate".into()); } }
    if let Some(limit) = policy.max_kernel_count { if measurement.kernel_count > limit { reasons.push("kernel count exceeds gate".into()); } }
    if measurement.fallback_frequency > policy.max_fallback_frequency { reasons.push("fallback frequency exceeds gate".into()); }
    if measurement.receipt_completeness < policy.min_receipt_completeness { reasons.push("receipt completeness below gate".into()); }
    let mut receipt = ExecutionGraphAdmissionReceipt { graph_digest: graph.graph_digest.clone(), admitted: reasons.is_empty(), measured: true, reasons, receipt_digest: String::new() };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("execution graph canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
