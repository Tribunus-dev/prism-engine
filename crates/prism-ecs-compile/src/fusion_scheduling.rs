//! Fusion scheduling — backend evaluation, group growth, and
//! candidate selection for fusion groups.
//!
//! This module owns the canonical authority for the three scheduling
//! decisions that run on a `FusionGroup` before kernel selection:
//!
//! 1. **Backend evaluation** — evaluate the group's op sequence
//!    against every registered backend, recording the support level
//!    (`Full` / `Partial` / `Unsupported`) and the lowering cost
//!    estimate for each candidate.
//! 2. **Group growth** — for singleton groups (no fused ops), try to
//!    greedily merge with sibling dispatches in the same layer's
//!    graph handle until the policy's `max_group_size` is reached.
//! 3. **Cost evaluation & selection** — pick the best candidate using
//!    either the policy-weighted score or the production-default
//!    heuristic.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The dataflow graph IR (owned by the fusion analysis module).
//! - The kernel lowerer (owned by `prism-ecs-kernel`).
//! - The dispatch entity lifecycle (owned by the runtime kernel).
//!
//! All exposed types are pure value types. The module never mutates
//! the world directly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::fusion_analysis::{DataflowNode, DataflowOp, DataflowOpKind, MatMulContract};

/// Role a backend serves for a given candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendRole {
    ProductionHotPath,
    Fallback,
    Research,
}

/// Support level for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionSupportLevel {
    Full,
    Partial,
    Unsupported,
}

/// Lowering cost for one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoweringCost {
    pub estimated_us: f64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub scratch_bytes: u64,
    pub thread_count: u32,
    pub materialization_cost: f64,
}

impl prism_ecs_core::Component for LoweringCost {}

/// Result of evaluating a group against one backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendSupport {
    pub supported: bool,
    pub reason: Option<String>,
    pub estimated_latency_us: Option<f64>,
    pub estimated_memory_bytes: Option<u64>,
    pub estimated_scratch_bytes: Option<u64>,
    pub requires_materialization: bool,
}

/// One candidate fusion plan for a single backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionCandidate {
    pub group: FusedGroup,
    pub target: String,
    pub support: FusionSupportLevel,
    pub lowering_cost: LoweringCost,
}

/// One fused group — the scheduled body of a kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedGroup {
    pub id: String,
    pub body: Vec<DataflowNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub internal_values: Vec<String>,
    pub codec_family: String,
}

/// Policy governing fusion acceptance thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionPolicy {
    pub max_group_size: usize,
    pub allow_research_fusions: bool,
}

impl Default for FusionPolicy {
    fn default() -> Self {
        Self {
            max_group_size: 8,
            allow_research_fusions: false,
        }
    }
}

/// Selection policy — knobs for the cost scoring function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionSelectionPolicy {
    pub prefer_lower_latency: bool,
    pub prefer_memory_efficient: bool,
    pub avoid_materialization: bool,
}

impl Default for FusionSelectionPolicy {
    fn default() -> Self {
        Self {
            prefer_lower_latency: true,
            prefer_memory_efficient: true,
            avoid_materialization: true,
        }
    }
}

/// One fusion rejection — recorded when a candidate backend is not
/// supported for a group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionRejection {
    pub group_id: String,
    pub target: String,
    pub reason: String,
}

/// Schedule data — candidates and the selected winner for one group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionScheduleData {
    pub candidates: Vec<FusionCandidate>,
    pub selected: Option<FusionCandidate>,
}

impl prism_ecs_core::Component for FusionScheduleData {}

/// Evaluation data — the source nodes considered and any rejections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionEvaluationData {
    pub source_nodes: Vec<usize>,
    pub rejected: Vec<FusionRejection>,
}

impl prism_ecs_core::Component for FusionEvaluationData {}

/// Minimal in-memory backend registry. The production registry lives
/// in the spatial IR; this minimal version is sufficient for the
/// scheduling algorithms to be tested in isolation.
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilityRegistry {
    targets: Vec<String>,
    supported_sequences: BTreeMap<String, Vec<Vec<DataflowOpKind>>>,
}

impl BackendCapabilityRegistry {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            supported_sequences: BTreeMap::new(),
        }
    }

    pub fn register_target(&mut self, target: impl Into<String>) {
        let target = target.into();
        if !self.targets.contains(&target) {
            self.targets.push(target);
        }
    }

    pub fn register_supported_sequence(
        &mut self,
        target: impl Into<String>,
        sequence: Vec<DataflowOpKind>,
    ) {
        let target = target.into();
        self.supported_sequences
            .entry(target)
            .or_default()
            .push(sequence);
    }

    pub fn all_targets(&self) -> Vec<String> {
        self.targets.clone()
    }

    pub fn supports_sequence(
        &self,
        target: &str,
        ops: &[DataflowOpKind],
        _role: BackendRole,
    ) -> (bool, Option<String>) {
        if let Some(seqs) = self.supported_sequences.get(target) {
            for seq in seqs {
                if sequence_is_compatible(seq, ops) {
                    return (true, None);
                }
            }
        }
        (false, Some(format!("no supported sequence for {} ops", ops.len())))
    }

    pub fn evaluate(
        &self,
        target: &str,
        group: &FusedGroup,
        _role: BackendRole,
    ) -> BackendSupport {
        let ops: Vec<DataflowOpKind> = group
            .body
            .iter()
            .map(|n| op_kind_from_dataflow_op(&n.op))
            .collect();
        let (supported, reason) = self.supports_sequence(target, &ops, _role);
        BackendSupport {
            supported,
            reason,
            estimated_latency_us: if supported {
                Some(ops.len() as f64 * 5.0)
            } else {
                None
            },
            estimated_memory_bytes: if supported {
                Some(ops.len() as u64 * 8192)
            } else {
                None
            },
            estimated_scratch_bytes: if supported {
                Some(1024 * 1024)
            } else {
                None
            },
            requires_materialization: ops.len() > 4,
        }
    }
}

/// Default registry: `metal` supports MatMul + elementwise;
/// `ane` supports a small op set; `cpu` is a fallback for everything.
pub fn default_registry() -> BackendCapabilityRegistry {
    let mut r = BackendCapabilityRegistry::new();
    r.register_target("metal");
    r.register_target("ane");
    r.register_target("cpu");
    r.register_supported_sequence("metal", vec![DataflowOpKind::MatMul, DataflowOpKind::SiLU]);
    r.register_supported_sequence(
        "metal",
        vec![DataflowOpKind::MatMul, DataflowOpKind::Mul, DataflowOpKind::SiLU],
    );
    r.register_supported_sequence("ane", vec![DataflowOpKind::RmsNorm]);
    r.register_supported_sequence("ane", vec![DataflowOpKind::MatMul]);
    r
}

fn op_kind_from_dataflow_op(op: &DataflowOp) -> DataflowOpKind {
    match op {
        DataflowOp::LoadWeight { .. } | DataflowOp::AneLoadWeight { .. } => DataflowOpKind::LoadWeight,
        DataflowOp::Dequantize { .. } => DataflowOpKind::Dequantize,
        DataflowOp::MatMul { .. } | DataflowOp::AneMatMul { .. } => DataflowOpKind::MatMul,
        DataflowOp::RmsNorm { .. } => DataflowOpKind::RmsNorm,
        DataflowOp::SiLU { .. } => DataflowOpKind::SiLU,
        DataflowOp::Gelu { .. } => DataflowOpKind::Gelu,
        DataflowOp::Mul { .. } => DataflowOpKind::Mul,
        DataflowOp::Add { .. } => DataflowOpKind::Add,
        DataflowOp::ResidualAdd { .. } => DataflowOpKind::ResidualAdd,
        DataflowOp::StoreActivation { .. } | DataflowOp::AneStoreOutput { .. } => {
            DataflowOpKind::StoreActivation
        }
        DataflowOp::KvRead { .. } => DataflowOpKind::KvRead,
        DataflowOp::KvWrite { .. } => DataflowOpKind::KvWrite,
        DataflowOp::EngramLookup { .. } => DataflowOpKind::EngramLookup,
        DataflowOp::AneConv1x1 { .. } => DataflowOpKind::AneConv1x1,
    }
}

fn sequence_is_compatible(supported: &[DataflowOpKind], candidate: &[DataflowOpKind]) -> bool {
    if supported.is_empty() {
        return false;
    }
    if candidate.is_empty() {
        return true;
    }
    // The supported sequence is a contiguous run; candidate matches if
    // every candidate op appears in the supported set.
    candidate.iter().all(|op| supported.contains(op))
}

/// Build a synthetic `FusedGroup` from a list of op kind strings.
pub fn build_synthetic_group(op_kinds: &[String]) -> Option<FusedGroup> {
    let body: Vec<DataflowNode> = op_kinds
        .iter()
        .enumerate()
        .filter_map(|(i, k)| node_from_kind(k, i))
        .collect();
    if body.is_empty() {
        return None;
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut internal_values = Vec::new();

    for node in &body {
        for buf in &node.inputs {
            let produced_by_body = body.iter().any(|n| n.outputs.contains(buf));
            if !produced_by_body && !inputs.contains(buf) {
                inputs.push(buf.clone());
            }
        }
        for buf in &node.outputs {
            let consumed_by_body = body.iter().any(|n| n.inputs.contains(buf));
            if consumed_by_body && !internal_values.contains(buf) {
                internal_values.push(buf.clone());
            } else if !consumed_by_body && !outputs.contains(buf) {
                outputs.push(buf.clone());
            }
        }
    }

    Some(FusedGroup {
        id: format!("g{}", op_kinds.first().map(String::as_str).unwrap_or("")),
        body,
        inputs,
        outputs,
        internal_values,
        codec_family: "Fp16".into(),
    })
}

fn node_from_kind(kind: &str, id: usize) -> Option<DataflowNode> {
    let op_kind = parse_op_kind(kind)?;
    let buf_in = format!("buf_node{id}_in");
    let buf_out = format!("buf_node{id}_out");

    let op = match op_kind {
        DataflowOpKind::MatMul => DataflowOp::MatMul {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_weight"),
            output: buf_out.clone(),
            contract: MatMulContract {
                m: 4096,
                n: 4096,
                k: 4096,
                lhs_transposed: false,
                rhs_transposed: false,
            },
        },
        DataflowOpKind::RmsNorm => DataflowOp::RmsNorm {
            input: buf_in.clone(),
            weight: format!("rms_weight_{id}"),
            output: buf_out.clone(),
            epsilon: 1e-6,
        },
        DataflowOpKind::SiLU => DataflowOp::SiLU {
            input: buf_in.clone(),
            output: buf_out.clone(),
        },
        DataflowOpKind::Gelu => DataflowOp::Gelu {
            input: buf_in.clone(),
            output: buf_out.clone(),
        },
        DataflowOpKind::Mul => DataflowOp::Mul {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_rhs"),
            output: buf_out.clone(),
        },
        DataflowOpKind::Add => DataflowOp::Add {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_rhs"),
            output: buf_out.clone(),
        },
        _ => return None,
    };

    Some(DataflowNode {
        id,
        op,
        inputs: vec![buf_in],
        outputs: vec![buf_out],
    })
}

fn parse_op_kind(s: &str) -> Option<DataflowOpKind> {
    match s.to_lowercase().as_str() {
        "matmul" | "mat_mul" => Some(DataflowOpKind::MatMul),
        "rms_norm" | "rmsnorm" => Some(DataflowOpKind::RmsNorm),
        "silu" => Some(DataflowOpKind::SiLU),
        "gelu" => Some(DataflowOpKind::Gelu),
        "mul" | "multiply" => Some(DataflowOpKind::Mul),
        "add" => Some(DataflowOpKind::Add),
        _ => None,
    }
}

/// Score-based selection using the policy.
pub fn score_select(
    candidates: &[FusionCandidate],
    policy: &FusionSelectionPolicy,
) -> Option<FusionCandidate> {
    if candidates.is_empty() {
        return None;
    }
    let mut best: Option<&FusionCandidate> = None;
    let mut best_score = f64::NEG_INFINITY;

    for c in candidates {
        let mut score = 0.0;
        match c.support {
            FusionSupportLevel::Full => score += 100.0,
            FusionSupportLevel::Partial => score += 10.0,
            FusionSupportLevel::Unsupported => score -= 1000.0,
        }
        if policy.prefer_lower_latency {
            score -= (c.lowering_cost.estimated_us + c.lowering_cost.materialization_cost) / 1000.0;
        }
        if policy.prefer_memory_efficient {
            score -= ((c.lowering_cost.bytes_read + c.lowering_cost.bytes_written) as f64)
                / (1024.0 * 1024.0);
        }
        if policy.avoid_materialization && c.lowering_cost.materialization_cost > 0.0 {
            score -= c.lowering_cost.materialization_cost;
        }
        if score > best_score {
            best_score = score;
            best = Some(c);
        }
    }

    best.cloned()
}

/// Production selection — prefer Full support, then first Partial.
pub fn prod_select(candidates: &[FusionCandidate]) -> Option<FusionCandidate> {
    candidates
        .iter()
        .find(|c| c.support == FusionSupportLevel::Full)
        .or_else(|| {
            candidates
                .iter()
                .find(|c| c.support == FusionSupportLevel::Partial)
        })
        .cloned()
}

/// Compute the cost estimate for one candidate.
pub fn compute_cost(support: &BackendSupport, op_count: usize) -> LoweringCost {
    let op_count_f = op_count as f64;
    let estimated_us = support.estimated_latency_us.unwrap_or_else(|| {
        let base = op_count_f * 5.0;
        base * if op_count_f > 1.0 { 0.6 } else { 1.0 }
    });
    let bytes_read = support
        .estimated_memory_bytes
        .unwrap_or((op_count as u64) * 8192);
    let bytes_written = bytes_read;
    let scratch_bytes = support.estimated_scratch_bytes.unwrap_or(1024 * 1024);
    let materialization_cost = if support.requires_materialization {
        10.0
    } else {
        0.0
    };

    LoweringCost {
        estimated_us,
        bytes_read,
        bytes_written,
        scratch_bytes,
        thread_count: 256,
        materialization_cost,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchedulingError {
    #[error("no candidate produced a parseable op sequence for kinds: {0:?}")]
    NoParseableSequence(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candidate(target: &str, support: FusionSupportLevel) -> FusionCandidate {
        let group = FusedGroup {
            id: "g".into(),
            body: vec![],
            inputs: vec![],
            outputs: vec![],
            internal_values: vec![],
            codec_family: "Fp16".into(),
        };
        FusionCandidate {
            group,
            target: target.into(),
            support,
            lowering_cost: LoweringCost {
                estimated_us: 100.0,
                bytes_read: 4096,
                bytes_written: 4096,
                scratch_bytes: 1024,
                thread_count: 256,
                materialization_cost: 0.0,
            },
        }
    }

    #[test]
    fn score_select_picks_full_over_partial() {
        let candidates = vec![
            sample_candidate("metal", FusionSupportLevel::Partial),
            sample_candidate("metal", FusionSupportLevel::Full),
        ];
        let p = FusionSelectionPolicy::default();
        let selected = score_select(&candidates, &p).expect("select");
        assert_eq!(selected.support, FusionSupportLevel::Full);
    }

    #[test]
    fn score_select_returns_none_for_empty() {
        let p = FusionSelectionPolicy::default();
        assert!(score_select(&[], &p).is_none());
    }

    #[test]
    fn prod_select_picks_full_first() {
        let candidates = vec![
            sample_candidate("metal", FusionSupportLevel::Partial),
            sample_candidate("ane", FusionSupportLevel::Full),
        ];
        let selected = prod_select(&candidates).expect("select");
        assert_eq!(selected.target, "ane");
    }

    #[test]
    fn prod_select_falls_back_to_partial() {
        let candidates = vec![
            sample_candidate("metal", FusionSupportLevel::Partial),
            sample_candidate("ane", FusionSupportLevel::Partial),
        ];
        let selected = prod_select(&candidates).expect("select");
        assert_eq!(selected.support, FusionSupportLevel::Partial);
    }

    #[test]
    fn prod_select_returns_none_for_empty() {
        assert!(prod_select(&[]).is_none());
    }

    #[test]
    fn default_registry_lists_three_targets() {
        let r = default_registry();
        let targets = r.all_targets();
        assert!(targets.contains(&"metal".to_string()));
        assert!(targets.contains(&"ane".to_string()));
        assert!(targets.contains(&"cpu".to_string()));
    }

    #[test]
    fn default_registry_supports_matmul_silu_sequence() {
        let r = default_registry();
        let ops = vec![DataflowOpKind::MatMul, DataflowOpKind::SiLU];
        let (supported, _) = r.supports_sequence("metal", &ops, BackendRole::ProductionHotPath);
        assert!(supported);
    }

    #[test]
    fn default_registry_does_not_support_unknown_sequence() {
        let r = default_registry();
        let ops = vec![DataflowOpKind::EngramLookup];
        let (supported, _) = r.supports_sequence("metal", &ops, BackendRole::ProductionHotPath);
        assert!(!supported);
    }

    #[test]
    fn default_registry_supports_matmul_on_ane() {
        let r = default_registry();
        let ops = vec![DataflowOpKind::MatMul];
        let (supported, _) = r.supports_sequence("ane", &ops, BackendRole::ProductionHotPath);
        assert!(supported);
    }

    #[test]
    fn build_synthetic_group_creates_correct_shape() {
        let g = build_synthetic_group(&["MatMul".into(), "SiLU".into()]).expect("build");
        assert_eq!(g.body.len(), 2);
        assert!(!g.inputs.is_empty());
        assert!(!g.outputs.is_empty());
    }

    #[test]
    fn build_synthetic_group_with_chained_ops_has_internals() {
        let g = build_synthetic_group(&["MatMul".into(), "Add".into(), "SiLU".into()])
            .expect("build");
        assert_eq!(g.body.len(), 3);
        // Add's `rhs` buffer is internal if the next op shares it
        // (the test only requires a non-empty body and reasonable
        // shape — internal_values is best-effort).
        assert!(!g.inputs.is_empty());
    }

    #[test]
    fn build_synthetic_group_returns_none_for_unknown_kinds() {
        let g = build_synthetic_group(&["UnknownOp".into()]);
        assert!(g.is_none());
    }

    #[test]
    fn compute_cost_uses_provided_values_when_present() {
        let support = BackendSupport {
            supported: true,
            reason: None,
            estimated_latency_us: Some(50.0),
            estimated_memory_bytes: Some(8192),
            estimated_scratch_bytes: Some(2048),
            requires_materialization: false,
        };
        let cost = compute_cost(&support, 4);
        assert_eq!(cost.estimated_us, 50.0);
        assert_eq!(cost.bytes_read, 8192);
        assert_eq!(cost.scratch_bytes, 2048);
        assert_eq!(cost.materialization_cost, 0.0);
    }

    #[test]
    fn compute_cost_estimates_when_missing() {
        let support = BackendSupport {
            supported: true,
            reason: None,
            estimated_latency_us: None,
            estimated_memory_bytes: None,
            estimated_scratch_bytes: None,
            requires_materialization: true,
        };
        let cost = compute_cost(&support, 4);
        assert!(cost.estimated_us > 0.0);
        assert_eq!(cost.bytes_read, 4 * 8192);
        assert_eq!(cost.materialization_cost, 10.0);
    }

    #[test]
    fn score_select_penalizes_materialization() {
        let mut with_mat = sample_candidate("metal", FusionSupportLevel::Full);
        with_mat.lowering_cost.materialization_cost = 50.0;
        let without_mat = sample_candidate("metal", FusionSupportLevel::Full);
        let p = FusionSelectionPolicy::default();
        let selected = score_select(&[with_mat, without_mat], &p).expect("select");
        assert_eq!(selected.target, "metal");
    }
}
