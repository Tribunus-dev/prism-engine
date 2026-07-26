use crate::ecs::plan::fusion_scheduler_types::{
    FusionCandidate, FusionPolicy, FusionRejection, FusionSupportLevel,
};
use crate::ecs::Component;
use serde::{Deserialize, Serialize};

// ── FusionGroup ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    pub root_op_kind: String,
    pub fused_op_kinds: Vec<String>,
    pub binding_slots: u32,
    pub accepted: bool,
    pub reject_reason: Option<String>,
}
impl Component for FusionGroup {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkgroupCount(pub u32, pub u32, pub u32);
impl Component for WorkgroupCount {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TileSize(pub u32, pub u32, pub u32);
impl Component for TileSize {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BindingCapacity {
    pub max_slots: u32,
    pub max_bytes_per_slot: u64,
}
impl Component for BindingCapacity {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowGraphHandle(pub String); // filename/key referencing DataflowGraph
impl Component for DataflowGraphHandle {}

// ── FusionPolicyComp ───────────────────────────────────────────────────────

/// ECS component wrapping FusionPolicy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPolicyComp(pub Box<FusionPolicy>);
impl Component for FusionPolicyComp {}

// ── LoweringCost ───────────────────────────────────────────────────────────

/// ECS component wrapping a lowering cost estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweringCost(pub crate::ecs::plan::fusion_scheduler_types::LoweringCost);
impl Component for LoweringCost {}

// ── FusionScheduleData ─────────────────────────────────────────────────────

/// ECS component storing a full FusionEvaluation result for a dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionScheduleData {
    pub candidates: Vec<FusionCandidate>,
    pub selected: Option<FusionCandidate>,
}
impl Component for FusionScheduleData {}

// ── FusionEvaluationData ───────────────────────────────────────────────────

/// ECS component storing rejection details from evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionEvaluationData {
    pub source_nodes: Vec<usize>,
    pub rejected: Vec<FusionRejection>,
}
impl Component for FusionEvaluationData {}

// ── FusionSupportLevelV2 ───────────────────────────────────────────────────

/// ECS component wrapping FusionSupportLevel.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FusionSupportLevelV2(pub FusionSupportLevel);
impl Component for FusionSupportLevelV2 {}

// ── PowerClass ─────────────────────────────────────────────────────────────

/// ECS component wrapping backend PowerClass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PowerClass(pub crate::ecs::plan::backend_capability::PowerClass);
impl Component for PowerClass {}

// ── PrecisionClassEnum ─────────────────────────────────────────────────────

/// ECS component wrapping PrecisionClass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PrecisionClassEnum(pub crate::ecs::plan::backend_capability::PrecisionClass);
impl Component for PrecisionClassEnum {}
