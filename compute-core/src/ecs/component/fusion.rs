use crate::ecs::Component;
use serde::{Deserialize, Serialize};

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
