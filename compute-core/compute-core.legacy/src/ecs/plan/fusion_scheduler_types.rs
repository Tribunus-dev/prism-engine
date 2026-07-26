//! Metadata types for the fusion scheduler pipeline.
//!
//! These type definitions are shared between the ECS fusion systems and the
//! legacy execution_plan module.  They were extracted from `fusion_scheduler.rs`
//! so that ECS system files import from a type-only module rather than a
//! mixed types-and-functions module.

use serde::{Deserialize, Serialize};

use crate::ecs::plan::backend_capability::BackendLoweringTarget;
use crate::ecs::plan::fusion::FusedGroup;
use crate::ecs::plan::ExecutionMode;

// ── BackendTarget (receipts.rs compatibility alias) ─────────────────────────

/// Backend target for lowering — alias for `BackendLoweringTarget`.
pub type BackendTarget = crate::ecs::plan::backend_capability::BackendLoweringTarget;

// ── FusionPolicy ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPolicy {
    pub max_group_size: usize,
    pub allow_materialization: bool,
    pub allow_research_fusions: bool,
    /// How the scheduler treats groups without a viable backend.
    pub execution_mode: ExecutionMode,
}

impl Default for FusionPolicy {
    fn default() -> Self {
        Self {
            max_group_size: 8,
            allow_materialization: true,
            allow_research_fusions: false,
            execution_mode: ExecutionMode::Explore,
        }
    }
}

// ── LoweringCost ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoweringCost {
    pub estimated_us: f64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub scratch_bytes: u64,
    pub thread_count: u32,
    pub materialization_cost: f64,
}

// ── FusionSupportLevel ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionSupportLevel {
    Full,
    Partial,
    Unsupported,
}

// ── FusionCandidate ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCandidate {
    pub group: FusedGroup,
    pub target: BackendLoweringTarget,
    pub support: FusionSupportLevel,
    pub lowering_cost: LoweringCost,
}

// ── FusionRejection ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionRejection {
    pub group_id: String,
    pub target: BackendLoweringTarget,
    pub reason: String,
}

// ── FusionEvaluation ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionEvaluation {
    pub source_nodes: Vec<usize>,
    pub candidates: Vec<FusionCandidate>,
    pub selected: Option<FusionCandidate>,
    pub rejected: Vec<FusionRejection>,
}

// ── FusionSchedule ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSchedule {
    pub groups: Vec<FusedGroup>,
    pub receipts: Vec<FusionEvaluation>,
}

// ── FusionSelectionPolicy ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSelectionPolicy {
    pub prefer_lower_latency: bool,
    pub prefer_memory_efficient: bool,
    pub avoid_materialization: bool,
}

impl Default for FusionSelectionPolicy {
    fn default() -> Self {
        Self {
            prefer_lower_latency: true,
            prefer_memory_efficient: false,
            avoid_materialization: true,
        }
    }
}

// ── FusionError ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum FusionError {
    EmptyGraph,
    NoViableBackend { group_id: String, reason: String },
    UnselectedGroupInCompileMode { group_id: String },
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::EmptyGraph => write!(f, "fusion schedule received an empty graph"),
            FusionError::NoViableBackend { group_id, reason } => {
                write!(f, "no viable backend for group {group_id}: {reason}")
            }
            FusionError::UnselectedGroupInCompileMode { group_id } => {
                write!(f, "unselected group {group_id} in Compile mode")
            }
        }
    }
}

impl std::error::Error for FusionError {}
