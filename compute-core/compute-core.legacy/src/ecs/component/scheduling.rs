use crate::ecs::Component;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// WorkState — lifecycle state for a work item
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkState {
    Created,
    Ready,
    Selected,
    CapacityReserved,
    SlotsReserved,
    Submitted,
    Running,
    Denied,
    Cancelling,
    FallbackPending,
    FallbackRunning,
    Complete,
    Failed,
}

// ---------------------------------------------------------------------------
// BackpressureLevel — aggregate backpressure severity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BackpressureLevel {
    None,
    Mild,
    Moderate,
    Severe,
    Critical,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Tracks the lifecycle state of a work item in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkRegistryComponent {
    pub state: WorkState,
    pub created_at: u64,
}
impl Component for WorkRegistryComponent {}

/// Aggregate backpressure level and queue depth for a resource or lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureComponent {
    pub level: BackpressureLevel,
    pub queue_depth: u32,
}
impl Component for BackpressureComponent {}
/// Session state — decode step, active model, generation params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub decode_step: u64,
    pub active_model: String,
    pub generation_params_json: String,
}
impl Component for SessionState {}

/// Phase DAG state — phase names, edges, current phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDagState {
    pub phase_names: Vec<String>,
    pub edges: Vec<(String, String)>,
    pub current_phase: String,
}
impl Component for PhaseDagState {}

/// Ready queue state — pending work item IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyQueueState {
    pub pending_items: Vec<String>,
}
impl Component for ReadyQueueState {}
