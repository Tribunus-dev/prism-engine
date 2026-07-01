//! ECS component types for Prism Engine worker supervision.
//!
//! Each component represents one axis of a worker-bound inference request's
//! lifecycle state.  Components are indexed by entity ID (one entity = one
//! submitted request).

pub mod worker_request;
pub mod agent_core;
pub mod worker_assignment;
pub mod worker_lifecycle;
pub mod worker_stream;
pub mod worker_health;

pub use worker_request::WorkerRequest;
pub use worker_assignment::WorkerAssignment;
pub use worker_lifecycle::{WorkerLifecycle, WorkerRequestPhase};
pub use worker_stream::WorkerStream;
pub use worker_stream::HardwareStreamHandle;
pub use worker_health::{WorkerHeartbeat, WorkerOutcome};
pub use agent_core::{AgentSlot, AgentPayload, KVCacheRef, ToolRegistry, ToolDef, AgentStatus};

// ── Stable component IDs ──────────────────────────────────────────────

pub const WORKER_REQUEST_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 10;
pub const WORKER_ASSIGNMENT_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 11;
pub const WORKER_LIFECYCLE_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 12;
pub const WORKER_STREAM_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 13;
pub const WORKER_HEARTBEAT_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 14;
pub const WORKER_OUTCOME_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 15;
pub const WORKER_HARDWARE_STREAM_COMPONENT: crate::runtime::scheduling::component_id::ComponentId = 16;

// ── Stable system IDs ─────────────────────────────────────────────────

pub const WORKER_INGRESS_SYSTEM: crate::runtime::scheduling::metadata::SystemId =
    crate::runtime::scheduling::metadata::SystemId(100);
pub const WORKER_EVENT_DRAIN_SYSTEM: crate::runtime::scheduling::metadata::SystemId =
    crate::runtime::scheduling::metadata::SystemId(101);
pub const WORKER_WATCHDOG_SYSTEM: crate::runtime::scheduling::metadata::SystemId =
    crate::runtime::scheduling::metadata::SystemId(102);
