//! Prism ECS constitutional commands — session, work, execution, compilation,
//! multimodal, agent, distributed, and ingress authority systems.

pub mod admission_gates;
pub mod agent_exec;
pub mod agent_plan;
pub mod agent_reflection;
pub mod agent_state;
pub mod artifact;
pub mod command;
pub mod compilation;
pub mod config;
pub mod device;
pub mod distributed;
pub mod driver;
pub mod envelope;
pub mod event_store;
pub mod execution;
pub mod ingress;
pub mod lifecycle;
pub mod lifecycle_command;
pub mod migration;
pub mod multimodal;
pub mod persistence;
pub mod pipeline_bridge;
pub mod residency;
pub mod scheduler;
pub mod schema;
pub mod session;
pub mod sparse_set;
pub mod system_desc;
pub mod types;
pub mod work;
pub mod world_txn;

pub use agent_exec::*;
pub use agent_plan::*;
pub use agent_reflection::*;
pub use artifact::*;
pub use command::*;
pub use compilation::*;
pub use device::*;
pub use distributed::*;
pub use driver::*;
pub use envelope::*;
pub use event_store::*;
pub use execution::*;
pub use ingress::*;
pub use types::*;
// `ffi` re-export removed: the C-ABI bridge now lives in its own
// crate `prism-ecs-ffi`. Callers should `use prism_ecs_ffi::*;`
// (the FFI surface re-exports its symbols at the crate root).
pub use lifecycle::*;
pub use lifecycle_command::*;
pub use migration::*;
pub use multimodal::*;
pub use persistence::*;
pub use pipeline_bridge::*;
pub use residency::*;
pub use scheduler::*;
pub use schema::*;
pub use session::*;
pub use sparse_set::*;
pub use system_desc::*;
pub use work::*;
pub use world_txn::*;
