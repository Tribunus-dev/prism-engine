//! Scheduling state (Components) — authoritative scheduling data.
//!
//! Every type in this module is **authoritative ECS state** in the C
//! bucket. A mutation becomes visible to other systems only after a
//! `ConstitutionalWorldTxn` commits it. Transient implementation state
//! (kernel fences, backend queues, GPU completion callbacks) does not
//! live here — it lives in `prism-ecs-kernel::backend::*` and re-enters
//! this state only through the runtime completion-reconciliation system.

pub mod activation_transaction;
pub mod batch;
pub mod execution_context;
pub mod lane_capacity;
pub mod lane_work;
pub mod lease;
pub mod phase;
pub mod phase_cancellation;
pub mod phase_engine_state;
pub mod phase_invocation;
pub mod ready_queue;
pub mod request;
pub mod token_budget;
pub mod work_registry;
