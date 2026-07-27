//! Scheduling state (Components) — authoritative scheduling data.
//!
//! Every type in this module is **authoritative ECS state** in the C
//! bucket. A mutation becomes visible to other systems only after a
//! `ConstitutionalWorldTxn` commits it. Transient implementation state
//! (kernel fences, backend queues, GPU completion callbacks) does not
//! live here — it lives in `prism-ecs-kernel::backend::*` and re-enters
//! this state only through the runtime completion-reconciliation system.

pub mod lane_capacity;
pub mod lane_work;
