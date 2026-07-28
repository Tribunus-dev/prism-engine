//! Re-exports of `LaneAdmissionGate` and `RiskPolicy` from the constitutional
//! `admission_gates` module. Engine callers that imported these from
//! `prism_ecs_compile::compilation::*` continue to work without churn.

pub use prism_ecs_constitutional::admission_gates::{LaneAdmissionGate, RiskPolicy};
