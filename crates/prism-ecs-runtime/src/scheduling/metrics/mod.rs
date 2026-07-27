//! Scheduling metrics (M bucket) — advisory metrics, rebuildable projection.
//!
//! Metrics in this module are NOT evidence. They are rebuildable
//! projections of the canonical scheduling state. A metric becomes
//! evidence only when a measurement is admitted into an immutable
//! receipt (see `prism-ecs-runtime::evidence::*`).
//!
//! # Migration status
//!
//! Per the inventory v2.1 (steps 53-55), the metrics files
//! (outlier_detector, scheduler_metrics, phase_telemetry) move here
//! from the engine.

pub mod outlier_detector;
pub mod phase_telemetry;
pub mod scheduler_metrics;
