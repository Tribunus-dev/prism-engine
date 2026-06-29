//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Typed backpressure signals.
//!
//! Provides typed backpressure events, severity levels, and a controller
//! that tracks active backpressure across all heterogeneous execution lanes
//! (Metal GPU, ANE, CPU, etc.) for the request scheduler.
//!
//! The controller translates active events into a single [`BackpressureLevel`]
//! used by the scheduler to throttle or cancel work.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::backend::placement::ExecutionLane;

// ---------------------------------------------------------------------------
// BackpressureReason — specific resource or capacity cause
// ---------------------------------------------------------------------------

/// Specific reason for a backpressure signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackpressureReason {
    /// Metal GPU command buffer / compute pipeline at capacity.
    MetalCapacity,
    /// Apple Neural Engine at capacity.
    AneCapacity,
    /// CPU compute lanes saturated.
    CpuCapacity,
    /// Activation arena slot exhaustion.
    ActivationSlots,
    /// IOSurface pool depleted or fragmented.
    IOSurfacePool,
    /// Per-session resource quota exceeded.
    SessionQuota,
    /// Global dispatch queue depth limit reached.
    GlobalQueue,
    /// ANE artifact-cache miss causing cold-load delay.
    ArtifactCold,
}

impl BackpressureReason {
    /// Human-readable description of this backpressure reason.
    pub fn description(&self) -> &'static str {
        match self {
            Self::MetalCapacity => "Metal GPU command buffer or compute pipeline at capacity",
            Self::AneCapacity => "Apple Neural Engine at capacity",
            Self::CpuCapacity => "CPU compute lanes saturated",
            Self::ActivationSlots => "Activation arena slot exhaustion",
            Self::IOSurfacePool => "IOSurface pool depleted or fragmented",
            Self::SessionQuota => "Per-session resource quota exceeded",
            Self::GlobalQueue => "Global dispatch queue depth limit reached",
            Self::ArtifactCold => "ANE artifact-cache miss causing cold-load delay",
        }
    }

    /// Whether this reason represents a transient condition.
    /// All built-in reasons are transient by default.
    pub fn is_transient(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// BackpressureEvent — a single observed backpressure occurrence
// ---------------------------------------------------------------------------

/// A backpressure event with reason, severity, and affected lane.
#[derive(Debug, Clone)]
pub struct BackpressureEvent {
    /// The specific resource or capacity reason.
    pub reason: BackpressureReason,
    /// The execution lane experiencing backpressure, if lane-specific.
    pub lane: Option<ExecutionLane>,
    /// Session identifier, if the event is session-scoped.
    pub affected_session: Option<String>,
    /// Monotonic timestamp of when the event was observed.
    pub timestamp: Instant,
    /// Free-form details for diagnostics or observability.
    pub details: String,
}

// ---------------------------------------------------------------------------
// BackpressureLevel — aggregate severity for the request scheduler
// ---------------------------------------------------------------------------

/// Backpressure level for the request scheduler.
///
/// Levels are ordered from least to most severe. The controller derives the
/// current level from active events based on the most severe contributing
/// reason category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BackpressureLevel {
    /// Normal operation — no backpressure.
    None,
    /// Mild — new work should be delayed briefly.
    Mild,
    /// Moderate — only high-priority work should be admitted.
    Moderate,
    /// Severe — no new work should be admitted.
    Severe,
    /// Critical — in-flight work may need to be cancelled.
    Critical,
}

// ---------------------------------------------------------------------------
// BackpressureController — tracks state across all resources
// ---------------------------------------------------------------------------

/// Tracks backpressure state across all resources.
///
/// Maintains a vector of active [`BackpressureEvent`]s and derives a current
/// [`BackpressureLevel`] based on the most severe category present.
pub struct BackpressureController {
    active_events: Vec<BackpressureEvent>,
    level: BackpressureLevel,
    max_events: usize,
}

impl BackpressureController {
    /// Creates a new controller with default capacity (256 events).
    pub fn new() -> Self {
        Self {
            active_events: Vec::new(),
            level: BackpressureLevel::None,
            max_events: 256,
        }
    }

    /// Creates a new controller with a custom event capacity.
    pub fn with_max_events(max: usize) -> Self {
        Self {
            active_events: Vec::with_capacity(max.min(4096)),
            level: BackpressureLevel::None,
            max_events: max.min(4096),
        }
    }

    /// Record a backpressure event.
    ///
    /// Adds the event to the active set and recalculates the aggregate
    /// backpressure level. If the event buffer is full, the oldest event
    /// is evicted.
    pub fn report(&mut self, event: BackpressureEvent) {
        if self.active_events.len() >= self.max_events {
            self.active_events.remove(0);
        }
        self.active_events.push(event);
        self.recalculate_level();
    }

    /// Clear expired/resolved events older than the given instant.
    ///
    /// After removal the controller recalculates the aggregate level.
    pub fn clear_before(&mut self, cutoff: Instant) {
        self.active_events.retain(|e| e.timestamp >= cutoff);
        self.recalculate_level();
    }

    /// Get current backpressure level.
    pub fn level(&self) -> BackpressureLevel {
        self.level
    }

    /// Manually set backpressure level (e.g., after clearing events externally).
    pub fn set_level(&mut self, level: BackpressureLevel) {
        self.level = level;
    }

    /// Get all active events.
    pub fn events(&self) -> &[BackpressureEvent] {
        &self.active_events
    }

    /// Clear all events and reset to `None`.
    pub fn clear(&mut self) {
        self.active_events.clear();
        self.level = BackpressureLevel::None;
    }

    /// Recalculate the aggregate level from active events.
    ///
    /// Priority (highest to lowest):
    /// - **Critical** if any `SessionQuota` or `GlobalQueue` events are present.
    /// - **Severe** if any lane-capacity events (`MetalCapacity`, `AneCapacity`,
    ///   `CpuCapacity`) are present.
    /// - **Moderate** if any `IOSurfacePool` or `ActivationSlots` events are present.
    /// - **Mild** if any `ArtifactCold` events are present.
    /// - **None** if no events are present.
    fn recalculate_level(&mut self) {
        self.level = derive_level(&self.active_events);
    }

    /// Produce a serializable summary of the current backpressure state.
    pub fn summary(&self) -> BackpressureSummary {
        let mut reasons: Vec<BackpressureReason> = Vec::new();
        let mut affected_lanes: Vec<ExecutionLane> = Vec::new();

        for event in &self.active_events {
            if !reasons.contains(&event.reason) {
                reasons.push(event.reason);
            }
            if let Some(lane) = &event.lane {
                if !affected_lanes.contains(lane) {
                    affected_lanes.push(*lane);
                }
            }
        }

        BackpressureSummary {
            level: self.level,
            active_event_count: self.active_events.len(),
            reasons,
            affected_lanes,
        }
    }
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BackpressureSummary — snapshot for observability / telemetry
// ---------------------------------------------------------------------------

/// Summary of backpressure state for observability.
#[derive(Debug, Clone, Serialize)]
pub struct BackpressureSummary {
    /// Aggregate severity level.
    pub level: BackpressureLevel,
    /// Number of active events contributing to the level.
    pub active_event_count: usize,
    /// Unique list of backpressure reasons present among active events.
    pub reasons: Vec<BackpressureReason>,
    /// Unique set of lanes that have active backpressure events.
    pub affected_lanes: Vec<ExecutionLane>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive the aggregate [`BackpressureLevel`] from a slice of events.
///
/// Uses the highest-priority category present:
/// 1. `SessionQuota` / `GlobalQueue` → Critical
/// 2. `MetalCapacity` / `AneCapacity` / `CpuCapacity` → Severe
/// 3. `IOSurfacePool` / `ActivationSlots` → Moderate
/// 4. `ArtifactCold` → Mild
/// 5. No events → None
fn derive_level(events: &[BackpressureEvent]) -> BackpressureLevel {
    let mut level = BackpressureLevel::None;

    for event in events {
        match event.reason {
            // Critical — session or global resource exhaustion
            BackpressureReason::SessionQuota | BackpressureReason::GlobalQueue => {
                return BackpressureLevel::Critical;
            }
            // Severe — lane-level capacity
            BackpressureReason::MetalCapacity
            | BackpressureReason::AneCapacity
            | BackpressureReason::CpuCapacity => {
                if level < BackpressureLevel::Severe {
                    level = BackpressureLevel::Severe;
                }
            }
            // Moderate — shared pool / slot exhaustion
            BackpressureReason::IOSurfacePool | BackpressureReason::ActivationSlots => {
                if level < BackpressureLevel::Moderate {
                    level = BackpressureLevel::Moderate;
                }
            }
            // Mild — transient delays from cold caches
            BackpressureReason::ArtifactCold => {
                if level < BackpressureLevel::Mild {
                    level = BackpressureLevel::Mild;
                }
            }
        }
    }

    level
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_level_is_none() {
        let ctrl = BackpressureController::new();
        assert_eq!(ctrl.level(), BackpressureLevel::None);
        assert!(ctrl.events().is_empty());
    }

    #[test]
    fn test_report_artifact_cold_is_mild() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::ArtifactCold,
            lane: None,
            affected_session: Some("sess-1".into()),
            timestamp: Instant::now(),
            details: "cache miss on layer 3".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Mild);
        assert_eq!(ctrl.events().len(), 1);
    }

    #[test]
    fn test_report_iosurface_pool_is_moderate() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::IOSurfacePool,
            lane: Some(ExecutionLane::CoreMlAne),
            affected_session: None,
            timestamp: Instant::now(),
            details: "pool utilization 94%".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Moderate);
    }

    #[test]
    fn test_report_activation_slots_is_moderate() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::ActivationSlots,
            lane: None,
            affected_session: Some("sess-2".into()),
            timestamp: Instant::now(),
            details: "slots exhausted".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Moderate);
    }

    #[test]
    fn test_report_lane_capacity_is_severe() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::MetalCapacity,
            lane: Some(ExecutionLane::MlxGpu),
            affected_session: None,
            timestamp: Instant::now(),
            details: "GPU command buffer full".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Severe);
    }

    #[test]
    fn test_report_session_quota_is_critical() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::SessionQuota,
            lane: None,
            affected_session: Some("sess-1".into()),
            timestamp: Instant::now(),
            details: "quota exceeded: 512 tokens".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Critical);
    }

    #[test]
    fn test_report_global_queue_is_critical() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::GlobalQueue,
            lane: None,
            affected_session: None,
            timestamp: Instant::now(),
            details: "dispatch depth 2048".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Critical);
    }

    #[test]
    fn test_highest_priority_wins() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::ArtifactCold,
            lane: None,
            affected_session: None,
            timestamp: Instant::now(),
            details: "cold".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Mild);

        // A moderate event should override mild
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::ActivationSlots,
            lane: None,
            affected_session: None,
            timestamp: Instant::now(),
            details: "slots".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Moderate);

        // A severe event should override moderate
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::AneCapacity,
            lane: Some(ExecutionLane::CoreMlAne),
            affected_session: None,
            timestamp: Instant::now(),
            details: "ANE full".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Severe);

        // A critical event should override severe
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::SessionQuota,
            lane: None,
            affected_session: Some("sess-1".into()),
            timestamp: Instant::now(),
            details: "quota".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Critical);
    }

    #[test]
    fn test_clear_before_removes_old_events() {
        let mut ctrl = BackpressureController::new();
        let now = Instant::now();

        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::ArtifactCold,
            lane: None,
            affected_session: None,
            timestamp: now - Duration::from_secs(10),
            details: "old".into(),
        });
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::MetalCapacity,
            lane: None,
            affected_session: None,
            timestamp: now,
            details: "recent".into(),
        });

        // All events have timestamp >= now - 5
        ctrl.clear_before(now - Duration::from_secs(5));
        assert_eq!(ctrl.events().len(), 1);
        assert_eq!(ctrl.level(), BackpressureLevel::Severe);

        ctrl.clear_before(now + Duration::from_secs(1));
        assert_eq!(ctrl.events().len(), 0);
        assert_eq!(ctrl.level(), BackpressureLevel::None);
    }

    #[test]
    fn test_clear_resets_to_none() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::GlobalQueue,
            lane: None,
            affected_session: None,
            timestamp: Instant::now(),
            details: "queue full".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Critical);

        ctrl.clear();
        assert_eq!(ctrl.level(), BackpressureLevel::None);
        assert!(ctrl.events().is_empty());
    }

    #[test]
    fn test_set_level_manual_override() {
        let mut ctrl = BackpressureController::new();
        ctrl.set_level(BackpressureLevel::Severe);
        assert_eq!(ctrl.level(), BackpressureLevel::Severe);
    }

    #[test]
    fn test_summary_aggregates_reasons_and_lanes() {
        let mut ctrl = BackpressureController::new();

        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::MetalCapacity,
            lane: Some(ExecutionLane::MlxGpu),
            affected_session: None,
            timestamp: Instant::now(),
            details: "gpu full".into(),
        });
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::ArtifactCold,
            lane: Some(ExecutionLane::CoreMlAne),
            affected_session: None,
            timestamp: Instant::now(),
            details: "cold".into(),
        });
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::MetalCapacity,
            lane: Some(ExecutionLane::MlxGpu),
            affected_session: None,
            timestamp: Instant::now(),
            details: "gpu still full".into(),
        });

        let summary = ctrl.summary();
        assert_eq!(summary.level, BackpressureLevel::Severe);
        assert_eq!(summary.active_event_count, 3);
        assert_eq!(summary.reasons.len(), 2);
        assert!(summary.reasons.contains(&BackpressureReason::MetalCapacity));
        assert!(summary.reasons.contains(&BackpressureReason::ArtifactCold));
        assert_eq!(summary.affected_lanes.len(), 2);
    }

    #[test]
    fn test_with_max_events_caps_buffer() {
        let mut ctrl = BackpressureController::with_max_events(3);
        for i in 0..5 {
            ctrl.report(BackpressureEvent {
                reason: BackpressureReason::ArtifactCold,
                lane: None,
                affected_session: None,
                timestamp: Instant::now(),
                details: format!("event {i}"),
            });
        }
        assert_eq!(ctrl.events().len(), 3);
        // Oldest event was evicted three times, so "event 2" is the oldest survivor
        assert!(ctrl.events().iter().any(|e| e.details == "event 2"));
        assert!(ctrl.events().iter().any(|e| e.details == "event 4"));
    }

    #[test]
    fn test_backpressure_reason_description() {
        assert_eq!(
            BackpressureReason::MetalCapacity.description(),
            "Metal GPU command buffer or compute pipeline at capacity"
        );
        assert!(BackpressureReason::AneCapacity.is_transient());
        assert!(BackpressureReason::GlobalQueue.is_transient());
    }

    #[test]
    fn test_multiple_severe_lanes_stay_severe() {
        let mut ctrl = BackpressureController::new();
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::MetalCapacity,
            lane: Some(ExecutionLane::MlxGpu),
            affected_session: None,
            timestamp: Instant::now(),
            details: "".into(),
        });
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::AneCapacity,
            lane: Some(ExecutionLane::CoreMlAne),
            affected_session: None,
            timestamp: Instant::now(),
            details: "".into(),
        });
        ctrl.report(BackpressureEvent {
            reason: BackpressureReason::CpuCapacity,
            lane: Some(ExecutionLane::CandleCpu),
            affected_session: None,
            timestamp: Instant::now(),
            details: "".into(),
        });
        assert_eq!(ctrl.level(), BackpressureLevel::Severe);
    }

    #[test]
    fn test_default_impl() {
        let ctrl = BackpressureController::default();
        assert_eq!(ctrl.level(), BackpressureLevel::None);
    }
}
