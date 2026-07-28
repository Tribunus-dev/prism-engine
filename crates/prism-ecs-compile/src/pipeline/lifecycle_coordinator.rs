//! `pipeline::lifecycle_coordinator` — production composition root types.
//!
//! This file owns the canonical authority for the public type-level
//! surface of the lifecycle coordinator: the
//! [`CompilerRequest`] / [`LifecycleResult`] contract, the
//! [`PolicyConfig`] / [`PromotionPolicy`] knobs, and the
//! [`LifecycleCoordinator`] shell. The full hardware-gated coordinator
//! implementation lives in the engine; the constitutional surface here
//! provides the typed contract that callers depend on.

use std::collections::BTreeMap;

use super::event_emitter::CompilerEventStream;

/// Input to a single lifecycle run.
#[derive(Debug, Clone)]
pub struct CompilerRequest {
    /// Source model identifier.
    pub source_id: String,
    /// Precision / codec families the lifecycle should target.
    pub precision_targets: Vec<String>,
    /// Whether engram training should be included in the lifecycle.
    pub engram_training: bool,
}

impl Default for CompilerRequest {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            precision_targets: Vec::new(),
            engram_training: false,
        }
    }
}

/// Output from a completed or cancelled lifecycle.
#[derive(Debug, Clone)]
pub struct LifecycleResult {
    /// Identifier of the generated artifact, if any.
    pub generation_id: Option<String>,
    /// Compiled kernel artifacts from this lifecycle (keyed by id).
    pub artifacts: BTreeMap<String, KernelArtifact>,
    /// Full event stream for the lifecycle.
    pub event_stream: CompilerEventStream,
    /// Receipt bundle identifier, if available.
    pub receipt_bundle: Option<String>,
    /// Whether the lifecycle completed successfully.
    pub success: bool,
    /// Rejection reason if the lifecycle was rejected.
    pub rejection_reason: Option<String>,
    /// Number of kernels dispatched during evaluation.
    pub dispatch_count: usize,
    /// Maximum measured GPU latency in nanoseconds across all dispatches.
    pub measured_latency_ns: u64,
    /// Maximum absolute numerical error from the CPU oracle comparison.
    pub numerical_max_error: f64,
    /// Optional smoke test result, if attempted.
    pub smoke_result: Option<SmokeResult>,
    /// Whether the generation was promoted without full numerical
    /// confidence and requires manual validation.
    pub needs_validation: bool,
}

/// Compiled kernel artifact carrier.
#[derive(Debug, Clone)]
pub struct KernelArtifact {
    /// Identifier of the implementation that produced this artifact.
    pub implementation_id: String,
    /// Compiled bytes (the .metallib on Apple platforms).
    pub compiled_bytes: Vec<u8>,
}

/// Result of an admission smoke test.
#[derive(Debug, Clone)]
pub struct SmokeResult {
    /// Total latency for the prefill phase in nanoseconds.
    pub prefill_latency_ns: u64,
    /// Total latency for the first decode step in nanoseconds.
    pub decode_latency_ns: u64,
    /// Maximum absolute element-wise error vs the CPU reference.
    pub max_error_vs_cpu: f64,
    /// Root mean squared error vs the CPU reference.
    pub rmse_vs_cpu: f64,
    /// Number of target layers that were successfully dispatched.
    pub layers_dispatched: usize,
}

/// Budget and constraint configuration that governs a lifecycle.
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Maximum allowed runtime in seconds.
    pub max_runtime_seconds: u64,
    /// Maximum allowed memory in bytes.
    pub max_memory_bytes: u64,
    /// Required receipt identifiers for admission.
    pub required_receipts: Vec<String>,
    /// Required device identifiers.
    pub device_requirements: Vec<String>,
    /// Promotion policy.
    pub promotion_policy: PromotionPolicy,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            max_runtime_seconds: 300,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            required_receipts: vec![],
            device_requirements: vec![],
            promotion_policy: PromotionPolicy::BestEffort,
        }
    }
}

/// How the promotion gate handles incomplete evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionPolicy {
    /// Reject unless every receipt resolves.
    FailClosed,
    /// Accept if critical receipts pass.
    BestEffort,
}

/// The narrow production composition root for the complete compilation
/// lifecycle.
///
/// The constitutional surface provides the typed shell; the engine's
/// hardware-gated implementation supplies the full coordinator state.
#[derive(Debug, Clone, Default)]
pub struct LifecycleCoordinator {
    /// Whether a lifecycle is currently in progress.
    pub active: bool,
    /// Compiled kernel artifacts accumulated during the active lifecycle.
    pub artifacts: BTreeMap<String, KernelArtifact>,
    /// Event stream for the active lifecycle.
    pub event_stream: CompilerEventStream,
    /// Policy configuration.
    pub policy: PolicyConfig,
}

impl LifecycleCoordinator {
    /// Create a new coordinator with default state and policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the policy configuration and return the modified coordinator.
    pub fn with_policy(mut self, policy: PolicyConfig) -> Self {
        self.policy = policy;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_best_effort() {
        assert_eq!(
            PolicyConfig::default().promotion_policy,
            PromotionPolicy::BestEffort
        );
    }

    #[test]
    fn default_max_memory_is_8gib() {
        assert_eq!(
            PolicyConfig::default().max_memory_bytes,
            8 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn default_max_runtime_is_5min() {
        assert_eq!(PolicyConfig::default().max_runtime_seconds, 300);
    }

    #[test]
    fn coordinator_starts_inactive() {
        let coord = LifecycleCoordinator::new();
        assert!(!coord.active);
        assert!(coord.artifacts.is_empty());
    }

    #[test]
    fn with_policy_replaces_policy() {
        let coord = LifecycleCoordinator::new().with_policy(PolicyConfig {
            max_runtime_seconds: 60,
            max_memory_bytes: 1024,
            required_receipts: vec!["r1".into()],
            device_requirements: vec!["metal".into()],
            promotion_policy: PromotionPolicy::FailClosed,
        });
        assert_eq!(coord.policy.max_runtime_seconds, 60);
        assert_eq!(coord.policy.max_memory_bytes, 1024);
        assert_eq!(coord.policy.promotion_policy, PromotionPolicy::FailClosed);
    }
}
