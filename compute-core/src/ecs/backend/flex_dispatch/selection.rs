//! FlexDispatch — kernel and path selection logic.
//!
//! Routes each operation to MLX (GPU), Core ML (ANE), or Accelerate
//! (CPU/NEON) based on the real-time [`SystemState`] sampled by
//! [`profiling`](crate::ecs::backend::flex_dispatch::profiling).

use super::profiling::SystemState;
use crate::ecs::backend::heterogeneous_executor::HeterogeneousExecutor;
use crate::ecs::backend::routing::*;
use crate::ecs::backend::routing::{BACKEND_ACCELERATE, BACKEND_ANE, BACKEND_MLX};
use crate::ecs::compilation::phase_types::PhaseType;
use crate::ecs::scheduling::outlier_detector::OutlierDetector;
use std::sync::{Arc, Mutex};

/// Default decode-step interval for sampling system state.
pub const DEFAULT_SAMPLE_INTERVAL: u32 = 16;

// ── Operation classification ──────────────────────────────────────────────

/// Simplified operation classification for dispatch decisions.
///
/// The five families map directly to the dispatch `match` in
/// [`FlexDispatch::dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchFamily {
    MatMul,
    Attention,
    ElementWise,
    Softmax,
    LayerNorm,
}

/// Classify a [`routing::OperationFamily`] into a [`DispatchFamily`] for
/// the flex dispatcher.
pub(crate) fn classify_family(family: OperationFamily) -> DispatchFamily {
    use OperationFamily::*;
    match family {
        Matmul | QuantizedMatmul | MlpBlock => DispatchFamily::MatMul,
        AttentionBlock | DecoderLayer | PrefillFragment => DispatchFamily::Attention,
        Silu | Add | Multiply | Transpose | Reshape | IndexSelect | Sampling | Reduction => {
            DispatchFamily::ElementWise
        }
        Softmax => DispatchFamily::Softmax,
        RmsNorm | RoPE | LayoutTransform | Checksum => DispatchFamily::LayerNorm,
        VisionEncode | AudioEncode | MultimodalProject => DispatchFamily::ElementWise,
    }
}

/// Check if ANE offload is beneficial given current system state.
pub fn prefer_ane(state: &SystemState, family: OperationFamily) -> bool {
    state.should_offload_to_ane()
        && state.ane_active
        && matches!(
            family,
            OperationFamily::AttentionBlock
                | OperationFamily::MlpBlock
                | OperationFamily::DecoderLayer
        )
}

// ── FlexDispatch ──────────────────────────────────────────────────────────

/// Runtime backend dispatcher — adapts to real-time system conditions.
///
/// Every `sample_interval` decode steps, `FlexDispatch` samples the full
/// [`SystemState`] and uses it to route each incoming operation to the
/// best backend *right now*.
///
/// The dispatcher is stateless between samples; the decision logic is a
/// pure function of the current state and the operation family.
pub struct FlexDispatch {
    /// Last sampled system state.
    pub last_state: SystemState,
    /// How often to re-sample the system state (in decode steps).
    pub sample_interval: u32,
    /// Steps since the last sample.
    pub steps_since_sample: u32,
    /// Optional outlier detector for precision overrides.
    pub outlier_detector: Option<Arc<Mutex<OutlierDetector>>>,
}

impl FlexDispatch {
    /// Create a new dispatch controller with default sampling interval
    /// (16 decode steps).
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
            sample_interval: 16,
            steps_since_sample: u32::MAX, // Sample on first call.
            outlier_detector: None,
        }
    }

    /// Create a dispatch controller with a custom sampling interval.
    pub fn with_interval(steps: u32) -> Self {
        Self {
            last_state: SystemState::default(),
            sample_interval: steps,
            steps_since_sample: u32::MAX,
            outlier_detector: None,
        }
    }

    /// Force a system-state sample right now.
    pub fn sample_now(&mut self) {
        if let Ok(state) = SystemState::sample() {
            self.last_state = state;
        }
        self.steps_since_sample = 0;
    }

    /// Pick the best backend for an operation given current system state.
    ///
    /// Samples system state every `sample_interval` steps.  The decision
    /// logic is:
    ///
    /// | Family | GPU free & no throttle | GPU saturated | Throttling |
    /// |---|---|---|---|
    /// | MatMul | MLX (GPU) | MLX (GPU) | Core ML (ANE) |
    /// | Attention | MLX (GPU) | Core ML (ANE) | Accelerate (CPU) |
    /// | ElementWise | MLX (GPU) | Accelerate (CPU) | Accelerate (CPU) |
    /// | Softmax | Accelerate (CPU) | Accelerate (CPU) | Accelerate (CPU) |
    /// | LayerNorm | Accelerate (CPU) | Accelerate (CPU) | Accelerate (CPU) |
    pub fn dispatch(&mut self, op: &OperationDescriptor, _sequence: u32) -> BackendId {
        // Sample system state every N steps.
        self.steps_since_sample = self.steps_since_sample.wrapping_add(1);
        if self.steps_since_sample >= self.sample_interval {
            if let Ok(state) = SystemState::sample() {
                self.last_state = state;
            }
            self.steps_since_sample = 0;
        }

        let state = &self.last_state;
        let family = classify_family(op.family);

        // When GPU is saturated but the system shouldn't throttle,
        // offload attention and MLP blocks to ANE.
        if state.should_offload_to_ane()
            && matches!(
                op.family,
                OperationFamily::AttentionBlock | OperationFamily::MlpBlock
            )
        {
            return BACKEND_ANE;
        }

        match family {
            DispatchFamily::MatMul => {
                // MatMul is GPU-bound — prefer MLX unless throttling.
                if state.should_throttle() {
                    BACKEND_ANE // Core ML (ANE — most efficient per watt)
                } else {
                    BACKEND_MLX // MLX (GPU — fastest)
                }
            }
            DispatchFamily::Attention => {
                // Attention is memory-bandwidth-bound — offload to ANE
                // when the GPU is saturated, use CPU when throttling,
                // GPU otherwise.
                if state.gpu_saturated() {
                    BACKEND_ANE // Core ML (ANE)
                } else if state.should_throttle() {
                    BACKEND_ACCELERATE // Accelerate (CPU — most power efficient)
                } else {
                    BACKEND_MLX // MLX (GPU)
                }
            }
            DispatchFamily::ElementWise => {
                // Element-wise ops are cheap everywhere — use whichever
                // backend does not compete with the GPU.
                if state.gpu_saturated() || state.gpu_utilization > 0.5 {
                    BACKEND_ACCELERATE // Accelerate (CPU — doesn't compete)
                } else {
                    BACKEND_MLX // MLX (GPU — fast and available)
                }
            }
            DispatchFamily::Softmax | DispatchFamily::LayerNorm => {
                // These run fine on any backend; prefer CPU to keep GPU free.
                BACKEND_ACCELERATE // Accelerate (CPU NEON)
            }
        }
    }

    /// Update a [`HeterogeneousExecutor`]'s per-operation routing table
    /// based on the current system state.
    ///
    /// Iterates every operation in the executor's registry, calls
    /// [`dispatch`](Self::dispatch) for each one, and writes the result
    /// into `executor.routing_table`.
    ///
    /// This allows the executor to use the flex-dispatch routes during the
    /// next [`execute_boundaries`] call without sampling the system on
    /// every single operation.
    pub fn reroute(&mut self, executor: &mut HeterogeneousExecutor) -> Result<(), String> {
        // Force a fresh sample so all routes are based on the same state.
        self.sample_now();

        let state = &self.last_state;
        // Collect operation IDs first to avoid conflicting borrows on executor
        let op_ids: Vec<_> = executor.operation_registry.keys().copied().collect();

        for op_id in op_ids {
            let op_desc = &executor.operation_registry[&op_id];
            let family = classify_family(op_desc.family);

            // When GPU is saturated but the system shouldn't throttle,
            // offload attention and MLP blocks to ANE.
            if state.should_offload_to_ane()
                && matches!(
                    op_desc.family,
                    OperationFamily::AttentionBlock | OperationFamily::MlpBlock
                )
            {
                executor.set_route(op_id, BACKEND_ANE);
                continue;
            }

            let backend_id = match family {
                DispatchFamily::MatMul => {
                    if state.should_throttle() {
                        BACKEND_ANE
                    } else {
                        BACKEND_MLX
                    }
                }
                DispatchFamily::Attention => {
                    if state.gpu_saturated() {
                        BACKEND_ANE
                    } else if state.should_throttle() {
                        BACKEND_ACCELERATE
                    } else {
                        BACKEND_MLX
                    }
                }
                DispatchFamily::ElementWise => {
                    if state.gpu_saturated() || state.gpu_utilization > 0.5 {
                        BACKEND_ACCELERATE
                    } else {
                        BACKEND_MLX
                    }
                }
                DispatchFamily::Softmax | DispatchFamily::LayerNorm => BACKEND_ACCELERATE,
            };

            executor.set_route(op_id, backend_id);
        }

        Ok(())
    }

    /// Install an outlier detector for precision-override decisions.
    pub fn set_outlier_detector(&mut self, detector: Arc<Mutex<OutlierDetector>>) {
        self.outlier_detector = Some(detector);
    }

    /// Check whether a matrix has active precision overrides.
    ///
    /// Returns `Some("bf16")` if any channel for this matrix is flagged
    /// as an outlier, or `None` if no override is active.
    pub fn check_precision_override(&self, matrix: &str) -> Option<&str> {
        let detector = self.outlier_detector.as_ref()?;
        let guard = detector.lock().ok()?;
        let has_override = guard.active_overrides().iter().any(|(id, _)| id == matrix);
        if has_override {
            Some("bf16")
        } else {
            None
        }
    }

    /// Overload: check ANE offload eligibility including precision overrides.
    ///
    /// If the matrix has a precision override, offload is suppressed so the
    /// high-precision path runs on GPU. Otherwise delegates to the system
    /// state check.
    pub fn should_offload_to_ane(&self, matrix: &str) -> bool {
        if self.check_precision_override(matrix).is_some() {
            return false;
        }
        self.last_state.should_offload_to_ane()
    }

    /// Observe activations at a PhaseType tap point.
    ///
    /// Only the `TapQkvProj`, `TapOProj`, `TapFfnGate`, `TapFfnUp`, and
    /// `TapFfnDown` variants are treated as observation points. Other phase
    /// types are silently ignored.
    pub fn observe_activations(&self, tap: PhaseType, activations: &[f32]) {
        let matrix_id = match tap {
            PhaseType::TapQkvProj => "tap.qkv_proj",
            PhaseType::TapOProj => "tap.o_proj",
            PhaseType::TapFfnGate => "tap.ffn_gate",
            PhaseType::TapFfnUp => "tap.ffn_up",
            PhaseType::TapFfnDown => "tap.ffn_down",
            _ => return,
        };
        let Some(detector) = &self.outlier_detector else {
            return;
        };
        let _ = detector
            .lock()
            .map(|mut guard| guard.observe(&matrix_id.to_string(), activations));
    }
}

impl Default for FlexDispatch {
    fn default() -> Self {
        Self::new()
    }
}
