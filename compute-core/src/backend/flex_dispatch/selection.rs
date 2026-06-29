//! FlexDispatch — kernel and path selection logic.
//!
//! Routes each operation to MLX (GPU), Core ML (ANE), or Accelerate
//! (CPU/NEON) based on the real-time [`SystemState`] sampled by
//! [`profiling`](crate::backend::flex_dispatch::profiling).

use super::profiling::{SystemState, ThermalState};
use crate::backend::heterogeneous_executor::HeterogeneousExecutor;
use crate::backend::routing::*;

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
    }
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
}

impl FlexDispatch {
    /// Create a new dispatch controller with default sampling interval
    /// (16 decode steps).
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
            sample_interval: 16,
            steps_since_sample: u32::MAX, // Sample on first call.
        }
    }

    /// Create a dispatch controller with a custom sampling interval.
    pub fn with_interval(steps: u32) -> Self {
        Self {
            last_state: SystemState::default(),
            sample_interval: steps,
            steps_since_sample: u32::MAX,
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

        match family {
            DispatchFamily::MatMul => {
                // MatMul is GPU-bound — prefer MLX unless throttling.
                if state.should_throttle() {
                    BackendId(2) // Core ML (ANE — most efficient per watt)
                } else {
                    BackendId(0) // MLX (GPU — fastest)
                }
            }
            DispatchFamily::Attention => {
                // Attention is memory-bandwidth-bound — offload to ANE
                // when the GPU is saturated, use CPU when throttling,
                // GPU otherwise.
                if state.gpu_saturated() {
                    BackendId(2) // Core ML (ANE)
                } else if state.should_throttle() {
                    BackendId(1) // Accelerate (CPU — most power efficient)
                } else {
                    BackendId(0) // MLX (GPU)
                }
            }
            DispatchFamily::ElementWise => {
                // Element-wise ops are cheap everywhere — use whichever
                // backend does not compete with the GPU.
                if state.gpu_saturated() || state.gpu_utilization > 0.5 {
                    BackendId(1) // Accelerate (CPU — doesn't compete)
                } else {
                    BackendId(0) // MLX (GPU — fast and available)
                }
            }
            DispatchFamily::Softmax | DispatchFamily::LayerNorm => {
                // These run fine on any backend; prefer CPU to keep GPU free.
                BackendId(1) // Accelerate (CPU NEON)
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
            let backend_id = match family {
                DispatchFamily::MatMul => {
                    if state.should_throttle() {
                        BackendId(2)
                    } else {
                        BackendId(0)
                    }
                }
                DispatchFamily::Attention => {
                    if state.gpu_saturated() {
                        BackendId(2)
                    } else if state.should_throttle() {
                        BackendId(1)
                    } else {
                        BackendId(0)
                    }
                }
                DispatchFamily::ElementWise => {
                    if state.gpu_saturated() || state.gpu_utilization > 0.5 {
                        BackendId(1)
                    } else {
                        BackendId(0)
                    }
                }
                DispatchFamily::Softmax | DispatchFamily::LayerNorm => BackendId(1),
            };

            executor.set_route(op_id, backend_id);
        }

        Ok(())
    }
}

impl Default for FlexDispatch {
    fn default() -> Self {
        Self::new()
    }
}
