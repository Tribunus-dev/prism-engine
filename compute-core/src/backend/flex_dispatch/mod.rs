//! FlexDispatch — runtime backend scheduler.
//!
//! Dynamically assigns each operation to MLX (GPU), Core ML (ANE), or
//! Accelerate (CPU/NEON) based on real-time system state sampled from
//! IOKit, Mach, and iOS/IOKit power-source APIs.  No compile-time static
//! routing — the dispatcher adapts to actual running conditions.
//!
//! # Design
//!
//! Every `N` decode steps the controller samples GPU utilization, CPU
//! load, thermal state and battery.  Each operation is classified into one
//! of five families — `MatMul`, `Attention`, `ElementWise`, `Softmax`,
//! `LayerNorm` — and routed to the most appropriate backend for the
//! *current* system state.
//!
//! - **MatMul** (GPU-bound) → MLX, unless thermal/battery throttling
//!   demands the more efficient ANE path.
//! - **Attention** (memory-bandwidth-bound) → ANE when GPU is saturated,
//!   Accelerate when throttling, MLX otherwise.
//! - **ElementWise** (cheap) → Accelerate (CPU) when GPU is busy, MLX
//!   when it is free.
//! - **Softmax / LayerNorm** → Accelerate (NEON) to keep the GPU free for
//!   matmuls.

pub mod profiling;
pub mod selection;

pub use profiling::*;
pub use selection::*;

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::routing::*;
    use crate::backend::heterogeneous_executor::HeterogeneousExecutor;
    use crate::backend::DType;

    fn make_matmul_op(id: u64) -> OperationDescriptor {
        OperationDescriptor {
            operation_id: OperationId(id),
            family: OperationFamily::Matmul,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: vec![1, 64] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F32, DType::F32],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1, 64] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    fn make_attention_op(id: u64) -> OperationDescriptor {
        OperationDescriptor {
            operation_id: OperationId(id),
            family: OperationFamily::AttentionBlock,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: vec![1, 64] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F32],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1, 64] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    #[test]
    fn test_system_state_default() {
        let state = SystemState::default();
        assert_eq!(state.gpu_utilization, 0.0);
        assert_eq!(state.thermal_state, ThermalState::Nominal);
        assert_eq!(state.battery_remaining, 1.0);
        assert!(state.ac_power);
        assert!(!state.gpu_saturated());
        assert!(!state.should_throttle());
    }

    #[test]
    fn test_gpu_saturated_threshold() {
        let mut state = SystemState::default();
        // Below threshold
        state.gpu_utilization = 0.5;
        assert!(!state.gpu_saturated());

        // Above utilization threshold
        state.gpu_utilization = 0.9;
        assert!(state.gpu_saturated());

        // Memory threshold
        state.gpu_utilization = 0.5;
        state.gpu_memory_fraction = 0.95;
        assert!(state.gpu_saturated());
    }

    #[test]
    fn test_should_throttle_thermal() {
        let mut state = SystemState::default();
        assert!(!state.should_throttle());

        state.thermal_state = ThermalState::Serious;
        assert!(state.should_throttle());

        state.thermal_state = ThermalState::Critical;
        assert!(state.should_throttle());

        state.thermal_state = ThermalState::Fair;
        assert!(!state.should_throttle());
    }

    #[test]
    fn test_should_throttle_battery() {
        let mut state = SystemState::default();
        state.ac_power = false;
        state.battery_remaining = 0.15;
        assert!(state.should_throttle());

        state.battery_remaining = 0.5;
        assert!(!state.should_throttle());
    }

    #[test]
    fn test_classify_family() {
        assert_eq!(
            classify_family(OperationFamily::Matmul),
            DispatchFamily::MatMul
        );
        assert_eq!(
            classify_family(OperationFamily::QuantizedMatmul),
            DispatchFamily::MatMul
        );
        assert_eq!(
            classify_family(OperationFamily::AttentionBlock),
            DispatchFamily::Attention
        );
        assert_eq!(
            classify_family(OperationFamily::Silu),
            DispatchFamily::ElementWise
        );
        assert_eq!(
            classify_family(OperationFamily::Softmax),
            DispatchFamily::Softmax
        );
        assert_eq!(
            classify_family(OperationFamily::RmsNorm),
            DispatchFamily::LayerNorm
        );
    }

    #[test]
    fn test_dispatch_matmul_normal() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState::default();
        flex.steps_since_sample = 0; // Skip sampling.

        let op = make_matmul_op(1);
        let backend = flex.dispatch(&op, 0);
        // Default state: GPU free, AC power, nominal temps → MLX (GPU).
        assert_eq!(backend, BackendId(0));
    }

    #[test]
    fn test_dispatch_matmul_throttle() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            thermal_state: ThermalState::Serious,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = make_matmul_op(1);
        let backend = flex.dispatch(&op, 0);
        // Throttling → ANE (Core ML).
        assert_eq!(backend, BackendId(2));
    }

    #[test]
    fn test_dispatch_attention_gpu_saturated() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            gpu_utilization: 0.9,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = make_attention_op(1);
        let backend = flex.dispatch(&op, 0);
        // GPU saturated → ANE.
        assert_eq!(backend, BackendId(2));
    }

    #[test]
    fn test_dispatch_attention_throttle() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            thermal_state: ThermalState::Critical,
            gpu_utilization: 0.3,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = make_attention_op(1);
        let backend = flex.dispatch(&op, 0);
        // Throttling (but GPU not saturated) → CPU (Accelerate).
        assert_eq!(backend, BackendId(1));
    }

    #[test]
    fn test_dispatch_elementwise_gpu_busy() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            gpu_utilization: 0.7,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = OperationDescriptor {
            family: OperationFamily::Silu,
            ..make_matmul_op(2)
        };
        let backend = flex.dispatch(&op, 0);
        // GPU utilization > 0.5 → CPU (Accelerate).
        assert_eq!(backend, BackendId(1));
    }

    #[test]
    fn test_dispatch_softmax_always_cpu() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = OperationDescriptor {
            family: OperationFamily::Softmax,
            ..make_matmul_op(3)
        };
        let backend = flex.dispatch(&op, 0);
        // Softmax always routes to CPU.
        assert_eq!(backend, BackendId(1));
    }

    #[test]
    fn test_sample_interval() {
        let mut flex = FlexDispatch::new();
        flex.sample_interval = 5;

        // Each dispatch call increments the counter. After interval, sampling
        // resets. We verify the internal state by checking the step counter.
        for i in 0..4 {
            let op = make_matmul_op(i);
            flex.dispatch(&op, i as u32);
            assert!(
                flex.steps_since_sample <= 5,
                "steps_since_sample should be <= 5 after {i} dispatches"
            );
        }
    }

    #[test]
    fn test_reroute_populates_routing_table() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState::default();

        let mut executor = HeterogeneousExecutor::new();

        // Populate the operation registry.
        let mut registry = std::collections::HashMap::new();
        registry.insert(OperationId(1), make_matmul_op(1));
        registry.insert(OperationId(2), make_attention_op(2));
        executor.set_operation_registry(registry);

        // Reroute based on current state.
        flex.reroute(&mut executor).unwrap();

        // With default state (GPU free, nominal temps, AC power):
        //   MatMul → MLX (BackendId(0))
        //   Attention → MLX (BackendId(0))
        let route1 = executor.get_route(&OperationId(1));
        let route2 = executor.get_route(&OperationId(2));
        assert_eq!(route1, Some(BackendId(0)));
        assert_eq!(route2, Some(BackendId(0)));
    }
}
