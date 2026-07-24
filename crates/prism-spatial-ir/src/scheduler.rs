//! Dependency-aware replay of an AOT heterogeneous execution plan.

use crate::execution::{BackendId, HeterogeneousExecutionReceipt, StepExecutionEvidence};
use crate::execution_plan::{ExecutionPlan, FusedScheduleStep, PlanBackend};
use std::time::Instant;

pub trait HeterogeneousExecutor {
    fn ensure_residency(&mut self, window_id: usize) -> Result<(), String>;
    fn dispatch(&mut self, backend: PlanBackend, step: &FusedScheduleStep) -> Result<(), String>;
    /// Dispatch a step after its CImage/IOSurface bindings have been resolved.
    /// Implementors that need the actual buffers override this method; the
    /// default preserves compatibility with route-only executors.
    fn dispatch_resolved(
        &mut self,
        backend: PlanBackend,
        resolved: &mut ResolvedStep<'_>,
    ) -> Result<(), String> {
        self.dispatch(backend, resolved.step)
    }
    fn synchronize(&mut self, step: &FusedScheduleStep) -> Result<(), String>;
}

/// Explicit hardware route contract for production AOT dispatch.
///
/// Keeping these entry points distinct prevents an executor from accepting
/// ANE/Metal/Accelerate labels while silently routing every step through one
/// generic backend.
pub trait RouteDispatch {
    fn ensure_residency(&mut self, window_id: usize) -> Result<(), String>;
    fn dispatch_ane_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_ane_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_accelerate(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_metal(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_cpu(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_xdna(
        &mut self,
        _step: &FusedScheduleStep,
        _inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("XDNA route is not implemented by this executor".into())
    }
    fn synchronize(&mut self, step: &FusedScheduleStep) -> Result<(), String>;
}

/// Adapter that turns the explicit route contract into the scheduler's
/// dependency/residency-aware executor interface.
pub struct RoutedExecutor<R> {
    pub routes: R,
}

impl<R: RouteDispatch> HeterogeneousExecutor for RoutedExecutor<R> {
    fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
        self.routes.ensure_residency(window_id)
    }

    fn dispatch(&mut self, _backend: PlanBackend, _step: &FusedScheduleStep) -> Result<(), String> {
        Err("route dispatch requires resolved bindings".into())
    }

    fn dispatch_resolved(
        &mut self,
        backend: PlanBackend,
        resolved: &mut ResolvedStep<'_>,
    ) -> Result<(), String> {
        match backend {
            PlanBackend::AnePlanar => self.routes.dispatch_ane_planar(
                resolved.step,
                &resolved.inputs,
                &mut resolved.outputs,
            ),
            PlanBackend::AneMatrix => self.routes.dispatch_ane_matrix(
                resolved.step,
                &resolved.inputs,
                &mut resolved.outputs,
            ),
            PlanBackend::Accelerate => self.routes.dispatch_accelerate(
                resolved.step,
                &resolved.inputs,
                &mut resolved.outputs,
            ),
            PlanBackend::Metal => {
                self.routes
                    .dispatch_metal(resolved.step, &resolved.inputs, &mut resolved.outputs)
            }
            PlanBackend::Cpu => {
                self.routes
                    .dispatch_cpu(resolved.step, &resolved.inputs, &mut resolved.outputs)
            }
            PlanBackend::Xdna => {
                self.routes
                    .dispatch_xdna(resolved.step, &resolved.inputs, &mut resolved.outputs)
            }
        }
    }

    fn synchronize(&mut self, step: &FusedScheduleStep) -> Result<(), String> {
        self.routes.synchronize(step)
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedBuffer {
    pub name: String,
    pub element_type: String,
    pub region: String,
    pub byte_length: usize,
    pub zero_copy: bool,
    pub file_offset: Option<u64>,
    pub storage: BufferStorage,
    pub shape: Vec<u64>,
    /// Optional materialized payload for runtime-owned/intermediate buffers.
    /// Mapped CImage buffers leave this unset and use `file_offset` instead.
    pub payload: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferStorage {
    MappedCImage,
    RuntimeOwned,
}

pub trait BindingResolver {
    fn resolve_inputs(&mut self, step: &FusedScheduleStep) -> Result<Vec<ResolvedBuffer>, String>;
    fn resolve_outputs(&mut self, step: &FusedScheduleStep) -> Result<Vec<ResolvedBuffer>, String>;
    fn commit_outputs(
        &mut self,
        _step: &FusedScheduleStep,
        _outputs: &[ResolvedBuffer],
    ) -> Result<(), String> {
        Ok(())
    }
}

pub struct ResolvedStep<'a> {
    pub step: &'a FusedScheduleStep,
    pub inputs: Vec<ResolvedBuffer>,
    pub outputs: Vec<ResolvedBuffer>,
}

pub struct AotScheduler;

impl AotScheduler {
    pub fn replay<E: HeterogeneousExecutor>(
        plan: &ExecutionPlan,
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, String> {
        plan.validate()?;
        let started = Instant::now();
        let mut evidence = Vec::with_capacity(plan.fused_steps.len());
        for step in &plan.fused_steps {
            let step_started = Instant::now();
            let step_offset_ns = started.elapsed().as_nanos() as u64;
            let window_id = plan
                .residency_windows
                .iter()
                .find(|window| {
                    window
                        .prefetch_step
                        .is_some_and(|prefetch| prefetch <= step.step_id)
                        && window
                            .eviction_step
                            .is_none_or(|evict| step.step_id < evict)
                })
                .map(|window| window.window_id)
                .unwrap_or(0);
            executor.ensure_residency(window_id)?;
            executor.dispatch(step.backend, step)?;
            executor.synchronize(step)?;
            evidence.push(StepExecutionEvidence {
                step_id: step.step_id,
                backend: backend_id(step.backend),
                started_ns: step_offset_ns,
                elapsed_ns: step_started.elapsed().as_nanos() as u64,
                input_region: step.input_region.clone(),
                output_region: step.output_region.clone(),
                zero_copy: step.zero_copy,
                residency_window: window_id,
                fusion_strategy: step.fusion_strategy.clone(),
            });
        }
        Ok(HeterogeneousExecutionReceipt {
            plan_id: format!("aot-{}", started.elapsed().as_nanos()),
            steps: evidence,
            model_residency_windows: plan.residency_windows.len(),
            total_elapsed_ns: started.elapsed().as_nanos() as u64,
        })
    }

    pub fn replay_resolved<E: HeterogeneousExecutor, R: BindingResolver>(
        plan: &ExecutionPlan,
        resolver: &mut R,
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, String> {
        plan.validate()?;
        let started = Instant::now();
        let mut evidence = Vec::with_capacity(plan.fused_steps.len());
        for step in &plan.fused_steps {
            let resolved = Self::resolve_step(resolver, step)?;
            if resolved.inputs.len() != step.input_tensors.len()
                || resolved.outputs.len() != step.output_tensors.len()
            {
                return Err(format!(
                    "step {} binding count changed during resolution",
                    step.step_id
                ));
            }
            let mut resolved = resolved;
            let step = resolved.step;
            let step_started = Instant::now();
            let window_id = plan
                .residency_windows
                .iter()
                .find(|window| {
                    window
                        .prefetch_step
                        .is_some_and(|prefetch| prefetch <= step.step_id)
                        && window
                            .eviction_step
                            .is_none_or(|evict| step.step_id < evict)
                })
                .map(|window| window.window_id)
                .unwrap_or(0);
            executor.ensure_residency(window_id)?;
            executor.dispatch_resolved(step.backend, &mut resolved)?;
            executor.synchronize(step)?;
            resolver.commit_outputs(step, &resolved.outputs)?;
            evidence.push(StepExecutionEvidence {
                step_id: step.step_id,
                backend: backend_id(step.backend),
                started_ns: started.elapsed().as_nanos() as u64,
                elapsed_ns: step_started.elapsed().as_nanos() as u64,
                input_region: step.input_region.clone(),
                output_region: step.output_region.clone(),
                zero_copy: step.zero_copy,
                residency_window: window_id,
                fusion_strategy: step.fusion_strategy.clone(),
            });
        }
        Ok(HeterogeneousExecutionReceipt {
            plan_id: format!("aot-{}", started.elapsed().as_nanos()),
            steps: evidence,
            model_residency_windows: plan.residency_windows.len(),
            total_elapsed_ns: started.elapsed().as_nanos() as u64,
        })
    }

    pub fn resolve_step<'a, R: BindingResolver>(
        resolver: &mut R,
        step: &'a FusedScheduleStep,
    ) -> Result<ResolvedStep<'a>, String> {
        let inputs = resolver.resolve_inputs(step)?;
        let outputs = resolver.resolve_outputs(step)?;
        if step.zero_copy
            && inputs.iter().any(|buffer| !buffer.zero_copy)
            && outputs.iter().any(|buffer| !buffer.zero_copy)
        {
            return Err(format!(
                "step {} violates its zero-copy contract",
                step.step_id
            ));
        }
        Ok(ResolvedStep {
            step,
            inputs,
            outputs,
        })
    }
}

fn backend_id(backend: PlanBackend) -> BackendId {
    match backend {
        PlanBackend::AnePlanar | PlanBackend::AneMatrix => BackendId::Ane,
        PlanBackend::Accelerate => BackendId::Accelerate,
        PlanBackend::Metal => BackendId::Metal,
        PlanBackend::Cpu => BackendId::Cpu,
        PlanBackend::Xdna => BackendId::Xdna,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::{
        ExecutionMode, FusedScheduleStep, PlanBackend, ResidencyWindow, ResidencyWorkload,
    };
    use crate::fused_ops::FusionStrategy;

    struct RecordingExecutor {
        events: Vec<String>,
    }

    impl HeterogeneousExecutor for RecordingExecutor {
        fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
            self.events.push(format!("resident:{window_id}"));
            Ok(())
        }

        fn dispatch(
            &mut self,
            backend: PlanBackend,
            step: &FusedScheduleStep,
        ) -> Result<(), String> {
            self.events
                .push(format!("dispatch:{backend:?}:{}", step.step_id));
            Ok(())
        }

        fn synchronize(&mut self, step: &FusedScheduleStep) -> Result<(), String> {
            self.events.push(format!("sync:{}", step.step_id));
            Ok(())
        }
    }

    #[test]
    fn replays_interleaved_schedule_with_residency_and_receipt() {
        let mut plan = ExecutionPlan::new(ExecutionMode::Batch, vec![], 1, false);
        plan.fused_steps = vec![
            FusedScheduleStep {
                step_id: 0,
                model_id: None,
                node_ids: vec![],
                backend: PlanBackend::AnePlanar,
                depends_on: vec![],
                input_region: "ane-memory".into(),
                output_region: "ane-memory".into(),
                zero_copy: true,
                estimated_latency_ns: 10,
                input_tensors: vec![],
                output_tensors: vec![],
                dispatch_geometry: [1, 1, 1],
                fusion_strategy: Some(FusionStrategy::PersistentMegakernel {
                    search_generation: 7,
                }),
            },
            FusedScheduleStep {
                step_id: 1,
                model_id: None,
                node_ids: vec![],
                backend: PlanBackend::Metal,
                depends_on: vec![0],
                input_region: "unified-memory".into(),
                output_region: "unified-memory".into(),
                zero_copy: true,
                estimated_latency_ns: 20,
                input_tensors: vec![],
                output_tensors: vec![],
                dispatch_geometry: [8, 1, 1],
                fusion_strategy: None,
            },
        ];
        plan.residency_windows = vec![ResidencyWindow {
            window_id: 0,
            model_bytes: 1024,
            required_workloads: vec![
                ResidencyWorkload::RealtimeText,
                ResidencyWorkload::BatchedText,
                ResidencyWorkload::BatchedAudio,
            ],
            resident_devices: vec!["unified-memory".into()],
            prefetch_step: Some(0),
            eviction_step: Some(2),
        }];
        let mut executor = RecordingExecutor { events: vec![] };
        let receipt = AotScheduler::replay(&plan, &mut executor).unwrap();
        assert_eq!(receipt.steps.len(), 2);
        assert_eq!(receipt.model_residency_windows, 1);
        assert!(matches!(
            receipt.steps[0].fusion_strategy,
            Some(FusionStrategy::PersistentMegakernel {
                search_generation: 7
            })
        ));
        assert_eq!(
            executor.events,
            vec![
                "resident:0",
                "dispatch:AnePlanar:0",
                "sync:0",
                "resident:0",
                "dispatch:Metal:1",
                "sync:1",
            ]
        );
    }

    struct TestResolver;
    impl BindingResolver for TestResolver {
        fn resolve_inputs(
            &mut self,
            step: &FusedScheduleStep,
        ) -> Result<Vec<ResolvedBuffer>, String> {
            Ok(step
                .input_tensors
                .iter()
                .map(|binding| ResolvedBuffer {
                    name: binding.name.clone(),
                    element_type: binding.element_type.clone(),
                    region: step.input_region.clone(),
                    byte_length: 16,
                    zero_copy: step.zero_copy,
                    file_offset: None,
                    storage: BufferStorage::RuntimeOwned,
                    shape: vec![],
                    payload: None,
                })
                .collect())
        }
        fn resolve_outputs(
            &mut self,
            step: &FusedScheduleStep,
        ) -> Result<Vec<ResolvedBuffer>, String> {
            Ok(step
                .output_tensors
                .iter()
                .map(|binding| ResolvedBuffer {
                    name: binding.name.clone(),
                    element_type: binding.element_type.clone(),
                    region: step.output_region.clone(),
                    byte_length: 16,
                    zero_copy: step.zero_copy,
                    file_offset: None,
                    storage: BufferStorage::RuntimeOwned,
                    shape: vec![],
                    payload: None,
                })
                .collect())
        }
    }

    #[test]
    fn resolves_zero_copy_bindings_before_dispatch() {
        let step = FusedScheduleStep {
            step_id: 0,
            model_id: None,
            node_ids: vec![],
            backend: PlanBackend::AneMatrix,
            depends_on: vec![],
            input_region: "ane-memory".into(),
            output_region: "ane-memory".into(),
            zero_copy: true,
            estimated_latency_ns: 1,
            input_tensors: vec![],
            output_tensors: vec![],
            dispatch_geometry: [1, 1, 1],
            fusion_strategy: None,
        };
        let resolved = AotScheduler::resolve_step(&mut TestResolver, &step).unwrap();
        assert!(resolved.inputs.is_empty());
        assert!(resolved.outputs.is_empty());
    }

    #[test]
    fn routed_executor_keeps_all_hardware_routes_distinct() {
        struct Routes(Vec<PlanBackend>);
        impl RouteDispatch for Routes {
            fn ensure_residency(&mut self, _window_id: usize) -> Result<(), String> {
                Ok(())
            }
            fn dispatch_ane_planar(
                &mut self,
                _step: &FusedScheduleStep,
                _inputs: &[ResolvedBuffer],
                _outputs: &mut [ResolvedBuffer],
            ) -> Result<(), String> {
                self.0.push(PlanBackend::AnePlanar);
                Ok(())
            }
            fn dispatch_ane_matrix(
                &mut self,
                _step: &FusedScheduleStep,
                _inputs: &[ResolvedBuffer],
                _outputs: &mut [ResolvedBuffer],
            ) -> Result<(), String> {
                self.0.push(PlanBackend::AneMatrix);
                Ok(())
            }
            fn dispatch_accelerate(
                &mut self,
                _step: &FusedScheduleStep,
                _inputs: &[ResolvedBuffer],
                _outputs: &mut [ResolvedBuffer],
            ) -> Result<(), String> {
                self.0.push(PlanBackend::Accelerate);
                Ok(())
            }
            fn dispatch_metal(
                &mut self,
                _step: &FusedScheduleStep,
                _inputs: &[ResolvedBuffer],
                _outputs: &mut [ResolvedBuffer],
            ) -> Result<(), String> {
                self.0.push(PlanBackend::Metal);
                Ok(())
            }
            fn dispatch_cpu(
                &mut self,
                _step: &FusedScheduleStep,
                _inputs: &[ResolvedBuffer],
                _outputs: &mut [ResolvedBuffer],
            ) -> Result<(), String> {
                self.0.push(PlanBackend::Cpu);
                Ok(())
            }
            fn synchronize(&mut self, _step: &FusedScheduleStep) -> Result<(), String> {
                Ok(())
            }
        }

        let step = FusedScheduleStep {
            step_id: 0,
            model_id: None,
            node_ids: vec![],
            backend: PlanBackend::Cpu,
            depends_on: vec![],
            input_region: "unified-memory".into(),
            output_region: "unified-memory".into(),
            zero_copy: false,
            estimated_latency_ns: 1,
            input_tensors: vec![],
            output_tensors: vec![],
            dispatch_geometry: [1, 1, 1],
            fusion_strategy: None,
        };
        let mut executor = RoutedExecutor {
            routes: Routes(vec![]),
        };
        for backend in [
            PlanBackend::AnePlanar,
            PlanBackend::AneMatrix,
            PlanBackend::Accelerate,
            PlanBackend::Metal,
            PlanBackend::Cpu,
        ] {
            let routed_step = FusedScheduleStep {
                backend,
                ..step.clone()
            };
            let mut resolved = ResolvedStep {
                step: &routed_step,
                inputs: vec![],
                outputs: vec![],
            };
            executor.dispatch_resolved(backend, &mut resolved).unwrap();
        }
        assert_eq!(
            executor.routes.0,
            vec![
                PlanBackend::AnePlanar,
                PlanBackend::AneMatrix,
                PlanBackend::Accelerate,
                PlanBackend::Metal,
                PlanBackend::Cpu,
            ]
        );
    }
}
