//! AOT replay admission and the `replay_aot*` family.
//!
//! This module owns the canonical authority for validating the
//! compiler-emitted heterogeneous execution plan and for replaying it
//! through concrete backend hooks. The replay family composes the
//! smaller entities — [`super::super::binding::CImageBindingResolver`],
//! [`super::super::ane_backend::EmbeddedAneRouteBackend`],
//! [`super::super::kernel_dispatch::KernelRouteDispatcher`], and
//! [`super::super::xdna_dispatch::CImageXdnaRouteDispatcher`] — but
//! does not own their authority.
//!
//! All entry points call [`super::UnifiedRuntime::validate_aot_schedule`]
//! first; the validation is the runtime admission gate for dependency
//! order and streamed-model workload coverage.

use std::collections::HashMap;

use prism_amd_npu_runtime::{XdnaCommandSubmitter, XdnaExecutionPhase};
use prism_ecs_kernel::{AccelerateBackend, CpuBackend, MetalBackend};
use prism_spatial_ir::execution::HeterogeneousExecutionReceipt;
use prism_spatial_ir::execution_plan::{ExecutionPlan, InferencePhase, PlanBackend};
use prism_spatial_ir::{AotScheduler, HeterogeneousExecutor, RouteDispatch, RoutedExecutor, WorkloadScenario};

use super::super::ane_backend::EmbeddedAneRouteBackend;
use super::super::binding::CImageBindingResolver;
use super::super::kernel_dispatch::KernelRouteDispatcher;
use super::super::xdna_dispatch::CImageXdnaRouteDispatcher;
use super::super::RuntimeError;
use super::UnifiedRuntime;

impl UnifiedRuntime {
    pub(super) fn active_execution_plan(&self) -> Option<&ExecutionPlan> {
        match self.mode {
            super::ExecutionMode::Batch => self.model.execution_plan.as_ref(),
            super::ExecutionMode::RealtimePrefill | super::ExecutionMode::RealtimeDecode => self
                .model
                .realtime_execution_plan
                .as_ref()
                .or(self.model.execution_plan.as_ref()),
        }
    }

    /// Validate the AOT heterogeneous schedule before replay. This is the
    /// runtime admission gate for dependency order and streamed-model
    /// workload coverage.
    pub fn validate_aot_schedule(&self) -> Result<(), RuntimeError> {
        let plan = self.active_execution_plan().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
        })?;
        plan.validate().map_err(RuntimeError::InvalidCImage)?;
        for step in &plan.fused_steps {
            if step
                .depends_on
                .iter()
                .any(|dependency| *dependency >= step.step_id)
            {
                return Err(RuntimeError::InvalidCImage(format!(
                    "AOT schedule step {} has a forward dependency",
                    step.step_id
                )));
            }
        }
        if !plan.supports_all_streamed_workloads() {
            return Err(RuntimeError::UnsupportedMode(
                "AOT residency window does not support realtime text, batched text, and batched audio".into(),
            ));
        }
        for step in &plan.fused_steps {
            self.model.model_for_fused_step(step)?;
        }
        Ok(())
    }

    /// Replay the compiler-emitted heterogeneous plan through concrete
    /// backend hooks. The hook implementation owns buffer resolution and
    /// backend-specific dispatch; this method owns plan admission and receipt
    /// production.
    pub fn replay_aot<E: HeterogeneousExecutor>(
        &self,
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self.active_execution_plan().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
        })?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        AotScheduler::replay_resolved(plan, &mut resolver, executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay the plan specialized for one realtime or batch workload. The
    /// selected fusion strategy is attached to each dispatch step before the
    /// backend sees it.
    pub fn replay_aot_for_workload<E: HeterogeneousExecutor>(
        &self,
        scenario: WorkloadScenario,
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .try_specialize_for_workload(scenario)
            .map_err(RuntimeError::UnsupportedMode)?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        AotScheduler::replay_resolved(&plan, &mut resolver, executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay a phase-specialized plan using current backend queue telemetry.
    /// The plan migrates only dispatchable XDNA/Metal/CPU islands; fixed
    /// CPU-side attention and ANE routes remain unchanged.
    pub fn replay_aot_for_phase<E: HeterogeneousExecutor>(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .specialize_for_phase(phase, queue_depths);
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        AotScheduler::replay_resolved(&plan, &mut resolver, executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay an AOT plan through the explicit ANE/Metal/Accelerate/CPU route
    /// table. This is the preferred integration point for production runtime
    /// backends because route labels cannot be silently collapsed into one
    /// generic dispatch method.
    pub fn replay_aot_routed<R: RouteDispatch>(
        &self,
        routes: R,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self.active_execution_plan().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
        })?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        let mut executor = RoutedExecutor { routes };
        AotScheduler::replay_resolved(plan, &mut resolver, &mut executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Routed counterpart to [`Self::replay_aot_for_workload`].
    pub fn replay_aot_routed_for_workload<R: RouteDispatch>(
        &self,
        scenario: WorkloadScenario,
        routes: R,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .try_specialize_for_workload(scenario)
            .map_err(RuntimeError::UnsupportedMode)?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        let mut executor = RoutedExecutor { routes };
        AotScheduler::replay_resolved(&plan, &mut resolver, &mut executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Routed phase-aware replay using live queue telemetry.
    pub fn replay_aot_routed_for_phase<R: RouteDispatch>(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
        routes: R,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .specialize_for_phase(phase, queue_depths);
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        let mut executor = RoutedExecutor { routes };
        AotScheduler::replay_resolved(&plan, &mut resolver, &mut executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay the compiler-emitted plan through Prism's assembled Apple
    /// route table. This is the production convenience entry point: ANE
    /// programs use the embedded Core ML/IOSurface adapter, while Metal,
    /// Accelerate, and CPU use the shared kernel backend contract.
    pub fn replay_aot_apple(&self) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        let mut ane = EmbeddedAneRouteBackend {
            runtime: self,
            outputs: HashMap::new(),
        };
        let accelerate = AccelerateBackend;
        let metal = MetalBackend::new();
        let cpu = CpuBackend;
        let routes = KernelRouteDispatcher {
            model: &self.model,
            ane: &mut ane,
            accelerate: &accelerate,
            metal: &metal,
            cpu: &cpu,
            xdna: None,
        };
        self.replay_aot_routed(routes)
    }

    /// Replay the assembled Apple routes using the strategy selected for a
    /// concrete realtime or batch workload scenario.
    pub fn replay_aot_apple_for_workload(
        &self,
        scenario: WorkloadScenario,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        let mut ane = EmbeddedAneRouteBackend {
            runtime: self,
            outputs: HashMap::new(),
        };
        let accelerate = AccelerateBackend;
        let metal = MetalBackend::new();
        let cpu = CpuBackend;
        let routes = KernelRouteDispatcher {
            model: &self.model,
            ane: &mut ane,
            accelerate: &accelerate,
            metal: &metal,
            cpu: &cpu,
            xdna: None,
        };
        self.replay_aot_routed_for_workload(scenario, routes)
    }

    /// Replay the compiler-emitted plan with a native XDNA island included in
    /// the same route table as the Apple/CPU backends. This is the public
    /// heterogeneous entry point for deployments that provide an
    /// `XdnaDevice` implementation.
    pub fn replay_aot_with_xdna<D: XdnaCommandSubmitter>(
        &self,
        device: D,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        let mut ane = EmbeddedAneRouteBackend {
            runtime: self,
            outputs: HashMap::new(),
        };
        let accelerate = AccelerateBackend;
        let metal = MetalBackend::new();
        let cpu = CpuBackend;
        let mut xdna = CImageXdnaRouteDispatcher::new(&self.model, device)
            .map_err(RuntimeError::InvalidCImage)?;
        if matches!(self.mode, super::ExecutionMode::RealtimePrefill) {
            xdna.set_phase(XdnaExecutionPhase::Prefill { tokens: 1 });
        }
        let routes = KernelRouteDispatcher {
            model: &self.model,
            ane: &mut ane,
            accelerate: &accelerate,
            metal: &metal,
            cpu: &cpu,
            xdna: Some(&mut xdna),
        };
        self.replay_aot_routed(routes)
    }
}
