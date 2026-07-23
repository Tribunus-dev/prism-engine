//! Execution plan types for mode-specific lowering.
//!
//! [`ExecutionMode`] and [`ExecutionPlan`] carry batch vs realtime scheduling
//! intent through the lowering pipeline. The [`lower_to_manifest`] function
//! produces a [`target::KernelManifest`] populated with both mode-specific plans.
//!
//! Batch mode targets multi-token throughput with GEMM-oriented dispatch;
//! realtime mode targets single-token latency with GEMV-oriented dispatch and
//! KV-cache persistence.

use crate::cost::{CodecVariant, CostEstimate};
use crate::fused_ops::{
    FusionStrategyCandidate, FusionStrategyEvaluation, WorkloadScenario, WorkloadStrategyEvaluation,
};
use crate::graph::{
    ComputeIntensity, ComputeKind, SpatialGraph, SpatialNode, SpatialNodeId, TileGeometry,
};
use crate::target::{KernelDescriptor, KernelManifest};
use prism_ecs_ir::evolution::compile_plan::FormatPlan;
use prism_ecs_ir::evolution::foundation::AneUnitAxis;
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ExecutionMode
// ---------------------------------------------------------------------------

/// Execution mode for a lowered schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Batch execution: process multiple tokens, GEMM-oriented.
    Batch,
    /// Realtime autoregressive decode: single token, GEMV-oriented.
    Realtime,
}

impl ExecutionMode {
    /// Returns `true` for batch mode.
    pub fn is_batch(&self) -> bool {
        matches!(self, Self::Batch)
    }

    /// Returns `true` for realtime mode.
    pub fn is_realtime(&self) -> bool {
        matches!(self, Self::Realtime)
    }

    /// Returns the default batch size for this mode.
    pub fn default_batch_size(&self) -> u32 {
        match self {
            Self::Batch => 32,
            Self::Realtime => 1,
        }
    }

    /// Returns whether KV-cache should be persisted across tokens in this mode.
    pub fn default_persistent_cache(&self) -> bool {
        match self {
            Self::Batch => false,
            Self::Realtime => true,
        }
    }
}

// ---------------------------------------------------------------------------
// ScheduleEntry
// ---------------------------------------------------------------------------

/// A single entry in an execution plan's schedule.
///
/// Each entry references a kernel descriptor by index into the manifest's
/// kernel list and supplies the mode-specific dispatch geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleEntry {
    /// Index into the manifest's `kernels` vector.
    pub kernel_index: usize,
    /// Threadgroup width for this mode-specific invocation.
    pub threadgroup_width: u32,
    /// Threadgroup height for this mode-specific invocation.
    pub threadgroup_height: u32,
}

/// Backend selected by the AOT heterogeneous scheduler for one fused island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlanBackend {
    AnePlanar,
    AneMatrix,
    Accelerate,
    Metal,
    Cpu,
    Xdna,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferencePhase {
    Prefill,
    Decode,
}

/// Workload policy for heterogeneous APUs. Prefill is NPU-first on systems
/// that expose XDNA; decode remains latency-sensitive and may select the
/// least-loaded eligible device at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousDispatchPolicy {
    pub prefill: PlanBackend,
    pub decode_candidates: Vec<PlanBackend>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcieBoundaryPayload {
    Activation,
    WorkDescriptor,
    StreamedWeight,
    ResidentWeight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIslandPolicy {
    pub isolated: bool,
    pub allow_streamed_weights: bool,
}

impl Default for DeviceIslandPolicy {
    fn default() -> Self {
        Self {
            isolated: true,
            allow_streamed_weights: false,
        }
    }
}

impl DeviceIslandPolicy {
    pub fn validate_boundary(&self, payload: PcieBoundaryPayload) -> Result<(), String> {
        if !self.isolated {
            return Ok(());
        }
        match payload {
            PcieBoundaryPayload::Activation | PcieBoundaryPayload::WorkDescriptor => Ok(()),
            PcieBoundaryPayload::StreamedWeight if self.allow_streamed_weights => Ok(()),
            PcieBoundaryPayload::StreamedWeight => {
                Err("isolated device island forbids streamed weights by policy".into())
            }
            PcieBoundaryPayload::ResidentWeight => {
                Err("resident weights cannot cross an isolated PCIe boundary".into())
            }
        }
    }
}

impl Default for HeterogeneousDispatchPolicy {
    fn default() -> Self {
        Self {
            prefill: PlanBackend::Xdna,
            decode_candidates: vec![PlanBackend::Xdna, PlanBackend::Metal, PlanBackend::Cpu],
        }
    }
}

impl HeterogeneousDispatchPolicy {
    pub fn backend_for(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
    ) -> PlanBackend {
        if matches!(phase, InferencePhase::Prefill) {
            return self.prefill;
        }
        let selected = self
            .decode_candidates
            .iter()
            .copied()
            .min_by_key(|backend| {
                queue_depths
                    .iter()
                    .find(|(candidate, _)| candidate == backend)
                    .map(|(_, depth)| *depth)
                    .unwrap_or(u32::MAX)
            });
        if queue_depths.is_empty()
            || selected.is_none_or(|backend| {
                !queue_depths
                    .iter()
                    .any(|(candidate, _)| *candidate == backend)
            })
        {
            return self
                .decode_candidates
                .iter()
                .copied()
                .find(|backend| *backend == PlanBackend::Cpu)
                .or(selected)
                .unwrap_or(PlanBackend::Cpu);
        }
        selected.unwrap_or(PlanBackend::Cpu)
    }
}

/// Workloads that every streamed model residency window must support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyWorkload {
    RealtimeText,
    BatchedText,
    BatchedAudio,
}

/// A model-memory residency window. Streaming may move the model between
/// devices, but each window is valid only if the whole model remains usable
/// for all declared inference workloads before eviction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyWindow {
    pub window_id: usize,
    pub model_bytes: u64,
    pub required_workloads: Vec<ResidencyWorkload>,
    pub resident_devices: Vec<String>,
    pub prefetch_step: Option<usize>,
    pub eviction_step: Option<usize>,
}

/// A dependency-aware fused execution island. The scheduler keeps these
/// records separate from legacy kernel indices so runtime dispatch can
/// interleave ANE, Accelerate, Metal, and CPU work without guessing routes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedScheduleStep {
    pub step_id: usize,
    /// Namespaced model owner for mixed-model CImages.
    #[serde(default)]
    pub model_id: Option<String>,
    pub node_ids: Vec<SpatialNodeId>,
    pub backend: PlanBackend,
    pub depends_on: Vec<usize>,
    pub input_region: String,
    pub output_region: String,
    pub zero_copy: bool,
    pub estimated_latency_ns: u64,
    pub input_tensors: Vec<TensorBinding>,
    pub output_tensors: Vec<TensorBinding>,
    pub dispatch_geometry: [u32; 3],
    /// Strategy selected for the active workload, when one was calibrated.
    #[serde(default)]
    pub fusion_strategy: Option<crate::fused_ops::FusionStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBinding {
    pub name: String,
    pub role: String,
    pub element_type: String,
    #[serde(default)]
    pub shape: Vec<u64>,
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// A mode-specific execution schedule over the graph.
///
/// Each plan contains an ordered list of schedule entries (in topological
/// or dispatch order), the expected batch size for the mode, and whether
/// the KV-cache is persisted across tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Execution mode this plan targets.
    pub mode: ExecutionMode,
    /// Schedule entries in execution order for this mode.
    pub schedule: Vec<ScheduleEntry>,
    /// Expected batch size (1 for realtime).
    pub batch_size: u32,
    /// Whether KV-cache is persisted across tokens.
    pub persistent_cache: bool,
    #[serde(default)]
    pub dispatch_policy: HeterogeneousDispatchPolicy,
    #[serde(default)]
    pub device_island: DeviceIslandPolicy,
    #[serde(default)]
    pub fused_steps: Vec<FusedScheduleStep>,
    #[serde(default)]
    pub residency_windows: Vec<ResidencyWindow>,
    /// Comparative evidence for standard, interleaved, and per-operation
    /// fusion strategies considered for this plan.
    #[serde(default)]
    pub fusion_evaluations: Vec<FusionStrategyEvaluation>,
    /// Scenario-specific winners used for dynamic workload dispatch.
    #[serde(default)]
    pub workload_evaluations: Vec<WorkloadStrategyEvaluation>,
}

impl ExecutionPlan {
    /// Create a new execution plan for the given mode.
    pub fn new(
        mode: ExecutionMode,
        schedule: Vec<ScheduleEntry>,
        batch_size: u32,
        persistent_cache: bool,
    ) -> Self {
        Self {
            mode,
            schedule,
            batch_size,
            persistent_cache,
            dispatch_policy: HeterogeneousDispatchPolicy::default(),
            device_island: DeviceIslandPolicy::default(),
            fused_steps: Vec::new(),
            residency_windows: Vec::new(),
            fusion_evaluations: Vec::new(),
            workload_evaluations: Vec::new(),
        }
    }

    pub fn backend_for_phase(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
    ) -> PlanBackend {
        self.dispatch_policy.backend_for(phase, queue_depths)
    }

    /// Specialize dispatchable islands for a live inference phase. Fixed CPU
    /// attention/softmax and ANE-only steps are preserved; XDNA/Metal/CPU
    /// islands may migrate among the policy's eligible decode backends.
    pub fn specialize_for_phase(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
    ) -> Self {
        let selected = self.backend_for_phase(phase, queue_depths);
        let eligible = match phase {
            InferencePhase::Prefill => vec![self.dispatch_policy.prefill],
            InferencePhase::Decode => self.dispatch_policy.decode_candidates.clone(),
        };
        let mut specialized = self.clone();
        for step in &mut specialized.fused_steps {
            if eligible.contains(&step.backend)
                && matches!(
                    step.backend,
                    PlanBackend::Xdna | PlanBackend::Metal | PlanBackend::Cpu
                )
                && eligible.contains(&selected)
            {
                step.backend = selected;
            }
        }
        specialized
    }

    /// Attach measured or modeled fusion alternatives to this deployable plan.
    pub fn with_fusion_evaluations(mut self, evaluations: Vec<FusionStrategyEvaluation>) -> Self {
        self.fusion_evaluations = evaluations;
        self
    }

    /// Attach realtime/batch and batch-size-specific strategy evidence.
    pub fn with_workload_evaluations(
        mut self,
        evaluations: Vec<WorkloadStrategyEvaluation>,
    ) -> Self {
        self.workload_evaluations = evaluations;
        self
    }

    /// Resolve the selected strategy for a runtime workload. Exact scenario
    /// matches win; otherwise the nearest scenario with the same realtime or
    /// batch mode is used as a calibrated fallback.
    pub fn selected_workload_strategy(
        &self,
        scenario: crate::fused_ops::WorkloadScenario,
    ) -> Option<&crate::fused_ops::FusionStrategy> {
        let evaluation = self
            .workload_evaluations
            .iter()
            .filter(|candidate| candidate.scenario.realtime == scenario.realtime)
            .min_by_key(|candidate| {
                let batch_distance =
                    candidate.scenario.batch_size.abs_diff(scenario.batch_size) as u64;
                let sequence_distance = candidate
                    .scenario
                    .sequence_length
                    .abs_diff(scenario.sequence_length)
                    as u64;
                (
                    (!(candidate.scenario.batch_size == scenario.batch_size
                        && candidate.scenario.sequence_length == scenario.sequence_length))
                        as u8,
                    batch_distance + sequence_distance,
                )
            })?;
        evaluation
            .evaluation
            .candidates
            .get(evaluation.evaluation.selected)
            .map(|candidate| &candidate.strategy)
    }

    /// Create a dispatch-ready plan annotated with the strategy selected for
    /// one concrete workload scenario.
    pub fn specialize_for_workload(&self, scenario: crate::fused_ops::WorkloadScenario) -> Self {
        self.try_specialize_for_workload(scenario)
            .unwrap_or_else(|_| self.clone())
    }

    /// Fallible workload specialization used by runtime admission paths.
    pub fn try_specialize_for_workload(
        &self,
        scenario: crate::fused_ops::WorkloadScenario,
    ) -> Result<Self, String> {
        scenario.validate()?;
        let Some(strategy) = self.selected_workload_strategy(scenario).cloned() else {
            return Ok(self.clone());
        };
        let mut specialized = self.clone();
        for step in &mut specialized.fused_steps {
            step.fusion_strategy = Some(strategy.clone());
        }
        Ok(specialized)
    }

    /// Returns the number of kernel dispatches in this plan.
    pub fn dispatch_count(&self) -> usize {
        self.schedule.len()
    }

    /// Returns the total estimated thread count for the plan.
    ///
    /// For each dispatch, the total thread count is width × height of
    /// the schedule entry's threadgroup.
    pub fn total_threads(&self) -> u64 {
        self.schedule
            .iter()
            .map(|e| e.threadgroup_width as u64 * e.threadgroup_height as u64)
            .sum()
    }

    pub fn route_names(&self) -> Vec<&'static str> {
        self.fused_steps
            .iter()
            .map(|step| match step.backend {
                PlanBackend::AnePlanar => "ane-planar",
                PlanBackend::AneMatrix => "ane-matrix",
                PlanBackend::Accelerate => "accelerate",
                PlanBackend::Metal => "metal",
                PlanBackend::Cpu => "cpu",
                PlanBackend::Xdna => "xdna",
            })
            .collect()
    }

    pub fn supports_all_streamed_workloads(&self) -> bool {
        !self.residency_windows.is_empty()
            && self.residency_windows.iter().all(|window| {
                window
                    .required_workloads
                    .contains(&ResidencyWorkload::RealtimeText)
                    && window
                        .required_workloads
                        .contains(&ResidencyWorkload::BatchedText)
                    && window
                        .required_workloads
                        .contains(&ResidencyWorkload::BatchedAudio)
            })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.dispatch_policy.decode_candidates.is_empty() {
            return Err("heterogeneous decode policy has no eligible backend".into());
        }
        let mut decode_backends = std::collections::HashSet::new();
        for backend in &self.dispatch_policy.decode_candidates {
            if !decode_backends.insert(*backend) {
                return Err(format!(
                    "heterogeneous decode policy repeats backend {:?}",
                    backend
                ));
            }
        }
        if self
            .dispatch_policy
            .backend_for(InferencePhase::Decode, &[])
            == PlanBackend::Cpu
            && !self
                .dispatch_policy
                .decode_candidates
                .contains(&PlanBackend::Cpu)
        {
            return Err("heterogeneous decode policy has no deterministic fallback".into());
        }
        for (index, entry) in self.schedule.iter().enumerate() {
            if entry.threadgroup_width == 0 || entry.threadgroup_height == 0 {
                return Err(format!("schedule entry {index} has zero-sized threadgroup"));
            }
            let threads = (entry.threadgroup_width as u64)
                .checked_mul(entry.threadgroup_height as u64)
                .ok_or_else(|| format!("schedule entry {index} threadgroup overflows"))?;
            if threads > 1024 {
                return Err(format!(
                    "schedule entry {index} threadgroup has {threads} threads"
                ));
            }
        }
        for (expected, step) in self.fused_steps.iter().enumerate() {
            if step.step_id != expected {
                return Err(format!(
                    "fused step id {} is not contiguous at {expected}",
                    step.step_id
                ));
            }
            if step
                .depends_on
                .iter()
                .any(|dependency| *dependency >= step.step_id)
            {
                return Err(format!(
                    "step {} has a forward or self dependency",
                    step.step_id
                ));
            }
            if step.depends_on.iter().any(|dependency| {
                !self
                    .fused_steps
                    .iter()
                    .any(|candidate| candidate.step_id == *dependency)
            }) {
                return Err(format!("step {} depends on a missing step", step.step_id));
            }
            if step.zero_copy && step.input_region != step.output_region {
                return Err(format!(
                    "zero-copy step {} changes memory region",
                    step.step_id
                ));
            }
            if step.dispatch_geometry[0] == 0
                || step.dispatch_geometry[1] == 0
                || step.dispatch_geometry[2] == 0
            {
                return Err(format!(
                    "step {} has zero-sized dispatch geometry",
                    step.step_id
                ));
            }
            for tensor in step.input_tensors.iter().chain(step.output_tensors.iter()) {
                let role = tensor.role.to_ascii_lowercase();
                let payload = if role.contains("resident") && role.contains("weight") {
                    Some(PcieBoundaryPayload::ResidentWeight)
                } else if role.contains("stream") && role.contains("weight") {
                    Some(PcieBoundaryPayload::StreamedWeight)
                } else if role.contains("activation") {
                    Some(PcieBoundaryPayload::Activation)
                } else if role.contains("work") || role.contains("descriptor") {
                    Some(PcieBoundaryPayload::WorkDescriptor)
                } else {
                    None
                };
                if let Some(payload) = payload {
                    self.device_island
                        .validate_boundary(payload)
                        .map_err(|error| {
                            format!(
                                "step {} tensor {} violates device-island policy: {error}",
                                step.step_id, tensor.name
                            )
                        })?;
                }
            }
        }
        for (index, evaluation) in self.fusion_evaluations.iter().enumerate() {
            if evaluation.candidates.is_empty() {
                return Err(format!("fusion evaluation {index} has no candidates"));
            }
            if evaluation.selected >= evaluation.candidates.len() {
                return Err(format!(
                    "fusion evaluation {index} selects {} beyond candidate count {}",
                    evaluation.selected,
                    evaluation.candidates.len()
                ));
            }
            if evaluation
                .candidates
                .iter()
                .any(|candidate| candidate.kernel_count == 0 || !candidate.score.is_finite())
            {
                return Err(format!(
                    "fusion evaluation {index} contains an invalid candidate"
                ));
            }
        }
        let mut workload_scenarios = std::collections::HashSet::new();
        for workload in &self.workload_evaluations {
            workload
                .scenario
                .validate()
                .map_err(|error| format!("invalid workload strategy scenario: {error}"))?;
            if !workload_scenarios.insert(workload.scenario) {
                return Err(format!(
                    "duplicate workload strategy scenario: realtime={}, batch={}, sequence={}",
                    workload.scenario.realtime,
                    workload.scenario.batch_size,
                    workload.scenario.sequence_length
                ));
            }
            if workload.evaluation.candidates.is_empty() {
                return Err("workload strategy evaluation has no candidates".into());
            }
            if workload.evaluation.selected >= workload.evaluation.candidates.len() {
                return Err(format!(
                    "workload strategy selection {} exceeds candidate count {}",
                    workload.evaluation.selected,
                    workload.evaluation.candidates.len()
                ));
            }
            if workload
                .evaluation
                .candidates
                .iter()
                .any(|candidate| candidate.kernel_count == 0 || !candidate.score.is_finite())
            {
                return Err("workload strategy evaluation contains an invalid candidate".into());
            }
        }
        if !self.supports_all_streamed_workloads() {
            return Err("residency windows do not cover all required workloads".into());
        }
        for window in &self.residency_windows {
            if window.resident_devices.is_empty() {
                return Err(format!(
                    "residency window {} has no resident device",
                    window.window_id
                ));
            }
            if window.required_workloads.is_empty() {
                return Err(format!(
                    "residency window {} declares no workloads",
                    window.window_id
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Lowering: spatial graph → KernelManifest
// ---------------------------------------------------------------------------

/// Lower a spatial graph into a [`KernelManifest`] with batch and realtime
/// execution plans.
///
/// # Arguments
///
/// * `graph` — The spatial dataflow graph to lower.
/// * `cost` — Pre-computed cost estimate for the graph.
/// * `format_plan` — Optional per-tensor format assignment from evolution
///   search. When provided, the format plan can be used to populate the
///   `codec` field on [`KernelDescriptor`] from per-tensor
///   [`TensorFormat`] assignments, and to override tile geometry with the
///   plan's per-tensor tile dimensions.
///
/// # Steps
///
/// 1. Walk every compute node in topological order and build one
///    [`KernelDescriptor`] per node, reading codec from graph annotations
///    (set by the caller via [`SpatialGraph::set_codec`]) and resolving
///    tile geometry from annotations or heuristics.
/// 2. Build the **batch plan**: standard topological schedule with
///    GEMM-oriented threadgroup sizes (width × height from the tile
///    geometry annotation, falling back to 16×16).
/// 3. Build the **realtime plan**: same topological order but with
///    single-row threadgroups (height = 1) to produce GEMV-shaped
///    dispatches. KV-cache persistence is enabled.
///
/// # Returns
///
/// `None` if the graph cannot be topologically sorted (cycle detected).
pub fn lower_to_manifest(
    graph: &SpatialGraph,
    cost: CostEstimate,
    format_plan: Option<&FormatPlan>,
) -> Option<KernelManifest> {
    let selected_metal_tile = format_plan.and_then(|plan| {
        let tiling = plan.joint_tiling?;
        let geometry = TileGeometry {
            width: tiling.metal_threadgroup_width as usize,
            height: tiling.metal_threadgroup_height as usize,
        };
        crate::tiling::validate_joint_tiling_geometry(geometry).ok()?;
        Some((
            tiling.metal_threadgroup_width as usize,
            tiling.metal_threadgroup_height as usize,
        ))
    });
    let topo = graph.topological_sort()?;

    // Phase 1: build kernel descriptors for every compute node.
    let mut kernel_descriptors: Vec<KernelDescriptor> = Vec::new();
    let mut node_to_kernel: HashMap<SpatialNodeId, usize> = HashMap::new();

    for (i, &node_id) in topo.iter().enumerate() {
        let node = graph.get_node(node_id)?;
        match node {
            SpatialNode::Compute {
                kind, intensity, ..
            } => {
                let tile_geo = selected_metal_tile
                    .map(|(width, height)| TileGeometry { width, height })
                    .or_else(|| tile_geometry_from_meta(graph, node_id, kind, *intensity));
                let threadgroup_total = tile_geo
                    .as_ref()
                    .map_or(256, |t| (t.width * t.height) as u32);

                let idx = kernel_descriptors.len();
                kernel_descriptors.push(KernelDescriptor {
                    node_id,
                    codec: codec_from_annotations(graph, node_id, format_plan),
                    tile_geometry: tile_geo,
                    threadgroup_size: threadgroup_total,
                    schedule_index: i,
                });
                node_to_kernel.insert(node_id, idx);
            }
            _ => {}
        }
    }

    if kernel_descriptors.is_empty() {
        return Some(KernelManifest {
            kernels: kernel_descriptors,
            total_cost: cost,
            batch_plan: None,
            realtime_plan: None,
        });
    }

    // Phase 2: build batch execution plan — standard GEMM-oriented schedule.
    let batch_schedule: Vec<ScheduleEntry> = topo
        .iter()
        .filter_map(|id| node_to_kernel.get(id))
        .map(|&kidx| {
            let kd = &kernel_descriptors[kidx];
            let (w, h) = kd
                .tile_geometry
                .as_ref()
                .map_or((16, 16), |t| (t.width as u32, t.height as u32));
            ScheduleEntry {
                kernel_index: kidx,
                threadgroup_width: w,
                threadgroup_height: h,
            }
        })
        .collect();

    let any_kv_kernel = kernel_descriptors.iter().any(|kd| {
        matches!(
            graph.get_node(kd.node_id),
            Some(SpatialNode::Compute { kind, .. })
                if matches!(kind, ComputeKind::Attention | ComputeKind::SSM)
        )
    });

    let batch_plan = ExecutionPlan::new(
        ExecutionMode::Batch,
        batch_schedule,
        ExecutionMode::Batch.default_batch_size(),
        !any_kv_kernel,
    );
    let batch_plan = with_fused_steps(
        batch_plan,
        graph,
        &topo,
        &node_to_kernel,
        &kernel_descriptors,
        cost.latency.as_nanos() as u64,
        format_plan.and_then(|plan| plan.joint_tiling.map(|t| (t.ane_tile_m, t.ane_tile_n))),
        format_plan.and_then(|plan| plan.joint_tiling.map(|t| t.ane_unit)),
    );
    let (fusion_evaluations, workload_evaluations) = graph_strategy_evidence(graph);
    let batch_plan = batch_plan
        .with_fusion_evaluations(fusion_evaluations.clone())
        .with_workload_evaluations(workload_evaluations.clone());

    // Phase 3: build realtime execution plan — GEMV-oriented schedule.
    let realtime_schedule: Vec<ScheduleEntry> = topo
        .iter()
        .filter_map(|id| node_to_kernel.get(id))
        .map(|&kidx| {
            let kd = &kernel_descriptors[kidx];
            let w = kd.tile_geometry.as_ref().map_or(16, |t| t.width as u32);
            ScheduleEntry {
                kernel_index: kidx,
                threadgroup_width: w,
                threadgroup_height: 1,
            }
        })
        .collect();

    let realtime_plan = ExecutionPlan::new(
        ExecutionMode::Realtime,
        realtime_schedule,
        ExecutionMode::Realtime.default_batch_size(),
        ExecutionMode::Realtime.default_persistent_cache(),
    );
    let realtime_plan = with_fused_steps(
        realtime_plan,
        graph,
        &topo,
        &node_to_kernel,
        &kernel_descriptors,
        cost.latency.as_nanos() as u64,
        format_plan.and_then(|plan| plan.joint_tiling.map(|t| (t.ane_tile_m, t.ane_tile_n))),
        format_plan.and_then(|plan| plan.joint_tiling.map(|t| t.ane_unit)),
    );
    let realtime_plan = realtime_plan
        .with_fusion_evaluations(fusion_evaluations)
        .with_workload_evaluations(workload_evaluations);

    Some(KernelManifest {
        kernels: kernel_descriptors,
        total_cost: cost,
        batch_plan: Some(batch_plan),
        realtime_plan: Some(realtime_plan),
    })
}

/// Derive executable strategy evidence while the graph is still available.
/// Keeping this at lowering time ensures the plan embedded in a CImage is
/// never disconnected from the fusion regions that produced its kernels.
fn graph_strategy_evidence(
    graph: &SpatialGraph,
) -> (
    Vec<FusionStrategyEvaluation>,
    Vec<WorkloadStrategyEvaluation>,
) {
    let regions = graph.available_fusions();
    let fusion_evaluations = graph
        .evaluate_available_fusions()
        .into_iter()
        .flat_map(|(_, evaluations)| evaluations)
        .collect::<Vec<_>>();
    if regions.is_empty() {
        return (fusion_evaluations, Vec::new());
    }
    let scenarios = [
        WorkloadScenario {
            realtime: true,
            batch_size: 1,
            sequence_length: 1,
        },
        WorkloadScenario {
            realtime: false,
            batch_size: 1,
            sequence_length: 1,
        },
        WorkloadScenario {
            realtime: false,
            batch_size: 4,
            sequence_length: 1,
        },
        WorkloadScenario {
            realtime: false,
            batch_size: 8,
            sequence_length: 1,
        },
    ];
    let workload_evaluations = scenarios
        .into_iter()
        .filter_map(|scenario| {
            let region_evaluations = regions
                .iter()
                .flat_map(|(anchor, permutations)| {
                    let element_count = graph
                        .get_node(*anchor)
                        .and_then(|node| match node {
                            SpatialNode::Compute { shape, .. } => shape.out_shapes.first(),
                            _ => None,
                        })
                        .and_then(|shape| {
                            shape.dims.iter().try_fold(1u64, |count, dimension| {
                                count.checked_mul(*dimension as u64)
                            })
                        })
                        .unwrap_or(1);
                    permutations.iter().filter_map(move |permutation| {
                        crate::fused_ops::evaluate_fusion_strategies_for_workload(
                            permutation,
                            element_count,
                            scenario,
                        )
                        .ok()
                    })
                })
                .collect::<Vec<_>>();
            let first = region_evaluations.first()?.clone();
            let candidates = (0..first.candidates.len())
                .map(|candidate_index| {
                    let strategy = first.candidates[candidate_index].strategy.clone();
                    let latency_ns = region_evaluations
                        .iter()
                        .map(|evaluation| {
                            evaluation.candidates[candidate_index].estimated_latency_ns
                        })
                        .fold(0u64, u64::saturating_add);
                    let materialized_bytes = region_evaluations
                        .iter()
                        .map(|evaluation| {
                            evaluation.candidates[candidate_index].estimated_materialized_bytes
                        })
                        .fold(0u64, u64::saturating_add);
                    let measured = region_evaluations
                        .iter()
                        .all(|evaluation| evaluation.candidates[candidate_index].measured);
                    FusionStrategyCandidate {
                        kernel_count: region_evaluations
                            .iter()
                            .map(|evaluation| evaluation.candidates[candidate_index].kernel_count)
                            .sum(),
                        score: latency_ns as f64 + materialized_bytes as f64 * 0.01,
                        strategy,
                        estimated_latency_ns: latency_ns,
                        estimated_materialized_bytes: materialized_bytes,
                        measured,
                    }
                })
                .collect::<Vec<_>>();
            let selected = candidates
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.score.total_cmp(&right.score))
                .map(|(index, _)| index)
                .unwrap_or(0);
            Some(WorkloadStrategyEvaluation {
                scenario,
                evaluation: FusionStrategyEvaluation {
                    candidates,
                    selected,
                },
            })
        })
        .collect();
    (fusion_evaluations, workload_evaluations)
}

fn with_fused_steps(
    mut plan: ExecutionPlan,
    graph: &SpatialGraph,
    topo: &[SpatialNodeId],
    node_to_kernel: &HashMap<SpatialNodeId, usize>,
    descriptors: &[KernelDescriptor],
    total_latency_ns: u64,
    ane_tile: Option<(u32, u32)>,
    ane_unit: Option<AneUnitAxis>,
) -> ExecutionPlan {
    let compute_nodes: Vec<SpatialNodeId> = topo
        .iter()
        .copied()
        .filter(|id| node_to_kernel.contains_key(id))
        .collect();
    let per_step = total_latency_ns
        .checked_div(compute_nodes.len().max(1) as u64)
        .unwrap_or(0);
    plan.fused_steps = compute_nodes
        .iter()
        .enumerate()
        .map(|(step_id, &node_id)| {
            let kernel_index = node_to_kernel[&node_id];
            let descriptor = &descriptors[kernel_index];
            let schedule_entry = plan
                .schedule
                .iter()
                .find(|entry| entry.kernel_index == kernel_index);
            let forced_xdna = graph
                .get_annotations(node_id)
                .and_then(|meta| meta.placement.as_deref())
                .map(|placement| {
                    placement.eq_ignore_ascii_case("xdna") || placement.eq_ignore_ascii_case("npu")
                })
                .unwrap_or(false);
            let xdna_shape_eligible = matches!(
                graph.get_node(node_id),
                Some(SpatialNode::Compute {
                    kind: ComputeKind::MatMul,
                    ..
                })
            );
            let xdna_auto_eligible = xdna_shape_eligible
                && matches!(
                    descriptor.codec,
                    Some(CodecVariant::Ternary | CodecVariant::Ternary1_58 | CodecVariant::Int8)
                );
            // XDNA is an explicit heterogeneous lane, but it can also be
            // selected by the host after it has opted into AMD target
            // discovery. Keeping the switch in the scheduler makes XDNA a
            // real peer of ANE/Metal without making macOS builds probe Linux
            // device nodes or silently changing their default route.
            let auto_xdna = std::env::var("PRISM_ENABLE_XDNA_AUTO").ok().as_deref() == Some("1")
                && matches!(
                    graph.get_node(node_id),
                    Some(SpatialNode::Compute {
                        kind: ComputeKind::MatMul,
                        ..
                    })
                )
                && matches!(
                    descriptor.codec,
                    Some(CodecVariant::Ternary | CodecVariant::Ternary1_58 | CodecVariant::Int8)
                );
            let backend = if (forced_xdna && xdna_shape_eligible)
                || (auto_xdna && xdna_auto_eligible)
            {
                PlanBackend::Xdna
            } else {
                match graph.get_node(node_id) {
                    Some(SpatialNode::Compute { kind, .. }) => match kind {
                        ComputeKind::MatMul
                            if matches!(
                                descriptor.codec,
                                Some(
                                    CodecVariant::Ternary
                                        | CodecVariant::Ternary1_58
                                        | CodecVariant::Int8
                                )
                            ) =>
                        {
                            match ane_unit.unwrap_or_default() {
                                AneUnitAxis::Planar => PlanBackend::AnePlanar,
                                AneUnitAxis::Auto | AneUnitAxis::Matrix => PlanBackend::AneMatrix,
                            }
                        }
                        // Attention has irregular score/softmax and masking
                        // behavior that is not a good XDNA contract. Keep it
                        // on the CPU-side accelerated route, where the host
                        // can use its vector math implementation and avoid
                        // an NPU round trip.
                        ComputeKind::Attention => PlanBackend::Accelerate,
                        ComputeKind::MatMul => PlanBackend::Metal,
                        ComputeKind::Elementwise
                        | ComputeKind::Normalization
                        | ComputeKind::RoPE
                        | ComputeKind::Softmax => match ane_unit.unwrap_or_default() {
                            AneUnitAxis::Matrix => PlanBackend::AneMatrix,
                            AneUnitAxis::Auto | AneUnitAxis::Planar => PlanBackend::AnePlanar,
                        },
                        _ => PlanBackend::Accelerate,
                    },
                    _ => PlanBackend::Cpu,
                }
            };
            let region = match backend {
                PlanBackend::AnePlanar | PlanBackend::AneMatrix => "ane-memory",
                PlanBackend::Metal => "unified-memory",
                PlanBackend::Accelerate | PlanBackend::Cpu | PlanBackend::Xdna => "unified-memory",
            };
            let (input_shape, output_shape) = match graph.get_node(node_id) {
                Some(SpatialNode::Compute { shape, .. }) => (
                    shape
                        .in_shapes
                        .first()
                        .map(|s| s.dims.iter().map(|&dim| dim as u64).collect())
                        .unwrap_or_default(),
                    shape
                        .out_shapes
                        .first()
                        .map(|s| s.dims.iter().map(|&dim| dim as u64).collect())
                        .unwrap_or_default(),
                ),
                _ => (Vec::new(), Vec::new()),
            };
            FusedScheduleStep {
                step_id,
                model_id: None,
                node_ids: vec![node_id],
                backend,
                depends_on: step_id.checked_sub(1).into_iter().collect(),
                input_region: region.into(),
                output_region: region.into(),
                zero_copy: matches!(backend, PlanBackend::AnePlanar | PlanBackend::AneMatrix),
                estimated_latency_ns: per_step,
                input_tensors: vec![TensorBinding {
                    name: graph
                        .get_annotations(node_id)
                        .and_then(|meta| meta.tensor_key.clone())
                        .unwrap_or_else(|| format!("node_{:?}_input", node_id)),
                    role: "input".into(),
                    element_type: if matches!(backend, PlanBackend::AneMatrix) {
                        "int8"
                    } else {
                        "fp16"
                    }
                    .into(),
                    shape: input_shape,
                }],
                output_tensors: vec![TensorBinding {
                    name: format!("node_{:?}_output", node_id),
                    role: "output".into(),
                    element_type: if matches!(backend, PlanBackend::AneMatrix) {
                        "int32"
                    } else {
                        "fp16"
                    }
                    .into(),
                    shape: output_shape,
                }],
                dispatch_geometry: if matches!(
                    backend,
                    PlanBackend::AnePlanar | PlanBackend::AneMatrix
                ) {
                    ane_tile.map(|(m, n)| [m, n, 1]).or_else(|| {
                        schedule_entry
                            .map(|entry| [entry.threadgroup_width, entry.threadgroup_height, 1])
                    })
                } else {
                    schedule_entry
                        .map(|entry| [entry.threadgroup_width, entry.threadgroup_height, 1])
                }
                .unwrap_or([descriptor.threadgroup_size, 1, 1]),
                fusion_strategy: None,
            }
        })
        .collect();
    plan.residency_windows = vec![ResidencyWindow {
        window_id: 0,
        model_bytes: 0,
        required_workloads: vec![
            ResidencyWorkload::RealtimeText,
            ResidencyWorkload::BatchedText,
            ResidencyWorkload::BatchedAudio,
        ],
        resident_devices: vec!["unified-memory".into()],
        prefetch_step: Some(0).filter(|_| !plan.fused_steps.is_empty()),
        eviction_step: Some(plan.fused_steps.len()).filter(|_| !plan.fused_steps.is_empty()),
    }];
    plan
}

/// Extract the codec variant for a node from graph annotations, falling back
/// to format-plan lookup when the annotations carry a `tensor_key` that
/// matches a format plan entry.
fn codec_from_annotations(
    graph: &SpatialGraph,
    node_id: SpatialNodeId,
    format_plan: Option<&FormatPlan>,
) -> Option<CodecVariant> {
    // Read the codec from graph annotations (set by the graph builder via
    // `SpatialGraph::set_codec`). When the format plan is provided and the
    // graph later carries tensor keys on node annotations, this function
    // can also look up `TensorFormat` from the plan and convert it to a
    // `CodecVariant` via `tensor_format_to_codec_variant`.
    if let Some(codec) = graph.get_annotations(node_id).and_then(|m| m.codec) {
        return Some(codec);
    }

    // Evolution search owns the format assignment when a graph annotation
    // does not provide one.  Preserve that assignment through lowering so
    // backend selection and the emitted AOT fused steps see the same format
    // that was scored by search.
    let tensor_key = graph
        .get_annotations(node_id)
        .and_then(|metadata| metadata.tensor_key.as_deref())?;
    let format = format_plan?.per_tensor.get(tensor_key)?.format;
    Some(match format {
        TensorFormat::Fp16 => CodecVariant::Fp16,
        TensorFormat::Bf16 => CodecVariant::Bf16,
        TensorFormat::Int8 => CodecVariant::Int8,
        TensorFormat::Int4 => CodecVariant::SymInt4,
        TensorFormat::Nf4 => CodecVariant::Nf4,
        TensorFormat::Nf8 => CodecVariant::Fp16,
        TensorFormat::Palettized4Bit => CodecVariant::Nf4,
        TensorFormat::Ternary158 => CodecVariant::Ternary1_58,
        TensorFormat::Binary1 => CodecVariant::Ternary,
        TensorFormat::TernaryTile640 => CodecVariant::Ternary,
    })
}

/// Map a [`TensorFormat`] from the evolution search framework to the
/// corresponding [`CodecVariant`] in the spatial IR.
#[allow(dead_code)]
fn tensor_format_to_codec_variant(tf: &TensorFormat) -> CodecVariant {
    match tf {
        TensorFormat::Fp16 => CodecVariant::Fp16,
        TensorFormat::Bf16 => CodecVariant::Bf16,
        TensorFormat::Int8 => CodecVariant::Int8,
        TensorFormat::Int4 => CodecVariant::SymInt4,
        TensorFormat::Nf4 => CodecVariant::Nf4,
        TensorFormat::Nf8 => CodecVariant::Q8_0,
        TensorFormat::Palettized4Bit => CodecVariant::Q4_0,
        TensorFormat::Ternary158 => CodecVariant::Ternary1_58,
        TensorFormat::Binary1 => CodecVariant::Ternary,
        TensorFormat::TernaryTile640 => CodecVariant::Ternary,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve tile geometry from graph annotations and node traits.
///
/// Resolve tile geometry from graph annotations and node traits.
fn tile_geometry_from_meta(
    graph: &SpatialGraph,
    node_id: SpatialNodeId,
    kind: &ComputeKind,
    intensity: ComputeIntensity,
) -> Option<TileGeometry> {
    if let Some(meta) = graph.get_annotations(node_id) {
        if let Some(size) = meta.batch_threadgroup_size {
            let dim = (size as f64).sqrt().ceil() as usize;
            let dim = dim.max(1).min(256);
            return Some(TileGeometry {
                width: dim,
                height: dim,
            });
        }
    }

    if let Some(meta) = graph.get_annotations(node_id) {
        if meta.tile_geometry.is_some() {
            return meta.tile_geometry;
        }
    }

    Some(heuristic_tile(kind, intensity))
}

/// Heuristic tile dimensions based on compute kind and intensity.
fn heuristic_tile(kind: &ComputeKind, intensity: ComputeIntensity) -> TileGeometry {
    let (w, h) = match (kind, intensity) {
        (ComputeKind::MatMul, ComputeIntensity::ComputeBound) => (32, 32),
        (ComputeKind::MatMul, _) => (16, 16),
        (ComputeKind::Attention, ComputeIntensity::ComputeBound) => (16, 16),
        (ComputeKind::Attention, _) => (8, 8),
        (ComputeKind::Convolution, _) => (8, 8),
        (ComputeKind::Elementwise, _) => (8, 4),
        (ComputeKind::Normalization, _) => (8, 4),
        (ComputeKind::Softmax, _) => (8, 4),
        (ComputeKind::RoPE, _) => (8, 4),
        (ComputeKind::SSM, _) => (16, 4),
        (ComputeKind::Reshape, _) => (4, 4),
        (ComputeKind::Gather, _) => (4, 4),
        (ComputeKind::Custom(_), _) => (8, 8),
    };
    TileGeometry {
        width: w,
        height: h,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::CostEstimate;
    use crate::graph::{
        ComputeIntensity, ComputeKind, EdgeDirection, MemoryKind, MemoryRegion, ShapeContract,
        SpatialEdge, SpatialEdgeId, SpatialGraph, SpatialNode,
    };
    use prism_ecs_ir::cimage_types::TensorShape;
    use std::time::Duration;

    fn make_simple_graph() -> SpatialGraph {
        let mut g = SpatialGraph::new();

        let _input_id = g.add_node(SpatialNode::Memory {
            id: SpatialNodeId(0),
            kind: MemoryKind::WeightStorage,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![64, 64] },
                element_size: 2,
                strides: vec![],
            },
        });

        let matmul_id = g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![
                    TensorShape { dims: vec![64, 64] },
                    TensorShape { dims: vec![64, 64] },
                ],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });

        let attn_id = g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::Attention,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![64, 64] }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });

        let elem_id = g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(3),
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![64, 64] }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });

        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: _input_id,
            sink: matmul_id,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![64, 64] }),
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(2),
            source: matmul_id,
            sink: attn_id,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![64, 64] }),
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(3),
            source: attn_id,
            sink: elem_id,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![64, 64] }),
        });

        g
    }

    #[test]
    fn test_execution_mode_defaults() {
        assert_eq!(ExecutionMode::Batch.default_batch_size(), 32);
        assert_eq!(ExecutionMode::Realtime.default_batch_size(), 1);
        assert!(!ExecutionMode::Batch.default_persistent_cache());
        assert!(ExecutionMode::Realtime.default_persistent_cache());
        assert!(ExecutionMode::Batch.is_batch());
        assert!(!ExecutionMode::Batch.is_realtime());
        assert!(ExecutionMode::Realtime.is_realtime());
        assert!(!ExecutionMode::Realtime.is_batch());
    }

    #[test]
    fn execution_plan_persists_fusion_evaluation_evidence() {
        let permutation = crate::fused_ops::check_fusion_legality(&[
            crate::fused_ops::FusableOp::FpGemv,
            crate::fused_ops::FusableOp::Silu,
        ])
        .unwrap();
        let evaluation = crate::fused_ops::evaluate_fusion_strategies(&permutation, 1024);
        let workload = crate::fused_ops::WorkloadStrategyEvaluation {
            scenario: crate::fused_ops::WorkloadScenario {
                realtime: true,
                batch_size: 1,
                sequence_length: 1,
            },
            evaluation: evaluation.clone(),
        };
        let plan = ExecutionPlan::new(ExecutionMode::Realtime, vec![], 1, true)
            .with_fusion_evaluations(vec![evaluation.clone()])
            .with_workload_evaluations(vec![workload.clone()]);
        let encoded = serde_json::to_string(&plan).unwrap();
        let decoded: ExecutionPlan = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.fusion_evaluations, vec![evaluation]);
        assert!(
            decoded.fusion_evaluations[0].selected < decoded.fusion_evaluations[0].candidates.len()
        );
        assert_eq!(decoded.workload_evaluations, vec![workload]);
        assert!(decoded
            .selected_workload_strategy(crate::fused_ops::WorkloadScenario {
                realtime: true,
                batch_size: 1,
                sequence_length: 1,
            })
            .is_some());
        assert!(decoded
            .selected_workload_strategy(crate::fused_ops::WorkloadScenario {
                realtime: false,
                batch_size: 32,
                sequence_length: 128,
            })
            .is_none());
        let specialized = decoded.specialize_for_workload(crate::fused_ops::WorkloadScenario {
            realtime: true,
            batch_size: 1,
            sequence_length: 1,
        });
        assert!(specialized
            .fused_steps
            .iter()
            .all(|step| step.fusion_strategy.is_some()));
    }

    #[test]
    fn test_lower_to_manifest_produces_both_plans() {
        let graph = make_simple_graph();
        let cost = CostEstimate::new(
            Duration::from_millis(10),
            1024 * 1024,
            512 * 1024,
            2,
            0.5,
            0.1,
        );

        let manifest = lower_to_manifest(&graph, cost.clone(), None).unwrap();

        // Three compute nodes → three kernel descriptors.
        assert_eq!(manifest.kernels.len(), 3);

        // Both plans present.
        let batch = manifest.batch_plan.as_ref().expect("batch plan must exist");
        let realtime = manifest
            .realtime_plan
            .as_ref()
            .expect("realtime plan must exist");

        // Batch plan checks.
        assert_eq!(batch.mode, ExecutionMode::Batch);
        assert_eq!(batch.batch_size, 32);
        assert_eq!(batch.dispatch_count(), 3);
        assert!(!batch.persistent_cache);

        // Realtime plan checks.
        assert_eq!(realtime.mode, ExecutionMode::Realtime);
        assert_eq!(realtime.batch_size, 1);
        assert_eq!(realtime.dispatch_count(), 3);
        assert!(realtime.persistent_cache);

        // Realtime: single-row threadgroups (GEMV).
        for entry in &realtime.schedule {
            assert_eq!(entry.threadgroup_height, 1);
        }

        // Batch: standard threadgroup dimensions.
        for entry in &batch.schedule {
            assert!(entry.threadgroup_width >= 1);
            assert!(entry.threadgroup_height >= 1);
        }
    }

    #[test]
    fn test_lower_manifest_batch_reference_same_kernels() {
        let graph = make_simple_graph();
        let cost = CostEstimate::new(
            Duration::from_millis(5),
            1024 * 1024,
            512 * 1024,
            1,
            0.3,
            0.05,
        );
        let manifest = lower_to_manifest(&graph, cost, None).unwrap();

        let batch = manifest.batch_plan.as_ref().unwrap();
        let realtime = manifest.realtime_plan.as_ref().unwrap();

        assert_eq!(batch.schedule.len(), realtime.schedule.len());
        for (b_entry, r_entry) in batch.schedule.iter().zip(realtime.schedule.iter()) {
            assert_eq!(b_entry.kernel_index, r_entry.kernel_index);
            assert_eq!(b_entry.threadgroup_width, r_entry.threadgroup_width);
            assert_eq!(r_entry.threadgroup_height, 1);
        }
    }

    #[test]
    fn explicit_xdna_placement_selects_xdna_route() {
        let mut graph = make_simple_graph();
        graph.set_annotation(SpatialNodeId(1), "placement", "xdna".into());
        let manifest = lower_to_manifest(&graph, CostEstimate::zero(), None).unwrap();
        assert_eq!(
            manifest.batch_plan.unwrap().fused_steps[0].backend,
            PlanBackend::Xdna
        );
    }

    #[test]
    fn explicit_xdna_placement_does_not_claim_attention_support() {
        let mut graph = make_simple_graph();
        if let Some(SpatialNode::Compute { kind, .. }) = graph.get_node_mut(SpatialNodeId(1)) {
            *kind = ComputeKind::Attention;
        }
        graph.set_annotation(SpatialNodeId(1), "placement", "xdna".into());
        let manifest = lower_to_manifest(&graph, CostEstimate::zero(), None).unwrap();
        assert_ne!(
            manifest.batch_plan.unwrap().fused_steps[0].backend,
            PlanBackend::Xdna
        );
    }

    #[test]
    fn isolated_device_island_keeps_weights_local() {
        let policy = DeviceIslandPolicy::default();
        assert!(policy
            .validate_boundary(PcieBoundaryPayload::Activation)
            .is_ok());
        assert!(policy
            .validate_boundary(PcieBoundaryPayload::WorkDescriptor)
            .is_ok());
        assert!(policy
            .validate_boundary(PcieBoundaryPayload::ResidentWeight)
            .is_err());
        assert!(policy
            .validate_boundary(PcieBoundaryPayload::StreamedWeight)
            .is_err());
        let streaming = DeviceIslandPolicy {
            isolated: true,
            allow_streamed_weights: true,
        };
        assert!(streaming
            .validate_boundary(PcieBoundaryPayload::StreamedWeight)
            .is_ok());
    }

    #[test]
    fn heterogeneous_policy_prefers_xdna_for_prefill_and_least_loaded_decode() {
        let policy = HeterogeneousDispatchPolicy::default();
        assert_eq!(
            policy.backend_for(InferencePhase::Prefill, &[]),
            PlanBackend::Xdna
        );
        assert_eq!(
            policy.backend_for(InferencePhase::Decode, &[]),
            PlanBackend::Cpu
        );
        assert_eq!(
            policy.backend_for(
                InferencePhase::Decode,
                &[
                    (PlanBackend::Xdna, 4),
                    (PlanBackend::Metal, 1),
                    (PlanBackend::Cpu, 2)
                ]
            ),
            PlanBackend::Metal
        );
    }

    #[test]
    fn phase_specialization_migrates_only_dispatchable_islands() {
        let mut plan = ExecutionPlan::new(ExecutionMode::Realtime, vec![], 1, true);
        plan.fused_steps = vec![
            FusedScheduleStep {
                step_id: 0,
                model_id: None,
                node_ids: vec![],
                backend: PlanBackend::Metal,
                depends_on: vec![],
                input_region: "unified-memory".into(),
                output_region: "unified-memory".into(),
                zero_copy: true,
                estimated_latency_ns: 1,
                input_tensors: vec![],
                output_tensors: vec![],
                dispatch_geometry: [1, 1, 1],
                fusion_strategy: None,
            },
            FusedScheduleStep {
                step_id: 1,
                model_id: None,
                node_ids: vec![],
                backend: PlanBackend::Accelerate,
                depends_on: vec![0],
                input_region: "unified-memory".into(),
                output_region: "unified-memory".into(),
                zero_copy: true,
                estimated_latency_ns: 1,
                input_tensors: vec![],
                output_tensors: vec![],
                dispatch_geometry: [1, 1, 1],
                fusion_strategy: None,
            },
        ];
        let specialized = plan.specialize_for_phase(
            InferencePhase::Decode,
            &[(PlanBackend::Metal, 4), (PlanBackend::Cpu, 1)],
        );
        assert_eq!(specialized.fused_steps[0].backend, PlanBackend::Cpu);
        assert_eq!(specialized.fused_steps[1].backend, PlanBackend::Accelerate);
    }

    #[test]
    fn test_lower_empty_graph() {
        let g = SpatialGraph::new();
        let cost = CostEstimate::new(Duration::ZERO, 0, 0, 0, 0.0, 1.0);
        let manifest = lower_to_manifest(&g, cost, None).unwrap();
        assert_eq!(manifest.kernels.len(), 0);
        assert!(manifest.batch_plan.is_none());
        assert!(manifest.realtime_plan.is_none());
    }
}
