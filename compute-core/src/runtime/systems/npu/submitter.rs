//! NPU Submitter System — dispatches compiled graphs to the NPU.
//!
//! Runs during the Intake stage.  For each worker with an [`NpuExecutionState`]
//! component that has not yet been submitted, fires the asynchronous FFI call
//! to submit the compiled graph.  The worker transitions to [`Streaming`] and
//! the ECS scheduler moves on immediately without blocking.
//!
//! After the FFI call returns a submission ID, the system flags the state as
//! submitted and transitions the worker lifecycle.  [`NpuObserverSystem`]
//! (Maintenance stage) polls for completion.
//!
//! # Lifecycle contract
//!
//! The submitter only acts on workers whose lifecycle phase is
//! [`AwaitingFirstEvent`] — the only phase from which a [`Streaming`]
//! transition is valid.  Workers in any other phase are skipped even if
//! [`NpuExecutionState`] is present.

use lazy_static::lazy_static;

use crate::backend::npu::ffi::{NpuBuffer, TargetNpu};
use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::component_id::{ComponentId, SchedulableComponent};
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::components::{WorkerLifecycle, WorkerRequestPhase};
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// Component — NpuExecutionState
// ---------------------------------------------------------------------------

/// ECS component tracking NPU execution state for a single worker-bound
/// inference request.
///
/// Owns the vendor-specific session handle and the input/output buffer
/// descriptors.  The `is_submitted` flag ensures exactly one FFI dispatch
/// per graph, even if the system is invoked over multiple ticks.
///
/// # Safety
///
/// `session` and `target` are opaque handles managed by the C FFI layer.
/// The `NpuSubmitterSystem` never dereferences or deallocates them.
pub struct NpuExecutionState {
    /// Target NPU accelerator (Apple ANE, Intel VPU, AMD XDNA).
    pub target: TargetNpu,
    /// Opaque vendor-specific session handle returned by `load_graph`.
    pub session: *mut std::ffi::c_void,
    /// Input buffers registered with the NPU DMA engine.
    pub inputs: Vec<NpuBuffer>,
    /// Output buffers registered with the NPU DMA engine.
    pub outputs: Vec<NpuBuffer>,
    /// Monotonically increasing submission ID assigned by `submit_execution`.
    pub submission_id: u64,
    /// Whether the graph has been submitted for execution.
    pub is_submitted: bool,
}

// Safety: raw pointers are only used for FFI dispatch; the session handle
// lifetime is managed externally by the graph loader.
unsafe impl Send for NpuExecutionState {}
unsafe impl Sync for NpuExecutionState {}

impl NpuExecutionState {
    /// Create a new execution state for a compiled NPU graph.
    pub fn new(
        target: TargetNpu,
        session: *mut std::ffi::c_void,
        inputs: Vec<NpuBuffer>,
        outputs: Vec<NpuBuffer>,
    ) -> Self {
        Self {
            target,
            session,
            inputs,
            outputs,
            submission_id: 0,
            is_submitted: false,
        }
    }
}

impl Default for NpuExecutionState {
    fn default() -> Self {
        Self {
            target: TargetNpu::AppleAne,
            session: std::ptr::null_mut(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            submission_id: 0,
            is_submitted: false,
        }
    }
}

impl SchedulableComponent for NpuExecutionState {
    const COMPONENT_ID: ComponentId = 17;
    const NAME: &'static str = "NpuExecutionState";
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// Dispatches compiled NPU graphs via an asynchronous FFI call.
///
/// Scans entities with [`NpuExecutionState`] during [`Stage::Intake`],
/// filters for unsubmitted graphs on workers in [`AwaitingFirstEvent`]
/// phase, fires the FFI [`submit_execution`], and transitions the worker
/// to [`Streaming`].
pub struct NpuSubmitterSystem {
    _private: (),
}

impl NpuSubmitterSystem {
    /// Create a new NPU submitter system.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for NpuSubmitterSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpec for NpuSubmitterSystem {
    type Reads = (WorkerLifecycle, NpuExecutionState);
    type Writes = (WorkerLifecycle, NpuExecutionState);
    type ReadResources = ();
    type WriteResources = ();

    const NAME: &'static str = "npu_submitter";
    const ID: SystemId = SystemId(201);
    const STAGE: Stage = Stage::Intake;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// Static metadata (lazy_static per system convention)
// ---------------------------------------------------------------------------

lazy_static! {
    static ref NPU_SUBMITTER_META: SystemMetadata = SystemMetadata {
        id: SystemId(201),
        name: "npu_submitter",
        stage: Stage::Intake,
        reads: <(WorkerLifecycle, NpuExecutionState) as ComponentSet>::mask().unwrap(),
        writes: <(WorkerLifecycle, NpuExecutionState) as ComponentSet>::mask().unwrap(),
        reads_resources: <() as ResourceSet>::mask().unwrap(),
        writes_resources: <() as ResourceSet>::mask().unwrap(),
        after: &[],
        before: &[],
        order: 0,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for NpuSubmitterSystem {
    fn metadata(&self) -> &SystemMetadata {
        &NPU_SUBMITTER_META
    }

    /// Execute the NPU submitter for the current tick.
    ///
    /// 1. Collects entity IDs with [`NpuExecutionState`] whose graph has not
    ///    yet been submitted and whose lifecycle is in [`AwaitingFirstEvent`].
    /// 2. For each candidate, fires [`submit_execution`] via FFI.
    /// 3. Updates the execution state with the returned submission ID.
    /// 4. Transitions the worker lifecycle to [`Streaming`].
    fn run(
        &mut self,
        world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // Collect entity IDs first to avoid borrow conflicts with get_mut below.
        let candidate_entities: Vec<_> = world
            .iter_entities_with::<NpuExecutionState>()
            .filter(|entity| {
                // Must be unsubmitted and in a phase that allows ->Streaming.
                let npu_ready = world
                    .get::<NpuExecutionState>(*entity)
                    .map(|s| !s.is_submitted)
                    .unwrap_or(false);
                if !npu_ready {
                    return false;
                }
                world
                    .get::<WorkerLifecycle>(*entity)
                    .map(|lc| lc.phase == WorkerRequestPhase::AwaitingFirstEvent)
                    .unwrap_or(false)
            })
            .collect();

        for entity in candidate_entities {
            // -- 1. Fire the asynchronous FFI call --------------------------
            // Extract FFI parameters from the mutable borrow, then release
            // it before re-borrowing for the state update below.
            let submission_id = {
        let (target, session, in_buf, out_buf, in_bytes, out_bytes) = {
            let ns = match world.get_mut::<NpuExecutionState>(entity) {
                    Some(s) => s,
                None => continue,
            };
            let t = ns.target;
            let s = ns.session;
            let ib = if !ns.inputs.is_empty() { ns.inputs[0].ptr } else { std::ptr::null_mut() };
            let ob = if !ns.outputs.is_empty() { ns.outputs[0].ptr } else { std::ptr::null_mut() };
            let isz = if !ns.inputs.is_empty() { ns.inputs[0].size } else { 0 };
            let osz = if !ns.outputs.is_empty() { ns.outputs[0].size } else { 0 };
            (t, s, ib, ob, isz, osz)
                };
        unsafe { crate::backend::npu::ffi::npu_submit_execution(target, session, in_buf, out_buf, in_bytes, out_bytes) }
            };

            // -- 2. Mark the execution state as submitted -------------------
            if let Some(npu_state) = world.get_mut::<NpuExecutionState>(entity) {
                npu_state.submission_id = submission_id;
                npu_state.is_submitted = true;
            }

            // -- 3. Transition the worker lifecycle to Streaming ------------
            //
            // The check above guarantees phase == AwaitingFirstEvent, so
            // transition_to(Streaming) is infallible here.
            if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                let _ = lc.transition_to(WorkerRequestPhase::Streaming);
            }
        }

        SystemResult::ok()
    }
}
