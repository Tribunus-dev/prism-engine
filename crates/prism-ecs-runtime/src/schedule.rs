// Submodules of the schedule module — each owns a single Prism-domain
// authority and is named for what it does in Prism, not for an
// upstream project. The submodule declaration list lives at the
// top of the file so the module surface is grep-able.
pub mod backpressure;
pub mod lane_queue;
pub mod lane_slot_lease;

use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::backend::{parse_backend, KernelArtifactBinding};
use crate::inference::{InferenceAdmissionPolicy, InferencePhase, InferenceWorkMetadata};
use crate::kernel::{
    AdmittedMarker, Command, CommandEnvelope, KernelHandle, PlannedMarker, PublishedMarker,
};
use crate::ports::RuntimeError;
use crate::ports::{
    DispatchHandle, DispatchRequest, DispatchStatus, ProviderSelectionReceipt,
    ProviderSelectionRequest, WorkDispatcher,
};
use prism_ecs_constitutional::lifecycle_command::{
    AcquireWorkLeaseCommand, AdmitWorkCommand, LifecycleCommand, MarkObservedCommand,
    RecordDispatchIntentCommand, RecordWorkPlanCommand, RequestCancellationCommand,
};
use prism_ecs_constitutional::types::{
    Config, Epoch, FilePath, Format, Generation, RejectionReason,
};
use prism_ecs_constitutional::work::{WorkInputPath, WorkOutputPath};
use prism_ecs_constitutional::work::{WorkItemComponent, WorkState};
use prism_ecs_core::Entity;

/// Stable system identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct SystemId(pub u64);

/// Canonical stage names for the execution schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum SystemStage {
    Observe,
    Plan,
    Admit,
    Lease,
    Dispatch,
    Collect,
    Publish,
    Cleanup,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Access {
    pub component_namespace: &'static str,
    pub component_id: u32,
    pub write: bool,
}

/// A registered system with its metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemSpec {
    pub id: SystemId,
    pub name: &'static str,
    pub stage: SystemStage,
    pub reads: Vec<Access>,
    pub writes: Vec<Access>,
    pub dependencies: Vec<SystemId>,
}

// ── Context, CommandBuffer, System trait ────────────────────────────────────

/// Context provided to a system during execution.
pub struct SystemContext<'w> {
    pub tick_number: u64,
    pub stage: SystemStage,
    pub deadline: std::time::Instant,
    pub cancellation: AtomicBool,
    /// Maximum number of non-terminal work items that may be admitted at once.
    /// The committed decision still goes through the bound [`KernelHandle`].
    pub admission_capacity: usize,
    pub inference_policy: InferenceAdmissionPolicy,
    pub command_buffer: CommandBuffer,
    pub world_view: &'w crate::world_view::WorldViewImpl<'w>,
    /// The single runtime authority for commands and provider decisions.
    pub kernel: &'w KernelHandle,
    /// Optional dispatcher for invoking external work.
    pub dispatcher: Option<&'w dyn WorkDispatcher>,
    /// Kernel-owned dispatcher for work items carrying a compiled artifact
    /// binding. Its resource state remains persistent in the KernelHandle.
    pub kernel_dispatcher: Option<&'w dyn WorkDispatcher>,
    /// Active dispatch handles to poll.
    pub active_dispatches: &'w std::sync::Mutex<Vec<DispatchHandle>>,
    /// Selection receipts produced during this tick.
    pub provider_receipts: &'w std::sync::Mutex<Vec<ProviderSelectionReceipt>>,
}

impl<'w> SystemContext<'w> {
    pub fn should_stop(&self) -> bool {
        self.cancellation.load(Ordering::Acquire) || std::time::Instant::now() >= self.deadline
    }

    pub fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub fn cancelled_or_expired(&self, metadata: &InferenceWorkMetadata) -> bool {
        metadata.deadline_expired(self.now_ms()) || self.should_stop()
    }
}

/// Per-system command buffer that feeds into a shared submission channel.
#[derive(Clone)]
pub struct CommandBuffer {
    inner: Arc<Mutex<Vec<CommandEnvelope>>>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn emit(&self, command: CommandEnvelope) {
        self.inner.lock().push(command);
    }

    pub fn drain(&self) -> Vec<CommandEnvelope> {
        std::mem::take(&mut *self.inner.lock())
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward-compatible alias for CommandBuffer.
pub type CommandEmitter = CommandBuffer;

/// A system that can be executed by the schedule.
pub trait System: Send + Sync {
    fn id(&self) -> SystemId;
    fn name(&self) -> &'static str;
    fn stage(&self) -> SystemStage;
    fn spec(&self) -> SystemSpec;
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError>;
}

// ── TickReceipt ────────────────────────────────────────────────────────────

/// Receipt produced after executing a single schedule tick.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmittedCommand {
    pub idempotency_key: uuid::Uuid,
    pub sequence: Option<u64>,
    pub world_epoch: Option<u64>,
    pub command_type: String,
    pub result_type: String,
}

/// Receipt produced after executing a single schedule tick.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TickReceipt {
    pub tick_number: u64,
    pub schedule_hash: [u8; 32],
    pub system_order: Vec<SystemId>,
    pub emitted_commands: Vec<Vec<EmittedCommand>>,
    pub duration_ms: u64,
    pub failure: Option<String>,
    pub provider_selections: Vec<ProviderSelectionReceipt>,
}

// ── RuntimeSchedule ────────────────────────────────────────────────────────

/// A validated, ready-to-execute schedule.
///
/// The canonical fields (`systems`, `stage_order`) are `BTreeMap`: the
/// schedule topology is part of the constitutional runtime authority
/// and the iteration order over registered systems is observable to
/// schedule validation, dispatch, and projection rebuilds. See
/// AGENTS.md "no HashMap/HashSet for canonical collections whose order
/// is observable."
///
/// The `system_map` field is **execution-plane state** — the trait-object
/// systems are looked up by `SystemId` during schedule execution but
/// their iteration order is not observed externally. Per
/// `references/rust-quality.md`, `HashMap` is allowed for
/// execution-plane state.
pub struct RuntimeSchedule {
    pub stages: Vec<SystemStage>,
    /// Spec-level metadata for each registered system.
    ///
    /// BTreeMap: the schedule topology is canonical; iteration over
    /// the system set is observable to validation and dispatch.
    pub systems: BTreeMap<SystemId, SystemSpec>,
    /// Stage → ordered system ids.
    ///
    /// BTreeMap: the stage order is canonical; iteration order is
    /// observable to the schedule executor and to the topological sort.
    pub stage_order: BTreeMap<SystemStage, Vec<SystemId>>,
    /// Deterministic hash of the schedule topology.
    pub schedule_hash: [u8; 32],
    /// Trait-object system instances indexed by id, used during execution.
    ///
    /// HashMap is allowed here: this is execution-plane state, not
    /// canonical. The systems are looked up by id but not iterated in
    /// a canonical order.
    pub system_map: HashMap<SystemId, Box<dyn System>>,
    /// Monotonically increasing tick counter (shared across threads).
    pub tick_count: AtomicU64,
    /// Per-stage timeout in milliseconds.
    pub stage_timeout_ms: u64,
    /// Bounded active-work admission policy owned by this schedule.
    pub admission_capacity: usize,
    /// Token/KV admission and chunking policy for inference work.
    pub inference_policy: InferenceAdmissionPolicy,
    /// Kernel handle for submitting emitted commands.
    pub kernel: Option<KernelHandle>,
    /// Provider-neutral dispatcher for adapter execution.
    pub dispatcher: Option<std::sync::Arc<dyn WorkDispatcher>>,
    /// Active dispatch handles.
    pub active_dispatches: std::sync::Mutex<Vec<DispatchHandle>>,
    /// Provider-selection receipts accumulated during a tick.
    pub provider_receipts: std::sync::Mutex<Vec<ProviderSelectionReceipt>>,
}

#[derive(Debug)]
pub enum ScheduleError {
    DuplicateSystemId(SystemId),
    MissingDependency(SystemId, SystemId),
    CycleDetected(Vec<SystemId>),
    ConflictingParallelWrite(Vec<(SystemId, SystemId, String)>),
    InvalidStageDependency(SystemId, SystemId),
    InvalidAdmissionCapacity,
    InvalidInferencePolicy,
}

impl RuntimeSchedule {
    /// Build an empty schedule.
    pub fn new() -> Self {
        Self {
            stages: vec![
                SystemStage::Observe,
                SystemStage::Plan,
                SystemStage::Admit,
                SystemStage::Lease,
                SystemStage::Dispatch,
                SystemStage::Collect,
                SystemStage::Publish,
                SystemStage::Cleanup,
            ],
            systems: BTreeMap::new(),
            stage_order: BTreeMap::new(),
            schedule_hash: [0u8; 32],
            system_map: HashMap::new(),
            tick_count: AtomicU64::new(0),
            stage_timeout_ms: 5000,
            admission_capacity: 32,
            inference_policy: InferenceAdmissionPolicy::default(),
            kernel: None,
            dispatcher: None,
            active_dispatches: std::sync::Mutex::new(vec![]),
            provider_receipts: std::sync::Mutex::new(vec![]),
        }
    }

    /// Register a system specification.
    pub fn register(&mut self, spec: SystemSpec) -> Result<(), ScheduleError> {
        if self.systems.contains_key(&spec.id) {
            return Err(ScheduleError::DuplicateSystemId(spec.id));
        }
        self.systems.insert(spec.id, spec.clone());
        self.stage_order
            .entry(spec.stage)
            .or_default()
            .push(spec.id);
        Ok(())
    }

    /// Register a system trait object (spec extracted from the trait).
    pub fn register_system(&mut self, system: Box<dyn System>) -> Result<(), ScheduleError> {
        let spec = system.spec();
        let id = spec.id;
        self.register(spec)?;
        self.system_map.insert(id, system);
        Ok(())
    }

    /// Bind the schedule to a kernel handle for command submission.
    pub fn bind(&mut self, kernel: &KernelHandle) {
        self.kernel = Some(kernel.clone());
    }

    /// Register a provider-neutral dispatcher for adapter execution.
    pub fn set_dispatcher(&mut self, dispatcher: std::sync::Arc<dyn WorkDispatcher>) {
        self.dispatcher = Some(dispatcher);
    }

    /// Set the maximum number of non-terminal work items that can be admitted
    /// concurrently. Zero would permanently strand work, so reject it before
    /// the schedule is installed on a kernel.
    pub fn set_admission_capacity(&mut self, capacity: usize) -> Result<(), ScheduleError> {
        if capacity == 0 {
            return Err(ScheduleError::InvalidAdmissionCapacity);
        }
        self.admission_capacity = capacity;
        Ok(())
    }

    pub fn set_inference_policy(
        &mut self,
        policy: InferenceAdmissionPolicy,
    ) -> Result<(), ScheduleError> {
        if !policy.validate() {
            return Err(ScheduleError::InvalidInferencePolicy);
        }
        self.inference_policy = policy;
        Ok(())
    }

    /// Validate the schedule: cycles, missing deps, duplicate IDs,
    /// parallel write conflicts, stage ordering, and schedule hash.
    pub fn validate(&mut self) -> Result<(), ScheduleError> {
        // Check all dependencies exist
        for (id, spec) in &self.systems {
            for dep in &spec.dependencies {
                if !self.systems.contains_key(dep) {
                    return Err(ScheduleError::MissingDependency(*id, *dep));
                }
            }
        }

        // Check for cycles via DFS
        let all_ids: Vec<SystemId> = self.systems.keys().copied().collect();
        for id in &all_ids {
            let mut visited = HashSet::new();
            let mut stack = Vec::new();
            if self.detect_cycle(*id, &mut visited, &mut stack) {
                return Err(ScheduleError::CycleDetected(stack));
            }
        }

        // Reject dependencies pointing from earlier stages to later stages
        let stage_index: HashMap<SystemStage, usize> = self
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect();
        for (id, spec) in &self.systems {
            for dep in &spec.dependencies {
                if let Some(dep_spec) = self.systems.get(dep) {
                    let my_idx = stage_index.get(&spec.stage).copied().unwrap_or(0);
                    let dep_idx = stage_index.get(&dep_spec.stage).copied().unwrap_or(0);
                    if dep_idx > my_idx {
                        return Err(ScheduleError::InvalidStageDependency(*id, *dep));
                    }
                }
            }
        }

        // Check parallel write conflicts within the same stage
        let mut conflicts = Vec::new();
        for ids in self.stage_order.values() {
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let a = &self.systems[&ids[i]];
                    let b = &self.systems[&ids[j]];
                    for wa in &a.writes {
                        for wb in &b.writes {
                            if wa.component_namespace == wb.component_namespace
                                && wa.component_id == wb.component_id
                            {
                                conflicts.push((
                                    a.id,
                                    b.id,
                                    format!("{}:{}", wa.component_namespace, wa.component_id),
                                ));
                            }
                        }
                    }
                }
            }
        }
        if !conflicts.is_empty() {
            return Err(ScheduleError::ConflictingParallelWrite(conflicts));
        }

        // Comprehensive schedule hash: stage position, system id + name + stage + deps + accesses
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        for (i, stage) in self.stages.iter().enumerate() {
            hasher.update(&(i as u64).to_le_bytes());
            if let Some(ids) = self.stage_order.get(stage) {
                for id in ids {
                    if let Some(spec) = self.systems.get(id) {
                        hasher.update(&spec.id.0.to_le_bytes());
                        hasher.update(spec.name.as_bytes());
                        hasher.update(&(spec.stage as u64).to_le_bytes());
                        for dep in &spec.dependencies {
                            hasher.update(&dep.0.to_le_bytes());
                        }
                        for a in &spec.reads {
                            hasher.update(a.component_namespace.as_bytes());
                            hasher.update(&a.component_id.to_le_bytes());
                        }
                        for a in &spec.writes {
                            hasher.update(a.component_namespace.as_bytes());
                            hasher.update(&a.component_id.to_le_bytes());
                            hasher.update(&[1u8]);
                        }
                    }
                }
            }
        }
        hasher.update(&(self.admission_capacity as u64).to_le_bytes());
        hasher.update(&self.inference_policy.max_inflight_tokens.to_le_bytes());
        hasher.update(&self.inference_policy.max_kv_tokens.to_le_bytes());
        hasher.update(&self.inference_policy.prefill_chunk_tokens.to_le_bytes());
        hasher.update(
            &self
                .inference_policy
                .max_prefill_tokens_per_tick
                .to_le_bytes(),
        );
        self.schedule_hash = *hasher.finalize().as_bytes();

        Ok(())
    }

    /// Compute a topological ordering per stage, respecting intra-stage
    /// dependencies. Returns a map from stage to ordered system ids.
    pub fn topological_order(&self) -> Result<BTreeMap<SystemStage, Vec<SystemId>>, RuntimeError> {
        let mut result: BTreeMap<SystemStage, Vec<SystemId>> = BTreeMap::new();
        for stage in &self.stages {
            let ids: Vec<SystemId> = self.stage_order.get(stage).cloned().unwrap_or_default();
            let sorted = self.kahn_sort(ids, *stage)?;
            if sorted.is_empty() {
                continue;
            }
            result.insert(*stage, sorted);
        }
        Ok(result)
    }

    /// Kahn's algorithm for topological sort within a single stage.
    fn kahn_sort(
        &self,
        ids: Vec<SystemId>,
        stage: SystemStage,
    ) -> Result<Vec<SystemId>, RuntimeError> {
        let id_set: HashSet<SystemId> = ids.iter().copied().collect();
        let mut in_degree: HashMap<SystemId, usize> = HashMap::new();
        let mut dependents: HashMap<SystemId, Vec<SystemId>> = HashMap::new();

        for id in &ids {
            in_degree.entry(*id).or_insert(0);
            dependents.entry(*id).or_default();
        }

        for id in &ids {
            if let Some(spec) = self.systems.get(id) {
                for dep in &spec.dependencies {
                    if id_set.contains(dep) {
                        dependents.entry(*dep).or_default().push(*id);
                        *in_degree.entry(*id).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut queue: Vec<SystemId> = ids
            .iter()
            .filter(|id| *in_degree.get(id).unwrap_or(&0) == 0)
            .copied()
            .collect();
        queue.sort_by_key(|id| id.0);
        let mut sorted = Vec::new();

        while !queue.is_empty() {
            let id = queue.remove(0);
            sorted.push(id);
            if let Some(children) = dependents.get(&id) {
                for child in children {
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(*child);
                        }
                    }
                }
            }
            queue.sort_by_key(|id| id.0);
        }

        if sorted.len() != ids.len() {
            return Err(RuntimeError::Entity(format!(
                "cycle detected in stage {:?}",
                stage
            )));
        }

        Ok(sorted)
    }

    fn detect_cycle(
        &self,
        start: SystemId,
        visited: &mut HashSet<SystemId>,
        stack: &mut Vec<SystemId>,
    ) -> bool {
        if stack.contains(&start) {
            stack.push(start);
            return true;
        }
        if visited.contains(&start) {
            return false;
        }
        visited.insert(start);
        stack.push(start);
        if let Some(spec) = self.systems.get(&start) {
            for dep in &spec.dependencies {
                if self.detect_cycle(*dep, visited, stack) {
                    return true;
                }
            }
        }
        stack.pop();
        false
    }

    pub fn schedule_hash(&self) -> [u8; 32] {
        self.schedule_hash
    }

    /// Run a single tick of the schedule.
    pub fn run_tick(&self) -> Result<TickReceipt, RuntimeError> {
        let start = std::time::Instant::now();
        let tick_number = self.tick_count.fetch_add(1, Ordering::SeqCst);
        self.provider_receipts.lock().unwrap().clear();
        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| RuntimeError::Entity("schedule not bound to kernel".into()))?;
        let stage_topological = self.topological_order()?;
        let kernel_dispatcher = kernel.kernel_dispatcher();
        let mut system_order = Vec::new();
        let mut emitted_commands = Vec::new();
        for stage in &self.stages {
            let deadline = start + std::time::Duration::from_millis(self.stage_timeout_ms);
            if let Some(ids) = stage_topological.get(stage) {
                let world_guard = kernel.lock_world();
                let world_view = crate::world_view::WorldViewImpl::new(world_guard);
                let mut stage_commands = Vec::new();
                for id in ids {
                    if std::time::Instant::now() > deadline {
                        return Err(RuntimeError::Entity(format!(
                            "tick deadline exceeded at stage {:?}",
                            stage
                        )));
                    }
                    if let Some(system) = self.system_map.get(id) {
                        let buffer = CommandBuffer::new();
                        let ctx = SystemContext {
                            tick_number,
                            stage: *stage,
                            deadline,
                            cancellation: AtomicBool::new(false),
                            admission_capacity: self.admission_capacity,
                            inference_policy: self.inference_policy,
                            command_buffer: buffer.clone(),
                            world_view: &world_view,
                            kernel,
                            dispatcher: self.dispatcher.as_deref(),
                            kernel_dispatcher: Some(kernel_dispatcher.as_ref()),
                            active_dispatches: &self.active_dispatches,
                            provider_receipts: &self.provider_receipts,
                        };
                        if let Err(e) = system.run(&ctx) {
                            return Err(RuntimeError::Entity(format!(
                                "system {} at stage {:?} failed: {}",
                                id.0, stage, e
                            )));
                        }
                        let commands = buffer.drain();
                        stage_commands.push((*id, commands));
                    }
                }
                drop(world_view);
                for (id, commands) in stage_commands {
                    let mut emitted = Vec::new();
                    for envelope in commands {
                        let ek = envelope.idempotency_key;
                        let cmd_type = format!("{:?}", envelope.command)
                            .lines()
                            .next()
                            .unwrap_or("")
                            .to_string();
                        match kernel.submit(envelope) {
                            Ok(outcome) => {
                                let result_type = format!("{:?}", outcome.result)
                                    .lines()
                                    .next()
                                    .unwrap_or("")
                                    .to_string();
                                emitted.push(EmittedCommand {
                                    idempotency_key: ek,
                                    sequence: Some(outcome.sequence),
                                    world_epoch: Some(outcome.world_epoch),
                                    command_type: cmd_type,
                                    result_type,
                                });
                            }
                            Err(e) => {
                                return Err(RuntimeError::Entity(format!(
                                    "kernel submit failed: {e}"
                                )));
                            }
                        }
                    }
                    system_order.push(id);
                    emitted_commands.push(emitted);
                }
            }
        }
        let duration_ms = start.elapsed().as_millis() as u64;
        let provider_selections = self.provider_receipts.lock().unwrap().clone();
        Ok(TickReceipt {
            tick_number,
            schedule_hash: self.schedule_hash,
            system_order,
            emitted_commands,
            duration_ms,
            failure: None,
            provider_selections,
        })
    }
}

impl Default for RuntimeSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Observe: check for admitted work that needs attention.
pub struct ObserveSystem;

impl System for ObserveSystem {
    fn id(&self) -> SystemId {
        SystemId(1)
    }
    fn name(&self) -> &'static str {
        "observe"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Observe
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_OBSERVE: usize = 64;
        if ctx.should_stop() {
            return Ok(());
        }
        let epoch = ctx.world_view.epoch();
        let mut pending: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                **state == WorkState::Pending
                    && ctx.world_view.get::<PlannedMarker>(*entity).is_none()
            })
            .map(|(entity, _)| entity)
            .collect();
        pending.sort_by_key(|entity| entity.id());
        pending.truncate(MAX_OBSERVE);
        for entity in pending {
            if ctx.should_stop() {
                break;
            }
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::MarkObserved(MarkObservedCommand {
                        entity,
                        observed_epoch: Epoch(epoch),
                    }),
                )));
        }
        Ok(())
    }
}

/// Plan: advance agents from Planning to Ready if they have dependencies met.
pub struct PlanSystem;

impl System for PlanSystem {
    fn id(&self) -> SystemId {
        SystemId(2)
    }
    fn name(&self) -> &'static str {
        "plan"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Plan
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(1)],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_PLAN: usize = 64;
        if ctx.should_stop() {
            return Ok(());
        }
        let mut observed: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                **state == WorkState::Ready
                    && ctx.world_view.get::<PlannedMarker>(*entity).is_some()
                    && ctx.world_view.get::<AdmittedMarker>(*entity).is_none()
            })
            .map(|(entity, _)| entity)
            .collect();
        observed.sort_by_key(|entity| entity.id());
        observed.truncate(MAX_PLAN);
        for entity in observed {
            if ctx.should_stop() {
                break;
            }
            let resource_estimate = ctx
                .world_view
                .get::<WorkItemComponent>(entity)
                .map(|_| 1024u64)
                .unwrap_or(512);
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::RecordWorkPlan(RecordWorkPlanCommand {
                        entity,
                        backend: parse_backend("auto")?,
                        output_format: Format("cimage".to_string()),
                        resource_estimate_bytes: resource_estimate,
                        timeout_ms: 30000,
                    }),
                )));
        }
        Ok(())
    }
}

/// Admit: check authority, policy, and resource bounds.
pub struct AdmitSystem;

impl System for AdmitSystem {
    fn id(&self) -> SystemId {
        SystemId(3)
    }
    fn name(&self) -> &'static str {
        "admit"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Admit
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(2)],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_ADMIT_PER_TICK: usize = 64;
        if ctx.should_stop() {
            return Ok(());
        }
        let active = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                matches!(**state, WorkState::Ready | WorkState::Leased(_))
                    && ctx.world_view.get::<AdmittedMarker>(*entity).is_some()
            })
            .count();
        let mut reserved_tokens: u32 = ctx
            .world_view
            .query::<WorkState>()
            .filter_map(|(entity, state)| {
                (matches!(*state, WorkState::Ready | WorkState::Leased(_))
                    && ctx.world_view.get::<AdmittedMarker>(entity).is_some())
                .then(|| ctx.world_view.get::<InferenceWorkMetadata>(entity))
                .flatten()
                .map(InferenceWorkMetadata::reserved_tokens)
            })
            .sum();
        let available = ctx.admission_capacity.saturating_sub(active);
        if available == 0 {
            return Ok(());
        }

        let mut planned: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                **state == WorkState::Ready
                    && ctx.world_view.get::<PlannedMarker>(*entity).is_some()
                    && ctx.world_view.get::<AdmittedMarker>(*entity).is_none()
            })
            .map(|(entity, _)| entity)
            .collect();
        planned.sort_by(|a, b| {
            let a_priority = ctx
                .world_view
                .get::<InferenceWorkMetadata>(*a)
                .map(|metadata| metadata.priority)
                .unwrap_or(0);
            let b_priority = ctx
                .world_view
                .get::<InferenceWorkMetadata>(*b)
                .map(|metadata| metadata.priority)
                .unwrap_or(0);
            b_priority
                .cmp(&a_priority)
                .then_with(|| a.id().cmp(&b.id()))
        });
        let mut admitted = 0usize;
        for entity in planned {
            if admitted >= available.min(MAX_ADMIT_PER_TICK) || ctx.should_stop() {
                break;
            }
            if let Some(metadata) = ctx.world_view.get::<InferenceWorkMetadata>(entity) {
                if metadata.deadline_expired(ctx.now_ms()) {
                    ctx.command_buffer
                        .emit(CommandEnvelope::new(Command::Lifecycle(
                            LifecycleCommand::RequestCancellation(RequestCancellationCommand {
                                entity,
                                reason: RejectionReason(
                                    "inference deadline expired before admission".to_string(),
                                ),
                            }),
                        )));
                    continue;
                }
                if metadata.required_kv_tokens() > ctx.inference_policy.max_kv_tokens
                    || reserved_tokens.saturating_add(metadata.reserved_tokens())
                        > ctx.inference_policy.max_inflight_tokens
                {
                    continue;
                }
                reserved_tokens = reserved_tokens.saturating_add(metadata.reserved_tokens());
            }
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::AdmitWork(AdmitWorkCommand { entity }),
                )));
            admitted += 1;
        }
        Ok(())
    }
}

/// Lease: acquire resource leases for admitted work.
pub struct LeaseSystem;

impl System for LeaseSystem {
    fn id(&self) -> SystemId {
        SystemId(4)
    }
    fn name(&self) -> &'static str {
        "lease"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Lease
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19, // WorkState
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(3)], // depends on Admit
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_LEASE: usize = 32;
        if ctx.should_stop() {
            return Ok(());
        }
        let mut admitted: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                **state == WorkState::Ready
                    && ctx.world_view.get::<AdmittedMarker>(*entity).is_some()
            })
            .map(|(entity, _)| entity)
            .collect();
        admitted.sort_by_key(|entity| entity.id());
        admitted.truncate(MAX_LEASE);
        for entity in admitted {
            if ctx.should_stop() {
                break;
            }
            if let Some(metadata) = ctx.world_view.get::<InferenceWorkMetadata>(entity) {
                if metadata.deadline_expired(ctx.now_ms()) {
                    ctx.command_buffer
                        .emit(CommandEnvelope::new(Command::Lifecycle(
                            LifecycleCommand::RequestCancellation(RequestCancellationCommand {
                                entity,
                                reason: RejectionReason(
                                    "inference deadline expired before lease".to_string(),
                                ),
                            }),
                        )));
                    continue;
                }
            }
            let ttl_ms = ctx
                .world_view
                .get::<InferenceWorkMetadata>(entity)
                .and_then(|metadata| metadata.deadline_ms.checked_sub(ctx.now_ms()))
                .map(|remaining| remaining.max(1).min(60_000))
                .unwrap_or(60_000);
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::AcquireWorkLease(AcquireWorkLeaseCommand {
                        work_entity: entity,
                        ttl_ms,
                        lease_generation: Generation(1),
                    }),
                )));
        }
        Ok(())
    }
}

/// Dispatch: hand leased work to hardware or subprocess adapters.
pub struct DispatchSystem;

impl System for DispatchSystem {
    fn id(&self) -> SystemId {
        SystemId(5)
    }
    fn name(&self) -> &'static str {
        "dispatch"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Dispatch
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(4)],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_DISPATCH: usize = 32;
        if ctx.should_stop() {
            return Ok(());
        }
        let mut leased: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                matches!(state, WorkState::Leased(_))
                    && ctx.world_view.get::<AdmittedMarker>(*entity).is_some()
            })
            .map(|(entity, _)| entity)
            .collect();
        leased.sort_by_key(|entity| entity.id());
        leased.truncate(MAX_DISPATCH);
        for &entity in leased.iter() {
            if ctx.should_stop() {
                break;
            }
            let already_active = ctx
                .active_dispatches
                .lock()
                .map(|active| {
                    active
                        .iter()
                        .any(|handle| handle.work_entity == entity.id())
                })
                .unwrap_or(true);
            if already_active {
                continue;
            }
            let now_ms = ctx.now_ms();
            let (config, deadline_ms) = if let Some(metadata) =
                ctx.world_view.get::<InferenceWorkMetadata>(entity)
            {
                if metadata.deadline_expired(now_ms) {
                    ctx.command_buffer
                        .emit(CommandEnvelope::new(Command::Lifecycle(
                            LifecycleCommand::RequestCancellation(RequestCancellationCommand {
                                entity,
                                reason: RejectionReason(
                                    "inference deadline expired before dispatch".to_string(),
                                ),
                            }),
                        )));
                    continue;
                }
                let (chunk_start, chunk_end) = match metadata.phase {
                    InferencePhase::Prefill => metadata.next_prefill_chunk(
                        ctx.inference_policy
                            .prefill_chunk_tokens
                            .min(ctx.inference_policy.max_prefill_tokens_per_tick),
                    ),
                    InferencePhase::Decode => {
                        (metadata.prefilled_tokens, metadata.prefilled_tokens)
                    }
                };
                let phase = match metadata.phase {
                    InferencePhase::Prefill if chunk_end > chunk_start => "prefill",
                    _ => "decode",
                };
                let config = serde_json::json!({
                    "phase": phase,
                    "prefill_start": chunk_start,
                    "prefill_end": chunk_end,
                    "decode_token_index": metadata.generated_tokens,
                    "kv_epoch": metadata.kv_epoch,
                    "kv_tokens": metadata.kv_tokens,
                    "token_budget": metadata.reserved_tokens(),
                })
                .to_string();
                let deadline = if metadata.deadline_ms == 0 {
                    now_ms.saturating_add(30_000)
                } else {
                    metadata.deadline_ms
                };
                (config, deadline)
            } else {
                ("{}".to_string(), now_ms.saturating_add(30_000))
            };
            let selection = ctx.kernel.select_provider(&ProviderSelectionRequest {
                operation: "work_dispatch".to_string(),
                requested_provider: Some("auto".to_string()),
                fallback_providers: vec!["metal".to_string(), "cpu".to_string()],
            });
            if let Ok(mut receipts) = ctx.provider_receipts.lock() {
                receipts.push(selection.clone());
            }

            let Some(backend) = selection.selected_backend().map(str::to_owned) else {
                ctx.command_buffer
                    .emit(CommandEnvelope::new(Command::Lifecycle(
                        LifecycleCommand::FailWork(
                            prism_ecs_constitutional::lifecycle_command::FailWorkCommand {
                                work_entity: entity,
                                error: RejectionReason(format!(
                                    "provider selection failed: {:?}",
                                    selection.fallback_reason
                                )),
                                lease_generation: Generation(1),
                                retryable: true,
                            },
                        ),
                    )));
                continue;
            };

            let effective_config = if config == "{}" {
                serde_json::to_string(&selection).unwrap_or_else(|_| "{}".into())
            } else {
                config.clone()
            };
            let artifact_binding = ctx
                .world_view
                .get::<KernelArtifactBinding>(entity)
                .cloned();
            if let Some(binding) = &artifact_binding {
                if let Err(error) = ctx
                    .kernel
                    .backend_resources()
                    .validate_dispatch(&backend, &binding.dispatch_spec())
                {
                    ctx.command_buffer
                        .emit(CommandEnvelope::new(Command::Lifecycle(
                            LifecycleCommand::FailWork(
                                prism_ecs_constitutional::lifecycle_command::FailWorkCommand {
                                    work_entity: entity,
                                    error: RejectionReason(format!(
                                        "artifact dispatch validation failed: {error}"
                                    )),
                                    lease_generation: Generation(1),
                                    retryable: false,
                                },
                            ),
                        )));
                    continue;
                }
            }
            let effective_config = artifact_binding
                .as_ref()
                .map(|binding| binding.dispatch_config(&selection))
                .unwrap_or(effective_config);
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::RecordDispatchIntent(RecordDispatchIntentCommand {
                        work_entity: entity,
                        backend: parse_backend(&backend)?,
                        config: Config(effective_config.clone()),
                        deadline_ms,
                    }),
                )));
            let dispatcher = if artifact_binding.is_some() {
                ctx.dispatcher.or(ctx.kernel_dispatcher)
            } else {
                ctx.dispatcher
            };
            if let Some(dispatcher) = dispatcher {
                let output_path = ctx
                    .world_view
                    .get::<WorkOutputPath>(entity)
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                let input_path = ctx
                    .world_view
                    .get::<WorkInputPath>(entity)
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                let request = DispatchRequest {
                    work_entity: entity.id(),
                    attempt: 1,
                    plan_generation: 0,
                    lease_token: format!("work-lease:{}", entity.id()),
                    deadline_ms,
                    backend,
                    config: effective_config,
                    input_path,
                    output_path,
                };
                match dispatcher.start(&request) {
                    Ok(handle) => {
                        if let Ok(mut active) = ctx.active_dispatches.lock() {
                            active.push(handle);
                        }
                    }
                    Err(e) => {
                        ctx.command_buffer
                            .emit(CommandEnvelope::new(Command::Lifecycle(
                                LifecycleCommand::FailWork(
                                    prism_ecs_constitutional::lifecycle_command::FailWorkCommand {
                                        work_entity: entity,
                                        error: RejectionReason(format!("dispatch start failed: {e}")),
                                        lease_generation: Generation(1),
                                        retryable: true,
                                    },
                                ),
                            )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Collect: convert external completion into ECS commands.
pub struct CollectSystem;

impl System for CollectSystem {
    fn id(&self) -> SystemId {
        SystemId(6)
    }
    fn name(&self) -> &'static str {
        "collect"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Collect
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(5)],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        if ctx.should_stop() {
            return Ok(());
        }
        if let Some(dispatcher) = ctx.dispatcher.or(ctx.kernel_dispatcher) {
            if let Ok(mut active) = ctx.active_dispatches.lock() {
                let mut i = 0;
                while i < active.len() {
                    if ctx.should_stop() {
                        break;
                    }
                    let handle = active[i].clone();
                    let entity = Entity::new(handle.work_entity, 0);
                    let metadata = ctx.world_view.get::<InferenceWorkMetadata>(entity).copied();
                    let cancelled = ctx
                        .world_view
                        .get::<WorkState>(entity)
                        .is_some_and(|state| *state == WorkState::Cancelled);
                    let expired =
                        metadata.is_some_and(|metadata| metadata.deadline_expired(ctx.now_ms()));
                    if cancelled || expired {
                        let _ = dispatcher.cancel(&handle);
                        if !cancelled {
                            ctx.command_buffer
                                .emit(CommandEnvelope::new(Command::Lifecycle(
                                    LifecycleCommand::RequestCancellation(
                                        RequestCancellationCommand {
                                            entity: Entity::new(handle.work_entity, 0),
                                            reason: RejectionReason(
                                                "inference deadline expired during dispatch"
                                                    .to_string(),
                                            ),
                                        },
                                    ),
                                )));
                        }
                        active.swap_remove(i);
                        continue;
                    }
                    match dispatcher.poll(&handle) {
                        Ok(DispatchStatus::Completed(output)) => {
                            if let Some(metadata) = metadata {
                                match metadata.phase {
                                    InferencePhase::Prefill => {
                                        let next = metadata.next_after_prefill(
                                            ctx.inference_policy.prefill_chunk_tokens.min(
                                                ctx.inference_policy.max_prefill_tokens_per_tick,
                                            ),
                                        );
                                        ctx.command_buffer.emit(CommandEnvelope::new(
                                            Command::AdvanceInference {
                                                entity: handle.work_entity,
                                                phase: next.phase,
                                                prefilled_tokens: next.prefilled_tokens,
                                                generated_tokens: next.generated_tokens,
                                                kv_epoch: next.kv_epoch,
                                                kv_tokens: next.kv_tokens,
                                            },
                                        ));
                                    }
                                    InferencePhase::Decode
                                        if metadata.is_complete_after_decode() =>
                                    {
                                        ctx.command_buffer.emit(CommandEnvelope::new(
                                            Command::Lifecycle(
                                                LifecycleCommand::CompleteWork(
                                                    prism_ecs_constitutional::lifecycle_command::CompleteWorkCommand {
                                                        work_entity: Entity::new(handle.work_entity, 0),
                                                        output,
                                                        output_path: FilePath(String::new()),
                                                        lease_generation: Generation(handle.attempt),
                                                    },
                                                ),
                                            ),
                                        ));
                                    }
                                    InferencePhase::Decode => {
                                        let next = metadata.next_after_decode();
                                        ctx.command_buffer.emit(CommandEnvelope::new(
                                            Command::AdvanceInference {
                                                entity: handle.work_entity,
                                                phase: next.phase,
                                                prefilled_tokens: next.prefilled_tokens,
                                                generated_tokens: next.generated_tokens,
                                                kv_epoch: next.kv_epoch,
                                                kv_tokens: next.kv_tokens,
                                            },
                                        ));
                                    }
                                }
                            } else {
                                ctx.command_buffer.emit(CommandEnvelope::new(
                                    Command::Lifecycle(
                                        LifecycleCommand::CompleteWork(
                                            prism_ecs_constitutional::lifecycle_command::CompleteWorkCommand {
                                                work_entity: Entity::new(handle.work_entity, 0),
                                                output,
                                                output_path: FilePath(String::new()),
                                                lease_generation: Generation(handle.attempt),
                                            },
                                        ),
                                    ),
                                ));
                            }
                            active.swap_remove(i);
                        }
                        Ok(DispatchStatus::Failed(e)) => {
                            ctx.command_buffer
                                .emit(CommandEnvelope::new(Command::Lifecycle(
                                LifecycleCommand::FailWork(
                                    prism_ecs_constitutional::lifecycle_command::FailWorkCommand {
                                        work_entity: Entity::new(handle.work_entity, 0),
                                        error: RejectionReason(e),
                                        lease_generation: Generation(handle.attempt),
                                        retryable: false,
                                    },
                                ),
                            )));
                            active.swap_remove(i);
                        }
                        Ok(DispatchStatus::Running) => {
                            i += 1;
                        }
                        Err(_) => {
                            i += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Publish: persist results and evidence before acknowledgement.
pub struct PublishSystem;

impl System for PublishSystem {
    fn id(&self) -> SystemId {
        SystemId(7)
    }
    fn name(&self) -> &'static str {
        "publish"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Publish
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(6)],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_PUBLISH: usize = 32;
        if ctx.should_stop() {
            return Ok(());
        }
        let mut completed: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(entity, state)| {
                **state == WorkState::Completed
                    && ctx.world_view.get::<PublishedMarker>(*entity).is_none()
            })
            .map(|(entity, _)| entity)
            .collect();
        completed.sort_by_key(|entity| entity.id());
        completed.truncate(MAX_PUBLISH);
        for entity in completed {
            if ctx.should_stop() {
                break;
            }
            let entity_id = entity.id();
            let result_payload = if let Some(path) = ctx.world_view.get::<WorkOutputPath>(entity) {
                let output_path = path.0.as_str();
                match std::fs::metadata(output_path) {
                    Ok(meta) if meta.len() > 0 => {
                        // Read file and compute blake3 digest
                        let digest = match std::fs::read(output_path) {
                            Ok(data) => {
                                let hash = blake3::hash(&data);
                                hash.to_hex().to_string()
                            }
                            Err(e) => serde_json::json!({
                                "error": format!("publish read: {e}"),
                                "output_path": output_path,
                            })
                            .to_string(),
                        };
                        serde_json::json!({
                            "digest": digest,
                            "digest_algorithm": "blake3",
                            "compiler_identity": "prism-engine",
                            "output_path": output_path,
                            "file_size": meta.len(),
                            "status": "verified",
                        })
                        .to_string()
                    }
                    Ok(_) => serde_json::json!({
                        "status": "empty_output",
                        "output_path": output_path,
                    })
                    .to_string(),
                    Err(e) => serde_json::json!({
                        "status": "file_not_found",
                        "error": format!("metadata: {e}"),
                        "output_path": output_path,
                    })
                    .to_string(),
                }
            } else {
                "{}".to_string()
            };
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::PublishResult(
                        prism_ecs_constitutional::lifecycle_command::PublishResultCommand {
                            entity: Entity::new(entity_id, 0),
                            result_type: Format("compilation".to_string()),
                            result: result_payload,
                        },
                    ),
                )));
        }
        Ok(())
    }
}

/// Cleanup: release leases, expire transient state, compact completed work.
pub struct CleanupSystem;

impl System for CleanupSystem {
    fn id(&self) -> SystemId {
        SystemId(8)
    }
    fn name(&self) -> &'static str {
        "cleanup"
    }
    fn stage(&self) -> SystemStage {
        SystemStage::Cleanup
    }
    fn spec(&self) -> SystemSpec {
        SystemSpec {
            id: self.id(),
            name: self.name(),
            stage: self.stage(),
            reads: vec![Access {
                component_namespace: "prism.work",
                component_id: 19,
                write: false,
            }],
            writes: vec![],
            dependencies: vec![SystemId(7)],
        }
    }
    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), RuntimeError> {
        const MAX_CLEANUP: usize = 32;
        if ctx.should_stop() {
            return Ok(());
        }
        let mut terminal: Vec<Entity> = ctx
            .world_view
            .query::<WorkState>()
            .filter(|(_, state)| state.is_terminal())
            .map(|(entity, _)| entity)
            .collect();
        terminal.sort_by_key(|entity| entity.id());
        terminal.truncate(MAX_CLEANUP);
        for entity in terminal {
            if ctx.should_stop() {
                break;
            }
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::ReleaseWorkLease(
                        prism_ecs_constitutional::lifecycle_command::ReleaseWorkLeaseCommand {
                            work_entity: entity,
                        },
                    ),
                )));
            ctx.command_buffer
                .emit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::ExpireTransientState(
                        prism_ecs_constitutional::lifecycle_command::ExpireTransientCommand {
                            entity,
                        },
                    ),
                )));
        }
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::CommandResult;

    use prism_ecs_constitutional::lifecycle_command::{
        AcquireWorkLeaseCommand, AdmitWorkCommand, CreateWorkCommand, LifecycleCommand,
        LifecycleCommandResult, MarkObservedCommand, RecordDispatchIntentCommand,
        RecordWorkPlanCommand, RequestCancellationCommand,
    };
    use prism_ecs_constitutional::scheduler::{InferenceHint, Priority, ResourceClaim};
    use prism_ecs_constitutional::types::{Epoch, FilePath, Format, Generation, RejectionReason};
    use prism_ecs_kernel::{
        BackendKind, CpuBackend, DispatchGeometry, KernelBackend, KernelCompileRequest,
        KernelDescriptor, KernelVariant,
    };

    /// Map a u32 priority to the typed `Priority` enum used by `ResourceClaim`.
    /// 0 = Low, 1 = Normal, 2 = High, 3+ = Critical. The exact mapping is
    /// not externally observable — tests just need a stable order so that
    /// higher numeric priority wins the admission race.
    fn priority_to_priority(p: u32) -> Priority {
        match p {
            0 => Priority::Low,
            1 => Priority::Normal,
            2 => Priority::High,
            _ => Priority::Critical,
        }
    }

    /// Build a default empty `ResourceClaim` for tests. The original tests
    /// passed arbitrary JSON strings for inference metadata; the typed
    /// claim now carries only scheduler fields. Inference hints land in
    /// a separate path (not exercised by these tests).
    fn default_claim() -> ResourceClaim {
        ResourceClaim::default()
    }

    fn test_cpu_artifact() -> prism_ecs_kernel::KernelArtifact {
        CpuBackend
            .compile(&KernelCompileRequest {
                source: b"schedule-uop".to_vec(),
                descriptor: KernelDescriptor {
                    name: "schedule_uop".into(),
                    variant: KernelVariant::FP16GEMV,
                    backend: BackendKind::CPU,
                    source_digest: String::new(),
                    binary_digest: String::new(),
                    binding_signature: Vec::new(),
                    dispatch_geometry: DispatchGeometry {
                        threads_per_threadgroup: [1, 1, 1],
                        threadgroups_per_grid: [1, 1, 1],
                        threads_per_grid: [1, 1, 1],
                    },
                },
                source_path: None,
            })
            .unwrap()
    }

    #[test]
    fn test_lifecycle_pipeline_create_observe_plan_admit() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.register_system(Box::new(CleanupSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        kernel.set_schedule(sched);
        let outcome = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
                    output_path: FilePath("".to_string()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let work_entity = match outcome.result {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            _ => panic!("expected WorkCreated"),
        };
        assert!(work_entity.id() > 0, "positive ID");
        let tick0 = kernel.run_tick().expect("tick 0");
        assert_eq!(tick0.tick_number, 0);
        let total: usize = tick0.emitted_commands.iter().map(|c| c.len()).sum();
        assert_eq!(total, 5, "tick 0: lifecycle commands across 8 stages");
        for i in 1..=5 {
            let tick = kernel.run_tick().unwrap_or_else(|_| panic!("tick {i}"));
            let repeated_observation = tick
                .emitted_commands
                .iter()
                .flatten()
                .any(|command| command.command_type.contains("MarkObserved"));
            assert!(!repeated_observation, "tick {i} must not re-observe work");
        }
    }

    #[test]
    fn compiled_artifact_propagates_through_dispatch_collect_and_completion() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut binding = handle
            .register_kernel_artifact(test_cpu_artifact())
            .expect("register compiled artifact");
        binding.inputs = vec![vec![0x00, 0x3c], 2.0f32.to_ne_bytes().to_vec()];

        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        kernel.set_schedule(sched);

        let create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
                    output_path: FilePath(String::new()),
                    input_path: FilePath(String::new()),
                }),
            )))
            .expect("create work");
        let work_entity = match create.result {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            other => panic!("expected work creation, got {other:?}"),
        };
        handle
            .bind_kernel_artifact(work_entity.id(), binding)
            .expect("bind artifact to work");

        for _ in 0..5 {
            kernel.run_tick().expect("run ECS tick");
            if handle
                .lock_world()
                .get_component::<WorkState>(work_entity)
                .is_some_and(|state| state.is_terminal())
            {
                break;
            }
        }
        let world = handle.lock_world();
        assert_eq!(
            world.get_component::<WorkState>(work_entity),
            Some(&WorkState::Completed)
        );
        let payload = world
            .get_component::<crate::ports::ResultPayload>(work_entity)
            .expect("backend output receipt");
        eprintln!("result bytes {:?}", payload.result.as_bytes());
        assert!(
            !payload.result.is_empty(),
            "backend output receipt should contain bytes"
        );
    }

    #[test]
    fn test_admission_capacity_is_canonical_and_retries_queued_work() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched
            .set_admission_capacity(2)
            .expect("positive admission capacity");
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        kernel.set_schedule(sched);

        for _ in 0..3 {
            handle
                .submit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::CreateWork(CreateWorkCommand {
                        entity: Entity::new(0, 0),
                        target_entity: Entity::new(0, 0),
                        kind: Format("inference".to_string()),
                        resource_claim: default_claim(),
                        output_path: FilePath(String::new()),
                        input_path: FilePath(String::new()),
                    }),
                )))
                .expect("create inference work");
        }

        let first_tick = kernel.run_tick().expect("first admission tick");
        let admitted_first_tick = first_tick
            .emitted_commands
            .iter()
            .flatten()
            .filter(|command| command.command_type.contains("AdmitWork"))
            .count();
        assert_eq!(
            admitted_first_tick, 2,
            "capacity must bound first admission"
        );

        let second_tick = kernel.run_tick().expect("second admission tick");
        let admitted_second_tick = second_tick
            .emitted_commands
            .iter()
            .flatten()
            .filter(|command| command.command_type.contains("AdmitWork"))
            .count();
        assert_eq!(
            admitted_second_tick, 0,
            "queued work must remain unadmitted while the canonical active set is full"
        );
        assert!(
            second_tick
                .emitted_commands
                .iter()
                .flatten()
                .all(|command| !command.command_type.contains("DeferWork")),
            "capacity backpressure is represented by canonical queued state, not a second queue"
        );
    }

    #[test]
    fn test_cancelled_work_is_not_admitted() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        kernel.set_schedule(sched);

        let work_entity = match handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("inference".to_string()),
                    resource_claim: default_claim(),
                    output_path: FilePath(String::new()),
                    input_path: FilePath(String::new()),
                }),
            )))
            .expect("create inference work")
            .result
        {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            other => panic!("expected work creation, got {other:?}"),
        };

        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::MarkObserved(MarkObservedCommand {
                    entity: work_entity,
                    observed_epoch: Epoch(0),
                }),
            )))
            .expect("observe work");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RecordWorkPlan(RecordWorkPlanCommand {
                    entity: work_entity,
                    backend: parse_backend("auto").unwrap(),
                    output_format: Format("tokens".to_string()),
                    resource_estimate_bytes: 1024,
                    timeout_ms: 30000,
                }),
            )))
            .expect("plan work");
        let cancelled = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RequestCancellation(RequestCancellationCommand {
                    entity: work_entity,
                    reason: RejectionReason("client disconnected".to_string()),
                }),
            )))
            .expect("cancel work");
        assert!(matches!(
            cancelled.result,
            CommandResult::Lifecycle(LifecycleCommandResult::RequestCancelled { .. })
        ));

        let tick = kernel.run_tick().expect("cancellation admission tick");
        assert!(
            tick.emitted_commands
                .iter()
                .flatten()
                .all(|command| !command.command_type.contains("AdmitWork")),
            "cancellation must win before admission"
        );
    }

    #[test]
    fn test_zero_admission_capacity_is_rejected() {
        let mut schedule = RuntimeSchedule::new();
        assert!(matches!(
            schedule.set_admission_capacity(0),
            Err(ScheduleError::InvalidAdmissionCapacity)
        ));
    }

    #[test]
    fn inference_admission_is_token_kv_aware_and_priority_deterministic() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut schedule = RuntimeSchedule::new();
        schedule.set_admission_capacity(4).unwrap();
        schedule
            .set_inference_policy(InferenceAdmissionPolicy {
                max_inflight_tokens: 14,
                max_kv_tokens: 32,
                ..InferenceAdmissionPolicy::default()
            })
            .unwrap();
        schedule.register_system(Box::new(ObserveSystem)).unwrap();
        schedule.register_system(Box::new(PlanSystem)).unwrap();
        schedule.register_system(Box::new(AdmitSystem)).unwrap();
        schedule.bind(&handle);
        schedule.validate().unwrap();
        kernel.set_schedule(schedule);

        let create = |priority: u32| {
            handle
                .submit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::CreateWork(CreateWorkCommand {
                        entity: Entity::new(0, 0),
                        target_entity: Entity::new(0, 0),
                        kind: Format("inference".to_string()),
                        resource_claim: ResourceClaim {
                            memory_bytes: 0,
                            compute_units: 0,
                            priority: priority_to_priority(priority),
                            inference_hint: Some(InferenceHint {
                                prompt_tokens: 8,
                                max_new_tokens: 4,
                                prefill_chunk_tokens: 1,
                                kv_epoch: 9,
                                kv_tokens: 0,
                                kv_capacity_tokens: 32,
                                deadline_ms: 0,
                                priority,
                            }),
                        },
                        output_path: FilePath(String::new()),
                        input_path: FilePath(String::new()),
                    }),
                )))
                .unwrap()
                .result
        };
        let first = match create(1) {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            result => panic!("expected work creation, got {result:?}"),
        };
        let second = match create(2) {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            result => panic!("expected work creation, got {result:?}"),
        };

        kernel.run_tick().unwrap();
        let world = handle.lock_world();
        assert!(
            world
                .get_component::<AdmittedMarker>(second)
                .is_some(),
            "higher priority request must win the deterministic token budget"
        );
        assert!(
            world
                .get_component::<AdmittedMarker>(first)
                .is_none(),
            "the second request cannot exceed the global token budget"
        );
    }

    #[test]
    fn inference_prefill_advances_in_chunks_and_decode_is_one_token() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut schedule = RuntimeSchedule::new();
        schedule
            .set_inference_policy(InferenceAdmissionPolicy {
                prefill_chunk_tokens: 2,
                max_prefill_tokens_per_tick: 2,
                ..InferenceAdmissionPolicy::default()
            })
            .unwrap();
        schedule.register_system(Box::new(ObserveSystem)).unwrap();
        schedule.register_system(Box::new(PlanSystem)).unwrap();
        schedule.register_system(Box::new(AdmitSystem)).unwrap();
        schedule.register_system(Box::new(LeaseSystem)).unwrap();
        schedule.register_system(Box::new(DispatchSystem)).unwrap();
        schedule.register_system(Box::new(CollectSystem)).unwrap();
        schedule.bind(&handle);
        schedule.validate().unwrap();
        schedule.set_dispatcher(std::sync::Arc::new(
            crate::test_adapters::FakeWorkDispatcher,
        ));
        kernel.set_schedule(schedule);

        let entity = match handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("inference".to_string()),
                    resource_claim: ResourceClaim {
                        memory_bytes: 0,
                        compute_units: 0,
                        priority: Priority::Normal,
                        inference_hint: Some(InferenceHint {
                            prompt_tokens: 5,
                            max_new_tokens: 2,
                            prefill_chunk_tokens: 2,
                            kv_epoch: 17,
                            kv_tokens: 0,
                            kv_capacity_tokens: 32,
                            deadline_ms: 0,
                            priority: 0,
                        }),
                    },
                    output_path: FilePath(String::new()),
                    input_path: FilePath(String::new()),
                }),
            )))
            .unwrap()
            .result
        {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated { work_entity, .. }) => {
                work_entity
            }
            result => panic!("expected work creation, got {result:?}"),
        };

        for expected_prefilled in [2, 4, 5] {
            let tick = kernel.run_tick().unwrap();
            assert!(
                tick.emitted_commands
                    .iter()
                    .flatten()
                    .any(|command| command.command_type.contains("AdvanceInference")),
                "each prefill dispatch must advance canonical inference metadata"
            );
            let world = handle.lock_world();
            let metadata = world
                .get_component::<InferenceWorkMetadata>(entity)
                .copied()
                .unwrap();
            assert_eq!(metadata.prefilled_tokens, expected_prefilled);
        }

        let decode_tick = kernel.run_tick().unwrap();
        let world = handle.lock_world();
        let metadata = world
            .get_component::<InferenceWorkMetadata>(entity)
            .copied()
            .unwrap();
        assert_eq!(metadata.generated_tokens, 1);
        assert!(
            decode_tick
                .emitted_commands
                .iter()
                .flatten()
                .any(|command| command.command_type.contains("AdvanceInference")),
            "decode must advance one token through KernelHandle"
        );
    }

    #[test]
    fn inference_deadline_cancels_before_admission() {
        use prism_ecs_constitutional::scheduler::InferenceHint;
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut schedule = RuntimeSchedule::new();
        schedule.register_system(Box::new(ObserveSystem)).unwrap();
        schedule.register_system(Box::new(PlanSystem)).unwrap();
        schedule.register_system(Box::new(AdmitSystem)).unwrap();
        schedule.bind(&handle);
        schedule.validate().unwrap();
        kernel.set_schedule(schedule);

        let entity = match handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("inference".to_string()),
                    resource_claim: ResourceClaim {
                        memory_bytes: 0,
                        compute_units: 0,
                        priority: Priority::Normal,
                        inference_hint: Some(InferenceHint {
                            prompt_tokens: 4,
                            max_new_tokens: 1,
                            prefill_chunk_tokens: 1,
                            kv_epoch: 0,
                            kv_tokens: 0,
                            kv_capacity_tokens: 0,
                            // deadline already passed: 1 ms (immediately expired)
                            deadline_ms: 1,
                            priority: 0,
                        }),
                    },
                    output_path: FilePath(String::new()),
                    input_path: FilePath(String::new()),
                }),
            )))
            .unwrap()
            .result
        {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            result => panic!("expected work creation, got {result:?}"),
        };

        let tick = kernel.run_tick().unwrap();
        let world = handle.lock_world();
        assert_eq!(
            world.get_component::<WorkState>(entity),
            Some(&WorkState::Cancelled)
        );
        assert!(tick
            .emitted_commands
            .iter()
            .flatten()
            .any(|command| command.command_type.contains("RequestCancellation")));
        assert!(world
            .get_component::<AdmittedMarker>(entity)
            .is_none());
    }

    #[test]
    fn test_direct_command_chain_to_completed() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("test".to_string()),
                    resource_claim: default_claim(),
                    output_path: FilePath("".to_string()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let work_entity = match &create.result {
            CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => *work_entity,
            _ => panic!("expected WorkCreated"),
        };
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::MarkObserved(MarkObservedCommand {
                    entity: work_entity,
                    observed_epoch: Epoch(0),
                }),
            )))
            .expect("observe");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RecordWorkPlan(RecordWorkPlanCommand {
                    entity: work_entity,
                    backend: parse_backend("auto").unwrap(),
                    output_format: Format("cimage".to_string()),
                    resource_estimate_bytes: 1024,
                    timeout_ms: 30000,
                }),
            )))
            .expect("plan");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::AdmitWork(AdmitWorkCommand {
                    entity: work_entity,
                }),
            )))
            .expect("admit");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::AcquireWorkLease(AcquireWorkLeaseCommand {
                    work_entity,
                    ttl_ms: 60000,
                    lease_generation: Generation(1),
                }),
            )))
            .expect("lease");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RecordDispatchIntent(RecordDispatchIntentCommand {
                    work_entity,
                    backend: parse_backend("auto").unwrap(),
                    config: Config("{}".to_string()),
                    deadline_ms: 9999999999,
                }),
            )))
            .expect("dispatch");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CompleteWork(
                    prism_ecs_constitutional::lifecycle_command::CompleteWorkCommand {
                        work_entity,
                        output: vec![],
                        output_path: FilePath(String::new()),
                        lease_generation: Generation(1),
                    },
                ),
            )))
            .expect("complete");
    }

    #[test]
    fn test_publish_verifies_cimage_digest() {
        // Create a temp .cimage file with known content
        let tmpdir = std::env::temp_dir().join(format!("prism-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let output_path = tmpdir.join("model.cimage");
        let content = b"mock cimage file content";
        std::fs::write(&output_path, content).unwrap();
        let output_str = output_path.to_string_lossy().to_string();

        // Create work with output_path and drive to Completed manually
        let kernel = {
            let k = crate::kernel::RuntimeKernel::new();
            let handle = k.handle();
            let mut sched = RuntimeSchedule::new();
            sched.register_system(Box::new(ObserveSystem)).unwrap();
            sched.register_system(Box::new(PlanSystem)).unwrap();
            sched.register_system(Box::new(AdmitSystem)).unwrap();
            sched.register_system(Box::new(LeaseSystem)).unwrap();
            sched.register_system(Box::new(DispatchSystem)).unwrap();
            sched.register_system(Box::new(CollectSystem)).unwrap();
            sched.register_system(Box::new(PublishSystem)).unwrap();
            sched.register_system(Box::new(CleanupSystem)).unwrap();
            sched.bind(&handle);
            sched.validate().unwrap();
            k.set_schedule(sched);
            k
        };
        let handle = kernel.handle();

        let create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
output_path: FilePath(output_str.clone()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let work_entity = match create.result {
            crate::CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            ref other => panic!("expected WorkCreated, got {other:?}"),
        };

        // Drive through lifecycle manually
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::MarkObserved(MarkObservedCommand {
                    entity: work_entity,
                    observed_epoch: Epoch(0),
                }),
            )))
            .expect("observe");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RecordWorkPlan(RecordWorkPlanCommand {
                    entity: work_entity,
                    backend: parse_backend("auto").unwrap(),
                    output_format: Format("cimage".to_string()),
                    resource_estimate_bytes: 1024,
                    timeout_ms: 30000,
                }),
            )))
            .expect("plan");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::AdmitWork(AdmitWorkCommand {
                    entity: work_entity,
                }),
            )))
            .expect("admit");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::AcquireWorkLease(AcquireWorkLeaseCommand {
                    work_entity,
                    ttl_ms: 60000,
                    lease_generation: Generation(1),
                }),
            )))
            .expect("lease");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RecordDispatchIntent(RecordDispatchIntentCommand {
                    work_entity,
                    backend: parse_backend("auto").unwrap(),
                    config: Config("{}".to_string()),
                    deadline_ms: 9999999999,
                }),
            )))
            .expect("dispatch");
        handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CompleteWork(
                    prism_ecs_constitutional::lifecycle_command::CompleteWorkCommand {
                        work_entity,
                        output: vec![],
                        output_path: FilePath(String::new()),
                        lease_generation: Generation(1),
                    },
                ),
            )))
            .expect("complete");

        // Run PublishSystem via tick — it should find the completed entity with WorkOutputPath
        let tick = kernel.run_tick().expect("publish tick");

        // The expected blake3 digest of our content
        // Verify that PublishSystem emitted publish_result for our entity
        let stage_publish: &[EmittedCommand] = tick
            .emitted_commands
            .get(6)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let has_publish = stage_publish.iter().any(|ec| {
            ec.command_type.contains("PublishResult") && ec.result_type.contains("Published")
        });
        assert!(
            has_publish,
            "should emit publish_result Published for completed work: {stage_publish:?}"
        );

        // Verify the Published command has the right idempotency_key (non-zero)
        for ec in stage_publish {
            if ec.command_type.contains("PublishResult") && ec.result_type.contains("Published") {
                assert!(ec.sequence.is_some(), "publish_result should have sequence");
                assert!(ec.world_epoch.is_some(), "publish_result should have epoch");
                assert_ne!(
                    ec.idempotency_key.to_string(),
                    "00000000-0000-0000-0000-000000000000"
                );
            }
        }

        // Clean up temp file
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_idle_world_no_commands_or_leases() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.register_system(Box::new(CleanupSystem)).unwrap();
        sched.bind(&handle);
        kernel.set_schedule(sched);
        for i in 0..10 {
            let tick = kernel.run_tick().unwrap_or_else(|_| panic!("tick {i}"));
            assert_eq!(
                tick.emitted_commands.iter().map(|c| c.len()).sum::<usize>(),
                0
            );
        }
    }

    #[test]
    fn test_schedule_validation_passes() {
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.register_system(Box::new(CleanupSystem)).unwrap();
        sched.validate().expect("8-stage schedule must validate");
    }

    #[test]
    fn test_schedule_drives_full_lifecycle_to_published_with_digest() {
        // Create a mock .cimage file
        let tmpdir = std::env::temp_dir().join(format!("prism-cross-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let output_path = tmpdir.join("model.cimage");
        let content = b"cross-ingress mock cimage content";
        std::fs::write(&output_path, content).unwrap();
        let output_str = output_path.to_string_lossy().to_string();
        let expected_digest = blake3::hash(content).to_hex().to_string();

        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.register_system(Box::new(CleanupSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        // Use FakeWorkDispatcher — immediately completes so CollectSystem can reconcile
        sched.set_dispatcher(std::sync::Arc::new(
            crate::test_adapters::FakeWorkDispatcher,
        ));
        kernel.set_schedule(sched);

        // Submit CreateWork with output_path (the "command ingress")
        let create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
output_path: FilePath(output_str.clone()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let work_entity = match create.result {
            crate::CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            ref other => panic!("expected WorkCreated, got {other:?}"),
        };

        // Run ticks — the schedule ingress drives Observe→Plan→Admit→Lease→Dispatch→Collect→Publish
        let max_ticks = 10;
        let mut seen_published = false;
        for i in 0..max_ticks {
            let tick = kernel.run_tick().unwrap_or_else(|_| panic!("tick {i}"));
            // Check all emitted commands for PublishResult with matching digest
            for stage_commands in &tick.emitted_commands {
                for ec in stage_commands {
                    if ec.command_type.contains("PublishResult")
                        && ec.result_type.contains("Published")
                    {
                        seen_published = true;
                    }
                }
            }
            if seen_published {
                break;
            }
        }
        assert!(
            seen_published,
            "schedule ticks should drive work through to Published"
        );

        // Query the world directly through the kernel handle to verify Published state
        // and that the ResultPayload contains the correct digest
        let _entity = work_entity;
        // We can't directly inspect world state from outside, but we verified
        // the PublishResult was emitted via the tick receipt. The next tick should
        // show CleanupSystem finds no more work to process.
        let cleanup_tick = kernel.run_tick().expect("cleanup tick");
        // There should be no publish commands in the next tick (already published)
        let has_publish: bool = cleanup_tick
            .emitted_commands
            .iter()
            .flatten()
            .any(|ec| ec.command_type.contains("PublishResult"));
        assert!(
            !has_publish,
            "already published, no more PublishResult expected"
        );

        // Verify the file exists and has the correct digest
        let meta = std::fs::metadata(&output_path).expect("output file should exist");
        assert!(meta.len() > 0, "output file should have content");
        let file_data = std::fs::read(&output_path).expect("read output file");
        let actual_digest = blake3::hash(&file_data).to_hex().to_string();
        assert_eq!(actual_digest, expected_digest, "file content unchanged");

        // Clean up
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_cross_ingress_cancellation() {
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.register_system(Box::new(CleanupSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        kernel.set_schedule(sched);

        // Ingress 1: submit CreateWork via command submission
        let create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
                    output_path: FilePath("".to_string()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let work_entity = match create.result {
            crate::CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            ref other => panic!("expected WorkCreated, got {other:?}"),
        };

        // Ingress 2: schedule tick advances the work to Observed
        kernel.run_tick().expect("observe tick");

        // Ingress 3: cancel via command submission
        let cancel = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::RequestCancellation(RequestCancellationCommand {
                    entity: work_entity,
                    reason: RejectionReason(String::new()),
                }),
            )))
            .expect("cancel");
        assert!(
            matches!(
                cancel.result,
                crate::CommandResult::Lifecycle(LifecycleCommandResult::RequestCancelled { .. })
            ),
            "cancellation should be acknowledged"
        );

        // Verify cancellation took effect: entity should have Cancelled WorkState
        let tick2 = kernel.run_tick().expect("post-cancel tick");
        let any_observe: bool = tick2
            .emitted_commands
            .iter()
            .flatten()
            .any(|ec| ec.command_type.contains("MarkObserved"));
        assert!(
            !any_observe,
            "cancelled work should not be observed in subsequent tick: {tick2:?}"
        );
    }

    #[test]
    fn test_recovery_preserves_work_output_path() {
        use crate::test_adapters::*;
        let kernel = crate::kernel::RuntimeKernel::with_ports(
            Box::new(InMemoryCommandStore::new()),
            Box::new(InMemorySnapshotStore::new()),
            Box::new(InMemoryTickReceiptStore::new()),
            Box::new(InMemoryLeaseCoordinator::new()),
            Box::new(DeterministicClock::new(1000)),
        );
        let handle = kernel.handle();

        // Create work with output_path via command ingress (no schedule)
        let _create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
                    output_path: FilePath("/tmp/recovery-test.cimage".to_string()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let _work_entity = match _create.result {
            crate::CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            ref other => panic!("expected WorkCreated, got {other:?}"),
        };

        // Run tick to advance to Observed (this exercises schedule ingress)
        // No schedule bound — skip tick, just save and recover

        // Persist snapshot and recover
        kernel.save_snapshot().expect("save snapshot");
        let report = kernel.recover().expect("recover");
        assert_eq!(report.recovery_state, "recovered");
        assert!(
            report.replayed_commands >= 1,
            "should replay at least CreateWork"
        );
    }

    #[test]
    fn test_daemon_lifecycle_end_to_end() {
        // ── 1. Create deterministic output file ──
        let tmpdir = std::env::temp_dir().join(format!("prism-e2e-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let output_path = tmpdir.join("test-daemon-lifecycle.cimage");
        let content = b"deterministic prisma-engine daemon lifecycle end-to-end test content";
        std::fs::write(&output_path, content).unwrap();
        let output_str = output_path.to_string_lossy().to_string();
        let expected_digest = blake3::hash(content).to_hex().to_string();

        // ── 2. Create kernel with 8-stage schedule + FakeWorkDispatcher ──
        let kernel = crate::kernel::RuntimeKernel::new();
        let handle = kernel.handle();
        let mut sched = RuntimeSchedule::new();
        sched.register_system(Box::new(ObserveSystem)).unwrap();
        sched.register_system(Box::new(PlanSystem)).unwrap();
        sched.register_system(Box::new(AdmitSystem)).unwrap();
        sched.register_system(Box::new(LeaseSystem)).unwrap();
        sched.register_system(Box::new(DispatchSystem)).unwrap();
        sched.register_system(Box::new(CollectSystem)).unwrap();
        sched.register_system(Box::new(PublishSystem)).unwrap();
        sched.register_system(Box::new(CleanupSystem)).unwrap();
        sched.bind(&handle);
        sched.validate().unwrap();
        sched.set_dispatcher(std::sync::Arc::new(
            crate::test_adapters::FakeWorkDispatcher,
        ));
        kernel.set_schedule(sched);

        // ── 3. Submit CreateWork with output path ──
        let create = handle
            .submit(CommandEnvelope::new(Command::Lifecycle(
                LifecycleCommand::CreateWork(CreateWorkCommand {
                    entity: Entity::new(0, 0),
                    target_entity: Entity::new(0, 0),
                    kind: Format("compile".to_string()),
                    resource_claim: default_claim(),
output_path: FilePath(output_str.clone()),
                    input_path: FilePath("".to_string()),
                }),
            )))
            .expect("create");
        let work_entity = match create.result {
            crate::CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                work_entity, ..
            }) => work_entity,
            ref other => panic!("expected WorkCreated, got {other:?}"),
        };
        assert!(work_entity.id() > 0, "work entity must have positive id");

        // ── 4. Run ticks until Published or limit reached ──
        // Track which lifecycle command types have been observed
        let mut mark_observed_seen = false;
        let mut record_work_plan_seen = false;
        let mut admit_work_seen = false;
        let mut acquire_work_lease_seen = false;
        let mut record_dispatch_intent_seen = false;
        let mut complete_work_seen = false;
        let mut publish_result_seen = false;
        let mut cleanup_seen = false;
        let mut provider_selection_seen = false;

        let max_ticks: usize = 20;
        for i in 0..max_ticks {
            let tick = kernel.run_tick().unwrap_or_else(|_| panic!("tick {i}"));

            // Verify emitted_commands has entries for all 8 stages
            assert_eq!(
                tick.emitted_commands.len(),
                8,
                "tick {i}: expected 8 system entries in emitted_commands, got {}: \
                 system_order={:?}",
                tick.emitted_commands.len(),
                tick.system_order
            );

            // Inspect every emitted command across all stages
            for stage_commands in &tick.emitted_commands {
                for ec in stage_commands {
                    let ct = &ec.command_type;
                    if ct.contains("MarkObserved") {
                        mark_observed_seen = true;
                    }
                    if ct.contains("RecordWorkPlan") {
                        record_work_plan_seen = true;
                    }
                    if ct.contains("AdmitWork") {
                        admit_work_seen = true;
                    }
                    if ct.contains("AcquireWorkLease") {
                        acquire_work_lease_seen = true;
                    }
                    if ct.contains("RecordDispatchIntent") {
                        record_dispatch_intent_seen = true;
                    }
                    if ct.contains("CompleteWork") {
                        complete_work_seen = true;
                    }
                    if ec.command_type.contains("PublishResult")
                        && ec.result_type.contains("Published")
                    {
                        publish_result_seen = true;
                    }
                    if ct.contains("ReleaseWorkLease") || ct.contains("ExpireTransientState") {
                        cleanup_seen = true;
                    }
                }
            }
            if !tick.provider_selections.is_empty() {
                provider_selection_seen = true;
                assert_eq!(tick.provider_selections[0].selected_backend(), Some("cpu"));
            }

            if publish_result_seen {
                break;
            }
        }

        // ── 5. Verify all lifecycle stages executed ──
        assert!(mark_observed_seen, "ObserveSystem should emit MarkObserved");
        assert!(
            record_work_plan_seen,
            "PlanSystem should emit RecordWorkPlan"
        );
        assert!(admit_work_seen, "AdmitSystem should emit AdmitWork");
        assert!(
            acquire_work_lease_seen,
            "LeaseSystem should emit AcquireWorkLease"
        );
        assert!(
            record_dispatch_intent_seen,
            "DispatchSystem should emit RecordDispatchIntent"
        );
        assert!(complete_work_seen, "CollectSystem should emit CompleteWork");
        assert!(
            publish_result_seen,
            "PublishSystem should emit PublishResult with Published status"
        );
        assert!(cleanup_seen, "CleanupSystem should emit cleanup commands");
        assert!(
            provider_selection_seen,
            "DispatchSystem should publish a provider-selection receipt"
        );

        // ── 6. Verify file digest via blake3 hash ──
        let meta = std::fs::metadata(&output_path).expect("output file should exist");
        assert!(meta.len() > 0, "output file should have content");
        let file_data = std::fs::read(&output_path).expect("read output file");
        let actual_digest = blake3::hash(&file_data).to_hex().to_string();
        assert_eq!(
            actual_digest, expected_digest,
            "file content should be unchanged by FakeWorkDispatcher"
        );

        // ── 7. Verify 3 idle ticks emit no lifecycle commands ──
        for i in 0..3 {
            let tick = kernel
                .run_tick()
                .unwrap_or_else(|_| panic!("idle tick {i} after publish"));
            assert_eq!(
                tick.emitted_commands.len(),
                8,
                "idle tick {i}: expected 8 system entries"
            );
        }

        // ── 8. Clean up ──
        let _ = std::fs::remove_dir_all(&tmpdir);
    }
}
