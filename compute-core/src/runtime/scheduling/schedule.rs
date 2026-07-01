//! Schedule compiler and executor.
//!
//! The eight-step compilation procedure:
//! 1. Validate uniqueness of IDs and names.
//! 2. Barrier-group systems by stage.
//! 3. Inject explicit `after` and `before` edges.
//! 4. Detect write/write overlaps across components and resources.
//! 5. Enforce serialization policy — reject undeclared hazards.
//! 6. Resolve legal hazards (`StableOrder` policy).
//! 7. Topological sort via Kahn with deterministic priority queue.
//! 8. Emit canonical `ScheduleManifest`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::error::ScheduleError;
use crate::runtime::scheduling::graph::{EdgeKind, GraphBuilder};
use crate::runtime::scheduling::manifest::{
    ManifestBuilder, ManifestWarningKind, ScheduleManifest,
};
use crate::runtime::scheduling::metadata::{
    ErasedSystem, Stage, SystemId, SystemMetadata, SystemResult,
    SerializationPolicy,
};
use crate::runtime::world::World;
use crate::runtime::ledger::{
    DeterministicReceiptPayload, ReceiptHasher,
    ComponentTypeRegistry, SemanticReceipt, SemanticStampedCommand,
    TransitionLedgerResource, TransitionReceipt,
};
use crate::runtime::ledger::entry::TRANSITION_RECEIPT_SCHEMA_VERSION;
use crate::runtime::ledger::error::LedgerProjectionError;

// ---------------------------------------------------------------------------
// PriorityKey — deterministic ready-queue ordering
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ScheduleCommitError {
    Validation(String),
    LedgerProjection(LedgerProjectionError),
    Apply(String),
}

impl std::fmt::Display for ScheduleCommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "command batch validation failed: {msg}"),
            Self::LedgerProjection(e) => write!(f, "ledger projection failed: {e}"),
            Self::Apply(msg) => write!(f, "command application failed: {msg}"),
        }
    }
}

impl std::error::Error for ScheduleCommitError {}

// ---------------------------------------------------------------------------
// CommandOrderKey — deterministic sorting key for commands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommandOrderKey {
    pub stage_rank: u16,
    pub system_id: u32,
    pub entity: u32,
    pub entity_generation: u64,
    pub sequence: u64,
}

// ---------------------------------------------------------------------------
// PriorityKey — deterministic ready-queue ordering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PriorityKey {
    stage: u8,
    order: i32,
    system_id: u32,
}

fn priority_key(s: &SystemMetadata) -> PriorityKey {
    PriorityKey {
        stage: s.stage as u8,
        order: s.order,
        system_id: s.id.0,
    }
}

// ---------------------------------------------------------------------------
// MicrocycleConfig
// ---------------------------------------------------------------------------

/// Configuration for a post-stage Maintenance microcycle.
///
/// After Stage::Intake commits, if the event signal is set, the scheduler
/// runs one bounded pass through the designated Maintenance systems.
/// At most one microcycle per frame, no recursive re-entry.
pub struct MicrocycleConfig {
    /// Index (in the systems vec) of the first Maintenance system to run.
    pub maintenance_start: usize,
    /// Index (exclusive) of the last Maintenance system to run.
    pub maintenance_end: usize,
    /// Signal checked after Intake commit: true = one microcycle requested.
    pub event_signal: Arc<AtomicBool>,
    /// Whether the microcycle has already been taken this frame.
    pub microcycle_taken: bool,
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// A compiled, immutable schedule that can be executed repeatedly.
///
/// Once constructed via `Schedule::compile`, the schedule's execution order
/// is fixed and its manifest is hashable for receipt correlation.
pub struct Schedule {
    /// Systems in execution order.
    systems: Vec<Box<dyn ErasedSystem>>,
    /// Metadata for each system (index-aligned with `systems`).
    /// Owned so we avoid lifetime gymnastics with the borrowed metadata
    /// from the ErasedSystem trait.
    metadata: Vec<SystemMetadata>,
    /// The compiled manifest.
    manifest: ScheduleManifest,
    /// Pre-allocated command buffer for each stage.
    /// Indexed by stage discriminant.
    command_buffers: Vec<Vec<crate::runtime::scheduling::command::StampedCommand>>,
    /// Optional ledger for semantic receipt recording.
    ledger: Option<TransitionLedgerResource>,
    /// Optional post-Intake Maintenance microcycle configuration.
    pub microcycle: Option<MicrocycleConfig>,
}

impl std::fmt::Debug for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Schedule")
            .field("system_count", &self.systems.len())
            .field("manifest", &self.manifest)
            .field("ledger_installed", &self.ledger.is_some())
            .finish_non_exhaustive()
    }
}

impl Schedule {
    /// Number of systems in the schedule.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// The compiled manifest.
    pub fn manifest(&self) -> &ScheduleManifest {
        &self.manifest
    }

    /// Attach a `MicrocycleConfig` to this schedule.
    pub fn set_microcycle(&mut self, config: MicrocycleConfig) {
        self.microcycle = Some(config);
    }

    /// Reset the microcycle signal after a `run()` cycle.
    ///
    /// Clears the taken flag and resets the atomic signal so the microcycle
    /// can be requested again on the next frame.
    pub fn reset_microcycle_signal(&mut self) {
        if let Some(ref mut mc) = self.microcycle {
            mc.microcycle_taken = false;
            mc.event_signal.store(false, Ordering::Release);
        }
    }

    /// Attach a transition ledger resource to this schedule.
    pub fn set_ledger(&mut self, ledger: TransitionLedgerResource) {
        self.ledger = Some(ledger);
    }

    // ═════════════════════════════════════════════════════════════════════
    //  compile
    // ═════════════════════════════════════════════════════════════════════

    /// Compile a flat list of systems into a validated, sorted `Schedule`.
    ///
    /// ## Eight-step procedure
    ///
    /// 1. **Validate** — reject duplicate IDs and names.
    /// 2. **Barrier-group** — partition systems by `Stage`.
    /// 3. **Explicit edges** — inject `after` and `before` declarations.
    /// 4. **Detect hazards** — find write/write overlaps.
    /// 5. **Enforce policy** — reject undeclared hazards.
    /// 6. **Resolve legal** — apply `StableOrder` fallback edges.
    /// 7. **Topological sort** — Kahn with deterministic priority.
    /// 8. **Emit manifest** — build hashable `ScheduleManifest`.
    pub fn compile(
        systems: Vec<Box<dyn ErasedSystem>>,
    ) -> Result<Self, ScheduleError> {
        // ── Step 0: extract metadata ───────────────────────────────────
        let metadata: Vec<SystemMetadata> =
            systems.iter().map(|s| s.metadata().clone()).collect();

        // ── Step 1: validate uniqueness ─────────────────────────────────
        let mut id_set: HashSet<SystemId> = HashSet::new();
        let mut name_set: HashSet<&'static str> = HashSet::new();
        for meta in &metadata {
            if !id_set.insert(meta.id) {
                return Err(ScheduleError::SystemIdCollision(meta.id));
            }
            if !name_set.insert(meta.name) {
                return Err(ScheduleError::SystemNameCollision(meta.name));
            }
        }

        // ── Step 2: barrier-group by stage ─────────────────────────────
        // Systems are already grouped logically by their Stage discriminant.
        // We record stage barriers for the manifest.

        let mut manifest_builder = ManifestBuilder::new();

        let mut stage_groups: HashMap<Stage, Vec<usize>> = HashMap::new();
        for (i, meta) in metadata.iter().enumerate() {
            stage_groups.entry(meta.stage).or_default().push(i);
        }

        // Detect stage barrier positions: the last index of each stage.
        // We'll emit edges between the last system of stage N and the
        // first system of stage N+1 during graph building.

        // ── Step 3: inject explicit edges ───────────────────────────────
        let mut builder = GraphBuilder::new(metadata.clone());

        for meta in &metadata {
            for &after_id in meta.after {
                builder.add_explicit_after(meta.id, after_id)?;
            }
            for &before_id in meta.before {
                builder.add_explicit_before(meta.id, before_id)?;
            }
        }

        // ── Step 4: detect write/write overlaps ─────────────────────────
        // Two systems have a hazard when they write the same component or
        // resource.

        let mut hazards: Vec<(usize, usize, &'static str)> = Vec::new();

        for i in 0..metadata.len() {
            for j in (i + 1)..metadata.len() {
                let a = &metadata[i];
                let b = &metadata[j];

                let comp_overlap = a.writes.overlaps(&b.writes);
                let res_overlap = a.writes_resources.overlaps(&b.writes_resources);

                if comp_overlap {
                    hazards.push((i, j, "component write/write overlap"));
                }
                if res_overlap {
                    hazards.push((i, j, "resource write/write overlap"));
                }
            }
        }

        // ── Step 5: enforce serialization policy ────────────────────────
        let mut pending_hazards: Vec<(usize, usize)> = Vec::new();

        for &(i, j, reason) in &hazards {
            let a = &metadata[i];
            let b = &metadata[j];

            // Check if an explicit edge already resolves this.
            let has_explicit = a
                .after
                .iter()
                .any(|id| *id == b.id)
                || a.before.iter().any(|id| *id == b.id)
                || b.after.iter().any(|id| *id == a.id)
                || b.before.iter().any(|id| *id == a.id);

            if has_explicit {
                manifest_builder.record_hazard(
                    a.id,
                    b.id,
                    reason,
                    true,
                    "resolved by explicit edge",
                );
                continue;
            }

            match (a.serialization, b.serialization) {
                (SerializationPolicy::Commutative, SerializationPolicy::Commutative) => {
                    manifest_builder.record_hazard(
                        a.id,
                        b.id,
                        reason,
                        true,
                        "commutative — no edge required",
                    );
                    manifest_builder.record_warning(
                        ManifestWarningKind::CommutativeHazard,
                        format!(
                            "commutative hazard between {} and {}",
                            a.name, b.name
                        ),
                    );
                }
                (SerializationPolicy::Reject, _)
                | (_, SerializationPolicy::Reject) => {
                    return Err(ScheduleError::IllegalHazard {
                        system_a: a.id,
                        system_b: b.id,
                        reason,
                    });
                }
                (SerializationPolicy::ExplicitOnly, _)
                | (_, SerializationPolicy::ExplicitOnly) => {
                    return Err(ScheduleError::IllegalHazard {
                        system_a: a.id,
                        system_b: b.id,
                        reason,
                    });
                }
                // StableOrder → resolved in step 6.
                _ => {
                    pending_hazards.push((i, j));
                }
            }
        }

        // ── Step 6: resolve legal hazards ───────────────────────────────
        for &(i, j) in &pending_hazards {
            let a = &metadata[i];
            let b = &metadata[j];

            let key_a = priority_key(a);
            let key_b = priority_key(b);

            if key_a <= key_b {
                builder.add_serialization_edge(a.id, b.id);
            } else {
                builder.add_serialization_edge(b.id, a.id);
            }

            manifest_builder.record_hazard(
                a.id,
                b.id,
                "write/write overlap resolved by stable order",
                true,
                "deterministic stable order",
            );
        }

        // ── Stage barrier edges ─────────────────────────────────────────
        // Emit edges between consecutive non-empty stages.
        // This correctly handles gaps where intermediate stages have no
        // systems — e.g. Decode → Receipt when PostDecode, ToolExecution,
        // and Maintenance are empty.
        let mut stage_barriers: Vec<(SystemId, SystemId)> = Vec::new();
        let non_empty_stages: Vec<Stage> = Stage::ALL
            .iter()
            .copied()
            .filter(|s| stage_groups.contains_key(s))
            .collect();
        for pair in non_empty_stages.windows(2) {
            let stage_a = pair[0];
            let stage_b = pair[1];
            let group_a = &stage_groups[&stage_a];
            let group_b = &stage_groups[&stage_b];

            // Last system in stage_a by (order, id).
            let last_in_a = group_a
                .iter()
                .max_by(|&&i, &&j| {
                    let a = &metadata[i];
                    let b = &metadata[j];
                    a.order
                        .cmp(&b.order)
                        .then_with(|| a.id.cmp(&b.id))
                })
                .unwrap();
            // First system in stage_b by (order, id).
            let first_in_b = group_b
                .iter()
                .min_by(|&&i, &&j| {
                    let a = &metadata[i];
                    let b = &metadata[j];
                    a.order
                        .cmp(&b.order)
                        .then_with(|| a.id.cmp(&b.id))
                })
                .unwrap();

            let last_id = metadata[*last_in_a].id;
            let first_id = metadata[*first_in_b].id;

            builder.add_explicit_after(first_id, last_id)?;
            stage_barriers.push((last_id, first_id));
        }

        // Record stage barrier edges in the manifest.
        for &(last, first) in &stage_barriers {
            manifest_builder.record_edge(last, first, EdgeKind::StageBarrier);
        }

        // ── Step 7: topological sort ────────────────────────────────────
        let graph = builder.build().map_err(|e| match e {
            ScheduleError::CycleDetected(_) => e,
            other => other,
        })?;
        let order = graph.topological_order().map_err(|cycle_nodes| {
            // Build a cycle path for diagnostics.
            ScheduleError::CycleDetected(cycle_nodes)
        })?;

        // Reorder systems to match the topological order.
        let mut system_map: HashMap<SystemId, Box<dyn ErasedSystem>> = HashMap::new();
        let mut meta_map: HashMap<SystemId, &SystemMetadata> = HashMap::new();
        for (s, m) in systems.into_iter().zip(metadata.iter()) {
            system_map.insert(m.id, s);
            meta_map.insert(m.id, m);
        }

        let mut reordered: Vec<Box<dyn ErasedSystem>> = Vec::with_capacity(order.len());
        let mut reordered_meta: Vec<SystemMetadata> = Vec::with_capacity(order.len());
        for id in &order {
            reordered.push(system_map.remove(id).unwrap());
            reordered_meta.push(meta_map.remove(id).unwrap().clone());
        }

        // ── Step 8: emit manifest ──────────────────────────────────────
        let manifest = manifest_builder.build(order);

        // Pre-allocate command buffers (one per stage).
        let num_stages = Stage::ALL.len();
        let command_buffers = (0..num_stages).map(|_| Vec::new()).collect();

        Ok(Schedule {
            systems: reordered,
            metadata: reordered_meta,
            manifest,
            command_buffers,
            ledger: None,
            microcycle: None,
        })
    }

    // ═════════════════════════════════════════════════════════════════════
    //  commit_stage_commands
    // ═════════════════════════════════════════════════════════════════════

    /// Sort, validate, project, hash, apply, and commit a stage's command
    /// buffer to the transition ledger.
    fn commit_stage_commands(
        &mut self,
        stage: Stage,
        scheduler_epoch: u64,
        microcycle: u32,
        world: &mut World,
    ) -> Result<(), ScheduleCommitError> {
        let ledger = self.ledger.as_ref().ok_or_else(|| {
            ScheduleCommitError::Apply("no ledger installed".into())
        })?;

        let stage_idx = stage as usize;
        let buffer = &self.command_buffers[stage_idx];
        if buffer.is_empty() {
            return Ok(());
        }

        // 1. Sort commands deterministically
        let mut commands = buffer.clone();
        commands.sort_by(|a, b| {
            CommandOrderKey {
                stage_rank: stage as u16,
                system_id: a.system_id.0,
                entity: a.entity.map(|e| e.0).unwrap_or(u32::MAX),
                entity_generation: 0,
                sequence: a.sequence,
            }
            .cmp(&CommandOrderKey {
                stage_rank: stage as u16,
                system_id: b.system_id.0,
                entity: b.entity.map(|e| e.0).unwrap_or(u32::MAX),
                entity_generation: 0,
                sequence: b.sequence,
            })
        });

        // 2. Validate complete batch
        for cmd in &commands {
            if let Some(entity) = cmd.entity {
                if !world.is_alive(entity) {
                    return Err(ScheduleCommitError::Validation(
                        format!("entity {} not alive for command seq {}", entity.0, cmd.sequence)
                    ));
                }
            }
        }

        // 3. Project semantic receipts
        let registry = ComponentTypeRegistry::new_core();
        let semantic_commands: Vec<SemanticStampedCommand> = commands
            .iter()
            .map(|cmd| cmd.semantic_receipt(&registry))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ScheduleCommitError::LedgerProjection)?;

        // 4. Build deterministic payload and hash
        let payload = DeterministicReceiptPayload {
            schema_version: TRANSITION_RECEIPT_SCHEMA_VERSION,
            receipt_sequence: ledger.next_receipt_sequence(),
            scheduler_epoch,
            microcycle,
            stage,
            command_count: commands.len() as u32,
            commands: semantic_commands,
        };
        let hasher = ReceiptHasher::production();
        let deterministic_digest = hasher.compute(&payload)
            .map_err(|e| ScheduleCommitError::Apply(e.to_string()))?;

        // 5. Apply commands to World
        Self::apply_command_buffer(world, &commands);

        // 6. Build receipt with optional timestamp
        let receipt = TransitionReceipt {
            payload,
            deterministic_digest,
            observed_at_ns: None,
        };

        // 7. Commit to ledger
        ledger.commit(receipt);

        // 8. Clear buffer
        self.command_buffers[stage_idx].clear();

        Ok(())
    }

    // ═════════════════════════════════════════════════════════════════════
    //  run
    // ═════════════════════════════════════════════════════════════════════

    /// Execute all systems in compiled order.
    ///
    /// Groups systems by stage.  After each stage completes, the command
    /// buffer is drained and applied deterministically — sorted by
    /// (system_id, entity_id, sequence).
    ///
    /// Returns execution results for each system.
    pub fn run(
        &mut self,
        world: &mut World,
    ) -> Vec<(SystemId, SystemResult)> {
        let mut results = Vec::with_capacity(self.systems.len());
        let scheduler_epoch: u64 = 1;
        // Track the current stage — when it changes, drain the previous
        // stage's buffer.
        let mut previous_stage: Option<Stage> = None;

        for idx in 0..self.systems.len() {
            let stage_idx = self.metadata[idx].stage as usize;
            let meta_stage = self.metadata[idx].stage;
            let meta_id = self.metadata[idx].id;

            // Stage boundary: process commands through ledger.
            if let Some(prev) = previous_stage {
                if prev != meta_stage && stage_idx < self.command_buffers.len() {
                    let _ = self.commit_stage_commands(prev, scheduler_epoch, 0, world);

                    // After Intake commits, check for a requested microcycle.
                    if prev == Stage::Intake {
                        if let Some(ref mut mc) = self.microcycle {
                            if mc.event_signal.load(Ordering::Acquire) && !mc.microcycle_taken {
                                // Bounded maintenance pass — one shot, no re-entry.
                                let mut micro_buffer: Vec<
                                    crate::runtime::scheduling::command::StampedCommand,
                                > = Vec::new();
                                for i in mc.maintenance_start..mc.maintenance_end {
                                    if i < self.systems.len() {
                                        let mut writer = CommandWriter::new(
                                            &mut micro_buffer,
                                            Stage::Maintenance,
                                            self.metadata[i].id,
                                        );
                                        self.systems[i].run(world, &mut writer);
                                    }
                                }
                                Self::apply_command_buffer(world, &micro_buffer);
                                mc.microcycle_taken = true;
                            }
                        }
                    }
                }
            }
            previous_stage = Some(meta_stage);

            // Create the command writer for this system.
            {
                let mut writer: CommandWriter<'_> =
                    CommandWriter::new(&mut self.command_buffers[stage_idx], meta_stage, meta_id);
                let result = self.systems[idx].run(world, &mut writer);
                results.push((meta_id, result));
            }
        }

        // Drain the final stage's buffer through ledger.
        if let Some(last_stage) = previous_stage {
            let _ = self.commit_stage_commands(last_stage, scheduler_epoch, 0, world);
        }

        // Reset the microcycle signal for the next frame.
        self.reset_microcycle_signal();

        results
    }

    /// Apply a stage's command buffer to the World.
    ///
    /// Commands are sorted deterministically by (system_id, entity, sequence)
    /// before application.
    fn apply_command_buffer(
        world: &mut World,
        buffer: &[crate::runtime::scheduling::command::StampedCommand],
    ) {
        if buffer.is_empty() {
            return;
        }

        // Sort a copy deterministically.
        let mut sorted: Vec<_> = buffer.to_vec();
        sorted.sort_by(|a, b| {
            a.system_id
                .0
                .cmp(&b.system_id.0)
                .then_with(|| {
                    let ea = a.entity.map(|e| e.0).unwrap_or(u32::MAX);
                    let eb = b.entity.map(|e| e.0).unwrap_or(u32::MAX);
                    ea.cmp(&eb)
                })
                .then_with(|| a.sequence.cmp(&b.sequence))
        });

        for cmd in &sorted {
            match &cmd.command {
                crate::runtime::scheduling::command::Command::Spawn => {
                    world.spawn();
                }
                crate::runtime::scheduling::command::Command::Despawn(entity) => {
                    world.despawn(*entity);
                }
                crate::runtime::scheduling::command::Command::Insert {
                    entity,
                    type_id,
                    payload,
                } => {
                    let _ = world.insert_raw(*entity, *type_id, payload);
                }
                crate::runtime::scheduling::command::Command::Remove {
                    entity,
                    type_id,
                } => {
                    let _ = (entity, type_id);
                }
            }
        }
    }
}
