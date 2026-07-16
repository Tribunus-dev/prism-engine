//! SPU-style workgroup scheduling — RPCS3 Cell/B.E. SPU workgroup abstraction.
//!
//! Maps the lv2_spu_group concept onto ECS components: a [`WorkGroup`] entity
//! owns a pool of SPU threads, and each thread becomes a [`WorkUnit`] entity
//! that tracks execution budget, core assignment, and completion.
//!
//! The [`WorkGroupSystem`] resource manages the lifecycle: creating groups,
//! distributing units across available lanes, and tracking completion.
//!
//! # Components
//!
//! * [`WorkGroup`] — a group of work units (analogue of lv2_spu_group).
//! * [`WorkUnit`] — a single schedulable unit of work.
//! * [`WorkGroupStatus`] — lifecycle state of a work group.
//! * [`SchedulerPolicy`] — scheduling discipline for work distribution.
//!
//! # Systems
//!
//! * [`WorkGroupSystem`] — ECS resource that creates groups, distributes units,
//!   and processes completions.

use prism_ecs_core::{Component, Entity, EntityKind, World, WorldError};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SchedulerPolicy
// ---------------------------------------------------------------------------

/// Scheduling discipline for work distribution within a work group.
///
/// Analogous to RPCS3's SPU scheduler policy: each SPU thread group can
/// be pinned to specific cores, spread round-robin, or managed as a
/// single gang that must start together on contiguous cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchedulerPolicy {
    /// Pinned to a specific core — the work group runs on a designated host thread.
    Pinned { core: u32 },
    /// Round-robin distribution across available lanes.
    RoundRobin,
    /// Gang scheduling — all work units in the group must start together
    /// on contiguous cores.  Common in Cell SPU tight loops where threads
    /// synchronise through raw DMA.
    Gang,
    /// Default policy — distributes work to the first available lane.
    Default,
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        SchedulerPolicy::Default
    }
}

// ---------------------------------------------------------------------------
// WorkGroupStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a work group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkGroupStatus {
    /// Group has been created but not yet scheduled.
    Pending,
    /// Group is actively executing on assigned cores.
    Running,
    /// All work units in the group have completed.
    Completed,
    /// Group was cancelled before completion.
    Cancelled,
}

impl Default for WorkGroupStatus {
    fn default() -> Self {
        WorkGroupStatus::Pending
    }
}

// ---------------------------------------------------------------------------
// WorkGroup component
// ---------------------------------------------------------------------------

/// A group of work units sharing a scheduling policy and lifecycle.
///
/// Analogous to RPCS3's `lv2_spu_group`: a collection of SPU threads that
/// are created, scheduled, and destroyed together.  The group tracks its
/// constituent work unit entities, the scheduling policy used to distribute
/// them, and the overall lifecycle status.
///
/// Every work group entity carries this component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkGroup {
    /// Number of worker threads / work units in this group.
    pub worker_count: usize,
    /// Entities of the work units belonging to this group.
    pub entity_list: Vec<Entity>,
    /// Scheduling policy for distributing work units.
    pub scheduler_policy: SchedulerPolicy,
    /// Current lifecycle status.
    pub status: WorkGroupStatus,
}

impl Component for WorkGroup {}

impl WorkGroup {
    /// Create a new work group component with the given worker count and policy.
    pub fn new(worker_count: usize, scheduler_policy: SchedulerPolicy) -> Self {
        Self {
            worker_count,
            entity_list: Vec::with_capacity(worker_count),
            scheduler_policy,
            status: WorkGroupStatus::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// WorkUnit component
// ---------------------------------------------------------------------------

/// A single schedulable unit of work within a work group.
///
/// Analogous to one SPU thread in an RPCS3 `spu_group`: each work unit has
/// a budget of execution steps, a core assignment, and an optional completion
/// event entity that is signalled when the unit finishes.
///
/// Every work unit entity carries this component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    /// The work group entity that owns this unit.
    pub parent_group: Entity,
    /// Number of execution steps / instructions budgeted for this unit.
    /// When the budget reaches zero the unit is considered complete.
    pub execution_budget: u64,
    /// The assigned core (lane / execution slot) for this unit.
    pub assigned_core: Option<u32>,
    /// Optional fence/event entity to signal on completion.
    /// Downstream executor systems wait on this before reclaiming resources.
    pub completion_event: Option<Entity>,
}

impl Component for WorkUnit {}

impl WorkUnit {
    /// Create a new work unit belonging to `parent_group` with the given budget.
    pub fn new(parent_group: Entity, execution_budget: u64) -> Self {
        Self {
            parent_group,
            execution_budget,
            assigned_core: None,
            completion_event: None,
        }
    }

    /// Returns `true` if this work unit has exhausted its execution budget.
    pub fn is_exhausted(&self) -> bool {
        self.execution_budget == 0
    }

    /// Consume one unit of the execution budget. Returns the remaining budget.
    pub fn tick(&mut self) -> u64 {
        self.execution_budget = self.execution_budget.saturating_sub(1);
        self.execution_budget
    }
}

// ---------------------------------------------------------------------------
// WorkGroupSystem
// ---------------------------------------------------------------------------

/// ECS resource for SPU-style workgroup scheduling.
///
/// Manages the lifecycle of [`WorkGroup`] and [`WorkUnit`] entities:
///
/// * [`create_group`](Self::create_group) — spawns a new work group with the
///   requested number of worker entities, each carrying a [`WorkUnit`]
///   component.
/// * [`distribute`](Self::distribute) — assigns pending work units to
///   execution lanes according to the group's [`SchedulerPolicy`].
/// * [`complete_unit`](Self::complete_unit) — marks a work unit as finished
///   and checks whether the parent group is fully completed.
/// * [`group_status`](Self::group_status) — reads the current lifecycle
///   status of a group entity.
///
/// This is a lightweight coordinator — lane state and pending queues are
/// stored on entity components in the world, not inside this struct.
#[derive(Debug, Default)]
pub struct WorkGroupSystem {
    /// Total number of execution cores / lanes available for assignment.
    pub available_cores: u32,
    /// Round-robin cursor for the RoundRobin policy.
    rr_cursor: u32,
}

impl WorkGroupSystem {
    /// Create a new work group system with a given number of available cores.
    pub fn new(available_cores: u32) -> Self {
        Self {
            available_cores,
            rr_cursor: 0,
        }
    }

    /// Spawn a new work group entity in the world with `count` work units.
    ///
    /// Each work unit is spawned as a separate entity carrying a [`WorkUnit`]
    /// component linked back to the group.  The group entity itself carries
    /// [`WorkGroup`] with the given scheduling policy.
    ///
    /// Returns the work group entity handle.
    pub fn create_group(
        &mut self,
        world: &mut World,
        count: usize,
        policy: SchedulerPolicy,
        budget: u64,
    ) -> Result<Entity, WorldError> {
        // ── Spawn the group entity ──────────────────────────────────────
        let group: Entity = world
            .spawn(
                EntityKind::WorkGroup,
                Some(format!("workgroup_{}", world.entity_count())),
            )?
            .into();
        let group_component = WorkGroup::new(count, policy);
        world.add_component(group, group_component.clone())?;

        // ── Spawn work units ────────────────────────────────────────────
        for i in 0..count {
            let unit: Entity = world
                .spawn(
                    EntityKind::WorkUnit,
                    Some(format!("workunit_{}_{}", group.id(), i)),
                )?
                .into();
            let unit_component = WorkUnit::new(group, budget);
            world.add_component(unit, unit_component)?;
            // Track in the group's entity list — must re-borrow after each insert
            if let Some(gc) = world.get_component_mut::<WorkGroup>(group) {
                gc.entity_list.push(unit);
            }
        }

        Ok(group)
    }

    /// Distribute pending work units across execution cores.
    ///
    /// Walks every group marked as [`WorkGroupStatus::Pending`] and assigns
    /// each of its work units to a core number based on the group's
    /// [`SchedulerPolicy`].
    ///
    /// Returns the total number of work units assigned.
    pub fn distribute(&mut self, world: &mut World) -> usize {
        let mut assigned = 0usize;

        // Collect group entities first to avoid borrow conflicts
        let group_entities: Vec<Entity> = world.query::<WorkGroup>().map(|(e, _)| e).collect();

        for group_entity in group_entities {
            let policy = world
                .get_component::<WorkGroup>(group_entity)
                .map(|g| g.scheduler_policy)
                .unwrap_or(SchedulerPolicy::Default);

            let is_pending = world
                .get_component::<WorkGroup>(group_entity)
                .is_some_and(|g| g.status == WorkGroupStatus::Pending);

            if !is_pending {
                continue;
            }

            let unit_entities: Vec<Entity> = {
                let group = world.get_component::<WorkGroup>(group_entity);
                match group {
                    Some(g) => g.entity_list.clone(),
                    None => continue,
                }
            };

            if self.available_cores == 0 {
                break;
            }

            match policy {
                SchedulerPolicy::Pinned { core } => {
                    for unit_entity in &unit_entities {
                        if let Some(unit) = world.get_component_mut::<WorkUnit>(*unit_entity) {
                            unit.assigned_core = Some(core % self.available_cores);
                            assigned += 1;
                        }
                    }
                }
                SchedulerPolicy::RoundRobin => {
                    for unit_entity in &unit_entities {
                        let core = self.rr_cursor % self.available_cores;
                        self.rr_cursor = self.rr_cursor.wrapping_add(1);
                        if let Some(unit) = world.get_component_mut::<WorkUnit>(*unit_entity) {
                            unit.assigned_core = Some(core);
                            assigned += 1;
                        }
                    }
                }
                SchedulerPolicy::Gang => {
                    // Gang scheduling: all units must fit on contiguous cores.
                    // Pick the first core where `count` contiguous slots are available.
                    let count = unit_entities.len() as u32;
                    if count > self.available_cores {
                        // Cannot gang-schedule — leave as pending.
                        continue;
                    }
                    let start = self.rr_cursor % (self.available_cores - count + 1);
                    for (offset, unit_entity) in unit_entities.iter().enumerate() {
                        if let Some(unit) = world.get_component_mut::<WorkUnit>(*unit_entity) {
                            unit.assigned_core = Some(start + offset as u32);
                            assigned += 1;
                        }
                    }
                    self.rr_cursor = start.wrapping_add(count);
                }
                SchedulerPolicy::Default => {
                    // Assign to first available core.
                    for unit_entity in &unit_entities {
                        if let Some(unit) = world.get_component_mut::<WorkUnit>(*unit_entity) {
                            unit.assigned_core = Some(0);
                            assigned += 1;
                        }
                    }
                }
            }

            // Mark the group as running.
            if let Some(group) = world.get_component_mut::<WorkGroup>(group_entity) {
                group.status = WorkGroupStatus::Running;
            }
        }

        assigned
    }

    /// Mark a work unit as completed.
    ///
    /// Sets the unit's budget to zero.  If all units in the parent group
    /// are exhausted, the group transitions to [`WorkGroupStatus::Completed`].
    ///
    /// Returns `true` if the parent group is now fully completed.
    pub fn complete_unit(
        &mut self,
        world: &mut World,
        unit_entity: Entity,
    ) -> Result<bool, WorldError> {
        let parent_group = world
            .get_component::<WorkUnit>(unit_entity)
            .map(|u| u.parent_group)
            .ok_or_else(|| {
                // Synthesise a clear error — no WorldError variant fits perfectly,
                // so we use MissingResource as a proxy for "entity is not a WorkUnit".
                WorldError::MissingResource {
                    type_name: "prism_ecs_server::workgroup::WorkUnit",
                }
            })?;

        // Exhaust this unit's budget.
        if let Some(unit) = world.get_component_mut::<WorkUnit>(unit_entity) {
            unit.execution_budget = 0;
        }

        // Check whether every unit in the parent group is exhausted.
        let group_units: Vec<Entity> = world
            .get_component::<WorkGroup>(parent_group)
            .map(|g| g.entity_list.clone())
            .unwrap_or_default();

        let all_done = group_units.iter().all(|e| {
            world
                .get_component::<WorkUnit>(*e)
                .is_some_and(|u| u.is_exhausted())
        });

        if all_done {
            if let Some(group) = world.get_component_mut::<WorkGroup>(parent_group) {
                group.status = WorkGroupStatus::Completed;
            }
        }

        Ok(all_done)
    }

    /// Read the current lifecycle status of a work group.
    pub fn group_status(&self, world: &World, group_entity: Entity) -> Option<WorkGroupStatus> {
        world
            .get_component::<WorkGroup>(group_entity)
            .map(|g| g.status)
    }

    /// Cancel a work group, transitioning it and all its units to cancelled state.
    pub fn cancel_group(
        &mut self,
        world: &mut World,
        group_entity: Entity,
    ) -> Result<(), WorldError> {
        if let Some(group) = world.get_component_mut::<WorkGroup>(group_entity) {
            group.status = WorkGroupStatus::Cancelled;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::World;

    #[test]
    fn create_work_group() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(4);

        let group = system
            .create_group(&mut world, 3, SchedulerPolicy::Default, 100)
            .expect("create_group failed");

        let wg = world.get_component::<WorkGroup>(group);
        assert!(wg.is_some());
        assert_eq!(wg.unwrap().worker_count, 3);
        assert_eq!(wg.unwrap().entity_list.len(), 3);
        assert_eq!(wg.unwrap().status, WorkGroupStatus::Pending);

        // Each unit should exist and reference the group
        for unit_entity in &wg.unwrap().entity_list {
            let unit = world.get_component::<WorkUnit>(*unit_entity);
            assert!(unit.is_some());
            assert_eq!(unit.unwrap().parent_group, group);
            assert_eq!(unit.unwrap().execution_budget, 100);
        }
    }

    #[test]
    fn distribute_round_robin() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(4);

        let group = system
            .create_group(&mut world, 4, SchedulerPolicy::RoundRobin, 50)
            .expect("create_group failed");

        let assigned = system.distribute(&mut world);
        assert_eq!(assigned, 4);

        let status = system.group_status(&world, group);
        assert_eq!(status, Some(WorkGroupStatus::Running));

        // Verify each unit got a unique core
        let group_component = world.get_component::<WorkGroup>(group).unwrap();
        let mut seen_cores: Vec<u32> = group_component
            .entity_list
            .iter()
            .filter_map(|e| {
                world
                    .get_component::<WorkUnit>(*e)
                    .and_then(|u| u.assigned_core)
            })
            .collect();
        seen_cores.sort();
        assert_eq!(seen_cores, vec![0, 1, 2, 3]);
    }

    #[test]
    fn distribute_gang_scheduling() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(4);

        let group = system
            .create_group(&mut world, 3, SchedulerPolicy::Gang, 50)
            .expect("create_group failed");

        let assigned = system.distribute(&mut world);
        assert_eq!(assigned, 3);

        // Verify all units are assigned and form a contiguous block
        let group_component = world.get_component::<WorkGroup>(group).unwrap();
        let mut cores: Vec<u32> = group_component
            .entity_list
            .iter()
            .filter_map(|e| {
                world
                    .get_component::<WorkUnit>(*e)
                    .and_then(|u| u.assigned_core)
            })
            .collect();
        cores.sort();
        // Contiguous: e.g. [0, 1, 2]
        for i in 1..cores.len() {
            assert_eq!(cores[i], cores[i - 1] + 1);
        }
    }

    #[test]
    fn distribute_pinned() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(4);

        let group = system
            .create_group(&mut world, 2, SchedulerPolicy::Pinned { core: 2 }, 50)
            .expect("create_group failed");

        let assigned = system.distribute(&mut world);
        assert_eq!(assigned, 2);

        let group_component = world.get_component::<WorkGroup>(group).unwrap();
        for unit_entity in &group_component.entity_list {
            let unit = world.get_component::<WorkUnit>(*unit_entity).unwrap();
            assert_eq!(unit.assigned_core, Some(2));
        }
    }

    #[test]
    fn complete_unit_triggers_group_completion() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(2);

        let group = system
            .create_group(&mut world, 2, SchedulerPolicy::Default, 10)
            .expect("create_group failed");

        // Distribute to move to Running state.
        system.distribute(&mut world);
        assert_eq!(
            system.group_status(&world, group),
            Some(WorkGroupStatus::Running)
        );

        // Complete each unit.
        let unit_entities = world
            .get_component::<WorkGroup>(group)
            .unwrap()
            .entity_list
            .clone();
        for (i, ue) in unit_entities.iter().enumerate() {
            let is_last = i == unit_entities.len() - 1;
            let completed = system
                .complete_unit(&mut world, *ue)
                .expect("complete_unit failed");
            // Only the last unit triggers group completion.
            assert_eq!(completed, is_last);
        }

        assert_eq!(
            system.group_status(&world, group),
            Some(WorkGroupStatus::Completed)
        );
    }

    #[test]
    fn cancel_group() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(2);

        let group = system
            .create_group(&mut world, 1, SchedulerPolicy::Default, 10)
            .expect("create_group failed");

        system
            .cancel_group(&mut world, group)
            .expect("cancel failed");
        assert_eq!(
            system.group_status(&world, group),
            Some(WorkGroupStatus::Cancelled)
        );
    }

    #[test]
    fn work_unit_tick() {
        let mut world = World::new();
        let mut system = WorkGroupSystem::new(1);

        let group = system
            .create_group(&mut world, 1, SchedulerPolicy::Default, 5)
            .expect("create_group failed");

        let unit_entity = world.get_component::<WorkGroup>(group).unwrap().entity_list[0];
        let unit = world.get_component_mut::<WorkUnit>(unit_entity).unwrap();
        assert!(!unit.is_exhausted());

        assert_eq!(unit.tick(), 4);
        assert_eq!(unit.tick(), 3);
        assert_eq!(unit.tick(), 2);
        assert_eq!(unit.tick(), 1);
        assert_eq!(unit.tick(), 0);
        assert!(unit.is_exhausted());
    }
}
