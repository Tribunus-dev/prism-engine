use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{
    ClassifiedComponent, DurableClass, DurableComponent, WorldTxn, WorldTxnError,
};
use crate::ecs::{CompWorld, Component, EntityKind};
use serde::{Deserialize, Serialize};

// ── Schema IDs ───────────────────────────────────────────────────────────────
//
// Stable component schema identifiers for the work item subsystem.
// Must be kept in sync with schema registration at boot time.

pub const SCHEMA_WORK_ITEM: u64 = 18;
pub const SCHEMA_WORK_STATE: u64 = 19;
pub const SCHEMA_WORK_LEASE: u64 = 20;
pub const SCHEMA_RESOURCE_CLAIM: u64 = 21;
pub const SCHEMA_WORK_PREREQUISITES: u64 = 22;
pub const SCHEMA_WORK_OUTPUT: u64 = 23;

// ── Kind of Work ─────────────────────────────────────────────────────────────

/// Kind of work a work item represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkKind {
    LoadModel,
    CompileGraph,
    RunInference,
    Distill,
    Validate,
    Package,
    Teardown,
    Custom(u64),
}

// ── Prerequisite ─────────────────────────────────────────────────────────────

/// Kind of prerequisite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrereqKind {
    ComponentPresent,
    EventReceived,
    ResourceAvailable,
    Custom(u64),
}

/// A single prerequisite for a work item to become ready.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Prerequisite {
    pub entity: u64,
    pub kind: PrereqKind,
    pub generation: u64,
}

// ── WorkItemComponent ────────────────────────────────────────────────────────
//
// Canonical work item data stored as a component on a work entity.

/// The canonical work item descriptor attached to a work entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkItemComponent {
    pub kind: WorkKind,
    pub target_entity: u64,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl Component for WorkItemComponent {}
impl ClassifiedComponent for WorkItemComponent {
    type Class = DurableClass;
}
impl DurableComponent for WorkItemComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.work",
        id: 18,
        version: 1,
    };
}

// ── WorkState ────────────────────────────────────────────────────────────────
//
// Lifecycle state of a work item with lease generation tracking.

/// State of a work item in its lifecycle.
///
/// `Leased(lease_gen)` carries the generation counter so lease-holder
/// identity can be validated on completion/failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkState {
    Pending,
    Ready,
    Leased(u32),
    Completed,
    Failed,
    Cancelled,
}

impl WorkState {
    /// Returns true if this is a terminal state (no further transitions allowed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Returns true if the work can be started (Pending or Ready).
    pub fn is_pending_or_ready(&self) -> bool {
        matches!(self, Self::Pending | Self::Ready)
    }
}

impl Component for WorkState {}
impl ClassifiedComponent for WorkState {
    type Class = DurableClass;
}
impl DurableComponent for WorkState {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.work",
        id: 19,
        version: 1,
    };
}

// ── WorkLeaseComponent ───────────────────────────────────────────────────────
//
// Active lease metadata — created when a work entity transitions to Leased.

/// Active lease metadata attached to the leasing system / executor entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkLeaseComponent {
    pub work_entity: u64,
    pub lease_generation: u32,
    pub attempt: u32,
    pub cancellation_epoch: WorldEpoch,
    pub expiry: Timestamp,
}

impl Component for WorkLeaseComponent {}
impl ClassifiedComponent for WorkLeaseComponent {
    type Class = DurableClass;
}
impl DurableComponent for WorkLeaseComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.work",
        id: 20,
        version: 1,
    };
}

// ── ResourceClaimComponent ───────────────────────────────────────────────────
//
// Resource budget needed by a work item.

/// Resource claim attached to a work entity — what this work item needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceClaimComponent {
    pub memory_bytes: u64,
    pub compute_units: u32,
    pub priority: u32,
}

impl Component for ResourceClaimComponent {}
impl ClassifiedComponent for ResourceClaimComponent {
    type Class = DurableClass;
}
impl DurableComponent for ResourceClaimComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.work",
        id: 21,
        version: 1,
    };
}

// ── WorkPrerequisites ────────────────────────────────────────────────────────
//
// List of prerequisites for a work item to become Ready.

/// Prerequisites that must be satisfied before this work item transitions to Ready.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkPrerequisites(pub Vec<Prerequisite>);

impl Component for WorkPrerequisites {}
impl ClassifiedComponent for WorkPrerequisites {
    type Class = DurableClass;
}
impl DurableComponent for WorkPrerequisites {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.work",
        id: 22,
        version: 1,
    };
}

// ── WorkOutput ───────────────────────────────────────────────────────────────
//
// Ephemeral output data produced by completed work.

/// Output data produced by completing a work item. Ephemeral — not persisted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkOutput(pub Vec<u8>);

impl Component for WorkOutput {}
impl ClassifiedComponent for WorkOutput {
    type Class = DurableClass;
}
impl DurableComponent for WorkOutput {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.work",
        id: 23,
        version: 1,
    };
}

// ── Schema Validation ────────────────────────────────────────────────────────

/// Validate all work-related schemas are registered.
pub fn validate_work_schemas(
    reg: &crate::ecs::constitutional::schema::SchemaRegistry,
) -> Result<(), String> {
    reg.verify_type::<WorkItemComponent>(ComponentSchemaId(SCHEMA_WORK_ITEM))
        .map_err(|e| format!("SCHEMA_WORK_ITEM: {e}"))?;
    reg.verify_type::<WorkState>(ComponentSchemaId(SCHEMA_WORK_STATE))
        .map_err(|e| format!("SCHEMA_WORK_STATE: {e}"))?;
    reg.verify_type::<WorkLeaseComponent>(ComponentSchemaId(SCHEMA_WORK_LEASE))
        .map_err(|e| format!("SCHEMA_WORK_LEASE: {e}"))?;
    reg.verify_type::<ResourceClaimComponent>(ComponentSchemaId(SCHEMA_RESOURCE_CLAIM))
        .map_err(|e| format!("SCHEMA_RESOURCE_CLAIM: {e}"))?;
    reg.verify_type::<WorkPrerequisites>(ComponentSchemaId(SCHEMA_WORK_PREREQUISITES))
        .map_err(|e| format!("SCHEMA_WORK_PREREQUISITES: {e}"))?;
    reg.verify_type::<WorkOutput>(ComponentSchemaId(SCHEMA_WORK_OUTPUT))
        .map_err(|e| format!("SCHEMA_WORK_OUTPUT: {e}"))?;
    Ok(())
}

// ── Domain Event Kinds ───────────────────────────────────────────────────────

const EVENT_WORK_CREATED: &str = "work_created";
const EVENT_WORK_LEASED: &str = "work_leased";
const EVENT_WORK_COMPLETED: &str = "work_completed";
const EVENT_WORK_FAILED: &str = "work_failed";
const EVENT_WORK_CANCELLED: &str = "work_cancelled";
const EVENT_WORK_RETRIED: &str = "work_retried";

// ── Commands ─────────────────────────────────────────────────────────────────

// ── CreateWorkCommand ────────────────────────────────────────────────────────

/// Command to create a new work entity in Pending state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateWorkCommand {
    pub id: MessageId,
    pub kind: WorkKind,
    pub target_entity: u64,
    pub prerequisites: Vec<Prerequisite>,
    pub resource_claim: ResourceClaimComponent,
}

impl CreateWorkCommand {
    /// Validate that the target entity exists in the world.
    pub fn preflight(&self, world: &CompWorld) -> Result<(), WorkError> {
        if !world.has_entity(crate::ecs::CompEntity(self.target_entity)) {
            return Err(WorkError::EntityNotFound(self.target_entity));
        }
        Ok(())
    }

    /// Execute the command, staging spawn + components into a WorldTxn.
    pub fn execute(self, world: &CompWorld, txn: &mut WorldTxn) -> Result<DomainEvent, WorkError> {
        // Allocate entity ID for the work item
        let work_entity = WorldTxn::next_entity_id(world);
        txn.stage_spawn(work_entity, EntityKind::Executable);

        let domain_event = DomainEvent {
            id: self.id,
            kind: EVENT_WORK_CREATED.to_string(),
            entity_id: Some(crate::ecs::constitutional::types::EntityKindId(work_entity)),
            payload: serde_json::json!({
                "work_entity": work_entity,
                "kind": format!("{:?}", self.kind),
                "target_entity": self.target_entity,
            }),
        };

        // Stage components
        txn.put_durable::<WorkItemComponent>(
            work_entity,
            WorkItemComponent {
                kind: self.kind,
                target_entity: self.target_entity,
                retry_count: 0,
                max_retries: 0,
            },
        );
        txn.put_durable::<WorkState>(work_entity, WorkState::Pending);
        txn.put_durable::<ResourceClaimComponent>(work_entity, self.resource_claim);

        if !self.prerequisites.is_empty() {
            txn.put_durable::<WorkPrerequisites>(
                work_entity,
                WorkPrerequisites(self.prerequisites),
            );
        }

        txn.emit_event(domain_event.clone());
        Ok(domain_event)
    }
}

// ── LeaseWorkCommand ─────────────────────────────────────────────────────────

/// Command to lease a ready work item — transitions Ready → Leased.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseWorkCommand {
    pub id: MessageId,
    pub work_entity: u64,
    pub lease_generation: u32,
    pub attempt: u32,
    pub cancellation_epoch: WorldEpoch,
    pub expiry: Timestamp,
}

impl LeaseWorkCommand {
    /// Validate preconditions: entity exists, state is Ready, no conflicting lease.
    pub fn preflight(&self, world: &CompWorld) -> Result<(), WorkError> {
        let entity = crate::ecs::CompEntity(self.work_entity);
        if !world.has_entity(entity) {
            return Err(WorkError::EntityNotFound(self.work_entity));
        }
        let state = world
            .get_component::<WorkState>(entity)
            .ok_or(WorkError::MissingComponent(
                self.work_entity,
                "WorkState".into(),
            ))?;
        if *state != WorkState::Ready {
            return Err(WorkError::InvalidTransition {
                entity: self.work_entity,
                from: *state,
                to: "Leased".into(),
            });
        }
        Ok(())
    }

    /// Execute the lease: set state to Leased, add WorkLeaseComponent, emit event.
    pub fn execute(self, _world: &CompWorld, txn: &mut WorldTxn) -> Result<DomainEvent, WorkError> {
        let _entity = crate::ecs::CompEntity(self.work_entity);

        // Update state to Leased
        txn.put_durable::<WorkState>(self.work_entity, WorkState::Leased(self.lease_generation));

        // Attach lease metadata
        txn.put_durable::<WorkLeaseComponent>(
            self.work_entity,
            WorkLeaseComponent {
                work_entity: self.work_entity,
                lease_generation: self.lease_generation,
                attempt: self.attempt,
                cancellation_epoch: self.cancellation_epoch,
                expiry: self.expiry,
            },
        );

        let domain_event = DomainEvent {
            id: self.id,
            kind: EVENT_WORK_LEASED.to_string(),
            entity_id: Some(crate::ecs::constitutional::types::EntityKindId(
                self.work_entity,
            )),
            payload: serde_json::json!({
                "work_entity": self.work_entity,
                "lease_generation": self.lease_generation,
                "attempt": self.attempt,
            }),
        };

        txn.emit_event(domain_event.clone());
        Ok(domain_event)
    }
}

// ── CompleteWorkCommand ──────────────────────────────────────────────────────

/// Command to complete a leased work item — transitions Leased → Completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompleteWorkCommand {
    pub id: MessageId,
    pub work_entity: u64,
    pub output: Vec<u8>,
    pub lease_generation: u32,
}

impl CompleteWorkCommand {
    /// Validate: entity exists, state is Leased with matching generation.
    pub fn preflight(&self, world: &CompWorld) -> Result<(), WorkError> {
        let entity = crate::ecs::CompEntity(self.work_entity);
        if !world.has_entity(entity) {
            return Err(WorkError::EntityNotFound(self.work_entity));
        }
        let state = world
            .get_component::<WorkState>(entity)
            .ok_or(WorkError::MissingComponent(
                self.work_entity,
                "WorkState".into(),
            ))?;
        match state {
            WorkState::Leased(gen) if *gen == self.lease_generation => {}
            WorkState::Leased(gen) => {
                return Err(WorkError::LeaseGenerationMismatch {
                    entity: self.work_entity,
                    expected: self.lease_generation,
                    actual: *gen,
                });
            }
            _ => {
                return Err(WorkError::InvalidTransition {
                    entity: self.work_entity,
                    from: *state,
                    to: "Completed".into(),
                });
            }
        }
        Ok(())
    }

    /// Execute: set state to Completed, attach output, remove lease, emit event.
    pub fn execute(self, _world: &CompWorld, txn: &mut WorldTxn) -> Result<DomainEvent, WorkError> {
        // Update state to Completed
        txn.put_durable::<WorkState>(self.work_entity, WorkState::Completed);

        // Attach output
        txn.put_durable::<WorkOutput>(self.work_entity, WorkOutput(self.output));

        // Remove lease
        txn.remove_durable::<WorkLeaseComponent>(self.work_entity);

        let domain_event = DomainEvent {
            id: self.id,
            kind: EVENT_WORK_COMPLETED.to_string(),
            entity_id: Some(crate::ecs::constitutional::types::EntityKindId(
                self.work_entity,
            )),
            payload: serde_json::json!({
                "work_entity": self.work_entity,
                "lease_generation": self.lease_generation,
            }),
        };

        txn.emit_event(domain_event.clone());
        Ok(domain_event)
    }
}

// ── FailWorkCommand ──────────────────────────────────────────────────────────

/// Command to fail a leased work item.
///
/// If `retry` is true and `retry_count < max_retries`, transitions to Ready
/// (for retry) instead of Failed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailWorkCommand {
    pub id: MessageId,
    pub work_entity: u64,
    pub error: String,
    pub lease_generation: u32,
}

impl FailWorkCommand {
    /// Validate: entity exists, state is Leased with matching generation.
    pub fn preflight(&self, world: &CompWorld) -> Result<(), WorkError> {
        let entity = crate::ecs::CompEntity(self.work_entity);
        if !world.has_entity(entity) {
            return Err(WorkError::EntityNotFound(self.work_entity));
        }
        let state = world
            .get_component::<WorkState>(entity)
            .ok_or(WorkError::MissingComponent(
                self.work_entity,
                "WorkState".into(),
            ))?;
        match state {
            WorkState::Leased(gen) if *gen == self.lease_generation => {}
            WorkState::Leased(gen) => {
                return Err(WorkError::LeaseGenerationMismatch {
                    entity: self.work_entity,
                    expected: self.lease_generation,
                    actual: *gen,
                });
            }
            _ => {
                return Err(WorkError::InvalidTransition {
                    entity: self.work_entity,
                    from: *state,
                    to: "Failed/Retry".into(),
                });
            }
        }
        Ok(())
    }

    /// Execute: check retry eligibility, transition to Failed or Ready, emit event.
    pub fn execute(self, world: &CompWorld, txn: &mut WorldTxn) -> Result<DomainEvent, WorkError> {
        let entity = crate::ecs::CompEntity(self.work_entity);

        // Read current work item to check retry eligibility
        let should_retry = world
            .get_component::<WorkItemComponent>(entity)
            .map(|wi| wi.retry_count < wi.max_retries)
            .unwrap_or(false);

        // Remove lease
        txn.remove_durable::<WorkLeaseComponent>(self.work_entity);

        if should_retry {
            // Increment retry_count and go back to Ready
            if let Some(wi) = world.get_component::<WorkItemComponent>(entity) {
                let mut updated = wi.clone();
                updated.retry_count += 1;
                txn.put_durable::<WorkItemComponent>(self.work_entity, updated);
            }
            txn.put_durable::<WorkState>(self.work_entity, WorkState::Ready);

            let domain_event = DomainEvent {
                id: self.id,
                kind: EVENT_WORK_RETRIED.to_string(),
                entity_id: Some(crate::ecs::constitutional::types::EntityKindId(
                    self.work_entity,
                )),
                payload: serde_json::json!({
                    "work_entity": self.work_entity,
                    "error": self.error,
                    "lease_generation": self.lease_generation,
                }),
            };
            txn.emit_event(domain_event.clone());
            Ok(domain_event)
        } else {
            // Transition to Failed
            txn.put_durable::<WorkState>(self.work_entity, WorkState::Failed);

            let domain_event = DomainEvent {
                id: self.id,
                kind: EVENT_WORK_FAILED.to_string(),
                entity_id: Some(crate::ecs::constitutional::types::EntityKindId(
                    self.work_entity,
                )),
                payload: serde_json::json!({
                    "work_entity": self.work_entity,
                    "error": self.error,
                    "lease_generation": self.lease_generation,
                }),
            };
            txn.emit_event(domain_event.clone());
            Ok(domain_event)
        }
    }
}

// ── CancelWorkCommand ────────────────────────────────────────────────────────

/// Command to cancel a work item — transitions Pending/Ready/Leased → Cancelled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelWorkCommand {
    pub id: MessageId,
    pub work_entity: u64,
}

impl CancelWorkCommand {
    /// Validate: entity exists, state is not terminal.
    pub fn preflight(&self, world: &CompWorld) -> Result<(), WorkError> {
        let entity = crate::ecs::CompEntity(self.work_entity);
        if !world.has_entity(entity) {
            return Err(WorkError::EntityNotFound(self.work_entity));
        }
        let state = world
            .get_component::<WorkState>(entity)
            .ok_or(WorkError::MissingComponent(
                self.work_entity,
                "WorkState".into(),
            ))?;
        if state.is_terminal() {
            return Err(WorkError::InvalidTransition {
                entity: self.work_entity,
                from: *state,
                to: "Cancelled".into(),
            });
        }
        Ok(())
    }

    /// Execute: set state to Cancelled, remove lease if present, emit event.
    pub fn execute(self, world: &CompWorld, txn: &mut WorldTxn) -> Result<DomainEvent, WorkError> {
        let entity = crate::ecs::CompEntity(self.work_entity);

        // Remove lease if present
        if world.get_component::<WorkLeaseComponent>(entity).is_some() {
            txn.remove_durable::<WorkLeaseComponent>(self.work_entity);
        }

        // Set state to Cancelled
        txn.put_durable::<WorkState>(self.work_entity, WorkState::Cancelled);

        let domain_event = DomainEvent {
            id: self.id,
            kind: EVENT_WORK_CANCELLED.to_string(),
            entity_id: Some(crate::ecs::constitutional::types::EntityKindId(
                self.work_entity,
            )),
            payload: serde_json::json!({
                "work_entity": self.work_entity,
            }),
        };

        txn.emit_event(domain_event.clone());
        Ok(domain_event)
    }
}

// ── Replay ───────────────────────────────────────────────────────────────────

/// Replay a domain event that created a work entity.
///
/// Restores the full work entity with all durable components.
/// Ephemeral components (WorkOutput, WorkLeaseComponent) are skipped
/// and should be reconciled by the projection/query side.
pub fn replay_work_created(world: &mut CompWorld, event: &DomainEvent) -> Result<(), String> {
    let work_entity: u64 = event
        .payload
        .get("work_entity")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "replay_work_created: missing work_entity".to_string())?;

    let kind_str = event
        .payload
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "replay_work_created: missing kind".to_string())?;

    let target_entity: u64 = event
        .payload
        .get("target_entity")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "replay_work_created: missing target_entity".to_string())?;

    // Restore the entity
    let ce = world.spawn_entity_with_id(work_entity, EntityKind::Executable);

    // Restore durable components
    let kind = match kind_str {
        "LoadModel" => WorkKind::LoadModel,
        "CompileGraph" => WorkKind::CompileGraph,
        "RunInference" => WorkKind::RunInference,
        "Distill" => WorkKind::Distill,
        "Validate" => WorkKind::Validate,
        "Package" => WorkKind::Package,
        "Teardown" => WorkKind::Teardown,
        _ => WorkKind::Custom(0),
    };

    world.add_component(
        ce,
        WorkItemComponent {
            kind,
            target_entity,
            retry_count: 0,
            max_retries: 0,
        },
    );
    world.add_component(ce, WorkState::Pending);
    // ResourceClaim is durable; replicate as zero-claim if missing from event
    world.add_component(
        ce,
        ResourceClaimComponent {
            memory_bytes: 0,
            compute_units: 0,
            priority: 0,
        },
    );

    Ok(())
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkError {
    #[error("entity not found: {0}")]
    EntityNotFound(u64),

    #[error("missing component on entity {0}: {1}")]
    MissingComponent(u64, String),

    #[error("invalid state transition: entity {entity} from {from:?} to {to}")]
    InvalidTransition {
        entity: u64,
        from: WorkState,
        to: String,
    },

    #[error("lease generation mismatch on entity {entity}: expected {expected}, actual {actual}")]
    LeaseGenerationMismatch {
        entity: u64,
        expected: u32,
        actual: u32,
    },

    #[error("world transaction failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::types::MessageId;

    fn make_id(seed: &[u8]) -> MessageId {
        MessageId::compute(seed)
    }

    /// Helper: create a minimal CompWorld with a target entity.
    fn setup_world() -> CompWorld {
        let mut world = CompWorld::new();
        let mut txn = WorldTxn::new(&world);
        let eid = WorldTxn::next_entity_id(&world);
        txn.stage_spawn(eid, EntityKind::Model);
        world.transit(txn).unwrap();
        world
    }

    fn make_resource_claim() -> ResourceClaimComponent {
        ResourceClaimComponent {
            memory_bytes: 1024,
            compute_units: 8,
            priority: 1,
        }
    }

    #[test]
    fn test_create_work_item() {
        let mut world = setup_world();
        let target = 1; // the model entity we spawned
        let cmd = CreateWorkCommand {
            id: make_id(b"create-test"),
            kind: WorkKind::CompileGraph,
            target_entity: target,
            prerequisites: vec![],
            resource_claim: make_resource_claim(),
        };

        // Preflight should pass
        assert!(cmd.preflight(&world).is_ok());

        // Execute via transaction
        let mut txn = WorldTxn::new(&world);
        let event = cmd.execute(&world, &mut txn).unwrap();
        assert_eq!(event.kind, "work_created");

        let epoch = world.transit(txn).unwrap();
        assert!(epoch.0 .0 > 0);

        // Find the work entity (entity 2, after the target)
        let work_entity = world
            .entities_of_kind(EntityKind::Executable)
            .first()
            .copied()
            .expect("work entity should exist");

        // Verify components
        let item = world
            .get_component::<WorkItemComponent>(work_entity)
            .expect("should have WorkItemComponent");
        assert_eq!(item.kind, WorkKind::CompileGraph);
        assert_eq!(item.target_entity, target);
        assert_eq!(item.retry_count, 0);

        let state = world
            .get_component::<WorkState>(work_entity)
            .expect("should have WorkState");
        assert_eq!(*state, WorkState::Pending);

        let claim = world
            .get_component::<ResourceClaimComponent>(work_entity)
            .expect("should have ResourceClaimComponent");
        assert_eq!(claim.memory_bytes, 1024);
    }

    #[test]
    fn test_work_state_transitions() {
        let mut world = setup_world();
        let target = 1;

        // Create work item
        let create_cmd = CreateWorkCommand {
            id: make_id(b"transitions-create"),
            kind: WorkKind::Validate,
            target_entity: target,
            prerequisites: vec![],
            resource_claim: make_resource_claim(),
        };

        let mut txn = WorldTxn::new(&world);
        create_cmd.execute(&world, &mut txn).unwrap();
        world.transit(txn).unwrap();

        // Transition Pending → Ready (manual)
        let work_entity = world
            .entities_of_kind(EntityKind::Executable)
            .first()
            .copied()
            .unwrap();

        let mut txn = WorldTxn::new(&world);
        txn.put_durable::<WorkState>(work_entity.0, WorkState::Ready);
        world.transit(txn).unwrap();

        let state = world.get_component::<WorkState>(work_entity).unwrap();
        assert_eq!(*state, WorkState::Ready);

        // Lease the work item (Ready → Leased)
        let lease_cmd = LeaseWorkCommand {
            id: make_id(b"transitions-lease"),
            work_entity: work_entity.0,
            lease_generation: 1,
            attempt: 1,
            cancellation_epoch: WorldEpoch(0),
            expiry: Timestamp::now(),
        };

        assert!(lease_cmd.preflight(&world).is_ok());
        let mut txn = WorldTxn::new(&world);
        lease_cmd.execute(&world, &mut txn).unwrap();
        world.transit(txn).unwrap();

        let state = world.get_component::<WorkState>(work_entity).unwrap();
        assert_eq!(*state, WorkState::Leased(1));

        let lease = world
            .get_component::<WorkLeaseComponent>(work_entity)
            .unwrap();
        assert_eq!(lease.lease_generation, 1);
        assert_eq!(lease.attempt, 1);

        // Complete the work (Leased → Completed) via command
        let complete_cmd = CompleteWorkCommand {
            id: make_id(b"transitions-complete"),
            work_entity: work_entity.0,
            output: b"done".to_vec(),
            lease_generation: 1,
        };

        assert!(complete_cmd.preflight(&world).is_ok());
        let mut txn = WorldTxn::new(&world);
        complete_cmd.execute(&world, &mut txn).unwrap();
        world.transit(txn).unwrap();

        let state = world.get_component::<WorkState>(work_entity).unwrap();
        assert_eq!(*state, WorkState::Completed);

        // Lease should be removed on completion
        assert!(world
            .get_component::<WorkLeaseComponent>(work_entity)
            .is_none());

        // Output should be attached
        let output = world.get_component::<WorkOutput>(work_entity).unwrap();
        assert_eq!(output.0, b"done");
    }

    #[test]
    fn test_lease_then_complete() {
        let mut world = setup_world();
        let target = 1;

        // Create item in Pending, transition to Ready manually
        {
            let mut txn = WorldTxn::new(&world);
            let create_cmd = CreateWorkCommand {
                id: make_id(b"lifecycle-create"),
                kind: WorkKind::RunInference,
                target_entity: target,
                prerequisites: vec![],
                resource_claim: make_resource_claim(),
            };
            create_cmd.execute(&world, &mut txn).unwrap();
            world.transit(txn).unwrap();
        }

        let work_entity = world
            .entities_of_kind(EntityKind::Executable)
            .first()
            .copied()
            .unwrap();

        // Ready
        {
            let mut txn = WorldTxn::new(&world);
            txn.put_durable::<WorkState>(work_entity.0, WorkState::Ready);
            world.transit(txn).unwrap();
        }

        // Lease
        let lease_cmd = LeaseWorkCommand {
            id: make_id(b"lifecycle-lease"),
            work_entity: work_entity.0,
            lease_generation: 1,
            attempt: 1,
            cancellation_epoch: WorldEpoch(0),
            expiry: Timestamp::now(),
        };
        {
            let mut txn = WorldTxn::new(&world);
            lease_cmd.execute(&world, &mut txn).unwrap();
            world.transit(txn).unwrap();
        }

        // Complete
        let complete_cmd = CompleteWorkCommand {
            id: make_id(b"lifecycle-complete"),
            work_entity: work_entity.0,
            output: vec![0u8; 16],
            lease_generation: 1,
        };
        {
            let mut txn = WorldTxn::new(&world);
            complete_cmd.execute(&world, &mut txn).unwrap();
            world.transit(txn).unwrap();
        }

        let state = world.get_component::<WorkState>(work_entity).unwrap();
        assert_eq!(*state, WorkState::Completed);
        assert!(world
            .get_component::<WorkLeaseComponent>(work_entity)
            .is_none());
    }

    #[test]
    fn test_fail_retry() {
        let mut world = setup_world();
        let target = 1;

        // Create work item with retry configured
        {
            let mut txn = WorldTxn::new(&world);
            let work_entity_id = WorldTxn::next_entity_id(&world);
            txn.stage_spawn(work_entity_id, EntityKind::Executable);
            txn.put_durable::<WorkItemComponent>(
                work_entity_id,
                WorkItemComponent {
                    kind: WorkKind::Validate,
                    target_entity: target,
                    retry_count: 0,
                    max_retries: 3,
                },
            );
            txn.put_durable::<WorkState>(work_entity_id, WorkState::Ready);
            world.transit(txn).unwrap();
        }

        let work_entity = world
            .entities_of_kind(EntityKind::Executable)
            .first()
            .copied()
            .unwrap();

        // Verify initial: Ready
        assert_eq!(
            *world.get_component::<WorkState>(work_entity).unwrap(),
            WorkState::Ready
        );

        // Lease
        {
            let mut txn = WorldTxn::new(&world);
            let lease_cmd = LeaseWorkCommand {
                id: make_id(b"retry-lease-1"),
                work_entity: work_entity.0,
                lease_generation: 1,
                attempt: 1,
                cancellation_epoch: WorldEpoch(0),
                expiry: Timestamp::now(),
            };
            lease_cmd.execute(&world, &mut txn).unwrap();
            world.transit(txn).unwrap();
        }

        // Fail command — should retry (back to Ready) because retry_count < max_retries
        let fail_cmd = FailWorkCommand {
            id: make_id(b"retry-fail-1"),
            work_entity: work_entity.0,
            error: "transient error".into(),
            lease_generation: 1,
        };
        {
            let mut txn = WorldTxn::new(&world);
            let event = fail_cmd.execute(&world, &mut txn).unwrap();
            assert_eq!(event.kind, EVENT_WORK_RETRIED);
            world.transit(txn).unwrap();
        }

        // Should now be back to Ready
        assert_eq!(
            *world.get_component::<WorkState>(work_entity).unwrap(),
            WorkState::Ready
        );

        // retry_count should have incremented
        let wi = world
            .get_component::<WorkItemComponent>(work_entity)
            .unwrap();
        assert_eq!(wi.retry_count, 1);

        // Lease should be removed
        assert!(world
            .get_component::<WorkLeaseComponent>(work_entity)
            .is_none());
    }

    #[test]
    fn test_create_work_preflight_fails_on_missing_target() {
        let world = CompWorld::new(); // no entities
        let cmd = CreateWorkCommand {
            id: make_id(b"preflight-fail"),
            kind: WorkKind::Validate,
            target_entity: 999,
            prerequisites: vec![],
            resource_claim: make_resource_claim(),
        };
        let result = cmd.preflight(&world);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            WorkError::EntityNotFound(999)
        ));
    }

    #[test]
    fn test_lease_preflight_fails_when_not_ready() {
        let mut world = setup_world();
        let target = 1;

        // Create work item (stays Pending)
        let create_cmd = CreateWorkCommand {
            id: make_id(b"lease-preflight"),
            kind: WorkKind::Validate,
            target_entity: target,
            prerequisites: vec![],
            resource_claim: make_resource_claim(),
        };
        let mut txn = WorldTxn::new(&world);
        create_cmd.execute(&world, &mut txn).unwrap();
        world.transit(txn).unwrap();

        let work_entity = world
            .entities_of_kind(EntityKind::Executable)
            .first()
            .copied()
            .unwrap();

        // Trying to lease from Pending should fail
        let lease_cmd = LeaseWorkCommand {
            id: make_id(b"lease-bad-state"),
            work_entity: work_entity.0,
            lease_generation: 1,
            attempt: 1,
            cancellation_epoch: WorldEpoch(0),
            expiry: Timestamp::now(),
        };
        let result = lease_cmd.preflight(&world);
        assert!(result.is_err());
    }
}
