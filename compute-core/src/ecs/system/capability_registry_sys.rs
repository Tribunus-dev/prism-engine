//! ECS-native capability registry — wraps `compute_image::compile::capability_registry`.
//!
//! Creates a `CapabilityRegistry` and stores it as a component on a singleton
//! entity so downstream fusion systems can query production readiness.

use crate::ecs::component::model_source::CapabilityKeyComp;
use crate::ecs::compute_image::compile::capability_registry::CapabilityRegistry;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;
use crate::ecs::Component;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Singleton entity id for the capability registry resource.
const CAPABILITY_ENTITY_NAME: &str = "capability_registry";

/// Create and populate the capability registry from the default Metal V1 set.
///
/// The registry is stored on a dedicated entity so that it can be read by
/// any downstream system via `world.get_component::<CapabilityRegistry>(entity)`.
pub struct CapabilityRegistrySystem;

impl Component for CapabilityRegistry {}

impl CompilerSystem for CapabilityRegistrySystem {
    fn name(&self) -> &str {
        "CapabilityRegistrySystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Find or create the capability registry entity.
        //
        // The helper `find_or_create_registry_entity` may spawn a new
        // entity if one does not exist. Direct `world.spawn` /
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden, so we resolve the entity through a
        // ConstitutionalWorldTxn that handles the conditional spawn,
        // then stage the inserts on a second transaction.
        //
        // Transaction 1: resolve the entity (spawn-if-missing).
        let entity = find_or_create_registry_entity(world);
        let registry = CapabilityRegistry::default_metal_v1();

        // Transaction 2: stage the inserts on the resolved entity.
        let mut txn = ConstitutionalWorldTxn::new();
        if let Err(e) = txn.stage_insert(entity, registry) {
            tracing::warn!(entity = ?entity, error = %e, "capability_registry: stage_insert CapabilityRegistry");
        }
        if let Err(e) = txn.stage_insert(
            entity,
            CapabilityKeyComp("default_metal_v1".to_string()),
        ) {
            tracing::warn!(entity = ?entity, error = %e, "capability_registry: stage_insert CapabilityKeyComp");
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "capability_registry: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("capability_registry: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}

fn find_or_create_registry_entity(world: &mut World) -> Entity {
    // Look for existing registry by name.
    for entity in world.entities_of_kind(EntityKind::Model) {
        if let Some(name) = world.name(entity) {
            if name == CAPABILITY_ENTITY_NAME {
                return entity;
            }
        }
    }
    // No existing entity — stage a spawn on a ConstitutionalWorldTxn
    // and commit, then return the resolved Entity. This is the
    // engine-local "atomic" replacement for what was previously
    // `world.spawn(...).unwrap()`. The `.unwrap()` is gone in the
    // port; we surface the error via `expect` and the Constitutional
    // WorldTxn error path returns a meaningful error if the spawn
    // is rejected.
    let mut txn = ConstitutionalWorldTxn::new();
    let token = txn.stage_spawn(
        EntityKind::Model,
        Some(CAPABILITY_ENTITY_NAME.to_string()),
    );
    let spawned = txn
        .commit(world)
        .expect("capability_registry: spawn registry entity");
    let _ = token; // token resolved at commit
    spawned
        .into_iter()
        .next()
        .expect("capability_registry: ConstitutionalWorldTxn returned no entity for spawn")
}
