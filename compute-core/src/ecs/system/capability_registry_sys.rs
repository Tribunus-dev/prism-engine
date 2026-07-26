//! ECS-native capability registry — wraps `compute_image::compile::capability_registry`.
//!
//! Creates a `CapabilityRegistry` and stores it as a component on a singleton
//! entity so downstream fusion systems can query production readiness.

use crate::ecs::component::model_source::CapabilityKeyComp;
use crate::ecs::compute_image::compile::capability_registry::CapabilityRegistry;
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
        let entity = find_or_create_registry_entity(world);
        let registry = CapabilityRegistry::default_metal_v1();

        world.add_component(entity, registry)?;

        // Also tag the entity so other systems can find it by component type.
        world.add_component(entity, CapabilityKeyComp("default_metal_v1".to_string()))?;

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
    world
        .spawn(EntityKind::Model, Some(CAPABILITY_ENTITY_NAME.to_string()))
        .unwrap()
        .into()
}
