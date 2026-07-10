pub mod aot;
pub mod compile_session;
pub mod adapter;
pub mod config;
pub mod plan;
pub mod component;
pub mod entity;
pub mod system;
#[cfg(test)]
mod tests;

pub use component::backend::*;
pub use component::memory::*;
pub use component::quality::*;
pub use component::tensor::*;

use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;

pub type EntityId = u64;

/// Opaque entity handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompEntity(pub EntityId);

/// Tag trait for data attached to entities.
pub trait Component: std::fmt::Debug + Send + Sync + 'static {}

/// Phase in the compiler pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulePhase {
    ModelLoading,     // Phase A
    Quantization,     // Phase B
    MemoryPlanning,   // Phase C
    FusionDispatch,   // Phase D
    KernelGeneration, // Phase E
    Compilation,      // Phase F
    Packaging,        // Phase G
    Validation,       // Phase H — final admission gates and completeness checks
}

/// A compiler pass over the ECS world.
pub trait CompilerSystem: Send + Sync {
    fn name(&self) -> &str;
    fn phase(&self) -> SchedulePhase;
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()>;
}

/// The ECS world — all entities, components, and systems.
pub struct CompWorld {
    next_id: EntityId,
    entities: Vec<EntityMeta>,
    components: ComponentStore,
    systems: Vec<Box<dyn CompilerSystem>>,
}

impl std::fmt::Debug for CompWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompWorld")
            .field("entity_count", &self.entities.len())
            .field("system_count", &self.systems.len())
            .field("component_type_count", &self.components.data.len())
            .finish()
    }
}

/// Manages the lifecycle of a single executable argument region.
#[derive(Debug)]
struct EntityMeta {
    name: Option<String>,
    kind: EntityKind,
}

impl Default for EntityMeta {
    fn default() -> Self {
        Self {
            name: None,
            kind: EntityKind::Model,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Model,
    Tensor,
    Layer,
    Expert,
    Dispatch,
    Kernel,
    Buffer,
    CommandBuffer,
    Executable,
    Fence,
}

/// Type-erased storage for components, indexed by (TypeId, EntityId).
#[derive(Debug)]
pub struct ComponentStore {
    data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for ComponentStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl CompWorld {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entities: Vec::new(),
            components: ComponentStore::default(),
            systems: Vec::new(),
        }
    }

    pub fn spawn(&mut self, kind: EntityKind, name: Option<String>) -> CompEntity {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(EntityMeta { name, kind });
        CompEntity(id)
    }

    pub fn kind(&self, entity: CompEntity) -> Option<EntityKind> {
        self.entities.get(entity.0 as usize - 1).map(|m| m.kind)
    }

    pub fn name(&self, entity: CompEntity) -> Option<&str> {
        self.entities
            .get(entity.0 as usize - 1)
            .and_then(|m| m.name.as_deref())
    }

    pub fn add_component<T: Component>(&mut self, entity: CompEntity, component: T) {
        self.components.insert(entity, component);
    }

    pub fn get_component<T: Component>(&self, entity: CompEntity) -> Option<&T> {
        self.components.get(entity)
    }

    pub fn get_component_mut<T: Component>(&mut self, entity: CompEntity) -> Option<&mut T> {
        self.components.get_mut(entity)
    }

    pub fn remove_component<T: Component>(&mut self, entity: CompEntity) -> Option<T> {
        self.components.remove(entity)
    }

    pub fn add_system(&mut self, system: Box<dyn CompilerSystem>) {
        self.systems.push(system);
    }

    pub fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()> {
        // Take systems out of self to avoid borrow conflict, then restore.
        let systems = std::mem::take(&mut self.systems);
        for system in &systems {
            if system.phase() == phase {
                system.run(self)?;
            }
        }
        self.systems = systems;
        Ok(())
    }

    pub fn run_all(&mut self) -> anyhow::Result<()> {
        let phases = [
            SchedulePhase::ModelLoading,
            SchedulePhase::Quantization,
            SchedulePhase::MemoryPlanning,
            SchedulePhase::FusionDispatch,
            SchedulePhase::KernelGeneration,
            SchedulePhase::Compilation,
            SchedulePhase::Packaging,
            SchedulePhase::Validation,
        ];
        for phase in phases {
            self.run_phase(phase)?;
        }
        Ok(())
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<CompEntity> {
        self.entities
            .iter()
            .enumerate()
            .filter(|(_, m)| m.kind == kind)
            .map(|(i, _)| CompEntity(i as u64 + 1))
            .collect()
    }
}

impl ComponentStore {
    fn insert<T: Component>(&mut self, entity: CompEntity, component: T) {
        let tid = TypeId::of::<T>();
        let slot = self
            .data
            .entry(tid)
            .or_insert_with(|| Box::<HashMap<EntityId, T>>::default());
        if let Some(map) = slot.downcast_mut::<HashMap<EntityId, T>>() {
            map.insert(entity.0, component);
        }
    }

    fn get<T: Component>(&self, entity: CompEntity) -> Option<&T> {
        let tid = TypeId::of::<T>();
        self.data
            .get(&tid)
            .and_then(|slot| slot.downcast_ref::<HashMap<EntityId, T>>())
            .and_then(|map| map.get(&entity.0))
    }

    fn get_mut<T: Component>(&mut self, entity: CompEntity) -> Option<&mut T> {
        let tid = TypeId::of::<T>();
        self.data
            .get_mut(&tid)
            .and_then(|slot| slot.downcast_mut::<HashMap<EntityId, T>>())
            .and_then(|map| map.get_mut(&entity.0))
    }

    fn remove<T: Component>(&mut self, entity: CompEntity) -> Option<T> {
        let tid = TypeId::of::<T>();
        self.data
            .get_mut(&tid)
            .and_then(|slot| slot.downcast_mut::<HashMap<EntityId, T>>())
            .and_then(|map| map.remove(&entity.0))
    }
}
