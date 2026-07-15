use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::column::Column;
use crate::component::Component;
use crate::entity::Entity;
use crate::error::WorldError;

/// Type-erased storage for components, indexed by (TypeId, EntityId).
///
/// This is the original HashMap-based store. The newer `ColumnStore` provides
/// generation-aware SparseSet storage; this wrapper provides backward-compatible
/// access on top of columnar storage.
#[derive(Debug)]
pub struct ComponentStore {
    pub data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for ComponentStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl ComponentStore {
    /// Get or create a column for component type T.
    pub fn column_mut<T: Component>(&mut self) -> &mut Column<T> {
        let key = TypeId::of::<Column<T>>();
        self.data
            .entry(key)
            .or_insert_with(|| Box::new(Column::<T>::new()))
            .downcast_mut::<Column<T>>()
            .expect("Column<T> type mismatch in ComponentStore")
    }

    /// Get a shared reference to a column.
    pub fn column<T: Component>(&self) -> Option<&Column<T>> {
        let key = TypeId::of::<Column<T>>();
        self.data.get(&key)?.downcast_ref::<Column<T>>()
    }

    /// Canonical: insert or replace a component.
    pub fn insert_component<T: Component>(
        &mut self,
        entity: Entity,
        value: T,
    ) -> Result<(), WorldError> {
        self.insert::<T>(entity, value);
        Ok(())
    }

    /// Canonical: read a component.
    pub fn component<T: Component>(&self, entity: Entity) -> Result<&T, WorldError> {
        self.get::<T>(entity).ok_or(WorldError::MissingComponent {
            entity,
            type_name: std::any::type_name::<T>(),
        })
    }

    /// Canonical: mutable read of a component.
    pub fn component_mut<T: Component>(&mut self, entity: Entity) -> Result<&mut T, WorldError> {
        self.column_mut::<T>()
            .get_mut(entity)
            .ok_or(WorldError::MissingComponent {
                entity,
                type_name: std::any::type_name::<T>(),
            })
    }

    /// Canonical: check if entity has a component.
    pub fn has_component<T: Component>(&self, entity: Entity) -> bool {
        self.contains::<T>(entity)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.column_mut::<T>().insert(entity, value);
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.column::<T>()?.get(entity)
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.column_mut::<T>().remove(entity)
    }

    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.column::<T>().map(|c| c.has(entity)).unwrap_or(false)
    }
}

/// Type-erased storage for global resources (not per-entity).
#[derive(Debug)]
pub struct ResourceStore {
    pub data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for ResourceStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl ResourceStore {
    pub fn insert<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(resource));
    }

    pub fn get<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.data
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    pub fn contains<T: 'static + Send + Sync>(&self) -> bool {
        self.data.contains_key(&TypeId::of::<T>())
    }

    pub fn remove<T: 'static + Send + Sync>(&mut self) -> Option<T> {
        self.data
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    pub fn get_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
        self.data
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }
}
