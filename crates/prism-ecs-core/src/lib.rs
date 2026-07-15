//! Prism ECS core types.
//!
//! Foundation types for the ECS (Entity Component System) used throughout
//! the Tribunus compute kernel. These types are shared across crates and
//! extracted into their own crate to minimize compilation overhead.

pub mod canonical;
pub mod capacity;
pub mod column;
pub mod compilation;
pub mod component;
pub mod entity;
pub mod epoch;
pub mod error;
pub mod mutation;
pub mod nf4tile640;
pub mod query;
pub mod resource;
pub mod store;
pub mod ternary;
pub mod world;

pub use capacity::{ComponentStoreCapacity, WorldCapacity};
pub use column::{Column, ColumnStore, ErasedColumn};
pub use component::Component;
pub use entity::{
    CompEntity, Entity, EntityAllocation, EntityId, EntityKind, PendingEntity, SpawnedEntity,
};
pub use epoch::WorldEpoch;
pub use error::WorldError;
pub use mutation::MutationPolicy;
pub use query::{Query, Query2, Query3, QueryMut};
pub use resource::{Resource, ResourceMut, ResourceRef};
pub use store::{ComponentStore, ResourceStore};
pub use world::{EntityRef, World};
