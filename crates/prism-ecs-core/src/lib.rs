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
pub mod identity;
pub mod memory_model;
pub mod memory_tier;
pub mod mutation;
pub mod nf4tile640;
pub mod query;
pub mod resource;
pub mod scheduling;
pub mod snapshot;
pub mod store;
pub mod ternary;
pub mod world;

pub use capacity::{ComponentStoreCapacity, WorldCapacity};
pub use column::{Column, ColumnStore, ErasedColumn};
pub use component::Component;
pub use entity::{Entity, EntityAllocation, EntityKind, PendingEntity, SpawnedEntity};
pub use epoch::WorldEpoch;
pub use error::WorldError;
pub use mutation::MutationPolicy;
pub use query::{Query, Query2, Query3, QueryMut};
pub use resource::{Resource, ResourceMut, ResourceRef};
pub use store::{ComponentStore, MemoryStore, MemoryStoreError, ResourceStore};
pub use world::{EntityRef, World};
pub mod observability;
pub use observability::{global_context, StateRecord, StateSnapshot, StateStream, TraceContext};
