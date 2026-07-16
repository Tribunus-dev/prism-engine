//! Hash dispatch table — RPCS3 inspired O(1) SPU JIT code dispatch.
//!
//! RPCS3's `spu_runtime` uses a 2^20 entry hash table (hashed on the SPU
//! program counter and thread context) to achieve O(1) dispatch to JIT-compiled
//! SPU recompiler blocks.  This module generalises that pattern: any kernel
//! entity can be registered in a hash table indexed by a deterministic hash
//! of the kernel's name and feature profile.
//!
//! # Components
//!
//! * [`HashDispatchTable`] — a sparse ECS component mapping a hash index to
//!   a kernel entity.  Attached to a dispatch-table entity in the world.
//! * [`DispatchHash`] — component holding the hash index for a kernel entity,
//!   enabling O(1) reverse lookup from the table.
//!
//! # Systems
//!
//! * [`HashDispatchSystem`] — resource that manages insertion, lookup, and
//!   eviction in a hash dispatch table.
//!
//! # Integration
//!
//! The [`backend_dispatch`](crate::backend_dispatch) module reads
//! [`HashDispatchTable`] components during dispatch to find compiled kernels
//! without a linear scan.

use prism_ecs_core::{Component, Entity, EntityKind, World, WorldError};
use serde::{Deserialize, Serialize};

// Default table size: 2^20 = 1_048_576 entries, matching RPCS3's `spu_runtime`.
const DEFAULT_TABLE_SIZE: usize = 1 << 20;

// ---------------------------------------------------------------------------
// DispatchHash — component on kernel entities
// ---------------------------------------------------------------------------

/// The hash index assigned to a kernel entity in a hash dispatch table.
///
/// Attached to a kernel entity after registration so that the hash can be
/// reused for fast table-bucket lookup without recomputing the hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DispatchHash(pub u64);

impl Component for DispatchHash {}

// ---------------------------------------------------------------------------
// HashDispatchTable — component on a dispatch-table entity
// ---------------------------------------------------------------------------

/// A hash-indexed dispatch table mapping kernel hashes to kernel entities.
///
/// Modelled after RPCS3's `spu_runtime` 2^20-entry hash table.  Each slot
/// optionally holds a kernel entity.  Lookup is `O(1)` — hash the kernel
/// name and feature fingerprint, then index into the vector.
///
/// The table is stored as a component on a "dispatch-table" entity in the
/// world, allowing multiple independent tables (one per model, compile
/// target, or execution context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashDispatchTable {
    /// The actual dispatch slots.  Each slot is `None` (empty) or `Some(entity)`.
    pub slots: Vec<Option<Entity>>,
    /// Mask for fast index computation: `hash & mask`.
    /// Guaranteed to be `slots.len() - 1` which is a power-of-two.
    pub mask: u64,
    /// Number of occupied slots (for capacity monitoring).
    pub occupied: usize,
}

impl Component for HashDispatchTable {}

impl HashDispatchTable {
    /// Create a new dispatch table with the given size (rounded up to the
    /// next power of two).  The default size is 2^20.
    pub fn new(size: Option<usize>) -> Self {
        let raw = size.unwrap_or(DEFAULT_TABLE_SIZE);
        let capacity = raw.next_power_of_two();
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            mask: (capacity as u64).wrapping_sub(1),
            occupied: 0,
        }
    }

    /// Return the slot index for a hash value (capped by the table mask).
    #[inline]
    pub fn index_for(&self, hash: u64) -> usize {
        (hash & self.mask) as usize
    }

    /// Look up a kernel entity by its dispatch hash.
    ///
    /// Returns `Some(entity)` if a kernel is registered at the indexed slot,
    /// `None` otherwise.
    #[inline]
    pub fn lookup(&self, hash: u64) -> Option<Entity> {
        let idx = self.index_for(hash);
        self.slots[idx]
    }

    /// Register a kernel entity at the slot determined by `hash`.
    ///
    /// Overwrites any previous occupant.  Attaches a [`DispatchHash`]
    /// component to the kernel entity for reverse lookup.
    ///
    /// Returns the previous occupant, if any.
    pub fn insert(&mut self, hash: u64, kernel_entity: Entity) -> Option<Entity> {
        let idx = self.index_for(hash);
        let prev = self.slots[idx].take();
        self.slots[idx] = Some(kernel_entity);
        if prev.is_none() {
            self.occupied += 1;
        }
        prev
    }

    /// Remove the occupant at `hash`, returning it if present.
    pub fn remove(&mut self, hash: u64) -> Option<Entity> {
        let idx = self.index_for(hash);
        let prev = self.slots[idx].take();
        if prev.is_some() {
            self.occupied = self.occupied.saturating_sub(1);
        }
        prev
    }

    /// Returns `true` if the slot at `hash` is occupied.
    pub fn contains(&self, hash: u64) -> bool {
        self.slots[self.index_for(hash)].is_some()
    }

    /// Clear all entries in the table.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.occupied = 0;
    }

    /// Load factor as a fraction of occupied / capacity.
    pub fn load_factor(&self) -> f64 {
        if self.slots.is_empty() {
            return 0.0;
        }
        self.occupied as f64 / self.slots.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Hash computation — deterministic hash from name + feature fingerprint
// ---------------------------------------------------------------------------

/// Compute a dispatch hash from a kernel name and an optional feature
/// fingerprint string.
///
/// Uses FxHash (the crate's existing hash) for speed.  The 64-bit result
/// is masked by the table size before indexing.
///
/// This is the default hashing strategy, analogous to RPCS3's hashing of
/// SPU program counter and thread context.
pub fn compute_dispatch_hash(name: &str, features: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = fxhash::FxHasher::default();
    name.hash(&mut hasher);
    if let Some(f) = features {
        f.hash(&mut hasher);
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// HashDispatchSystem — resource-level coordinator
// ---------------------------------------------------------------------------

/// ECS resource for managing hash dispatch tables.
///
/// Provides helpers to spawn a dispatch-table entity in the world, register
/// kernel entities by name and features, and look up kernels for dispatch.
///
/// # Usage
///
/// ```ignore
/// // On init
/// let table_entity = system.create_table(&mut world, None)?;
///
/// // After compilation
/// system.register(&mut world, table_entity, "matmul_16x16", None, &kernel_entity)?;
///
/// // On dispatch
/// let hash = compute_dispatch_hash("matmul_16x16", None);
/// if let Some(found) = system.lookup(&world, table_entity, hash) {
///     // dispatch found entity …
/// }
/// ```
#[derive(Debug, Default)]
pub struct HashDispatchSystem;

impl HashDispatchSystem {
    /// Create a new dispatch-table entity with a [`HashDispatchTable`]
    /// component of the given size.
    ///
    /// When `size` is `None`, defaults to 2^20 entries.
    pub fn create_table(
        &self,
        world: &mut World,
        size: Option<usize>,
    ) -> Result<Entity, WorldError> {
        let table: Entity = world
            .spawn(EntityKind::Node, Some("hash_dispatch_table".into()))?
            .into();
        world.add_component(table, HashDispatchTable::new(size))?;
        Ok(table)
    }

    /// Register a kernel entity in the dispatch table under the given name.
    ///
    /// Computes the dispatch hash from `name` and `features`, then inserts
    /// the kernel entity.  Attaches a [`DispatchHash`] component to the
    /// kernel so the hash can be read back without recomputation.
    pub fn register(
        &self,
        world: &mut World,
        table_entity: Entity,
        name: &str,
        features: Option<&str>,
        kernel_entity: Entity,
    ) -> Result<(), WorldError> {
        let hash = compute_dispatch_hash(name, features);

        // Attach DispatchHash to the kernel for reverse lookup.
        world.add_component(kernel_entity, DispatchHash(hash))?;

        // Insert into the table.
        if let Some(table) = world.get_component_mut::<HashDispatchTable>(table_entity) {
            table.insert(hash, kernel_entity);
        }

        Ok(())
    }

    /// Look up a kernel entity in the table by hash.
    ///
    /// Returns `None` when the slot is empty or the table entity is missing
    /// the [`HashDispatchTable`] component.
    pub fn lookup(&self, world: &World, table_entity: Entity, hash: u64) -> Option<Entity> {
        world
            .get_component::<HashDispatchTable>(table_entity)
            .and_then(|t| t.lookup(hash))
    }

    /// Remove a kernel from the table by name and features.
    pub fn unregister(
        &self,
        world: &mut World,
        table_entity: Entity,
        name: &str,
        features: Option<&str>,
    ) -> Result<Option<Entity>, WorldError> {
        let hash = compute_dispatch_hash(name, features);
        let removed = world
            .get_component_mut::<HashDispatchTable>(table_entity)
            .and_then(|t| t.remove(hash));
        Ok(removed)
    }

    /// Read the current load factor of a dispatch table.
    pub fn load_factor(&self, world: &World, table_entity: Entity) -> Option<f64> {
        world
            .get_component::<HashDispatchTable>(table_entity)
            .map(|t| t.load_factor())
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
    fn create_table_default_size() {
        let mut world = World::new();
        let system = HashDispatchSystem;
        let table = system.create_table(&mut world, None).expect("create_table");
        let comp = world.get_component::<HashDispatchTable>(table).unwrap();
        assert_eq!(comp.slots.len(), 1 << 20);
        assert_eq!(comp.mask, (1u64 << 20) - 1);
        assert_eq!(comp.occupied, 0);
    }

    #[test]
    fn create_table_custom_size() {
        let mut world = World::new();
        let system = HashDispatchSystem;
        let table = system
            .create_table(&mut world, Some(16))
            .expect("create_table");
        let comp = world.get_component::<HashDispatchTable>(table).unwrap();
        // next_power_of_two(16) == 16
        assert_eq!(comp.slots.len(), 16);
        assert_eq!(comp.mask, 15);
    }

    #[test]
    fn register_and_lookup() {
        let mut world = World::new();
        let system = HashDispatchSystem;

        let table = system
            .create_table(&mut world, Some(64))
            .expect("create_table");
        let kernel: Entity = world
            .spawn(EntityKind::Kernel, Some("test_kernel".into()))
            .expect("spawn kernel")
            .into();

        system
            .register(&mut world, table, "matmul_16x16", Some("fp16"), kernel)
            .expect("register");

        // Kernel entity should now carry a DispatchHash
        let dh = world.get_component::<DispatchHash>(kernel).unwrap();
        let hash = dh.0;

        // Lookup should find it
        let found = system.lookup(&world, table, hash);
        assert_eq!(found, Some(kernel));

        // Lookup by recomputed hash should match
        let computed = compute_dispatch_hash("matmul_16x16", Some("fp16"));
        let found_computed = system.lookup(&world, table, computed);
        assert_eq!(found_computed, Some(kernel));
    }

    #[test]
    fn register_overwrites_previous() {
        let mut world = World::new();
        let system = HashDispatchSystem;

        let table = system
            .create_table(&mut world, Some(16))
            .expect("create_table");
        let k1: Entity = world.spawn(EntityKind::Kernel, None).expect("k1").into();
        let k2: Entity = world.spawn(EntityKind::Kernel, None).expect("k2").into();

        // Both with the same name — second overwrites first.
        system
            .register(&mut world, table, "same", None, k1)
            .expect("register k1");
        system
            .register(&mut world, table, "same", None, k2)
            .expect("register k2");

        let hash = compute_dispatch_hash("same", None);
        let found = system.lookup(&world, table, hash);
        assert_eq!(found, Some(k2));
    }

    #[test]
    fn unregister_removes_entry() {
        let mut world = World::new();
        let system = HashDispatchSystem;

        let table = system
            .create_table(&mut world, Some(16))
            .expect("create_table");
        let kernel: Entity = world.spawn(EntityKind::Kernel, None).expect("k").into();

        system
            .register(&mut world, table, "op", None, kernel)
            .expect("register");

        let removed = system
            .unregister(&mut world, table, "op", None)
            .expect("unregister");
        assert_eq!(removed, Some(kernel));

        let hash = compute_dispatch_hash("op", None);
        assert!(system.lookup(&world, table, hash).is_none());
    }

    #[test]
    fn load_factor_tracking() {
        let mut world = World::new();
        let system = HashDispatchSystem;

        let table = system
            .create_table(&mut world, Some(4))
            .expect("create_table");

        // Initially empty
        assert!((system.load_factor(&world, table).unwrap() - 0.0).abs() < 1e-9);

        let k1: Entity = world.spawn(EntityKind::Kernel, None).expect("k1").into();
        system
            .register(&mut world, table, "a", None, k1)
            .expect("reg a");
        assert!((system.load_factor(&world, table).unwrap() - 0.25).abs() < 1e-9);

        // Removal reduces load
        system
            .unregister(&mut world, table, "a", None)
            .expect("unreg");
        assert!((system.load_factor(&world, table).unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn compute_hash_deterministic() {
        let h1 = compute_dispatch_hash("matmul_16x16", Some("fp16"));
        let h2 = compute_dispatch_hash("matmul_16x16", Some("fp16"));
        assert_eq!(h1, h2);

        let h3 = compute_dispatch_hash("matmul_16x16", None);
        let h4 = compute_dispatch_hash("matmul_16x16", Some("int8"));
        assert_ne!(h1, h3);
        assert_ne!(h1, h4);
    }
}
