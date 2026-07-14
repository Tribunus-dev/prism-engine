# ADR-003: Canonical ECS World

## Status

In Progress — Phases 1–6 committed (all 3 Waves complete), Phases 7–10 in progress.

## Context

Prism Engine has three overlapping ECS world implementations:

| World | Entity model | Storage | Used by |
|---|---|---|---|
| `CompWorld` (ecs/mod.rs) | u64 index, no generation | `HashMap<TypeId, HashMap<EntityId, T>>` | Compiler pipeline, constitutional governance |
| `runtime::world::World` (runtime/world.rs) | u32 + generation counter | `HashMap<TypeId, ComponentVec<T>>` (SparseSet) | Worker pipeline, inference scheduling |
| `WorldTxn` (constitutional/world_txn.rs) | wraps CompWorld | CompWorld + epoch journal + schema | Constitutional commands and replay |

Each world was built for a different subsystem. The cost of maintaining two independent entity models, component stores, query conventions, and mutation protocols grows with every new feature. The architecture cannot survive three worlds; it barely survives two.

## Decision

Adopt `CompWorld` as the **constitutional foundation** because it already owns the difficult guarantees:

- Schema registration and durable/transient classification
- Optimistic concurrency with read-version tracking
- Epoch-based commit ordering
- Atomic transactions via `WorldTxn` → `PreparedWorldTxn`
- Domain events emitted only after successful commit
- Mutation journal for replay

Absorb the `runtime::world::World` features into the constitutional foundation:

- **Generational entity handles** — each entity carries a generation counter incremented on despawn, catching stale references
- **Dense/sparse component storage** — `ComponentVec<T>` (SparseSet) replaces `HashMap<EntityId, T>` for cache-friendly iteration
- **Typed queries** — `Query<&T>` iterates the dense array without requiring `TypeId` downcasting at every access
- **Typed resources** — ergonomic `Res<T>` / `ResMut<T>` wrappers

The result is one world type with one entity model, one component contract, one resource API, one transaction protocol, and one event model across compiler, promotion, runtime, and serving.

## Core API

```rust
// Entity with generation tracking
pub struct Entity(u64, u32);
impl Entity {
    pub fn id(&self) -> u64;
    pub fn generation(&self) -> u32;
}

// Canonical world
pub struct World { /* CompWorld + SparseSet storage + generations */ }

impl World {
    pub fn spawn(&mut self) -> Entity;
    pub fn despawn(&mut self, entity: Entity) -> bool;
    pub fn is_alive(&self, entity: Entity) -> bool;

    // Component operations (for system implementation)
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T>;
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&T>;

    // Resources
    pub fn insert_resource<T: Resource>(&mut self, resource: T);
    pub fn get_resource<T: Resource>(&self) -> Option<&T>;
    pub fn get_resource_mut<T: Resource>(&mut self) -> Option<&mut T>;

    // Queries (immutable)
    pub fn query<Q: QueryParam>(&self) -> Query<'_, Q>;

    // Transaction support
    pub fn current_epoch(&self) -> WorldEpoch;
    pub fn prepare(&self, commands: Commands, catalogue: Option<&SchemaCatalogue>)
        -> Result<PreparedTransaction, TransactionError>;
    pub fn apply(&mut self, prepared: PreparedTransaction) -> CommitReceipt;
}

// Deferred mutation builder (replaces WorldTxn)
pub struct Commands { /* staged spawns, inserts, removes, events */ }
impl Commands {
    pub fn spawn(&mut self) -> PendingEntity;
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T);
    pub fn remove<T: Component>(&mut self, entity: Entity);
    pub fn insert_resource<T: Resource>(&mut self, resource: T);
    pub fn emit(&mut self, event: DomainEvent);
}

// Validated, ready-to-apply transaction
pub struct PreparedTransaction { /* ops, journal, events, expected_epoch */ }

// Typed query
pub struct Query<'w, Q> { /* dense iterator */ }
impl<'w, T: Component> Iterator for Query<'w, &T> { type Item = (Entity, &'w T); }
impl<'w, A: Component, B: Component> Iterator for Query<'w, (&A, &B)> { ... }
```

## Storage architecture

```
Component column (per type):
  DenseVec<T>:
    dense: Vec<T>              // contiguous component values
    sparse: Vec<Option<u32>>   // entity_id -> dense index
    entity_ids: Vec<u64>       // dense index -> entity_id

World holds:
  columns: HashMap<TypeId, Box<dyn Any>>  // TypeId -> DenseVec<T>
  entities: Vec<Option<EntityMeta>>        // entity_id -> metadata
  generations: Vec<u32>                    // entity_id -> generation
  resources: HashMap<TypeId, Box<dyn Any>>
```

## Transaction protocol

1. **Build**: Systems construct `Commands` by reading world state and staging mutations
2. **Prepare** (`prepare()`): Validates epoch, entity existence, schema consistency, read versions, and conflict freedom. Returns `PreparedTransaction` or error.
3. **Apply** (`apply()`): Applies all mutations atomically, advances epoch, persists journal, emits events. Infallible for valid prepared transactions.
4. **Events**: Published only after successful commit. Consumed by downstream schedules or external subscribers.

## Migration strategy

| Wave | Change | Acceptance |
|---|---|---|
| 1 | World contract (this ADR) | Reviewed and approved ✅ |
| 2 | Build `CanonicalWorld` with SparseSet storage + typed queries | Tests pass: spawn, get, insert, remove, query, schema, epoch, events, replay ✅ |
| 3 | Make transactions the only mutation path | Existing WorldTxn callers work unchanged through compat layer ✅ (Phase 6) |
| 4–9 | Progressive migration of `CompWorld` callers, retire `runtime::World`, fold `WorldTxn` | All production code uses one world |

## Consequences

### Positive

- Single entity model across the entire codebase
- Cache-friendly component iteration without HashMap overhead
- Generational handles prevent use-after-despawn bugs
- Typed queries eliminate downcasting boilerplate
- Transactional guarantees are the default, not an overlay

### Negative

- Migration cost: every `CompWorld` caller must adapt to the new query/resource API
- A compat layer is needed during migration (temporary technical debt)

### Neutral

- Internal storage can be partitioned for concurrency while preserving the one-world contract
- Schema catalogue integration (B6) is deferred — the storage change does not require it

## References

- [Existing CompWorld implementation](compute-core/src/ecs/mod.rs)
- [Runtime world ComponentVec storage](compute-core/src/ecs/runtime/world.rs)
- [WorldTxn transaction protocol](compute-core/src/ecs/constitutional/world_txn.rs)
- [SchemaRegistry](compute-core/src/ecs/constitutional/schema.rs)
- [Revised canonical plan](file:///Users/user/.codex/plans/prism-canonical-ecs-refactor.md)
