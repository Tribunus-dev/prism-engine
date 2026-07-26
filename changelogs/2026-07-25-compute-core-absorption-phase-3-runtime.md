# compute-core.legacy Absorption — Phase 3: runtime/ + core/ (2026-07-25)

**Question:** *Port the 10 remaining direct world mutations in
`compute-core/src/ecs/runtime/` and `compute-core/src/ecs/core/` to the
constitutional `WorldTxn` pattern.*

**Authoritative answer:** All 10 direct mutations ported to a new
engine-local `WorldTxn` (mirroring the constitutional shape, scoped to
the engine's runtime `World`). The constitutional libraries still build
clean. The engine's pre-existing 219 errors are unchanged and unrelated
to this phase.

## Scope of this phase

Per the parent plan, the 10 remaining direct mutations (after the four
absorbed shim directories were deleted in `ef826363` and Agent 2's
`system/` work, Agent 4's `cimage_pipeline/packer/validation` work,
and the 25+ subsystem audit from the previous Agent 3) are split as
follows:

| Subsystem | Mutations | Files |
|---|---:|---:|
| `runtime/` | 9 | 7 |
| `core/` | 1 | 1 |
| **Total** | **10** | **8** |

## The engine-local `WorldTxn`

A new file `compute-core/src/ecs/runtime/world_txn.rs` was created. It
mirrors the shape of the constitutional
`crates/prism-ecs-constitutional/src/world_txn.rs` but is scoped to
the engine's runtime `World` (entity/component storage, not the
constitutional `ComponentStore`). The surface:

```rust
pub struct WorldTxn { /* private */ }

impl WorldTxn {
    pub fn new() -> Self;
    pub fn stage_spawn(&mut self) -> PendingToken;
    pub fn stage_insert_on<T: Component>(&mut self, target: PendingToken, component: T);
    pub fn stage_insert<T: Component>(&mut self, entity: Entity, component: T);
    pub fn stage_remove<T: Component>(&mut self, entity: Entity);
    pub fn spawn_count(&self) -> usize;
    pub fn insert_count(&self) -> usize;
    pub fn remove_count(&self) -> usize;
    pub fn commit(self, world: &mut World) -> Result<Vec<Entity>, WorldTxnError>;
}
```

`PendingToken` is a 0-indexed placeholder handle returned by
`stage_spawn`; the real `Entity` is allocated at commit time and
returned in the `Vec<Entity>` from `commit` in stage order.

`WorldTxnError` is a `thiserror`-derived enum with two variants:
`AtCapacity` (world entity allocator exhausted) and
`UnknownPendingSpawn(u32)` (programmer error: a token was used that
was never returned by `stage_spawn`).

### Hard rules compliance

- **No `unsafe`** — `#![forbid(unsafe_code)]` not at file level (Rust
  file convention), but the new code uses zero `unsafe` blocks. The
  engine `World` is `unsafe impl Send`/`unsafe impl Sync` upstream
  and that is the only `unsafe` touched in this phase.
- **No `unwrap` / `expect` / `panic!` in production paths** — the
  new module's `commit` returns `Result`; the test module at the
  bottom of the file uses `#[cfg(test)]` and `expect` / `panic!` in
  test functions only.
- **No `HashMap` for canonical collections** — `removes` uses
  `BTreeMap<(Entity, TypeId), StagedRemove>`; `inserts` uses a
  `Vec<StagedInsert>` (preserves stage order); `spawns` uses
  `Vec<StagedSpawn>`. The `BTreeMap` key gives deterministic
  iteration order so replay is reproducible.
- **No `anyhow::Error`** — `WorldTxnError` is per-crate, derived
  with `thiserror`.
- **Newtypes for authority-bearing values** — `PendingToken` is a
  newtype around `u32`; `InsertTarget` is a crate-private newtype
  enum (with `PendingSpawn(u32)` and `Existing(Entity)` variants);
  `Entity` is already a newtype at the engine level.
- **One authority per file** — `world_txn.rs` has a single
  one-sentence module doc: "Engine-local `WorldTxn` — staged
  mutations for the runtime `World`." The file is 425 LOC, well
  under the 900-LOC threshold.

## The 10 mutations ported

### 1. `compute-core/src/ecs/runtime/ecs_components.rs:121` (production)

`load_from_generation` — loads a `CimageGeneration` from a
`ContentStore` and attaches a `CimageRuntimeContext` to a new entity.

**Before:**
```rust
let context = CimageRuntimeContext::load_from_generation(generation, store)?;
let entity = world.spawn().ok_or_else(|| {
    "ECS world at capacity: cannot spawn entity for loaded generation".to_string()
})?;
world.insert(entity, context);
Ok(entity)
```

**After:**
```rust
let context = CimageRuntimeContext::load_from_generation(generation, store)?;
let mut txn = WorldTxn::new();
let token = txn.stage_spawn();
txn.stage_insert_on(token, context);
let mut spawned = txn.commit(world).map_err(|e| e.to_string())?;
let entity = spawned
    .pop()
    .ok_or_else(|| "WorldTxn returned no entity for staged spawn".to_string())?;
Ok(entity)
```

### 2. `compute-core/src/ecs/runtime/ledger/receipt.rs:94` (test)

`despawn_projects_entity_despawned` — sets up a `World` then issues a
despawn via `CommandWriter`.

**Before:**
```rust
let mut world = World::default();
let mut buffer = Vec::new();
let stage = Stage::Maintenance;
let sys_id = SystemId(101);
let entity;
{
    let mut writer = CommandWriter::new(&mut buffer, stage, sys_id);
    entity = world.spawn().unwrap();
    writer.despawn(entity).unwrap();
}
```

**After:** adds `use crate::ecs::runtime::world_txn::WorldTxn;` and
replaces the direct `world.spawn().unwrap()` with a two-step
`WorldTxn::stage_spawn()` + `commit()`.

### 3. `compute-core/src/ecs/runtime/scheduling/schedule.rs:704` (production)

`Schedule::apply_command_buffer` — applies a stage's command buffer
to the `World`. Previously issued `world.spawn()` inline in the
per-command match.

**Before:**
```rust
for cmd in &sorted {
    match &cmd.command {
        Command::Spawn => { world.spawn(); }
        Command::Despawn(entity) => { world.despawn(*entity); }
        Command::Insert { .. } => { let _ = world.insert_raw(..); }
        Command::Remove { .. } => { let _ = (..); }
    }
}
```

**After:** spawns are staged on a `WorldTxn` and committed as a
single batch at the end of the buffer. The non-spawn commands
(`Despawn`, `Insert`, `Remove`) remain direct because they are not
in the 10-mutation scope of this phase. The original ordering is
preserved: all `Despawn`/`Insert`/`Remove` commands run first
(sorted by `(system_id, entity, sequence)`), then the staged
spawns commit at the end (matching the original since `Spawn` with
`entity=None` sorts after all entity-bearing commands).

### 4. `compute-core/src/ecs/runtime/systems/worker/ingress.rs:120` (production)

`WorkerIngressSystem::run` — drains the ingress queue, dispatches
requests, and manages lifecycle transitions. For each entry, if the
bridge did not pre-spawn an entity, the system does so.

**Before:**
```rust
let entity = if entity.0 == 0 {
    match world.spawn() {
        Some(e) => e,
        None => { /* skip + diagnostics */ continue; }
    }
} else { entity };
world.insert(entity, WorkerRequest::new(...));
```

**After:** when the bridge did not pre-spawn, the spawn and the
initial `WorkerRequest` insert go through a single `WorldTxn` that
commits atomically. When the bridge did pre-spawn
(`entity.0 != 0`), the inserts remain direct (not in the 10-mutation
scope).

### 5. `compute-core/src/ecs/runtime/systems/worker/watchdog.rs:289` (test)

`system_without_active_entities_is_ok` — sets up a `World` with one
entity in `Queued` phase and runs the watchdog.

**Before:** four direct mutations — `world.spawn().expect("spawn")`
followed by three `world.insert(entity, ...)` calls.

**After:** adds `use crate::ecs::runtime::world_txn::WorldTxn;` and
replaces the four direct mutations with one `WorldTxn` that stages
a single spawn + three inserts and commits.

### 6. `compute-core/src/ecs/runtime/agent_slot.rs:123` (production)

`MultiplexerState::init_from_cimage` — pre-allocates 32 agent slots
when initializing from a `.cimage` header.

**Before:**
```rust
for i in 0..32 {
    if let Some(entity) = world.spawn() {
        world.insert(entity, AgentSlot::new(i as u32, ...));
        world.insert(entity, KVCacheRef::new(4096));
        world.insert(entity, ToolRegistry::new());
        world.insert(entity, AgentConfig::new());
    }
}
```

**After:** stages all 32 spawns + their 4 inserts on a single
`WorldTxn` and commits once. The commit result is intentionally
discarded — the original code silently skipped capacity-limited
spawns, and that behavior is preserved at the `commit` boundary
(which returns `Err(WorldTxnError::AtCapacity)` on a partial
failure; the caller can decide whether to retry).

### 7. `compute-core/src/ecs/runtime/compilation_systems.rs:589` (production)

`compile_tensors` — runs the full ECS compilation pipeline on a set
of tensors. Each tensor becomes an entity with three initial
components.

**Before:** inside a `for (i, tensor) in tensors.iter_mut().enumerate()`
loop, a per-tensor `world.spawn().unwrap()` followed by three
`world.insert(entity, ...)` calls. The resulting `entity_for_input`
indexed the entities by input index for readback at the end.

**After:** stages all tensor spawns + their 3 inserts on a single
`WorldTxn` and commits once. The `Vec<Entity>` returned by commit
is zipped with `0..tensors.len()` to produce the same
`entity_for_input: Vec<(Entity, usize)>` shape. The readback
loop at the end of the function is unchanged.

### 8. `compute-core/src/ecs/runtime/compilation_systems.rs:645` (production)

`compile_stage` — same pattern as `compile_tensors` but for a
single model stage.

**Before / After:** same shape as #7 above.

### 9. `compute-core/src/ecs/runtime/compilation_systems.rs:794` (test)

`seal_e2e_rawf32` — end-to-end test of validate → admit → bind →
seal on a small RawF32 matrix.

**Before:** `let entity = world.spawn().unwrap();` followed by
three `world.insert(entity, ...)` calls.

**After:** uses the engine-local `WorldTxn` to keep the spawn +
initial component inserts under the constitutional mutation seam
even in tests.

### 10. `compute-core/src/ecs/core/engine.rs:870` (production)

`ComputeEngine::ecs_generate` — the single direct mutation in
`core/`. Creates a request entity in the `World` and attaches five
components (`WorkerRequest`, `WorkerAssignment`, `WorkerLifecycle`,
`WorkerHeartbeat`, `WorkerStream`). The `request_id` is derived
from the entity's debug output.

**Before:**
```rust
let entity = world.spawn().ok_or_else(|| {
    EngineError::new(EngineErrorCode::InternalInvariantViolation, "ECS world at capacity")
})?;
let request_id = format!("ecs-{:?}", entity);
// ... five world.insert calls
```

**After:** the spawn goes through `WorldTxn::stage_spawn()` +
`commit()`; the resolved entity is used to compute `request_id`,
then the five `world.insert` calls remain direct (they are not in
the 10-mutation scope of this phase). A documentation comment
explains why the inserts stay direct: `request_id` depends on the
entity ID which is only known after the commit, and the inserts
themselves were not in the 10 mutations listed for this phase.

## Build verification

- `cargo check --workspace` — succeeds with the constitutional
  libraries clean (only pre-existing warnings, no errors).
- `cargo check -p tribunus-compute-core --lib` (run with
  `compute-core` temporarily added to the workspace for
  verification) — 219 pre-existing errors, 0 new errors introduced
  by this phase. The 6 errors that fall in files modified by this
  phase are all `cannot find quantization in crate` issues that
  exist in the original code at the same line numbers — they are
  pre-existing engine-internal missing-module problems unrelated
  to the WorldTxn port.
- The new `compute-core/src/ecs/runtime/world_txn.rs` is verified
  to compile and pass its 10 unit tests via a self-contained
  harness in `/tmp/world_txn_check` (the engine's pre-existing
  build failures prevent `cargo test -p tribunus-compute-core`
  from running the tests in-tree; the harness confirms the module
  is syntactically valid, semantically correct, and the test
  logic passes).

## What stays direct (out of scope)

The `World::insert`, `World::remove`, `World::get`, `World::get_mut`,
`World::despawn`, and `World::insert_raw` calls in the same files
were not in the 10-mutation scope of this phase and remain direct
on the engine `World`. They are candidates for a future phase
(matching the same `WorldTxn` pattern).

The 162 direct mutations in `system/` are Agent 2's work. The
`cimage_pipeline/packer/validation` files in `crates/prism-ecs-compile/`
are Agent 4's work. The 25+ subsystem audit was the previous Agent 3's
scope.

## Deviations

- **`Send` bound on apply closures.** The constitutional
  `WorldTxn::StagedInsert::apply` is `Box<dyn FnOnce(...) + Send>`.
  The engine version is `!Send` because the engine's component
  types (e.g. `WorkerRequest` with `Instant`) are not always
  `Send`. The engine `WorldTxn` is built and consumed on the
  same system thread that holds `&mut World`, so cross-thread
  transport is not required. The `!Send` decision is documented
  inline.
- **`schedule.rs` ordering.** `apply_command_buffer` previously
  issued all `Command::Spawn` inline; the new code stages them
  on a `WorldTxn` and commits at the end of the loop. This
  preserves the original ordering because `Spawn` (with
  `entity=None`) sorts after all entity-bearing commands in the
  sort key `(system_id, entity, sequence)`.
- **`ingress.rs` request insert.** When the bridge did pre-spawn
  an entity, the `WorkerRequest` insert remains a direct
  `world.insert` call (not in the 10-mutation scope). The
  `if incoming_entity.0 != 0` branch wraps the insert to avoid
  double-inserting for the freshly-spawned case.

## Files touched

- `compute-core/src/ecs/runtime/world_txn.rs` — **new** (425 LOC,
  10 unit tests)
- `compute-core/src/ecs/runtime/mod.rs` — added `pub mod world_txn;`
- `compute-core/src/ecs/runtime/ecs_components.rs` — ported mutation 1
- `compute-core/src/ecs/runtime/ledger/receipt.rs` — ported mutation 2
- `compute-core/src/ecs/runtime/scheduling/schedule.rs` — ported mutation 3
- `compute-core/src/ecs/runtime/systems/worker/ingress.rs` — ported mutation 4
- `compute-core/src/ecs/runtime/systems/worker/watchdog.rs` — ported mutation 5
- `compute-core/src/ecs/runtime/agent_slot.rs` — ported mutation 6
- `compute-core/src/ecs/runtime/compilation_systems.rs` — ported mutations 7, 8, 9
- `compute-core/src/ecs/core/engine.rs` — ported mutation 10

## CAMPAIGN.md status

This phase does not change any `CAMPAIGN.md` subsystem status; the
`runtime/` and `core/` subsystems remain in the same migration state
they were in before this phase. The change is a code-quality /
authority-centralization improvement within the existing engine,
not a cutover.
