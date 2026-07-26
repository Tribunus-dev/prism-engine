# Compute-core Absorption — Phase 2.5 Batch 5: Verification, No-Op

**Date:** 2026-07-26
**Agent:** Branch session, Phase 2.5 batch 5
**Status:** No-op verification — the 8 files in this batch were already
fully ported in commit `e633567e` (Phase 2.5, Agent 2 of 5 parallel
agents). **Zero direct `world.spawn()` / `world.add_component()` /
`world.remove_component()` / `world.get_component_mut()` /
`world.insert()` calls remain in any of the 8 files.**

## Task as received

Port 10 remaining direct world mutations across 8 system/ files to the
engine-local `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs`:

- `compiler_systems.rs` (2 mutations)
- `backend_compile.rs` (2 mutations)
- `work_dispatch_tick.rs` (1 mutation)
- `work_dispatch.rs` (1 mutation)
- `token_budget_tick.rs` (1 mutation)
- `ternary_pipeline.rs` (1 mutation)
- `slot_lease_tick.rs` (1 mutation)
- `session_decode_tick.rs` (1 mutation)

## What was actually found

A direct grep over the 8 files for the
`world\.(spawn|add_component|remove_component|get_component_mut|insert|remove)\b`
pattern, anchored to statement-leading position, returns **zero
matches**. The only references to those API names in the 8 files are
inside `//` comments that say "Direct `world.add_component` calls
outside the WorldTxn seam are forbidden" — these are guard-rail
comments placed at each mutation site by the previous batch to document
the discipline.

Every mutation in the 8 files already flows through
`ConstitutionalWorldTxn` (added in commit `e633567e` to
`compute-core/src/ecs/runtime/constitutional_world_txn.rs`). The shape
is consistent across all 8 files:

```rust
let mut txn = ConstitutionalWorldTxn::new();
for entity in &entities {
    // immutable read → local mutation → staged insert
    let Some(value) = world.get_component::<T>(*entity).cloned() else {
        continue;
    };
    let mut updated = value;
    // ... mutate `updated` ...
    if let Err(e) = txn.stage_insert(*entity, updated) {
        tracing::warn!(entity = ?entity, error = %e, "<system>: stage_insert T");
    }
}
let _ = txn.commit(world).map_err(|e| {
    tracing::error!(error = %e, "<system>: ConstitutionalWorldTxn commit failed");
    anyhow::anyhow!("<system>: ConstitutionalWorldTxn commit failed: {e}")
})?;
```

## Why the engine-local `WorldTxn` was not used

The task brief said to use the **engine-local** `WorldTxn` (from
`compute-core/src/ecs/runtime/world_txn.rs`, commit `ebcaf2bc`), but
the 8 system/ files target the **constitutional**
`prism_ecs_core::World`, not the engine-local `World`. The two
`World` types differ in ways that make them mutually exclusive at the
`WorldTxn` seam:

| Aspect                       | Engine-local `World`               | Constitutional `prism_ecs_core::World`               |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------- |
| `spawn` signature            | `spawn() -> Option<Entity>`        | `spawn(EntityKind, Option<String>) -> ...`            |
| `insert` / `add_component`   | `insert(entity, T)` (panics on bad) | `add_component(entity, T) -> Result<(), WorldError>` |
| `remove` / `remove_component` | `remove::<T>(entity) -> Option<T>` | `remove_component::<T>(entity) -> Result<...>`        |
| Entity metadata              | none (just `Entity(u32)`)          | `EntityKind` + optional name                          |
| `Entity` type                 | `crate::ecs::runtime::world::Entity` | `prism_ecs_core::Entity`                              |
| `Component` trait bound       | `Component: 'static` (blanket)     | `Component: 'static` (the legacy pattern)             |

The engine-local `WorldTxn::stage_spawn()` takes no arguments — it
cannot capture the `EntityKind` and `Option<String>` that the system
files' `world.spawn(EntityKind::Kernel, None)` calls require. Using
the engine-local `WorldTxn` against the constitutional `World` would
fail to compile (`World` mismatches across the seam, and the engine-
local `WorldTxn` apply closures call `world.spawn().ok_or(...)` /
`world.insert(entity, component)` which don't exist on the
constitutional `World`).

The previous batch's changelog
(`changelogs/2026-07-25-compute-core-absorption-phase-2.5-system-rest.md`,
lines 32-60) explicitly documents this design decision:

> The existing `WorldTxn` in
> `compute-core/src/ecs/runtime/world_txn.rs` (the "engine-local" one
> from commit `ebcaf2bc`) operates on the engine-local `World` (a
> simpler entity/component store with no `EntityKind` / no name). It
> could not be used as-is in the system files because they target the
> **constitutional** `prism_ecs_core::World` (with `EntityKind` / name
> / `add_component` / `remove_component`).
>
> ...
>
> **Decision:** add a parallel `ConstitutionalWorldTxn` in
> `compute-core/src/ecs/runtime/constitutional_world_txn.rs` that
> mirrors the engine-local `WorldTxn` pattern (staged spawns, staged
> inserts on `PendingToken` or existing entity, staged removes) but
> targets the constitutional `World`. It accepts any
> `prism_ecs_core::Component` (no classification required).

So the 8 files in this batch are correct as-is. The
`ConstitutionalWorldTxn` is the right seam for them. The engine-local
`WorldTxn` is the right seam for `compilation_systems.rs`-style code
that builds a short-lived engine-local `World` (see
`compute-core/src/ecs/runtime/compilation_systems.rs` lines 580-660
and 780-820, which the task brief cited — those examples DO use the
engine-local `WorldTxn` correctly, because they construct their own
engine-local `World` rather than receiving the constitutional one
through `&mut World`).

## Per-file audit

| File                              | Lines | Direct mutations | Txn seam in use           | Note |
| --------------------------------- | ----- | ---------------- | -------------------------- | ---- |
| `compiler_systems.rs`             | 320   | **0**            | `ConstitutionalWorldTxn`   | `GraphOptimizerSystem::run` and `GraphEqualizationSystem::run` already use the seam. |
| `backend_compile.rs`              | 267   | **0**            | `ConstitutionalWorldTxn`   | `BackendCompilationSystem::run` and `ExecutableCachingSystem::run` already use the seam. Side effects (`xcrun metal`) are correctly outside the txn. |
| `work_dispatch_tick.rs`           | 82    | **0**            | `ConstitutionalWorldTxn`   | Two extract-mutate-insert loops on `ReadyQueueState` and `WorkRegistryComponent`. |
| `work_dispatch.rs`                | 74    | **0**            | `ConstitutionalWorldTxn`   | Single extract-mutate-insert loop on `WorkRegistryComponent`. |
| `token_budget_tick.rs`            | 69    | **0**            | `ConstitutionalWorldTxn`   | Single extract-mutate-insert loop on `TokenBudgetComponent`. |
| `ternary_pipeline.rs`             | 128   | **0**            | `ConstitutionalWorldTxn`   | Single `stage_insert` of `CimageBinaryComp` on the model entity. |
| `slot_lease_tick.rs`              | 76    | **0**            | `ConstitutionalWorldTxn`   | Single extract-mutate-insert loop on `SlotLeaseComponent`. |
| `session_decode_tick.rs`          | 44    | **0**            | `ConstitutionalWorldTxn`   | Single extract-mutate-insert loop on `SessionState`. |
| **Total**                         | 1060  | **0**            | 8/8 files use the seam    | No mutations to port. |

## Build status

`cargo check -p tribunus-compute-core --lib --no-default-features`
emits 242 pre-existing errors, all out of scope for this task (per
the task brief: "Do NOT touch the engine's 100+ pre-existing build
errors — they're out of scope"). Filtering the output for paths under
`compute-core/src/ecs/system/{compiler_systems,backend_compile,
work_dispatch_tick,work_dispatch,token_budget_tick,ternary_pipeline,
slot_lease_tick,session_decode_tick}.rs` returns **zero lines** —
the 8 files in this batch compile cleanly. No new errors, no new
warnings.

## Patterns noted (for the parent)

These patterns are stable across all 8 files and the rest of the
system/ directory:

1. **Extract-mutate-insert on existing entity** (the dominant
   pattern, ~6 of 8 files). `get_component` (immutable borrow) →
   `cloned` → local `mut` mutation → `stage_insert` on the same
   `Entity`. The constitutional `WorldTxn` doesn't have a
   `get_mut_and_stage` API (and intentionally so — the
   `get_component_mut → &mut` flow was the last-mile leakage
   `WorldTxn` was designed to close).
2. **Single-insert per system** (1 file, `ternary_pipeline.rs`).
   Just a `stage_insert` + `commit`, no loop.
3. **No spawns in these 8 files.** All 8 files operate on entities
   that were spawned upstream (typically in `validation.rs`,
   `kernel_catalog.rs`, `source_load.rs`, or by the system host).
   `ConstitutionalWorldTxn::stage_spawn` is not used in any of them,
   which is why the engine-local vs. constitutional seam question
   never arises in practice for these files.

## Recommendation to the parent

The 8 files in this batch are **already complete**. The
"Phase 2.5 batch 5" task is a no-op. Future batches targeting
remaining `system/` mutations should focus on the files that still
have direct mutations as of this writing (verified by grep, anchored
to statement-leading position):

```
catalog_validation.rs:71-72, 92       (3 mutations)
kernel_catalog.rs:97-98, 117-118, 137, 149-150  (7 mutations)
variant_select.rs:260-262, 268-269    (4 mutations)
variant_gen.rs:232-233, 241           (3 mutations)
```

These 4 files contain the last 17 production-path direct mutations in
`compute-core/src/ecs/system/`. They will need either (a) the
existing `ConstitutionalWorldTxn` pattern (matching what the 8 files
in this batch use) or (b) the engine-local `WorldTxn` only if the
constitutional `World` is replaced by an engine-local `World` at the
`CompilerSystem::run` boundary — a much larger architectural change
that is out of scope for Phase 2.5.

## Files changed in this commit

- `changelogs/2026-07-25-compute-core-absorption-phase-2-5-batch-5.md`
  (this file)

No `.rs` files were modified. The 8 batch files are unchanged because
they are already at the target state.
