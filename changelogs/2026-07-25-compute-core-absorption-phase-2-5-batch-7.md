# Compute-core.legacy Absorption — Phase 2.5 Batch 7: No-Op Confirmation

**Date:** 2026-07-26
**Agent:** Batch 7 of Phase 2.5
**Status:** No-op — all 6 files in batch already ported in commit `e633567e`

## TL;DR

The brief for this batch asked to port 6 direct world mutations in 6
`compute-core/src/ecs/system/` files to the engine-local `WorldTxn` (added
in commit `ebcaf2bc`). On inspection, **all 6 files have zero direct
mutations remaining**. They were already ported to `ConstitutionalWorldTxn`
in commit `e633567e` (Phase 2.5 — "port 100 remaining system/ mutations
to WorldTxn"). This changelog records the verification and the
reason the engine-local `WorldTxn` was not the target pattern in the
first place.

## Verification

Files in batch (all checked via `grep -nE "(world\.spawn|world\.add_component|world\.remove_component|world\.get_component_mut|world\.insert)\b"`):

| File | Direct mutations remaining | Already on `*WorldTxn`? | Phase 2.5 mapping |
|---|---:|---|---|
| `compute-core/src/ecs/system/metal_cleanup.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e` line 97: 2/2 prod `remove_component` → `stage_remove` |
| `compute-core/src/ecs/system/fusion/scalar.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e` line 105: 1/1 prod `add_component` → `stage_insert` |
| `compute-core/src/ecs/system/fusion/heuristic.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e` line 85: 4/4 prod extract-mutate-insert |
| `compute-core/src/ecs/system/completion_ingest.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e` line 116: 1/1 prod extract-mutate-insert |
| `compute-core/src/ecs/system/backpressure_tick.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e` line 117: 1/1 prod extract-mutate-insert |
| `compute-core/src/ecs/system/backend_eval.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e` line 118: 1/1 prod extract-mutate-insert |

The only matches for the direct-mutation patterns in these 6 files are
inside comments documenting the discipline (e.g.
"// `world.remove_component` calls outside the WorldTxn seam are
forbidden."). Zero call sites.

Total mutations already ported across the 6 files (per the Phase 2.5
per-file table): **10** (2 + 1 + 4 + 1 + 1 + 1), not 6 as the brief
estimated. Either way, all ported.

## Why `ConstitutionalWorldTxn` and not the engine-local `WorldTxn`

The brief assumed the engine-local `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs` (added in `ebcaf2bc`) is
the target. It is not, for the system files. The two WorldTxn types
operate on different `World` types:

| `WorldTxn` flavor | File | Operates on | Spawn API | Insert API |
|---|---|---|---|---|
| Engine-local | `compute-core/src/ecs/runtime/world_txn.rs` | Engine runtime `World` (entity + component store, no `EntityKind`) | `stage_spawn()` | `world.insert(entity, comp)` |
| Constitutional (engine bridge) | `compute-core/src/ecs/runtime/constitutional_world_txn.rs` | Constitutional `prism_ecs_core::World` (with `EntityKind` / name) | `stage_spawn(kind, name)` | `world.add_component(entity, comp)` / `world.remove_component::<T>(entity)` |

The `system/` files receive a `&mut World` (the constitutional one) in
every `CompilerSystem::run` call, and every `add_component` /
`remove_component` they call is the constitutional `World`'s API. The
engine-local `WorldTxn` calls `world.insert(entity, comp)` and
`world.remove::<T>(entity)` — those methods exist on the engine
runtime `World` but **not** on the constitutional `World`. The
`constitutional_world_txn.rs` module docstring states this explicitly
(lines 12-19):

> The system files (`compute-core/src/ecs/system/`) cannot use the
> engine-local `WorldTxn` because the World types differ. They also
> cannot use the full constitutional `WorldTxn` in
> `crates/prism-ecs-constitutional/src/world_txn.rs` because that API
> gates `put_durable` / `put_transient` on the
> `DurableComponent` / `TransientComponent` traits, and the system
> files' components only implement `prism_ecs_core::Component` (the
> legacy pattern that the engine's `CompilerSystem`s rely on).

So the engine-local `WorldTxn` is the right target for the
`compilation_systems.rs` and the engine's `runtime/` subsystem (which
uses the engine runtime `World`), but the wrong target for the
`system/` files. `ConstitutionalWorldTxn` is the bridge.

A direct port of these 6 files to the engine-local `WorldTxn` would
not compile: the constitutional `World` has no `insert` / `remove`
methods, and the system files have no other way to thread mutations
into it.

## Current pattern in each file (already correct)

All 6 files use the same shape:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let entities: Vec<Entity> = world.entities_of_kind(...);
    let mut txn = ConstitutionalWorldTxn::new();
    for entity in &entities {
        // ... extract via get_component + .cloned() ...
        // ... mutate the local copy ...
        if let Err(e) = txn.stage_insert(*entity, updated) {
            tracing::warn!(...);
        }
    }
    let _ = txn.commit(world).map_err(|e| { ... })?;
    Ok(())
}
```

`metal_cleanup.rs` uses `stage_remove` instead of `stage_insert`
(per-file pattern documented in `e633567e` line 97).

No direct `world.spawn` / `world.add_component` / `world.remove_component`
/ `world.get_component_mut` / `world.insert` calls in production paths.
Zero call sites in the 6 files.

## What this batch did

Nothing. The work was already complete in `e633567e`. This changelog
is the only artifact, and it exists to make the no-op state explicit
for the next agent that scans the system/ directory and to record the
verification that all 6 files are on the ConstitutionalWorldTxn seam.

## Recommendation to the parent session

- The Phase 2.5 absorption of `system/` is complete. No follow-up
  batches targeting direct mutations in this directory will find
  any.
- The "batch N" decomposition used to parallelize Phase 2.5 should be
  retired for the `system/` subsystem. Future compute-core
  absorption work should target the `core/` and `compute_image/`
  subsystems (per CAMPAIGN.md and the Phase 4B/4C/4C-cont
  changelogs), or the engine's `runtime/` subsystem (Phase 3 in
  `ebcaf2bc` already covered the engine-local `World` ops; the
  remaining `runtime/` work is a follow-up).
- If the brief's "engine-local `WorldTxn`" framing is the goal
  project-wide, consider migrating `ConstitutionalWorldTxn` to a
  thin alias over the engine-local `WorldTxn` so there is one
  `WorldTxn` type. That is a refactor, not a port.

## Files changed in this batch

- None in `compute-core/src/ecs/system/`.
- This changelog only.
