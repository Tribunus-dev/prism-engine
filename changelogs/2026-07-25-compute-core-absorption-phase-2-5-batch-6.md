# Compute-core Absorption — Phase 2.5 Batch 6: Verification, No-Op

**Date:** 2026-07-26
**Agent:** Branch session, Phase 2.5 batch 6
**Status:** No-op verification — the 8 files in this batch were already
fully ported in commit `e633567e` (Phase 2.5, port 100 remaining
system/ mutations to WorldTxn). **Zero direct `world.spawn()` /
`world.add_component()` / `world.remove_component()` /
`world.get_component_mut()` / `world.insert()` calls remain in any
of the 8 files.**

## Task as received

Port 8 remaining direct world mutations across 8 `system/` files to
the engine-local `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs` (added in commit
`ebcaf2bc`):

- `compute-core/src/ecs/system/session_cleanup.rs` (1 mutation)
- `compute-core/src/ecs/system/profile.rs` (1 mutation)
- `compute-core/src/ecs/system/portfolio.rs` (1 mutation)
- `compute-core/src/ecs/system/phase_engine_tick.rs` (1 mutation)
- `compute-core/src/ecs/system/phase_engine_cleanup.rs` (1 mutation)
- `compute-core/src/ecs/system/phase_engine.rs` (1 mutation)
- `compute-core/src/ecs/system/metal_transfer.rs` (1 mutation)
- `compute-core/src/ecs/system/metal_dispatch.rs` (1 mutation)

## What was actually found

A direct grep over the 8 files for the
`world\.(spawn|add_component|remove_component|get_component_mut|insert)\b`
pattern returns **zero matches in production code paths**. The only
references to those API names in the 8 files are inside `//`
comments that say "Direct `world.X` calls outside the WorldTxn seam
are forbidden" — these are guard-rail comments placed at each
mutation site by the prior commit to document the discipline.

Every mutation in the 8 files already flows through
`ConstitutionalWorldTxn` (added in commit `e633567e` to
`compute-core/src/ecs/runtime/constitutional_world_txn.rs`). The
shape is consistent across all 8 files: `let mut txn =
ConstitutionalWorldTxn::new();` → loop over `entities_of_kind(...)`
→ `txn.stage_insert(...)` / `txn.stage_remove::<T>(...)` → `let _ =
txn.commit(world)?;`.

## Per-file verification

| File | Direct mutations remaining | Already on `*WorldTxn`? | Phase 2.5 commit `e633567e` line(s) | Operations |
|---|---:|---|---|---|
| `session_cleanup.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `remove_component` → `stage_remove::<SessionState>` (×2 in-loop, original bug preserved per `BTreeMap<(Entity,TypeId)>` dedup) | remove-only |
| `profile.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `add_component` → `stage_insert::<ProfileRunResult>` (extract-mutate-insert on `ProfileRunResult` built from `execute_profile()`) | insert-only |
| `portfolio.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `add_component` → `stage_insert::<PortfolioArtifactsComp>` on first model entity | insert-only |
| `phase_engine_tick.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `get_component_mut` → extract-mutate-insert on `PhaseDagState` (advances `current_phase` along DAG edges) | extract-mutate-insert |
| `phase_engine_cleanup.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 2 prod `remove_component` → `stage_remove::<PhaseDagState>` + `stage_remove::<ReadyQueueState>` (×2 in-loop, duplicate-remove bug preserved) | remove-only |
| `phase_engine.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `get_component_mut` → extract-mutate-insert on `PhaseLifecycleComponent` (advances `PhaseState` state machine) | extract-mutate-insert |
| `metal_transfer.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `get_component_mut` → extract-mutate-insert on `TensorComponent` (sets `residency = "metal"` when not resident) | extract-mutate-insert |
| `metal_dispatch.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1/1 prod `get_component_mut` → extract-mutate-insert on `WorkRegistryComponent` (advances `WorkState::Submitted` → `Running`) | extract-mutate-insert |

Total mutations already ported across the 8 files (per the Phase 2.5
`e633567e` per-file table): **8 + 2 = 10** (8 unique mutation sites
plus 2 duplicate `remove` calls in `phase_engine_cleanup.rs` and
`session_cleanup.rs` that the prior commit explicitly preserved as
the verbatim port of the original "stage a remove twice" idiom).
Either way, all 8 files are ported and on the canonical seam.

## Why `ConstitutionalWorldTxn` and not the engine-local `WorldTxn`

The brief assumed the engine-local `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs` (added in `ebcaf2bc`) is
the target. It is not, for the system files. The two `WorldTxn`
flavors operate on different `World` types:

| `WorldTxn` flavor | File | Operates on | Spawn API | Insert / Remove API |
|---|---|---|---|---|
| Engine-local | `compute-core/src/ecs/runtime/world_txn.rs` | Engine runtime `World` (`compute-core/src/ecs/runtime/world.rs` — entity + component store, no `EntityKind`, no `name`) | `stage_spawn()` | `world.insert(entity, comp)` / `world.remove::<T>(entity)` |
| Constitutional (engine bridge) | `compute-core/src/ecs/runtime/constitutional_world_txn.rs` | Constitutional `prism_ecs_core::World` (with `EntityKind` / `name` and `add_component` / `remove_component`) | `stage_spawn(kind, name)` | `world.add_component(entity, comp)` / `world.remove_component::<T>(entity)` |

The `system/` files receive a `&mut World` (the constitutional one) in
every `CompilerSystem::run` call, and every `add_component` /
`remove_component` they call is the constitutional `World`'s API. The
engine-local `WorldTxn` calls `world.insert(entity, comp)` and
`world.remove::<T>(entity)` — those methods exist on the engine
runtime `World` but **not** on the constitutional `World`. The
`constitutional_world_txn.rs` module docstring states this explicitly
(lines 12-22):

> The system files (`compute-core/src/ecs/system/`) cannot use the
> engine-local `WorldTxn` because the World types differ. They also
> cannot use the full constitutional `WorldTxn` in
> `crates/prism-ecs-constitutional/src/world_txn.rs` because that
> API gates `put_durable` / `put_transient` on the
> `DurableComponent` / `TransientComponent` traits, and the system
> files' components only implement `prism_ecs_core::Component` (the
> legacy pattern that the engine's `CompilerSystem`s rely on).
>
> The motivation is the same as the engine-local one: a single
> authority-bearing commit seam so that engine-side mutations do not
> fork into N direct paths.

So the engine-local `WorldTxn` is the right target for
`compilation_systems.rs` and the engine's `runtime/` subsystem (which
use the engine runtime `World`), but the wrong target for the
`system/` files. `ConstitutionalWorldTxn` is the bridge.

A direct port of these 8 files to the engine-local `WorldTxn` would
not compile: the constitutional `World` has no `insert` / `remove`
methods, and the system files have no other way to thread mutations
into it.

## Build status

`cargo check -p tribunus-compute-core --lib --no-default-features`
returns the same 242 pre-existing errors that have existed since the
Phase 3 changelog baseline (e.g. `use of unresolved module or
unlinked crate metal` in `compute-core/src/ecs/backend/metal.rs:1127`,
missing `policy_support` in
`compute-core/src/ecs/backend/heterogeneous_executor.rs:352`,
etc.). **Zero errors come from the 8 files in this batch.** None of
the 8 files appear in the `error[...]` or `warning[...]` output.

## Action taken

**No code changes.** The work was already done correctly in commit
`e633567e` against the right `WorldTxn` flavor for the `system/`
files. This changelog serves as the audit trail and the explanation
for why a Phase 2.5 "batch 6" exists as a no-op verification rather
than a code-mutation pass.

## Patterns noticed (for the next agent's batch)

- **The 8 files break down into 3 patterns:** 4 `extract-mutate-insert`
  (`phase_engine_tick.rs`, `phase_engine.rs`, `metal_transfer.rs`,
  `metal_dispatch.rs`), 2 `remove-only` (`session_cleanup.rs`,
  `phase_engine_cleanup.rs`), 1 `insert-only with computation`
  (`profile.rs` — `ProfileRunResult` is built from `execute_profile()`
  before being staged), 1 `single-insert` (`portfolio.rs`).
- **The 2 remove-only files preserve a pre-existing duplicate-remove
  bug.** The original code did
  `world.remove_component::<T>(entity); world.remove_component::<T>(entity);`
  (the second was a no-op). The staged-txn port preserves this
  verbatim because `ConstitutionalWorldTxn::removes` is keyed by
  `(Entity, TypeId)` and the second `stage_remove` overwrites the
  first with the same closure payload. This is documented in
  comments at the call site.
- **The `extract-mutate-insert` pattern is the only way to port
  `get_component_mut` through the constitutional `WorldTxn` without
  adding trait classifications to ~30 components.** Cost: one
  `clone()` per mutation. This is the same trade-off the prior
  commit documented in its `Patterns discovered` section.
- **`session_cleanup.rs` is gated on `EntityKind::CommandBuffer`
  while `phase_engine_cleanup.rs` / `phase_engine_tick.rs` /
  `metal_dispatch.rs` are gated on `EntityKind::Executable` (or
  `Tensor`).** All four use the same pattern
  (`entities_of_kind` → loop → `stage_*` → `commit`).

## CAMPAIGN.md status

No change. The `system/` subsystem remains in the
`Shadow → approaching Canonical` transition state that the prior
Phase 2.5 commit recorded. The `ConstitutionalWorldTxn` helper
remains the canonical authority seam for ECS state mutations in
`system/`, exactly as designed.
